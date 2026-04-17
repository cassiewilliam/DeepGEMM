// =====================================================================================
// tests/cpp/test_mega_ffn_baseline.cu
//
// MegaFFN baseline 对比测试：
//   Baseline A: 最基础的 naive BF16 GEMM kernel（不用 Tensor Core, 不用 SMEM）
//               3 kernels: Linear1 + SwiGLU + Linear2
//   Baseline B: cuBLAS BF16 未融合（cublasGemmEx, 走 Tensor Core）
//               2 GEMM + 1 SwiGLU + 1 final cast
//
// 输入生成方式 (seed 0xBEEF) 与 test_sm100_fp8_mega_ffn.cu 完全一致，保证公平对比。
// 权重/激活在 FP8 侧量化后一次性 dequant 到 BF16 (不计入 timing)，然后跑 BF16 GEMM。
//
// 构建：./tests/cpp/build_mega_ffn.sh  (会额外构建此 baseline binary)
// =====================================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d : %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        std::exit(1); \
    } \
} while (0)

#define CUBLAS_CHECK(x) do { \
    cublasStatus_t _s = (x); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuBLAS error %s:%d code=%d\n", __FILE__, __LINE__, (int)_s); \
        std::exit(1); \
    } \
} while (0)

// -----------------------------------------------------------------------------
// 形状常量 (Qwen3-0.6B) —— 与主测试保持一致
// -----------------------------------------------------------------------------
constexpr uint32_t kHidden       = 1024;
constexpr uint32_t kIntermediate = 3072;
constexpr uint32_t kMaxM         = 128;
constexpr uint32_t BLOCK_N       = 128;    // SwiGLU 的 gate/up pairing 粒度
constexpr uint32_t kGranK        = 32;

// -----------------------------------------------------------------------------
// Host 端 FP8 / UE8M0 helper (与主测试同)
// -----------------------------------------------------------------------------
static inline uint8_t float_to_fp8_e4m3(float x) {
    __nv_fp8_e4m3 v(x);
    return *reinterpret_cast<uint8_t*>(&v);
}
static inline float ue8m0_to_float(uint8_t e) {
    if (e == 0) return 0.f;
    uint32_t bits = static_cast<uint32_t>(e) << 23;
    float v; std::memcpy(&v, &bits, 4);
    return v;
}
static inline uint8_t float_to_ue8m0_ceil(float x) {
    if (x <= 0.f or !std::isfinite(x)) return 0;
    uint32_t bits; std::memcpy(&bits, &x, 4);
    uint32_t exp = (bits >> 23) & 0xff;
    uint32_t man = bits & ((1u << 23) - 1);
    if (man != 0) exp += 1;
    if (exp > 0xff) exp = 0xff;
    return static_cast<uint8_t>(exp);
}
static void quantize_fp8_ue8m0(const float* src, uint32_t rows, uint32_t cols,
                               std::vector<uint8_t>& fp8_out,
                               std::vector<uint8_t>& sf_out) {
    fp8_out.resize(static_cast<size_t>(rows) * cols);
    sf_out.resize(static_cast<size_t>(rows) * (cols / kGranK));
    for (uint32_t r = 0; r < rows; ++ r)
        for (uint32_t kg = 0; kg < cols / kGranK; ++ kg) {
            float amax = 0.f;
            for (uint32_t t = 0; t < kGranK; ++ t)
                amax = std::max(amax, std::fabs(src[r * cols + kg * kGranK + t]));
            const float factor = amax / 448.f;
            uint8_t sf = float_to_ue8m0_ceil(factor);
            float sf_f = ue8m0_to_float(sf);
            float inv  = sf_f > 0.f ? 1.f / sf_f : 0.f;
            sf_out[r * (cols / kGranK) + kg] = sf;
            for (uint32_t t = 0; t < kGranK; ++ t) {
                float v = src[r * cols + kg * kGranK + t];
                fp8_out[r * cols + kg * kGranK + t] = float_to_fp8_e4m3(v * inv);
            }
        }
}

// -----------------------------------------------------------------------------
// Device kernels
// -----------------------------------------------------------------------------

// Dequant: fp8+ue8m0 (per-32 per-row) → bf16，layout 均为 row-major [rows, cols]
__global__ void dequant_fp8_to_bf16(
    const uint8_t* __restrict__ fp8, const uint8_t* __restrict__ sf,
    nv_bfloat16* __restrict__ out, uint32_t rows, uint32_t cols)
{
    const uint32_t r = blockIdx.y;
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || j >= cols) return;
    const uint32_t kg = j / kGranK;
    const uint8_t  sf_b = sf[r * (cols / kGranK) + kg];
    const float    scale = sf_b == 0 ? 0.f : __uint_as_float(static_cast<uint32_t>(sf_b) << 23);
    __nv_fp8_e4m3 v; *reinterpret_cast<uint8_t*>(&v) = fp8[r * cols + j];
    out[r * cols + j] = __float2bfloat16(static_cast<float>(v) * scale);
}

// SwiGLU: in_raw [M, 2I] → out [M, I]，每 BLOCK_N 内前半=gate 后半=up
__global__ void swiglu_bf16(
    const nv_bfloat16* __restrict__ in_raw, nv_bfloat16* __restrict__ out,
    uint32_t M, uint32_t I)
{
    const uint32_t m = blockIdx.y;
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || j >= I) return;
    const uint32_t block_n = j / (BLOCK_N / 2);
    const uint32_t jj      = j % (BLOCK_N / 2);
    const uint32_t g_idx   = m * 2 * I + block_n * BLOCK_N + jj;
    const uint32_t u_idx   = g_idx + BLOCK_N / 2;
    const float g = __bfloat162float(in_raw[g_idx]);
    const float u = __bfloat162float(in_raw[u_idx]);
    const float silu_g = g / (1.f + __expf(-g));
    out[m * I + j] = __float2bfloat16(silu_g * u);
}

// Naive GEMM-T (row-major): C[M, N] = A[M, K] @ B[N, K]^T, BF16 w/ FP32 acc.
// 每 thread 1 个输出元素，无 SMEM，无 Tensor Core —— 最朴素 baseline。
__global__ void naive_gemm_transB_bf16(
    const nv_bfloat16* __restrict__ A, const nv_bfloat16* __restrict__ B,
    nv_bfloat16* __restrict__ C,
    uint32_t M, uint32_t N, uint32_t K)
{
    const uint32_t m = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;
    float acc = 0.f;
    for (uint32_t k = 0; k < K; ++ k)
        acc += __bfloat162float(A[m * K + k]) * __bfloat162float(B[n * K + k]);
    C[m * N + n] = __float2bfloat16(acc);
}

// -----------------------------------------------------------------------------
// CPU 参考: 与 main test 一致的 FP8-faithful pipeline
//   X, W1, W2 都是 FP8+UE8M0, 中间 interm 也再量化到 FP8+UE8M0 (模拟 MMA 的 SF-dequant)
// -----------------------------------------------------------------------------
static inline float fp8_e4m3_to_float(uint8_t x) {
    __nv_fp8_e4m3 v; *reinterpret_cast<uint8_t*>(&v) = x;
    return static_cast<float>(v);
}
static void cpu_reference_ffn(
    const std::vector<uint8_t>& x_fp8, const std::vector<uint8_t>& x_sf,
    const std::vector<uint8_t>& w1_fp8, const std::vector<uint8_t>& w1_sf,
    const std::vector<uint8_t>& w2_fp8, const std::vector<uint8_t>& w2_sf,
    uint32_t M, std::vector<nv_bfloat16>& y_bf16)
{
    auto deq_act = [&](const std::vector<uint8_t>& a, const std::vector<uint8_t>& s,
                       uint32_t rows, uint32_t cols) {
        std::vector<float> out(static_cast<size_t>(rows) * cols);
        for (uint32_t r = 0; r < rows; ++ r)
            for (uint32_t kg = 0; kg < cols / kGranK; ++ kg) {
                float scale = ue8m0_to_float(s[r * (cols / kGranK) + kg]);
                for (uint32_t t = 0; t < kGranK; ++ t)
                    out[r * cols + kg * kGranK + t] =
                        fp8_e4m3_to_float(a[r * cols + kg * kGranK + t]) * scale;
            }
        return out;
    };
    auto X  = deq_act(x_fp8, x_sf, M, kHidden);
    auto W1 = deq_act(w1_fp8, w1_sf, 2 * kIntermediate, kHidden);
    auto W2 = deq_act(w2_fp8, w2_sf, kHidden, kIntermediate);
    std::vector<float> interm_raw(static_cast<size_t>(M) * (2 * kIntermediate), 0.f);
    for (uint32_t m = 0; m < M; ++ m)
        for (uint32_t n = 0; n < 2 * kIntermediate; ++ n) {
            float acc = 0.f;
            for (uint32_t k = 0; k < kHidden; ++ k)
                acc += X[m * kHidden + k] * W1[n * kHidden + k];
            interm_raw[m * (2 * kIntermediate) + n] = acc;
        }
    std::vector<float> interm(static_cast<size_t>(M) * kIntermediate, 0.f);
    auto silu = [](float v) { return v / (1.f + std::exp(-v)); };
    const uint32_t n_blocks = (2 * kIntermediate) / BLOCK_N;
    for (uint32_t m = 0; m < M; ++ m)
        for (uint32_t b = 0; b < n_blocks; ++ b)
            for (uint32_t j = 0; j < BLOCK_N / 2; ++ j) {
                float g = interm_raw[m * (2 * kIntermediate) + b * BLOCK_N + j];
                float u = interm_raw[m * (2 * kIntermediate) + b * BLOCK_N + BLOCK_N / 2 + j];
                interm[m * kIntermediate + b * (BLOCK_N / 2) + j] = silu(g) * u;
            }
    std::vector<uint8_t> interm_fp8, interm_sf;
    quantize_fp8_ue8m0(interm.data(), M, kIntermediate, interm_fp8, interm_sf);
    auto interm_dq = deq_act(interm_fp8, interm_sf, M, kIntermediate);
    y_bf16.assign(static_cast<size_t>(M) * kHidden, nv_bfloat16(0.f));
    for (uint32_t m = 0; m < M; ++ m)
        for (uint32_t h = 0; h < kHidden; ++ h) {
            float acc = 0.f;
            for (uint32_t i = 0; i < kIntermediate; ++ i)
                acc += interm_dq[m * kIntermediate + i] * W2[h * kIntermediate + i];
            y_bf16[m * kHidden + h] = __float2bfloat16(acc);
        }
}

// -----------------------------------------------------------------------------
// Compare + report
// -----------------------------------------------------------------------------
static void compare_and_report(const char* tag, uint32_t M,
                               const std::vector<nv_bfloat16>& y_gpu,
                               const std::vector<nv_bfloat16>& y_ref,
                               double us_per_iter, uint32_t num_iters)
{
    double l1 = 0.0, mx = 0.0;
    for (uint32_t m = 0; m < M; ++ m)
        for (uint32_t h = 0; h < kHidden; ++ h) {
            float g = __bfloat162float(y_gpu[m * kHidden + h]);
            float r = __bfloat162float(y_ref[m * kHidden + h]);
            float d = std::fabs(g - r);
            l1 += d;
            mx = std::max<double>(mx, d);
        }
    double denom = static_cast<double>(M) * kHidden;
    std::printf("[%s] M=%u  mean|Δ|=%.4f  max|Δ|=%.4f   latency=%.3f us (%u iters)\n",
                tag, M, l1 / denom, mx, us_per_iter, num_iters);
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    uint32_t M         = argc > 1 ? static_cast<uint32_t>(std::atoi(argv[1])) : 1;
    uint32_t num_iters = argc > 2 ? static_cast<uint32_t>(std::atoi(argv[2])) : 100;
    const char* which  = argc > 3 ? argv[3] : "all";   // "all" | "naive" | "cublas"

    if (M == 0 or M > kMaxM) {
        std::fprintf(stderr, "Invalid M=%u (1..%u)\n", M, kMaxM);
        return 1;
    }

    CUDA_CHECK(cudaFree(nullptr));

    // ---- Generate identical inputs to main test (seed 0xBEEF) ----
    std::mt19937 rng(0xBEEFu);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x_fp32 (static_cast<size_t>(kMaxM) * kHidden, 0.f);
    std::vector<float> w1_fp32(static_cast<size_t>(2 * kIntermediate) * kHidden, 0.f);
    std::vector<float> w2_fp32(static_cast<size_t>(kHidden) * kIntermediate, 0.f);
    for (size_t i = 0; i < static_cast<size_t>(M) * kHidden; ++ i) x_fp32[i] = dist(rng);
    for (auto& v : w1_fp32) v = 0.3f * dist(rng);
    for (auto& v : w2_fp32) v = 0.3f * dist(rng);

    std::vector<uint8_t> x_fp8, x_sf, w1_fp8, w1_sf, w2_fp8, w2_sf;
    quantize_fp8_ue8m0(x_fp32.data(),  kMaxM,               kHidden,       x_fp8,  x_sf);
    quantize_fp8_ue8m0(w1_fp32.data(), 2 * kIntermediate,   kHidden,       w1_fp8, w1_sf);
    quantize_fp8_ue8m0(w2_fp32.data(), kHidden,             kIntermediate, w2_fp8, w2_sf);

    // ---- Device alloc ----
    const size_t bytes_x   = static_cast<size_t>(kMaxM) * kHidden;
    const size_t bytes_xs  = static_cast<size_t>(kMaxM) * (kHidden / kGranK);
    const size_t bytes_w1  = static_cast<size_t>(2 * kIntermediate) * kHidden;
    const size_t bytes_w1s = static_cast<size_t>(2 * kIntermediate) * (kHidden / kGranK);
    const size_t bytes_w2  = static_cast<size_t>(kHidden) * kIntermediate;
    const size_t bytes_w2s = static_cast<size_t>(kHidden) * (kIntermediate / kGranK);

    uint8_t *d_x_fp8, *d_x_sf, *d_w1_fp8, *d_w1_sf, *d_w2_fp8, *d_w2_sf;
    CUDA_CHECK(cudaMalloc(&d_x_fp8,  bytes_x));
    CUDA_CHECK(cudaMalloc(&d_x_sf,   bytes_xs));
    CUDA_CHECK(cudaMalloc(&d_w1_fp8, bytes_w1));
    CUDA_CHECK(cudaMalloc(&d_w1_sf,  bytes_w1s));
    CUDA_CHECK(cudaMalloc(&d_w2_fp8, bytes_w2));
    CUDA_CHECK(cudaMalloc(&d_w2_sf,  bytes_w2s));
    CUDA_CHECK(cudaMemcpy(d_x_fp8,  x_fp8.data(),  bytes_x,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_sf,   x_sf.data(),   bytes_xs,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1_fp8, w1_fp8.data(), bytes_w1,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1_sf,  w1_sf.data(),  bytes_w1s, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2_fp8, w2_fp8.data(), bytes_w2,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2_sf,  w2_sf.data(),  bytes_w2s, cudaMemcpyHostToDevice));

    // BF16 工作空间
    nv_bfloat16 *d_x_bf16, *d_w1_bf16, *d_w2_bf16;
    nv_bfloat16 *d_interm_raw_bf16, *d_interm_bf16, *d_y_bf16;
    CUDA_CHECK(cudaMalloc(&d_x_bf16,          static_cast<size_t>(kMaxM) * kHidden * sizeof(nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_w1_bf16,         static_cast<size_t>(2 * kIntermediate) * kHidden * sizeof(nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_w2_bf16,         static_cast<size_t>(kHidden) * kIntermediate * sizeof(nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_interm_raw_bf16, static_cast<size_t>(kMaxM) * 2 * kIntermediate * sizeof(nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_interm_bf16,     static_cast<size_t>(kMaxM) * kIntermediate * sizeof(nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_y_bf16,          static_cast<size_t>(kMaxM) * kHidden * sizeof(nv_bfloat16)));

    // ---- Dequant X / W1 / W2 → BF16 (一次性, 不测时) ----
    {
        dim3 blk(256);
        {
            dim3 grd((kHidden + 255) / 256, kMaxM);
            dequant_fp8_to_bf16<<<grd, blk>>>(d_x_fp8, d_x_sf, d_x_bf16, kMaxM, kHidden);
        }
        {
            dim3 grd((kHidden + 255) / 256, 2 * kIntermediate);
            dequant_fp8_to_bf16<<<grd, blk>>>(d_w1_fp8, d_w1_sf, d_w1_bf16, 2 * kIntermediate, kHidden);
        }
        {
            dim3 grd((kIntermediate + 255) / 256, kHidden);
            dequant_fp8_to_bf16<<<grd, blk>>>(d_w2_fp8, d_w2_sf, d_w2_bf16, kHidden, kIntermediate);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- Build CPU reference ----
    std::vector<nv_bfloat16> y_ref;
    cpu_reference_ffn(x_fp8, x_sf, w1_fp8, w1_sf, w2_fp8, w2_sf, M, y_ref);

    std::vector<nv_bfloat16> y_host(static_cast<size_t>(kMaxM) * kHidden);

    // =============================================================================
    // Baseline A: 最朴素 naive BF16 GEMM (no TC, no SMEM)
    // =============================================================================
    if (std::strcmp(which, "naive") == 0 || std::strcmp(which, "all") == 0) {
        auto run_naive = [&](cudaStream_t s = 0) {
            {
                dim3 blk(32, 8), grd((2 * kIntermediate + 31) / 32, (M + 7) / 8);
                naive_gemm_transB_bf16<<<grd, blk, 0, s>>>(
                    d_x_bf16, d_w1_bf16, d_interm_raw_bf16, M, 2 * kIntermediate, kHidden);
            }
            {
                dim3 blk(256, 1), grd((kIntermediate + 255) / 256, M);
                swiglu_bf16<<<grd, blk, 0, s>>>(d_interm_raw_bf16, d_interm_bf16, M, kIntermediate);
            }
            {
                dim3 blk(32, 8), grd((kHidden + 31) / 32, (M + 7) / 8);
                naive_gemm_transB_bf16<<<grd, blk, 0, s>>>(
                    d_interm_bf16, d_w2_bf16, d_y_bf16, M, kHidden, kIntermediate);
            }
        };

        // warmup
        for (int i = 0; i < 3; ++ i) run_naive();
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        for (uint32_t i = 0; i < num_iters; ++ i) run_naive();
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        CUDA_CHECK(cudaEventDestroy(e0));
        CUDA_CHECK(cudaEventDestroy(e1));

        CUDA_CHECK(cudaMemcpy(y_host.data(), d_y_bf16,
                              static_cast<size_t>(kMaxM) * kHidden * sizeof(nv_bfloat16),
                              cudaMemcpyDeviceToHost));
        compare_and_report("NAIVE-BF16", M, y_host, y_ref, 1000.0 * ms / num_iters, num_iters);
    }

    // =============================================================================
    // Baseline B: cuBLAS BF16 未融合 (cublasGemmEx, Tensor Core)
    // =============================================================================
    if (std::strcmp(which, "cublas") == 0 || std::strcmp(which, "all") == 0) {
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        // 默认即使用 Tensor Core；显式声明 TF32 fallback 允许（BF16 不受影响）
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

        const float alpha = 1.f, beta = 0.f;

        auto run_cublas = [&]() {
            // Linear1: out[M, 2I] = X[M, H] @ W1[2I, H]^T
            //   col-major 等效: C_cm[2I, M] = W1_cm^T[2I, H] × X_cm[H, M]
            CUBLAS_CHECK(cublasGemmEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                2 * kIntermediate, M, kHidden,
                &alpha,
                d_w1_bf16, CUDA_R_16BF, kHidden,
                d_x_bf16,  CUDA_R_16BF, kHidden,
                &beta,
                d_interm_raw_bf16, CUDA_R_16BF, 2 * kIntermediate,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            // SwiGLU (BF16 in/out)
            dim3 blk(256, 1), grd((kIntermediate + 255) / 256, M);
            swiglu_bf16<<<grd, blk>>>(d_interm_raw_bf16, d_interm_bf16, M, kIntermediate);

            // Linear2: y[M, H] = interm[M, I] @ W2[H, I]^T
            CUBLAS_CHECK(cublasGemmEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                kHidden, M, kIntermediate,
                &alpha,
                d_w2_bf16,     CUDA_R_16BF, kIntermediate,
                d_interm_bf16, CUDA_R_16BF, kIntermediate,
                &beta,
                d_y_bf16, CUDA_R_16BF, kHidden,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };

        // warmup
        for (int i = 0; i < 5; ++ i) run_cublas();
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        for (uint32_t i = 0; i < num_iters; ++ i) run_cublas();
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        CUDA_CHECK(cudaEventDestroy(e0));
        CUDA_CHECK(cudaEventDestroy(e1));

        CUDA_CHECK(cudaMemcpy(y_host.data(), d_y_bf16,
                              static_cast<size_t>(kMaxM) * kHidden * sizeof(nv_bfloat16),
                              cudaMemcpyDeviceToHost));
        compare_and_report("CUBLAS-BF16", M, y_host, y_ref, 1000.0 * ms / num_iters, num_iters);

        cublasDestroy(handle);
    }

    // ---- Cleanup ----
    cudaFree(d_x_fp8);  cudaFree(d_x_sf);
    cudaFree(d_w1_fp8); cudaFree(d_w1_sf);
    cudaFree(d_w2_fp8); cudaFree(d_w2_sf);
    cudaFree(d_x_bf16); cudaFree(d_w1_bf16); cudaFree(d_w2_bf16);
    cudaFree(d_interm_raw_bf16); cudaFree(d_interm_bf16); cudaFree(d_y_bf16);
    return 0;
}
