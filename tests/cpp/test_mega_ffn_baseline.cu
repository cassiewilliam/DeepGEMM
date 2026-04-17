// =====================================================================================
// tests/cpp/test_mega_ffn_baseline.cu
//
// MegaFFN baseline: cuBLAS FP16 未融合 (cublasGemmEx FP16 × Tensor Core)
//   2 × GEMM(FP16) + 1 × SwiGLU(FP16) — 作为与 MegaFFN kernel 对比的强 baseline
//
// 输入种子 (0xBEEF) 与 test_sm100_fp8_mega_ffn.cu 一致，保证公平对比。
// 权重/激活在 FP8+UE8M0 量化后一次性 dequant 到 FP16 (不计入 timing)，再跑 FP16 GEMM。
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
#include <cuda_fp16.h>
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
// 形状常量 (Qwen3-0.6B) —— 与主测试一致
// -----------------------------------------------------------------------------
constexpr uint32_t kHidden       = 1024;
constexpr uint32_t kIntermediate = 3072;
constexpr uint32_t kMaxM         = 128;
constexpr uint32_t BLOCK_N       = 128;
constexpr uint32_t kGranK        = 32;

// -----------------------------------------------------------------------------
// Host 端 FP8 / UE8M0 helpers (与主测试同)
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

// Dequant fp8 + ue8m0 → fp16，row-major [rows, cols]
__global__ void dequant_fp8_to_fp16(
    const uint8_t* __restrict__ fp8, const uint8_t* __restrict__ sf,
    __half* __restrict__ out, uint32_t rows, uint32_t cols)
{
    const uint32_t r = blockIdx.y;
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || j >= cols) return;
    const uint32_t kg    = j / kGranK;
    const uint8_t  sf_b  = sf[r * (cols / kGranK) + kg];
    const float    scale = sf_b == 0 ? 0.f : __uint_as_float(static_cast<uint32_t>(sf_b) << 23);
    __nv_fp8_e4m3 v; *reinterpret_cast<uint8_t*>(&v) = fp8[r * cols + j];
    out[r * cols + j] = __float2half(static_cast<float>(v) * scale);
}

// SwiGLU(FP16 in/out, FP32 accumulate): in_raw[M, 2I] → out[M, I]
// 每 BLOCK_N 内前半=gate 后半=up, out[m, b*(BLOCK_N/2) + j] = silu(g)*u
__global__ void swiglu_fp16(
    const __half* __restrict__ in_raw, __half* __restrict__ out,
    uint32_t M, uint32_t I)
{
    const uint32_t m = blockIdx.y;
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || j >= I) return;
    const uint32_t block_n = j / (BLOCK_N / 2);
    const uint32_t jj      = j % (BLOCK_N / 2);
    const uint32_t g_idx   = m * 2 * I + block_n * BLOCK_N + jj;
    const uint32_t u_idx   = g_idx + BLOCK_N / 2;
    const float g = __half2float(in_raw[g_idx]);
    const float u = __half2float(in_raw[u_idx]);
    const float silu_g = g / (1.f + __expf(-g));
    out[m * I + j] = __float2half(silu_g * u);
}

// -----------------------------------------------------------------------------
// CPU 参考: FP8-faithful (与主 test 一致)
// -----------------------------------------------------------------------------
static inline float fp8_e4m3_to_float(uint8_t x) {
    __nv_fp8_e4m3 v; *reinterpret_cast<uint8_t*>(&v) = x;
    return static_cast<float>(v);
}
static void cpu_reference_ffn(
    const std::vector<uint8_t>& x_fp8, const std::vector<uint8_t>& x_sf,
    const std::vector<uint8_t>& w1_fp8, const std::vector<uint8_t>& w1_sf,
    const std::vector<uint8_t>& w2_fp8, const std::vector<uint8_t>& w2_sf,
    uint32_t M, std::vector<__half>& y_fp16)
{
    auto deq = [&](const std::vector<uint8_t>& a, const std::vector<uint8_t>& s,
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
    auto X  = deq(x_fp8,  x_sf,  M,                   kHidden);
    auto W1 = deq(w1_fp8, w1_sf, 2 * kIntermediate,   kHidden);
    auto W2 = deq(w2_fp8, w2_sf, kHidden,             kIntermediate);
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
    auto interm_dq = deq(interm_fp8, interm_sf, M, kIntermediate);
    y_fp16.assign(static_cast<size_t>(M) * kHidden, __half(0.f));
    for (uint32_t m = 0; m < M; ++ m)
        for (uint32_t h = 0; h < kHidden; ++ h) {
            float acc = 0.f;
            for (uint32_t i = 0; i < kIntermediate; ++ i)
                acc += interm_dq[m * kIntermediate + i] * W2[h * kIntermediate + i];
            y_fp16[m * kHidden + h] = __float2half(acc);
        }
}

static void compare_and_report(const char* tag, uint32_t M,
                               const std::vector<__half>& y_gpu,
                               const std::vector<__half>& y_ref,
                               double us_per_iter, uint32_t num_iters)
{
    double l1 = 0.0, mx = 0.0;
    for (uint32_t m = 0; m < M; ++ m)
        for (uint32_t h = 0; h < kHidden; ++ h) {
            float g = __half2float(y_gpu[m * kHidden + h]);
            float r = __half2float(y_ref[m * kHidden + h]);
            float d = std::fabs(g - r);
            l1 += d;
            mx = std::max<double>(mx, d);
        }
    double denom = static_cast<double>(M) * kHidden;
    std::printf("[%s] M=%u  mean|Δ|=%.4f  max|Δ|=%.4f   latency=%.3f us (%u iters)\n",
                tag, M, l1 / denom, mx, us_per_iter, num_iters);
}

int main(int argc, char** argv) {
    uint32_t M         = argc > 1 ? static_cast<uint32_t>(std::atoi(argv[1])) : 1;
    uint32_t num_iters = argc > 2 ? static_cast<uint32_t>(std::atoi(argv[2])) : 100;

    if (M == 0 or M > kMaxM) {
        std::fprintf(stderr, "Invalid M=%u (1..%u)\n", M, kMaxM);
        return 1;
    }

    CUDA_CHECK(cudaFree(nullptr));

    // ---- 与主测试一致的输入 (seed 0xBEEF) ----
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

    __half *d_x_fp16, *d_w1_fp16, *d_w2_fp16;
    __half *d_interm_raw_fp16, *d_interm_fp16, *d_y_fp16;
    CUDA_CHECK(cudaMalloc(&d_x_fp16,          static_cast<size_t>(kMaxM) * kHidden * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_w1_fp16,         static_cast<size_t>(2 * kIntermediate) * kHidden * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_w2_fp16,         static_cast<size_t>(kHidden) * kIntermediate * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_interm_raw_fp16, static_cast<size_t>(kMaxM) * 2 * kIntermediate * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_interm_fp16,     static_cast<size_t>(kMaxM) * kIntermediate * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_y_fp16,          static_cast<size_t>(kMaxM) * kHidden * sizeof(__half)));

    // ---- Dequant → FP16 (一次性, 不测时) ----
    {
        dim3 blk(256);
        {
            dim3 grd((kHidden + 255) / 256, kMaxM);
            dequant_fp8_to_fp16<<<grd, blk>>>(d_x_fp8, d_x_sf, d_x_fp16, kMaxM, kHidden);
        }
        {
            dim3 grd((kHidden + 255) / 256, 2 * kIntermediate);
            dequant_fp8_to_fp16<<<grd, blk>>>(d_w1_fp8, d_w1_sf, d_w1_fp16, 2 * kIntermediate, kHidden);
        }
        {
            dim3 grd((kIntermediate + 255) / 256, kHidden);
            dequant_fp8_to_fp16<<<grd, blk>>>(d_w2_fp8, d_w2_sf, d_w2_fp16, kHidden, kIntermediate);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- CPU reference ----
    std::vector<__half> y_ref;
    cpu_reference_ffn(x_fp8, x_sf, w1_fp8, w1_sf, w2_fp8, w2_sf, M, y_ref);

    std::vector<__half> y_host(static_cast<size_t>(kMaxM) * kHidden);

    // =============================================================================
    // Baseline: cuBLAS FP16 未融合
    // =============================================================================
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    const __half alpha = __float2half(1.f);
    const __half beta  = __float2half(0.f);

    auto run_cublas = [&]() {
        // Linear1: out[M, 2I] = X[M, H] @ W1[2I, H]^T
        //   col-major 等效: C_cm[2I, M] = W1_cm^T × X_cm (ldW=H, ldX=H, ldC=2I)
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            2 * kIntermediate, M, kHidden,
            &alpha,
            d_w1_fp16, CUDA_R_16F, kHidden,
            d_x_fp16,  CUDA_R_16F, kHidden,
            &beta,
            d_interm_raw_fp16, CUDA_R_16F, 2 * kIntermediate,
            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // SwiGLU (FP16 in/out)
        dim3 blk(256, 1), grd((kIntermediate + 255) / 256, M);
        swiglu_fp16<<<grd, blk>>>(d_interm_raw_fp16, d_interm_fp16, M, kIntermediate);

        // Linear2: y[M, H] = interm[M, I] @ W2[H, I]^T
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            kHidden, M, kIntermediate,
            &alpha,
            d_w2_fp16,     CUDA_R_16F, kIntermediate,
            d_interm_fp16, CUDA_R_16F, kIntermediate,
            &beta,
            d_y_fp16, CUDA_R_16F, kHidden,
            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

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

    CUDA_CHECK(cudaMemcpy(y_host.data(), d_y_fp16,
                          static_cast<size_t>(kMaxM) * kHidden * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    compare_and_report("CUBLAS-FP16", M, y_host, y_ref, 1000.0 * ms / num_iters, num_iters);

    cublasDestroy(handle);
    cudaFree(d_x_fp8);  cudaFree(d_x_sf);
    cudaFree(d_w1_fp8); cudaFree(d_w1_sf);
    cudaFree(d_w2_fp8); cudaFree(d_w2_sf);
    cudaFree(d_x_fp16); cudaFree(d_w1_fp16); cudaFree(d_w2_fp16);
    cudaFree(d_interm_raw_fp16); cudaFree(d_interm_fp16); cudaFree(d_y_fp16);
    return 0;
}
