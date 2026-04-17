// =====================================================================================
// tests/cpp/test_sm100_fp8_mega_ffn.cu
//
// 独立 C++ 测试入口，用于校验 deep_gemm/include/deep_gemm/impls/sm100_fp8_mega_ffn.cuh
// 在 B200 (SM100a) 上的正确性与时延。
//
// 构建方式见同目录 build_mega_ffn.sh：
//   ./tests/cpp/build_mega_ffn.sh
//
// 运行参数：
//   ./test_mega_ffn [M=32] [num_iters=50]
//
// 调优建议（需要在真实 B200 上 NCU 迭代）：
//   M  | BLOCK_M | kNumStages | kNumEpilogueThreads | 备注
//   ---+---------+------------+---------------------+-----------------------------
//    1 |   32    |    3       |       128           | UMMA_M 自动 padding 到 64
//    2 |   32    |    3       |       128           | 同上
//    4 |   32    |    3       |       128           |
//    8 |   32    |    4       |       128           | 提升 K-pipeline 吞吐
//   16 |   32    |    4       |       128           |
//   32 |   64    |    4       |       256           | 双 warpgroup epilogue
//
// 设计决策 (v1)：
//   - kClusterDim = 1；intermediate 走 per-CTA HBM workspace（驻留 L2，避免 HBM 带宽消耗）
//   - 2 个 epilogue stage 交错：Linear1 量化写 FP8 → Linear2 消费 FP8
//   - W1 权重在 N=2I 方向布局为 [gate_block0, up_block0, gate_block1, up_block1, ...]
//     （即每 BLOCK_N 内部 N/2 为 gate，N/2 为 up；与 MegaMoE 的 SwiGLU 配对语义一致）
//
// 说明：
//   1) 不依赖 PyTorch / DeepGEMM 的 JIT。
//   2) 构造 11 个 CUtensorMap（TMA descriptor），直接 cuTensorMapEncodeTiled。
//   3) CPU 参考实现：按 UE8M0 block-scale FP8 语义 dequant 后做矩阵乘 + SwiGLU。
//   4) 与 GPU 输出做 max/L1 误差比较。
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

// DeepGEMM kernel
#include <deep_gemm/impls/sm100_fp8_mega_ffn.cuh>

// -----------------------------------------------------------------------------
// 错误处理宏
// -----------------------------------------------------------------------------
#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d : %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        std::exit(1); \
    } \
} while (0)

#define DRV_CHECK(x) do { \
    CUresult _r = (x); \
    if (_r != CUDA_SUCCESS) { \
        const char* msg = nullptr; cuGetErrorString(_r, &msg); \
        std::fprintf(stderr, "CUDA Driver error %s:%d : %s\n", __FILE__, __LINE__, msg ? msg : "(null)"); \
        std::exit(1); \
    } \
} while (0)

// -----------------------------------------------------------------------------
// 形状常量（Qwen3-0.6B）
// -----------------------------------------------------------------------------
constexpr uint32_t kHidden       = 1024;
constexpr uint32_t kIntermediate = 3072;
// kMaxM == BLOCK_M 作为 HBM padding，保证 TMA 在 M 方向读到合法地址；
// runtime 有效 token 数由命令行参数 M 控制，epilogue 里按 valid_m 截断。
constexpr uint32_t kMaxM         = 128;

// SM100 1-CTA MXF8F6F4 block-scaled UMMA 硬件要求 UMMA_M=128（CUTLASS static_assert M==128）。
// 因此 BLOCK_M=128 是 SMEM/寄存器侧的物理 padding；kMaxM=32 是 runtime 的有效 token 上限，
// epilogue 中按 valid_m 截断无效行（不会写出 HBM，也不会读到非法 SF）。
constexpr uint32_t BLOCK_M       = 128;
constexpr uint32_t BLOCK_N       = 128;
constexpr uint32_t BLOCK_K       = 128;
constexpr uint32_t kNumStages    = 3;

constexpr uint32_t kNumNonEpiThreads = 128;
constexpr uint32_t kNumEpiThreads    = 128;  // 4 warps = 128 threads
constexpr uint32_t kClusterDim       = 1;
constexpr uint32_t kNumCTAs          = kHidden / BLOCK_N;   // 8
constexpr uint32_t kL2NPerCta        = 1;                    // (每 CTA 处理 1 个 Linear2 输出 N-tile)

constexpr uint32_t kGranK = 32;

// -----------------------------------------------------------------------------
// TensorMap 构造 helper
// -----------------------------------------------------------------------------
static CUtensorMap make_tma_2d(const char* name,
                               void* gmem_ptr,
                               CUtensorMapDataType dtype,
                               uint64_t gmem_inner, uint64_t gmem_outer,
                               uint32_t smem_inner, uint32_t smem_outer,
                               uint64_t gmem_outer_stride_bytes,
                               CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
    CUtensorMap tm;
    cuuint64_t dims[2]    = {gmem_inner, gmem_outer};
    cuuint64_t strides[1] = {gmem_outer_stride_bytes};
    cuuint32_t box[2]     = {smem_inner, smem_outer};
    cuuint32_t elem_str[2]= {1, 1};

    std::fprintf(stderr,
        "[TMA-2D %s] dims=(%llu,%llu) box=(%u,%u) stride1=%llu swizzle=%d ptr=%p\n",
        name, (unsigned long long)dims[0], (unsigned long long)dims[1],
        box[0], box[1], (unsigned long long)strides[0], (int)swizzle, gmem_ptr);

    CUresult _r = cuTensorMapEncodeTiled(
        &tm, dtype, 2, gmem_ptr, dims, strides, box, elem_str,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (_r != CUDA_SUCCESS) {
        const char* msg = nullptr; cuGetErrorString(_r, &msg);
        std::fprintf(stderr, "[TMA-2D %s] FAILED: %s\n", name, msg ? msg : "(null)");
        std::exit(1);
    }
    return tm;
}

static CUtensorMap make_tma_3d(const char* name,
                               void* gmem_ptr,
                               CUtensorMapDataType dtype,
                               uint64_t dim0, uint64_t dim1, uint64_t dim2,
                               uint32_t box0, uint32_t box1, uint32_t box2,
                               uint64_t stride1_bytes, uint64_t stride2_bytes,
                               CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
    CUtensorMap tm;
    cuuint64_t dims[3]    = {dim0, dim1, dim2};
    cuuint64_t strides[2] = {stride1_bytes, stride2_bytes};
    cuuint32_t box[3]     = {box0, box1, box2};
    cuuint32_t elem_str[3]= {1, 1, 1};

    std::fprintf(stderr,
        "[TMA-3D %s] dims=(%llu,%llu,%llu) box=(%u,%u,%u) stride=(%llu,%llu) swizzle=%d ptr=%p\n",
        name,
        (unsigned long long)dims[0], (unsigned long long)dims[1], (unsigned long long)dims[2],
        box[0], box[1], box[2],
        (unsigned long long)strides[0], (unsigned long long)strides[1], (int)swizzle, gmem_ptr);

    CUresult _r = cuTensorMapEncodeTiled(
        &tm, dtype, 3, gmem_ptr, dims, strides, box, elem_str,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (_r != CUDA_SUCCESS) {
        const char* msg = nullptr; cuGetErrorString(_r, &msg);
        std::fprintf(stderr, "[TMA-3D %s] FAILED: %s\n", name, msg ? msg : "(null)");
        std::exit(1);
    }
    return tm;
}

// -----------------------------------------------------------------------------
// FP8 e4m3 / UE8M0 helper（仅用于 CPU 侧参考实现）
// -----------------------------------------------------------------------------
static inline float fp8_e4m3_to_float(uint8_t x) {
    __nv_fp8_e4m3 v; *reinterpret_cast<uint8_t*>(&v) = x;
    return static_cast<float>(v);
}

static inline uint8_t float_to_fp8_e4m3(float x) {
    __nv_fp8_e4m3 v(x);
    return *reinterpret_cast<uint8_t*>(&v);
}

// ue8m0: 8-bit exponent, value = 2^(e - 127), e == 0 means 0
static inline float ue8m0_to_float(uint8_t e) {
    if (e == 0) return 0.f;
    uint32_t bits = static_cast<uint32_t>(e) << 23;
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

static inline uint8_t float_to_ue8m0_ceil(float x) {
    if (x <= 0.f or !std::isfinite(x)) return 0;
    // ceil(log2(x)) as exponent
    uint32_t bits; std::memcpy(&bits, &x, 4);
    uint32_t exp = (bits >> 23) & 0xff;
    uint32_t man = bits & ((1u << 23) - 1);
    if (man != 0) exp += 1;
    if (exp > 0xff) exp = 0xff;
    return static_cast<uint8_t>(exp);
}

// Host-side：将 M-major uint8 SF `[rows][cols/32]` 转成 K-major uint32
// `[cols/128][rows]`，每个 uint32 打包 4 个沿 K 方向连续的 UE8M0 字节。
// 这是 SM100 block-scaled UMMA 的标准布局，TMA inner=rows 维度 ≥ 16B。
static std::vector<uint32_t> sf_to_kmajor_uint32(const std::vector<uint8_t>& sf_mmajor,
                                                 uint32_t rows, uint32_t cols) {
    const uint32_t kK4 = cols / 128;            // 每 4 个 32-element K-chunk 打成 1 uint32
    std::vector<uint32_t> out(static_cast<size_t>(kK4) * rows, 0u);
    for (uint32_t r = 0; r < rows; ++ r) {
        for (uint32_t k4 = 0; k4 < kK4; ++ k4) {
            uint32_t packed = 0;
            for (uint32_t t = 0; t < 4; ++ t) {
                uint8_t v = sf_mmajor[r * (cols / kGranK) + k4 * 4 + t];
                packed |= static_cast<uint32_t>(v) << (t * 8);
            }
            out[static_cast<size_t>(k4) * rows + r] = packed;
        }
    }
    return out;
}

// 把 BF16 序列量化为 FP8 + UE8M0 scale（每 32 个元素一个 scale）
// 返回 scale 向量（大小 = hidden/32）
static void quantize_fp8_ue8m0(const float* src, uint32_t rows, uint32_t cols,
                               std::vector<uint8_t>& fp8_out,
                               std::vector<uint8_t>& sf_out) {
    fp8_out.resize(static_cast<size_t>(rows) * cols);
    sf_out.resize(static_cast<size_t>(rows) * (cols / kGranK));
    for (uint32_t r = 0; r < rows; ++ r) {
        for (uint32_t kg = 0; kg < cols / kGranK; ++ kg) {
            float amax = 0.f;
            for (uint32_t t = 0; t < kGranK; ++ t) {
                float v = src[r * cols + kg * kGranK + t];
                amax = std::max(amax, std::fabs(v));
            }
            // sf 使 amax / sf ∈ [-448, 448]
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
}

// -----------------------------------------------------------------------------
// CPU 参考：FFN SwiGLU
//   y = down(silu(W1_gate x) * (W1_up x))
// 假定 W1 在 N=2I 方向布局为 [gate_half || up_half] 交错：
//   每 BLOCK_N 内，前 BLOCK_N/2 是 gate_piece，后 BLOCK_N/2 是 up_piece
// 这是为了与 kernel 的 SwiGLU TMEM 配对假设一致。
// -----------------------------------------------------------------------------
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

    // Linear1: interm_raw[m, n] = sum_k X[m,k] * W1[n,k]
    std::vector<float> interm_raw(static_cast<size_t>(M) * (2 * kIntermediate), 0.f);
    for (uint32_t m = 0; m < M; ++ m)
        for (uint32_t n = 0; n < 2 * kIntermediate; ++ n) {
            float acc = 0.f;
            for (uint32_t k = 0; k < kHidden; ++ k)
                acc += X[m * kHidden + k] * W1[n * kHidden + k];
            interm_raw[m * (2 * kIntermediate) + n] = acc;
        }

    // SwiGLU: 对每个 BLOCK_N，前半 gate，后半 up
    //   对应 intermediate[m, block*(BLOCK_N/2) + j] = silu(gate) * up
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

    // Linear2: y[m, h] = sum_i interm[m, i] * W2[h, i]
    y_bf16.assign(static_cast<size_t>(M) * kHidden, nv_bfloat16(0.f));
    for (uint32_t m = 0; m < M; ++ m)
        for (uint32_t h = 0; h < kHidden; ++ h) {
            float acc = 0.f;
            for (uint32_t i = 0; i < kIntermediate; ++ i)
                acc += interm[m * kIntermediate + i] * W2[h * kIntermediate + i];
            y_bf16[m * kHidden + h] = __float2bfloat16(acc);
        }
}

// -----------------------------------------------------------------------------
// 主函数：构造数据、TensorMap、发起 cluster launch、校验
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    uint32_t M          = argc > 1 ? static_cast<uint32_t>(std::atoi(argv[1])) : 1;
    uint32_t num_iters  = argc > 2 ? static_cast<uint32_t>(std::atoi(argv[2])) : 50;

    if (M == 0 or M > kMaxM) {
        std::fprintf(stderr, "Invalid M=%u (1..%u)\n", M, kMaxM);
        return 1;
    }

    // --- Ensure cuInit has been called by the runtime ---
    CUDA_CHECK(cudaFree(nullptr));

    // --- Generate random inputs (BF16-equivalent float) on host ---
    std::mt19937 rng(0xBEEFu);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> x_fp32(static_cast<size_t>(kMaxM) * kHidden, 0.f);
    std::vector<float> w1_fp32(static_cast<size_t>(2 * kIntermediate) * kHidden, 0.f);
    std::vector<float> w2_fp32(static_cast<size_t>(kHidden) * kIntermediate, 0.f);
    for (size_t i = 0; i < static_cast<size_t>(M) * kHidden; ++ i) x_fp32[i] = dist(rng);
    for (auto& v : w1_fp32) v = 0.3f * dist(rng);
    for (auto& v : w2_fp32) v = 0.3f * dist(rng);

    // --- Quantize to FP8 + UE8M0 SF ---
    std::vector<uint8_t> x_fp8, x_sf, w1_fp8, w1_sf, w2_fp8, w2_sf;
    quantize_fp8_ue8m0(x_fp32.data(), kMaxM, kHidden, x_fp8, x_sf);
    quantize_fp8_ue8m0(w1_fp32.data(), 2 * kIntermediate, kHidden, w1_fp8, w1_sf);
    quantize_fp8_ue8m0(w2_fp32.data(), kHidden, kIntermediate, w2_fp8, w2_sf);

    // --- Device allocations ---
    uint8_t* d_x       = nullptr;
    uint8_t* d_x_sf    = nullptr;
    uint8_t* d_w1      = nullptr;
    uint8_t* d_w1_sf   = nullptr;
    uint8_t* d_w2      = nullptr;
    uint8_t* d_w2_sf   = nullptr;
    nv_bfloat16* d_y   = nullptr;
    uint8_t* d_ws      = nullptr;    // per-cta intermediate FP8 [num_ctas, kMaxM, kIntermediate]
    uint8_t* d_ws_sf   = nullptr;    // per-cta intermediate SF  [num_ctas, kMaxM, kIntermediate/32]

    const size_t bytes_x  = static_cast<size_t>(kMaxM) * kHidden;
    const size_t bytes_xs = static_cast<size_t>(kMaxM) * (kHidden / kGranK);
    const size_t bytes_w1 = static_cast<size_t>(2 * kIntermediate) * kHidden;
    const size_t bytes_w1s= static_cast<size_t>(2 * kIntermediate) * (kHidden / kGranK);
    const size_t bytes_w2 = static_cast<size_t>(kHidden) * kIntermediate;
    const size_t bytes_w2s= static_cast<size_t>(kHidden) * (kIntermediate / kGranK);
    const size_t bytes_y  = static_cast<size_t>(kMaxM) * kHidden * sizeof(nv_bfloat16);
    const size_t bytes_ws = static_cast<size_t>(kNumCTAs) * kMaxM * kIntermediate;
    const size_t bytes_ws_sf = static_cast<size_t>(kNumCTAs) * kMaxM * (kIntermediate / kGranK);

    CUDA_CHECK(cudaMalloc(&d_x,     bytes_x));
    CUDA_CHECK(cudaMalloc(&d_x_sf,  bytes_xs));
    CUDA_CHECK(cudaMalloc(&d_w1,    bytes_w1));
    CUDA_CHECK(cudaMalloc(&d_w1_sf, bytes_w1s));
    CUDA_CHECK(cudaMalloc(&d_w2,    bytes_w2));
    CUDA_CHECK(cudaMalloc(&d_w2_sf, bytes_w2s));
    CUDA_CHECK(cudaMalloc(&d_y,     bytes_y));
    CUDA_CHECK(cudaMalloc(&d_ws,    bytes_ws));
    CUDA_CHECK(cudaMalloc(&d_ws_sf, bytes_ws_sf));

    CUDA_CHECK(cudaMemcpy(d_x,     x_fp8.data(),  bytes_x,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1,    w1_fp8.data(), bytes_w1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2,    w2_fp8.data(), bytes_w2, cudaMemcpyHostToDevice));

    // SF: host 量化出的是 M-major uint8，UMMA + TMA 需要 K-major uint32。
    //   X_sf:  [kMaxM, H/32]  →  [H/128, kMaxM]   uint32
    //   W1_sf: [2I, H/32]     →  [H/128, 2I]      uint32
    //   W2_sf: [H, I/32]      →  [I/128, H]       uint32
    auto x_sf_km  = sf_to_kmajor_uint32(x_sf,  kMaxM,              kHidden);
    auto w1_sf_km = sf_to_kmajor_uint32(w1_sf, 2 * kIntermediate,  kHidden);
    auto w2_sf_km = sf_to_kmajor_uint32(w2_sf, kHidden,            kIntermediate);
    CUDA_CHECK(cudaMemcpy(d_x_sf,  x_sf_km.data(),  bytes_xs, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1_sf, w1_sf_km.data(), bytes_w1s,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2_sf, w2_sf_km.data(), bytes_w2s,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, bytes_y));
    CUDA_CHECK(cudaMemset(d_ws, 0, bytes_ws));
    CUDA_CHECK(cudaMemset(d_ws_sf, 0, bytes_ws_sf));

    // --- Build CUtensorMap descriptors ---
    // X: [M, H] FP8, swizzle 128B (BLOCK_K bytes)
    auto tma_x = make_tma_2d("X", d_x, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                             kHidden, kMaxM,
                             BLOCK_K, BLOCK_M,
                             static_cast<uint64_t>(kHidden) * 1,
                             CU_TENSOR_MAP_SWIZZLE_128B);
    // X_sf: K-major uint32 布局 [H/128, M] —— inner=M (连续 ≥16B 满足 TMA), outer=K/128。
    // 每次 TMA 读一个 k128 列的 SF_BLOCK_M=128 tokens，共 128 × 4 = 512 字节。
    auto tma_x_sf = make_tma_2d("X_sf", d_x_sf, CU_TENSOR_MAP_DATA_TYPE_UINT32,
                                kMaxM, kHidden / 128,
                                128, 1,
                                static_cast<uint64_t>(kMaxM) * sizeof(uint32_t));

    // W1: [2I, H] FP8
    auto tma_w1 = make_tma_2d("W1", d_w1, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                              kHidden, 2 * kIntermediate,
                              BLOCK_K, BLOCK_N,
                              static_cast<uint64_t>(kHidden),
                              CU_TENSOR_MAP_SWIZZLE_128B);
    // W1_sf: K-major uint32 布局 [H/128, 2I] —— inner=2I, outer=K/128。
    //   每次 TMA 读一个 k128 列的 BLOCK_N 行，共 BLOCK_N × 4 = 512 字节。
    auto tma_w1_sf = make_tma_2d("W1_sf", d_w1_sf, CU_TENSOR_MAP_DATA_TYPE_UINT32,
                                 2 * kIntermediate, kHidden / 128,
                                 BLOCK_N, 1,
                                 static_cast<uint64_t>(2 * kIntermediate) * sizeof(uint32_t));

    // W2: [H, I] FP8
    auto tma_w2 = make_tma_2d("W2", d_w2, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                              kIntermediate, kHidden,
                              BLOCK_K, BLOCK_N,
                              static_cast<uint64_t>(kIntermediate),
                              CU_TENSOR_MAP_SWIZZLE_128B);
    // W2_sf: K-major uint32 布局 [I/128, H] —— inner=H, outer=I/128。
    auto tma_w2_sf = make_tma_2d("W2_sf", d_w2_sf, CU_TENSOR_MAP_DATA_TYPE_UINT32,
                                 kHidden, kIntermediate / 128,
                                 BLOCK_N, 1,
                                 static_cast<uint64_t>(kHidden) * sizeof(uint32_t));

    // workspace intermediate FP8 [num_ctas, kMaxM, kIntermediate]
    //   写入视图 (SM90_TMA_STORE_3D)：inner=I, outer=M, batch=cta_idx
    //   tile  = (L1_OUT_BLOCK_N, BLOCK_M, 1) -- 但 TMA store 固定 box = (L1_OUT_BLOCK_N, STORE_BLOCK_M)
    auto tma_interm = make_tma_3d("interm_w", d_ws, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                  kIntermediate, kMaxM, kNumCTAs,
                                  BLOCK_N / 2, BLOCK_M, 1,
                                  static_cast<uint64_t>(kIntermediate),
                                  static_cast<uint64_t>(kMaxM) * kIntermediate,
                                  CU_TENSOR_MAP_SWIZZLE_64B);
    // 读视图：给 Linear2 TMA-A 用，box = (BLOCK_K, BLOCK_M)
    auto tma_interm_load = make_tma_3d("interm_r", d_ws, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                       kIntermediate, kMaxM, kNumCTAs,
                                       BLOCK_K, BLOCK_M, 1,
                                       static_cast<uint64_t>(kIntermediate),
                                       static_cast<uint64_t>(kMaxM) * kIntermediate,
                                       CU_TENSOR_MAP_SWIZZLE_128B);

    // workspace SF: K-major uint32 布局 [num_ctas, I/128, kMaxM]
    //   写视图（当前 epilogue 走字节写，不经 TMA；保留 descriptor 仅作占位）
    //   读视图：Linear2 TMA-A 的 SFA 源 —— inner=M, outer=I/128, batch=num_ctas
    auto tma_interm_sf = make_tma_3d("interm_sf_w", d_ws_sf, CU_TENSOR_MAP_DATA_TYPE_UINT32,
                                     kMaxM, kIntermediate / 128, kNumCTAs,
                                     128, 1, 1,
                                     static_cast<uint64_t>(kMaxM) * sizeof(uint32_t),
                                     static_cast<uint64_t>(kMaxM) * (kIntermediate / 128) * sizeof(uint32_t));
    auto tma_interm_sf_load = make_tma_3d("interm_sf_r", d_ws_sf, CU_TENSOR_MAP_DATA_TYPE_UINT32,
                                          kMaxM, kIntermediate / 128, kNumCTAs,
                                          128, 1, 1,
                                          static_cast<uint64_t>(kMaxM) * sizeof(uint32_t),
                                          static_cast<uint64_t>(kMaxM) * (kIntermediate / 128) * sizeof(uint32_t));

    // Y: [M, H] BF16 —— swizzle 128B 时 box[0]*2 必须 = 128 字节，故 box[0]=64.
    // 当前 kernel epilogue 对 BLOCK_N=128 只发一次 TMA store，先用 SWIZZLE_NONE 跑通，
    // 后续待与 epilogue STSM 对齐再切回 swizzle.
    auto tma_y = make_tma_2d("Y", d_y, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                             kHidden, kMaxM,
                             BLOCK_N, BLOCK_M,
                             static_cast<uint64_t>(kHidden) * sizeof(nv_bfloat16),
                             CU_TENSOR_MAP_SWIZZLE_NONE);

    // --- Kernel launch ---
    auto kernel = &deep_gemm::sm100_fp8_mega_ffn_impl<
        kHidden, kIntermediate, kMaxM,
        BLOCK_M, BLOCK_N, BLOCK_K,
        kNumStages,
        kNumNonEpiThreads, kNumEpiThreads,
        kClusterDim>;

    // 计算 dynamic smem size (与 kernel 中一致)
    constexpr uint32_t UMMA_N = BLOCK_N;
    constexpr uint32_t STORE_BLOCK_M_ = BLOCK_M;
    constexpr uint32_t SF_BLOCK_M_    = 128;
    constexpr uint32_t SF_BLOCK_N_    = BLOCK_N;
    constexpr uint32_t kNumEpiWarps_  = kNumEpiThreads / 32;
    constexpr uint32_t kNumEpiWG_     = kNumEpiWarps_ / 4;
    constexpr uint32_t L1_OUT_BLOCK_N_= BLOCK_N / 2;

    constexpr uint32_t SMEM_CD_L1_ = kNumEpiWG_ * STORE_BLOCK_M_ * L1_OUT_BLOCK_N_ * 1 * 2;
    constexpr uint32_t SMEM_CD_L2_ = kNumEpiWG_ * STORE_BLOCK_M_ * BLOCK_N * sizeof(nv_bfloat16) * 2;
    constexpr uint32_t SMEM_CD_    = SMEM_CD_L1_ > SMEM_CD_L2_ ? SMEM_CD_L1_ : SMEM_CD_L2_;
    constexpr uint32_t SMEM_A_     = BLOCK_M * BLOCK_K * 1;
    constexpr uint32_t SMEM_B_     = BLOCK_N * BLOCK_K * 1;
    constexpr uint32_t SMEM_SFA_   = SF_BLOCK_M_ * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_   = SF_BLOCK_N_ * sizeof(uint32_t);
    constexpr uint32_t SMEM_AMAX_  = kNumEpiWarps_ * (STORE_BLOCK_M_ / 2) * sizeof(float2);
    constexpr uint32_t SMEM_BAR_   = (2 * kNumStages + 2 * 2) * 8 + 16;  // 粗估, 每 barrier 8B + tmem_ptr
    constexpr uint32_t smem_size   = SMEM_CD_
                                   + kNumStages * (SMEM_A_ + SMEM_B_)
                                   + kNumStages * (SMEM_SFA_ + SMEM_SFB_)
                                   + SMEM_AMAX_ + SMEM_BAR_;

    // 在大于 48KB 时必须开启 opt-in
    std::printf("[kernel] smem_size=%u bytes (%.2f KB), SMEM_CD=%u, SMEM_A/B=%u, stages=%u\n",
                smem_size, smem_size / 1024.0,
                SMEM_CD_, SMEM_A_, kNumStages);
    {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        int max_dyn_smem = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&max_dyn_smem,
            cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        std::printf("[kernel] max dynamic SMEM per block opt-in = %d bytes (%.2f KB)\n",
                    max_dyn_smem, max_dyn_smem / 1024.0);
    }
    if (smem_size > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    // Launch with cluster dim = 1 (v1)
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim  = dim3(kNumCTAs, 1, 1);
    cfg.blockDim = dim3(kNumNonEpiThreads + kNumEpiThreads, 1, 1);
    cfg.dynamicSmemBytes = smem_size;
    cfg.stream = 0;

    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim = {kClusterDim, 1, 1};
    cfg.attrs = attr;
    cfg.numAttrs = 1;

    // Warmup + correctness
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel,
        static_cast<void*>(d_y),
        static_cast<void*>(d_ws),
        static_cast<void*>(d_ws_sf),
        M,
        tma_x, tma_x_sf, tma_w1, tma_w1_sf, tma_w2, tma_w2_sf,
        tma_interm, tma_interm_load, tma_interm_sf, tma_interm_sf_load,
        tma_y));
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Validate against CPU reference ---
    std::vector<nv_bfloat16> y_ref;
    cpu_reference_ffn(x_fp8, x_sf, w1_fp8, w1_sf, w2_fp8, w2_sf, M, y_ref);

    std::vector<nv_bfloat16> y_gpu(static_cast<size_t>(kMaxM) * kHidden);
    CUDA_CHECK(cudaMemcpy(y_gpu.data(), d_y, bytes_y, cudaMemcpyDeviceToHost));

    double l1 = 0.0, mx = 0.0;
    size_t nonzero = 0;
    for (uint32_t m = 0; m < M; ++ m)
        for (uint32_t h = 0; h < kHidden; ++ h) {
            float g = __bfloat162float(y_gpu[m * kHidden + h]);
            float r = __bfloat162float(y_ref[m * kHidden + h]);
            float d = std::fabs(g - r);
            l1 += d;
            mx = std::max<double>(mx, d);
            if (std::fabs(r) > 1e-4f) ++ nonzero;
        }
    double denom = static_cast<double>(M) * kHidden;
    std::printf("[MegaFFN] M=%u  mean|Δ|=%.4f  max|Δ|=%.4f  nonzero_ref=%zu/%.0f\n",
                M, l1 / denom, mx, nonzero, denom);

    // --- Timing ---
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    // warmup
    for (int i = 0; i < 5; ++ i) {
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel,
            static_cast<void*>(d_y), static_cast<void*>(d_ws), static_cast<void*>(d_ws_sf),
            M,
            tma_x, tma_x_sf, tma_w1, tma_w1_sf, tma_w2, tma_w2_sf,
            tma_interm, tma_interm_load, tma_interm_sf, tma_interm_sf_load,
            tma_y));
    }
    CUDA_CHECK(cudaEventRecord(e0));
    for (uint32_t i = 0; i < num_iters; ++ i) {
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel,
            static_cast<void*>(d_y), static_cast<void*>(d_ws), static_cast<void*>(d_ws_sf),
            M,
            tma_x, tma_x_sf, tma_w1, tma_w1_sf, tma_w2, tma_w2_sf,
            tma_interm, tma_interm_load, tma_interm_sf, tma_interm_sf_load,
            tma_y));
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    std::printf("[MegaFFN] avg latency = %.3f us  (over %u iters)\n",
                1000.0 * ms / num_iters, num_iters);

    // --- Cleanup ---
    cudaFree(d_x); cudaFree(d_x_sf);
    cudaFree(d_w1); cudaFree(d_w1_sf);
    cudaFree(d_w2); cudaFree(d_w2_sf);
    cudaFree(d_y);
    cudaFree(d_ws); cudaFree(d_ws_sf);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 0;
}
