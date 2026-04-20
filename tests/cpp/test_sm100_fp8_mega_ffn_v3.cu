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

// DeepGEMM kernel — **v3.0** = Swap AB + Per-Tensor FP8 + Pre-merged W1 + L2 slot layout [cta][h][m]
#include <deep_gemm/impls/sm100_fp8_mega_ffn_v3.cuh>

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

#ifndef MFFN_STAGES
#define MFFN_STAGES 4
#endif
#ifndef MFFN_EPI_THREADS
#define MFFN_EPI_THREADS 128
#endif
#ifndef MFFN_CLUSTER_DIM
// Step 15: 默认 cluster=kL2KSplit=8，post_l2_sync 走 hardware cluster_sync
#define MFFN_CLUSTER_DIM 8
#endif
// Step 3/4：Linear2 K 拆分份数。=1 回到 Step 2；>=2 gridDim = 8*kL2KSplit，用 fp32 atomicAdd 合并。
// Step 4 放松 L1 N-tile 整除约束 + 升级 grid_sync (bit31 flip + ld.acq 轮询) 后，
// kL2KSplit=8 (gridDim=64) 在 M=1..32 上最优，~14% 快于 Step 3，已追平 cuBLAS FP16 unfused。
#ifndef MFFN_L2_K_SPLIT
#define MFFN_L2_K_SPLIT 8
#endif

constexpr uint32_t kNumStages    = MFFN_STAGES;

constexpr uint32_t kNumNonEpiThreads = 128;
constexpr uint32_t kNumEpiThreads    = MFFN_EPI_THREADS;  // 4 warps = 128 threads
constexpr uint32_t kClusterDim       = MFFN_CLUSTER_DIM;
constexpr uint32_t kL2KSplit         = MFFN_L2_K_SPLIT;
constexpr uint32_t kNumCTAs          = (kHidden / BLOCK_N) * kL2KSplit;  // 8 or 16
constexpr uint32_t kL2NPerCta        = 1;                    // (每 CTA 处理 1 个 Linear2 输出 N-tile)

// Step 3：fp32 → BF16 cast kernel（K-split 合并后执行一次）
__global__ void cast_y_fp32_to_bf16_kernel(nv_bfloat16* __restrict__ y_bf16,
                                           const float* __restrict__ y_fp32,
                                           uint32_t M, uint32_t H) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = M * H;
    if (tid >= total) return;
    y_bf16[tid] = __float2bfloat16(y_fp32[tid]);
}

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

// **v3** Pre-merged W1 layout 重排: W1 [2I, H], 把每 BLOCK_N=128 行块从 [gate(0..63), up(0..63)]
// 重排为 [g0..15, u0..15, g16..31, u16..31, g32..47, u32..47, g48..63, u48..63].
// 这样 swap MMA 后 TMEM 每 warp 的 32 行内同时有 16 gate + 16 up, 配对靠 intra-warp shfl_xor(16).
// In-place 不行 (W1 太大), 输出新 buffer.
static std::vector<float> permute_w1_premerged(const std::vector<float>& w1_orig,
                                               uint32_t total_n, uint32_t hidden) {
    std::vector<float> w1_new(w1_orig.size());
    for (uint32_t b = 0; b * BLOCK_N < total_n; ++ b) {           // n_block index
        const uint32_t base = b * BLOCK_N;
        for (uint32_t w = 0; w < 4; ++ w) {                        // sub-block
            for (uint32_t j = 0; j < 16; ++ j) {                   // offset within sub-block
                const uint32_t new_gate_row = base + w * 32 + j;
                const uint32_t new_up_row   = base + w * 32 + 16 + j;
                const uint32_t old_gate_row = base + w * 16 + j;          // gate in [0..63] of block
                const uint32_t old_up_row   = base + 64 + w * 16 + j;     // up in [64..127] of block
                std::memcpy(&w1_new[new_gate_row * hidden],
                            &w1_orig[old_gate_row * hidden],
                            hidden * sizeof(float));
                std::memcpy(&w1_new[new_up_row * hidden],
                            &w1_orig[old_up_row * hidden],
                            hidden * sizeof(float));
            }
        }
    }
    return w1_new;
}

// **Per-Tensor FP8 量化**: 整张 tensor 单一 scale, scale = max|x| / 448, fp8[i] = round(x[i] / scale).
// 返回 scale (host 端在 CPU ref + kernel launch 时都需要).
static float quantize_fp8_pt(const float* src, size_t n,
                             std::vector<uint8_t>& fp8_out) {
    float amax = 0.f;
    for (size_t i = 0; i < n; ++ i) amax = std::max(amax, std::fabs(src[i]));
    const float scale = amax > 0.f ? amax / 448.f : 1.f;
    const float inv   = 1.f / scale;
    fp8_out.resize(n);
    for (size_t i = 0; i < n; ++ i)
        fp8_out[i] = float_to_fp8_e4m3(src[i] * inv);
    return scale;
}

// -----------------------------------------------------------------------------
// **PT** CPU 参考: Linear1 + SwiGLU + Per-Tensor FP8 量化 intermediate.
// 输入: per-tensor FP8 X / W1 + 各自 scale.
// 输出: ws_fp8[num_ctas, kMaxM, kIntermediate] (跨 CTA broadcast),
//       interm_out (real fp32, 用作下游 Linear2 ref),
//       返回值: scale_intermediate (kernel 端需要 1/scale 作为 scale_inv_intermediate).
// -----------------------------------------------------------------------------
static float cpu_reference_l1_workspace_pt(
    const std::vector<uint8_t>& x_fp8, float scale_x,
    const std::vector<uint8_t>& w1_fp8, float scale_w1,
    uint32_t M, uint32_t num_ctas,
    std::vector<uint8_t>& ws_fp8,
    std::vector<float>& interm_out)
{
    auto deq = [&](const std::vector<uint8_t>& src, float scale, size_t n) {
        std::vector<float> out(n);
        for (size_t i = 0; i < n; ++ i) out[i] = fp8_e4m3_to_float(src[i]) * scale;
        return out;
    };
    auto X  = deq(x_fp8, scale_x, static_cast<size_t>(M) * kHidden);
    auto W1 = deq(w1_fp8, scale_w1, static_cast<size_t>(2 * kIntermediate) * kHidden);

    // Linear1
    std::vector<float> interm_raw(static_cast<size_t>(M) * (2 * kIntermediate), 0.f);
    for (uint32_t m = 0; m < M; ++ m)
        for (uint32_t n = 0; n < 2 * kIntermediate; ++ n) {
            float acc = 0.f;
            for (uint32_t k = 0; k < kHidden; ++ k)
                acc += X[m * kHidden + k] * W1[n * kHidden + k];
            interm_raw[m * (2 * kIntermediate) + n] = acc;
        }

    // **v3 SwiGLU**: 适配 pre-merged W1 layout.
    // 每 BLOCK_N=128 块内的 N-position layout: [g0..15, u0..15, g16..31, u16..31, g32..47, u32..47, g48..63, u48..63]
    // 即每 32-row sub-block 前 16 是 gate 后 16 是 up. 分 4 sub-blocks/block, 每 sub 输出 16 个 output_n.
    interm_out.assign(static_cast<size_t>(M) * kIntermediate, 0.f);
    auto silu = [](float v) { return v / (1.f + std::exp(-v)); };
    const uint32_t n_blocks = (2 * kIntermediate) / BLOCK_N;
    for (uint32_t m = 0; m < M; ++ m)
        for (uint32_t b = 0; b < n_blocks; ++ b)
            for (uint32_t w = 0; w < 4; ++ w)              // sub-block within BLOCK_N=128
                for (uint32_t j = 0; j < 16; ++ j) {        // offset within sub-block
                    const uint32_t output_n = b * 64 + w * 16 + j;
                    const uint32_t g_pos    = b * BLOCK_N + w * 32 + j;       // gate row in interm_raw
                    const uint32_t u_pos    = b * BLOCK_N + w * 32 + 16 + j;  // up row
                    float g = interm_raw[m * (2 * kIntermediate) + g_pos];
                    float u = interm_raw[m * (2 * kIntermediate) + u_pos];
                    interm_out[m * kIntermediate + output_n] = silu(g) * u;
                }

    // Per-Tensor 量化 intermediate → FP8 (single scale)
    std::vector<uint8_t> one_fp8;
    const float scale_intermediate = quantize_fp8_pt(interm_out.data(),
                                                     static_cast<size_t>(M) * kIntermediate,
                                                     one_fp8);

    // 广播到 num_ctas 个 CTA, 仅 valid m 行有数据 (rest = 0 from assign())
    const size_t per_cta_fp8 = static_cast<size_t>(kMaxM) * kIntermediate;
    ws_fp8.assign(per_cta_fp8 * num_ctas, 0);
    for (uint32_t c = 0; c < num_ctas; ++ c)
        for (uint32_t m = 0; m < M; ++ m)
            std::memcpy(&ws_fp8[c * per_cta_fp8 + m * kIntermediate],
                        &one_fp8[m * kIntermediate], kIntermediate);
    return scale_intermediate;
}

// -----------------------------------------------------------------------------
// **PT** CPU 参考: Linear2 only (Linear1 + SwiGLU 已在 cpu_reference_l1_workspace_pt 算完).
//   y = Linear2(intermediate_fp8 dequant, W2 dequant)
// 输入 intermediate_fp8 来自 L1 ref 的 per-tensor 量化结果, scale_intermediate 也来自那里.
static void cpu_reference_ffn_pt(
    const std::vector<uint8_t>& interm_fp8, float scale_intermediate,
    const std::vector<uint8_t>& w2_fp8, float scale_w2,
    uint32_t M, std::vector<nv_bfloat16>& y_bf16)
{
    std::vector<float> Interm(static_cast<size_t>(M) * kIntermediate);
    for (size_t i = 0; i < Interm.size(); ++ i)
        Interm[i] = fp8_e4m3_to_float(interm_fp8[i]) * scale_intermediate;
    std::vector<float> W2(static_cast<size_t>(kHidden) * kIntermediate);
    for (size_t i = 0; i < W2.size(); ++ i)
        W2[i] = fp8_e4m3_to_float(w2_fp8[i]) * scale_w2;

    y_bf16.assign(static_cast<size_t>(M) * kHidden, nv_bfloat16(0.f));
    for (uint32_t m = 0; m < M; ++ m)
        for (uint32_t h = 0; h < kHidden; ++ h) {
            float acc = 0.f;
            for (uint32_t i = 0; i < kIntermediate; ++ i)
                acc += Interm[m * kIntermediate + i] * W2[h * kIntermediate + i];
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

    // **v3** Pre-merged W1 layout (host 端 N 维重排, 让 swap 后 TMEM 同 warp 内有 gate+up).
    auto w1_fp32_premerged = permute_w1_premerged(w1_fp32, 2 * kIntermediate, kHidden);

    // --- **Per-Tensor** quantize: 单一 scale per tensor, 无 SF.  ---
    std::vector<uint8_t> x_fp8, w1_fp8, w2_fp8;
    const float scale_x  = quantize_fp8_pt(x_fp32.data(),           static_cast<size_t>(kMaxM) * kHidden,             x_fp8);
    const float scale_w1 = quantize_fp8_pt(w1_fp32_premerged.data(),static_cast<size_t>(2 * kIntermediate) * kHidden, w1_fp8);
    const float scale_w2 = quantize_fp8_pt(w2_fp32.data(),          static_cast<size_t>(kHidden) * kIntermediate,     w2_fp8);
    std::printf("[v3] scale_X=%.6e scale_W1=%.6e scale_W2=%.6e (W1 pre-merged)\n", scale_x, scale_w1, scale_w2);

    // --- Device allocations (no SF buffers) ---
    uint8_t* d_x       = nullptr;
    uint8_t* d_w1      = nullptr;
    uint8_t* d_w2      = nullptr;
    nv_bfloat16* d_y   = nullptr;
    float*    d_y_fp32           = nullptr;
    uint8_t*  d_ws               = nullptr;
    uint32_t* d_l1_done_counter  = nullptr;
    uint32_t* d_l2_tile_counters = nullptr;

    const size_t bytes_x  = static_cast<size_t>(kMaxM) * kHidden;
    const size_t bytes_w1 = static_cast<size_t>(2 * kIntermediate) * kHidden;
    const size_t bytes_w2 = static_cast<size_t>(kHidden) * kIntermediate;
    const size_t bytes_y  = static_cast<size_t>(kMaxM) * kHidden * sizeof(nv_bfloat16);
    const size_t bytes_y_fp32 = static_cast<size_t>(kNumCTAs) * kMaxM * BLOCK_N * sizeof(float);
    const size_t bytes_ws    = static_cast<size_t>(kMaxM) * kIntermediate;

    CUDA_CHECK(cudaMalloc(&d_x,     bytes_x));
    CUDA_CHECK(cudaMalloc(&d_w1,    bytes_w1));
    CUDA_CHECK(cudaMalloc(&d_w2,    bytes_w2));
    CUDA_CHECK(cudaMalloc(&d_y,     bytes_y));
    CUDA_CHECK(cudaMalloc(&d_y_fp32,           bytes_y_fp32));
    CUDA_CHECK(cudaMalloc(&d_ws,               bytes_ws));
    CUDA_CHECK(cudaMalloc(&d_l1_done_counter,  sizeof(uint32_t)));
    constexpr uint32_t kNumL2Tiles = kHidden / BLOCK_N;
    CUDA_CHECK(cudaMalloc(&d_l2_tile_counters, kNumL2Tiles * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_x,  x_fp8.data(),  bytes_x,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1, w1_fp8.data(), bytes_w1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, w2_fp8.data(), bytes_w2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, bytes_y));
    CUDA_CHECK(cudaMemset(d_ws, 0xCC, bytes_ws));   // sentinel; will be overwritten by L1 epi

    // --- Build CUtensorMap descriptors (**v3 swap**: 6 个, X/interm box 收到 (K=128, M=32)) ---
    constexpr uint32_t kSwapValidM = 32;   // 必须与 kernel 内 kMaxValidM 一致
    auto tma_x = make_tma_2d("X_swap", d_x, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                             kHidden, kMaxM,
                             BLOCK_K, kSwapValidM,
                             static_cast<uint64_t>(kHidden) * 1,
                             CU_TENSOR_MAP_SWIZZLE_128B);
    auto tma_w1 = make_tma_2d("W1", d_w1, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                              kHidden, 2 * kIntermediate,
                              BLOCK_K, BLOCK_N,
                              static_cast<uint64_t>(kHidden),
                              CU_TENSOR_MAP_SWIZZLE_128B);
    auto tma_w2 = make_tma_2d("W2", d_w2, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                              kIntermediate, kHidden,
                              BLOCK_K, BLOCK_N,
                              static_cast<uint64_t>(kIntermediate),
                              CU_TENSOR_MAP_SWIZZLE_128B);
    // intermediate 写视图保持 (BLOCK_N/2=64, BLOCK_M=128) — TMA store 写 128 m rows
    // (虽然只前 32 valid, padding rows 在 cudaMemset 0 下安全; 留给将来 box 收紧)
    auto tma_interm = make_tma_2d("interm_w", d_ws, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                  kIntermediate, kMaxM,
                                  BLOCK_N / 2, BLOCK_M,
                                  static_cast<uint64_t>(kIntermediate),
                                  CU_TENSOR_MAP_SWIZZLE_NONE);
    auto tma_interm_load = make_tma_2d("interm_r_swap", d_ws, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                       kIntermediate, kMaxM,
                                       BLOCK_K, kSwapValidM,
                                       static_cast<uint64_t>(kIntermediate),
                                       CU_TENSOR_MAP_SWIZZLE_128B);
    auto tma_y = make_tma_2d("Y", d_y, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                             kHidden, kMaxM,
                             BLOCK_N, BLOCK_M,
                             static_cast<uint64_t>(kHidden) * sizeof(nv_bfloat16),
                             CU_TENSOR_MAP_SWIZZLE_NONE);

    // --- Kernel launch ---
    auto kernel = &deep_gemm::sm100_fp8_mega_ffn_v3_impl<
        kHidden, kIntermediate, kMaxM,
        BLOCK_M, BLOCK_N, BLOCK_K,
        kNumStages,
        kNumNonEpiThreads, kNumEpiThreads,
        kClusterDim, kL2KSplit>;

    // 计算 dynamic smem size (与 PT kernel 一致, 无 SF SMEM)
    constexpr uint32_t STORE_BLOCK_M_ = BLOCK_M;
    constexpr uint32_t kNumEpiWarps_  = kNumEpiThreads / 32;
    constexpr uint32_t kNumEpiWG_     = kNumEpiWarps_ / 4;
    constexpr uint32_t L1_OUT_BLOCK_N_= BLOCK_N / 2;

    constexpr uint32_t SMEM_CD_L1_ = kNumEpiWG_ * STORE_BLOCK_M_ * L1_OUT_BLOCK_N_ * 1 * 2;
    constexpr uint32_t SMEM_CD_L2_ = (kL2KSplit == 1)
        ? (kNumEpiWG_ * STORE_BLOCK_M_ * BLOCK_N * sizeof(nv_bfloat16) * 2)
        : 0u;
    constexpr uint32_t SMEM_CD_    = SMEM_CD_L1_ > SMEM_CD_L2_ ? SMEM_CD_L1_ : SMEM_CD_L2_;
    // **v3 swap**: SMEM_A = W (BLOCK_N=128 × BLOCK_K), SMEM_B = X (kSwapValidM=32 × BLOCK_K).
    constexpr uint32_t SMEM_A_     = BLOCK_N * BLOCK_K * 1;             // W: 128*128 = 16KB
    constexpr uint32_t SMEM_B_     = kSwapValidM * BLOCK_K * 1;         // X: 32*128 = 4KB
    constexpr uint32_t SMEM_AMAX_  = kNumEpiWarps_ * (STORE_BLOCK_M_ / 2) * sizeof(float2);
    constexpr uint32_t SMEM_BAR_   = (2 * kNumStages + 2 * 2) * 8 + 16;
    constexpr uint32_t smem_size   = SMEM_CD_
                                   + kNumStages * (SMEM_A_ + SMEM_B_)
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

    // Step 6: per-kernel access policy window —— 把 W1 (6MB) 标记为 persisting，
    // 强制驻留 L2。B200 L2=80MB，9MB weights 完全装得下。需要先 cudaDeviceSetLimit
    // 把 persisting L2 carveout 调到 ≥9MB。
    {
        size_t persist_l2_size = 16ull * 1024ull * 1024ull;  // 16MB carveout, 容纳 W1+W2+SF
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_l2_size));
    }

    cudaLaunchAttribute attr[4];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim = {kClusterDim, 1, 1};
    attr[1].id = cudaLaunchAttributeAccessPolicyWindow;
    attr[1].val.accessPolicyWindow.base_ptr = d_w1;
    attr[1].val.accessPolicyWindow.num_bytes = bytes_w1;
    attr[1].val.accessPolicyWindow.hitRatio = 1.0f;
    attr[1].val.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr[1].val.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    attr[2].id = cudaLaunchAttributeProgrammaticStreamSerialization;
#ifdef MFFN_DISABLE_PDL
    attr[2].val.programmaticStreamSerializationAllowed = 0;
#else
    attr[2].val.programmaticStreamSerializationAllowed = 1;
#endif
    // Step 17: ClusterSchedulingPolicyPreference. LoadBalancing 让 cluster 内 CTA
    // 调度更紧凑，对 cluster_sync 延迟可能有帮助。
    attr[3].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    attr[3].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
    cfg.attrs = attr;
    cfg.numAttrs = (kClusterDim > 1) ? 4 : 3;

    // Step 5: per-launch memset / cast kernel 全部融进主 kernel。
    //   - l1_done_counter 靠 bit31-flip 抗 ABA，启动时一次清零即可
    //   - y_fp32 由 kernel 在 post-L2 cast 阶段顺手把 valid_m 行归零，首次 launch 前
    //     对整块 buffer 一次 host memset 即可
    //   - fp32→bf16 cast 由 kernel 内 l2_k_half==0 的 leader CTA 直接做，省一次 launch
    // Step 8: y_fp32 变成 slot buffer, 每 iter 每 slot 都会被对应 CTA 重写, 无需任何 memset。
    //         bit31-flip counter 也不需要 memset。这里只做一次 counter 初始化即可。
    CUDA_CHECK(cudaMemsetAsync(d_l1_done_counter, 0, sizeof(uint32_t), cfg.stream));
    CUDA_CHECK(cudaMemsetAsync(d_l2_tile_counters, 0, kNumL2Tiles * sizeof(uint32_t), cfg.stream));

    // --- CPU reference: do L1 first to derive scale_intermediate ---
    std::vector<uint8_t> ws_fp8_ref;
    std::vector<float>   interm_fp32;
    const float scale_intermediate = cpu_reference_l1_workspace_pt(
        x_fp8, scale_x, w1_fp8, scale_w1, M, /*num_ctas=*/1u,
        ws_fp8_ref, interm_fp32);
    // Pre-compute combined scales for kernel
    const float scale_xw1 = scale_x * scale_w1;
    const float scale_inv_intermediate = 1.f / scale_intermediate;
    const float scale_iw2 = scale_intermediate * scale_w2;
    std::printf("[PT] scale_intermediate=%.6e | scale_xw1=%.6e scale_inv_I=%.6e scale_iw2=%.6e\n",
                scale_intermediate, scale_xw1, scale_inv_intermediate, scale_iw2);

    auto do_launch = [&]() {
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel,
            static_cast<void*>(d_y),
            d_y_fp32,
            static_cast<void*>(d_ws),
            d_l1_done_counter,
            d_l2_tile_counters,
            M,
            scale_xw1, scale_inv_intermediate, scale_iw2,
            tma_x, tma_w1, tma_w2,
            tma_interm, tma_interm_load,
            tma_y));
    };

    // Warmup + correctness
    do_launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Validate Linear1 shared workspace (PT: 只比较 FP8 字节, 无 SF) ---
    {
        std::vector<uint8_t> ws_fp8_host(bytes_ws, 0);
        CUDA_CHECK(cudaMemcpy(ws_fp8_host.data(), d_ws, bytes_ws, cudaMemcpyDeviceToHost));

        double sum_abs = 0.0, max_abs = 0.0;
        size_t nz = 0, total = 0;
        size_t first_bad_cnt = 0;
        for (uint32_t m = 0; m < M; ++ m)
            for (uint32_t i = 0; i < kIntermediate; ++ i) {
                uint8_t got = ws_fp8_host[m * kIntermediate + i];
                uint8_t exp = ws_fp8_ref [m * kIntermediate + i];
                float got_f = fp8_e4m3_to_float(got) * scale_intermediate;
                float exp_f = fp8_e4m3_to_float(exp) * scale_intermediate;
                float ref_f = interm_fp32[m * kIntermediate + i];
                float d = std::fabs(got_f - exp_f);
                sum_abs += d;
                max_abs = std::max<double>(max_abs, d);
                if (std::fabs(ref_f) > 1e-4f) ++ nz;
                ++ total;
                if (first_bad_cnt < 10 && got != exp) {
                    std::printf("  [L1-WS diff] m=%u i=%u got=0x%02x exp=0x%02x got_f=%.4f exp_f=%.4f ref=%.4f\n",
                                m, i, got, exp, got_f, exp_f, ref_f);
                    ++ first_bad_cnt;
                }
            }
        std::printf("[L1-WS] mean|Δ_l1|=%.6f max|Δ_l1|=%.6f nonzero_ref=%zu/%zu\n",
                    sum_abs / (double)total, max_abs, nz, total);
    }

    // --- Validate Y against CPU reference (PT) ---
    // 用 CPU L1 ref 出来的 ws_fp8_ref 做 Linear2 reference (匹配 GPU 的 quantize-then-MMA 语义).
    std::vector<nv_bfloat16> y_ref;
    cpu_reference_ffn_pt(ws_fp8_ref, scale_intermediate, w2_fp8, scale_w2, M, y_ref);

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
    for (int i = 0; i < 5; ++ i) do_launch();
    CUDA_CHECK(cudaEventRecord(e0));
    for (uint32_t i = 0; i < num_iters; ++ i) do_launch();
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    std::printf("[MegaFFN] avg latency = %.3f us  (over %u iters)\n",
                1000.0 * ms / num_iters, num_iters);

    // --- Cleanup ---
    cudaFree(d_x);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_y); cudaFree(d_y_fp32);
    cudaFree(d_ws);
    cudaFree(d_l1_done_counter);
    cudaFree(d_l2_tile_counters);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 0;
}
