#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

// =====================================================================================
// SM100 (B200) 独立 Mega FFN Kernel —— 面向 Qwen3-0.6B 解码阶段的密集 FFN 融合
// -------------------------------------------------------------------------------------
// 本文件是一个 **独立的** device kernel，可以用 nvcc 手工编译，不依赖 DeepGEMM 的 JIT /
// csrc / python binding。它把 Linear1 (FP8×FP8, MX-FP8 block scale) + SwiGLU + Linear2
// (FP8×FP8, MX-FP8 block scale) 三段融合到同一个 persistent kernel 内。
//
// 与 MegaMoE 对比：
//   - 去掉 Dispatch / Combine / NVLink / SymBuffer / topk / 多 rank workspace
//   - 1-CTA UMMA (cta_group::1)，不做 2-CTA multicast
//   - Warp specialization：TMA-A, TMA-B, MMA, Epilogue (+ 占位冷 warp)
//   - 每 CTA 负责「一次完整的 Linear1 → 对应自己 N-tile 区间的 Linear2」，
//     intermediate 走 per-CTA HBM workspace（会驻留 L2，不消耗 HBM 带宽）
//
// kClusterDim 作为模板参数保留。当前实现支持 kClusterDim = 1，kClusterDim > 1 的 DSMEM
// 路径留作 follow-up（见下方 STUB 注释）。
//
// 参数约束（Qwen3-0.6B 默认）：
//   kHidden = 1024, kIntermediate = 3072, kMaxM <= 32, BLOCK_M = 32 (UMMA_M padded to 64),
//   BLOCK_N = 128, BLOCK_K = 128。
//
// 数据布局：
//   X          [kMaxM, kHidden]                    FP8 e4m3
//   X_sf       [kMaxM, kHidden / 32]               UE8M0
//   W1         [2 * kIntermediate, kHidden]        FP8 e4m3   (gate || up concat on N)
//   W1_sf      [2 * kIntermediate, kHidden / 32]   UE8M0
//   W2         [kHidden, kIntermediate]            FP8 e4m3
//   W2_sf      [kHidden, kIntermediate / 32]       UE8M0
//   Y          [kMaxM, kHidden]                    BF16
//   workspace  [gridDim.x, kMaxM, kIntermediate]   FP8 e4m3 (per-CTA intermediate)
//   ws_sf      [gridDim.x, kMaxM, kIntermediate/32] UE8M0
//
// =====================================================================================

#include <cstdint>
#include <type_traits>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// -------------------------------------------------------------------------------------
// Debug trace switch. 编译时加 -DMEGA_FFN_TRACE=1 打开；默认关。
// 只在 cta_idx==0 && lane 0 时打印，尽量减少 printf 开销 & divergence.
// -------------------------------------------------------------------------------------
#ifndef MEGA_FFN_TRACE
#define MEGA_FFN_TRACE 0
#endif
#if MEGA_FFN_TRACE
#define MFFN_TRACE(fmt, ...) do {                                   \
    if (blockIdx.x == 0 && threadIdx.x % 32 == 0)                   \
        printf("[cta%u w%u] " fmt "\n", blockIdx.x,                 \
               threadIdx.x / 32, ##__VA_ARGS__);                    \
} while(0)
#else
#define MFFN_TRACE(...) ((void)0)
#endif

// -------------------------------------------------------------------------------------
// 模板参数：
//   kHidden, kIntermediate, kMaxM         —— 形状常量
//   BLOCK_M, BLOCK_N, BLOCK_K             —— MMA 分块 (BLOCK_K 必须 128，对齐 swizzle-128B)
//   kNumStages                            —— K-pipeline 深度 (TMA-A/B ↔ MMA)
//   kNumNonEpilogueThreads                —— TMA-A/B + MMA + 冷 warp = 128
//   kNumEpilogueThreads                   —— Epilogue warp 数 (128 的倍数)
//   kClusterDim                           —— 1 (当前实现) / 计划扩展到 4
//   kActivationClamp                      —— SwiGLU 的 clamp 阈值；+inf 代表不 clamp
//   kFastMath                             —— 使用 APPROX rcp / __expf
// -------------------------------------------------------------------------------------
template <
    uint32_t kHidden,
    uint32_t kIntermediate,
    uint32_t kMaxM,
    uint32_t BLOCK_M,
    uint32_t BLOCK_N,
    uint32_t BLOCK_K,
    uint32_t kNumStages,
    uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kClusterDim    = 1,
    uint32_t kL2KSplit      = 1,   // Step3: Linear2 K 维拆分份数 (e.g. 2 → gridDim = 2*kL2OutputBlocksN)
    bool     kFastMath      = true,
    // Derived constants
    uint32_t kNumThreads    = kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_fp8_mega_ffn_impl(
    void* y,
    float* y_fp32,                 // Step3: fp32 累加 buffer, 仅当 kL2KSplit > 1 使用; host 每次 launch 前 memset 0
    void* workspace,               // FP8 intermediate, shared [kMaxM][kIntermediate] (共享, N-split 写)
    void* workspace_sf,            // UE8M0 scale factors, K-major [kIntermediate/128, kMaxM, 4]
    uint32_t* l1_done_counter,     // global counter, host 端 memset 0; 用作 Linear1→Linear2 cross-CTA barrier
    const uint32_t num_tokens,     // valid M (<= kMaxM)
    const __grid_constant__ cute::TmaDescriptor tensor_map_x,
    const __grid_constant__ cute::TmaDescriptor tensor_map_x_sf,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w1,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w1_sf,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w2,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w2_sf,
    const __grid_constant__ cute::TmaDescriptor tensor_map_interm,      // write view of workspace
    const __grid_constant__ cute::TmaDescriptor tensor_map_interm_load, // read view of workspace
    const __grid_constant__ cute::TmaDescriptor tensor_map_interm_sf,      // write view of ws_sf
    const __grid_constant__ cute::TmaDescriptor tensor_map_interm_sf_load, // read view of ws_sf
    const __grid_constant__ cute::TmaDescriptor tensor_map_y) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier   = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator1Sm;

    // -----------------------------------------------------------------------------
    // 编译期常量校验
    // -----------------------------------------------------------------------------
    DG_STATIC_ASSERT(kClusterDim == 1, "v1: only kClusterDim == 1 is implemented; DSMEM path is stubbed");
    DG_STATIC_ASSERT(kHidden        % BLOCK_K == 0, "Hidden must be divisible by BLOCK_K");
    DG_STATIC_ASSERT(kIntermediate  % BLOCK_K == 0, "Intermediate must be divisible by BLOCK_K");
    DG_STATIC_ASSERT((2 * kIntermediate) % BLOCK_N == 0, "2*Intermediate must be divisible by BLOCK_N");
    DG_STATIC_ASSERT(kHidden        % BLOCK_N == 0, "Hidden must be divisible by BLOCK_N");
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only BLOCK_K=128 is supported (matches 128B swizzle)");
    DG_STATIC_ASSERT(BLOCK_N == 128, "Only BLOCK_N=128 is supported");
    // SM100 1-CTA MXF8F6F4 block-scaled UMMA 硬件要求 UMMA_M == 128，因此 BLOCK_M 必须 128。
    // kMaxM 可以 <= 32（decoding 典型值），epilogue 中按 valid_m 截断无效行。
    DG_STATIC_ASSERT(BLOCK_M == 128, "BLOCK_M must be 128 (UMMA_M=128 for 1-CTA MXF8F6F4)");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Must be 128 (TMA-A/B/MMA/cold)");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Epilogue threads must be multiple of 128");
    DG_STATIC_ASSERT(kMaxM <= 128, "kMaxM constrained by UMMA_M padding");
    DG_STATIC_ASSERT(kNumStages >= 2, "Need at least 2 pipeline stages");

    using a_dtype_t  = cutlass::float_e4m3_t;    // activation (X / intermediate)
    using b_dtype_t  = cutlass::float_e4m3_t;    // weight (W1 / W2)
    using sf_dtype_t = cutlass::float_ue8m0_t;   // scale factor
    using cd_dtype_t = cutlass::bfloat16_t;      // output

    // UMMA 形状（1-CTA，无 A/B swap）
    // SM100 1-CTA MXF8F6F4 block-scaled UMMA 仅支持 UMMA_M=128（CUTLASS static_assert）。
    // 对于 kMaxM≤32 decoding，M-方向会有 ~3/4 冗余，但 hardware 限制无法规避；
    // epilogue 按 valid_m 截断，不会写出多余行。
    constexpr uint32_t UMMA_M = 128;
    constexpr uint32_t UMMA_N = BLOCK_N;
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;

    // SF granularity / UTCCP 对齐
    constexpr uint32_t kGranK = 32;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t SF_BLOCK_M = math::constexpr_align<uint32_t>(UMMA_M, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(SF_BLOCK_N == BLOCK_N, "No padding needed for SFB");

    // Swizzle configs（与 MegaMoE 一致）
    constexpr uint32_t kSwizzleAMode  = BLOCK_K * sizeof(a_dtype_t);  // 128
    constexpr uint32_t kSwizzleBMode  = BLOCK_K * sizeof(b_dtype_t);  // 128
    constexpr uint32_t kSwizzleCDMode = 128;
    DG_STATIC_ASSERT(BLOCK_N % kSwizzleCDMode == 0, "Invalid BLOCK_N vs swizzle CD mode");

    // Epilogue 流水
    constexpr uint32_t kNumEpilogueStages   = 2;
    constexpr uint32_t kNumTMAStoreStages   = 2;
    constexpr uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4;
    DG_STATIC_ASSERT(kNumEpilogueWarps % 4 == 0, "Epilogue warps must be warpgroup aligned");

    // -----------------------------------------------------------------------------
    // 形状派生
    // -----------------------------------------------------------------------------
    // Linear1: A=X[M,H]  B=W1[2I,H]  C=acc[M, BLOCK_N]，N 遍历 2I/BLOCK_N 次
    // Linear2: A=I[M,I]  B=W2[H,I]   C=acc[M, BLOCK_N]，每 CTA 负责若干个 H/BLOCK_N 的 N-tile
    constexpr uint32_t kL1OutputBlocksN   = (2 * kIntermediate) / BLOCK_N;   // e.g. 48
    constexpr uint32_t kL2OutputBlocksN   = kHidden / BLOCK_N;               // e.g. 8
    constexpr uint32_t kL1KBlocks         = kHidden / BLOCK_K;               // e.g. 8
    constexpr uint32_t kL2KBlocks         = kIntermediate / BLOCK_K;         // e.g. 24
    constexpr uint32_t L1_OUT_BLOCK_N     = BLOCK_N / 2;                     // SwiGLU 减半

    // Step 3：Linear2 K 拆分 (kL2KSplit) —— gridDim = kL2OutputBlocksN * kL2KSplit。
    //   - kL2KSplit == 1：Step 2 行为，每 CTA 跑满 kL2KBlocks 个 L2 K 块。
    //   - kL2KSplit >= 2：每 L2 N-tile 分给 kL2KSplit 个 CTA，每 CTA 只跑 kL2KBlocks/kL2KSplit
    //                     个 K 块产生 partial，再 atomicAdd fp32 到 y_fp32 合并。
    //
    // Step 4：放松 L1 N-tile 数必须整除 gridDim 的约束 —— 允许 gridDim > kL1OutputBlocksN。
    //   每个 CTA 最多负责 kL1NPerCtaCeil = ceil(kL1OutputBlocksN / kNumCTAs) 个 L1 N-tile；
    //   `l1_n_start` / `l1_n_count` 在运行期截断到 [0, kL1OutputBlocksN)。多余的 CTA 在 L1
    //   阶段 l1_n_count==0，直接跳过 L1，但仍参与 grid-sync 与 L2 计算。
    DG_STATIC_ASSERT(kL2KBlocks % kL2KSplit == 0,
                     "L2 K block count must be divisible by kL2KSplit");
    constexpr uint32_t kNumCTAs           = kL2OutputBlocksN * kL2KSplit;         // e.g. 8, 48, 96
    constexpr uint32_t kL1NPerCtaCeil     = (kL1OutputBlocksN + kNumCTAs - 1) / kNumCTAs;  // 6/1/1...
    constexpr uint32_t kL2KBlocksPerCta   = kL2KBlocks / kL2KSplit;               // 24/4/2/1

    // -----------------------------------------------------------------------------
    // 坐标系 / 线程角色
    // -----------------------------------------------------------------------------
    const uint32_t cta_idx   = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx  = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx  = ptx::get_lane_idx();

    // 本 CTA 负责 Linear2 的哪几个 N-tile（循环覆盖 kL2OutputBlocksN）
    const uint32_t num_ctas = gridDim.x;
    // Step3: 一个 CTA 只负责 1 个 L2 N-tile (即使 gridDim > kL2OutputBlocksN 也是)
    constexpr uint32_t kL2NPerCta = 1;

    // Step3: cta_idx 解码
    //   l2_n_tile_for_cta  = cta_idx % kL2OutputBlocksN   (归属的 L2 N-tile, 0..7)
    //   l2_k_half_for_cta  = cta_idx / kL2OutputBlocksN   (K 拆分里的第几份, 0..kL2KSplit-1)
    const uint32_t l2_n_tile_for_cta = cta_idx % kL2OutputBlocksN;
    const uint32_t l2_k_half_for_cta = cta_idx / kL2OutputBlocksN;
    const uint32_t l2_k_base_block   = l2_k_half_for_cta * kL2KBlocksPerCta;

    // Step 4：Linear1 N-split 允许 uneven —— 前若干个 CTA 做 1~kL1NPerCtaCeil 个 L1 tile，
    //         剩余 CTA (cta_idx * kL1NPerCtaCeil >= kL1OutputBlocksN) l1_n_count==0 直接跳过 L1。
    const uint32_t l1_n_start_raw = cta_idx * kL1NPerCtaCeil;
    const uint32_t l1_n_start     = l1_n_start_raw < kL1OutputBlocksN ? l1_n_start_raw : kL1OutputBlocksN;
    const uint32_t l1_n_end       = (l1_n_start + kL1NPerCtaCeil) < kL1OutputBlocksN
                                        ? (l1_n_start + kL1NPerCtaCeil)
                                        : kL1OutputBlocksN;
    const uint32_t l1_n_count     = l1_n_end - l1_n_start;  // 可能为 0

    // Prefetch 所有 TMA descriptor
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_x);
        cute::prefetch_tma_descriptor(&tensor_map_x_sf);
        cute::prefetch_tma_descriptor(&tensor_map_w1);
        cute::prefetch_tma_descriptor(&tensor_map_w1_sf);
        cute::prefetch_tma_descriptor(&tensor_map_w2);
        cute::prefetch_tma_descriptor(&tensor_map_w2_sf);
        cute::prefetch_tma_descriptor(&tensor_map_interm);
        cute::prefetch_tma_descriptor(&tensor_map_interm_load);
        cute::prefetch_tma_descriptor(&tensor_map_interm_sf);
        cute::prefetch_tma_descriptor(&tensor_map_interm_sf_load);
        cute::prefetch_tma_descriptor(&tensor_map_y);
    }

    // -----------------------------------------------------------------------------
    // SMEM 布局：
    //   [CD buffer] [A×kNumStages] [B×kNumStages] [SFA×kNumStages] [SFB×kNumStages]
    //   [amax reduction] [barriers ...] [tmem_ptr]
    // -----------------------------------------------------------------------------
    constexpr uint32_t STORE_BLOCK_M = BLOCK_M;  // 每个 warpgroup 处理全部 M
    constexpr uint32_t kSharedMemAlignment = 1024;

    // CD buffer: Linear1 输出 FP8 (经 SwiGLU, 宽度=L1_OUT_BLOCK_N) 或 Linear2 输出 BF16 (宽度=BLOCK_N)
    constexpr uint32_t SMEM_CD_L1_SIZE_PER_STAGE =
        kNumEpilogueWarpgroups * STORE_BLOCK_M * L1_OUT_BLOCK_N * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_CD_L1_SIZE  = SMEM_CD_L1_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_CD_L2_SIZE_PER_STAGE =
        kNumEpilogueWarpgroups * STORE_BLOCK_M * BLOCK_N * sizeof(cd_dtype_t);
    constexpr uint32_t SMEM_CD_L2_SIZE  = SMEM_CD_L2_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_CD_SIZE     = SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;

    constexpr uint32_t SMEM_A_SIZE_PER_STAGE   = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE   = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);

    DG_STATIC_ASSERT(SMEM_CD_SIZE % kSharedMemAlignment == 0 and
                     SMEM_A_SIZE_PER_STAGE % kSharedMemAlignment == 0 and
                     SMEM_B_SIZE_PER_STAGE % kSharedMemAlignment == 0,
                     "SMEM buffers must be 1024-byte aligned");

    extern __shared__ __align__(kSharedMemAlignment) uint8_t smem_buffer[];

    auto smem_cd_l1 = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(smem_buffer, i * SMEM_CD_L1_SIZE_PER_STAGE);
    });
    auto smem_cd_l2 = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<cd_dtype_t>(smem_buffer, i * SMEM_CD_L2_SIZE_PER_STAGE);
    });
    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(smem_buffer, SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(smem_buffer, SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_buffer,
        SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });
    // amax 归约区（epilogue 使用）
    auto smem_amax_reduction = reinterpret_cast<float2*>(smem_sfb[kNumStages]);
    constexpr uint32_t kAmaxReductionEntries = kNumEpilogueWarps * (STORE_BLOCK_M / 2);

    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_amax_reduction + kAmaxReductionEntries);
    auto full_barriers      = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
    auto tmem_full_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + 2 * kNumStages + i; });
    auto tmem_empty_barriers= utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + 2 * kNumStages + kNumEpilogueStages + i; });
    auto tmem_ptr_in_smem   = reinterpret_cast<uint32_t*>(barrier_start_ptr + 2 * kNumStages + 2 * kNumEpilogueStages);

    // -----------------------------------------------------------------------------
    // TMEM 分配
    //   [acc × kNumEpilogueStages columns] [SFA column] [SFB column]
    // -----------------------------------------------------------------------------
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols   = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols   = SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols      =
        utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;

    // -----------------------------------------------------------------------------
    // Barrier 初始化：warp 1 → mbarriers, warp 2 → TMEM 分配
    // -----------------------------------------------------------------------------
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(2);       // 2 producers: TMA-A warp + TMA-B warp
            empty_barriers[i]->init(1);      // consumer: MMA warp
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(kNumEpilogueThreads);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();
    MFFN_TRACE("post-init syncthreads passed");

    // -----------------------------------------------------------------------------
    // K-pipeline 游标（TMA 生产者 / MMA 消费者共用）
    // -----------------------------------------------------------------------------
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // 片内 Named Barrier 分配
    //   bar 0 —— epilogue 全 warp 同步 (count=kNumEpilogueThreads)
    //   bar 1 —— epilogue per-warpgroup 同步起点
    //   bar 15 —— Linear1→Linear2 CTA-wide 同步 (count=kNumThreads)
    //       专用 id 避免与 __syncthreads / epilogue 的 bar 0 冲突
    //       (同一 id 不同 thread-count 是 PTX 未定义行为，会触发 illegal instruction)
    constexpr uint32_t kEpilogueFullBarrierIdx   = 0;
    constexpr uint32_t kEpilogueWGBarrierStartIdx = 1;
    constexpr uint32_t kGridSyncBarrierIdx       = 15;

    // Linear1 → Linear2 之间跨 CTA grid-sync lambda (Step 4 重写, 灵感源自
    // deep_gemm::comm::grid_sync 的"bit31 翻转 + ld.acquire 轮询"模式)：
    //   * 所有非 CTA0 的 thread0 atomic_add_rel(+1)，CTA0 的 thread0 atomic_add_rel(kFinishTag - (N-1))；
    //     凑齐后 bit31 正好翻转一次，作为本轮"到齐"标志
    //   * spin 用 `ld.acquire.gpu`（普通 load + acquire 语义）而不是 atomicAdd(counter, 0)，
    //     把重度的全局 atomic 轮询换成轻量 load，大幅缓解 hot atomic 的 HBM/L2 排队
    //   * bit31 翻转天然抗 ABA，无需 host 每次 launch 前 memset counter
    constexpr uint32_t kGridSyncFinishTag = 0x80000000u;
    auto grid_sync_l1_to_l2 = [&] () {
        ptx::sync_aligned(kNumThreads, kGridSyncBarrierIdx);
        if (thread_idx == 0) {
            const uint32_t num_ctas_u = gridDim.x;
            const uint32_t delta = (cta_idx == 0)
                ? (kGridSyncFinishTag - (num_ctas_u - 1u))
                : 1u;
            const uint32_t old_val = ptx::atomic_add_rel(l1_done_counter, delta);
            uint32_t new_val;
            do {
                new_val = ptx::ld_acq(l1_done_counter);
            } while (((new_val ^ old_val) & kGridSyncFinishTag) == 0);
        }
        ptx::sync_aligned(kNumThreads, kGridSyncBarrierIdx);
    };

    // 寄存器预算
    constexpr uint32_t kNumNonEpilogueRegisters = 40;
    constexpr uint32_t kNumEpilogueRegisters    = 208;

    // =============================================================================
    // 合并的 "phase schedule": 每个 CTA 按顺序处理以下 block（BlockPhase）：
    //   1) Linear1 所有 N-block (kL1OutputBlocksN 次)           phase=L1
    //   2) Linear2 本 CTA 负责的 N-block (kL2NPerCta 次)        phase=L2
    // L1 与 L2 阶段串行（通过 __syncthreads + tma_store_wait）。L1 的输出经 TMA 写到
    // workspace[cta_idx] 中，L2 的输入通过 TMA 从同一 workspace 切片拉回。
    // =============================================================================
    enum class Phase : uint8_t { Linear1 = 0, Linear2 = 1 };

    // =========================================================================
    // 角色 ①：TMA-A warp —— 负责 A-tile (X 或 intermediate) + SFA
    // -------------------------------------------------------------------------
    if (warp_idx == 0) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
        MFFN_TRACE("TMA-A enter");

        // 注意：不能用 `const auto& d = cond ? ... : ...` 去绑定 `__grid_constant__` 的
        // TmaDescriptor 参数，nvcc 会把 `&d` 常量折叠成 0x0（null）导致 UTMALDG 非法指令。
        // 因此 Phase 用模板常量，descriptor 直接显式传入。
        auto tma_a_loop = [&](auto phase_tag_c,
                              const cute::TmaDescriptor* tensor_map_a_ptr,
                              const cute::TmaDescriptor* tensor_map_a_sf_ptr,
                              uint32_t num_n_blocks, uint32_t /*n_base*/) {
            constexpr Phase phase_tag = decltype(phase_tag_c)::value;
            MFFN_TRACE("TMA-A loop phase=%d n_blocks=%u", (int)phase_tag, num_n_blocks);
            constexpr uint32_t num_k_blocks =
                (phase_tag == Phase::Linear1) ? kL1KBlocks : kL2KBlocksPerCta;
            // Step 3: Linear2 的 K-block 基准偏移 (K-split 段起点)
            const uint32_t k_base_block =
                (phase_tag == Phase::Linear1) ? 0u : l2_k_base_block;
            const uint32_t m_idx = 0;

            for (uint32_t n_tile = 0; n_tile < num_n_blocks; ++ n_tile) {
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    MFFN_TRACE("TMA-A pre-empty_wait stage=%u phase=%u kblk=%u ntile=%u", stage_idx, phase, k_block_idx, n_tile);
                    empty_barriers[stage_idx]->wait(phase ^ 1);
                    MFFN_TRACE("TMA-A post-empty_wait stage=%u phase=%u kblk=%u", stage_idx, phase, k_block_idx);

                    const uint32_t abs_k_block = k_base_block + k_block_idx;
                    uint32_t k_idx        = abs_k_block * BLOCK_K;
                    uint32_t sfa_m_idx    = 0;
                    uint32_t sfa_k128_idx = abs_k_block;

                    if (cute::elect_one_sync()) {
                        MFFN_TRACE("TMA-A pre copy A stage=%u kblk=%u ntile=%u desc_a=%p desc_sf=%p",
                                   stage_idx, k_block_idx, n_tile, tensor_map_a_ptr, tensor_map_a_sf_ptr);
                        // Step 2 起 workspace 改为共享 2D，L1 / L2 两阶段的 TMA-A 都走 2D。
                        tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t, false>(
                            tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx],
                            k_idx, m_idx, 1);
                        tma::copy<SF_BLOCK_M, 1, 0, uint32_t, false>(
                            tensor_map_a_sf_ptr, full_barriers[stage_idx],
                            smem_sfa[stage_idx],
                            sfa_m_idx, sfa_k128_idx, 1);
                        MFFN_TRACE("TMA-A post copy A+SF stage=%u kblk=%u", stage_idx, k_block_idx);

                        uint32_t arrive_bytes = SMEM_A_SIZE_PER_STAGE + SF_BLOCK_M * sizeof(uint32_t);
                        full_barriers[stage_idx]->arrive_and_expect_tx(arrive_bytes);
                        MFFN_TRACE("TMA-A post arrive stage=%u kblk=%u", stage_idx, k_block_idx);
                    }
                    __syncwarp();
                }
            }
        };

        // Linear1 只做分配给本 CTA 的 l1_n_count 个 N-tile；l1_n_count==0 时不发 TMA。
        tma_a_loop(std::integral_constant<Phase, Phase::Linear1>{},
                   &tensor_map_x, &tensor_map_x_sf,
                   l1_n_count, 0);
        grid_sync_l1_to_l2();
        tma_a_loop(std::integral_constant<Phase, Phase::Linear2>{},
                   &tensor_map_interm_load, &tensor_map_interm_sf_load,
                   kL2NPerCta, 0);

    // =========================================================================
    // 角色 ②：TMA-B warp —— 负责 B-tile (W1 / W2) + SFB
    // -------------------------------------------------------------------------
    } else if (warp_idx == 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
        MFFN_TRACE("TMA-B enter");

        auto tma_b_loop = [&](auto phase_tag_c,
                              const cute::TmaDescriptor* tensor_map_b_ptr,
                              const cute::TmaDescriptor* tensor_map_b_sf_ptr,
                              uint32_t num_n_blocks, uint32_t n_base) {
            constexpr Phase phase_tag = decltype(phase_tag_c)::value;
            MFFN_TRACE("TMA-B loop phase=%d n_blocks=%u", (int)phase_tag, num_n_blocks);
            constexpr uint32_t num_k_blocks =
                (phase_tag == Phase::Linear1) ? kL1KBlocks : kL2KBlocksPerCta;
            // Step 3: Linear2 的 K-block 基准偏移 (K-split 段起点)
            const uint32_t k_base_block =
                (phase_tag == Phase::Linear1) ? 0u : l2_k_base_block;

            for (uint32_t n_tile = 0; n_tile < num_n_blocks; ++ n_tile) {
                const uint32_t n_idx = (n_base + n_tile) * BLOCK_N;
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    MFFN_TRACE("TMA-B pre-empty_wait stage=%u phase=%u kblk=%u ntile=%u", stage_idx, phase, k_block_idx, n_tile);
                    empty_barriers[stage_idx]->wait(phase ^ 1);
                    MFFN_TRACE("TMA-B post-empty_wait stage=%u phase=%u kblk=%u", stage_idx, phase, k_block_idx);

                    const uint32_t abs_k_block = k_base_block + k_block_idx;
                    uint32_t k_idx        = abs_k_block * BLOCK_K;
                    uint32_t sfb_n_idx    = n_idx;
                    uint32_t sfb_k128_idx = abs_k_block;

                    if (cute::elect_one_sync()) {
                        MFFN_TRACE("TMA-B pre copy B stage=%u kblk=%u ntile=%u desc_b=%p desc_sf=%p",
                                   stage_idx, k_block_idx, n_tile, tensor_map_b_ptr, tensor_map_b_sf_ptr);
                        // W1/W2/W1_sf/W2_sf 都是 2D
                        tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t, false>(
                            tensor_map_b_ptr, full_barriers[stage_idx], smem_b[stage_idx],
                            k_idx, n_idx, 1);
                        tma::copy<BLOCK_N, 1, 0, uint32_t, false>(
                            tensor_map_b_sf_ptr, full_barriers[stage_idx],
                            smem_sfb[stage_idx],
                            sfb_n_idx, sfb_k128_idx, 1);
                        MFFN_TRACE("TMA-B post copy B+SF stage=%u kblk=%u", stage_idx, k_block_idx);
                        uint32_t arrive_bytes = SMEM_B_SIZE_PER_STAGE + BLOCK_N * sizeof(uint32_t);
                        full_barriers[stage_idx]->arrive_and_expect_tx(arrive_bytes);
                        MFFN_TRACE("TMA-B post arrive stage=%u kblk=%u", stage_idx, k_block_idx);
                    }
                    __syncwarp();
                }
            }
        };

        // Linear1 N-split: CTA k 读 W1 的 [l1_n_start, l1_n_end) 个 N-tile (可能为空)。
        tma_b_loop(std::integral_constant<Phase, Phase::Linear1>{},
                   &tensor_map_w1, &tensor_map_w1_sf,
                   l1_n_count, l1_n_start);
        grid_sync_l1_to_l2();
        // Step 3：Linear2 的 N-tile 由 cta_idx % kL2OutputBlocksN 决定（与 K-split 无关）
        tma_b_loop(std::integral_constant<Phase, Phase::Linear2>{},
                   &tensor_map_w2, &tensor_map_w2_sf,
                   kL2NPerCta, l2_n_tile_for_cta);

    // =========================================================================
    // 角色 ③：MMA warp —— 发 UMMA 指令、UTCCP SF、commit barrier
    // -------------------------------------------------------------------------
    } else if (warp_idx == 2) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
        MFFN_TRACE("MMA enter");

        // UMMA block-scaled 指令描述符（FP8×FP8，UE8M0 scale）
        auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
            a_dtype_t, b_dtype_t, float, sf_dtype_t,
            UMMA_M, UMMA_N,
            cute::UMMA::Major::K, cute::UMMA::Major::K>();

        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);

        // 为每个 stage 缓存 descriptor base_lo 到 lane 寄存器
        DG_STATIC_ASSERT(kNumStages <= 32, "kNumStages must fit in one warp");
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * (SMEM_A_SIZE_PER_STAGE / 16) : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * (SMEM_B_SIZE_PER_STAGE / 16) : 0u;

        uint32_t current_iter_idx = 0;
        auto mma_loop = [&](Phase /*phase_tag*/, uint32_t num_n_blocks) {
            const uint32_t num_k_blocks = (current_iter_idx < kL1OutputBlocksN) ? kL1KBlocks : kL2KBlocks;  // not actually used here; use argument below
            (void)num_k_blocks;
            // NOTE: caller guarantees ordering; use proper k-block count below
        };
        (void)mma_loop;  // silence unused

        // UTCCP 需要的 SMEM 转置：
        //   TMA 把 SF 加载到 SMEM 后，按 K-major uint32 `[k128][m]` 排布。
        //   UTCCP 4x32dp128bit 期望的 SMEM 序是 4×32 转置 + XOR-swizzle。
        //   此 lambda 必须由整个 warp (32 lanes) 一起执行。
        auto utccp_smem_transpose = [&](uint32_t* smem_ptr) {
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
            __syncwarp();
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                ptx::st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);
        };

        auto run_mma = [&](Phase phase_tag, uint32_t num_n_blocks) {
            MFFN_TRACE("MMA loop phase=%d n_blocks=%u", (int)phase_tag, num_n_blocks);
            // Step 3：Linear2 在 K-split 下每 CTA 只跑 kL2KBlocksPerCta 个 K 块
            const uint32_t num_k_blocks =
                (phase_tag == Phase::Linear1) ? kL1KBlocks : kL2KBlocksPerCta;
            for (uint32_t n_tile = 0; n_tile < num_n_blocks; ++ n_tile) {
                const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                const auto accum_phase     = (current_iter_idx / kNumEpilogueStages) & 1;
                ++ current_iter_idx;

                MFFN_TRACE("MMA ENTER ntile=%u iter=%u acc_stage=%u acc_phase=%u stage_idx=%u phase=%u",
                           n_tile, current_iter_idx-1, accum_stage_idx, accum_phase, stage_idx, phase);
                tmem_empty_barriers[accum_stage_idx]->wait(accum_phase ^ 1);
                MFFN_TRACE("MMA post tmem_empty_wait iter=%u", current_iter_idx-1);
                ptx::tcgen05_after_thread_sync();

                auto empty_arrive = [&](bool do_tmem_full) {
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
                    if (do_tmem_full)
                        cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                    __syncwarp();
                };

                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    MFFN_TRACE("MMA pre full_wait stage=%u phase=%u kblk=%u", stage_idx, phase, k_block_idx);
                    full_barriers[stage_idx]->wait(phase);
                    MFFN_TRACE("MMA post full_wait stage=%u phase=%u kblk=%u", stage_idx, phase, k_block_idx);
                    ptx::tcgen05_after_thread_sync();

                    // 所有 32 lanes 协作：把 K-major SF SMEM 转置成 UTCCP 期望的 layout
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i)
                        utccp_smem_transpose(smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems);
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i)
                        utccp_smem_transpose(smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems);
                    cutlass::arch::fence_view_async_shared();
                    __syncwarp();

                    const auto a_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                    const auto b_base_lo = ptx::exchange(b_desc_lo, stage_idx);
                    MFFN_TRACE("MMA post exchange kblk=%u a_lo=%x b_lo=%x", k_block_idx, a_base_lo, b_base_lo);

                    if (cute::elect_one_sync()) {
                        // UTCCP: SMEM → TMEM scale factor copies
                        MFFN_TRACE("MMA pre UTCCP kblk=%u stage=%u", k_block_idx, stage_idx);
                        using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_1cta;
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                        }
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                        }
                        MFFN_TRACE("MMA post UTCCP kblk=%u", k_block_idx);

                        // K 方向展开 (BLOCK_K / UMMA_K 次 MMA)
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                            const auto runtime_desc =
                                mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k, k);
                            a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_base_lo, 0, k * UMMA_K);
                            b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_base_lo, 0, k * UMMA_K);
                            MFFN_TRACE("MMA pre fma kblk=%u k=%u scaleC=%d tmem=%u",
                                       k_block_idx, k, int(k_block_idx > 0 or k > 0),
                                       accum_stage_idx * UMMA_N);
                            ptx::SM100_MMA_MXF8F6F4_SS::fma(
                                a_desc, b_desc,
                                accum_stage_idx * UMMA_N,
                                k_block_idx > 0 or k > 0,
                                runtime_desc,
                                kTmemStartColOfSFA, kTmemStartColOfSFB);
                            MFFN_TRACE("MMA post fma kblk=%u k=%u", k_block_idx, k);
                        }
                    }
                    __syncwarp();
                    MFFN_TRACE("MMA pre empty_arrive kblk=%u last=%d", k_block_idx, int(k_block_idx == num_k_blocks - 1));
                    empty_arrive(k_block_idx == num_k_blocks - 1);
                    MFFN_TRACE("MMA post empty_arrive kblk=%u", k_block_idx);
                }
                MFFN_TRACE("MMA exit kblk loop ntile=%u", n_tile);
            }
            MFFN_TRACE("MMA exit ntile loop phase=%d", (int)phase_tag);
        };

        run_mma(Phase::Linear1, l1_n_count);
        grid_sync_l1_to_l2();
        run_mma(Phase::Linear2, kL2NPerCta);

        // Drain last tmem_empty
        if (current_iter_idx > 0) {
            const auto last = current_iter_idx - 1;
            tmem_empty_barriers[last % kNumEpilogueStages]->wait((last / kNumEpilogueStages) & 1);
        }

    // =========================================================================
    // 角色 ④：冷 warp（降低寄存器占用）
    // -------------------------------------------------------------------------
    } else if (warp_idx == 3) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
        grid_sync_l1_to_l2();

    // =========================================================================
    // 角色 ⑤：Epilogue warps —— SwiGLU+quant(L1) / BF16 cast(L2)
    // -------------------------------------------------------------------------
    } else {
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();
        MFFN_TRACE("EPI enter warp=%u", warp_idx);

        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        const uint32_t epilogue_warp_idx = warp_idx - 4;
        const uint32_t epilogue_wg_idx   = epilogue_warp_idx / 4;
        const uint32_t warp_idx_in_wg    = epilogue_warp_idx % 4;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        (void)epilogue_thread_idx;

        constexpr uint32_t WG_BLOCK_M      = BLOCK_M / kNumEpilogueWarpgroups;
        constexpr uint32_t ATOM_M          = 8;
        constexpr uint32_t kNumBankGroupBytes = 16u;
        constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M / ATOM_M;
        DG_STATIC_ASSERT(WG_BLOCK_M % STORE_BLOCK_M == 0, "Invalid WG_BLOCK_M");
        DG_STATIC_ASSERT(STORE_BLOCK_M % ATOM_M == 0, "Invalid STORE_BLOCK_M");
        DG_STATIC_ASSERT(BLOCK_M % kNumEpilogueWarpgroups == 0, "Invalid BLOCK_M");

        uint32_t current_iter_idx = 0;

        // -------- Linear1 epilogue: SwiGLU + UE8M0 quant + TMA store to workspace --------
        //
        // 新设计（per-lane = 1 M row）：
        //   - 每个 epilogue warp 负责 TMEM accumulator 的一个 32-row subpartition：
        //       warp 0 -> M rows 0..31, warp 1 -> 32..63, warp 2 -> 64..95, warp 3 -> 96..127
        //   - 对于 kMaxM<=128 的 decoding，实际 valid_m<=32，只有 warp 0 的部分 lane 产出有效数据，
        //     其余 warp/lane 仍正常读 TMEM 以维持 TMEM barrier 的一致性。
        //   - W1 在每个 BLOCK_N tile 内: gate=N[0..63], up=N[64..127]; 输出 L1_OUT_BLOCK_N=64 列。
        //   - TMEM_LOAD 32dp32b8x 每次读 32 rows × 8 cols = 每 lane 1 row × 8 列 floats。
        //     沿 N 方向步进 8，共 kNumAtomsGate=8 次读完 64 列 gate，再读 64 列 up。
        //   - 每 lane 收齐 64 个 SwiGLU 输出 → 2 个 32-element SF chunk → 2 个 UE8M0 字节 + 64B FP8。
        //   - FP8 用 st.shared.u32 写入线性 SMEM (m_row 行号, 每行 64B)；SF 字节直接写回 HBM workspace_sf。
        //   - 单 warp 0 elect_one 发起 SM90_TMA_STORE_3D 将 128*64B SMEM tile 写到 workspace。

        constexpr uint32_t ATOM_N         = 8;                      // 每次 32dp32b8x 读 8 N-cols
        constexpr uint32_t kNumAtomsGate  = L1_OUT_BLOCK_N / ATOM_N;// 64/8 = 8
        constexpr uint32_t kNumSfPerTile  = L1_OUT_BLOCK_N / kGranK;// 64/32 = 2

        const uint32_t m_row = epilogue_warp_idx * 32 + lane_idx;   // 0..127

        uint32_t tma_stage_idx_l1 = 0;
        MFFN_TRACE("EPI L1 loop start n_blocks=%u (split of %u)", l1_n_count, kL1OutputBlocksN);
        for (uint32_t n_tile = 0; n_tile < l1_n_count; ++ n_tile) {
            const uint32_t global_n_tile = l1_n_start + n_tile;  // 全局 N 序号
            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const auto accum_phase     = (current_iter_idx / kNumEpilogueStages) & 1;
            ++ current_iter_idx;

            MFFN_TRACE("EPI L1 pre tmem_full_wait iter=%u acc_stage=%u acc_phase=%u ntile=%u",
                       current_iter_idx-1, accum_stage_idx, accum_phase, n_tile);
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase);
            MFFN_TRACE("EPI L1 post tmem_full_wait iter=%u ntile=%u", current_iter_idx-1, n_tile);
            ptx::tcgen05_after_thread_sync();

            const uint32_t valid_m = num_tokens;

            // ---- Phase 1: TMEM load gate + up, 计算 SwiGLU ----
            float swiglu[L1_OUT_BLOCK_N];  // 每 lane 持有 64 个 SwiGLU 输出
            const uint32_t tmem_base = accum_stage_idx * UMMA_N;
            #pragma unroll
            for (uint32_t i = 0; i < kNumAtomsGate; ++ i) {
                uint32_t g[ATOM_N], u[ATOM_N];
                // gate: N = i*8 .. i*8+7
                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_base + i * ATOM_N,
                    g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7]);
                // up: N = 64 + i*8 .. 64 + i*8 + 7
                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_base + L1_OUT_BLOCK_N + i * ATOM_N,
                    u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7]);
                #pragma unroll
                for (uint32_t k = 0; k < ATOM_N; ++ k) {
                    const float gate_f = __uint_as_float(g[k]);
                    const float up_f   = __uint_as_float(u[k]);
                    const float ne     = kFastMath ? __expf(-gate_f) : expf(-gate_f);
                    const float denom  = 1.0f + ne;
                    const float silu_g = kFastMath ? gate_f * math::fast_rcp(denom) : gate_f / denom;
                    swiglu[i * ATOM_N + k] = silu_g * up_f;
                }
            }
            cutlass::arch::fence_view_async_tmem_load();

            // 所有 epilogue 线程都 arrive 一次（barrier 初始化 count = kNumEpilogueThreads）
            ptx::tcgen05_before_thread_sync();
            tmem_empty_barriers[accum_stage_idx]->arrive(0u);

            // ---- Phase 2: 等 SMEM CD 双缓冲空闲 ----
            const uint32_t tma_stage_idx = tma_stage_idx_l1;
            ptx::tma_store_wait<kNumTMAStoreStages - 1>();
            ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

            // ---- Phase 3: per-row amax, UE8M0 sf, FP8 量化 ----
            uint8_t fp8_row[L1_OUT_BLOCK_N];
            uint8_t sf_bytes_row[kNumSfPerTile];
            #pragma unroll
            for (uint32_t sf_idx = 0; sf_idx < kNumSfPerTile; ++ sf_idx) {
                float amax = 0.f;
                #pragma unroll
                for (uint32_t j = 0; j < kGranK; ++ j)
                    amax = fmaxf(amax, fabsf(swiglu[sf_idx * kGranK + j]));

                uint8_t sf_byte = 0;
                float   sf_inv  = 0.f;
                if (amax > 0.f) {
                    const float factor = amax * (1.0f / 448.0f);
                    uint32_t bits = __float_as_uint(factor);
                    uint32_t exp  = (bits >> 23) & 0xffu;
                    uint32_t man  = bits & 0x007fffffu;
                    if (man != 0) ++ exp;
                    if (exp > 0xffu) exp = 0xffu;
                    sf_byte = static_cast<uint8_t>(exp);
                    const float sf_f = __uint_as_float(exp << 23);
                    sf_inv = 1.0f / sf_f;
                }
                sf_bytes_row[sf_idx] = sf_byte;
                #pragma unroll
                for (uint32_t j = 0; j < kGranK; j += 4) {
                    const float4 v = make_float4(
                        swiglu[sf_idx * kGranK + j + 0] * sf_inv,
                        swiglu[sf_idx * kGranK + j + 1] * sf_inv,
                        swiglu[sf_idx * kGranK + j + 2] * sf_inv,
                        swiglu[sf_idx * kGranK + j + 3] * sf_inv);
                    __nv_fp8x4_e4m3 q(v);
                    uint32_t packed;
                    __builtin_memcpy(&packed, &q, 4);
                    __builtin_memcpy(&fp8_row[sf_idx * kGranK + j], &packed, 4);
                }
            }

            // ---- Phase 4: 写 SMEM (线性布局, m_row * 64 偏移) + 写 SF 字节到 workspace_sf ----
            auto smem_cd_base = reinterpret_cast<uint8_t*>(smem_cd_l1[tma_stage_idx]);
            if (m_row < valid_m) {
                auto dst = smem_cd_base + m_row * L1_OUT_BLOCK_N;
                #pragma unroll
                for (uint32_t j = 0; j < L1_OUT_BLOCK_N; j += 4) {
                    uint32_t packed;
                    __builtin_memcpy(&packed, &fp8_row[j], 4);
                    ptx::st_shared(reinterpret_cast<uint32_t*>(dst + j), packed);
                }

                // workspace_sf 布局: [I/128, kMaxM] uint32 (K-major, 共享, 无 cta_idx 维度)
                //   每 uint32 打包 4 个沿 K 方向连续的 UE8M0 字节。
                //   byte offset = k128*(kMaxM*4) + m_row*4 + sub
                //   其中 k_sf_idx = global_n_tile * kNumSfPerTile + sf_idx (全局 SF idx)
                auto ws_sf_base = reinterpret_cast<uint8_t*>(workspace_sf);
                const uint64_t k128_stride_bytes = static_cast<uint64_t>(kMaxM) * 4;
                #pragma unroll
                for (uint32_t sf_idx = 0; sf_idx < kNumSfPerTile; ++ sf_idx) {
                    const uint32_t k_sf_idx = global_n_tile * kNumSfPerTile + sf_idx;
                    const uint32_t k128     = k_sf_idx / 4;
                    const uint32_t sub      = k_sf_idx & 3;
                    const uint64_t off      = static_cast<uint64_t>(k128) * k128_stride_bytes
                                            + static_cast<uint64_t>(m_row) * 4
                                            + static_cast<uint64_t>(sub);
                    ws_sf_base[off] = sf_bytes_row[sf_idx];
                }
            }

            // SMEM 写 → TMA store 需要 fence + 全 epilogue 同步
            cute::tma_store_fence();
            ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

            // warp 0 elect_one 发起 2D TMA store: box = (L1_OUT_BLOCK_N, BLOCK_M) 写入共享 workspace
            if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                const uint32_t out_n_idx = global_n_tile * L1_OUT_BLOCK_N;
                const uint32_t out_m_idx = 0;
                cute::SM90_TMA_STORE_2D::copy(
                    &tensor_map_interm,
                    smem_cd_base,
                    out_n_idx, out_m_idx);
                cute::tma_store_arrive();
            }
            __syncwarp();
            tma_stage_idx_l1 = (tma_stage_idx_l1 + 1) % kNumTMAStoreStages;
        }

        // Linear1 结束：等所有 TMA store 落盘 → 让 Linear2 TMA-A 能从 workspace 正确读取
        ptx::tma_store_wait<0>();
        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

        // 跨 CTA grid-sync：等所有 CTA 的 Linear1 都完成，workspace 才对所有 CTA 一致。
        grid_sync_l1_to_l2();

        // --- Linear2 epilogue loop ---
        //
        // 新设计（per-lane = 1 M row）：
        //   - 每 lane 读自己所在 warp 的 TMEM subpartition 中的 1 行 × 128 N cols (kL2AtomsPerTile=16)
        //   - 每 8 个 float 打包成 4 个 BF16 pair (8 BF16 = 16 bytes)，直接 st.shared.v4.u32 线性写入 SMEM
        //   - SMEM 布局: [BLOCK_M][BLOCK_N] BF16, 线性无 swizzle (与 tma_y SWIZZLE_NONE 一致)
        //   - 每 n_tile 结束由 warp 0 elect_one 发起 SM90_TMA_STORE_2D
        //
        // 输出 SMEM 每行 BLOCK_N*2 = 256 bytes。m_row * 256 + col_offset。

        constexpr uint32_t kL2AtomsPerTile = BLOCK_N / ATOM_N;  // 128/8 = 16

        uint32_t tma_stage_idx_l2 = 0;
        for (uint32_t n_tile = 0; n_tile < kL2NPerCta; ++ n_tile) {
            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const auto accum_phase     = (current_iter_idx / kNumEpilogueStages) & 1;
            ++ current_iter_idx;

            tmem_full_barriers[accum_stage_idx]->wait(accum_phase);
            ptx::tcgen05_after_thread_sync();

            const uint32_t valid_m = num_tokens;
            // Step 3: global N-tile 由 cta_idx % kL2OutputBlocksN 决定 (与 K-split 无关)
            const uint32_t global_n_tile = l2_n_tile_for_cta + n_tile;

            // Phase 1: TMEM load 128 floats per lane
            uint32_t values[BLOCK_N];  // 128 uint32 (=floats) per lane
            const uint32_t tmem_base = accum_stage_idx * UMMA_N;
            #pragma unroll
            for (uint32_t i = 0; i < kL2AtomsPerTile; ++ i) {
                auto* v = &values[i * ATOM_N];
                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_base + i * ATOM_N,
                    v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
            }
            cutlass::arch::fence_view_async_tmem_load();

            ptx::tcgen05_before_thread_sync();
            tmem_empty_barriers[accum_stage_idx]->arrive(0u);

            if constexpr (kL2KSplit == 1) {
                // ---- Step 2 行为：cast BF16 + SMEM STSM + TMA store 到 tma_y ----
                // Phase 2: SMEM 写双缓冲等待
                const uint32_t tma_stage_idx = tma_stage_idx_l2;
                ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                // Phase 3: cast fp32 -> BF16, 写 SMEM (每 lane 1 row × 128 cols × 2B = 256B)
                auto smem_cd_base = reinterpret_cast<uint8_t*>(smem_cd_l2[tma_stage_idx]);
                if (m_row < valid_m) {
                    auto dst = smem_cd_base + m_row * BLOCK_N * sizeof(cd_dtype_t);
                    #pragma unroll
                    for (uint32_t i = 0; i < kL2AtomsPerTile; ++ i) {
                        auto* v = &values[i * ATOM_N];
                        const uint32_t p0 = math::cast_into_bf16_and_pack(v[0], v[1]);
                        const uint32_t p1 = math::cast_into_bf16_and_pack(v[2], v[3]);
                        const uint32_t p2 = math::cast_into_bf16_and_pack(v[4], v[5]);
                        const uint32_t p3 = math::cast_into_bf16_and_pack(v[6], v[7]);
                        auto row_dst = dst + i * ATOM_N * sizeof(cd_dtype_t);
                        ptx::st_shared(row_dst, p0, p1, p2, p3);
                    }
                }

                cute::tma_store_fence();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    const uint32_t out_n_idx = global_n_tile * BLOCK_N;
                    const uint32_t out_m_idx = 0;
                    cute::SM90_TMA_STORE_2D::copy(
                        &tensor_map_y,
                        smem_cd_base,
                        out_n_idx, out_m_idx);
                    cute::tma_store_arrive();
                }
                __syncwarp();
                tma_stage_idx_l2 = (tma_stage_idx_l2 + 1) % kNumTMAStoreStages;
            } else {
                // ---- Step 3 K-split 行为：直接 atomicAdd(float2) 到 y_fp32 合并 ----
                // y_fp32 是 [kMaxM, kHidden] fp32, host 每次 launch 前已 memset 0.
                // 每 lane 持有 BLOCK_N fp32, 按 2 个一组 (float2) 做 64 次 atomicAdd.
                if (m_row < valid_m) {
                    auto* y_row = y_fp32 + static_cast<size_t>(m_row) * kHidden
                                         + static_cast<size_t>(global_n_tile) * BLOCK_N;
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_N / 2; ++ i) {
                        const float a = __uint_as_float(values[2 * i + 0]);
                        const float b = __uint_as_float(values[2 * i + 1]);
                        atomicAdd(reinterpret_cast<float2*>(y_row + 2 * i),
                                  make_float2(a, b));
                    }
                }
            }
        }

        // 等所有 TMA store 完成 (kL2KSplit==1) / atomicAdd 本身立即可见 (kL2KSplit>=2)
        if constexpr (kL2KSplit == 1) {
            ptx::tma_store_wait<0>();
        }
        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

        // 释放 TMEM
        if (epilogue_warp_idx == 0)
            Allocator().free(0, kNumTmemCols);
    }

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "sm100_fp8_mega_ffn requires sm_100a");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
