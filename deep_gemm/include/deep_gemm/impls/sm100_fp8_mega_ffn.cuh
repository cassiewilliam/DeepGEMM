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
    bool     kFastMath      = true,
    // Derived constants
    uint32_t kNumThreads    = kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_fp8_mega_ffn_impl(
    void* y,
    void* workspace,               // FP8 intermediate, laid out [gridDim.x][kMaxM][kIntermediate]
    void* workspace_sf,            // UE8M0 scale factors, [gridDim.x][kMaxM][kIntermediate/32]
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

    // -----------------------------------------------------------------------------
    // 坐标系 / 线程角色
    // -----------------------------------------------------------------------------
    const uint32_t cta_idx   = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx  = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx  = ptx::get_lane_idx();

    // 本 CTA 负责 Linear2 的哪几个 N-tile（循环覆盖 kL2OutputBlocksN）
    const uint32_t num_ctas = gridDim.x;
    // 简化：gridDim.x 必须能均分 kL2OutputBlocksN（host 端保证）
    const uint32_t kL2NPerCta = kL2OutputBlocksN / num_ctas > 0 ? kL2OutputBlocksN / num_ctas : 1;

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
    constexpr uint32_t kEpilogueFullBarrierIdx = 0;
    constexpr uint32_t kEpilogueWGBarrierStartIdx = 1;

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
            constexpr uint32_t num_k_blocks = (phase_tag == Phase::Linear1) ? kL1KBlocks : kL2KBlocks;
            const uint32_t m_idx = 0;

            for (uint32_t n_tile = 0; n_tile < num_n_blocks; ++ n_tile) {
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    MFFN_TRACE("TMA-A pre-empty_wait stage=%u phase=%u kblk=%u ntile=%u", stage_idx, phase, k_block_idx, n_tile);
                    empty_barriers[stage_idx]->wait(phase ^ 1);
                    MFFN_TRACE("TMA-A post-empty_wait stage=%u phase=%u kblk=%u", stage_idx, phase, k_block_idx);

                    uint32_t k_idx    = k_block_idx * BLOCK_K;
                    uint32_t sfa_m_idx    = 0;
                    uint32_t sfa_k128_idx = k_block_idx;

                    if (cute::elect_one_sync()) {
                        MFFN_TRACE("TMA-A pre copy A stage=%u kblk=%u ntile=%u desc_a=%p desc_sf=%p",
                                   stage_idx, k_block_idx, n_tile, tensor_map_a_ptr, tensor_map_a_sf_ptr);
                        tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                            tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx],
                            k_idx, m_idx, 1);
                        MFFN_TRACE("TMA-A post copy A, pre copy SF stage=%u kblk=%u", stage_idx, k_block_idx);

                        tma::copy<SF_BLOCK_M, 1, 0, uint32_t>(
                            tensor_map_a_sf_ptr, full_barriers[stage_idx],
                            smem_sfa[stage_idx],
                            sfa_m_idx, sfa_k128_idx, 1);
                        MFFN_TRACE("TMA-A post copy SF stage=%u kblk=%u", stage_idx, k_block_idx);

                        uint32_t arrive_bytes = SMEM_A_SIZE_PER_STAGE + SF_BLOCK_M * sizeof(uint32_t);
                        full_barriers[stage_idx]->arrive_and_expect_tx(arrive_bytes);
                        MFFN_TRACE("TMA-A post arrive stage=%u kblk=%u", stage_idx, k_block_idx);
                    }
                    __syncwarp();
                }
            }
        };

        tma_a_loop(std::integral_constant<Phase, Phase::Linear1>{},
                   &tensor_map_x, &tensor_map_x_sf,
                   kL1OutputBlocksN, 0);
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
            constexpr uint32_t num_k_blocks = (phase_tag == Phase::Linear1) ? kL1KBlocks : kL2KBlocks;

            for (uint32_t n_tile = 0; n_tile < num_n_blocks; ++ n_tile) {
                const uint32_t n_idx = (n_base + n_tile) * BLOCK_N;
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    MFFN_TRACE("TMA-B pre-empty_wait stage=%u phase=%u kblk=%u ntile=%u", stage_idx, phase, k_block_idx, n_tile);
                    empty_barriers[stage_idx]->wait(phase ^ 1);
                    MFFN_TRACE("TMA-B post-empty_wait stage=%u phase=%u kblk=%u", stage_idx, phase, k_block_idx);

                    uint32_t k_idx     = k_block_idx * BLOCK_K;
                    uint32_t sfb_n_idx    = n_idx;
                    uint32_t sfb_k128_idx = k_block_idx;

                    if (cute::elect_one_sync()) {
                        MFFN_TRACE("TMA-B pre copy B stage=%u kblk=%u ntile=%u desc_b=%p desc_sf=%p",
                                   stage_idx, k_block_idx, n_tile, tensor_map_b_ptr, tensor_map_b_sf_ptr);
                        tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                            tensor_map_b_ptr, full_barriers[stage_idx], smem_b[stage_idx],
                            k_idx, n_idx, 1);
                        MFFN_TRACE("TMA-B post copy B, pre copy SF stage=%u kblk=%u", stage_idx, k_block_idx);
                        tma::copy<BLOCK_N, 1, 0, uint32_t>(
                            tensor_map_b_sf_ptr, full_barriers[stage_idx],
                            smem_sfb[stage_idx],
                            sfb_n_idx, sfb_k128_idx, 1);
                        MFFN_TRACE("TMA-B post copy SF stage=%u kblk=%u", stage_idx, k_block_idx);
                        uint32_t arrive_bytes = SMEM_B_SIZE_PER_STAGE + BLOCK_N * sizeof(uint32_t);
                        full_barriers[stage_idx]->arrive_and_expect_tx(arrive_bytes);
                        MFFN_TRACE("TMA-B post arrive stage=%u kblk=%u", stage_idx, k_block_idx);
                    }
                    __syncwarp();
                }
            }
        };

        tma_b_loop(std::integral_constant<Phase, Phase::Linear1>{},
                   &tensor_map_w1, &tensor_map_w1_sf,
                   kL1OutputBlocksN, 0);
        tma_b_loop(std::integral_constant<Phase, Phase::Linear2>{},
                   &tensor_map_w2, &tensor_map_w2_sf,
                   kL2NPerCta, cta_idx * kL2NPerCta);

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
            const uint32_t num_k_blocks = (phase_tag == Phase::Linear1) ? kL1KBlocks : kL2KBlocks;
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

        run_mma(Phase::Linear1, kL1OutputBlocksN);
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
        const uint32_t num_l1_n_blocks_paired = kL1OutputBlocksN / 2;   // 每 2 个 N-block = 1 对 (gate, up)
        DG_STATIC_ASSERT(kL1OutputBlocksN % 2 == 0, "kL1OutputBlocksN must be even (gate/up pairing)");

        // NOTES: Linear1 kernel 产出是 [2I / BLOCK_N] 个 N-block，每 2 个相邻 N-block 构成一对:
        //   偶数 N-block 是 gate[:, n_tile:n_tile+BLOCK_N]
        //   奇数 N-block 是 up  [:, n_tile:n_tile+BLOCK_N]
        // 但硬件 TMEM 交错布局 + 现有 MegaMoE 写法把 gate/up 塞到「同一个 N-block 的相邻列」。
        // 这里我们采用更直观的做法: 每个 N-block 单独量化成 L1_OUT_BLOCK_N 宽度的 intermediate。
        // 具体 SwiGLU 配对按照「gate_block = 2k, up_block = 2k+1」在写 workspace 时匹配。
        // 为简化，v1 实现按 MegaMoE 相同语义：每个 Linear1 N-block 自身含 gate/up 相邻列。
        //
        // WARNING: 这里的配对语义依赖于 host 端 W1 权重布局：在 N=2I 方向上应当是
        // [gate_0, up_0, gate_1, up_1, ...]（每 BLOCK_N 内部 N/2 是 gate, N/2 是 up）。
        // host 应当按照此布局准备 W1；否则需要改下面 SwiGLU 的 lane 配对。

        // Linear1 处理 kL1OutputBlocksN 次 epilogue；Linear2 处理 kL2NPerCta 次
        constexpr uint32_t kTotalIters = kL1OutputBlocksN;  // will be overwritten by loop count below

        // --- Linear1 epilogue loop ---
        uint32_t tma_stage_idx_l1 = 0;
        MFFN_TRACE("EPI L1 loop start n_blocks=%u", kL1OutputBlocksN);
        for (uint32_t n_tile = 0; n_tile < kL1OutputBlocksN; ++ n_tile) {
            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const auto accum_phase     = (current_iter_idx / kNumEpilogueStages) & 1;
            ++ current_iter_idx;

            MFFN_TRACE("EPI L1 pre tmem_full_wait iter=%u acc_stage=%u acc_phase=%u ntile=%u", current_iter_idx-1, accum_stage_idx, accum_phase, n_tile);
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase);
            MFFN_TRACE("EPI L1 post tmem_full_wait iter=%u ntile=%u", current_iter_idx-1, n_tile);
            ptx::tcgen05_after_thread_sync();

            const uint32_t valid_m = num_tokens;

            // 为一整个 BLOCK_M × BLOCK_N 的 TMEM accumulator 做:
            //   1) SwiGLU: pair (gate, up) → silu(gate)*up
            //   2) amax per-row → UE8M0 sf
            //   3) cast FP8 E4M3 → STSM → TMA store
            //
            // 注意 MegaMoE 的 lane 配对假设 gate/up 在 TMEM 中相邻列（granularity 8）。
            // 见 MegaMoE epilogue 中对 values[0..7] 的解释。

            float stored_cached_weight = 0;  // dense FFN: no topk weight, 恒为 1
            #pragma unroll
            for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                    ptx::tcgen05_before_thread_sync();
                    tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                    break;
                }

                float2 swiglu_values[kNumAtomsPerStore * 2];
                float2 amax_values[kNumAtomsPerStore];

                #pragma unroll
                for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                    const uint32_t j = s * kNumAtomsPerStore + i;

                    uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M;
                    uint32_t values[ATOM_M];
                    cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                        values[0], values[1], values[2], values[3]);
                    cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                        values[4], values[5], values[6], values[7]);
                    cutlass::arch::fence_view_async_tmem_load();

                    if (j == WG_BLOCK_M / ATOM_M - 1) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                    }

                    auto* fp32_values = reinterpret_cast<float*>(values);

                    // SwiGLU 配对与 MegaMoE 相同：偶索引 gate, 奇索引 up
                    #pragma unroll
                    for (uint32_t k = 0; k < 2; ++ k) {
                        auto bf16_gate = __float22bfloat162_rn(make_float2(fp32_values[k*4 + 0], fp32_values[k*4 + 1]));
                        auto bf16_up   = __float22bfloat162_rn(make_float2(fp32_values[k*4 + 2], fp32_values[k*4 + 3]));

                        auto gate = __bfloat1622float2(bf16_gate);
                        auto neg_gate_exp = make_float2(
                            kFastMath ? __expf(-gate.x) : expf(-gate.x),
                            kFastMath ? __expf(-gate.y) : expf(-gate.y));
                        const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
                        if constexpr (kFastMath) {
                            gate = __fmul2_rn(gate, {math::fast_rcp(denom.x), math::fast_rcp(denom.y)});
                        } else {
                            gate = {gate.x / denom.x, gate.y / denom.y};
                        }
                        const auto up = __bfloat1622float2(bf16_up);
                        swiglu_values[i * 2 + k] = __fmul2_rn(gate, up);
                    }

                    amax_values[i].x = math::warp_reduce<4, true>(
                        cute::max(cute::abs(swiglu_values[i*2 + 0].x), cute::abs(swiglu_values[i*2 + 1].x)),
                        math::ReduceMax<float>());
                    amax_values[i].y = math::warp_reduce<4, true>(
                        cute::max(cute::abs(swiglu_values[i*2 + 0].y), cute::abs(swiglu_values[i*2 + 1].y)),
                        math::ReduceMax<float>());
                    if (lane_idx < 4)
                        smem_amax_reduction[epilogue_warp_idx * (STORE_BLOCK_M / 2) + i * (ATOM_M / 2) + lane_idx] = amax_values[i];
                    __syncwarp();
                }

                const uint32_t tma_stage_idx = s % kNumTMAStoreStages;
                ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                #pragma unroll
                for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                    const float2 wp_amax =
                        smem_amax_reduction[(epilogue_warp_idx ^ 1) * (STORE_BLOCK_M / 2) + i * (ATOM_M / 2) + lane_idx % 4];
                    amax_values[i].x = cute::max(amax_values[i].x, wp_amax.x);
                    amax_values[i].y = cute::max(amax_values[i].y, wp_amax.y);

                    float2 sf, sf_inv;
                    math::get_e4m3_sf_and_sf_inv(amax_values[i], sf, sf_inv);

                    const float2 upper = __fmul2_rn(swiglu_values[i*2 + 0], sf_inv);
                    const float2 lower = __fmul2_rn(swiglu_values[i*2 + 1], sf_inv);
                    const auto fp8x4 = __nv_fp8x4_e4m3(make_float4(upper.x, upper.y, lower.x, lower.y));

                    uint32_t row = lane_idx;
                    uint32_t col = warp_idx_in_wg;
                    auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd_l1[tma_stage_idx])
                                    + epilogue_wg_idx * STORE_BLOCK_M * L1_OUT_BLOCK_N
                                    + i * ATOM_M * L1_OUT_BLOCK_N
                                    + row * L1_OUT_BLOCK_N
                                    + (col ^ (row / 2)) * kNumBankGroupBytes;
                    ptx::SM100_U8x4_STSM_T<__nv_fp8x4_e4m3>::copy(fp8x4, smem_ptr);

                    // 写 UE8M0 SF 到 workspace_sf
                    //   SF 的 K 方向粒度 32，每个 token 每 32 个元素一个 SF
                    //   这里我们负责当前 N-tile 的 [BLOCK_N/2] 输出对应的 SF（每行 BLOCK_N/2 / 32 = 2 个 uint8）
                    //   但由于 MegaMoE 的 TMEM 布局不是直接按 SF 行展开，我们只在 warp 0/2 × lane<4 写（与 MegaMoE 一致）
                    if (warp_idx_in_wg % 2 == 0 and lane_idx < 4) {
                        // workspace_sf 采用 K-major uint32 布局：
                        //   [num_ctas, I/128, kMaxM] uint32，每 uint32 打包 4 个沿 K 方向的 UE8M0
                        // k_sf_idx ∈ [0, I/32) 表示沿 K 方向的 32-element chunk 索引
                        //   k128 = k_sf_idx / 4,  sub = k_sf_idx % 4
                        //   byte offset = cta*(I/32*kMaxM) + k128*(kMaxM*4) + token_row*4 + sub
                        const uint32_t token_row = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M + lane_idx * 2;
                        const uint32_t k_sf_idx  = n_tile * 2 + warp_idx_in_wg / 2;
                        const uint32_t k128      = k_sf_idx / 4;
                        const uint32_t sub       = k_sf_idx % 4;
                        auto ws_sf_base = reinterpret_cast<uint8_t*>(workspace_sf);
                        const uint64_t cta_stride_bytes  = static_cast<uint64_t>(kMaxM) * (kIntermediate / 32);
                        const uint64_t k128_stride_bytes = static_cast<uint64_t>(kMaxM) * 4;
                        const uint64_t base = static_cast<uint64_t>(cta_idx) * cta_stride_bytes
                                            + static_cast<uint64_t>(k128) * k128_stride_bytes
                                            + static_cast<uint64_t>(sub);
                        const auto sf_x = (*reinterpret_cast<const uint32_t*>(&sf.x) >> 23) & 0xff;
                        const auto sf_y = (*reinterpret_cast<const uint32_t*>(&sf.y) >> 23) & 0xff;
                        if (token_row + 0 < valid_m)
                            ws_sf_base[base + static_cast<uint64_t>(token_row + 0) * 4] = sf_x;
                        if (token_row + 1 < valid_m)
                            ws_sf_base[base + static_cast<uint64_t>(token_row + 1) * 4] = sf_y;
                    }
                    __syncwarp();
                }
                ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                // warp 0 发起 TMA store 到 workspace
                if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                    uint32_t out_n_idx = n_tile * L1_OUT_BLOCK_N;
                    uint32_t out_m_idx = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M;
                    cute::tma_store_fence();
                    cute::SM90_TMA_STORE_3D::copy(
                        &tensor_map_interm,
                        reinterpret_cast<uint8_t*>(smem_cd_l1[tma_stage_idx]) + epilogue_wg_idx * STORE_BLOCK_M * L1_OUT_BLOCK_N,
                        out_n_idx, out_m_idx, cta_idx);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
            tma_stage_idx_l1 = (tma_stage_idx_l1 + 1) % kNumTMAStoreStages;
        }

        // Linear1 结束：等所有 TMA store 落盘 → 让 Linear2 TMA-A 能从 workspace 正确读取
        ptx::tma_store_wait<0>();
        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

        // --- Linear2 epilogue loop ---
        uint32_t tma_stage_idx_l2 = 0;
        for (uint32_t n_tile = 0; n_tile < kL2NPerCta; ++ n_tile) {
            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const auto accum_phase     = (current_iter_idx / kNumEpilogueStages) & 1;
            ++ current_iter_idx;

            tmem_full_barriers[accum_stage_idx]->wait(accum_phase);
            ptx::tcgen05_after_thread_sync();

            const uint32_t valid_m = num_tokens;
            const uint32_t global_n_tile = cta_idx * kL2NPerCta + n_tile;

            #pragma unroll
            for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                    ptx::tcgen05_before_thread_sync();
                    tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                    break;
                }

                const uint32_t tma_stage_idx = s % kNumTMAStoreStages;
                ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                #pragma unroll
                for (uint32_t i = 0; i < STORE_BLOCK_M / ATOM_M; ++ i) {
                    uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                    uint32_t values[ATOM_M];
                    cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                        values[0], values[1], values[2], values[3]);
                    cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                        values[4], values[5], values[6], values[7]);
                    cutlass::arch::fence_view_async_tmem_load();

                    if (s == WG_BLOCK_M / STORE_BLOCK_M - 1 and i == STORE_BLOCK_M / ATOM_M - 1) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                    }

                    // STSM with 128B swizzle pattern
                    uint32_t row = lane_idx % 8;
                    uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                    auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd_l2[tma_stage_idx])
                                    + epilogue_wg_idx * STORE_BLOCK_M * BLOCK_N * sizeof(cd_dtype_t)
                                    + (warp_idx_in_wg / 2) * STORE_BLOCK_M * kSwizzleCDMode
                                    + i * ATOM_M * kSwizzleCDMode
                                    + row * (kNumBankGroupBytes * 8)
                                    + (col ^ row) * kNumBankGroupBytes;
                    ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                        math::cast_into_bf16_and_pack(values[0], values[1]),
                        math::cast_into_bf16_and_pack(values[2], values[3]),
                        math::cast_into_bf16_and_pack(values[4], values[5]),
                        math::cast_into_bf16_and_pack(values[6], values[7]),
                        smem_ptr);
                }

                ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                    uint32_t out_n_idx = global_n_tile * BLOCK_N;
                    uint32_t out_m_idx = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M;
                    cute::tma_store_fence();
                    cute::SM90_TMA_STORE_2D::copy(
                        &tensor_map_y,
                        reinterpret_cast<uint8_t*>(smem_cd_l2[tma_stage_idx])
                            + epilogue_wg_idx * STORE_BLOCK_M * BLOCK_N * sizeof(cd_dtype_t),
                        out_n_idx, out_m_idx);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
            tma_stage_idx_l2 = (tma_stage_idx_l2 + 1) % kNumTMAStoreStages;
        }

        // 等所有 TMA store 完成
        ptx::tma_store_wait<0>();
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
