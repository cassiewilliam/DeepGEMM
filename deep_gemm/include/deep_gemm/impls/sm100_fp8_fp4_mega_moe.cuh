#pragma once

#include <cstdint>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

// =====================================================================================
// 核心 Kernel (L4)：SM100 FP8×FP4 Mega MoE —— 五段融合大内核
// -------------------------------------------------------------------------------------
// 功能概述：
//   在同一个 persistent kernel 内一次性完成：
//     ① Dispatch      —— 从各 rank 拉取本 rank 所属 expert 的 token + SF + topk 权重
//     ② Linear1 GEMM  —— FP8 × FP4 = FP32 累加（SwiGLU 的 gate/up 对半输出）
//     ③ SwiGLU        —— silu(gate) * up * weight，顺便做 amax 并 cast 回 FP8 E4M3
//     ④ Linear2 GEMM  —— FP8 × FP4 = FP32 → BF16 输出
//     ⑤ Combine       —— 把 Linear2 输出通过 NVLink 写回源 rank 对应 topk 槽，再做 top-k 规约
//
// 并发分工：Warp-specialized，一个 block 内按 warp_idx 划分角色：
//   [0 ..  kNumDispatchWarps)                          → Dispatch 4 warps
//   [kNumDispatchWarps + 0]                            → GEMM TMA-A（token + SFA）
//   [kNumDispatchWarps + 1]                            → GEMM TMA-B（weight + SFB）
//   [kNumDispatchWarps + 2]                            → GEMM MMA issue（仅 leader CTA）
//   [kNumDispatchWarps + 3]                            → 占位（仅做寄存器降配）
//   [kNumDispatchWarps + kNumMMANonEpilogueWarps .. ]  → Epilogue + Combine 多个 warp
//
// Output-Stationary：同一个 pool_block 的 token 在 Linear1/Linear2 都由同一个 SM 算出，
// 避免跨 SM 规约；只需 per-block arrival (L1 count / L2 mask) 做 k-wise 同步。
//
// 关键同步对象（详见 `cluster_sync` 附近初始化）：
//   dispatch_barriers   —— 每个 dispatch warp 一个 mbarrier，用于"pull TMA 到达 + flip phase"
//   full_barriers       —— GEMM TMA 完成 → MMA 消费（transaction-based，expect_tx）
//   empty_barriers      —— MMA / Epilogue 消费完 → TMA 重用（umma_arrive_multicast_2x1SM）
//   tmem_full_barriers  —— UMMA 完成 → Epilogue 读取累加器（tcgen05.commit）
//   tmem_empty_barriers —— Epilogue 读完累加器 → UMMA 重用
//   combine_barriers    —— Combine 载入 TMA 完成（transaction-based）
// =====================================================================================

namespace deep_gemm {

// ---------------------------------------------------------------------------
// 模板参数分组：
//   【规模】kNumMaxTokensPerRank, kHidden, kIntermediateHidden, kNumExperts, kNumTopk
//   【分块】BLOCK_M/N/K, STORE_BLOCK_M, SF_BLOCK_M/N
//   【池】 kNumMaxPoolTokens, kNumPaddedSFPoolTokens
//   【调度】kNumExpertsPerWave, kNumStages
//   【线程】kNumDispatchThreads, kNumNonEpilogueThreads, kNumEpilogueThreads
//   【拓扑】kNumSMs, kNumRanks
//   【数值】kActivationClamp, kFastMath
// 派生参数（默认值，可被外部重写）：
//   L1_SHAPE_{N,K} / L2_SHAPE_{N,K} —— Linear1/Linear2 的 N×K 全局形状
//   kNumDispatchWarps / kNumMMANonEpilogueWarps / kNumEpilogueWarps —— 按 32 线程/warp 折算
//   kNumTokensPerWarp = 32 / kNumTopk  —— dispatch count 阶段每 warp 并行处理的 token 数
//                                         （lane_idx 表达 "token * topk + topk_idx"）
// ---------------------------------------------------------------------------
template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M,
    uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumStages,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,  // gate||up 并排输出
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_fp8_fp4_mega_moe_impl(void* y,
                            const uint32_t num_tokens,
                            const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights_sf) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    // Cluster 级 transaction barrier（64-bit mbarrier + expect_tx）；2-SM TMEM allocator
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator2Sm;

    // 模板不变量：
    //   * Dispatch 至少 1 warpgroup（128）；MMA 段严格 128 线程（4 warps）；
    //   * Epilogue 以 warpgroup 为单位（每 warpgroup 自占 STSM / TMA store 通道）；
    //   * expert 数必须能均分到 rank。
    DG_STATIC_ASSERT(kNumDispatchThreads % 128 == 0, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Invalid number of MMA non-epilogue threads");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of MMA epilogue and combine threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");

    // 坐标系：cluster 内 2 个 CTA，0 号是 leader（负责 MMA issue / TMA multicast 源）
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();

    // 尽早 prefetch 所有 TMA descriptor（使其进入 descriptor cache，减少首次访问延迟）
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights_sf);
    }

    // --------------------------- GLOBAL MEMORY LAYOUT ---------------------------
    // sym_buffer 是各 rank 的 symmetric memory 基地址数组，base 指向的是本 rank 自己的起点。
    // Workspace 占位从 sym_buffer 起，后续各 Buffer 的 base 依次接在 Workspace 之后。
    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk, BLOCK_M);

    // 各 Data 描述符（单条 token 占多少字节 / 是否需要 TMA 对齐）
    constexpr auto fp8_token_layout = layout::Data(kHidden);                                     // FP8 e4m3 token
    constexpr auto bf16_token_layout = layout::Data(kHidden * sizeof(nv_bfloat16));              // combine 写回 BF16 token
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);            // Linear1 → Linear2 的中间 FP8 token
    constexpr auto fp8_sf_layout = layout::Data(kHidden / 32);                                   // Linear1 输入 SF（每 32 元素 1 byte）
    constexpr auto fp8_intermediate_sf_layout = layout::Data(kIntermediateHidden / 32);          // Linear2 输入 SF
    constexpr auto input_topk_idx_layout = layout::Data(kNumTopk * sizeof(int64_t), false);      // token→expert 索引（非 TMA）
    constexpr auto input_topk_weights_layout = layout::Data(kNumTopk * sizeof(float), false);    // topk 权重（非 TMA）
    constexpr auto l1_topk_weights_layout = layout::Data(sizeof(float), false);                  // pool 槽位上的单条权重（供 SwiGLU 使用）

    // Registered inputs
    const auto input_token_buffer = layout::Buffer(
        fp8_token_layout, 1, kNumMaxTokensPerRank,
        workspace.get_end_ptr());
    const auto input_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, kNumMaxTokensPerRank,
        input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, kNumMaxTokensPerRank,
        input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, 1, kNumMaxTokensPerRank,
        input_topk_idx_buffer.get_end_ptr());

    // SF 相关参数
    //   kGranK = 32：每 32 个 K 元素共享一个 scale factor（MX-FP8/FP4 的硬件粒度）
    //   kNumUTCCPAlignedElems = 128：UTCCP 4x32 指令要求 SF 按 128 元素分组
    constexpr uint32_t kGranK = 32;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    DG_STATIC_ASSERT(SF_BLOCK_M == math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems), "Invalid SF_BLOCK_M");
    DG_STATIC_ASSERT(SF_BLOCK_N == BLOCK_N, "No padding is needed for SFB");

    // -----------------------------------------------------------------------------
    // UTCCP 4x32 的 SF 行内映射：为了让 tensor core 按 "4 组 × 32 lane" 取 SF，
    // 一个 128-token 分组内的 token_idx 要被重排为 (idx&31)*4 + ((idx>>5)&3)。
    // 公式：
    //   pool_block_offset = token_idx_in_expert / BLOCK_M   （按 pool block 粒度定位）
    //   inside_block      = token_idx_in_expert % BLOCK_M
    //   128-group 头部    = inside_block & ~127
    //   组内重排          = (inside_block & 31) * 4 + ((inside_block >> 5) & 3)
    // -----------------------------------------------------------------------------
    const auto transform_sf_token_idx = [](const uint32_t& token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M +
               (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
    };

    // L1 inputs
    const auto l1_token_buffer = layout::Buffer(
        fp8_token_layout, 1, kNumMaxPoolTokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, kNumPaddedSFPoolTokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, kNumMaxPoolTokens,
        l1_sf_buffer.get_end_ptr());

    // L2 inputs
    const auto l2_token_buffer = layout::Buffer(
        fp8_intermediate_token_layout, 1, kNumMaxPoolTokens,
        l1_topk_weights_buffer.get_end_ptr()
    );
    const auto l2_sf_buffer = layout::Buffer(
        fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens,
        l2_token_buffer.get_end_ptr()
    );

    // Combine inputs
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, kNumTopk, kNumMaxTokensPerRank,
        l2_sf_buffer.get_end_ptr()
    );

    // Data types
    // NOTES: activations are FP8 (e4m3), weights are FP4 (e2m1)
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::detail::float_e2m1_unpacksmem_t;

    // -----------------------------------------------------------------------------
    // MMA 形状（2-CTA UMMA）：
    //   * 全局始终做 A↔B 交换，这样 M 轴对应 weight-N、N 轴对应 token-M，
    //     便于把 weight 的 N 分到 cluster 里的 2 个 CTA 并行算（UMMA_M = 256）。
    //   * K-major：A/B 两矩阵都是 K 方向连续，匹配 TMA 2D 拷贝。
    //   * LOAD_BLOCK_M = BLOCK_M/2：multicast 在 A（token），每个 CTA 只加载一半 token。
    // -----------------------------------------------------------------------------
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;        // 2-CTA 合并后的 M 维
    constexpr uint32_t UMMA_N = BLOCK_M;                 // A/B swap 后 N 维 = token M
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2;
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_M % 32 == 0, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_N == LAYOUT_AD_M, "Invalid block N");
    DG_STATIC_ASSERT(BLOCK_K == 128, "Invalid block K");

    // Swizzle configs
    constexpr uint32_t kSwizzleAMode = BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t kSwizzleBMode = BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t kSwizzleCDMode = 128;
    DG_STATIC_ASSERT(BLOCK_N % kSwizzleCDMode == 0, "Invalid block N");

    // Epilogue configs
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;

    // -----------------------------------------------------------------------------
    // Shared memory：动态 SMEM 大小由 host 端启发式预估，总 layout 从低到高依次是
    //   [expert_count] [send_buffers] [CD buffer] [A×stages] [B×stages] [SFA×stages]
    //   [SFB×stages] [amax_reduction] [barriers...] [tmem_ptr]
    // 1024B 对齐是 SM100 的 shared memory bank 对齐要求（保证 TMA/STSM 高吞吐）。
    // -----------------------------------------------------------------------------
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // L1 输出是 SwiGLU 后的"前一半 N"（即 BLOCK_N/2），经过 FP8 E4M3 压缩后走 TMA store 2 stage；
    // L2 输出是 BF16 + 通过 NVLink store（无 TMA）只需 1 stage。两者共用 SMEM（取 max）。
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;
    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumDispatchWarps, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
    constexpr uint32_t SMEM_CD_L1_SIZE =
        kNumEpilogueWarpgroups * STORE_BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t) * kNumTMAStoreStages;
    constexpr uint32_t SMEM_CD_L2_SIZE =
        kNumEpilogueWarpgroups * STORE_BLOCK_M * BLOCK_N * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_L1_SIZE_PER_STAGE = SMEM_CD_L1_SIZE / kNumTMAStoreStages;
    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % kSharedMemoryAlignment == 0 and
                     SMEM_A_SIZE_PER_STAGE % kSharedMemoryAlignment == 0 and
                     SMEM_B_SIZE_PER_STAGE % kSharedMemoryAlignment == 0,
                     "Shared memory of CD/A/B must be aligned to 1024 bytes");

    // Tensor memory size
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Assign shared memory for dispatch warps
    const auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = layout::Buffer(
        fp8_token_layout, kNumDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));

    // GEMM shared memory: C/D, A, B
    // NOTES: GEMM shared memory starts after the dispatch region, aligned to 1024 bytes
    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE
    );

    // D/A/B shared memory
    auto smem_cd = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<uint8_t>(smem_gemm_base, i * SMEM_CD_L1_SIZE_PER_STAGE);
    });
    auto smem_cd_l2 = smem_cd[0];
    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // SF shared memory: SFA and SFB per pipeline stage
    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_gemm_base,
        SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    // Epilogue amax reduction shared memory
    auto smem_amax_reduction = reinterpret_cast<float2*>(smem_sfb[kNumStages]);

    // -----------------------------------------------------------------------------
    // Barrier 分组（在 SMEM 中线性排布，PatternVisitor 提供 O(1) 下标映射）：
    //   ┌─────────────────────┬───────────────────────────────────────────────┐
    //   │ dispatch_barriers  [kNumDispatchWarps]         每 dispatch warp 一把  │
    //   │ full_barriers      [kNumStages]                TMA 生产 → MMA 消费   │
    //   │ empty_barriers     [kNumStages]                MMA 释放 → TMA 重用   │
    //   │ tmem_full_barriers [kNumEpilogueStages]        UMMA 完成 → Epilogue  │
    //   │ tmem_empty_barriers[kNumEpilogueStages]        Epilogue 释放 → UMMA  │
    //   │ combine_barriers   [kNumEpilogueWarps * 2]     Combine TMA 载入 2 级 │
    //   └─────────────────────┴───────────────────────────────────────────────┘
    // tmem_ptr_in_smem：TMEM 分配后返回的起始 col index，所有 Epilogue warp 共读。
    // -----------------------------------------------------------------------------
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_amax_reduction + STORE_BLOCK_M * kNumEpilogueWarps / 2);
    auto dispatch_barriers      = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto full_barriers          = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + i); });
    auto empty_barriers         = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + kNumStages + i); });
    auto tmem_full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + kNumStages * 2 + i); });
    auto tmem_empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + kNumStages * 2 + kNumEpilogueStages + i); });
    auto combine_barriers       = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + kNumStages * 2 + kNumEpilogueStages * 2 + i); });
    auto tmem_ptr_in_smem       = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + kNumEpilogueStages * 2 + kNumEpilogueWarps * 2);

    // 2-CTA TMEM 分配必须有 cluster 级同步：保证 leader/follower CTA 看到一致的分配起点
    cute::cluster_sync();

    // -----------------------------------------------------------------------------
    // 初始化分工：让 4 个不同 warp 并行做 init，以缩短 cluster_sync 之间的 kernel 序章
    //   warp 0：bulk 清零 smem_expert_count（后面 dispatch count 用 atomicAdd_block 累加）
    //   warp 1：dispatch m-barrier init(1) —— 每次 TMA load 只 arrive 1 次
    //   warp 2：各 GEMM/Epilogue/Combine barrier init
    //     * full_barriers[i] init(2*2)：2 个 producer warp（TMA-A/TMA-B）× 2 个 CTA
    //     * empty_barriers[i] init(1)：只有 MMA warp 在 leader CTA 发 arrive（通过 umma_arrive_multicast）
    //     * tmem_full_barriers init(1)：UMMA commit 发一次
    //     * tmem_empty_barriers init(2*kNumEpilogueThreads)：Epilogue 每 thread 贡献一次，跨 2 CTA
    //     * combine_barriers init(1)：每个 TMA load 到达一次
    //   warp 3：在 leader/follower CTA 同时分配 TMEM
    // 最后 `cluster_sync` 保证所有 barrier init 对 cluster 内 producer/consumer 都可见。
    // -----------------------------------------------------------------------------
    if (warp_idx == 0) {
        if (cute::elect_one_sync())
            ptx::st_shared_bulk(smem_expert_count, kNumExperts * sizeof(uint32_t));
    } else if (warp_idx == 1) {
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            dispatch_barriers[i]->init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                full_barriers[i]->init(2 * 2);
                empty_barriers[i]->init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
                tmem_full_barriers[i]->init(1);
                tmem_empty_barriers[i]->init(2 * kNumEpilogueThreads);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                combine_barriers[i]->init(1);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 3) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    cute::cluster_sync();

    // Task scheduler
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank,
        kNumExpertsPerWave,
        kNumSMs, kNumRanks>(workspace);

    // -----------------------------------------------------------------------------
    // MMA pipeline 相位维护
    //   stage_idx : 0..kNumStages-1 之间循环（对应 full/empty barrier 的 slot）
    //   phase     : 每次 stage_idx 归零时翻转一次（mbarrier 的 parity bit）
    // advance_pipeline 在每个 k_block 结束时统一推进，TMA/MMA warp 共用同一套游标。
    // -----------------------------------------------------------------------------
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // -----------------------------------------------------------------------------
    // 块内 Named Barrier 分配（`bar.sync` / `barrier.sync` 使用的 0..15 ID 空间）：
    //   0  kDispatchBarrierIdx              —— Dispatch warps 内部对齐
    //   1  kDispatchWithEpilogueBarrierIdx  —— Dispatch ∩ Epilogue 的**防死锁**交叠屏障
    //                                          （见 Pull 前/Combine 后的双方对齐）
    //   2  kEpilogueFullBarrierIdx          —— Epilogue warpgroup 全体对齐
    //   3..  kEpilogueWGBarrierStartIdx     —— 每个 warpgroup 自占一个 barrier id
    // -----------------------------------------------------------------------------
    constexpr uint32_t kDispatchBarrierIdx = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx = 3;

    // NVLink barrier 的 tag（仅用于超时日志区分，不影响语义）
    constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 3;

    // -----------------------------------------------------------------------------
    // 寄存器异构分配（`warpgroup_reg_alloc/dealloc`）：
    //   * Dispatch    ×  kNumDispatchThreads      → 48/thread（轻量；TMA-driven）
    //   * NonEpilogue ×  kNumNonEpilogueThreads   → 40/thread（更轻；只发 TMA/MMA）
    //   * Epilogue    ×  kNumEpilogueThreads      → 208/thread（重；存 SwiGLU 值/amax/STSM 缓冲）
    //   总上限 64512 个寄存器 / SM（SM100 物理限制）
    // -----------------------------------------------------------------------------
    constexpr uint32_t kNumDispatchRegisters = 48;
    constexpr uint32_t kNumNonEpilogueRegisters = 40;
    constexpr uint32_t kNumEpilogueRegisters = 208;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    // Dispatch 与 Epilogue 走各自的 grid_sync counter，避免语义冲突
    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    // ==========================================================================
    // Warp 角色分发：下面的所有分支共享 scheduler/barrier/smem_xxx 变量
    // --------------------------------------------------------------------------
    //   ① [0, kNumDispatchWarps)                                    Dispatch
    //   ② [kNumDispatchWarps + 0]                                   GEMM TMA-A
    //   ③ [kNumDispatchWarps + 1]                                   GEMM TMA-B
    //   ④ [kNumDispatchWarps + 2]                                   GEMM MMA
    //   ⑤ [kNumDispatchWarps + 3]                                   占位（冷 warp）
    //   ⑥ [kNumDispatchWarps + kNumMMANonEpilogueWarps, 末尾)       Epilogue / Combine
    // ==========================================================================

    // ==========================================================================
    // ① Dispatch warps —— 负责：count + 写源索引 + NVLink 拉取 + 工作区清理
    // --------------------------------------------------------------------------
    // 阶段 A: 统计每个 expert 本 SM 收到的 token 数（smem 原子累加）
    // 阶段 B: 把 per-SM 的 send count 汇总到 workspace.send_count 全局槽
    //         并在高 32 位累计"已完成的 SM 数"，全部到齐 scheduler 才能吃进 token 数
    // 阶段 C: 对每条 (token, topk) 写入对端 rank 的 src_token_topk_idx 槽
    //         —— 远端 dispatch pull 时按 (expert, rank, slot) 查该表
    // 阶段 D: Grid sync → SM0 再把最终 recv count / recv count sum 通过 NVLink 发给所有 rank
    // 阶段 E: NVLink barrier 确保所有 rank 都可以开始 pull
    // 阶段 F: 每条 token 轮询可拉取 rank（min-peeling round-robin），TMA 拉到 pool
    // 阶段 G: 与 Epilogue 交叠之下做 workspace 清理（为下一次 kernel 调用准备）
    // ==========================================================================
    if (warp_idx < kNumDispatchWarps) {
        // 降低 per-thread 寄存器数，释放给 Epilogue warp 使用
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        // 一个 warp 每次并行处理 kNumTokensPerWarp 个 token，每 token 有 kNumTopk 条 topk
        // → lane_idx ∈ [0, kNumActivateLanes) 对应一个 (token, topk) 对
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            // 外循环跨 SM×warp 并行：(sm * nWarps + warp) 作为起点，步长 totalWarps * tokensPerWarp
            // TODO: figure out better unrolling
            // Now, `unroll` is better than `unroll 8`
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    // 每条 topk 是 int64，可能包含 -1（未激活），用 __ldg 走 L1 只读缓存
                    expert_idx = static_cast<int>(
                        __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        // 阶段 A: 用 smem 原子计数本 SM 每个 expert 的 token 数（后续还会原子"加 slot"）
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
           atomicAdd_block(smem_expert_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // 阶段 B: 把本 SM 计数合并到全局 send_count（64-bit：低 32 位 token 数 / 高 32 位 SM 数）
        //   - 返回的 old 值同时给出"当前 SM 在该 expert 的基础 slot 偏移"
        //   - 原子 add(1<<32 | local_count) 让 scheduler 用高 32 位判断"是否所有 SM 都到"
        // (~6.5 us)
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // 阶段 C: 写源索引表 —— 对每条 (token, topk) 计算其在"目标 rank 的该 expert 槽"中的位置
        //   - dst_rank    = expert 所属 rank
        //   - dst_slot    = smem 累加的本 SM 偏移（此时 smem_expert_count 已被覆写成基准偏移，
        //                   再继续 atomicAdd 即可得到线性 slot）
        // (~2 us with 512 tokens)
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
            // sym_buffer.map(ptr, r) → 把本 rank 的 ptr 映射到远端 rank r 的同偏移地址
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });

        // 阶段 D 开头：rank 内 Grid sync，保证本 rank 所有 SM 的 send_count / 源索引都写完
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        // 阶段 D: SM0 专属 —— 把本 rank 的 send_count 通过 NVLink 写到对端 recv_count
        // 其它 rank 的 scheduler 要的是"高 32 位 = kNumSMs*kNumRanks" 才往下走，
        // 所以这里要对每个远端 rank 做 system-scope atomic_add（触发远端 acquire）
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                // 细分 recv count（本 rank 发给对方多少）
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffff;
                // 汇总 recv count sum：64-bit add（低 = token 数、高 = SM×rank 到齐计数）
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // 阶段 E: 跨 rank barrier，确保所有 rank 都准备好被别人 pull
        //   sync_prologue=false：前面刚做过 grid_sync，无需重复
        //   sync_epilogue=true ：barrier 后要求所有 SM 都看到"可以开始拉"
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* After the grid sync above, there is no more writes by other SMs (except 0) */ false,
            /* After the NVLink barrier, there is a grid sync */ true
        );

        // 关键防死锁屏障：让 Dispatch 的"进入 pull"与 Epilogue 的"进入 combine 后 clean"互斥
        //   Epilogue warps 在 combine 之前会先到 kDispatchWithEpilogueBarrierIdx，
        //   只有两侧都到达才会放行 Dispatch 进入 pull —— 这样本次 kernel 的 pull 阶段
        //   不会与上一次残留的 epilogue 写回交叠。（详见 Epilogue 段对同一 barrier 的解释）
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // ----------------------------------------------------------------------
        // 阶段 F: Pull 循环 —— 为本 rank 的每个 expert-token 从某个源 rank 拉数据
        //   token_idx ∈ [0, Σ recv_tokens[e]) 按全局 warp 轮询
        //   拉取策略：在"命中该 expert 的所有 rank"之间做 min-peeling round-robin，
        //             保证相邻 token 尽量来自不同 rank（降低 NVLink 单链路热度）
        // ----------------------------------------------------------------------
        uint32_t pull_mbarrier_phase = 0;
        // 每个 dispatch warp 独占一个 SMEM 暂存 buffer + 一把 mbarrier（前面 init(1)）
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = dispatch_barriers[warp_idx];

        // 缓存本 rank 所有 expert 的 token 数（等待高 32 位 == kNumSMs*kNumRanks 表示确定）
        scheduler.fetch_expert_recv_count();

        // 本 warp 当前处理的 expert 状态（切换 expert 时重新加载 per-rank 分布）
        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
            // 推进 expert 指针：token_idx 可能已跨出当前 expert 的 token 范围
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++ current_expert_idx >= kNumExpertsPerRank)
                    break;

                // 进入新 expert：pool 偏移累加上一 expert 的 m-block 数
                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);

                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }

            if (current_expert_idx >= kNumExpertsPerRank)
                break;

            // 切换 expert 时才重新把 per-rank 分布拉入寄存器（避免每 token 都访 workspace）
            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    // TODO: this is not coalesced
                    stored_rank_count[i] = j < kNumRanks ?
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                }
            }

            // --------------------------------------------------------------
            // Min-peeling round-robin：给定每 rank 贡献 tokens[r]，
            // 线性 slot_idx 要被映射成 (rank, idx_in_rank)，且相邻 slot 来自不同 rank。
            //
            // 算法：
            //   每一轮取 length = min(active ranks) —— 在这一轮每个活跃 rank 各出 length 条，
            //   总计 num_round_tokens = length * num_active_ranks 条 token 可分配。
            //   若 slot_idx 落在当前轮，则：
            //     rank = 第 (slot_idx % num_active_ranks) 个仍活跃的 rank
            //     在 rank 内偏移 = offset + slot_idx / num_active_ranks
            //   否则 slot_idx -= num_round_tokens，offset += length，
            //   把每个 rank 扣掉 length 后进入下一轮（耗尽的 rank 从"活跃"中剔除）。
            //
            // 正确性：每轮所有活跃 rank 消费相同 length，自然平衡 NVLink 方向的负载。
            // --------------------------------------------------------------
            uint32_t current_rank_in_expert_idx;
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] = stored_rank_count[i];
            uint32_t offset = 0;
            uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t slot_idx = token_idx_in_expert;
            uint32_t token_idx_in_rank;
            while (true) {
                // 每 lane 先算局部，再一次 warp reduce（减少 shuffle 指令）
                uint32_t num_actives_in_lane = 0;
                uint32_t min_in_lane = 0xffffffff;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    num_actives_in_lane += remaining[i] > 0;
                    if (remaining[i] > 0)
                        min_in_lane = cute::min(min_in_lane, remaining[i]);
                }
                const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {
                    // 命中本轮：slot_idx_in_round 指明"活跃 rank 序列中的第几个"
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t num_seen_ranks = 0;
                    current_rank_in_expert_idx = 0;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                        const uint32_t num_active_lanes = __popc(mask);
                        // __fns(mask, 0, k) 找 mask 中第 k 个 1 的位置（lane 号）
                        if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                            current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }

                // 进入下一轮
                slot_idx -= num_round_tokens;
                offset += length;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] -= cute::min(remaining[i], length);
            }

            // 读源索引（已由远端 dispatch 写入），解包出 src_token_idx + src_topk_idx
            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx = src_token_topk_idx % kNumTopk;

            // Step 1: 发起 TMA 1D load，从远端 rank 的 token buffer 拉到 SMEM 暂存 buffer
            //   （用 elect_one_sync 让 warp 内仅 1 lane 发指令，其它 lane 自己走 scalar 路径）
            if (cute::elect_one_sync()) {
                ptx::tma_load_1d(
                    pull_buffer.get_base_ptr(),
                    sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                                   current_rank_in_expert_idx),
                    pull_mbarrier, kHidden);
            }
            __syncwarp();

            // Step 2: SF 直接 LDG→STG（小数据，与上面的 TMA load 并行覆盖 kHidden 数据）
            //   - SF 按 (k, token) 转置摆放到 pool（kNumPaddedSFPoolTokens 为行跨步）
            //   - transform_sf_token_idx 完成 UTCCP 4×32 所需的组内重排
            constexpr uint32_t kNumSFUint32 = kHidden / 128;
            DG_STATIC_ASSERT(kNumSFUint32 > 0 and kHidden % 128 == 0, "Invalid SF");
            const auto remote_sf_ptr = sym_buffer.map(
                input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(),
                current_rank_in_expert_idx);
            const auto local_sf_ptr = l1_sf_buffer.get_base_ptr<uint32_t>();
            const auto sf_pool_token_idx = expert_pool_block_offset * SF_BLOCK_M +
                transform_sf_token_idx(token_idx_in_expert);
            #pragma unroll
            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFUint32, 32u); ++ i) {
                const uint32_t j = i * 32 + lane_idx;
                if (j < kNumSFUint32)
                    local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx] = remote_sf_ptr[j];
            }
            __syncwarp();

            // Step 3: 拿 topk 权重 + 等 TMA load 完成 + TMA store 入 pool + 写 combine 元数据
            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            if (cute::elect_one_sync()) {
                // 从远端拿 topk 权重（很小，普通 load 即可）
                const auto weight = *sym_buffer.map(
                    input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                    current_rank_in_expert_idx);
                *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;

                // mbarrier.arrive.expect_tx + mbarrier.try_wait.parity：等 kHidden 字节 TMA 到 SMEM
                ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

                // 再发一条 TMA store：SMEM → pool（本 rank 自己 symmetric memory 的 l1_token_buffer）
                ptx::tma_store_1d(
                    l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
                    pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());

                // 记录反向索引（供 Combine 阶段把 L2 输出写回源 rank 的源 topk 槽位）
                *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                // 等 TMA store 到达 + 给本 pool-block 的 arrival count +1（release 语义）
                //   arrival count == BLOCK_M（最后不足 BLOCK_M 的尾 block 用 valid_m）时
                //   Linear1 TMA warp 才会放行该 block 的 load。
                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();
                ptx::red_add_rel(
                    workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
            }
            __syncwarp();
        }

        // ----------------------------------------------------------------------
        // 阶段 G: 工作区清理（为下一次 kernel 调用做准备）
        //   与 Epilogue combine 阶段的 BF16 write-back 重叠：
        //     - 这里等 kDispatchWithEpilogueBarrierIdx 放行，表明 Epilogue 已进入 combine
        //       （pool token 不再需要、但 barrier/工作区尚被 Epilogue 使用）
        //     - 清零后再次 NVLink barrier，保证所有 rank 工作区都干净
        // 分工：
        //   SM0     负责清 expert_send_count（全局 send 计数）
        //   其它 SM 把 kNumExpertsPerRank 个 expert 按 (sm_idx-1) mod (kNumSMs-1) 分配
        //     - 每个 expert 清 recv_count_sum / per-rank recv_count / L1 L2 arrival
        // ----------------------------------------------------------------------
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
        } else {
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                // 注意：必须先读 recv_count_sum 再清零，否则 scheduler 的等待逻辑会丢信息
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                expert_pool_block_offset = scheduler.get_pool_block_offset(i);

                // 让 warp 内各 thread 先都读完 num_recv_tokens
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                if (thread_idx == 0)
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;

                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;

                // Per-block arrival：下一次 kernel 复用这些槽位，必须归零
                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + j) = 0;
                    *workspace.get_l2_arrival_mask_ptr(expert_pool_block_offset + j) = 0;
                }
            }
        }

        // 跨 rank 同步：确认"所有 rank 都清干净了"；kernel 结尾不需要 epilogue grid_sync
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* Before the NVLink barrier, there is a grid sync */ true,
            /* At the end of kernel does not need to sync */ false
        );
    } else if (warp_idx == kNumDispatchWarps) {
        // ======================================================================
        // ② GEMM TMA-A warp —— 负责加载 activations (A) + scale-factor-A (SFA)
        // ----------------------------------------------------------------------
        // 按 scheduler 派发的 block 依次做：
        //   * 等 pool block 的 token 全部到齐（Linear1 等 L1 count，Linear2 等 L2 mask）
        //   * 每 k_block 等 empty_barrier（之前的消费者释放）
        //   * TMA 2D copy A + SFA 到 smem_a/smem_sfa 的当前 stage
        //   * leader CTA 发 arrive_and_expect_tx（挂 transaction 字节数），follower 只 arrive
        // ======================================================================
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            // Phase 选择不同的 TMA descriptor（L1 读原始 FP8 输入 / L2 读 SwiGLU 后的 intermediate）
            const auto tensor_map_a_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts : &tensor_map_l1_acts;
            const auto tensor_map_sfa_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

            const auto shape_k = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_K : L1_SHAPE_K;
            const auto shape_sfa_k = math::ceil_div(shape_k, kGranK * 4u);

            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;

            // --------------------------------------------------------------
            // Linear1：等 pool block 的所有 token 都到齐（l1_arrival_count == valid_m）
            //   注意用 acquire load，保证 Dispatch 阶段写入的 token 数据对 GEMM 可见。
            // --------------------------------------------------------------
            if (block_phase == sched::BlockPhase::Linear1) {
                const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                const auto expected = scheduler.template get_valid_m<false>();
                while (ptx::ld_acq(ptr) != expected);
            }

            // --------------------------------------------------------------
            // Linear2：等 Linear1 epilogue 把本 pool block 在所需 N 方向上都写好了
            //   - L1 epilogue 输出的 BLOCK_N 被 SwiGLU 减半为 BLOCK_N/2；
            //     一个 Linear2 的 k_block 对应两个 Linear1 的 n_block → needed 位图为 3<<(k*2)。
            //   - cached_l2_arrival_mask 是本 warp 的寄存器缓存，减少访存。
            // --------------------------------------------------------------
            uint64_t cached_l2_arrival_mask = 0;
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                if (block_phase == sched::BlockPhase::Linear2) {
                    DG_STATIC_ASSERT(BLOCK_K == BLOCK_N, "Invalid block sizes");
                    const uint64_t needed = 3ull << (k_block_idx * 2);
                    if ((cached_l2_arrival_mask & needed) != needed) {
                        const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                        do {
                            cached_l2_arrival_mask = ptx::ld_acq_gpu(ptr);
                        } while ((cached_l2_arrival_mask & needed) != needed);
                    }
                }

                // 等 empty_barrier：上一轮的 MMA 已读完该 stage → 可以覆盖
                empty_barriers[stage_idx]->wait(phase ^ 1);

                // TMA 2D 拷贝坐标
                uint32_t m_idx = pool_block_idx * BLOCK_M;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t sfa_m_idx = pool_block_idx * SF_BLOCK_M;
                uint32_t sfa_k_idx = k_block_idx;

                // 2-CTA cluster：follower CTA 负责 token 的后半段
                if (not is_leader_cta)
                    m_idx += scheduler.template get_valid_m<true>() / 2;

                if (cute::elect_one_sync()) {
                    // TMA multicast=2：一次拷贝同时投到 cluster 内的 2 个 CTA smem
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx, 2);
                    tma::copy<SF_BLOCK_M, 1, 0>(
                        tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx], sfa_m_idx, sfa_k_idx, 2);
                    // full_barrier 的 transaction 账户：leader 挂总字节数，follower 只 arrive(0)
                    //   （因为一个 mbarrier 只能被 "expect_tx" 一次，多 CTA 要分责避免重复）
                    if (is_leader_cta) {
                        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE * 2 + SF_BLOCK_M * sizeof(uint32_t) * 2);
                    } else {
                        full_barriers[stage_idx]->arrive(0u);
                    }
                }
                __syncwarp();
            }
        });
    } else if (warp_idx == kNumDispatchWarps + 1) {
        // ======================================================================
        // ③ GEMM TMA-B warp —— 负责加载 weights (B) + scale-factor-B (SFB)
        // ----------------------------------------------------------------------
        // 结构与 TMA-A 对称，差异：
        //   * weight 不需要等 arrival count（weight 是持久化的全局张量，kernel 启动前就准备好）
        //   * weight 的 N 偏移包含 "local_expert_idx * shape_n" —— 表示"选当前 expert 的 weight 切片"
        //   * SFB 的 k 方向按 expert 连续排布（每个 expert 一个 shape_sfb_k 段）
        // ======================================================================
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const auto tensor_map_b_ptr =
                block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights : &tensor_map_l1_weights;
            const auto tensor_map_sfb_ptr =
                block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights_sf : &tensor_map_l1_weights_sf;

            const auto shape_k = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_K : L1_SHAPE_K;
            const auto shape_n = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_N : L1_SHAPE_N;
            const auto shape_sfb_k = math::ceil_div(shape_k, kGranK * 4u);

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                // weight 全局偏移：按 expert 选择子张量，再定位到 n_block / k_block
                uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                uint32_t sfb_k_idx = local_expert_idx * shape_sfb_k + k_block_idx;

                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                        tensor_map_b_ptr, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx, 2);
                    tma::copy<BLOCK_N, 1, 0>(
                        tensor_map_sfb_ptr, full_barriers[stage_idx], smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
                    if (is_leader_cta) {
                        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE + BLOCK_N * sizeof(uint32_t) * 2);
                    } else {
                        full_barriers[stage_idx]->arrive(0u);
                    }
                }
                __syncwarp();
            }
        });
    } else if (warp_idx == kNumDispatchWarps + 2) {
        // ======================================================================
        // ④ GEMM MMA issue warp —— 仅 leader CTA 真正发 UMMA 指令（2-CTA UMMA）
        // ----------------------------------------------------------------------
        // 流程：
        //   for each block（scheduler 派发）：
        //     * 等 tmem_empty_barrier（Epilogue 已释放对应 accum slot）
        //     * for each k_block：
        //         - 等 full_barrier（TMA-A / TMA-B 完成）
        //         - UTCCP 把 SFA / SFB 从 smem 拷到 TMEM 的 SF 列
        //         - 沿 UMMA_K 展开发 MMA（第一 k 之后 enable 累加）
        //         - tcgen05.commit 通过 umma_arrive_multicast_2x1SM 同时 arrive
        //           empty_barrier（释放 TMA slot）+ 末 k 时 arrive tmem_full_barrier
        //   kernel 末尾额外等一次 tmem_empty_barrier，确保所有 mbarrier 析构安全。
        // ======================================================================
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        if (is_leader_cta) {
            // UMMA 指令描述符：block-scaled（MX-FP8 × MX-FP4，UE8M0 标度）
            // A/B 已交换 —— 这里第一模板参数是 b_dtype_t（weight FP4）对应硬件 A，
            // 第二是 a_dtype_t（token FP8）对应硬件 B。
            auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                UMMA_M, UMMA_N,
                cute::UMMA::Major::K, cute::UMMA::Major::K
            >();
            auto sf_desc = mma::sm100::make_sf_desc(nullptr);

            // A/B 的 UMMA descriptor base：把 kNumStages 个 stage 的 smem 基址编码到 lane 内
            //   （lane_idx < kNumStages 的 lane 各存一个 stage 的 base_lo，后面用 shuffle 选当前 stage）
            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
            uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
            uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

            // 合法 UMMA shape 检查
            DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                             "Invalid MMA instruction shape");

            uint32_t current_iter_idx = 0;
            scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                         const uint32_t& local_expert_idx,
                                         const uint32_t& num_k_blocks,
                                         const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                // 动态把 UMMA 的 N 维改为本 block 的 valid_m（UMMA-对齐到 16）
                // —— 当前 block 尾巴 token 不足时减小硬件 N 维度以避免越界
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, scheduler.template get_valid_m<true>());

                // TMEM 累加器 slot（双 buffer）：iter 偶数 → slot 0，奇数 → slot 1；
                // phase 在每满 kNumEpilogueStages 轮后翻转一次（mbarrier 的 parity）
                const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
                tmem_empty_barriers[accum_stage_idx]->wait(accum_phase ^ 1);
                // tcgen05.fence::after_thread_sync：确保 TMEM 内容写完后 MMA 才读（硬件顺序提示）
                ptx::tcgen05_after_thread_sync();

                // empty / tmem_full 的 arrive 包装（umma_arrive_multicast_2x1SM 一指令两 CTA 同时 arrive）
                auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                    auto umma_arrive = [](const uint64_t* barrier) {
                        constexpr uint16_t kCTAMask = (1 << 2) - 1; // cluster 内 2 个 CTA 都 arrive
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    };
                    umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
                    // TMEM 累加 pipeline 独立于 multicast，但同样用 umma_arrive 省事
                    if (do_tmem_full_arrive)
                        umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                    __syncwarp();
                };

                // --------------------- 按 K 发 MMA ---------------------
                #pragma unroll 2
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    // 等 TMA load 完成（full_barrier 的 transaction count 归零时触发 wait）
                    full_barriers[stage_idx]->wait(phase);
                    ptx::tcgen05_after_thread_sync();

                    // 从 lane 缓存中取对应 stage 的 A/B descriptor base_lo
                    const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                    const auto b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);
                    if (cute::elect_one_sync()) {
                        // UTCCP：SMEM → TMEM 的 SF 专用拷贝（4×32 lane × 128 bit）
                        using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;
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

                        // 在 k 方向展开 MMA：BLOCK_K / UMMA_K 次
                        // 第一次 (k_block=0 且 k=0) 累加参数 enable_C=false（清零）；其余为 true（累加）
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                            const auto runtime_instr_desc =
                                mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k, k);
                            a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, 0, k * UMMA_K);
                            b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, 0, k * UMMA_K);
                            ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
                                b_desc, a_desc, accum_stage_idx * UMMA_N,
                                k_block_idx > 0 or k > 0, runtime_instr_desc,
                                kTmemStartColOfSFB, kTmemStartColOfSFA);
                        }
                    }
                    __syncwarp();

                    // commit：发 TMA empty + 若本 k 是最后一 k 则发 tmem_full
                    // tcgen05.commit 自带 before_thread_sync 语义，无需显式 fence
                    empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
                }
            });

            // 结束兜底：再等一次最后一个 tmem_empty，确保 barrier 析构不会撞到 pending arrive
            if (current_iter_idx > 0) {
                const auto accum_phase_idx = ((current_iter_idx - 1) / kNumEpilogueStages) & 1;
                tmem_empty_barriers[(current_iter_idx - 1) % kNumEpilogueStages]->wait(accum_phase_idx);
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 3) {
        // ⑤ 占位 warp（仅做寄存器降配，以免挤占 Epilogue warp 的寄存器预算）
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // ======================================================================
        // ⑥ Epilogue + Combine warps
        // ----------------------------------------------------------------------
        // 在同一段代码里处理 3 件事：
        //   Linear1 epilogue：TMEM → SwiGLU → cast FP8 → TMA store 到 L2 acts buffer
        //                     同时计算 per-token amax 并写 UE8M0 scale factor
        //   Linear2 epilogue：TMEM → cast BF16 → STSM → 通过 NVLink 直写远端 combine buffer
        //   Combine 阶段：   从远端 combine buffer 拉回本 rank 全部 topk 贡献，
        //                    求 reduce sum → cast BF16 → TMA store 到最终 y
        // ======================================================================
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        // Tensor memory 地址简化：硬件忽略 warp 索引位，无需 `tmem_ptr |= warp_idx*32 << 16`
        // 并且同一 SM 不允许两个 CTA 共享 TMEM，故此处直接断言起点为 0
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        // Epilogue warp 空间划分：
        //   epilogue_warp_idx ∈ [0, kNumEpilogueWarps)
        //   epilogue_wg_idx    = warp_idx / 4     —— 2 个 warpgroup 各分 BLOCK_M / 2
        //   warp_idx_in_wg     = warp_idx % 4     —— 4 个 warp 各分 BLOCK_N / 4
        const auto epilogue_warp_idx = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const auto epilogue_wg_idx = epilogue_warp_idx / 4;
        const auto epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const auto warp_idx_in_wg = epilogue_warp_idx % 4;
        DG_STATIC_ASSERT((kNumDispatchWarps + kNumMMANonEpilogueWarps) % 4 == 0 and
                         kNumEpilogueWarps % 4 == 0, "Invalid epilogue warps");

        // Epilogue 内部划分层次（M 方向）：
        //   BLOCK_M  (整 block)
        //   └─ WG_BLOCK_M   = BLOCK_M / num_warpgroups  （每 warpgroup 分一半）
        //      └─ STORE_BLOCK_M = TMA store 的 M 粒度（双 stage pipeline）
        //         └─ ATOM_M = 8             （STSM 指令的最小行数）
        constexpr uint32_t WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups;
        constexpr uint32_t ATOM_M = 8;
        constexpr uint32_t kNumBankGroupBytes = 16u;
        constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M / ATOM_M;
        DG_STATIC_ASSERT(BLOCK_M % kNumEpilogueWarpgroups == 0, "Invalid block M");
        DG_STATIC_ASSERT(WG_BLOCK_M % STORE_BLOCK_M == 0, "Invalid warpgroup block M");
        DG_STATIC_ASSERT(STORE_BLOCK_M % ATOM_M == 0, "Invalid store block M");
        DG_STATIC_ASSERT(BLOCK_N == 128, "Invalid block N");

        // 与 Dispatch 交叠的防死锁屏障（见 Dispatch 段同一 barrier 的说明）：
        // 保证"Epilogue 进入 combine"和"Dispatch 进入 pull"不会在同一时刻发生，
        // 否则会因共享 SMEM 工作区造成数据污染。
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        uint32_t current_iter_idx = 0;
        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            // 等 UMMA 把 accum 写入 TMEM —— 双 buffer 交替（accum_stage_idx 0/1）
            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase);
            ptx::tcgen05_after_thread_sync();

            // 用 __shfl 的 exchange 读取 valid_m，告诉 NVCC warp 内无 divergence（利于 unroll/SASS）
            const uint32_t valid_m = ptx::exchange(scheduler.template get_valid_m<false>(), 0);
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            uint32_t m_idx = pool_block_idx * BLOCK_M;
            uint32_t n_idx = n_block_idx * BLOCK_N;

            if (block_phase == sched::BlockPhase::Linear1) {
                // ==============================================================
                // Linear1 Epilogue：TMEM → SwiGLU → amax → cast FP8 → TMA store
                // --------------------------------------------------------------
                // 关键：SM100_TMEM_LOAD_16dp256b1x 一次取 256-bit，ATOM_M = 8 行。
                //   gate/up 在 TMEM 中是相邻列（granularity 8 交错），解包后的配对为：
                //     (values[0], values[2]), (values[1], values[3]),
                //     (values[4], values[6]), (values[5], values[7])
                //   —— 偶索引是 gate，奇索引是 up。
                // ==============================================================
                float stored_cached_weight = 0;

                #pragma unroll
                for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                    // 尾部 store block 可能整块超出 valid_m → 直接释放 TMEM slot，跳出
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

                        // 每 32 个 token（= 一个 warp 的 lane 覆盖范围）重读一次 topk 权重到寄存器
                        DG_STATIC_ASSERT(32 % ATOM_M == 0, "Invalid block size");
                        DG_STATIC_ASSERT(WG_BLOCK_M % 32 == 0, "Invalid block size");
                        if ((j * ATOM_M) % 32 == 0) {
                            stored_cached_weight = *l1_topk_weights_buffer
                                .get_data_buffer(m_idx + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M + lane_idx)
                                .get_base_ptr<float>();
                        }

                        // 从 register cache 读出当前 atom 的 2 个 float 权重（本 lane 在 atom 中的行）
                        const float2 weights = {
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 0),
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 1)
                        };

                        // TMEM → 寄存器：2 次 16dp256b 指令覆盖 ATOM_M=8 行 × 128 col
                        //   tmem_addr | 0x00100000 选择"列 bank 上半"（硬件位用于区分 gate/up 列组）
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M;
                        uint32_t values[ATOM_M];
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               values[0], values[1], values[2], values[3]);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               values[4], values[5], values[6], values[7]);
                        cutlass::arch::fence_view_async_tmem_load();

                        // 最后一个 atom 时释放 TMEM slot（让 MMA 能写下一个 iter 的 accum）
                        if (j == WG_BLOCK_M / ATOM_M - 1) {
                            ptx::tcgen05_before_thread_sync();
                            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        }

                        // SwiGLU 实际计算：先把 FP32 收缩到 BF16 做 clamp（省寄存器）
                        //   公式：out = silu(clamp(gate)) * clamp(up) * topk_weight
                        //        silu(x) = x / (1 + exp(-x))
                        auto fp32_values = reinterpret_cast<float*>(values);
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            auto bf16_gate = __float22bfloat162_rn(make_float2(fp32_values[k * 4], fp32_values[k * 4 + 1]));
                            auto bf16_up = __float22bfloat162_rn(make_float2(fp32_values[k * 4 + 2], fp32_values[k * 4 + 3]));

                            // 数值裁剪（可选）：避免 FP8 E4M3 表示范围外导致的溢出
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                                bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});
                                bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});
                                bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});
                            }

                            auto gate = __bfloat1622float2(bf16_gate);
                            auto neg_gate_exp = make_float2(
                                kFastMath ? __expf(-gate.x) : expf(-gate.x),
                                kFastMath ? __expf(-gate.y) : expf(-gate.y));
                            const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
                            if constexpr (kFastMath) {
                                // fast_rcp 用 APPROX 倒数指令，精度略低但吞吐高
                                gate = __fmul2_rn(gate, {math::fast_rcp(denom.x), math::fast_rcp(denom.y)});
                            } else {
                                gate = {gate.x / denom.x, gate.y / denom.y};
                            }
                            const auto up = __bfloat1622float2(bf16_up);
                            swiglu_values[i * 2 + k] = __fmul2_rn(__fmul2_rn(gate, up), weights);
                        }

                        // 跨 lane amax：同一 token 行的 4 个 lane（BLOCK_N / 4 个 warp × 1 token 的数据布局）做 reduce max
                        amax_values[i].x = math::warp_reduce<4, true>(
                            cute::max(cute::abs(swiglu_values[i * 2 + 0].x), cute::abs(swiglu_values[i * 2 + 1].x)),
                            math::ReduceMax<float>());
                        amax_values[i].y = math::warp_reduce<4, true>(
                            cute::max(cute::abs(swiglu_values[i * 2 + 0].y), cute::abs(swiglu_values[i * 2 + 1].y)),
                            math::ReduceMax<float>());
                        // 4 条 lane 分别写入 smem，供同 warpgroup 的"对偶 warp"再做一次跨 warp reduce
                        if (lane_idx < 4)
                            smem_amax_reduction[epilogue_warp_idx * (STORE_BLOCK_M / 2) + i * (ATOM_M / 2) + lane_idx] = amax_values[i];
                        __syncwarp();
                    }

                    // 等前一轮 TMA store 释放 smem 槽位 + fence smem_amax_reduction
                    // （sync_aligned(128, ...) 把本 warpgroup 4 个 warp 对齐到共享 amax 可见性后）
                    const uint32_t tma_stage_idx = s % kNumTMAStoreStages;
                    ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // 第二次 cast 循环：跨 warp 合并 amax → 计算 sf / sf_inv → cast FP8 → STSM → 写 SF
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        // 与对偶 warp（warp_idx_in_wg ^ 1）合并 amax，得到整行的 max
                        const float2 wp_amax =
                            smem_amax_reduction[(epilogue_warp_idx ^ 1) * (STORE_BLOCK_M / 2) + i * (ATOM_M / 2) + lane_idx % 4];
                        amax_values[i].x = cute::max(amax_values[i].x, wp_amax.x);
                        amax_values[i].y = cute::max(amax_values[i].y, wp_amax.y);

                        // 由 amax 推导 E4M3 的 scale 与其倒数（用于正向/反向量化）
                        float2 sf, sf_inv;
                        math::get_e4m3_sf_and_sf_inv(amax_values[i], sf, sf_inv);

                        // 执行 FP8 量化：swiglu_value * sf_inv，再打包成 fp8x4_e4m3
                        const float2 upper = __fmul2_rn(swiglu_values[i * 2 + 0], sf_inv);
                        const float2 lower = __fmul2_rn(swiglu_values[i * 2 + 1], sf_inv);
                        const auto fp8x4_values = __nv_fp8x4_e4m3(make_float4(upper.x, upper.y, lower.x, lower.y));

                        // STSM：row = lane_idx，col = warp 在 warpgroup 内位置（swizzle 避免 bank 冲突）
                        uint32_t row = lane_idx;
                        uint32_t col = warp_idx_in_wg;
                        const auto smem_ptr = smem_cd[tma_stage_idx] + epilogue_wg_idx * STORE_BLOCK_M * L1_OUT_BLOCK_N
                                                                     + i * ATOM_M * L1_OUT_BLOCK_N
                                                                     + row * L1_OUT_BLOCK_N
                                                                     + (col ^ (row / 2)) * kNumBankGroupBytes;
                        ptx::SM100_U8x4_STSM_T<__nv_fp8x4_e4m3>::copy(fp8x4_values, smem_ptr);

                        // 把 SF 写进 l2_sf_buffer（UE8M0 取 float 的指数位 8 bit）
                        //   布局为 MN-major：同一 K 方向上的 token SF 连续；相邻 token 的 sf_idx 差 4
                        //   只让每对中的 warp 0/2 写（因为 cross-warp reduce 后两 warp 持有相同值）
                        if (warp_idx_in_wg % 2 == 0 and lane_idx < 4) {
                            // TODO: I believe the expression can be optimized
                            const uint32_t token_idx_in_expert = m_block_idx * BLOCK_M
                                + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M + lane_idx * 2;
                            const uint32_t k_idx = n_block_idx * 2 + warp_idx_in_wg / 2;
                            const uint32_t k_uint_idx = k_idx / 4, byte_idx = k_idx % 4;
                            const uint32_t mn_stride = kNumPaddedSFPoolTokens * sizeof(uint32_t);
                            const auto sf_base_ptr = l2_sf_buffer.get_base_ptr<uint8_t>();
                            // NOTES: consecutive tokens (t, t + 1) are in the same 32-group, so `sf_idx` differs by 4
                            const auto sf_pool_token_idx = scheduler.get_current_pool_block_offset() * SF_BLOCK_M
                                + transform_sf_token_idx(token_idx_in_expert);
                            sf_base_ptr[k_uint_idx * mn_stride + sf_pool_token_idx * static_cast<uint32_t>(sizeof(uint32_t)) + byte_idx] =
                                (*reinterpret_cast<const uint32_t*>(&sf.x) >> 23);
                            sf_base_ptr[k_uint_idx * mn_stride + (sf_pool_token_idx + 4) * static_cast<uint32_t>(sizeof(uint32_t)) + byte_idx] =
                                (*reinterpret_cast<const uint32_t*>(&sf.y) >> 23);
                        }
                        __syncwarp();
                    }
                    // 每个 store block 完成后做一次 warpgroup aligned sync，
                    // 让 warp 0 发起 TMA store，其他 warp 等候 STSM 完成
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // warp 0 发起 TMA 2D store：把 FP8 activation 从 smem 写到 L2 activation buffer
                    //   目的地址由 tensor_map_l1_output 描述的 (N, M) 坐标定位
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd[tma_stage_idx] + epilogue_wg_idx * STORE_BLOCK_M * L1_OUT_BLOCK_N,
                            out_n_idx,
                            m_idx + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                }

                // 本 pool block 在 N 方向的这个 n_block 已经写完 —— 置位 l2_arrival_mask 的对应 bit
                //   Linear2 的 TMA-A/TMA-B 会 poll 这个 mask：等所有需要的 n_block 都置位才开始 load。
                //   TODO: 目前的 kEpilogueFullBarrierIdx 覆盖整个 epilogue，可以再细化
                ptx::tma_store_wait<0>();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    DG_STATIC_ASSERT(L2_SHAPE_K <= 64 * L1_OUT_BLOCK_N, "L2 shape K is too large");
                    // red.or.rel.gpu：atomic OR + release fence，可见于全 GPU 其它 SM
                    ptx::red_or_rel_gpu(
                        workspace.get_l2_arrival_mask_ptr(pool_block_idx),
                        1ull << n_block_idx
                    );
                }
                __syncwarp();
            } else {
                // ==========================================================
                // Linear2 Epilogue：TMEM → cast BF16 → STSM → NVLink 写回远端 combine buffer
                // ----------------------------------------------------------
                // 与 Linear1 不同，Linear2 的输出 **跨 rank 直写**：
                //   - 读取每一行 token 的 src_metadata（rank_idx, token_idx, topk_idx）
                //   - 通过 sym_buffer.map(dst_ptr, dst_rank_idx) 把本地 smem 里的 BF16 token
                //     写入目标 rank 上的 combine_token_buffer 的对应槽位
                //   - 后续 Combine 阶段再在目标 rank 上把 top-k 个贡献 reduce 成最终 token
                // 这里不需要 amax/SF，因为 BF16 是直接量化的。
                // ==========================================================
                DG_STATIC_ASSERT(STORE_BLOCK_M % 8 == 0, "Invalid store M");
                constexpr uint32_t kNumRowsPerWarp = STORE_BLOCK_M / 8;

                #pragma unroll
                for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                    // 整个 store block 超出 valid_m → 尾部 padding 行，直接释放 TMEM slot 并跳出
                    // TODO: check performance
                    if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        break;
                    }

                    #pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_M / ATOM_M; ++ i) {
                        // TMEM → 寄存器：16dp256b 指令一次取 16 行 × 256 bit，两条指令覆盖 ATOM_M = 8 行
                        //   两条指令通过 tmem_addr 的 bit 20 区分"上半 / 下半 column bank"
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                        uint32_t values[ATOM_M];
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               values[0], values[1], values[2], values[3]);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               values[4], values[5], values[6], values[7]);
                        cutlass::arch::fence_view_async_tmem_load();

                        // 等前一轮 NVLink store 释放 smem；第一个 atom 不用等（上一次 full_barrier 已保证）
                        if (i == 0 and s > 0)
                            ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                        // 最后一个 atom 时释放 TMEM：后续 MMA 可以 reuse 这块 accum 槽
                        if (s == WG_BLOCK_M / STORE_BLOCK_M - 1 and i == STORE_BLOCK_M / ATOM_M - 1) {
                            ptx::tcgen05_before_thread_sync();
                            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        }

                        // FP32 → BF16 + pack：cast_into_bf16_and_pack 把两个 float 压成一个 uint32_t
                        //   STSM 寻址：
                        //     row = lane_idx % 8        （8 行一组，对应 ATOM_M = 8）
                        //     col = 2 个 warp 共享一个 BF16 swizzle atom（warp 0/1 vs warp 2/3）
                        //     swizzle = col ^ row       （避免 bank 冲突）
                        uint32_t row = lane_idx % 8;
                        uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                        const auto smem_ptr = smem_cd_l2 +
                            epilogue_wg_idx * STORE_BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                            (warp_idx_in_wg / 2) * STORE_BLOCK_M * kSwizzleCDMode +
                            i * ATOM_M * kSwizzleCDMode +
                            row * (kNumBankGroupBytes * 8) +
                            (col ^ row) * kNumBankGroupBytes;
                        ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                            math::cast_into_bf16_and_pack(values[0], values[1]),
                            math::cast_into_bf16_and_pack(values[2], values[3]),
                            math::cast_into_bf16_and_pack(values[4], values[5]),
                            math::cast_into_bf16_and_pack(values[6], values[7]),
                            smem_ptr
                        );
                    }

                    // STSM 完成后，warpgroup 内 4 个 warp 对齐 → 保证 smem 可被所有 warp 读取
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // ------------------------------------------------------
                    // NVLink 写入阶段：每 warp 负责 kNumRowsPerWarp 行
                    //   布局不同于 STSM：这里以 "每 16 lane 写一行 × 16 个 float4" 的方式展开
                    //   1 行 = BLOCK_N * sizeof(bf16) / sizeof(float4) = 16 × 16B = 256B
                    // ------------------------------------------------------
                    const uint32_t row_in_atom = (warp_idx_in_wg * 2 + lane_idx / 16) % ATOM_M;
                    const uint32_t bank_group_idx = lane_idx % 8;

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_store = j * 8 + warp_idx_in_wg * 2 + lane_idx / 16;
                        const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + row_in_store;

                        // 跳过超过 expert 有效 token 数的 padding 行
                        if (m_idx_in_block >= valid_m)
                            break;

                        // 从 dispatch 阶段记录的 metadata 里拿回源 (rank, token, topk) —— 用于 Combine 回写
                        const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                        const uint32_t dst_rank_idx = src_metadata.rank_idx;
                        const uint32_t dst_token_idx = src_metadata.token_idx;
                        const uint32_t dst_topk_idx = src_metadata.topk_idx;

                        // 从 smem 读出 BF16 packed 数据（128B 对齐，load 一个 float4 = 8 个 bf16）
                        const auto smem_ptr = smem_cd_l2 +
                            epilogue_wg_idx * STORE_BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                            (lane_idx % 16 / 8) * STORE_BLOCK_M * kSwizzleCDMode +
                            row_in_store * kSwizzleCDMode +
                            (bank_group_idx ^ row_in_atom) * kNumBankGroupBytes;
                        const auto packed = ptx::ld_shared(reinterpret_cast<float4*>(smem_ptr));

                        // 写入远端 combine buffer：
                        //   combine_token_buffer.get_rank_buffer(dst_topk_idx)      本 rank 上第 topk_idx 个分片
                        //       .get_data_buffer(dst_token_idx)                      该分片内第 token_idx 个 token 槽
                        //   dst_ptr 按 n_idx + lane 的 16 byte 切片定位最终地址
                        //   sym_buffer.map(local_ptr, dst_rank_idx) → 转换为 NVLink 映射的远端地址
                        const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);
                        const auto dst_ptr = math::advance_ptr<float4>(
                            dst_token.get_base_ptr(),
                            n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) + (lane_idx % 16) * static_cast<uint32_t>(sizeof(float4)));
                        *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                    }
                }

                // 保证下一个 epilogue 可以安全 reuse smem_cd_l2
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            }
        });

        // ==================================================================
        // Epilogue 结束：释放 TMEM，并为 Combine 阶段做全局同步
        // ------------------------------------------------------------------
        //   1) TMEM free —— 必须由 2-CTA 中逻辑 warp ID 相同的线程 call（硬件约束）
        //   2) nvlink_barrier —— 确保所有 rank 都已完成 Linear2 写入目标 rank 的
        //      combine buffer；本 rank 才能安全地读本地的 combine buffer 做 reduce
        //   3) Dispatch+Epilogue 线程合流一次（unaligned barrier）→ Dispatch warps
        //      接下来会清理 workspace（count/mask/metadata 置 0），与 Combine 并行
        // ==================================================================
        if (epilogue_warp_idx == 0)
            Allocator().free(0, kNumTmemCols);

        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                             kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
            workspace, sym_buffer, sm_idx, epilogue_thread_idx,
            [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
        );

        // 与 Dispatch 线程汇合：此刻 Dispatch warps 阻塞在 kDispatchWithEpilogueBarrierIdx，
        //   放行后它们会做 workspace 清理（L1 count、L2 mask、metadata 归零）为下一 launch 做准备。
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // ==================================================================
        // Combine 阶段：在本 rank 上 reduce top-k 个贡献 → 最终输出 y
        // ------------------------------------------------------------------
        // 每个 token 在本 rank 的 combine_token_buffer 里有 **至多 kNumTopk 个槽位**，
        // 对应它在不同 expert 上的部分计算结果。此阶段要：
        //   1) 找出该 token 实际被 dispatch 到的 k' 个 expert（由 topk_idx == slot 不等于 -1 区分）
        //   2) 依次 TMA 加载每个 slot 的 chunk → 累加到 float 寄存器
        //   3) 累加完成后 cast 回 BF16 → TMA store 到最终输出 y
        //
        // 单 token/topk 的延迟约 3 us；整个 Combine 阶段的并行度是 SM × Epilogue warp。
        //
        // 共享内存复用策略：
        //   - smem 布局：[0 .. barrier_start_ptr) 是 epilogue 里 SwiGLU/STSM 用过的区域，
        //     barrier 之后已空闲 → Combine 直接从 smem_buffer(=0) 处复用
        //   - 每 warp 3 个 chunk slot：2 个 load stage（double-buffer）+ 1 个 store buffer
        // ==================================================================
        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        // 每 warp 3 slot：load_stage 0 / load_stage 1 / store
        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;

        // 根据 smem / 寄存器容量决定把 hidden 切成 1 块还是 2 块
        //   - 切 1 块：吞吐更高（少一次 TMA 往返），但需要 smem / 寄存器足够大
        //   - 切 2 块：更省 smem，代价是多一次同步
        constexpr uint32_t kNumChunks =
            kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE and kHidden <= 32 * kNumMaxRegistersForBuffer ? 1 : 2;
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks <= SMEM_BEFORE_BARRIER_SIZE, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements (one per lane)");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

        // Verify combined shared memory budget at runtime
        DG_DEVICE_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes <= static_cast<uint32_t>(
            reinterpret_cast<uint8_t*>(barrier_start_ptr) - smem_buffer));

        // Per-warp buffer：前 2*kNumEpilogueWarps*chunk_bytes 区是 load double-buffer，
        //   紧随其后的 kNumEpilogueWarps*chunk_bytes 是 store buffer
        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer  = math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);

        // 每 warp 2 个 combine_load barriers（load stage 0/1），已在 kernel 初始化阶段 init(1)
        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return combine_barriers[i + epilogue_warp_idx * 2];
        });

        // 遍历本 rank 输出的所有 token：SM × warp 交错分配（wave-uniform），负载均衡
        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
        for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumEpilogueWarps) {
            // 读该 token 的 topk slot index：
            //   - 每 lane 读一个（kNumTopk <= 32 时正好一个 warp）
            //   - slot_idx >= 0 表示该位置被某个 expert 选中，需要从 combine buffer 拉数据
            //   - total_mask 是所有"有效 slot"的 bitmask，后续用 __ffs 循环 pop
            DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            // 按 chunk 处理（当 hidden 较大时切 2 块；小的时候 chunk 就 = hidden）
            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;

                // move_mask_and_load：从 mask 中取最低有效位（对应下一个要 load 的 topk slot），
                //   发起 TMA 异步 load 到指定 stage 的 smem buffer，并设置 mbarrier 的 transaction bytes
                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) {
                    if (mask) {
                        const uint32_t slot_idx = __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;

                        // 只让 lane 0 发 TMA（TMA 是 per-warp 指令，但只需 1 thread 发起）
                        if (cute::elect_one_sync()) {
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot_idx)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                chunk_byte_offset);
                            ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                        }
                        __syncwarp();
                        return true;
                    }
                    return false;
                };

                // 发起首个 load（load_stage_idx 当前是 0 或 1，与上一 token 结束时相反）
                bool do_reduce = move_mask_and_load(load_stage_idx);

                // 累加器：每 lane 持有 chunk 的一部分数据，float 精度
                //   reduced[j] 的总大小 = kNumUint4PerLane * kNumElemsPerUint4 float2
                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                while (do_reduce) {
                    // 发起"下一个 slot"的 TMA load（stage = current ^ 1），与当前 accumulate 并行
                    do_reduce = move_mask_and_load(load_stage_idx ^ 1);

                    // 等当前 stage 的 load 完成（mbarrier phase 机制：每次 wait 后 phase 翻转）
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                    }
                    // 只翻转当前 stage 对应的 phase bit（两 stage 的 phase 独立追踪）
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                }

                // reduced 已经累加完所有 topk → cast 回 BF16 写入 store buffer
                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

                    // 第一个 element 前等上一次 TMA store 完成（store buffer 只有一份）
                    if (j == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }
                    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                   casted.x, casted.y, casted.z, casted.w);
                }
                __syncwarp();

                // TMA store：把本 warp 持有的 chunk 写入最终输出 y[token_idx, chunk_offset:]
                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                        combine_store_buffer, kNumChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
        }
        // 注：此处不再做全局 barrier —— Combine 写入的是每个 rank 自己的 y，
        //     无需跨 rank 同步；kernel 生命周期由 host side 的 stream 保证
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm
