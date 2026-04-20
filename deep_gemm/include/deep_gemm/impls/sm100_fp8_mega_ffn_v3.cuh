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
#include <cutlass/arch/grid_dependency_control.h>
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
sm100_fp8_mega_ffn_v3_impl(
    void* y,
    float* y_fp32,                 // fp32 slot buffer, layout **[cta][h_local][m]** (v3 new layout)
    void* workspace,               // FP8 intermediate [kMaxM][kIntermediate]
    uint32_t* l1_done_counter,
    uint32_t* l2_tile_counters,
    const uint32_t num_tokens,
    // **v3 = swap AB + PT**: Per-Tensor scales (代替 UE8M0 SF)
    const float scale_xw1,                 // = scale_X * scale_W1
    const float scale_inv_intermediate,    // = 1 / scale_intermediate
    const float scale_iw2,                 // = scale_intermediate * scale_W2
    const __grid_constant__ cute::TmaDescriptor tensor_map_x,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w1,          // **v3**: pre-merged layout
    const __grid_constant__ cute::TmaDescriptor tensor_map_w2,
    const __grid_constant__ cute::TmaDescriptor tensor_map_interm,      // write view
    const __grid_constant__ cute::TmaDescriptor tensor_map_interm_load, // read view
    const __grid_constant__ cute::TmaDescriptor tensor_map_y) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier   = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator1Sm;

    // -----------------------------------------------------------------------------
    // 编译期常量校验
    // -----------------------------------------------------------------------------
    // Step 14/15 cluster 修通: arrive(0u) → arrive() 修 cluster-aware bug;
    // post_l2_sync 在 cluster=kL2KSplit 时改用 hardware cluster_sync (省 HBM atomic)。
    DG_STATIC_ASSERT(kClusterDim == 1 or kClusterDim == kL2KSplit,
                     "kClusterDim must be 1 (legacy) or kL2KSplit (Step 15 cluster_sync)");
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

    // UMMA 形状（**Stage-B swap**：MMA-A 是 W, MMA-B 是 X）
    // 硬件 1-CTA MXF8F6F4 约束：M==128, N % 8 ∈ [8, 256]。
    // 把 W 当 A → UMMA_M=128 = W 的 BLOCK_N（全用满，无浪费）。
    // 把 X 当 B → UMMA_N=kMaxValidM=32（X 的有效 token 数）。TMEM accumulator
    // 从 [128 M × 128 N] 缩到 [128 W-N × 32 X-M]，4× 节省，且 SMEM_B 16KB→4KB。
    // 后果：TMEM 输出是 C^T，epi 需 transpose 读 + 跨 warp SMEM 交换才能做 SwiGLU。
    constexpr uint32_t kMaxValidM = 32;       // 当前 swap 仅适配 decoding (valid_m ≤ 32)
    DG_STATIC_ASSERT(kMaxM >= kMaxValidM, "Test kMaxM must cover swap kMaxValidM");
    constexpr uint32_t UMMA_M = 128;          // = W's BLOCK_N (fully used)
    constexpr uint32_t UMMA_N = kMaxValidM;   // = X's M
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_N;   // A 侧 SMEM tile MN-extent = W 的 N = 128
    constexpr uint32_t LOAD_BLOCK_N = kMaxValidM; // B 侧 SMEM tile MN-extent = X 的 valid M = 32
    DG_STATIC_ASSERT(UMMA_N % 8 == 0 and UMMA_N >= 8 and UMMA_N <= 256, "UMMA_N must be 8..256 step 8");

    // **v3 PT**: 无 SF pipeline. scale_up_comb = scale_xw1 * scale_inv_intermediate
    // 用于 L1 epi 把 silu(gate_real)*up_real 折算到 FP8 intermediate scale.

    // Swizzle configs（与 MegaMoE 一致）
    constexpr uint32_t kSwizzleAMode  = BLOCK_K * sizeof(a_dtype_t);  // 128
    constexpr uint32_t kSwizzleBMode  = BLOCK_K * sizeof(b_dtype_t);  // 128
    constexpr uint32_t kSwizzleCDMode = 128;
    DG_STATIC_ASSERT(BLOCK_N % kSwizzleCDMode == 0, "Invalid BLOCK_N vs swizzle CD mode");

    // Epilogue 流水
    // Step 34 实验: epi_stages 2→3 退步 +0.5%, 2→4 超 TMEM cap。kernel 不是 pipeline-depth
    // bound; 加深 epi 流水反而搅乱 mbarrier/scheduler。保留 2 stages。
    constexpr uint32_t kNumEpilogueStages   = 2;
    constexpr uint32_t kNumTMAStoreStages   = 2;
    constexpr uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4;
    DG_STATIC_ASSERT(kNumEpilogueWarps % 4 == 0, "Epilogue warps must be warpgroup aligned");

    // -----------------------------------------------------------------------------
    // 形状派生
    // -----------------------------------------------------------------------------
    // **swap 后**:
    //   Linear1: A=W1[2I,H]  B=X[M,H]   C^T=acc[2I(UMMA_M=128) × M(UMMA_N=32)]，N 遍历 2I/BLOCK_N 次
    //   Linear2: A=W2[H,I]   B=I[M,I]    C^T=acc[H_local(128) × M(32)]，每 CTA 负责 1 个 H/BLOCK_N 的 N-tile
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

    // Step3/6/10: cta_idx 解码
    //   即便 cluster launch=1，也始终用 cluster-style 映射（连续 cta_idx 共享 N-tile），
    //   让同一 N-tile 的 kL2KSplit 个 CTA 调度到连续 SM，slot 数据互相在同一 GPC L2 段，
    //   cast+reduce 阶段 8 个跨 CTA slot 读的 cache locality 显著改善。
    const uint32_t l2_n_tile_for_cta = cta_idx / kL2KSplit;   // 连续 8 个 CTA → 1 个 N-tile
    const uint32_t l2_k_half_for_cta = cta_idx % kL2KSplit;
    const uint32_t l2_k_base_block   = l2_k_half_for_cta * kL2KBlocksPerCta;

    // Step 4：Linear1 N-split 允许 uneven —— 前若干个 CTA 做 1~kL1NPerCtaCeil 个 L1 tile，
    //         剩余 CTA (cta_idx * kL1NPerCtaCeil >= kL1OutputBlocksN) l1_n_count==0 直接跳过 L1。
    const uint32_t l1_n_start_raw = cta_idx * kL1NPerCtaCeil;
    const uint32_t l1_n_start     = l1_n_start_raw < kL1OutputBlocksN ? l1_n_start_raw : kL1OutputBlocksN;
    const uint32_t l1_n_end       = (l1_n_start + kL1NPerCtaCeil) < kL1OutputBlocksN
                                        ? (l1_n_start + kL1NPerCtaCeil)
                                        : kL1OutputBlocksN;
    const uint32_t l1_n_count     = l1_n_end - l1_n_start;  // 可能为 0

    // Step 6+: PDL device-side trigger —— kernel 一进来就立刻释放 dependent，
    // 让 next launch 可以 overlap 整个当前 kernel 的执行。benchmark 场景下 iter 间
    // 完全无数据依赖（每 iter 重写同一个 y），所以早释放纯收益。
    cutlass::arch::launch_dependent_grids();

    // **v3**: prefetch 6 TMA descriptors (no SF descriptors).
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_x);
        cute::prefetch_tma_descriptor(&tensor_map_w1);
    } else if (warp_idx == 1) {
        cute::prefetch_tma_descriptor(&tensor_map_w2);
        cute::prefetch_tma_descriptor(&tensor_map_y);
    } else if (warp_idx == 2) {
        cute::prefetch_tma_descriptor(&tensor_map_interm);
        cute::prefetch_tma_descriptor(&tensor_map_interm_load);
    }

    // -----------------------------------------------------------------------------
    // SMEM 布局（**v3**: 无 SF buffer, 无 swap 交换 buffer — 预合并 W1 layout + intra-warp shfl 消除 cross-warp 交换）
    //   [CD buffer] [A×kNumStages (W)] [B×kNumStages (X)] [amax] [barriers] [tmem_ptr]
    // -----------------------------------------------------------------------------
    constexpr uint32_t STORE_BLOCK_M = BLOCK_M;
    constexpr uint32_t kSharedMemAlignment = 1024;

    constexpr uint32_t SMEM_CD_L1_SIZE_PER_STAGE =
        kNumEpilogueWarpgroups * STORE_BLOCK_M * L1_OUT_BLOCK_N * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_CD_L1_SIZE  = SMEM_CD_L1_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_CD_L2_SIZE_PER_STAGE = (kL2KSplit == 1)
        ? (kNumEpilogueWarpgroups * STORE_BLOCK_M * BLOCK_N * sizeof(cd_dtype_t))
        : 0u;
    constexpr uint32_t SMEM_CD_L2_SIZE  = SMEM_CD_L2_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_CD_SIZE     = SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;

    constexpr uint32_t SMEM_A_SIZE_PER_STAGE   = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);  // 128*128 = 16KB (W)
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE   = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);  // 32*128 = 4KB  (X)

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
    // amax 归约区（不再需要, 保留占位以便后续扩展）
    auto smem_amax_reduction = reinterpret_cast<float2*>(math::advance_ptr<uint8_t>(smem_buffer,
        SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE)));
    constexpr uint32_t kAmaxReductionEntries = kNumEpilogueWarps * (STORE_BLOCK_M / 2);

    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_amax_reduction + kAmaxReductionEntries);
    auto full_barriers      = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
    auto tmem_full_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + 2 * kNumStages + i; });
    auto tmem_empty_barriers= utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + 2 * kNumStages + kNumEpilogueStages + i; });
    auto tmem_ptr_in_smem   = reinterpret_cast<uint32_t*>(barrier_start_ptr + 2 * kNumStages + 2 * kNumEpilogueStages);

    // **v3** TMEM 分配 (PT: 只 accumulator, 无 SF cols)
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumTmemCols      =
        utils::get_num_aligned_tmem_cols<kNumAccumTmemCols>();

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
    // mbarrier 都是 CTA-local，__syncthreads 已足够提供初始化可见性。即使 cluster 模式
    // 下也无须 cluster_sync — 只要初始化 mbarrier 在 post-L2 cluster_sync 前对所有 CTA
    // 可见即可（自动满足，因为 cluster_sync 本身有 .acquire 语义）。
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
    constexpr uint32_t kGridSyncBarrierIdx       = 15;  // l1_to_l2_sync, count=kNumThreads
    constexpr uint32_t kPostL2SyncBarrierIdx     = 14;  // post_l2_sync_epi, count=kNumEpilogueThreads

    // Linear1 → Linear2 之间跨 CTA grid-sync lambda (Step 4 重写, 灵感源自
    // deep_gemm::comm::grid_sync 的"bit31 翻转 + ld.acquire 轮询"模式)：
    //   * 所有非 CTA0 的 thread0 atomic_add_rel(+1)，CTA0 的 thread0 atomic_add_rel(kFinishTag - (N-1))；
    //     凑齐后 bit31 正好翻转一次，作为本轮"到齐"标志
    //   * spin 用 `ld.acquire.gpu`（普通 load + acquire 语义）而不是 atomicAdd(counter, 0)，
    //     把重度的全局 atomic 轮询换成轻量 load，大幅缓解 hot atomic 的 HBM/L2 排队
    //   * bit31 翻转天然抗 ABA，无需 host 每次 launch 前 memset counter
    constexpr uint32_t kGridSyncFinishTag = 0x80000000u;
    // Step 18/19: l1_to_l2 grid_sync 排除 cold warp + TMA-B warp (count=192).
    //   epi 不能跳过 (epi 的 TMA store 必须在 thread 0 atomic 前同步)
    //   MMA 跳过 (Step 24) 反而退步 (11.14→11.31), 推测 MMA 提前进 L2 wait 影响 schedule
    // **v3.2 retry**: atomic master 已迁到 first epi thread, TMA-A 可安全跳过 grid_sync.
    // 仅 cold warp + TMA-A 不参与 (TMA-A 加载 W, 与 L1 输出无关).
    constexpr uint32_t kGridSyncThreads = kNumThreads - 32 - 32;
    auto grid_sync_l1_to_l2 = [&] () {
        ptx::sync_aligned(kGridSyncThreads, kGridSyncBarrierIdx);
        // **v3.2**: atomic master 改为首个 epi 线程 (thread_idx==kNumNonEpilogueThreads).
        // 之前用 thread_idx==0 但它在 TMA-A warp 里, 而 TMA-A 现在跳过 grid_sync,
        // 会导致 atomic 永不 increment, 全 grid hang/race.
        if (thread_idx == kNumNonEpilogueThreads) {
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
        ptx::sync_aligned(kGridSyncThreads, kGridSyncBarrierIdx);
    };

    // Step 6: post-L2 sync —— 当 cluster==kL2KSplit 时，一个 cluster 内的 8 个 CTA
    // 正好是同一 L2 N-tile 的 8 个 K-split 贡献者，它们的 atomicAdd 到 y_fp32
    // 只需要 cluster 内可见，不需要全 grid 可见。用硬件 cluster_sync 替代 global
    // atomic-counter 等待，省掉 ~1 µs 的 CTA barrier stall。
    // Step 7+9: post-L2 sync. 非 epi warp 在 L2 完成后已经无事可做，无须参与；
    // 将 CTA-scope bar.sync 从 256-thread 降到 128-thread (只 epi)，省一个全局 bar 等待。
    // bar id 仍用 kGridSyncBarrierIdx=15 (与 epilogue 内部的 bar id=0 不冲突)。
    auto post_l2_sync_cluster = [&] () {
        // Step 29: 用 cluster-scope fence (`fence.acq_rel.cluster`) 替 GPU-scope
        // `__threadfence()`. slot 是 HBM/L2 可达，但 peer reader 同 cluster
        // (同 GPC, 共享 L2)，cluster-scope fence 已足够保证可见性，且比
        // MEMBAR.ALL.GPU 便宜 (不下 sys-scope barrier，只在 cluster L2 域 fence)。
        // Step 31 实验把 fence 收窄到 epi only — 反而 -2% (11.02→11.34)。
        // Step 32 实验把 fence 折进 st.release.cluster — 灾难性 28µs (3x 退步),
        // 推测 release-store 每条都 stall 等待自身被 commit 才能继续, 16 store/CTA
        // 串行化整个 epi。保留 Step 29 fence 单点 + 普通 store。
        asm volatile("fence.acq_rel.cluster;");
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
    };

    // Step 33 实验: 非 epi warp 跳 cluster_wait 提前退出 — 11.02→11.13µs (+1.0%)。
    // 推测 cluster_wait 在 cluster scope 上低成本, exiting warp 反而搅乱 scheduler.
    // 保留全部 warp 都 cluster_arrive + cluster_wait (post_l2_sync_cluster)。

    // Step 9: epi-only per-tile counter post-L2 sync (默认路径)
    auto post_l2_sync_epi = [&] () {
        constexpr uint32_t kTileFinishTag = 0x80000000u;
        ptx::sync_aligned(kNumEpilogueThreads, kPostL2SyncBarrierIdx);
        if (thread_idx == kNumNonEpilogueThreads) {
            const uint32_t is_leader = (l2_k_half_for_cta == 0) ? 1u : 0u;
            const uint32_t delta = is_leader
                ? (kTileFinishTag - (kL2KSplit - 1u))
                : 1u;
            uint32_t* ctr = l2_tile_counters + l2_n_tile_for_cta;
            const uint32_t old_val = ptx::atomic_add_rel(ctr, delta);
            uint32_t new_val;
            do {
                new_val = ptx::ld_acq(ctr);
            } while (((new_val ^ old_val) & kTileFinishTag) == 0);
        }
        ptx::sync_aligned(kNumEpilogueThreads, kPostL2SyncBarrierIdx);
    };
    // L1→L2 sync 必须 grid-wide：intermediate 跨 cluster 共享 (HBM)，cluster_sync
    // 只同步本 cluster 的 CTA，其它 cluster 的 intermediate 可能没 commit 到 HBM。
    auto l1_to_l2_sync = [&] () {
        grid_sync_l1_to_l2();
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
    // 角色 ①：TMA-A warp（**v3 swap**：A = W → smem_a, 无 SF）
    // -------------------------------------------------------------------------
    if (warp_idx == 0) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        auto tma_a_loop = [&](auto phase_tag_c,
                              const cute::TmaDescriptor* tensor_map_w_ptr,
                              uint32_t num_n_blocks, uint32_t n_base) {
            constexpr Phase phase_tag = decltype(phase_tag_c)::value;
            constexpr uint32_t num_k_blocks =
                (phase_tag == Phase::Linear1) ? kL1KBlocks : kL2KBlocksPerCta;
            const uint32_t k_base_block =
                (phase_tag == Phase::Linear1) ? 0u : l2_k_base_block;

            for (uint32_t n_tile = 0; n_tile < num_n_blocks; ++ n_tile) {
                const uint32_t n_idx = (n_base + n_tile) * BLOCK_N;
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    empty_barriers[stage_idx]->wait(phase ^ 1);
                    const uint32_t abs_k_block = k_base_block + k_block_idx;
                    uint32_t k_idx = abs_k_block * BLOCK_K;

                    if (cute::elect_one_sync()) {
                        tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t, false>(
                            tensor_map_w_ptr, full_barriers[stage_idx], smem_a[stage_idx],
                            k_idx, n_idx, 1);
                        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE);
                    }
                    __syncwarp();
                }
            }
        };

        tma_a_loop(std::integral_constant<Phase, Phase::Linear1>{},
                   &tensor_map_w1, l1_n_count, l1_n_start);
        // **v3.2**: TMA-A skip l1_to_l2_sync. atomic master 在 epi 而非 thread 0, 所以 atomic 仍正常执行.
        // TMA-A 提前开始 W2 TMA load — 但 MMA 仍受 TMA-B (intermediate) 制约, 实测 win 待验证.
        tma_a_loop(std::integral_constant<Phase, Phase::Linear2>{},
                   &tensor_map_w2, kL2NPerCta, l2_n_tile_for_cta);
        if constexpr (kClusterDim == kL2KSplit)
            post_l2_sync_cluster();

    // =========================================================================
    // 角色 ②：TMA-B warp（**v3 swap**：B = X / intermediate → smem_b, 无 SF, small box=(K=128, M=32)）
    // -------------------------------------------------------------------------
    } else if (warp_idx == 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        auto tma_b_loop = [&](auto phase_tag_c,
                              const cute::TmaDescriptor* tensor_map_x_ptr,
                              uint32_t num_n_blocks) {
            constexpr Phase phase_tag = decltype(phase_tag_c)::value;
            constexpr uint32_t num_k_blocks =
                (phase_tag == Phase::Linear1) ? kL1KBlocks : kL2KBlocksPerCta;
            const uint32_t k_base_block =
                (phase_tag == Phase::Linear1) ? 0u : l2_k_base_block;
            constexpr uint32_t m_idx = 0;

            for (uint32_t n_tile = 0; n_tile < num_n_blocks; ++ n_tile) {
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    empty_barriers[stage_idx]->wait(phase ^ 1);
                    const uint32_t abs_k_block = k_base_block + k_block_idx;
                    uint32_t k_idx = abs_k_block * BLOCK_K;

                    if (cute::elect_one_sync()) {
                        tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t, false>(
                            tensor_map_x_ptr, full_barriers[stage_idx], smem_b[stage_idx],
                            k_idx, m_idx, 1);
                        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE);
                    }
                    __syncwarp();
                }
            }
        };

        tma_b_loop(std::integral_constant<Phase, Phase::Linear1>{},
                   &tensor_map_x, l1_n_count);
        // v3: TMA-B L2 读 intermediate (= L1 输出), 必须等 L1 done
        l1_to_l2_sync();
        tma_b_loop(std::integral_constant<Phase, Phase::Linear2>{},
                   &tensor_map_interm_load, kL2NPerCta);
        if constexpr (kClusterDim == kL2KSplit)
            post_l2_sync_cluster();

    // =========================================================================
    // 角色 ③：MMA warp —— 发 UMMA 指令、UTCCP SF、commit barrier
    // -------------------------------------------------------------------------
    } else if (warp_idx == 2) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // **v3 PT**: 非 block-scaled f8f6f4 MMA descriptor.
        auto instr_desc = cute::UMMA::make_instr_desc<
            a_dtype_t, b_dtype_t, float,
            UMMA_M, UMMA_N,
            cute::UMMA::Major::K, cute::UMMA::Major::K>();
        const uint64_t runtime_desc_u64 = cute::UMMA::make_runtime_instr_desc(instr_desc);

        auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);

        DG_STATIC_ASSERT(kNumStages <= 32, "kNumStages must fit in one warp");
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * (SMEM_A_SIZE_PER_STAGE / 16) : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * (SMEM_B_SIZE_PER_STAGE / 16) : 0u;

        uint32_t current_iter_idx = 0;

        auto run_mma = [&](Phase phase_tag, uint32_t num_n_blocks) {
            const uint32_t num_k_blocks =
                (phase_tag == Phase::Linear1) ? kL1KBlocks : kL2KBlocksPerCta;
            for (uint32_t n_tile = 0; n_tile < num_n_blocks; ++ n_tile) {
                const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                const auto accum_phase     = (current_iter_idx / kNumEpilogueStages) & 1;
                ++ current_iter_idx;

                tmem_empty_barriers[accum_stage_idx]->wait(accum_phase ^ 1);
                ptx::tcgen05_after_thread_sync();

                auto empty_arrive = [&](bool do_tmem_full) {
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
                    if (do_tmem_full)
                        cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                    __syncwarp();
                };

                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
                    ptx::tcgen05_after_thread_sync();

                    const auto a_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                    const auto b_base_lo = ptx::exchange(b_desc_lo, stage_idx);

                    if (cute::elect_one_sync()) {
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                            a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_base_lo, 0, k * UMMA_K);
                            b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_base_lo, 0, k * UMMA_K);
                            ptx::SM100_MMA_F8F6F4_SS::fma(
                                a_desc, b_desc,
                                accum_stage_idx * UMMA_N,
                                k_block_idx > 0 or k > 0,
                                runtime_desc_u64);
                        }
                    }
                    __syncwarp();
                    empty_arrive(k_block_idx == num_k_blocks - 1);
                }
            }
        };

        run_mma(Phase::Linear1, l1_n_count);
        l1_to_l2_sync();
        run_mma(Phase::Linear2, kL2NPerCta);

        // Drain last tmem_empty
        if (current_iter_idx > 0) {
            const auto last = current_iter_idx - 1;
            tmem_empty_barriers[last % kNumEpilogueStages]->wait((last / kNumEpilogueStages) & 1);
        }

        // Step 15: cluster=kL2KSplit 时 MMA warp 参与 cluster_sync
        if constexpr (kClusterDim == kL2KSplit)
            post_l2_sync_cluster();

    // =========================================================================
    // 角色 ④：冷 warp（降低寄存器占用）
    // -------------------------------------------------------------------------
    } else if (warp_idx == 3) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
        // Step 18: cold warp 跳过 l1_to_l2_sync（grid_sync 现在用 224-thread bar.sync）
        // 但仍参与 post-L2 cluster_sync（aligned 要求全 256 thread）
        if constexpr (kClusterDim == kL2KSplit)
            post_l2_sync_cluster();

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

        // -------- Linear1 epilogue (**v3: pre-merged W1 + intra-warp shfl_xor + PT**) --------
        //
        // **前置条件** (host 已做): W1 N 维按 [g0..15, u0..15, g16..31, u16..31, ...] 重排,
        //   每 32-row block 前 16 是 gate 后 16 是 up. 这样 swap 后 TMEM 每 warp 内同时持有
        //   gate + up, 不再需要 SMEM cross-warp exchange.
        //
        // TMEM C^T = [128 W-N rows × UMMA_N=32 X-M cols], 每 warp 32 rows. Lane mapping:
        //   warp w lane l ∈ [0, 32):
        //     l < 16:  TMEM row w*32+l = gate for output_n = w*16 + l
        //     l >= 16: TMEM row w*32+l = up   for output_n = w*16 + (l-16)
        //   全部 4 warps 覆盖 output_n [0, 64).
        //
        // 步骤:
        //   P1 (all): TMEM_LOAD own row's 32 fp32 vals.
        //   P2 (all): __shfl_xor_sync(_, v, 16) 拿 partner's row → 同 lane 得 (gate, up) 对.
        //   P3 (wait): tma_store_wait + sync_aligned.
        //   P4 (gate lanes l<16 only): compute silu(g*scale_xw1) * up * scale_up_comb → FP8;
        //     write 32 bytes to smem_cd[m, output_n_local=w*16+l].
        //   P5 (all): tma_store_fence + sync + warp 0 TMA store.

        const float scale_up_comb = scale_xw1 * scale_inv_intermediate;
        const bool is_gate_lane = lane_idx < 16;
        const uint32_t output_n_local = epilogue_warp_idx * 16 + (lane_idx & 15);  // 0..63 gate-lane global output_n

        constexpr uint32_t kSwapAtomN = 8;
        constexpr uint32_t kSwapAtomCount = UMMA_N / kSwapAtomN;  // = 4

        uint32_t tma_stage_idx_l1 = 0;
        for (uint32_t n_tile = 0; n_tile < l1_n_count; ++ n_tile) {
            const uint32_t global_n_tile = l1_n_start + n_tile;
            (void)global_n_tile;
            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const auto accum_phase     = (current_iter_idx / kNumEpilogueStages) & 1;
            ++ current_iter_idx;

            tmem_full_barriers[accum_stage_idx]->wait(accum_phase);
            ptx::tcgen05_after_thread_sync();

            const uint32_t valid_m = num_tokens;

            // ---- P1: TMEM_LOAD 1 W-N row × UMMA_N=32 X-M cols, 单次 32x atom ----
            // **v3.4**: 用 32dp32b32x 一次读 32 cols (vs 之前 4× 32dp32b8x). 省 3× TMEM 读延迟.
            uint32_t v[32];
            const uint32_t tmem_base = accum_stage_idx * UMMA_N;
            cute::SM100_TMEM_LOAD_32dp32b32x::copy(tmem_base,
                v[0],  v[1],  v[2],  v[3],  v[4],  v[5],  v[6],  v[7],
                v[8],  v[9],  v[10], v[11], v[12], v[13], v[14], v[15],
                v[16], v[17], v[18], v[19], v[20], v[21], v[22], v[23],
                v[24], v[25], v[26], v[27], v[28], v[29], v[30], v[31]);
            float own_vals[UMMA_N];
            #pragma unroll
            for (uint32_t k = 0; k < UMMA_N; ++ k)
                own_vals[k] = __uint_as_float(v[k]);
            cutlass::arch::fence_view_async_tmem_load();

            ptx::tcgen05_before_thread_sync();
            tmem_empty_barriers[accum_stage_idx]->arrive();

            // ---- P2: intra-warp shfl_xor(16) — 拿 partner's row values ----
            float partner_vals[UMMA_N];
            #pragma unroll
            for (uint32_t m = 0; m < UMMA_N; ++ m) {
                uint32_t own_bits = __float_as_uint(own_vals[m]);
                uint32_t partner_bits = __shfl_xor_sync(0xffffffffu, own_bits, 16);
                partner_vals[m] = __uint_as_float(partner_bits);
            }
            // gate lane (l<16): own=gate, partner=up.  up lane (l>=16): own=up, partner=gate (idle later).

            // ---- P3: wait SMEM CD 双缓冲空闲 ----
            // **v3.3**: 删除 sync_aligned. tma_store_wait 是 per-thread, 每 warp 独立等;
            // 各 warp 写的是 SMEM CD 不同 output_n 列 (不同 slice), 无 cross-warp 冲突, sync 多余.
            const uint32_t tma_stage_idx = tma_stage_idx_l1;
            ptx::tma_store_wait<kNumTMAStoreStages - 1>();

            // ---- P4: gate lanes only — SwiGLU + PT quantize + byte store to SMEM CD ----
            auto smem_cd_base = reinterpret_cast<uint8_t*>(smem_cd_l1[tma_stage_idx]);
            if (is_gate_lane) {
                uint8_t fp8_row[UMMA_N];
                #pragma unroll
                for (uint32_t m = 0; m < UMMA_N; m += 4) {
                    float fp8_f[4];
                    #pragma unroll
                    for (uint32_t kk = 0; kk < 4; ++ kk) {
                        const float gate_real = own_vals[m + kk] * scale_xw1;
                        const float up_acc    = partner_vals[m + kk];
                        const float ne     = kFastMath ? __expf(-gate_real) : expf(-gate_real);
                        const float denom  = 1.0f + ne;
                        const float silu_g = kFastMath ? gate_real * math::fast_rcp(denom) : gate_real / denom;
                        fp8_f[kk] = silu_g * (up_acc * scale_up_comb);
                    }
                    const float4 v4 = make_float4(fp8_f[0], fp8_f[1], fp8_f[2], fp8_f[3]);
                    __nv_fp8x4_e4m3 q(v4);
                    uint32_t packed;
                    __builtin_memcpy(&packed, &q, 4);
                    __builtin_memcpy(&fp8_row[m], &packed, 4);
                }
                // Write 32 bytes to SMEM_CD[m, output_n_local]. stride = L1_OUT_BLOCK_N=64 per m row.
                #pragma unroll
                for (uint32_t m = 0; m < UMMA_N; ++ m) {
                    if (m < valid_m)
                        smem_cd_base[m * L1_OUT_BLOCK_N + output_n_local] = fp8_row[m];
                }
            }

            // ---- P5: fence + sync + warp 0 TMA store ----
            cute::tma_store_fence();
            ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
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

        ptx::tma_store_wait<0>();
        l1_to_l2_sync();

        // -------- Linear2 epilogue (**v3: PT + slot[cta][h][m] vectorized**) --------
        //
        // TMEM C^T = [128 h_local rows × UMMA_N=32 X-M cols]. Per lane: 1 h_local row × 32 X-M cols.
        //
        // **v3 slot 重排**: y_fp32 layout 改为 [cta][h_local][m] (vs main 的 [cta][m][h_local]).
        //   每 lane 写 32 contiguous fp32 = 8 × float4 (vectorized). 消除 strided scalar 写.
        //   reduce 步骤相应改 indexing — vectorize on m, strided on h (但 reduce 总写量小, 可接受).

        constexpr uint32_t kSwapAtomN_L2     = 8;
        constexpr uint32_t kSwapAtomCount_L2 = UMMA_N / kSwapAtomN_L2;  // = 4

        uint32_t tma_stage_idx_l2 = 0;
        for (uint32_t n_tile = 0; n_tile < kL2NPerCta; ++ n_tile) {
            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const auto accum_phase     = (current_iter_idx / kNumEpilogueStages) & 1;
            ++ current_iter_idx;

            tmem_full_barriers[accum_stage_idx]->wait(accum_phase);
            ptx::tcgen05_after_thread_sync();

            const uint32_t valid_m = num_tokens;
            const uint32_t global_n_tile = l2_n_tile_for_cta + n_tile;
            const uint32_t h_local = epilogue_warp_idx * 32 + lane_idx;  // 0..127

            // ---- P1: TMEM_LOAD 1 h_local row × UMMA_N=32 X-M cols, 单次 32x atom ----
            // **v3.4**: same upgrade as L1 epi.
            uint32_t v_l2[32];
            const uint32_t tmem_base = accum_stage_idx * UMMA_N;
            cute::SM100_TMEM_LOAD_32dp32b32x::copy(tmem_base,
                v_l2[0],  v_l2[1],  v_l2[2],  v_l2[3],  v_l2[4],  v_l2[5],  v_l2[6],  v_l2[7],
                v_l2[8],  v_l2[9],  v_l2[10], v_l2[11], v_l2[12], v_l2[13], v_l2[14], v_l2[15],
                v_l2[16], v_l2[17], v_l2[18], v_l2[19], v_l2[20], v_l2[21], v_l2[22], v_l2[23],
                v_l2[24], v_l2[25], v_l2[26], v_l2[27], v_l2[28], v_l2[29], v_l2[30], v_l2[31]);
            float vals_l2[UMMA_N];
            #pragma unroll
            for (uint32_t k = 0; k < UMMA_N; ++ k)
                vals_l2[k] = __uint_as_float(v_l2[k]);
            cutlass::arch::fence_view_async_tmem_load();

            ptx::tcgen05_before_thread_sync();
            tmem_empty_barriers[accum_stage_idx]->arrive();

            if constexpr (kL2KSplit == 1) {
                // kL2KSplit==1 path (no K split): direct PT dequant + TMA store.
                // 保留 transposed scalar write 路径 (此路径 kernel 端用得少, 不优化).
                const uint32_t tma_stage_idx = tma_stage_idx_l2;
                ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                auto smem_cd_base = reinterpret_cast<uint8_t*>(smem_cd_l2[tma_stage_idx]);
                #pragma unroll
                for (uint32_t m = 0; m < UMMA_N; ++ m) {
                    if (m < valid_m) {
                        const uint32_t off = m * BLOCK_N * sizeof(cd_dtype_t) + h_local * sizeof(cd_dtype_t);
                        const __nv_bfloat16 bf = __float2bfloat16(vals_l2[m] * scale_iw2);
                        uint16_t bits;
                        __builtin_memcpy(&bits, &bf, 2);
                        const uint32_t smem_addr = __cvta_generic_to_shared(smem_cd_base + off);
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(smem_addr), "h"(bits));
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
                // **v3 K-split slot write [cta][h_local][m]** — vectorized contiguous fp32 writes.
                // Slot 偏移: cta * (BLOCK_N * kMaxM) + h_local * kMaxM + m
                // Per lane: 32 fp32 contiguous = 8 × float4 stores. NO bank conflict, NO strided.
                auto* lane_slot = y_fp32
                    + static_cast<size_t>(cta_idx) * BLOCK_N * kMaxM
                    + static_cast<size_t>(h_local) * kMaxM;
                #pragma unroll
                for (uint32_t m = 0; m < UMMA_N; m += 4) {
                    float4 v4 = make_float4(vals_l2[m], vals_l2[m+1], vals_l2[m+2], vals_l2[m+3]);
                    *reinterpret_cast<float4*>(lane_slot + m) = v4;
                }
            }
        }

        // 等所有 TMA store 完成 (kL2KSplit==1)。
        // K-split>1 路径下后面 post_l2_sync_epi 自带 sync_aligned，无需在此再做一次。
        if constexpr (kL2KSplit == 1) {
            ptx::tma_store_wait<0>();
            ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
        }

        // -----------------------------------------------------------------
        // Step 5: K-split 下把 host 侧的 cast_y_fp32_to_bf16 kernel 融进来。
        //   1) post-L2 grid-sync 等所有 CTA 的 atomicAdd 全局可见
        //   2) 每个 l2_n_tile 由其 l2_k_half==0 的 leader CTA 负责：
        //        - 读 y_fp32 slice [kMaxM][BLOCK_N]
        //        - cast 到 BF16 写入 y（仅对 m<num_tokens 的行）
        //        - 顺手把 y_fp32 slice 归零，省掉下次 launch 前 host memset
        //   3) 其他 l2_k_half!=0 的 CTA 只参与 grid-sync、不写 y
        // grid-sync 复用同一个 l1_done_counter：bit31 翻转天然兼容多轮。
        // -----------------------------------------------------------------
        if constexpr (kL2KSplit > 1) {
            if constexpr (kClusterDim == kL2KSplit)
                post_l2_sync_cluster();
            else
                post_l2_sync_epi();

            // **v3 reduce**: slot 现在是 [cta][h_local][m] layout.
            // Thread mapping: (h_local_in_cta_slice, m_quad). Vectorize on m direction (4 fp32 = float4).
            // 写 y_bf16[m][h] 是 strided over m (跨 kHidden 字节), 但 reduce 数据小 (~8KB/CTA),
            // 可接受 strided 写。
            constexpr uint32_t kColsPerCta = BLOCK_N / kL2KSplit;
            DG_STATIC_ASSERT(BLOCK_N % kL2KSplit == 0, "BLOCK_N must divide kL2KSplit");
            DG_STATIC_ASSERT(UMMA_N % 4 == 0, "UMMA_N must be multiple of 4 for m-direction float4 vectorize");
            const uint32_t valid_m = num_tokens;
            const uint32_t peer_base = l2_n_tile_for_cta * kL2KSplit;
            constexpr uint32_t peer_stride = 1u;
            constexpr uint32_t kMVec = 4;
            const uint32_t m_quads_per_row = (valid_m + kMVec - 1) / kMVec;  // valid_m=32 → 8

            auto* y_bf16 = reinterpret_cast<cd_dtype_t*>(y);
            const uint32_t total_units = kColsPerCta * m_quads_per_row;     // 16 × 8 = 128 = kNumEpilogueThreads
            for (uint32_t flat = epilogue_thread_idx; flat < total_units; flat += kNumEpilogueThreads) {
                const uint32_t h_in_cta = flat / m_quads_per_row;
                const uint32_t mq       = flat % m_quads_per_row;
                const uint32_t h_local  = l2_k_half_for_cta * kColsPerCta + h_in_cta;
                const uint32_t m_base   = mq * kMVec;

                float4 slots[kL2KSplit];
                #pragma unroll
                for (uint32_t k = 0; k < kL2KSplit; ++ k) {
                    const uint32_t peer_cta = peer_base + k * peer_stride;
                    const size_t slot_off = static_cast<size_t>(peer_cta) * BLOCK_N * kMaxM
                                          + static_cast<size_t>(h_local) * kMaxM
                                          + m_base;
                    slots[k] = *reinterpret_cast<const float4*>(y_fp32 + slot_off);
                }
                float4 acc = slots[0];
                #pragma unroll
                for (uint32_t k = 1; k < kL2KSplit; ++ k) {
                    acc.x += slots[k].x; acc.y += slots[k].y;
                    acc.z += slots[k].z; acc.w += slots[k].w;
                }
                // PT dequant
                acc.x *= scale_iw2; acc.y *= scale_iw2;
                acc.z *= scale_iw2; acc.w *= scale_iw2;

                // Write y_bf16[m_base..+4][h_global] — 4 strided BF16 stores
                const uint32_t h_global = l2_n_tile_for_cta * BLOCK_N + h_local;
                const float fvals[4] = {acc.x, acc.y, acc.z, acc.w};
                #pragma unroll
                for (uint32_t mm = 0; mm < kMVec; ++ mm) {
                    const uint32_t m = m_base + mm;
                    if (m < valid_m) {
                        const __nv_bfloat16 bf = __float2bfloat16(fvals[mm]);
                        // cd_dtype_t = cutlass::bfloat16_t. 直接用 reinterpret_cast 写底层 bits.
                        uint16_t bits;
                        __builtin_memcpy(&bits, &bf, 2);
                        reinterpret_cast<uint16_t*>(y_bf16)[static_cast<size_t>(m) * kHidden + h_global] = bits;
                    }
                }
            }
        }

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
