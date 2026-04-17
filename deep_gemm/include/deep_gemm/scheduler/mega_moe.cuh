#pragma once

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

// =====================================================================================
// 调度层 (L3)：Mega MoE 的 persistent block 调度器
// -------------------------------------------------------------------------------------
// 核心职责：在一个 **persistent kernel** 内部，让每个 SM 自行枚举"本 SM 该算哪些 block"。
//
// 调度维度（从外到内）：
//   1. Phase     —— Linear1（FP8×FP4 + SwiGLU）与 Linear2（FP8×FP4 → BF16 写回）轮换。
//   2. Wave      —— 每轮处理 `kNumExpertsPerWave` 个相邻的 local expert（wave 内 L1 都算完再统一进 L2）。
//   3. Expert    —— Wave 内按 expert 顺序遍历。
//   4. M / N     —— 每个 expert 内 token 方向 × N 方向的 block 枚举。
//
// 状态机流程：
//   next_phase = Linear1 → 遍历当前 wave 所有 expert 的 L1 blocks
//        ↓ wave 内 L1 全部派完
//   next_phase = Linear2 → 回退 expert 指针到 wave 起点，再遍历这些 expert 的 L2 blocks
//        ↓ wave 内 L2 全部派完
//   next_phase = Linear1, expert 推进到下一个 wave 起点 → 重复
//
// Block 划分：每次从 `block_idx` 开始取当前 expert 的 block，取走后 `block_idx += kNumSMs`
// 实现 SM 之间的 round-robin（类似 persistent GEMM 的经典技巧）。
//
// Per-expert token 数缓存：
//   `fetch_expert_recv_count` 一次性把"本 rank 上 kNumExpertsPerRank 个 expert 的 token 数"
//   缓存到 warp 寄存器（每 lane 负责 expert_idx % 32 == lane_idx 的 expert）。
//   `get_num_tokens(e)` 利用 warp shuffle 把 lane e%32 的值广播给全 warp，避免反复访存。
// =====================================================================================

namespace deep_gemm::sched {

// 当前 block 所处阶段：None 表示整个 kernel 的 block 已派完
enum class BlockPhase {
    None = 0,
    Linear1 = 1,
    Linear2 = 2
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumExpertsPerWave,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
          uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K,
          uint32_t kNumL2BlockKs = L2_SHAPE_K / BLOCK_K>
struct MegaMoEScheduler {
    DG_STATIC_ASSERT(L1_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(kNumExpertsPerRank % kNumExpertsPerWave == 0, "Invalid wave config");

    // 2-CTA cluster 不变量：相邻 CTA 必须落在同一个 m_block_idx、n_block_idx 只相差 1。
    // 故 SM 数 / N 方向 block 数都必须是偶数，否则 leader/ follower CTA 的配对会错位。
    DG_STATIC_ASSERT(kNumSMs % 2 == 0, "Number of SMs must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kNumL1BlockNs % 2 == 0, "L1 N block count must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kNumL2BlockNs % 2 == 0, "L2 N block count must be even for 2-CTA cluster");

    // 只读 workspace 引用：用于读取 expert recv count 等跨 rank 计数
    const layout::Workspace& workspace;

    // 下一次 get_next_block 的相位：初始从 Linear1 开始
    BlockPhase next_phase = BlockPhase::Linear1;

    // 当前游标（随 for_each_block 推进）：
    //   current_local_expert_idx  —— 当前正在派发的 local expert
    //   current_num_tokens        —— 该 expert 实际收到的 token 数
    //   current_pool_block_offset —— 该 expert 在 pool 中的起始 block（<=current_expert 的 block 累计和）
    //   block_idx                 —— 剩余线性 block 计数（每次派发 += kNumSMs 实现 SM 间 round-robin）
    //   m_block_idx / n_block_idx —— 本次派出的 block 坐标
    uint32_t current_local_expert_idx = 0;
    uint32_t current_num_tokens = 0;
    uint32_t current_pool_block_offset = 0;
    uint32_t block_idx = 0;
    uint32_t m_block_idx = 0;
    uint32_t n_block_idx = 0;

    // Per-warp 缓存：每 lane 保存 expert (i*32+lane_idx) 的 token 数
    // 通过 warp shuffle 提供 O(kNumExpertsPerLane) 查表（通常为 1）
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};

    CUTLASS_DEVICE explicit MegaMoEScheduler(const layout::Workspace& workspace): workspace(workspace) {
        // 初始 block_idx = 本 SM 编号；每派一次 block 就累加 kNumSMs，天然 round-robin
        block_idx = blockIdx.x;
    }

    // 当前 wave 的 expert 结束边界（向上对齐到 kNumExpertsPerWave）
    CUTLASS_DEVICE uint32_t get_wave_expert_end_idx() const {
        return math::align(current_local_expert_idx + 1, kNumExpertsPerWave);
    }

    // 从 warp-distributed 缓存里拿 expert_idx 的 token 数
    //   - 先在本 lane 内线性匹配（k=kNumExpertsPerLane，通常 1）
    //   - 再用 __shfl_sync(exchange) 把拥有该值的 lane 的结果广播给整 warp
    CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
        uint32_t valid_value;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_num_tokens_per_expert[i] : valid_value;
        }
        return ptx::exchange(valid_value, expert_idx % 32);
    }

    // 计算 expert_idx 在 pool 中的起始 block 偏移
    //   = Σ (ceil_div(tokens[e], BLOCK_M)) for e < expert_idx
    // 利用 per-lane 部分和 + warp reduce 求和
    CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) {
        uint32_t num_blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
        }
        return __reduce_add_sync(0xffffffff, num_blocks);
    }

    CUTLASS_DEVICE void advance_expert_idx() {
        current_pool_block_offset += get_current_num_m_blocks();
        current_local_expert_idx += 1;
        current_num_tokens = get_num_tokens(current_local_expert_idx);
    }

    CUTLASS_DEVICE void set_expert_idx(const uint32_t& expert_idx) {
        current_local_expert_idx = expert_idx;
        current_num_tokens = get_num_tokens(expert_idx);
        current_pool_block_offset = get_pool_block_offset(expert_idx);
    }

    CUTLASS_DEVICE uint32_t get_current_pool_block_offset() const {
        return current_pool_block_offset;
    }

    CUTLASS_DEVICE uint32_t get_current_num_m_blocks() const {
        return math::ceil_div(current_num_tokens, BLOCK_M);
    }

    template <bool kDoUMMAAligned = false>
    CUTLASS_DEVICE uint32_t get_valid_m() const {
        const auto m = cute::min(current_num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
        return kDoUMMAAligned ? math::align(m, 16u) : m;
    }

    // -----------------------------------------------------------------
    // fetch_next_l1_block / fetch_next_l2_block
    //   在当前 wave 内搜索下一个落在本 SM 上的合法 (m_block, n_block)：
    //     1. 计算当前 expert 的 m-block 总数；
    //     2. 若 block_idx 仍指向当前 expert 的范围内，命中；
    //     3. 否则减去当前 expert 消耗的 block 数，expert 游标 +1，继续搜。
    //   block_idx 的表示：线性展开成 (m_block * kNumL1BlockNs + n_block)，
    //   故 m_block_idx = block_idx / kNumL1BlockNs（L2 同理）。
    // -----------------------------------------------------------------
    CUTLASS_DEVICE bool fetch_next_l1_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            m_block_idx = block_idx / kNumL1BlockNs;
            if (m_block_idx < num_m_blocks)
                return true;

            // 当前 expert 的所有 block 已分完：扣除消耗，expert 游标推进
            block_idx -= num_m_blocks * kNumL1BlockNs;
            advance_expert_idx();
        }
        return false;
    }

    CUTLASS_DEVICE bool fetch_next_l2_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            if (block_idx < num_m_blocks * kNumL2BlockNs) {
                m_block_idx = block_idx / kNumL2BlockNs;
                return true;
            }

            block_idx -= num_m_blocks * kNumL2BlockNs;
            advance_expert_idx();
        }
        return false;
    }

    // -----------------------------------------------------------------
    // get_next_block —— 核心状态机
    //   返回 (phase, expert, m_block, n_block)；若已派完返回 phase=None。
    //
    // 阶段切换规则：
    //   Linear1 完成 → Linear2：把 expert 游标回退到当前 wave 的起点（align_down）
    //                          以便"同一 wave 的相同 expert 再次以 L2 模式遍历"。
    //   Linear2 完成 → Linear1：expert 游标已在 wave 末尾，自然进入下一 wave 的 L1。
    // -----------------------------------------------------------------
    CUTLASS_DEVICE cute::tuple<BlockPhase, uint32_t, uint32_t, uint32_t> get_next_block() {
        while (true) {
            if (current_local_expert_idx >= kNumExpertsPerRank)
                break;

            if (next_phase == BlockPhase::Linear1) {
                if (fetch_next_l1_block()) {
                    n_block_idx = block_idx - m_block_idx * kNumL1BlockNs;
                    // block_idx += kNumSMs —— SM 间 round-robin：下次从"再过 kNumSMs 个 block"开始找
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear1, current_local_expert_idx, m_block_idx, n_block_idx};
                } else {
                    // wave 内 L1 全部派完 → 切到 L2；回退 expert 到 wave 起点再跑一遍
                    next_phase = BlockPhase::Linear2;
                    set_expert_idx(math::align<uint32_t, false>(current_local_expert_idx - 1, kNumExpertsPerWave));
                }
            } else {
                if (fetch_next_l2_block()) {
                    n_block_idx = block_idx - m_block_idx * kNumL2BlockNs;
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear2, current_local_expert_idx, m_block_idx, n_block_idx};
                } else {
                    // wave 内 L2 派完：进入下一 wave 的 L1（expert 游标已由 advance_expert_idx 跨过 wave 边界）
                    next_phase = BlockPhase::Linear1;
                }
            }
        }
        return {BlockPhase::None, 0, 0, 0};
    }

    // -----------------------------------------------------------------
    // fetch_expert_recv_count —— 等待并缓存每个 local expert 的 token 数
    // workspace.get_expert_recv_count_sum_ptr 的 64-bit 值布局：
    //     低 32 位 = 累计 token 数， 高 32 位 = 已完成的 (rank × SM) 数
    // 当高 32 位 == kNumSMs * kNumRanks 时说明"全部 rank 的全部 dispatch SM 都完成了 count 阶段"
    // —— 此时 token 数才算最终确定，可以开始 GEMM 调度。
    // -----------------------------------------------------------------
    CUTLASS_DEVICE void fetch_expert_recv_count() {
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                do {
                    value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            }
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
        __syncwarp();
    }

    // -----------------------------------------------------------------
    // for_each_block —— 驱动整个 persistent schedule 的入口
    //
    // 对每次派发给本 SM 的 block，调用 func(phase, expert, num_k_blocks, m_block_idx, n_block_idx)。
    // `num_k_blocks` 依 phase 切换（L1 用 L1_SHAPE_K / BLOCK_K，L2 用 L2_SHAPE_K / BLOCK_K）。
    //
    // 调用者（TMA warp / MMA warp / Epilogue warp）共用同一套调度 —— 因此同一个 iter_idx
    // 对应同一个 block 的各阶段 warp，互相通过 barrier index 对齐生产/消费关系。
    // -----------------------------------------------------------------
    template <typename Func>
    CUTLASS_DEVICE void for_each_block(Func&& func) {
        // 等待所有 rank 的 dispatch count 完成，这样 token 数就稳定了
        fetch_expert_recv_count();

        // 从 expert 0 开始（为 current_num_tokens / pool_block_offset 置初值）
        set_expert_idx(0);

        // TODO: add swizzle within expert waves for better L2 cache utilization
        while (true) {
            CUTE_TIE_DECL(get_next_block(), block_phase, current_local_expert_idx, m_block_idx, n_block_idx);
            if (block_phase == BlockPhase::None)
                break;

            func(block_phase, current_local_expert_idx,
                 block_phase == BlockPhase::Linear2 ? kNumL2BlockKs : kNumL1BlockKs,
                 m_block_idx, n_block_idx);
        }
    }
};

} // namespace deep_gemm::sched
