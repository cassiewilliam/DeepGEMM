#pragma once

#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

// =====================================================================================
// 基础通信层 (L1)：Mega MoE kernel 的最底层同步原语
// -------------------------------------------------------------------------------------
// 本文件提供两类全局同步能力：
//   1. `grid_sync`    —— 单 rank 内 Grid 级同步（所有 SM 对齐到同一个"逻辑屏障点"），
//                       灵感来自 `cooperative_groups::this_grid().sync()`，但避免
//                       cooperative launch 的运行时开销。
//   2. `nvlink_barrier` —— 多 rank 跨 GPU 同步，基于 NVLink symmetric memory 的
//                          "±1 交替信号 + grid sync 前后夹心"模式。
//
// 设计关键点：
//   * 所有同步计数器都落在 Workspace 预分配的 32 字节区间，详见 `layout/mega_moe.cuh`。
//   * Grid sync 仅 SM0 的 thread 0 承担"合帐"职责（对其它 SM 加 1，自己加一个补偿值），
//     用 `kFinishSumTag = 0x80000000` 的 bit 31 翻转作为"本轮结束"信号，避免 ABA。
//   * NVLink barrier 采用 2 位 state：phase ∈ {0, 1} 选择交替 buffer；sign ∈ {0, 1}
//     决定 ±1 的方向。这样相邻两次 barrier 可分别"+1 到 N"、"-1 到 0"，
//     无需清零即可复用同一 signal 槽位。
// =====================================================================================

namespace deep_gemm::comm {

    // -------------------------------------------------------------------------------------
    // grid_sync —— Rank 内所有 SM 的 Grid 级屏障（每调用一次即一次屏障）
    // -------------------------------------------------------------------------------------
    // 算法概述（"bit 31 翻转"技巧，避免 ABA 问题并天然支持重入）：
    //   * 初始值：0
    //   * 第 i 次 barrier 中：
    //       - 非 SM0 的 thread 0 贡献 +1
    //       - SM0 的 thread 0 贡献 (kFinishSumTag - (kNumSMs - 1))
    //     累加后计数器恰好增加 kFinishSumTag = 1<<31，即 bit 31 翻转一次。
    //   * 每个 SM 用自己 atomic 返回的 old_value 与后续 ld_acquire 的 new_value 比较，
    //     只要 bit 31 变化就说明"所有 SM 都已 arrive"，无需清零即可进入下一轮。
    //   * 由于每轮只翻转一次 bit 31，不同轮次的 old/new 组合绝不会误判（抗 ABA）。
    //
    // 参数：
    //   kGridSyncIndex  —— Workspace 中 4 个 grid sync 计数器之一（允许 dispatch 与 epilogue
    //                      各占一个，避免语义交叉）
    //   sync_scope      —— 调用方提供的 SM 内对齐函数（例如 `bar.sync` 或整 warpgroup 同步），
    //                      用于"进入 barrier 之前/之后"把本 SM 内所有线程对齐到 thread 0。
    // -------------------------------------------------------------------------------------
    template <uint32_t kNumSMs, uint32_t kGridSyncIndex = 0, typename sync_scope_t>
    CUTLASS_DEVICE void grid_sync(const layout::Workspace &workspace,
                                  const uint32_t &sm_idx, const uint32_t &thread_idx,
                                  const sync_scope_t &sync_scope)
    {
        // NOTES: the implementation idea is from `cooperative_groups::this_grid().sync()`
        static constexpr uint32_t kFinishSumTag = 0x80000000u;

        // 进入 barrier 前：SM 内所有线程先对齐到 thread 0，保证之前的 shared / global 写可见
        sync_scope();
        if (thread_idx == 0)
        {
            const auto count_ptr = workspace.get_grid_sync_count_ptr<kGridSyncIndex>();
            // 关键一步：SM0 加 (kFinishSumTag - (kNumSMs-1))，其余 SM 加 1
            // 全员到齐后计数器正好增加 kFinishSumTag —— 即"bit 31 翻转一次"作为本轮完成标记
            const auto old_value = ptx::atomic_add_rel(
                count_ptr, sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1)) : 1);
            uint32_t new_value;
            // 轮询：只要本线程 atomic 后的值与当前 load 值的 bit 31 不同，就说明屏障翻转了
            do
            {
                new_value = ptx::ld_acq(count_ptr);
            } while (((new_value ^ old_value) & kFinishSumTag) == 0);
        }
        // 离开 barrier：thread 0 已经探测到翻转，再把整个 SM 对齐一次
        sync_scope();
}

// -------------------------------------------------------------------------------------
// nvlink_barrier —— 跨 rank（跨 GPU）NVLink 屏障，结构为
//                   [Grid sync] → [Cross-rank ±1 signal] → [Grid sync]
// -------------------------------------------------------------------------------------
// 状态机（counter 的低 2 bit 即 `status`）：
//   +------+----------+-----------+------------------------------------+
//   | phase | sign    | 动作      | target                             |
//   +-------+---------+-----------+------------------------------------+
//   |   0   |   0     | 向 phase0 |  signal += +1；等待 signal == N    |
//   |       |         | 槽位 +1   |                                    |
//   |   1   |   0     | 向 phase1 |  signal += +1；等待 signal == N    |
//   |       |         | 槽位 +1   |                                    |
//   |   0   |   1     | 向 phase0 |  signal += -1；等待 signal == 0    |
//   |       |         | 槽位 -1   |  （把上一轮的 +N 抹掉，无需清零）  |
//   |   1   |   1     | 向 phase1 |  signal += -1；等待 signal == 0    |
//   +-------+---------+-----------+------------------------------------+
// 每次调用结束 counter+=1，按 4 周期循环遍历上面 4 个状态。
// 这样 2 个 signal 槽位轮换使用，+1 / -1 方向交替，**永远不需要显式清零**，
// 天然抗 ABA：因为任何相邻两次 barrier 都落在不同 (phase, sign) 组合。
//
// 线程/SM 分工：
//   * 只有 SM0 参与跨 rank 通信（其它 SM 仅在 prologue/epilogue 的 grid_sync 中出现）。
//   * SM0 内 thread i (i < kNumRanks) 负责向 rank i 的 signal 槽位发 ±1。
//   * thread 0 额外承担状态 counter 的累加与轮询 signal 到达。
//
// 超时保护：
//   * 以 2 GHz 保守估算，30 秒对应 6e10 cycles；超时直接 assert+printf 退出，
//     便于定位跨 rank 挂死（NVLink 拓扑错误、对端未到等）。
// -------------------------------------------------------------------------------------
template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kNumThreads, uint32_t kGridSyncIndex, uint32_t kTag, typename sync_scope_t>
CUTLASS_DEVICE void nvlink_barrier(const layout::Workspace& workspace,
                                   const layout::SymBuffer<kNumRanks>& sym_buffer,
                                   const uint32_t& sm_idx, const uint32_t& thread_idx,
                                   const sync_scope_t& sync_scope,
                                   const bool& sync_prologue = true,
                                   const bool& sync_epilogue = true) {
    // 必须保证 SM0 内线程数 ≥ rank 数，因为每个 rank 对应一个线程发信号
    DG_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");

    // Prologue grid sync：所有 SM 的"本地工作"都已写回 + 对远端可见后，再让 SM0 发信号
    // 可选关闭（`sync_prologue=false`）的场景：调用方保证前置已有 grid sync
    if (sync_prologue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);

    // 跨 rank 信号：仅 SM 0 参与
    if (sm_idx == 0) {
        auto* counter_ptr = workspace.get_nvl_barrier_counter_ptr();
        // 取 counter 的低 2 bit 作为本轮 (phase, sign) 状态
        const auto status = (*counter_ptr) & 3;
        const auto signal_phase = status & 1, signal_sign = status >> 1;
        auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);

        // 向每个远端 rank 的本轮 signal 槽位做 system-scope red.add（NVLink 原子）
        // sign=0 → +1；sign=1 → -1（抵消上一轮累加的 +N）
        if (thread_idx < kNumRanks)
            ptx::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);
        sync_scope();

        // thread 0：推进本地 counter；轮询 signal 到达目标值；带 30s 超时兜底
        constexpr int64_t kNumTimeoutCycles = 30ll * 2000000000ll;
        if (thread_idx == 0) {
            ptx::red_add(counter_ptr, 1);
            const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
            const auto start_clock = clock64();
            while (ptx::ld_acq_sys(signal_ptr) != target) {
                if (clock64() - start_clock >= kNumTimeoutCycles) {
                    // 挂死诊断输出：kTag 区分调用点（dispatch / combine / cleanup）
                    printf("DeepGEMM NVLink barrier timeout (30s): rank=%d, counter=%d, signal=%d, target=%d, phase=%d, sign=%d, tag=%d\n",
                           sym_buffer.rank_idx, *counter_ptr, ptx::ld_acq_sys(signal_ptr), target, signal_phase, signal_sign, kTag);
                    DG_DEVICE_ASSERT(false and "NVLink barrier timeout");
                }
            }
        }
    }

    // Epilogue grid sync：让所有 SM 都"感知到"跨 rank 屏障已过
    // 可选关闭（`sync_epilogue=false`）的场景：调用方后续紧跟另一 grid sync，无需重复
    if (sync_epilogue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);
}

} // namespace deep_gemm::comm
