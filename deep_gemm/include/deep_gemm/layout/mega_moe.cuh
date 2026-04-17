#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>

// =====================================================================================
// 数据结构层 (L2)：Mega MoE 的全局内存布局 / 元数据 / 对称缓冲区封装
// -------------------------------------------------------------------------------------
// 本文件定义 Mega MoE kernel 在 symmetric memory（各 rank 等大小、可互相 NVLink 访问）
// 中占用的工作区布局，以及 token 数据缓冲区 / scale-factor buffer 的抽象。
//
// 两个关键概念：
//   1. Pool（共享 token 池）：
//      所有 local expert 的 token 被连续摆放在同一 pool 中，按 BLOCK_M 对齐分段。
//      这是 Mega MoE "output-stationary" 策略的基础 —— 同一个 pool block 的 token
//      在整个 kernel 生命周期里停留在同一 SM 上，从 L1 GEMM 到 L2 GEMM 到 Combine 写回
//      都不必跨 block 迁移。
//
//   2. Workspace（同步/索引表 / 31 metadata 区）：
//      从 Workspace.base 起依次排布：
//        [ 0, 32)            barrier 信号区（grid sync 计数 + NVLink barrier 槽位）
//        [ 32,  32+8E )      每 expert 的 send count（用于对端写回 recv count）
//        [ ...,  +8E       ) 每 expert 的 recv count（per-rank 细分）
//        [ ...,  +8*(E/R) )  每 local expert 的 recv count sum（per-rank 汇总）
//        [ ...,  +4*B*⌈...⌉) L1 arrival count（per pool-block，dispatch pull 计数）
//        [ ...,  +8*B      ) L2 arrival mask（per pool-block，epilogue 完成位图）
//        [ ..., + src_tok )  dispatch pulling 用的 token-topk 源索引
//        [ ..., + combine )  combine 写回需要的 (rank, token, topk) 元数据
//      其中 E = num_experts，R = num_ranks，B = num_max_pool_blocks。
//
// Data / Buffer 则是 **通用**的 per-token / per-rank 缓冲区描述符，
// 不仅 Mega MoE 使用，也服务于其它 dispatch/combine 场景。
// =====================================================================================

namespace deep_gemm::layout {

// ----------------------------------------------------------------------------
// Pool 容量估算
//   上界 = 所有 rank 全部 token * min(topk, experts_per_rank) + 每 expert BLOCK_M-1 的对齐余数
// 说明：
//   - 第一项是最坏情况下每个 token 被 `min(topk, experts_per_rank)` 个本地 expert 命中
//     所带来的 token 实体数（每被命中一次就占一个 pool 槽）。
//   - 第二项是"每个 expert 的末尾 block 最多浪费 BLOCK_M-1 行"的对齐 padding，
//     最外层再按 block_m 向上对齐，保证 pool 能按 BLOCK_M 切成整齐 pool blocks。
// ----------------------------------------------------------------------------
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_pool_tokens(T num_ranks, T num_max_tokens_per_rank, T num_topk,
                                                        T num_experts_per_rank, T block_m) {
    const auto num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
    const auto num_max_experts_per_token = math::constexpr_min(num_topk, num_experts_per_rank);
    return math::constexpr_align(
        num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (block_m - 1),
        block_m);
}

// ----------------------------------------------------------------------------
// SF Pool 容量
//   SF（scale factor）需要按 UTCCP 128 元素对齐存放 —— 每个 pool block 的 SF 区
//   要扩展为 align(block_m, 128) 行，才能匹配 `tcgen05 UTCCP` 指令要求的布局。
// ----------------------------------------------------------------------------
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_padded_sf_pool_tokens(T num_max_pool_tokens, T block_m) {
    return (num_max_pool_tokens / block_m) * math::constexpr_align(block_m, static_cast<T>(128));
}

// ----------------------------------------------------------------------------
// Combine 阶段需要的逐 token 反向索引
//   记录一个 pool 槽位里的 token 原本来自哪个 rank / 哪个 token idx / 哪条 topk,
//   用于 Linear2 的 BF16 写回时把结果推回"源 rank + 源 topk 槽"。
// ----------------------------------------------------------------------------
struct TokenSrcMetadata {
    uint32_t rank_idx;   // 源 rank
    uint32_t token_idx;  // 源 rank 内该 token 的索引
    uint32_t topk_idx;   // 该 token 被当前 rank 命中时对应的 topk 序号
};

// ============================================================================
// Workspace —— Mega MoE kernel 用于跨 rank 协作的元数据区描述符
// ----------------------------------------------------------------------------
// 该结构体本身不持有内存，仅封装"从 `base` 开始按固定顺序切片"的访问方式。
// 各子区段的 byte offset 由一串 `get_*_ptr` 的累加式计算给出（见下方实现），
// 使得 host 端和 device 端共享同一套布局约定。
//
// 层次化地看：
//   Barrier 信号区（32B）          —— grid_sync 4 个计数 + NVLink barrier 状态/信号
//   Expert 计数区（~E*16B）        —— dispatch 阶段用于"各 rank 发多少 / 收多少 token"
//   Pool block arrival 同步表      —— L1 / L2 两级 block-wise 的 per-block arrival
//   Dispatch 源索引                —— token 从哪条 (rank, slot) 拉取过来
//   Combine 源元数据               —— token 要回写到哪条 (rank, token, topk)
// ============================================================================
struct Workspace {
    void* base;

    // 基础拓扑
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_max_recv_tokens_per_expert;

    // Pool: 所有 local expert 共享一个连续 token pool —— output-stationary 策略的载体
    uint32_t num_max_pool_tokens;
    uint32_t num_max_pool_blocks;

    // Barrier 信号区固定 32 字节：供 grid sync（4×4B）+ NVLink barrier 状态/信号共用
    static constexpr uint64_t kNumBarrierSignalBytes = 32;

    CUTLASS_HOST_DEVICE
    Workspace(void* base,
              const uint32_t& num_ranks,
              const uint32_t& num_experts,
              const uint32_t& num_max_tokens_per_rank,
              const uint32_t& num_topk,
              const uint32_t& block_m):
        base(base),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
        num_max_pool_tokens = get_num_max_pool_tokens(
            num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank, block_m);
        num_max_pool_blocks = num_max_pool_tokens / block_m;
        DG_UNIFIED_ASSERT(num_max_tokens_per_rank % block_m == 0);
    }

    // -----------------------------------------------------------------
    // 计算完整 Workspace 的字节数
    //   顺序累加 = Workspace 内存 layout 的官方定义。
    //   最后 16B 对齐是为了让紧随其后的 token buffer 满足 TMA descriptor 要求。
    // -----------------------------------------------------------------
    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;

        num_bytes += kNumBarrierSignalBytes;

        // Expert send/recv count —— 每 expert 一个 64-bit 状态槽（低 32 位累计 token 数，高 32 位累计 SM 数）
        num_bytes += num_experts * sizeof(uint64_t) * 2;

        // 每 local expert 的"所有 rank 总和"槽
        num_bytes += num_experts_per_rank * sizeof(uint64_t);

        // L1 arrival count: 每个 pool block 一个 uint32 计数 —— 对齐到偶数个以便后面 L2 mask 的 8B 对齐
        num_bytes += math::align(num_max_pool_blocks, 2u) * sizeof(uint32_t);

        // L2 block arrival mask: 每个 pool block 一个 uint64 位图，bit_i 表示 n_block_i 的 epilogue 完成
        num_bytes += num_max_pool_blocks * sizeof(uint64_t);

        // Dispatch pulling 源索引：[expert][rank][token_slot] -> src_token_topk_idx
        num_bytes += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * sizeof(int);

        // Combine 源元数据：每 pool 槽一份 (rank_idx, token_idx, topk_idx)
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);

        // 16B 对齐：匹配 TMA descriptor 要求
        num_bytes = math::align<uint64_t>(num_bytes, 16);
        return num_bytes;
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    // -----------------------------------------------------------------
    // Barrier 信号区（`kNumBarrierSignalBytes = 32` 字节）的 bit-level layout：
    //   [ 0 .. 16)  4 × uint32_t  grid sync 计数器（Dispatch / Epilogue 可各占一个）
    //   [16 .. 20)  1 × uint32_t  NVLink barrier 的 (phase, sign) 状态 counter
    //   [20 .. 28)  2 × int32_t   NVLink barrier 的两个交替 signal 槽（phase 0 / 1）
    //   [28 .. 32)  保留对齐
    // -----------------------------------------------------------------
    static constexpr uint32_t kNumMaxGridSyncCounters = 4;

    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return static_cast<uint32_t*>(base) + kIndex;
    }

    CUTLASS_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters;
    }

    CUTLASS_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        // NOTES: the signal is signed, as we may minus
        return math::advance_ptr<int>(base, (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) + phase * sizeof(int));
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return math::advance_ptr<uint64_t>(base, kNumBarrierSignalBytes) + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts) + rank_idx * num_experts_per_rank + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts * 2) + expert_idx;
    }

    // -----------------------------------------------------------------
    // L1 arrival count（per pool-block 的 token 到齐计数，uint32）
    //   Dispatch 阶段每 pull 完一个 token 就 +1，待其达到该 block 的 valid_m 时，
    //   GEMM warp 才允许读取该 pool block 的 token 数据。
    //   使用 atomic + acquire 语义保证 Linear1 TMA warp 能看到 token 数据写入。
    // -----------------------------------------------------------------
    CUTLASS_DEVICE
    uint32_t* get_l1_arrival_count_ptr(const uint32_t& pool_block_idx = 0) const {
        const auto base = get_expert_recv_count_sum_ptr(num_experts_per_rank);
        return reinterpret_cast<uint32_t*>(base) + pool_block_idx;
    }

    // -----------------------------------------------------------------
    // L2 arrival mask（per pool-block 的位图，uint64）
    //   Linear1 epilogue 写完每个 n_block_idx 的 intermediate 后 set 对应 bit；
    //   Linear2 TMA warp 根据"某个 k_block 需要的 bit 组合已置位"来判定可读。
    //   位数上限 = 64 足够覆盖 L2_SHAPE_K / BLOCK_K（见 `DG_STATIC_ASSERT(L2_SHAPE_K <= 64*...)`）。
    //   8B 对齐通过把上一块 L1 count 的条目数 pad 到偶数实现。
    // -----------------------------------------------------------------
    CUTLASS_DEVICE
    uint64_t* get_l2_arrival_mask_ptr(const uint32_t& pool_block_idx = 0) const {
        const auto base = get_l1_arrival_count_ptr(math::align(num_max_pool_blocks, 2u));
        return reinterpret_cast<uint64_t*>(base) + pool_block_idx;
    }

    // -----------------------------------------------------------------
    // Dispatch 拉取用的源 token-topk 索引表
    //   布局：[expert][rank][token_slot] -> uint32（编码 src_token_idx * topk + src_topk_idx）
    //   远端 rank 的 dispatch warp 在 count 阶段把 "token-topk" 写入对端对应槽位；
    //   本地 dispatch warp 的 pull 阶段按 round-robin 选择 rank 后读取该表得到源 token。
    // -----------------------------------------------------------------
    CUTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto base = get_l2_arrival_mask_ptr(num_max_pool_blocks);
        return reinterpret_cast<uint32_t*>(base) +
            expert_idx * (num_ranks * num_max_recv_tokens_per_expert) +
            rank_idx * num_max_recv_tokens_per_expert + token_idx;
    }

    // -----------------------------------------------------------------
    // Combine 源元数据：按 pool 槽位索引逐条记录 (rank, token, topk)
    //   Dispatch pull 时顺手写入；Combine epilogue 按 pool_token_idx 取出，
    //   定位到远端 rank 的 `combine_token_buffer[topk][token_idx]` 写回。
    // -----------------------------------------------------------------
    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }
};

// ============================================================================
// Data —— 对"一段已知字节数、以某个指针为基址"的轻量级描述
// ----------------------------------------------------------------------------
// 作为 `Buffer` 的"每条 item"配置：仅携带 size + alignment 要求 + base 指针。
// TMA descriptor 要求 16B 对齐，若确知本 item 不参与 TMA 可放宽（如 topk idx 表）。
// ============================================================================
struct Data {
    uint32_t num_bytes;
    bool require_tma_alignment;
    void* base;

    CUTLASS_HOST_DEVICE
    constexpr explicit Data(
        const uint32_t& num_bytes,
        const bool& require_tma_alignment = true,
        void* base = nullptr) :
        num_bytes(num_bytes), require_tma_alignment(require_tma_alignment), base(base) {
        DG_UNIFIED_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
    }

    template <typename dtype_t = uint32_t>
    CUTLASS_HOST_DEVICE constexpr dtype_t get_num_bytes() const {
        return static_cast<dtype_t>(num_bytes);
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE void set_base_ptr(void* ptr) {
        base = ptr;
    }
};

// ============================================================================
// Buffer —— 二维（rank × token）的统一缓冲区描述
// ----------------------------------------------------------------------------
// 抽象层次：
//   Buffer             ≈ [rank][token_idx] -> Data
//   .get_rank_buffer() ≈ [token_idx]       -> Data
//   .get_data_buffer() ≈ 具体一条 token 的起始地址
// 在 kernel 中广泛用于：
//   - input_token_buffer / input_sf_buffer （单 rank 自己输入）
//   - l1_token_buffer / l2_token_buffer     （pool 内 token / intermediate）
//   - combine_token_buffer                   （多 topk 槽写回）
// ============================================================================
struct Buffer {
    Data data_layout;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;

    void* base;

    CUTLASS_HOST_DEVICE
    Buffer(const Data& data_layout,
           const uint32_t& num_ranks,
           const uint32_t& max_num_tokens_per_rank,
           void* base = nullptr) :
        data_layout(data_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(max_num_tokens_per_rank),
        base(base) {}

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * data_layout.get_num_bytes<uint64_t>();
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    CUTLASS_HOST_DEVICE
    Buffer get_rank_buffer(const uint32_t& rank_idx) const {
        return {
            data_layout,
            1, num_max_tokens_per_rank,
            math::advance_ptr(base, get_num_bytes_per_rank() * rank_idx)
        };
    }

    CUTLASS_HOST_DEVICE
    Data get_data_buffer(const uint32_t& token_idx, const bool& global = false) const {
        DG_DEVICE_ASSERT(num_ranks == 1 or global);
        return Data(
            data_layout.num_bytes,
            data_layout.require_tma_alignment,
            math::advance_ptr(base, data_layout.get_num_bytes<uint64_t>() * token_idx)
        );
    }
};

} // namespace deep_gemm::layout
