#pragma once

// ============================================================================
// L6 启发式：Mega MoE 的参数选择策略
// ----------------------------------------------------------------------------
// 目标：在给定 shape（num_tokens, hidden, intermediate_hidden, num_experts, topk）
//       和硬件（num_sms, smem_capacity）下，选出一组能最大化硬件利用率的配置：
//         - block_m/n/k           GEMM 的 tile 形状
//         - store_block_m         epilogue 的 STSM 单次跨行数
//         - sf_block_m/n          UTCCP 128-aligned 的 SF 块
//         - num_max_pool_tokens   dispatch pool 容量（上界）
//         - num_experts_per_wave  每 wave 处理几个 expert（负载均衡 + 粒度）
//         - num_stages            pipeline 级数（受 smem 容量约束）
//         - num_*_threads         Dispatch / Non-Epilogue / Epilogue 线程数
//
// 决策顺序：
//   1) block_m 固定 192（后续可进一步动态）
//   2) block_n = block_k = 128（硬件友好，UMMA ShapeN 对齐）
//   3) 基于 block 推出 sf_block_* 与 pool 大小
//   4) 估算 num_experts_per_wave：以"所有 SM 都有活干"为目标
//   5) 固定 thread 布局（dispatch=128, non-epi=128, epi=256）
//   6) 按 smem_capacity 反推 num_stages（至少 2 级 pipeline）
// ============================================================================

#include <algorithm>
#include <unordered_set>

#include <deep_gemm/layout/mega_moe.cuh>

#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../../utils/system.hpp"
#include "sm100.hpp"

namespace deep_gemm {

// MegaMoEConfig：启发式产出的完整配置，全部会被塞成 device kernel 的模板参数
struct MegaMoEConfig {
    // GEMM block 分块：block_m × block_n / 2-CTA 广播 → 每 CTA 处理 load_block_m = block_m/2
    int block_m, block_n, block_k;
    int load_block_m, load_block_n;
    // Epilogue STSM 一次覆盖的行数（= ATOM_M × kNumAtomsPerStore）
    int store_block_m;

    // UTCCP 对齐的 SF block 尺寸：硬件要求一次 copy 的 SF 是 128 元素对齐
    int sf_block_m, sf_block_n;

    // Dispatch pool 容量 + SF 对齐后的 token 数（layout 决定的最大上界）
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;

    // TMA descriptor 的 swizzle mode（按字节：32/64/128）
    int swizzle_acts_mode, swizzle_weights_mode;

    // 每 wave 处理的 expert 数量 —— 平衡负载与 TMA 开销
    int num_experts_per_wave;

    // pipeline 级数 + 最终动态 shared memory 大小
    int num_stages, smem_size;

    // 三类 warp 的线程数（Dispatch / Non-Epilogue / Epilogue）
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    friend std::ostream& operator << (std::ostream& os, const MegaMoEConfig& config) {
        os << "MegaMoEConfig("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", load_block_m=" << config.load_block_m << ", load_block_n=" << config.load_block_n
           << ", store_block_m=" << config.store_block_m
           << ", sf_block_m=" << config.sf_block_m << ", sf_block_n=" << config.sf_block_n
           << ", num_max_pool_tokens=" << config.num_max_pool_tokens
           << ", num_padded_sf_pool_tokens=" << config.num_padded_sf_pool_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_experts_per_wave=" << config.num_experts_per_wave
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads << ")";
        return os;
    }
};

// block_m 选择：目前固定 192（= 3 × 64 warp-level tile），经验上在 H2/B200 上最优
//   TODO: 根据 num_tokens/num_topk/num_experts 做动态调整
static int get_block_m_for_mega_moe(const int& num_ranks, const int& num_experts,
                                    const int& num_max_tokens_per_rank, const int& num_topk) {
    return 192;
}

// num_experts_per_wave 计算：每 wave 处理几个 expert
//   核心约束：一个 wave 内 SM 不能闲置，且 expert 数要能整除 num_experts_per_rank
//
//   规划：
//     1) 估算每 expert 的 L1 block 数 = ceil(expected_tokens / block_m) × (N / block_n)
//     2) 让全 wave 的 block 总数 ≥ kImbalanceFactor × num_sms（留余量应对 expert 负载不均）
//     3) clamp 到 [1, min(32, num_experts_per_rank)] 且必须整除 num_experts_per_rank
static int get_num_experts_per_wave_for_mega_moe(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n, const int& num_sms) {
    // 不均衡裕量：真实路由下 expert 的 token 分布不均，实际能 schedule 的 block 数要打对折
    constexpr int kImbalanceFactor = 2;

    // TODO: support num_experts_per_rank > 32
    // 找到 num_experts_per_rank 在 [1, 32] 内最大的因子作为上界
    //   → 保证最终选择能整除 num_experts_per_rank（每 wave 处理数相同）
    int max_num_experts_per_wave = std::min(32, num_experts_per_rank);
    while (max_num_experts_per_wave > 1 and num_experts_per_rank % max_num_experts_per_wave != 0)
        -- max_num_experts_per_wave;

    // 期望 expert 的 token 数（均匀假设）
    const int expected_tokens_per_expert =
        num_tokens * num_topk / num_experts_per_rank + 1;
    const int num_m_blocks = ceil_div(expected_tokens_per_expert, block_m);
    const int num_n_blocks = intermediate_hidden / block_n;
    const int num_l1_blocks_per_expert = num_m_blocks * num_n_blocks;

    // 让"全 wave 的 block 数 ≥ 2 × SM 数"，才能在负载不均时喂饱所有 SM
    int num_experts_per_wave = num_l1_blocks_per_expert > 0
        ? ceil_div(kImbalanceFactor * num_sms, num_l1_blocks_per_expert) : 1;
    num_experts_per_wave = std::min(num_experts_per_wave, max_num_experts_per_wave);

    // 向上取整到 num_experts_per_rank 的最小因子 → 确保每 wave 处理等量 expert
    while (num_experts_per_wave < max_num_experts_per_wave and num_experts_per_rank % num_experts_per_wave != 0)
        ++ num_experts_per_wave;

    return num_experts_per_wave;
}

// pipeline 配置：按 shared memory 容量反推最大 num_stages
//   smem layout（与 kernel 一致）：
//     ┌──────────── Fixed（不随 stage 数增长）────────────┐
//     │ dispatch region：expert count + send buffers     │
//     │ C/D 输出 region：L1 FP8 (2 stage) 或 L2 BF16     │
//     │ amax reduction 缓冲 + tmem ptr                   │
//     │ barriers（mbarrier 8B each）                     │
//     └──────────────────────────────────────────────────┘
//     ┌──────────── Per-stage × num_stages ───────────┐
//     │ A tile + B tile + SFA + SFB + full/empty bars │
//     └───────────────────────────────────────────────┘
//   返回 (num_stages, 实际 smem 总大小)
static std::pair<int, int> get_pipeline_config_for_mega_moe(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k, const int& store_block_m,
    const int& sf_block_m, const int& sf_block_n,
    const int& num_dispatch_warps, const int& num_epilogue_warps) {
    constexpr int kSmemAlignment = 1024;      // SM100 smem bank 对齐
    constexpr int kNumEpilogueStages = 2;     // epilogue 累加器双缓冲
    constexpr int kNumTMAStoreStages = 2;     // L1 epilogue 的 TMA store 双缓冲

    // 2-CTA multicast：A tile 在两个 CTA 之间切半
    const int load_block_m = block_m / 2;

    // Dispatch 区：expert 计数 + 每 warp 一个 send buffer（hidden bytes）
    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(hidden), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    // C/D 输出区：Linear1 FP8 (block_n/2, 2 TMA stage) vs Linear2 BF16 (block_n, 1 stage)
    //   取 max —— 两者共用同一块 smem（phase 切换时不会并存）
    const auto num_epilogue_warpgroups = num_epilogue_warps / 4;
    const int smem_cd_l1 = num_epilogue_warpgroups * store_block_m * (block_n / 2) * kNumTMAStoreStages;
    const int smem_cd_l2 = num_epilogue_warpgroups * store_block_m * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd = std::max(smem_cd_l1, smem_cd_l2);

    // Barriers（不随 stage 变）：
    //   dispatch barriers × dispatch warps
    // + tensor memory full/empty × 2 stage × 2（full+empty）
    // + combine barriers × epi warps × 2（每 warp 两个 load stage）
    //   每个 mbarrier 8B
    const int smem_barriers = (num_dispatch_warps + kNumEpilogueStages * 2 + num_epilogue_warps * 2) * 8;

    // amax 跨 warp reduce 的共享缓冲（每 warp store_block_m/2 个 float2 → 取 store_block_m×float）
    const int smem_amax_reduction = store_block_m * num_epilogue_warps * static_cast<int>(sizeof(float));

    // TMEM allocator 返回的 base 指针（4B）
    const int smem_tmem_ptr = 4;

    // SF per-stage 大小：UTCCP 对齐到 128 元素，每元素 4 byte → sf_block × 4
    const int smem_sfa_per_stage = sf_block_m * 4;
    const int smem_sfb_per_stage = sf_block_n * 4;

    // 单 stage 消耗：A_tile + B_tile + SFA + SFB + (full_bar + empty_bar) = 2×8B
    const int smem_per_stage = load_block_m * block_k + block_n * block_k + smem_sfa_per_stage + smem_sfb_per_stage + 2 * 8;

    // Fixed 部分总和
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_amax_reduction + smem_barriers + smem_tmem_ptr;

    // 最大 stage 数 = (容量 - Fixed) / 每 stage 消耗；要求至少 2 级才能开 pipeline
    const int num_stages = (smem_capacity - smem_fixed) / smem_per_stage;
    DG_HOST_ASSERT(num_stages >= 2);

    return {num_stages, smem_fixed + num_stages * smem_per_stage};
}

// 主入口：一次产出完整 MegaMoEConfig
//   ── 所有 heuristics 在此组合，返回值会成为 kernel 的模板参数
static MegaMoEConfig get_mega_moe_config(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden) {
    // ------- (1) Block tiling -------
    const int block_m = get_block_m_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_topk);
    const int block_n = 128;                         // UMMA-N 对齐，128 是 SM100 上的甜点
    const int block_k = 128;                         // UMMA-K 对齐（FP8/FP4 都是 128）
    const int load_block_m = block_m / 2;            // 2-CTA multicast A
    const int load_block_n = block_n;                // B tile 直接等同 block_n
    const int store_block_m = 32;                    // epilogue STSM 粒度（ATOM_M=8 × 4 atoms）
    const auto [sf_block_m, sf_block_n] = SM100ArchSpec::get_sf_uttcp_aligned_block_sizes(block_m, block_n, MmaKind::MXFP8FP4);
    // Pool 容量：基于最坏情况（每 rank 全部 token 都落到当前 expert）+ block_m 对齐 padding
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank, block_m);
    // SF 对齐的 token 数：SF tile 的 row/col 都要对齐到 UTCCP 128-element
    const int num_padded_sf_pool_tokens = layout::get_num_padded_sf_pool_tokens(num_max_pool_tokens, block_m);
    // NOTES: FP8 act 与 FP4 weight（smem 里 unpack 成 8-bit）都用 128B swizzle
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;

    // ------- (2) Wave 决策 -------
    const int num_sms = device_runtime->get_num_sms();
    const int num_experts_per_wave = get_num_experts_per_wave_for_mega_moe(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms);

    // ------- (3) 固定的 warp 布局（与 kernel 内预期保持一致） -------
    const int num_dispatch_threads = 128;         // 4 warps → Dispatch
    const int num_non_epilogue_threads = 128;     // 4 warps → TMA-A / TMA-B / MMA / Free
    const int num_epilogue_threads = 256;         // 8 warps → Epilogue + Combine

    // ------- (4) pipeline stage 数 & smem 总量 -------
    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe(
        SM100ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k, store_block_m,
        sf_block_m, sf_block_n,
        num_dispatch_threads / 32, num_epilogue_threads / 32);

    const auto config = MegaMoEConfig {
        block_m, block_n, block_k,
        load_block_m, load_block_n, store_block_m,
        sf_block_m, sf_block_n,
        num_max_pool_tokens, num_padded_sf_pool_tokens,
        swizzle_acts_mode, swizzle_weights_mode,
        num_experts_per_wave,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads
    };

    // 首次 print：便于调试时观察启发式选出的具体参数（同 shape 只 print 一次）
    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format(
            "MegaMoEConfig(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
