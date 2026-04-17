#pragma once

// ============================================================================
// L5 宿主启动层：Mega MoE Kernel 的 JIT 编译 + launch 入口
// ----------------------------------------------------------------------------
// 职责：
//   1) 封装 device kernel sm100_fp8_fp4_mega_moe_impl<...> 的编译（generate_impl）
//      与启动（launch_impl）
//   2) 面向 PyTorch 的 python binding（sm100_fp8_fp4_mega_moe 自由函数）：
//        - 从 torch.Tensor 提取原始 device 指针
//        - 从 sym_buffer_ptrs 构造 cross-rank SymBuffer（NVLink 对称地址）
//        - 根据 heuristics 决定 block/stage/线程数配置
//        - 调用 make_tma_2d_desc / make_tma_sf_desc 构造 9 个 TensorMap
//        - 组装 Args → JIT 实例化模板 → 启动
//
// Kernel 启动参数一览：
//   Template：    形状/stage 数/线程数等全部编译期常量
//   Runtime：     y（输出）、num_tokens、SymBuffer、9 个 TensorMap
//   Launch Args： grid = num_sms，block = dispatch + non_epi + epi，动态 smem
//
// NOTES: L1 output 与 L2 activations 共享一块 tensor（intermediate FP8 buffer）：
//   Linear1 epilogue 写入 post-SwiGLU FP8 结果，Linear2 的 TMA-A 从同一块拉取。
//   因此需要两份 TensorMap：
//     - tensor_map_l1_output：MN 方向宽度为 intermediate_hidden（= N/2）
//     - tensor_map_l2_acts：   MN 方向宽度同样是 intermediate_hidden（load 视角）
// ============================================================================

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/mega_moe.hpp"

namespace deep_gemm {

// JIT 运行时：继承 LaunchRuntime 模板，负责代码生成 + kernel 启动
class SM100FP8FP4MegaMoERuntime final : public LaunchRuntime<SM100FP8FP4MegaMoERuntime> {
public:
    struct Args {
        // ================================================================
        // 模板期常量：决定 kernel 的编译实例（同组常量 → 同一个 .cubin 缓存）
        // ================================================================
        int num_max_tokens_per_rank;                  // 各 rank 的最大 token 数（决定 pool 容量）
        int hidden, intermediate_hidden;              // hidden size 与 SwiGLU 中间维度（= N/2）
        int num_experts, num_topk;                    // 全局 expert 总数 / 每 token topk
        int num_ranks;                                // NVLink 对等 rank 数
        float activation_clamp;                       // SwiGLU clamp 阈值（+inf = 关闭）
        bool fast_math;                               // 开启 APPROX 倒数 / __expf 等快速数学
        MegaMoEConfig config;                         // 启发式选出的 block/stage/thread 配置

        // ================================================================
        // 运行时参数：每次调用才变化的部分（通过 cudaLaunchKernel 传入）
        // ================================================================
        void* y;                                      // 最终输出 tensor 指针（BF16 [num_tokens, hidden]）
        int num_tokens;                               // 当前 batch 的 token 数
        layout::SymBuffer<> sym_buffer_ptrs;          // 跨 rank SymBuffer（持有全部 rank 的 base 指针数组）

        // ================================================================
        // 9 个 TMA TensorMap：
        //   l1_acts        → Linear1 输入 FP8 activation（pool）
        //   l1_acts_sf     → Linear1 输入 SF（UE8M0）
        //   l1_weights     → Linear1 权重 FP4
        //   l1_weights_sf  → Linear1 权重 SF
        //   l1_output      → Linear1 post-SwiGLU FP8 输出（存回 L2 acts 所在 tensor）
        //   l2_acts        → Linear2 输入 FP8 activation（即 l1_output 的读视角）
        //   l2_acts_sf     → Linear2 输入 SF（epilogue 期间写入 l2_sf_buffer）
        //   l2_weights     → Linear2 权重 FP4
        //   l2_weights_sf  → Linear2 权重 SF
        // ================================================================
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        CUtensorMap tensor_map_l2_weights_sf;

        // Launch configs：grid/block/smem/cluster（cluster=2 对应 2-CTA MMA）
        LaunchArgs launch_args;
    };

    // 根据 Args 生成 kernel C++ 源码片段（交给 NVRTC 编译成 .cubin）
    // 模板参数顺序必须与 sm100_fp8_fp4_mega_moe_impl 的定义保持一致
    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {}
    >);
}};
)", args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.store_block_m,
    args.config.sf_block_m, args.config.sf_block_n,
    args.config.num_max_pool_tokens,
    args.config.num_padded_sf_pool_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false");
    }

    // 真正的 kernel launch：模板实例化完成后，按顺序填入 runtime 参数
    //   launch_kernel 底层走 cuLaunchKernelEx，支持 cluster launch（cluster_dim=2）
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy —— Args 当前按值传入，拷贝了 9 个 TensorMap，可考虑 move
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.num_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.tensor_map_l1_weights_sf,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.tensor_map_l2_weights_sf
        ));
    }
};

// ============================================================================
// Python 侧入口：sm100_fp8_fp4_mega_moe
//   - 由 torch::Tensor 取出原始指针和 stride
//   - 根据 heuristics 决定 block/stage 配置
//   - 构造 9 个 TensorMap（TMA 描述符）
//   - 打包成 Args → JIT 编译 + 启动
//
// 参数说明：
//   y                   输出 BF16 tensor [num_tokens, hidden]
//   l1_acts/sf          Linear1 FP8 activation + SF（dispatch 后的 pool 数据）
//   l2_acts/sf          Linear2 FP8 activation + SF（Linear1 epilogue 写入）
//   l*_weights/sf       两层 FP4 权重 + SF（按 expert 存储）
//   sym_buffer_ptrs     全 rank 的 SymBuffer base 指针数组（NVLink 对称内存）
//   rank_idx            当前 rank 序号
//   num_max_tokens_*    单 rank 最大 token 数（pool 容量上限）
//   num_experts_per_rank 每 rank 的 expert 数
//   num_tokens          本次实际 token 数
//   num_topk            每 token 激活 expert 数
//   hidden              token 维度
//   intermediate_hidden Linear1 输出 / Linear2 输入维度（SwiGLU 后的宽度）
//   activation_clamp    SwiGLU clamp 阈值（float inf = 不 clamp）
//   fast_math           是否使用快速倒数/exp
// ============================================================================
static void sm100_fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;

    // 调用启发式：根据 shape / rank / expert 数量选择 block_m/n/k、stage 数、线程数等
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden);

    // SF 的 K 方向粒度固定为 32（UE8M0 按每 32 个元素共享 1 个 scale）
    constexpr int kGranK = 32;
    // ------------------------------------------------------------------------
    // TensorMap 构造：9 个，分 Linear1 / Linear2 两组
    //   make_tma_2d_desc(tensor, gmem_cols, gmem_rows, tile_cols, tile_rows, stride, swizzle)
    //   gmem_cols 对应 K（activation）或 N（weight）方向；tile 形状即 kernel 里的 BLOCK 大小
    // ------------------------------------------------------------------------

    // L1 activation：[num_max_pool_tokens, hidden]，K-tile = block_k, M-tile = load_block_m
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    // L1 activation SF：按 MN-major 存储（每 sf_block_m 行 × kGranK 列共享一个 UE8M0）
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_padded_sf_pool_tokens, hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0);
    // L1 weight：N 方向 = num_experts_per_rank * intermediate_hidden * 2
    //   （×2 是因为 SwiGLU 需要 gate 和 up 两份）
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l1_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf,
                                                           intermediate_hidden * 2, hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0);
    // NOTES: L1 output 与 L2 activations 是同一块 tensor（复用 intermediate buffer）
    //   post-SwiGLU 的 N 方向宽度只剩一半（gate×up 合并为单一 activation），
    //   所以 tile_cols = block_n/2，swizzle_mode 也要减半（128B → 64B）
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_max_pool_tokens,
                                                       config.block_n / 2, config.store_block_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       config.swizzle_acts_mode / 2);
    // L2 activation：读视角，shape 跟 l1_output 写视角不同（block_k vs block_n/2）
    //   这是同一块物理 memory 的两个 TMA descriptor
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_padded_sf_pool_tokens, intermediate_hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0);
    // L2 weight：[num_experts_per_rank, hidden, intermediate_hidden]
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        intermediate_hidden, num_experts_per_rank * hidden,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l2_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_weights_sf,
                                                           hidden, intermediate_hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0);

    // ------------------------------------------------------------------------
    // Launch：grid = num_sms（Persistent kernel，每 SM 一个 CTA）
    //         block = dispatch + non-epilogue（TMA/MMA）+ epilogue
    //         cluster = 2（2-CTA MMA multicast）
    // ------------------------------------------------------------------------
    const auto num_sms = device_runtime->get_num_sms();
    const SM100FP8FP4MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .config = config,
        .y = y.data_ptr(),
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .tensor_map_l1_weights_sf = tensor_map_l1_weights_sf,
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .tensor_map_l2_weights_sf = tensor_map_l2_weights_sf,
        .launch_args = LaunchArgs(num_sms,
                                  config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, 2)
    };

    const auto code = SM100FP8FP4MegaMoERuntime::generate(args);
    const auto runtime = compiler->build("sm100_fp8_fp4_mega_moe", code);
    SM100FP8FP4MegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
