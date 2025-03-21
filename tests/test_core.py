import random
import torch
from typing import Tuple

import deep_gemm
from deep_gemm import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))


def construct(m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()

    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out


def construct_grouped(num_groups: int, m: int, k: int, n: int, is_masked: bool) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((num_groups, m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((num_groups, m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = torch.einsum('gmk,gnk->gmn', x, y)

    assert m % 4 == 0, f'TMA alignment error: {m}'
    x_fp8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty((num_groups, m, k // 128), device='cuda', dtype=torch.float))
    y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, (n + 127) // 128, k // 128), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i])
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

    # For non-masked input, we must merge the group and M dims
    if not is_masked:
        x_fp8 = (x_fp8[0].view(-1, k), per_token_cast_to_fp8(x.view(-1, k))[1])
        out, ref_out = out.view(-1, n), ref_out.view(-1, n)

    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out


def test_gemm() -> None:
    print('Testing GEMM:')
    for m in (256):
        for k, n in [(5120,  15360),]:
            x_fp8, y_fp8, out, ref_out = construct(m, k, n)
            deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
            diff = calc_diff(out, ref_out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

            # noinspection PyShadowingNames
            def test_func():
                # Construct new tensors every time to avoid L2 cache acceleration
                x_fp8, y_fp8, out, ref_out = construct(m, k, n)
                deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)

            t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
            print(f' > Performance (m={m:5}, n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
                  f'{(m * k + k * n + m * n * 2) / 1e9 / t:4.0f} GB/s')

            # Timing loop
            import time
            repeat=1000
            torch.cuda.synchronize()
            start = time.time()
            x_fp8, y_fp8, out, ref_out = construct(m, k, n)
            # noinspection PyShadowingNames
            def func():
                deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
            
            for _ in range(repeat):
                func()
            torch.cuda.synchronize()
            end = time.time()

            # Calculate timing and TFLOPS
            avg_time_ms = (end - start) / repeat * 1000
            avg_time_us = avg_time_ms * 1000
            tflops = 2 * m * n * k / (avg_time_ms * 1e-3) / 1e12
            gb_s = (m * k + k * n + m * n * 2) / 1e9 / (avg_time_ms * 1e-3)
            print(f"avg_time_us: {avg_time_us}, tflops: {tflops}, gb_s: {gb_s}")

    print()


def test_m_grouped_gemm_contiguous() -> None:
    print('Testing grouped contiguous GEMM:')

    for num_groups, m, k, n in ((4, 8192, 7168, 4096), (4, 8192, 2048, 7168), (8, 4096, 7168, 4096), (8, 4096, 2048, 7168)):
        # TODO: make a stronger test
        x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=False)
        m_indices = torch.arange(0, num_groups, device='cuda', dtype=torch.int)
        m_indices = m_indices.unsqueeze(-1).expand(num_groups, m).contiguous().view(-1)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)
        diff = calc_diff(out, ref_out)
        assert diff < 0.001, f'm={m * num_groups}, {k=}, {n=}, {diff:.5f}'

        # noinspection PyShadowingNames
        def test_func():
            # Construct new tensors every time to avoid L2 cache acceleration
            x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=False)
            m_indices = torch.arange(0, num_groups, device='cuda', dtype=torch.int)
            m_indices = m_indices.unsqueeze(-1).expand(num_groups, m).contiguous().view(-1)
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Performance ({num_groups=}, m_per_group={m:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
              f'throughput: {2 * num_groups * m * n * k / t / 1e12:4.0f} TFLOPS, '
              f'{(num_groups * (m * k + k * n + m * n * 2)) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_masked() -> None:
    print('Testing grouped masked GEMM:')

    for num_groups, m in ((4, 1)):
        for k, n in ((3072, 4096), (2048, 3072)):
            # Test correctness
            masked_m_candidates = list(filter(lambda candidate: candidate <= m, (64, 128, 192, 256, 320, 384)))
            for i in range(10):
                x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=True)
                masked_m = torch.empty((num_groups, ), device='cuda', dtype=torch.int)
                for j in range(num_groups):
                    masked_m[j] = random.choice(masked_m_candidates)
                expected_m = min(int(masked_m.float().mean()) + 1, m)
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, expected_m)
                for j in range(num_groups):
                    diff = calc_diff(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()])
                    assert diff < 0.001, f'{m=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'

            # noinspection PyShadowingNames
            def test_func():
                # Construct new tensors every time to avoid L2 cache acceleration
                x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=True)
                masked_m = torch.ones((num_groups, ), device='cuda', dtype=torch.int) * m
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, m)

            # Test performance with fixed shapes
            t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
            print(f' > Performance ({num_groups=}, m_per_group={m:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * num_groups * m * n * k / t / 1e12:4.0f} TFLOPS, '
                  f'{(num_groups * (m * k + k * n + m * n * 2)) / 1e9 / t:4.0f} GB/s')
    print()

def generate_numbers_with_constraints(count, target_sum, non_zero_count, max_value):
    """
    生成指定个数的非负整数，满足以下条件：
    1. 数的总和等于指定的总和。
    2. 非零数的个数由用户指定。
    3. 数组中的最大值不超过用户指定的正数。

    :param count: 生成的数的总个数
    :param target_sum: 数的总和
    :param non_zero_count: 非零数的个数
    :param max_value: 生成的数的最大值
    :return: 生成的数的列表
    """
    if count <= 0:
        raise ValueError("数的个数必须大于0")
    if target_sum < 0:
        raise ValueError("总和必须是非负数")
    if non_zero_count < 0 or non_zero_count > count:
        raise ValueError("非零数的个数必须在0到总个数之间")
    if target_sum < non_zero_count:
        raise ValueError("总和必须大于或等于非零数的个数")
    if max_value <= 0:
        raise ValueError("最大值必须是正数")
    if target_sum > max_value * non_zero_count:
        raise ValueError("总和不能超过最大值乘以非零数的个数")

    # 初始化一个列表，所有元素初始值为0
    numbers = [0] * count

    # 随机选择非零数的位置
    non_zero_indices = random.sample(range(count), non_zero_count)

    # 初始化非零数的值为1（确保每个非零数至少为1）
    for index in non_zero_indices:
        numbers[index] = 1

    # 剩余需要分配的总和
    remaining_sum = target_sum - non_zero_count

    # 随机分配剩余的总和到非零数中，同时确保每个数不超过最大值
    while remaining_sum > 0:
        random_index = random.choice(non_zero_indices)
        if numbers[random_index] < max_value:
            numbers[random_index] += 1
            remaining_sum -= 1

    return numbers

def test_m_grouped_gemm_masked_moe() -> None:
    print('Testing grouped masked GEMM MoE:')

    top_k = 4
    num_groups = 48

    # m = m_max, 单expert中最大的m
    for batch_size, act_expert_num, m in ((1, 4, 4), (2, 8, 4), (4, 15, 4), (8, 24, 4), (16, 36, 8), (32, 45, 8), (64, 48, 16), (128, 48, 24), (256, 48, 32), (512, 48, 64),):
        for k, n in ((3072, 4096), ):

            # 一共选择 bs*top_k个expert
            total_expert_count = batch_size * top_k

            for i in range(10):
                x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=True)
                
                masked_m = torch.empty((num_groups, ), device='cuda', dtype=torch.int)

                # 产生48个数，分别表示每个expert从m_max中取多少个数 真正计算
                random_mask = generate_numbers_with_constraints(num_groups, total_expert_count, act_expert_num, batch_size)
                for j in range(num_groups):
                    masked_m[j] = random_mask[j]#random.choice(masked_m_candidates)
                expected_m = min(int(masked_m.float().mean()) + 1, m)

                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, expected_m)
            
            # noinspection PyShadowingNames
            def test_func():
                # Construct new tensors every time to avoid L2 cache acceleration
                x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=True)
                masked_m = torch.empty((num_groups, ), device='cuda', dtype=torch.int)

                # 产生48个数，分别表示每个expert从m_max中取多少个数 真正计算
                random_mask = generate_numbers_with_constraints(num_groups, total_expert_count, act_expert_num, batch_size)
                for j in range(num_groups):
                    masked_m[j] = random_mask[j]#random.choice(masked_m_candidates)
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, m)

            # Test performance with fixed shapes
            t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)

            total_cal = 0
            total_mem = 0

            for j in range(num_groups):
                total_cal += 2 * masked_m[j] * n * k / 1e12
                total_mem += (masked_m[j] * k + k * n + masked_m[j] * n * 2)/ 1e9

            print(f' > Performance ({num_groups=}, m_per_group={m:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
                  f'throughput: {total_cal / t:4.0f} TFLOPS, '
                  f'{total_mem / t:4.0f} GB/s')

            # print(f' > Performance ({num_groups=}, m_per_group={m:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
            #       f'throughput: {2 * num_groups * m * n * k / t / 1e12:4.0f} TFLOPS, '
            #       f'{(num_groups * (m * k + k * n + m * n * 2)) / 1e9 / t:4.0f} GB/s')
            
    print()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_gemm()
    #test_m_grouped_gemm_contiguous()
    test_m_grouped_gemm_masked()
