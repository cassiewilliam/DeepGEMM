# MegaFFN Perf Log

Qwen3-0.6B FFN decode: H=1024, I=3072, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128.

Device: single B200, measured on device 3 (`CUDA_VISIBLE_DEVICES=3`).
Metric: `cudaEventElapsedTime / num_iters`, warmup=5 iters, measured iters shown per run.

## Theoretical limits
- HBM BW ~8 TB/s, W1(6MB)+W2(3MB)+misc ≈ 9 MB  → **~1.1 µs**
- FP8 compute ~10 PFLOPS dense, 604 MFLOPs total → **~60 ns**
- SM 资源: 148 SMs. Our kernel launches `gridDim=kHidden/BLOCK_N=8` → 5.4% SMs.

## External baselines (未融合, 仅供对照; GPU 6, 100 iters)
跑自 `tests/cpp/test_mega_ffn_baseline.cu`，与主测试同 seed (0xBEEF) 生成输入；
数值误差相对于 FP8-faithful CPU 参考约 1.37 (BF16 baseline 无中间 FP8 量化)。

| Baseline | 技术 | M=1 (µs) | M=4 (µs) | M=32 (µs) | Notes |
|----------|------|----------|----------|-----------|-------|
| Naive BF16 | 3 个朴素 CUDA kernel, 无 TC 无 SMEM | 279.62 | 528.39 | 1681.60 | 一 thread 一元素, 只做 upper bound 参考 |
| cuBLAS BF16 unfused | 2× cublasGemmEx(BF16) + SwiGLU kernel | **15.56** | **15.28** | **16.47** | Tensor Core, 走 cuBLAS 默认 heuristics |

> 注：cuBLAS 小 M 时走 GEMV-like split-K 能吃满 148 个 SM；当前 MegaFFN 只用 8 个 CTA, 这正是主要 gap。

## MegaFFN step-by-step

| Step | Change | kNumStages | kNumEpiTh | gridDim | M=1 (µs) | M=4 (µs) | M=32 (µs) | Notes |
|------|--------|-----------|-----------|---------|----------|----------|-----------|-------|
| 0 | baseline (post-correctness rewrite) | 3 | 128 | 8 | 147.29 | 147.45 | 149.63 | Kernel 每 CTA 冗余跑 Linear1；SM 使用率 5.4%。REG=208, SMEM=165KB, 无 spill。**vs cuBLAS 约 9.5× 慢** |
