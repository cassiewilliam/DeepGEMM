# MegaFFN Perf Log

Qwen3-0.6B FFN decode: H=1024, I=3072, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128.

Device: single B200, measured on device 3 (`CUDA_VISIBLE_DEVICES=3`).
Metric: `cudaEventElapsedTime / num_iters`, warmup=5 iters, measured iters shown per run.

## Theoretical limits
- HBM BW ~8 TB/s, W1(6MB)+W2(3MB)+misc ≈ 9 MB  → **~1.1 µs**
- FP8 compute ~10 PFLOPS dense, 604 MFLOPs total → **~60 ns**
- SM 资源: 148 SMs. Our kernel launches `gridDim=kHidden/BLOCK_N=8` → 5.4% SMs.

## Step-by-step

(table appended live below)

| Step | Change | kNumStages | kNumEpiTh | gridDim | M=1 (µs) | M=4 (µs) | M=32 (µs) | Notes |
|------|--------|-----------|-----------|---------|----------|----------|-----------|-------|
| 0 | baseline (post-correctness rewrite) | 3 | 128 | 8 | 147.29 | 147.45 | 149.63 | Kernel 每 CTA 冗余跑 Linear1；SM 使用率 5.4%。REG=208, SMEM=165KB, 无 spill |
