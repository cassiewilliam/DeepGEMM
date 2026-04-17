# MegaFFN Perf Log

Qwen3-0.6B FFN decode: H=1024, I=3072, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128.

Device: single B200, measured on device 3 (`CUDA_VISIBLE_DEVICES=3`).
Metric: `cudaEventElapsedTime / num_iters`, warmup=5 iters, measured iters shown per run.

## Theoretical limits
- HBM BW ~8 TB/s, W1(6MB)+W2(3MB)+misc ≈ 9 MB  → **~1.1 µs**
- FP8 compute ~10 PFLOPS dense, 604 MFLOPs total → **~60 ns**
- SM 资源: 148 SMs. Our kernel launches `gridDim=kHidden/BLOCK_N=8` → 5.4% SMs.

## External baseline (未融合, 仅供对照; GPU 6, 100 iters)
跑自 `tests/cpp/test_mega_ffn_baseline.cu`，与主测试同 seed (0xBEEF) 生成输入。
数值误差相对于 FP8-faithful CPU 参考约 1.37 (FP16 baseline 无中间 FP8 量化)。

| Baseline | 技术 | M=1 (µs) | M=4 (µs) | M=32 (µs) | Notes |
|----------|------|----------|----------|-----------|-------|
| cuBLAS FP16 unfused | 2× cublasGemmEx(FP16) + SwiGLU(FP16) | **24.63** | **22.64** | **22.63** | Tensor Core, cuBLAS 默认 heuristics |

> 注：cuBLAS 小 M 时走 GEMV-like split-K 能吃满 148 个 SM；当前 MegaFFN 只用 8 个 CTA 是主要 gap。

## Parameter sweep（GPU 6, 100 iters）

| Config | M=1 (µs) | M=32 (µs) |
|---|---|---|
| STAGES=2 EPI=128 | 186.39 | 188.58 |
| STAGES=3 EPI=128 | 147.48 | 149.56 |
| **STAGES=4 EPI=128** | **143.51** | **143.30** |
| STAGES=5 EPI=128 | SMEM=227KB → launch fail |
| STAGES=* EPI=256 | `WG_BLOCK_M % STORE_BLOCK_M != 0`, epilogue 架构仅支持 1 WG |

## MegaFFN step-by-step

| Step | Change | kNumStages | kNumEpiTh | gridDim | M=1 (µs) | M=4 (µs) | M=32 (µs) | Notes |
|------|--------|-----------|-----------|---------|----------|----------|-----------|-------|
| 0 | baseline (post-correctness rewrite) | 3 | 128 | 8 | 147.29 | 147.45 | 149.63 | Kernel 每 CTA 冗余跑 Linear1；SM 使用率 5.4%。REG=208, SMEM=165KB, 无 spill。**vs cuBLAS FP16 约 6.6× 慢** |
| 1 | STAGES 3 → 4 (更深 K-pipeline) | **4** | 128 | 8 | 143.51 | 143.23 | 143.30 | SMEM 增至 185KB, 仍无 spill。-3% M=1, -4% M=32 |
| 2 | Linear1 N-split + shared workspace + cross-CTA grid-sync | 4 | 128 | 8 | **35.83** | **35.74** | **36.85** | 每 CTA 只算 6/48 个 L1 tile (原 8× 冗余消除)；workspace 2D 跨 CTA 共享；atomic l1_done_counter spin 做 grid-sync。相比 Step1 **≈4×** 提速，vs cuBLAS FP16 gap 从 6.6× 缩到 **1.58×**。正确性 mean‖Δ‖≈0，M=32 max‖Δ‖=1 (BF16 量化边界)。**坑**: grid_sync 最初用 `__syncthreads()` (bar 0 count=256) 与 epilogue `sync_aligned(128,0)` 冲突 → illegal instruction，改用专用 bar id 15 修复 |
