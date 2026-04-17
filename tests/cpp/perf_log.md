# MegaFFN Perf Log

Qwen3-0.6B FFN decode: H=1024, I=3072, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128.

Device: single B200, measured on device 3 (`CUDA_VISIBLE_DEVICES=3`).
Metric: `cudaEventElapsedTime / num_iters`, warmup=5 iters, measured iters shown per run.

## Theoretical limits
- HBM BW ~8 TB/s, W1(6MB)+W2(3MB)+misc ≈ 9 MB  → **~1.1 µs**
- FP8 compute ~10 PFLOPS dense, 604 MFLOPs total → **~60 ns**
- SM 资源: 148 SMs. Step 0-2 launches `gridDim=kHidden/BLOCK_N=8` → 5.4% SMs; Step 3 `gridDim=48` → 32% SMs; Step 4 `gridDim=64` → 43% SMs (gridDim > 148 会因 1-block/SM 上限死锁)。

## External baselines (GPU 2, idle, 100 iters)

两份 baseline 都把 gate/up 合并成 `W1[2I, H]` 单次 GEMM（与 MegaFFN kernel 语义一致，
只有 SwiGLU 和 Linear2 三段是 unfused），`W2` 单独一次 GEMM，不是一个 kernel 融合实现。

| Baseline | Dtype | SiLU/Mul 精度 | M=1 (µs) | M=8 (µs) | M=16 (µs) | M=32 (µs) | 文件 |
|----------|-------|---------------|---------|---------|----------|----------|------|
| cuBLAS FP16 fused-W1 | FP16 | FP32 accum | 24.63 | 22.65 | 22.64 | 22.64 | `test_mega_ffn_baseline.cu` |
| **cuBLAS BF16 fused-W1** | **BF16** | **FP32 accum** | **15.61** | **15.09** | **15.45** | **15.90** | `test_ffn_cublas_baseline.cu` |

> ⚠️ **B200 上 BF16 cuBLAS 明显快于 FP16** (~1.5×)：推测是 FP16→FP32 accum 路径在 B200 tensor core 上比 BF16→FP32 accum 慢，或 cuBLAS heuristics 对 BF16 选了更优 kernel。
> 真实可比 baseline 应以 **BF16 fused** 为准，即 **~15 µs** @ M=1..32。
>
> 我们的 kernel 是 FP8 fused；理论上 B200 FP8 算力是 BF16 的 2×，应该比 BF16 cuBLAS 更快才合理。

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
| 3 | L2 K-split (kL2KSplit=6) + fp32 atomicAdd + post-cast kernel | 4 | 128 | **48** | **23.64** | **24.64** | **28.68** | gridDim 8→48：每 CTA 只算 1 个 L1 N-tile + 4 个 L2 K-block；Linear2 每 N-tile 由 6 个 CTA 在 K 维拆分，fp32 partial 走 `atomicAdd(float2*)` 合成，再由 `cast_y_fp32_to_bf16` 单独 kernel 降精度。相比 Step2 M=32 **22% 快**, M=1 **34% 快**; **已追平 cuBLAS FP16 unfused** (两者都 ~28 µs @ M=32)。K-split sweep: K=1/2/3/6 在 M=32 分别 36.93 / 36.93 / 32.80 / **28.68** µs, 越大 SM 越饱和; K=6 是 48/(8·K)=整数且 24/K=整数的上限。正确性 max‖Δ‖=0.0625 (atomicAdd 非关联带来的小抖动)。|
| 4 | 放松 L1 整除约束 + grid_sync 重写 (bit31 flip + ld.acq 轮询) | 4 | 128 | **64** | **22.53** | **22.58** | **24.61** | **ncu 先定位瓶颈**: 93.7% No-Eligible, L1TEX stall 37%, **CTA barrier stall 36%**, Compute 4.2% / DRAM 5.5% (严重 latency-bound 而非 throughput-bound)。从"K_SPLIT 6→12 不再加速" 可知 SM 并行度已非瓶颈。**核心优化**: 把 grid_sync 从"atomicAdd 热轮询" 换成 deep_gemm::comm::grid_sync 的 `atom.add.release.gpu` 一次 + `ld.acquire.gpu` 轻量轮询 + bit31 翻转抗 ABA。同时放松 `kL1OutputBlocksN % gridDim == 0`, 允许 kL2KSplit=8 (gridDim=64)。**效果**: 相比 Step 3 M=32 **14% 快**, 相比 Step 1 累计 **5.8× 快**; **打败 cuBLAS FP16 fused** (M=1 快 9%, M=8 持平, M=16/32 慢 4-9%)；但与真正的强 baseline **cuBLAS BF16 fused (~15 µs)** 还差 **1.4-1.6×**。正确性 max‖Δ‖=0.002~0.125 (远优于 cuBLAS FP16 的 max‖Δ‖=8.3)。|
