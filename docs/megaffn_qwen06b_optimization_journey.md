# MegaFFN Qwen3-0.6B Decode Optimization Journey (v0 → v3.0)

**Target hardware:** NVIDIA B200 (SM100a, Blackwell), single CTA cluster
**Model:** Qwen3-0.6B FFN (Linear1 → SwiGLU → Linear2)
**Shapes:** `kHidden=1024`, `kIntermediate=3072`, decode `M ≤ 32`
**Input dtype:** FP8 e4m3 (MX-FP8 with UE8M0 block-scale, or per-tensor FP8)
**Output dtype:** BF16
**Launch:** programmatic stream serialization (PDL) on

> This document summarizes the end-to-end optimization journey from a naive kernel
> (~143 µs @ M=1) to the current production ceiling v3.0 (**9.28 µs @ M=32**).
> The focus is on **what worked, what didn't, and why** — including the analytical
> mistakes that cost us iterations.

---

## 0. Problem Statement

Qwen3-0.6B's FFN dominates decode wall-clock latency. Each token runs:
```
y = Linear2( SwiGLU( Linear1(x) ) )
  = W2 · ( silu(W1_gate · x) ⊙ (W1_up · x) )
```
with dimensions `x ∈ R^{M×1024}`, `W1 ∈ R^{6144×1024}` (gate‖up), `W2 ∈ R^{1024×3072}`, `y ∈ R^{M×1024}`.

Key properties of **decode**:
- **M is small** (typically 1, sometimes up to 32 for speculative / short batches).
- Each fma of UMMA has minimum `UMMA_M = 128` (hardware), so naive kernels waste
  ≥75% of the TMEM accumulator on padding rows.
- Wall-clock is dominated by **barrier + TMA latency** rather than raw Tensor-Core FLOPS.
- Baseline cuBLAS BF16 fused-W1 ≈ 15.6 µs @ M=1..32 on B200.

**Theoretical floor** (HBM weight-read only, no kernel overhead):
- W1 + W2 = 6MB + 3MB = 9MB FP8
- 9MB / B200 HBM3e peak 8 TB/s ≈ **1.16 µs**
- cuBLAS BF16 = 13.5× theoretical
- v1 MX-FP8 (Step 29) = 9.5× theoretical
- **v3.0 = 8.0× theoretical** (9.28 µs @ M=32)

---

## 1. Baseline and Kernel Structure

We implement a **single monolithic kernel** that fuses Linear1, SwiGLU, and Linear2
with warp-specialized pipelining:

```
+-------------------+--------------------------------+---------------------+
| Role              | Work                           | Barrier             |
+-------------------+--------------------------------+---------------------+
| TMA-A warp (w=0)  | Stream A SMEM (X or W)          | full/empty mbarrier |
| TMA-B warp (w=1)  | Stream B SMEM (W or X)          | full/empty mbarrier |
| MMA warp (w=2)    | Issue UMMA fma + UTCCP SF       | tmem_full/empty     |
| Cold warp (w=3)   | Idle (register-budget holder)   | grid/cluster sync   |
| Epilogue (w=4..)  | SwiGLU quant, BF16 cast, stores | tmem_full/empty     |
+-------------------+--------------------------------+---------------------+
```

Across N-tiles: L1 writes FP8 intermediate to HBM workspace (cross-CTA shared),
grid-sync, L2 reads workspace and reduces across K-splits into `y` slots, another
cluster-sync, then reduce+cast to BF16 in the final iteration.

Grid topology (from early steps):
- `gridDim.x = kL2OutputBlocksN (8) × kL2KSplit (8) = 64 CTAs`
- `cluster_dim = 8` (matches `kL2KSplit` for post-L2 cluster_sync)
- `l2_n_tile = cta / 8`, `l2_k_half = cta % 8`
- Linear1 N-split: first 48 CTAs each do 1 N-tile (`kL1OutputBlocksN = 48`), last 16 skip L1

---

## 2. v1: MX-FP8 Monolith — 143 µs → 11.02 µs (Steps 1–29)

The MX-FP8 path uses per-32K block UE8M0 scales with `tcgen05.mma.kind::mxf8f6f4` UMMA.
Each of the 29 optimization steps is small; the **key breakthroughs** were:

### Step 2 — Linear1 N-split: 143 → 36 µs (−75%)

Naive v1 had each CTA computing the full `kL1OutputBlocksN=48` N-tiles.
Step 2 distributes them across CTAs: `cta k` does tiles `[k, k+stride, ...]`.
**Why it helped:** eliminated 48× redundant weight-loads; kernel became latency-bound rather than bandwidth-bound.

### Step 3 — Linear2 K-split + atomicAdd: 36 → 28 µs (−22%)

Linear2's K = 3072/128 = 24 blocks was serialized per CTA. Split K across 8 CTAs
per N-tile, each computes a partial sum, atomic-add into `y_fp32`. `gridDim.x = 8 × 8 = 64`.
**Why it helped:** parallelized the L2 critical path 8× without blowing up TMA-B bandwidth.

### Step 6 — Programmatic Dependent Launch (PDL): 16 → 13 µs (−19%)

`cudaLaunchAttributeProgrammaticStreamSerialization = 1`. Kernel calls
`cutlass::arch::launch_dependent_grids()` at entry so the next iter's CTAs can spin up
while the current kernel is still executing. **This is the single biggest decode-latency
win** — should be tried first for any decode kernel.

### Step 8 — Slot-based L2 reduce: 13 → 12.4 µs (−5%)

Replaced `atomicAdd` into `y_fp32[m, h]` with per-CTA `slot[cta, m, h]` writes
+ post-sync scalar reduce. **Why it helped:** eliminated HBM atomic contention
that was scaling poorly with kL2KSplit.

### Steps 14–16 — Cluster=8 + `cluster_sync`: 12.26 → 11.43 µs (−7%)

Replaced per-tile HBM counter with `cute::cluster_sync` (hardware barrier,
GPC-local). Required fixing `arrive(0u)` → `arrive()` race. Cluster size must
equal `kL2KSplit` for semantic correctness.

### Steps 18–22 — Bar.sync thread count reduction: 11.43 → 11.14 µs (−2.5%)

The L1→L2 `grid_sync` uses `bar.sync.15` for CTA-scope sync before the global
atomic-counter phase. We discovered that **cold warp + TMA-B warp don't need to
participate** (they have no L1-epi state). Dropped from 256 → 192 threads.

Further: removed redundant `sync_aligned` before `grid_sync` (the bar.sync.15
itself already waits for `tma_store_wait`).

### Steps 28–29 — Cluster-scope fence: 11.14 → 11.02 µs (−1%)

`cute::cluster_sync()` carries an implicit `MEMBAR.ALL.GPU` in `cluster_arrive`.
For our slot-write pattern (writes only read by cluster peers in same GPC/L2),
GPU-scope fence is overkill. Replaced with:
```cpp
asm volatile("fence.acq_rel.cluster;");
cute::cluster_arrive_relaxed();
cute::cluster_wait();
```
Cluster-scope fence is cheaper than GPU-scope fence by ~80 ns.

### What didn't work (Steps 30–34)

| # | Attempt | Result |
|---|---------|--------|
| 30 | `__nanosleep(40)` spin inside barrier wait | No measurable diff |
| 31 | Fence in epi only (not all warps) | +2.9% slower (serialization on epi critical path) |
| 32 | `st.release.cluster` per slot-write store | +156% slower (each release-store serializes 16-wide epi) |
| 33 | Non-epi warps skip `cluster_wait` (arrive-only) | +1.0% slower (warp early-exit confuses scheduler) |
| 34 | `kNumEpilogueStages 2→3` | +0.5% slower (over TMEM cap at 4; 3 deepens epi pipeline but scheduler churns) |

**ncu profile at Step 29 (`v1` end state):**
```
smsp__average_warps_issue_stalled_membar_per_issue_active.pct = 167%
One or More Eligible                                            = 8.14%
kernel duration (GPU time)                                       = 21.02 µs
wall-clock (PDL hides ~10µs)                                     = 11.02 µs
```

Interpretation: **barrier-bound**, not compute/pipeline-depth bound. This led us
to conclude (wrongly, as we'll see) that further speedup requires structural
changes to barrier topology.

### Step 34's indirect conclusion — rejection of swap AB (WRONG, corrected in v3)

At the time, Step 34's result (deeper epi stages = no benefit) was read as:
> "Swap AB's only benefit hypothesis is UMMA_N 128→32 → TMEM 4× shrink → enable
> more epi stages. Since deeper stages don't help, swap AB is empirically dead."

This **rejection was premature** — it only addressed one of swap's potential benefit
axes (epi stages). The actual v3 benefit came from cross-warp exchange elimination
and L2 slot layout, which Step 34 didn't probe.

---

## 3. v2: Per-Tensor FP8 — 11.02 → 9.52 µs (−14%)

**Insight:** Remove the entire SF (scale factor) pipeline. Per-tensor FP8 uses a
single fp32 scale per tensor (passed as kernel arg), not per-32K-block UE8M0.

### What was removed (PT vs MX-FP8)
| Component | Saving |
|-----------|--------|
| SF TMA loads (SFA + SFB per K-block) | 640 bytes × 384 K-blocks = 240KB/kernel |
| SF SMEM (4× 1KB per stage) | 4KB dynamic SMEM |
| UTCCP 4×32dp128bit instruction (2 per K-block) | ~30 cycles/K-block |
| `utccp_smem_transpose` lambda | ~32 SMEM L + 32 SMEM S per warp per K-block |
| SF TMEM columns (8 cols) | ~16KB TMEM |
| `tcgen05.mma.kind::mxf8f6f4.block_scale` → `kind::f8f6f4` | Simpler 5-arg fma, no SF addressing |

### What was added
- 3 kernel scale args: `scale_xw1 = s_X·s_W1`, `scale_inv_intermediate = 1/s_I`, `scale_iw2 = s_I·s_W2`
- L1 epi per element: `gate_real = g·scale_xw1`, `result = silu(gate_real) · (up·scale_up_comb)` where `scale_up_comb = scale_xw1·scale_inv_intermediate`
- L2 reduce: multiply `acc.xyzw *= scale_iw2` before BF16 cast

Net: much more removed than added → 14% speedup.

### Important correction to earlier analysis

I predicted **"barrier-bound → removing SF pipeline saves ~0.3 µs at best"**.
Actual saving: **1.6 µs**. Takeaway: **"barrier-bound" is not an excuse to ignore
per-K-block work**. MMA warp saving ~200 cycles/K-block × 384 K-blocks ≈ 77K total
warp-cycles, which shortens the critical path between barriers.

### v2 parameter sweep — all configs worse than canonical

Swept `(kNumStages, kL2KSplit, kClusterDim, kNumEpilogueThreads)`:

| Config | M=1 | M=8 | M=32 |
|--------|-----|-----|------|
| **4 / 8 / 8 / 128 (canonical)** | 9.53 | 9.59 | 10.34 |
| 5 / 8 / 8 / 128 (deeper K-pipeline) | 9.80 | 9.86 | 10.70 |
| 4 / 4 / 4 / 128 (fewer CTAs) | 11.66 | 11.75 | 13.12 |
| 4 / 8 / 1 / 128 (per-tile counter, no cluster_sync) | 10.38 | 10.47 | 11.04 |
| 5 / 4 / 4 / 128 | 11.56 | 11.65 | 12.85 |

Confirmed: canonical `(4, 8, 8, 128)` remains optimal. K-pipeline depth ≥5 loses ~3%
due to scheduler confusion (Step 34's conclusion still holds on v2).

### v2 L1 epi fusion attempt — REJECTED

Tried merging compute + SMEM-store per atom (saving 16 registers from `fp8_row[]`).
Result: **+5% slowdown**. Root cause: `tmem_empty_barriers->arrive()` was pushed
past all SMEM stores, extending MMA's next-iter wait.

**Critical invariant (saved for posterity):** in any epi rewrite,
`tmem_empty_arrive` **MUST happen immediately after TMEM_LOAD**, before SMEM CD writes.

---

## 4. Failed path: Stage-B Swap AB on MX-FP8 (+33 to +54% SLOWER)

Before v3, we tried Stage-B swap AB on the MX-FP8 kernel. Result:

| M  | Main MX-FP8 (Step 29) | Swap MX-FP8 (Stage-B) | Delta |
|----|------------------------|------------------------|-------|
| 1  | 11.13 µs               | 14.85 µs               | +33.4% |
| 8  | 11.19 µs               | 15.55 µs               | +38.9% |
| 32 | 12.01 µs               | 18.47 µs               | +53.8% |

The swap worked correctly but **paid heavily on four axes**:

| Swap-specific cost on MX-FP8 | Overhead @ M=32 |
|------------------------------|-----------------|
| Cross-warp SMEM exchange (gate↔up pairing) + `sync_aligned` | ~0.8 µs |
| Per-m butterfly amax (32-lane shfl × 32 m × 2 chunks) | ~1.5 µs |
| Transposed L2 slot writes (32 strided scalar stores per lane) | ~0.8 µs |
| SF pipeline (same as v1) | ~1.6 µs |
| **Total** | **~4.7 µs** |

This was the third rejection of swap AB:
1. Step 29 wrote "swap 即使全实现 (~400 LOC, 6-10 小时) 也带不来 wall-time 收益"
2. Stage-B empirical: +33-54% slower on MX, confirming prediction
3. Analytical estimate for **swap+PT**: ~13 µs, still slower than v2 PT (9.53 µs)

**All three rejections were wrong**, as v3 shows below. The missing insight was
that the swap's overheads are **not structural** — they're artifacts of a naive
port that didn't redesign layouts.

---

## 5. v3.0: swap AB + PT + pre-merged W1 + slot[cta][h][m] — 9.28 µs (−22.7% vs v1, −10% vs v2 at M=32)

### Bench summary

| M  | v1 Step 29 MX | v2 PT    | **v3.0**      | v3 vs v1   | v3 vs v2   |
|----|---------------|----------|----------------|------------|------------|
| 1  | 11.13 µs      | 9.53 µs  | **9.29 µs**   | −16.5%     | −2.5%      |
| 8  | 11.19 µs      | 9.59 µs  | **9.31 µs**   | −16.8%     | −2.9%      |
| 32 | 12.01 µs      | 10.34 µs | **9.28 µs ✅** | **−22.7%** | **−10.3%** |

**Most striking:** v3 is nearly **M-flat** (9.28–9.31 µs across M=1..32). This is
because UMMA_N=32 is fixed — every fma processes 32 X-M cols regardless of
`num_tokens`. L2 epi writes `UMMA_N` contiguous fp32 per lane (vectorized),
constant cost.

### The three structural changes

#### Change 1: Swap AB (A=W, B=X)

Hardware 1-CTA MXF8F6F4 constraint: `UMMA_M = 128`, `UMMA_N % 8 ∈ [8, 256]`.

Original (main):
- A = X [M=128 padded, K], B = W [N=128, K]
- TMEM accumulator = [128 M rows × 128 N cols]. For valid M=1..32, 75–99% wasted.

Swapped:
- A = W [N=128, K], B = X [M=32, K] (UMMA_N chosen = kMaxValidM = 32)
- TMEM accumulator = [128 W-N rows × 32 X-M cols]. **Zero M-waste**.
- MMA output is C^T (transposed view of original C). Epi reads transposed.

Implications:
- `SMEM_B_SIZE_PER_STAGE`: 16KB → 4KB (4× smaller)
- TMEM accumulator: 64KB → 16KB (4× smaller)
- MMA fma work per call: `128×128×32 → 128×32×32` (4× less, critical path ~4× shorter)
- TMA-A loads W (was loading X); TMA-B loads X (was loading W, smaller box now).
- **TMA-B must now wait `l1_to_l2_sync`** before L2 phase (it reads intermediate in swap, was W2 in main).

#### Change 2: Pre-merged W1 layout + intra-warp shfl_xor

The naive swap hit its worst penalty in the L1 epi — gate (W-N rows 0–63) and
up (W-N rows 64–127) land in **different warps** of TMEM, requiring a cross-warp
SMEM exchange.

**Solution: permute W1's N dimension on the host**, before any quantization,
so that within each `BLOCK_N=128` tile the layout is:
```
new_W1 rows:  [g0..15, u0..15, g16..31, u16..31, g32..47, u32..47, g48..63, u48..63]
                ^warp 0 owns^    ^warp 1 owns^   ^warp 2 owns^    ^warp 3 owns^
```

Now in TMEM:
- Warp `w` lane `l` (l∈[0,32)): TMEM row `w*32+l`
  - if `l < 16`: **gate** for `output_n = w*16 + l`
  - if `l ≥ 16`: **up**   for `output_n = w*16 + (l-16)`

Lane `l` and lane `l+16` own gate+up for the **same output_n within the same warp**.
Pairing is done by a single `__shfl_xor_sync(_, v, 16)` per element — no SMEM,
no cross-warp sync.

Host-side W1 permutation is a once-per-model-load cost, invisible to runtime.

#### Change 3: L2 slot layout `[cta][h_local][m]` (was `[cta][m][h_local]`)

After swap, each L2 epi lane has `TMEM[h_local, 0..31]` (1 H-row × 32 M-cols). To
write into the original slot layout `slot[cta][m][h_local]`, each lane wrote 32
scalar fp32 stores, stride = `BLOCK_N = 512 bytes`. 32 strided scalar writes/lane.

**Insight:** the slot buffer is **kernel-private** — no downstream consumer constrains
its layout. Rearranging to `slot[cta][h_local][m]` makes each lane's 32 writes
**contiguous**: `8 × float4 stores per lane`, fully vectorized, no bank conflicts.

The reduce step's indexing flips: we now vectorize on the m dimension when summing
across K-split peers, and the final BF16 write to `y[m][h]` becomes strided (4 BF16
scalar stores per thread, stride = `kHidden · 2`). Total strided-write bytes:
`valid_m × kHidden × 2B = ~64 KB` across the kernel — negligible at HBM 8 TB/s.

### Where my earlier analytical rejections were wrong

1. **"Cross-warp SMEM exchange is irreducible" — WRONG.**
   Pre-merging W1 makes gate+up co-warp; `shfl_xor` replaces SMEM.

2. **"L2 transposed writes = slow strided scalars" — WRONG.**
   Only true if the destination layout is fixed. For internal buffers, layout can match swap's natural output.

3. **"Main structurally better for M=32" — WRONG.**
   v3 beats v2 at M=32 by 10%. v2's "fewer active lanes at M=32" (only warp 0)
   is actually a DISADVANTAGE — it bottlenecks L2 epi on one warp's SMEM bank width.
   v3's all-4-warp participation spreads bank pressure.

4. **"MMA fma speedup hidden by TMA-A bottleneck" — PARTIALLY WRONG.**
   TMA-A is indeed the K-pipeline bottleneck. But the per-N-tile critical path also
   includes MMA → `tmem_full` barrier → epi. Shorter fma → earlier epi start →
   shorter wall-time per N-tile.

### Implementation outline

Files (uncommitted, on host `10.77.188.34`):
```
deep_gemm/include/deep_gemm/impls/sm100_fp8_mega_ffn_v3.cuh  (~1050 LOC)
tests/cpp/test_sm100_fp8_mega_ffn_v3.cu                       (with permute_w1_premerged helper)
tests/cpp/build_mega_ffn_v3.sh
```

Stages (S1→S4, each independently testable):
- **S1**: Swap MX → Swap PT. Drop SF TMA/SMEM/UTCCP/TMEM cols. `mxf8f6f4` → `f8f6f4` MMA. 3 scale args.
- **S2**: Pre-merged W1. Host permutation function. L1 epi: `__shfl_xor_sync(_, v, 16)` + gate-lane only writes. CPU ref updated.
- **S3**: L2 slot layout. Swap offset formula in both epi write and reduce read. Reduce vectorizes on m.
- **S4**: Correctness + bench.

---

## 6. Microarchitecture Lessons

### Lesson 1: "Barrier-bound" is NOT "per-iter-work-free"

ncu shows 92% of cycles with 0 eligible warps, but this doesn't mean removing
work from any one warp is useless. The critical path is:
```
warp N iter K → arrive(barrier) → all warps wait → barrier flips → warp N+1 iter K+1
```
Shortening any warp's pre-arrive work by 200 cycles shortens the critical path by
200 cycles × number_of_iters. Across 384 K-blocks per kernel, a 200-cycle saving
per K-block = 77K cycles = **55 µs of latent warp-time**, which translates to
~1.6 µs of wall-clock (if one warp is the bottleneck).

This explains why PT saves 1.6 µs despite "barrier-bound" label.

### Lesson 2: UMMA cta_group::1 fma latency is not fully hidden

`tcgen05.mma` is asynchronous (fma instruction returns quickly, work runs in the
background). But the `tmem_full_barriers[stage]->arrive` via `umma_arrive` only
fires after the actual tensor-core work completes. Smaller fma → earlier arrive →
earlier epi consumption.

For MXF8F6F4 1-CTA on B200, fma duration scales roughly with `M×N×K` tensor-core
ops. UMMA_N=32 vs 128 is 4× fewer ops → proportionally shorter critical path
contribution.

### Lesson 3: Internal buffer layouts are free variables

The slot buffer, the intermediate workspace in SMEM, any DSMEM, any scratch — all
are kernel-internal and NOT constrained by any external ABI. Optimize their layout
to match the access pattern **without** worrying about downstream readers.

This was the key insight unlocking v3's L2 slot win.

### Lesson 4: Pre-arrangement on the host can eliminate kernel overhead

W1 permutation is a one-time cost at model load. It eliminates **every per-iter**
cross-warp SMEM exchange. Runtime amortization: infinite.

For deployment pipelines, this means:
- Rearrange weights offline (one-time).
- Pass scales as kernel args (per-launch, cheap).
- Kernel becomes simpler + faster.

### Lesson 5: `tmem_empty_arrive` placement is critical

The v2 L1-epi fusion attempt failed because it pushed `tmem_empty_arrive` past the
SMEM CD write phase. MMA's next iter waited longer; total time grew.

**Invariant**: arrive on `tmem_empty_barriers` immediately after TMEM_LOAD completes,
before any SMEM write work. Place SMEM CD writes in a later phase with their own
sync.

---

## 7. Rejected Optimizations (don't re-try)

| Attempt | Why it fails | Evidence |
|---------|--------------|----------|
| kNumStages 4→5 on v1/v2 | scheduler confusion, no pipeline-depth wins available | Step 34 + v2 sweep |
| kNumEpilogueStages 2→3 | exceeds TMEM cap at 4; 3 deepens epi pipeline but no benefit | Step 34 |
| L1 epi compute+store fusion | pushes tmem_empty_arrive late, extends MMA wait | v2 attempt (+5%) |
| kClusterDim=1 (per-tile counter) | slower than cluster_sync; HBM atomic counter beats peer barriers only with small clusters | v2 sweep (+9%) |
| kL2KSplit=4 | halves CTA count, fewer workers per output tile | v2 sweep (+13-26%) |
| kL2KSplit=12 | BLOCK_N=128 not divisible by 12 | build failure |
| EPI_THREADS=256 | `WG_BLOCK_M` static_assert fails (requires multi-warpgroup epi) | build failure |
| Swap AB on MX-FP8 alone | 4 overheads (SF + cross-warp + amax + strided writes) × no PT win | measured +33-54% |
| 2-CTA UMMA multicast on current topology | DeepGEMM `2x1SM_SS` is 2-CTA only, incompatible with cluster=8 K-split | API analysis |
| `__nanosleep` spin inside barriers | no measurable diff | Step 30 |
| `st.release.cluster` per slot-write | serializes 16-wide epi, 2.5× slower | Step 32 |
| Warps arriving on cluster barriers with early-exit | confuses scheduler, +1% slower | Step 33 |

---

## 8. Next Directions (v4 candidates)

v3.0 sits at 9.28 µs @ M=32, which is **8.0× theoretical HBM floor** (1.16 µs).

### Candidate A: DSMEM intermediate (est. −1 to −2 µs, target ≤ 8 µs)

Skip the HBM round-trip for intermediate. L1 writes intermediate to cluster
distributed shared memory (DSMEM, 228KB × 8 CTAs = ~1.8MB per cluster — sufficient
for `kIntermediate × kMaxValidM × 1B = 96KB`). L2 reads from DSMEM directly.

Savings: eliminates L1 TMA store to workspace, eliminates L2 TMA load from
workspace. Net HBM bandwidth saved per kernel: `2 × 32 valid × 3072 = ~200KB`
per cluster × 8 clusters = 1.6 MB / 8 TB/s ≈ 0.2 µs. **BUT**, what's actually
saved is the HBM latency wait (the `l1_to_l2_sync` can potentially shorten if
DSMEM writes are visible faster within cluster than HBM).

Implementation complexity: medium-high (new DSMEM addressing, cluster rank
mapping, possible re-design of L1 N-split).

### Candidate B: Packed L1 byte writes (est. −0.2 µs)

L1 epi currently writes 1 FP8 byte per lane per m (64 writer lanes × 32 m = 2KB
per N-tile, with 4-way SMEM bank conflicts). Pack 4 bytes from 4 adjacent lanes
via `__shfl_sync` → 1 uint32 store per quad. Saves ~75% of SMEM store issue time.

Low effort, low confidence (might not be on critical path).

### Candidate C: kNumStages=5 on v3 — TESTED, marginal win (-0.03 µs)

v3 saves 4KB more SMEM than v2 (no SF region, no exchange buffer). STAGES=5 finally
fits without scheduler confusion (v2 was +3% slower with stages=5):

| STAGES | M=1 | M=8 | M=32 |
|--------|-------|-------|-------|
| 4 (v3 canonical) | 9.290 | 9.311 | 9.279 |
| **5** (v3.1 micro-win) | **9.247** | **9.273** | 9.262 |
| 6 | 9.251 | 9.276 | 9.276 |

v3.1 = v3 + STAGES=5 → **9.25 µs @ M=1** (additional −0.5% vs v3.0). STAGES=6 no further improvement.
Recommended as new default.

### Candidate D: Cross-token persistent kernel (est. −2 µs)

Amortize kernel launch overhead across multiple decode tokens by running a
persistent kernel that polls for new X inputs. Requires host-device signaling
protocol. High implementation cost, medium confidence.

### Candidate E: Smaller UMMA_N (est. 0 µs but cleaner)

For strict M=1 deployment, UMMA_N=8 is the minimum-supported shape. TMEM 2×
smaller, but likely no wall-time win (kernel is barrier-bound, not TMEM-bound).
Worth testing if M=1-only variant is deployed.

### Recommendation (post ncu profile, see Section 9)

**Stop at v3.1**. Profile confirms barrier-bound at 92.7% no-eligible-warp.
Compute (1.16%) and DRAM (7.45%) are far from saturated — there's nothing left
to feed. To break the barrier wall would require persistent kernel or a
fundamental topology change (DSMEM analyzed infeasible per Section 8).

---

## 9. ncu Profile Analysis (v3.1 @ M=32, on B200)

### 9.1 Speed of Light overview

| Subsystem | Throughput @ peak |
|-----------|-------------------|
| Memory | 8.32% |
| **DRAM** | **7.45%** (9.51 MB read = exact W1+W2 size, 0 written) |
| L2 Cache | 8.32% |
| L1/TEX | 14.26% |
| **Compute (SM)** | **3.36%** |
| **Tensor Core pipeline** | **1.16%** |
| **Eligible warps per active cycle** | **0.07** (out of max 1.0) |
| **% cycles with No Eligible** | **92.74%** |

**Verdict**: kernel is **scheduling-bound**, not compute or memory bound.
Every subsystem is sitting idle waiting for warps to issue.

### 9.2 Warp stall breakdown (% of issued cycle stalls)

Stall breakdown for v3.1 vs v2 PT (both @ M=32):

| Stall type | v2 PT | **v3.1** | Δ (v3 wins if ↓) |
|------------|-------|----------|-------------------|
| **barrier** (`bar.sync`) | 1219.71 | **1102.85** | **-117 ↓** |
| long_scoreboard (memory wait) | 749.54 | 797.18 | +48 |
| membar (`fence.acq_rel.cluster`) | 232.23 | **191.93** | **-40 ↓** |
| wait | 168.43 | 192.11 | +24 |
| no_instruction | 163.91 | 162.79 | -1 |
| short_scoreboard | (n/a) | 74.60 | — |
| tensor cycles active | 4.32% | **1.16%** (smaller MMA per fma) | — |

(Percentages can sum >100% because each stall cycle may count multiple reasons.)

**v3.1 reduced barrier stalls by ~10%** vs v2 PT — pre-merged W1 + slot layout
changes really did remove sync overhead. But barrier still dominates by 5×.

### 9.3 PC sampling (which stall reason hits the most)

| Stall reason | Samples | % of total |
|--------------|---------|------------|
| **barrier** | 120 | **57.4%** |
| long_scoreboard | 41 | 19.6% |
| membar | 20 | 9.6% |
| wait | 18 | 8.6% |
| short_scoreboard | 8 | 3.8% |

**57% of stall samples are CTA barrier waits** (`bar.sync` instructions, primarily
in `sync_aligned` calls within L1/L2 epi and `l1_to_l2_sync` grid_sync).

ncu's own recommendation: "On average, each warp spends 11.1 cycles being stalled
waiting for sibling warps at a CTA barrier... about 40.6% of the total average of
27.4 cycles between issuing two instructions."

### 9.4 Where are the barriers?

In v3.1's L1 epi loop (per N-tile, per CTA, ~6 N-tiles per CTA):
1. `tma_store_wait` + `sync_aligned` (Phase 2.5 — wait SMEM CD double-buffer free)
2. `tma_store_fence` + `sync_aligned` (Phase 4 — fence + sync before TMA store launch)

= 2 sync_aligned per L1 N-tile × 6 = **12 sync_aligned per CTA in L1 phase**.

Plus `l1_to_l2_sync` (called twice per kernel — once each from TMA-B and main path):
3. 2× `sync_aligned` for grid_sync (before & after atomic-counter spin)

Plus L2 phase (per N-tile, kL2KSplit=8 path):
4. Cluster_sync via `fence.acq_rel.cluster + cluster_arrive_relaxed + cluster_wait`
   (this is membar+wait, not bar.sync — counts in membar/wait stalls)

Total bar.sync per CTA per kernel: ~12 (L1) + 2 (grid_sync) = **~14 bar.sync** per CTA.

At ~11.1 cycles wasted per warp per barrier, and ~30+ cycles per kernel for these
14 barriers, the barrier overhead is structural and cannot be optimized away
without changing topology.

### 9.5 What this implies for v4

The profile rules out several proposed v4 directions:
- **Packed L1 byte writes** would reduce SMEM stores, but SMEM is at 14% throughput.
  No win expected.
- **Larger UMMA fma (e.g. UMMA_N=64)** would push tensor cores even further (still <2%);
  doesn't help.
- **Reduce TMA frequency / batched loads**: DRAM is at 7%, plenty of headroom.

### 9.6 Failed v3.2 experiment: TMA-A skip from grid_sync

**Hypothesis**: in v3, TMA-A loads W (W1 then W2), neither depends on L1 output.
By analogy with v1's Step 18-19 optimization (TMA-B skip, since TMA-B in v1 loaded W),
TMA-A in v3 should also be excluded from `l1_to_l2_sync`. Expected saving: ~0.1-0.3 µs.

**Implementation**: removed `l1_to_l2_sync()` call from TMA-A warp; lowered
`kGridSyncThreads` from 224 → 192 (also exclude TMA-A's 32 threads).

**Initial result**: latency dropped 9.25 → **7.74 µs** (−16%). Too good to be true.

**Catch**: correctness broken (`max|Δ|=263` on final y). Root cause:
`thread_idx == 0` was the atomic master for the global counter — and thread 0 lives
in TMA-A warp. With TMA-A skipping, the atomic was never executed by any thread,
counter never incremented, and grid sync became a no-op. CTAs raced through L2
without waiting for L1 to globally complete.

**Atomic-master fix**: moved master to `thread_idx == kNumNonEpilogueThreads`
(first epi thread). Re-tested; correctness restored. Latency: **9.24 µs at M=1**
— back to v3.1 baseline. **No real saving.**

### 9.7 Critical insight from the failed experiment

The "false speedup" (9.25 → 7.74 µs) reveals the **actual wait time of `l1_to_l2_sync`
is ~1.5 µs**, dominated NOT by `bar.sync` but by the **atomic spin** waiting for the
slowest CTA's L1 to globally signal completion.

After fixing atomic master placement (move from `thread_idx==0` in TMA-A warp to
`thread_idx==kNumNonEpilogueThreads` in epi), the TMA-A skip becomes correct. See
section 10 below for the corrected v3.2 result.

### 9.8 v3.1 was NOT the wall — corrected v3.2/3.3/3.4 break it (single-kernel)

I had concluded "v3.1 is the wall" prematurely. After re-examining the failed
experiment with proper atomic-master placement, three more single-kernel optimizations
succeeded:

| Version | M=1 | M=8 | M=32 | Δ vs v3.0 |
|---------|------|------|------|-----------|
| v3.0 baseline | 9.29 | 9.31 | 9.28 | — |
| v3.1 STAGES=5 (later reversed) | 9.25 | 9.27 | 9.26 | -0.04 |
| **v3.2** TMA-A skip + atomic→epi | **9.03** | **9.05** | **9.04** | **-0.26** |
| v3.3 drop redundant L1 sync_aligned | 9.02 | 9.05 | 9.04 | -0.27 |
| **v3.4** TMEM_LOAD 32x atom (single call) | **8.96** | **8.98** | **8.98** | **-0.32** ✅ |

(v3.4 reverts STAGES back to 4; TMEM 32x changed reg pressure such that STAGES=5 no longer wins.)

**v3.2 win mechanism (-0.21µs)**: TMA-A loads W2 in L2 phase. W2 doesn't depend on L1
output. By skipping `l1_to_l2_sync` for TMA-A, it can pre-load W2 K-pipeline stages
WHILE TMA-B (intermediate, L1-dependent) is still spinning on the global atomic. By
the time TMA-B's first K-block arrives, W2 is already staged → MMA starts immediately.
Without skip: TMA-A and TMA-B both wait at grid_sync, then race to load K-block 0;
TMA-A is the slower one (16KB vs 4KB), MMA waits for TMA-A.

**Critical bug fix for v3.2**: original v3 had atomic master at `thread_idx == 0`,
which is in TMA-A warp. Skipping TMA-A meant atomic never fired → grid_sync became
no-op → CTAs raced. Initial broken test showed "9.25→7.74µs" (false speedup of 1.5µs
from broken sync). Moving atomic master to `thread_idx == kNumNonEpilogueThreads`
(first epi thread) decouples it from TMA-A's participation; correctness restored.

**v3.4 win mechanism (-0.05µs additional)**: previously L1 + L2 epi each used 4× of
`SM100_TMEM_LOAD_32dp32b8x` (8 cols per call) per warp per N-tile. SM100 supports
up to 32dp32b128x. With UMMA_N=32, single call of `SM100_TMEM_LOAD_32dp32b32x`
covers all data. ncu confirms long_scoreboard stalls dropped 797% → 740% (-57%).

**v3.3 (drop sync_aligned)**: marginal (~0.01µs). The first sync_aligned after
`tma_store_wait` was redundant — each warp's `tma_store_wait` is per-thread, and
warps don't conflict on SMEM CD writes (different output_n slices). Sync was
overkill.

### 9.9 Updated v3.4 ncu profile (the new ceiling)

| Metric | v3.1 | v3.4 | Δ |
|--------|------|------|---|
| barrier stall % | 1102.85 | 1093.80 | -9 (noise) |
| long_scoreboard % | 797.18 | **740.41** | **-57 ↓** (TMEM 32x consolidation) |
| membar % | 191.93 | 187.11 | -5 |
| wait % | 192.11 | 187.92 | -4 |
| no_instruction % | 162.79 | 191.06 | +28 (more idle time, smaller compute) |
| tensor cycles | 1.16% | 1.20% | +0.04 |
| eligible warps/cycle | 0.07 | 0.08 | +0.01 |

Barrier stall is essentially unchanged (which is what we expected — bar.sync time
is dominated by the atomic spin, not participant count). The win came from
**reducing memory wait** (long_scoreboard) via TMEM consolidation.

### 9.10 Lesson learned from this iteration

**Never declare "the wall" without exhausting micro-optimizations.** Several ideas
I had analytically dismissed yielded real wins:
- TMA-A skip — I claimed "no benefit because MMA still waits for TMA-B". WRONG —
  TMA-A pre-loading lets MMA start sooner per K-block, since K-pipeline depth means
  TMA-A is ahead by the time TMA-B catches up.
- Larger TMEM atom — I dismissed as "compute is at 1.16%, memory not bottleneck".
  WRONG — long_scoreboard (chained TMEM loads) was a real cost that consolidation
  fixed.

**Methodology fix**: 30-iter quick tests are too noisy to detect 0.05-0.2µs
improvements. Always validate with 1500-iter × 3 reps before drawing conclusions.

### 9.11 Updated remaining directions

After v3.4, the single-kernel ceiling is **8.96 µs @ M=1**, which is:
- **-19.5%** vs v1 Step 29 MX-FP8 (11.13 µs)
- **-42%** vs cuBLAS BF16 (15.6 µs)
- **7.7× theoretical HBM floor** (1.16 µs)

What's left at single-kernel level (untested):
- Pack L1 byte writes via warp shfl — analyzed, likely break-even (shfl cost = bank conflict cost)
- Reduce register pressure to 168 → 2 blocks/SM — but our 64 CTAs only use 32 SMs anyway, no occupancy benefit
- Polynomial silu approximation — already use __expf hardware accel, marginal

Out-of-scope (not single-kernel):
- **Persistent kernel** (eliminates launch + grid_sync overhead across decode tokens). Est: -1 to -2µs. Impl: 1-2 weeks.
- **Cross-layer fusion** (FFN + attention together in one kernel). Different research direction.

---

## 10. File Manifest

```
deep_gemm/include/deep_gemm/impls/
├── sm100_fp8_mega_ffn.cuh          # v1 = Step 29 MX-FP8 (11.02 µs @ M=1)
├── sm100_fp8_mega_ffn_pt.cuh       # v2 = PT (9.52 µs @ M=1)
├── sm100_fp8_mega_ffn_v3.cuh       # v3 = swap+PT+premerged+slot (9.28 µs @ M=32)  <-- ceiling
├── sm100_fp8_mega_ffn_swap.cuh     # failed swap MX-FP8 attempt (kept for reference)

tests/cpp/
├── test_sm100_fp8_mega_ffn.cu          # v1 test
├── test_sm100_fp8_mega_ffn_pt.cu       # v2 test
├── test_sm100_fp8_mega_ffn_v3.cu       # v3 test (with permute_w1_premerged helper)
├── test_sm100_fp8_mega_ffn_swap.cu     # failed swap test
├── build_mega_ffn.sh                   # v1 build
├── build_mega_ffn_pt.sh                # v2 build
├── build_mega_ffn_v3.sh                # v3 build
```

All v1 files are committed on branch `feature/dev_mega_ffn`. v2/v3 files are
currently uncommitted working-tree state on host `10.77.188.34`.

---

## Appendix A: Canonical Runtime Configuration

```cpp
constexpr uint32_t kNumStages        = 4;      // K-pipeline depth
constexpr uint32_t kNumNonEpiThreads = 128;    // 4 warps (TMA-A, TMA-B, MMA, cold)
constexpr uint32_t kNumEpiThreads    = 128;    // 4 warps
constexpr uint32_t kClusterDim       = 8;      // matches kL2KSplit for cluster_sync
constexpr uint32_t kL2KSplit         = 8;      // kL2KBlocks (24) / 8 = 3 K-blocks per CTA
// Total: 8 × 8 = 64 CTAs, 8 clusters × 8 CTAs each.

constexpr uint32_t kL1OutputBlocksN   = (2 * kIntermediate) / BLOCK_N;  // 48
constexpr uint32_t kL2OutputBlocksN   = kHidden / BLOCK_N;              // 8
constexpr uint32_t kL1KBlocks         = kHidden / BLOCK_K;              // 8
constexpr uint32_t kL2KBlocks         = kIntermediate / BLOCK_K;        // 24
constexpr uint32_t kL2KBlocksPerCta   = kL2KBlocks / kL2KSplit;         // 3
constexpr uint32_t kL1NPerCtaCeil     = (48 + 64 - 1) / 64;             // 1 (first 48 CTAs do L1)
```

Launch attributes (all three versions):
```cpp
attr[0] = ClusterDimension (8, 1, 1)
attr[1] = AccessPolicyWindow (d_w1, bytes_w1, persisting)
attr[2] = ProgrammaticStreamSerialization (1)
attr[3] = ClusterSchedulingPolicyPreference (LoadBalancing)
```

L2 carveout: `cudaLimitPersistingL2CacheSize = 16 MB` (covers W1+W2 < 9 MB).

---

## Appendix B: v3 Per-Tensor Scale Derivation (host-side)

```cpp
// Per-tensor FP8 quantize: single scale per tensor, no UE8M0 SF.
float quantize_fp8_pt(const float* src, size_t n, std::vector<uint8_t>& out) {
    float amax = max_abs(src, n);
    float scale = amax > 0 ? amax / 448.f : 1.f;   // FP8 e4m3 max = 448
    float inv = 1.f / scale;
    for (size_t i = 0; i < n; ++i)
        out[i] = float_to_fp8_e4m3(src[i] * inv);
    return scale;
}

// CPU reference computes Linear1 + SwiGLU, derives scale_intermediate from amax.
float scale_X  = quantize_fp8_pt(X_fp32, M*H, x_fp8);
float scale_W1 = quantize_fp8_pt(W1_fp32_premerged, 2I*H, w1_fp8);  // pre-merged layout!
float scale_W2 = quantize_fp8_pt(W2_fp32, H*I, w2_fp8);
auto interm_fp32_cpu = cpu_linear1_swiglu(x_fp8, scale_X, w1_fp8, scale_W1);
float scale_intermediate = amax(interm_fp32_cpu) / 448;

// Kernel takes 3 fused scales (constant-folded):
float scale_xw1              = scale_X * scale_W1;
float scale_inv_intermediate = 1.f / scale_intermediate;
float scale_iw2              = scale_intermediate * scale_W2;
```

In production, `scale_X` is per-request (recomputed each forward), `scale_W1/W2`
are model constants, `scale_intermediate` is a calibrated constant.

---

## Appendix C: Bench Protocol

Run on host `10.77.188.34`, docker container `docker-env-lmdeploy-xx`,
`CUDA_VISIBLE_DEVICES=3` (single B200 GPU).

```bash
cd /home/workcode/DeepGEMM
# Build with canonical config (defaults)
CUDA_HOME=/usr/local/cuda \
CUTLASS_HOME=/home/workcode/DeepGEMM/third-party/cutlass \
  bash tests/cpp/build_mega_ffn_v3.sh

# 1500 iters × 3 reps at each M
cd tests/cpp
for i in 1 2 3; do
    for M in 1 8 32; do
        CUDA_VISIBLE_DEVICES=3 ./test_mega_ffn_v3 $M 1500 2>&1 | grep "avg latency"
    done
    echo ---
done
```

Expected output (v3.0):
```
avg latency = 9.29 µs  (M=1)
avg latency = 9.31 µs  (M=8)
avg latency = 9.28 µs  (M=32)
```

Variance across 3 reps: typically ≤ 0.05 µs with PDL + L2 persistence.
