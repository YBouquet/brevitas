# 2DP vs Hadamard: MXFP4 attention kernels on gfx950

These kernels and benchmarks time the MXFP4 attention path on an AMD MI350X/MI355X (gfx950):
QKV projection → RoPE → transform → MXFP4 → attention. They compare 2DP with the frozen plane and the
folded unshrink against the standard Hadamard pipeline.

## Kernels: which one to read

Three attention kernels were measured. Only the full flash-attention kernel is the real operator a model runs;
the other two are **partial kernels**, kept because they isolate one part of the cost.

| kernel | computes | writes | status |
|---|---|---|---|
| **Full flash attention** ([section](#full-flash-attention-the-real-kernel)) | QK^T → online softmax → P.V → O / l | O fp16 [T, n_q, D] + LSE | **the real kernel: the reference result** |
| Partial kernel 1: logit store (`p2_qk.py`, [results](#results-partial-kernel-1-logit-store)) | QK^T only | the full T x T fp32 logits | partial: bound by the T² store, so the rank-2 term looks almost free |
| Partial kernel 2: QK^T + online softmax (`p2_fa_stats.py`, [results](#partial-kernel-2-qkt--online-softmax-no-pv-no-output)) | QK^T + row max / row sum | m, l per row | partial: no P.V, so the rank-2 term's share of the work is overstated |

Full flash attention, MI350X, fastest vs fastest within one job. Each cell is the median [min, max] of the per-repeat
ratio over 5 interleaved repeats; — = not measured.
- 2DP-MFMA vs Had: the rank-2 term on MFMA, jobs 92366 (1B; T = 1024: job 92270) and 92099 (8B).
- 2DP-`v_dot2` vs Had: the rank-2 term on the vector unit (`vp=4`), against the Hadamard of the same job, jobs
  92415 (1B) and 92416 (8B). See [MFMA vs `v_dot2`](#rank-2-term-mfma-vs-v_dot2-full-flash-attention).
- fp16 / MXFP4: speedup of the MXFP4 FA kernel over fp16 flash attention (same kernel with an fp16 QK^T, same toolchain
  and search), FA kernel only, bench timing, strict spill policy, jobs 92428 (1B) and 92429 (8B); 2DP is the MFMA form. See
  [details](#mxfp4-vs-fp16-flash-attention).

Absolute times (FA kernel, front end, end to end) are in the per-section tables.

| model | T | FA: 2DP-MFMA vs Had | e2e: 2DP-MFMA vs Had | FA: 2DP-`v_dot2` vs Had | e2e: 2DP-`v_dot2` vs Had | fp16 / MXFP4 Had | fp16 / MXFP4 2DP-MFMA |
|---|---|---|---|---|---|---|---|
| 1B | 1024 | +6.7 % [+5.2, +8.1] | +1.5 % [+1.4, +1.6] | — | — | — | — |
| 1B | 2048 | +5.6 % [+5.5, +5.7] | −1.8 % [−2.0, −1.7] | +17.7 % [+17.4, +18.3] | +1.7 % [+1.6, +2.2] | 1.28x [1.17, 1.33] | 1.19x [1.19, 1.20] |
| 1B | 4096 | +6.0 % [+5.9, +6.4] | −2.9 % [−2.9, −2.8] | +24.5 % [+24.2, +24.7] | +8.9 % [+8.9, +9.0] | 1.30x [1.30, 1.30] | 1.22x [1.22, 1.22] |
| 1B | 8192 | +7.2 % [+7.0, +8.1] | +0.9 % [+0.8, +1.0] | +24.0 % [+23.4, +24.3] | +13.7 % [+13.3, +14.0] | 1.33x [1.32, 1.34] | 1.22x [1.21, 1.25] |
| 1B | 16384 | +6.8 % [+6.1, +7.6] | +4.2 % [+3.6, +5.0] | +22.6 % [+21.6, +23.2] | +17.2 % [+16.7, +17.4] | 1.33x [1.32, 1.34] | 1.24x [1.23, 1.25] |
| 8B | 2048 | — | — | +12.4 % [+12.2, +12.7] | −3.4 % [−3.6, −3.2] | 1.29x [1.28, 1.32] | 1.27x [1.25, 1.27] |
| 8B | 4096 | +4.2 % [+3.9, +4.5] | −4.6 % [−4.7, −4.3] | +13.1 % [+12.8, +13.3] | −0.0 % [−0.2, +0.1] | 1.33x [1.32, 1.34] | 1.29x [1.29, 1.30] |
| 8B | 8192 | +4.8 % [+4.0, +6.1] | −0.5 % [−1.7, +0.3] | +11.6 % [+11.0, +15.0] | +4.2 % [+3.7, +5.6] | 1.34x [1.32, 1.34] | 1.29x [1.28, 1.29] |
| 8B | 16384 | +4.2 % [+3.1, +4.9] | +0.8 % [+0.3, +1.3] | +13.6 % [+13.2, +14.0] | +9.4 % [+8.5, +9.7] | 1.32x [1.31, 1.33] | 1.28x [1.28, 1.29] |

| file | content |
|---|---|
| `p1_2dp.py` | P1 (Gluon): 8-wave ping-pong projection GEMM with the 2DP epilogue: RoPE, plane coordinates `a' = (1-gamma) a`, MXFP4 of the residual. `epi=0` is a plain fp16 store, used as the Hadamard projection |
| `p2_qk.py` | P2, **partial kernel 1** (Gluon): fp4 QK^T (`mfma_scaled`), plus the 2DP rank-2 term `a'.b` (`corr=1`); `corr=0` is the bare QK^T. One program computes `NL` consecutive logit tiles: the Q side is loaded once, and the next tile's K side is loaded before the current tile is stored |
| `had_j2.py` | Hadamard quantizer (Triton): RoPE + `H_D / sqrt(D)` + MXFP4 |
| `layout_2dp.py` | offline: weight column permutation, RoPE tables, per-tile plane tables |
| `reference.py` | torch fp32 references for the correctness check |
| `bench_2dp_vs_hadamard.py` | the benchmark (partial kernel 1) |
| `bench_rank2_variants.py` | the three rank-2 variants against Hadamard at one (model, T): correctness, end-to-end tile sweep, interleaved repeats |
| `p2_fa_stats.py` | P2, **partial kernel 2** (Gluon): fp4 QK^T (+ rank-2 term) consumed by an online softmax; only the row max `m` and row sum `l` are written, no logits |
| `bench_fa_proxy.py` | the benchmark with `p2_fa_stats` as P2 (partial kernel 2) |

```
2DP       P1: x @ W -> RoPE -> a = U^T x_LF, r = x - U a -> MXFP4(r), a' = (1-gamma) a     one kernel
          P2: Q(r_q) Q(r_k)^T + a' b^T                                                    one kernel
Hadamard  J1: x @ W (fp16 Q/K)  ->  J2: RoPE + H_D/sqrt(D) + MXFP4  ->  P2: Q(z_q) Q(z_k)^T
```

## Output scaling

P2 returns `(1-gamma) S / s`, with `s = 1/sqrt(D)`. The flash attention that consumes it uses
`s / (1-gamma)` as its softmax scale. This is exactly `--dc-correction unshrink --dc-gamma gamma`.

## Requirements

- A gfx950 GPU.
- Triton built at tag `gfx950-tutorial-v1.0`, which provides the Gluon gfx950 support.
- `triton` imported before `torch`. The benchmark does this.

## Run

```bash
cd src/brevitas_examples/papers/2dp/kernels
python bench_2dp_vs_hadamard.py --model 1b
python bench_2dp_vs_hadamard.py --model 8b --u-path U_selfgen_8b_2dp_allrand_s0.pt --u-layer 15
```

| option | effect |
|---|---|
| `--model` | `1b` = Llama-3.2-1B shapes (head_dim 64), `8b` = Llama-3.1-8B shapes (head_dim 128); 32 Q / 8 KV heads |
| `--ts` | sequence lengths |
| `--u-path` | a plane saved with `--dc-u-path`. Without it, a random orthonormal plane is used; latency does not depend on its values |
| `--out` | write a JSON of the results |

Activations and weights are random fp16. At the first T, both pipelines are checked against the torch
references: fp4 code mismatch, and logits relative L2 (about 1e-2, from rounding ties).

Timing uses `triton.testing.do_bench` (warmup 25 ms, timing budget 100 ms). It flushes L2 before every run
and returns the median, so the number of timed runs depends on the kernel's duration: about 45 at 2 ms,
several hundred at 0.1 ms.

- **Tile selection.** For each pipeline, the QK^T variant and tile `(BM, BN, warps, WM, WN, NL)` are picked on
  the end-to-end chain (projection → QK^T), from the same candidates (`p2_qk.tiles`, about 100 tilings). A QK^T kernel timed alone after the L2 flush does not
  rank the tiles as they run behind the front end. Picking tiles alone flipped the 8B T = 8192 result
  between jobs.
- **Repeats.** The two pipelines and their front ends are then timed in `--repeats` interleaved repeats
  (default 5). The table gives the median and the range of the per-repeat ratio.

## Results: partial kernel 1, logit store

*Partial kernel: QK^T only, the T x T fp32 logits are written. Not the attention operator; see
[Full flash attention](#full-flash-attention-the-real-kernel) for the real kernel.*

MI350X, in µs. Each time is the median over 3 separate jobs of each job's value (end to end: the median of 5
interleaved repeats; front end and QK^T inside the chain: the job's median). Each ratio column is one job: median
[min, max] of the per-repeat 2DP vs Hadamard end-to-end ratio. Jobs 91054 / 91056 / 91058 (1B) and 91053 / 91055 /
91057 (8B); bold: 2DP faster end to end.

| model | T | e2e 2DP (µs) | e2e Had (µs) | front end 2DP (µs) | front end Had (µs) | QK^T in chain 2DP (µs) | QK^T in chain Had (µs) | 2DP vs Had, job 1 | job 2 | job 3 |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | 1024 | **85.8** | 88.0 | 52.3 | 53.8 | 33.5 | 34.1 | −2.6 % [−2.7, −2.5] | −2.2 % [−2.4, −2.2] | −2.6 % [−2.7, −2.5] |
| Llama-3.2-1B | 2048 | **168.6** | 172.2 | 52.0 | 56.7 | 116.4 | 115.5 | −2.2 % [−2.3, −2.1] | −2.0 % [−2.5, −2.0] | −2.1 % [−2.3, −1.7] |
| Llama-3.2-1B | 4096 | **497.2** | 500.9 | 58.4 | 71.2 | 438.3 | 429.7 | −0.9 % [−0.9, −0.8] | −0.8 % [−0.8, −0.7] | −0.5 % [−0.6, −0.5] |
| Llama-3.2-1B | 8192 | 1912.3 | 1909.8 | 112.3 | 147.1 | 1800.0 | 1764.7 | −0.1 % [−0.2, −0.1] | +0.2 % [+0.1, +0.2] | +0.2 % [+0.2, +0.3] |
| Llama-3.1-8B | 1024 | **129.2** | 134.0 | 93.4 | 100.4 | 35.6 | 33.8 | −3.8 % [−3.8, −3.7] | −3.8 % [−3.9, −3.7] | −3.2 % [−3.3, −3.1] |
| Llama-3.1-8B | 2048 | **225.7** | 232.8 | 109.2 | 124.6 | 116.5 | 108.2 | −3.2 % [−3.4, −2.9] | −3.3 % [−3.4, −3.2] | −3.1 % [−3.2, −3.0] |
| Llama-3.1-8B | 4096 | **657.6** | 673.1 | 216.9 | 248.0 | 440.8 | 425.4 | −2.3 % [−2.4, −2.2] | −2.5 % [−2.6, −2.4] | −2.5 % [−2.6, −2.5] |
| Llama-3.1-8B | 8192 | **2177.1** | 2187.9 | 383.8 | 442.6 | 1793.6 | 1745.2 | −0.9 % [−1.1, −0.8] | −0.6 % [−0.8, −0.6] | −0.4 % [−0.6, −0.3] |

- **Front end.** The 2DP P1 is 3–24 % faster than Hadamard's J1 + J2, because RoPE, the plane and MXFP4 all
  run in the GEMM epilogue.
- **QK^T.** The rank-2 term costs 0–8 % over Hadamard's QK^T, measured inside the chain. This cost grows
  with T², while the front-end saving grows with T. At T = 8192, the two roughly cancel.
- **Large T.** Both pipelines are bound by the fp32 logit store. At T = 8192 the result moves by about
  ±1 % between jobs, so 1B is a tie there and 8B a small win.
- **The `NL` loop** brought 8B T = 8192 from +2.4 % (one tile per program) to −0.4 … −0.9 %; the best 2DP
  tile there is (256, 64, 4, 4, 1, NL = 2).

## The rank-2 term: MFMA, FMA or `v_dot2`

The three variants compute the same rank-2 term on different units. `v_dot2` is THE vector-unit (VALU)
approach; the 2-FMA variant is kept for reference only. The MFMA variant is kept for the MFMA vs `v_dot2`
comparison. The measurements in this section are on the partial kernels.

The fp4 product runs on the matrix core (`mfma_scaled`) in both pipelines. 2DP adds `a'.b`, a K = 2 product
per logit. `p2_qk.qk(corr=1, corr_mfma=...)` computes it in one of three ways:

| `corr_mfma` | variant | per logit | coefficient loads |
|---|---|---|---|
| 1 | MFMA | none on the VALU; one fp16 `32x32x16` MFMA per 32x32 sub-tile (`a'`, `b` padded with zeros from K = 2 to 16), accumulating into the fp4 accumulator | padded `[BM, 16]`, `[16, BN]` |
| 0 | FMA | 2 `v_fma_f32` on the accumulator's row and column slices | 4 fp16 loads, converted to fp32 |
| 2 | `v_dot2` | 1 `v_dot2_f32_f16`: `acc += a'0 b0 + a'1 b1` | one int32 (the packed fp16 pair P1 stores) per row and per column |

`v_dot2_f32_f16 D, A, B, C` computes `D = A.lo * B.lo + A.hi * B.hi + C`, with A and B fp16 pairs packed in
32 bits and C, D in fp32. Triton does not emit it, so `p2_qk.py` issues it through `inline_asm_elementwise`.
The fp16 products are exact in fp32. Against the FMA variant it differs by at most 7.6e-6 on logits up to 70,
with the same error against the reference.

T = 8192, each variant on its best end-to-end tile; median of 5 interleaved repeats in one job; vs Hadamard: median
[min, max] of the per-repeat end-to-end ratio (jobs 91172 / 91173, values from the job logs); the fastest 2DP variant
per model is in bold:

| model | method | best e2e tile | e2e (µs) | QK^T in chain (µs) | vs Had (e2e) |
|---|---|---|---|---|---|
| Llama-3.2-1B | Hadamard | (128, 64, 8, 4, 2, 1) | 1903.3 | 1758.0 | — |
| Llama-3.2-1B | MFMA | (128, 64, 4, 4, 1, 1) | 1911.4 | 1800.1 | +0.42 % [−3.58, +0.46] |
| Llama-3.2-1B | FMA | (256, 64, 8, 8, 1, 1) | **1903.6** | 1792.3 | **+0.01 % [−3.92, +0.18]** |
| Llama-3.2-1B | `v_dot2` | (64, 128, 4, 2, 2, 2) | 1922.3 | 1811.0 | +1.00 % [−2.88, +1.11] |
| Llama-3.1-8B | Hadamard | (128, 128, 4, 4, 1, 1) | 2176.6 | 1738.2 | — |
| Llama-3.1-8B | MFMA | (256, 64, 4, 4, 1, 2) | 2186.2 | 1805.5 | +0.44 % [+0.36, +0.56] |
| Llama-3.1-8B | FMA | (256, 128, 4, 4, 1, 1) | 2216.6 | 1835.9 | +1.84 % [+1.49, +1.95] |
| Llama-3.1-8B | `v_dot2` | (256, 128, 4, 4, 1, 1) | **2178.1** | 1797.4 | **+0.07 % [−0.08, +0.20]** |

- **No clear winner.**
  - At 8B, `v_dot2` beats the FMA variant (1797 vs 1836 µs QK^T inside the chain) and matches the MFMA one.
  - At 1B, it is the slowest of the three.
- **None of the three beats Hadamard in this job.** At T = 8192 the 2DP vs Hadamard result moves by about
  ±1 % between jobs: the same 8B MFMA configuration was −0.4 to −0.9 % in the 3 jobs of the results table
  above.
- **The remaining QK^T gap is not the rank-2 arithmetic.** At a matched tile, 2DP's QK^T costs the same as
  Hadamard's (1770 vs 1771 µs at 8B, tile (256, 64, 4, 4, 1, 2)). The gap comes from Hadamard's best tile,
  (128, 128, 4, 4, 1, 1) at 8B, at which every 2DP variant is slow.
- **Tuning.** `bench_2dp_vs_hadamard.py` tunes over all three variants end to end. The results table above
  was measured before `v_dot2` was added, over MFMA and FMA only.


### Over T: best MFMA, best FMA and best `v_dot2` against Hadamard

`bench_rank2_variants.py --endpoint {logits,fa}`: for each T, every variant and Hadamard get their own best
end-to-end tile; then the best chain of each in 5 interleaved repeats. One row per (T, method): end to end (front end +
P2) and front end in µs (median of the repeats; the three 2DP variants share the 2DP front end), and 2DP vs Hadamard end
to end, median [min, max] of the per-repeat ratio; the fastest 2DP variant per T is in bold. MI350X, one job per table.

**Partial kernel 1: logit store** (`p2_qk.qk`, T ≤ 8192). Logits rel L2 vs the torch target: 1.12e-2 (1B), 1.17e-2 (8B), the
same for the three variants.

Llama-3.2-1B (job 92106):

| T | method | best e2e tile | e2e (µs) | front end (µs) | vs Had (e2e) |
|---|---|---|---|---|---|
| 1024 | Hadamard | (256, 128, 4, 2, 2, 2) | 88.4 | 54.2 | — |
| 1024 | MFMA | (64, 64, 4, 2, 2, 1) | 86.6 | 52.1 | −2.1 % [−2.2, −1.9] |
| 1024 | FMA | (64, 64, 4, 2, 2, 1) | **86.4** | 52.1 | **−2.4 % [−2.4, −2.3]** |
| 1024 | `v_dot2` | (64, 64, 4, 2, 2, 1) | 86.4 | 52.1 | −2.3 % [−2.3, −2.2] |
| 2048 | Hadamard | (256, 128, 4, 2, 2, 4) | 173.7 | 57.6 | — |
| 2048 | MFMA | (128, 64, 8, 4, 2, 4) | **169.2** | 51.7 | **−2.5 % [−2.9, −2.3]** |
| 2048 | FMA | (128, 64, 8, 4, 2, 4) | 170.3 | 51.7 | −1.9 % [−2.1, −1.7] |
| 2048 | `v_dot2` | (64, 128, 4, 2, 2, 4) | 169.4 | 51.7 | −2.5 % [−2.7, −2.0] |
| 4096 | Hadamard | (64, 64, 4, 2, 2, 1) | 499.5 | 70.6 | — |
| 4096 | MFMA | (128, 64, 4, 4, 1, 1) | **497.5** | 57.1 | **−0.4 % [−0.6, −0.3]** |
| 4096 | FMA | (128, 64, 4, 4, 1, 1) | 503.6 | 57.1 | +0.8 % [+0.6, +0.9] |
| 4096 | `v_dot2` | (64, 128, 4, 1, 4, 2) | 506.8 | 57.1 | +1.4 % [+1.3, +1.6] |
| 8192 | Hadamard | (256, 128, 4, 2, 2, 2) | 1953.6 | 144.7 | — |
| 8192 | MFMA | (128, 128, 8, 4, 2, 1) | **1904.9** | 109.7 | **−2.5 % [−2.6, −2.5]** |
| 8192 | FMA | (256, 64, 8, 8, 1, 1) | 1926.8 | 109.7 | −1.4 % [−1.5, −1.3] |
| 8192 | `v_dot2` | (64, 128, 4, 2, 2, 2) | 1924.2 | 109.7 | −1.5 % [−1.6, −1.5] |

Llama-3.1-8B (job 92108):

| T | method | best e2e tile | e2e (µs) | front end (µs) | vs Had (e2e) |
|---|---|---|---|---|---|
| 1024 | Hadamard | (64, 64, 4, 2, 2, 2) | 134.9 | 96.8 | — |
| 1024 | MFMA | (128, 64, 4, 4, 1, 1) | 129.8 | 91.2 | −3.8 % [−3.9, −3.7] |
| 1024 | FMA | (128, 64, 4, 4, 1, 1) | **129.7** | 91.2 | **−3.9 % [−4.1, −3.9]** |
| 1024 | `v_dot2` | (128, 64, 4, 4, 1, 1) | 129.9 | 91.2 | −3.7 % [−3.8, −3.6] |
| 2048 | Hadamard | (128, 128, 4, 2, 2, 2) | 232.6 | 124.9 | — |
| 2048 | MFMA | (256, 128, 4, 2, 2, 1) | **227.9** | 109.1 | **−2.0 % [−2.1, −1.9]** |
| 2048 | FMA | (128, 256, 4, 2, 2, 1) | 233.2 | 109.1 | +0.3 % [+0.1, +0.4] |
| 2048 | `v_dot2` | (256, 128, 4, 4, 1, 1) | 231.0 | 109.1 | −0.7 % [−0.7, −0.6] |
| 4096 | Hadamard | (128, 64, 4, 2, 2, 2) | 671.2 | 246.8 | — |
| 4096 | MFMA | (256, 64, 4, 4, 1, 2) | **657.0** | 215.4 | **−2.1 % [−2.4, −2.0]** |
| 4096 | FMA | (256, 128, 4, 4, 1, 1) | 672.7 | 215.4 | +0.2 % [+0.0, +0.3] |
| 4096 | `v_dot2` | (256, 128, 4, 4, 1, 1) | 668.7 | 215.4 | −0.3 % [−0.5, −0.1] |
| 8192 | Hadamard | (128, 128, 4, 4, 1, 1) | 2186.2 | 442.0 | — |
| 8192 | MFMA | (256, 64, 4, 4, 1, 2) | **2179.2** | 383.4 | **−0.4 % [−0.4, −0.3]** |
| 8192 | FMA | (256, 128, 4, 4, 1, 1) | 2216.5 | 383.4 | +1.3 % [+1.2, +1.6] |
| 8192 | `v_dot2` | (256, 128, 4, 4, 1, 1) | 2188.1 | 383.4 | +0.1 % [−0.0, +0.1] |

**Partial kernel 2: QK^T + online softmax** (`p2_fa_stats.qk_stats`, no P.V, no output).
`m` / `l` vs torch: max abs 1e-6 / max rel 7e-7 for every variant.

Llama-3.2-1B (job 92107):

| T | method | best e2e tile | e2e (µs) | front end (µs) | vs Had (e2e) |
|---|---|---|---|---|---|
| 1024 | Hadamard | (128, 256, 4, 4, 1) | 71.4 | 53.8 | — |
| 1024 | MFMA | (128, 256, 4, 4, 1) | 76.9 | 52.0 | +7.7 % [+7.2, +7.8] |
| 1024 | FMA | (128, 128, 4, 4, 1) | 77.6 | 52.0 | +8.5 % [+8.4, +8.9] |
| 1024 | `v_dot2` | (128, 256, 4, 4, 1) | **70.9** | 52.0 | **−0.8 % [−1.1, −0.7]** |
| 2048 | Hadamard | (128, 256, 4, 4, 1) | 106.5 | 57.1 | — |
| 2048 | MFMA | (256, 128, 8, 8, 1) | 128.4 | 52.0 | +20.5 % [+20.4, +20.6] |
| 2048 | FMA | (256, 128, 8, 8, 1) | 134.2 | 52.0 | +26.0 % [+26.0, +26.1] |
| 2048 | `v_dot2` | (256, 256, 8, 8, 1) | **109.1** | 52.0 | **+2.4 % [+2.3, +2.4]** |
| 4096 | Hadamard | (128, 256, 4, 4, 1) | 245.8 | 70.9 | — |
| 4096 | MFMA | (128, 128, 4, 4, 1) | 336.9 | 57.6 | +37.0 % [+36.7, +37.3] |
| 4096 | FMA | (256, 128, 8, 8, 1) | 370.4 | 57.6 | +50.5 % [+50.3, +51.0] |
| 4096 | `v_dot2` | (256, 256, 8, 8, 1) | **268.6** | 57.6 | **+9.2 % [+9.0, +9.5]** |
| 8192 | Hadamard | (128, 256, 4, 4, 1) | 850.7 | 145.6 | — |
| 8192 | MFMA | (128, 128, 4, 4, 1) | 1242.4 | 112.3 | +45.8 % [+45.6, +46.2] |
| 8192 | FMA | (256, 128, 8, 8, 1) | 1380.2 | 112.3 | +62.2 % [+61.3, +64.0] |
| 8192 | `v_dot2` | (256, 256, 8, 8, 1) | **986.1** | 112.3 | **+15.8 % [+15.0, +16.2]** |
| 16384 | Hadamard | (128, 256, 4, 4, 1) | 2958.2 | 267.6 | — |
| 16384 | MFMA | (128, 128, 4, 4, 1) | 4593.4 | 204.7 | +55.1 % [+48.7, +56.7] |
| 16384 | FMA | (256, 128, 8, 8, 1) | 5150.7 | 204.7 | +74.1 % [+66.6, +75.7] |
| 16384 | `v_dot2` | (256, 256, 8, 8, 1) | **3561.1** | 204.7 | **+20.8 % [+16.7, +21.4]** |

Llama-3.1-8B (job 92109):

| T | method | best e2e tile | e2e (µs) | front end (µs) | vs Had (e2e) |
|---|---|---|---|---|---|
| 1024 | Hadamard | (128, 256, 4, 4, 1) | 121.6 | 100.3 | — |
| 1024 | MFMA | (128, 256, 4, 4, 1) | 120.1 | 90.5 | −1.3 % [−2.0, −1.2] |
| 1024 | FMA | (128, 128, 4, 4, 1) | 118.4 | 90.5 | −2.7 % [−2.8, −2.3] |
| 1024 | `v_dot2` | (128, 256, 4, 4, 1) | **113.8** | 90.5 | **−6.4 % [−6.5, −6.2]** |
| 2048 | Hadamard | (256, 256, 8, 8, 1) | 176.0 | 120.3 | — |
| 2048 | MFMA | (256, 128, 8, 8, 1) | 191.7 | 106.9 | +8.5 % [+8.3, +9.0] |
| 2048 | FMA | (256, 128, 8, 8, 1) | 191.1 | 106.9 | +8.6 % [+8.0, +8.6] |
| 2048 | `v_dot2` | (256, 256, 8, 8, 1) | **170.6** | 106.9 | **−3.2 % [−3.5, −1.6]** |
| 4096 | Hadamard | (256, 256, 8, 8, 1) | 453.5 | 241.0 | — |
| 4096 | MFMA | (128, 128, 4, 4, 1) | 532.3 | 209.8 | +17.2 % [+14.8, +17.8] |
| 4096 | FMA | (256, 128, 8, 8, 1) | 534.8 | 209.8 | +18.0 % [+17.8, +18.4] |
| 4096 | `v_dot2` | (256, 256, 8, 8, 1) | **455.5** | 209.8 | **−0.6 % [−1.6, +0.5]** |
| 8192 | Hadamard | (256, 256, 8, 8, 1) | 1332.9 | 427.2 | — |
| 8192 | MFMA | (128, 128, 4, 4, 1) | 1753.1 | 370.5 | +31.5 % [+29.7, +32.6] |
| 8192 | FMA | (256, 128, 8, 8, 1) | 1794.8 | 370.5 | +34.8 % [+33.2, +35.1] |
| 8192 | `v_dot2` | (256, 256, 8, 8, 1) | **1414.7** | 370.5 | **+6.1 % [+5.5, +6.5]** |
| 16384 | Hadamard | (256, 256, 8, 8, 1) | 4734.7 | 860.9 | — |
| 16384 | MFMA | (128, 128, 4, 4, 1) | 6276.9 | 750.4 | +32.4 % [+25.7, +33.3] |
| 16384 | FMA | (256, 128, 8, 8, 1) | 6457.4 | 750.4 | +36.2 % [+27.7, +37.5] |
| 16384 | `v_dot2` | (256, 256, 8, 8, 1) | **4978.4** | 750.4 | **+5.1 % [+4.9, +5.5]** |

- **Logit store:** MFMA is the best variant from T = 2048 on, and 2DP beats Hadamard at every T for both
  models. The three variants are within ~2 % of each other: the store bounds the kernel.
- **Softmax endpoint:** `v_dot2` is the best variant at every T; MFMA and FMA are 8–74 % slower than Hadamard except at 8B T = 1024. In this kernel the
  compiler schedules the padded MFMA / the FMAs poorly next to the softmax, while one `v_dot2` per logit
  stays cheap. 8B: 2DP wins up to T = 4096 and loses 5–6 % from 8192. 1B (head_dim 64: half the fp4 work per
  logit, the same rank-2 work) loses from T = 2048.
- The full flash-attention kernel below uses the MFMA form with a hand-placed schedule, and gets the rank-2
  term to +4–5 % of the FA kernel.

## Flash-attention endpoints

Partial kernel 1 ends at the fp32 logit store, which bounds both pipelines at large T. Flash attention never
writes the logits, so the rank-2 term is no longer hidden behind the store. Two flash-attention endpoints were
measured on Llama-3.1-8B / Llama-3.2-1B shapes, MI350X: a partial one (no P.V) and the full kernel.

### Partial kernel 2: QK^T + online softmax, no P.V, no output

*Partial kernel: P.V and the output are omitted, so the rank-2 term is a larger share of the work than in
the real kernel (e.g. 1B T = 16384: +16.1 % here in job 91651, +20.8 % in the variant sweep job 92107, +4.2 % end to end in
the full kernel).*

P2 = `p2_fa_stats.qk_stats`: fp4 QK^T, rank-2 term, exact online softmax (row max and row sum), `m` and `l` stored.
P.V is omitted. Each method is tuned on the end-to-end chain over the rank-2 variants and tiles; one job (91651).
Times in µs: end to end = median of 5 interleaved repeats; P2 alone and front end: medians. 2DP vs Hadamard: median
[min, max] of the per-repeat end-to-end ratio; bold: 2DP faster end to end. `m` / `l` match the torch reference (max
rel 7e-7).

| model | T | e2e 2DP (µs) | e2e Had (µs) | 2DP vs Had (e2e) | P2 alone 2DP (µs) | P2 alone Had (µs) | front end 2DP (µs) | front end Had (µs) | 2DP pick (variant, tile) | Had tile |
|---|---|---|---|---|---|---|---|---|---|---|
| 8B | 1024 | **114.4** | 117.9 | −2.9 % [−3.1, −2.8] | 27.5 | 25.2 | 91.3 | 97.0 | dot2, (128, 256, 4, 4, 1) | (128, 256, 4, 4, 1) |
| 8B | 2048 | **172.7** | 182.4 | −5.3 % [−5.9, −4.8] | 72.8 | 67.1 | 109.0 | 124.2 | dot2, (256, 256, 8, 8, 1) | (256, 256, 8, 8, 1) |
| 8B | 4096 | **461.2** | 463.6 | −0.5 % [−0.8, −0.2] | 259.2 | 236.4 | 216.9 | 248.4 | dot2, (256, 256, 8, 8, 1) | (256, 256, 8, 8, 1) |
| 8B | 8192 | 1447.0 | 1373.5 | +5.4 % [+4.8, +5.5] | 992.2 | 900.5 | 382.5 | 440.8 | dot2, (256, 256, 8, 8, 1) | (256, 256, 8, 8, 1) |
| 8B | 16384 | 5020.7 | 4772.0 | +5.3 % [+4.9, +5.7] | 3885.0 | 3513.0 | 770.3 | 885.6 | dot2, (256, 256, 8, 8, 1) | (256, 256, 8, 8, 1) |
| 1B | 1024 | **71.0** | 71.7 | −1.0 % [−1.1, −0.8] | 22.3 | 21.5 | 52.2 | 53.9 | dot2, (128, 256, 4, 4, 1) | (128, 256, 4, 4, 1) |
| 1B | 2048 | 109.0 | 107.0 | +1.8 % [+1.8, +1.9] | 61.5 | 53.4 | 51.8 | 57.3 | dot2, (256, 256, 8, 8, 1) | (128, 256, 4, 4, 1) |
| 1B | 4096 | 271.0 | 248.6 | +8.9 % [+8.8, +9.5] | 220.8 | 181.4 | 58.3 | 71.8 | dot2, (256, 256, 8, 8, 1) | (128, 256, 4, 4, 1) |
| 1B | 8192 | 983.5 | 853.1 | +15.3 % [+15.1, +15.4] | 849.8 | 681.7 | 112.0 | 144.7 | dot2, (256, 256, 8, 8, 1) | (128, 256, 4, 4, 1) |
| 1B | 16384 | 3601.3 | 3078.9 | +16.1 % [+15.1, +18.9] | 3337.3 | 2644.3 | 207.2 | 272.5 | dot2, (256, 256, 8, 8, 1) | (128, 256, 4, 4, 1) |

- **The rank-2 cost becomes visible.** Without the store, P2 is no longer memory-bound: 2DP's P2 is 8–11 %
  slower at 8B and 4–26 % at 1B (head_dim 64: the fp4 MFMA work per logit is half, the rank-2 work is not).
- This kernel computes the softmax only, and its issue schedule is the compiler's. The full kernel below shows
  that the gap is mostly a scheduling / load-path effect, not the rank-2 arithmetic.

### Full flash attention (the real kernel)

*The real attention operator. Code: [`fa/`](fa/) (see [Run: full flash attention](#run-full-flash-attention)).*

Kernel: the gfx950 Gluon tutorial `fmha_v3` (rotated 4-cluster warp-pipelined loop, eager online-softmax
rescale, P.V in fp16 with V through LDS, fp16 O + LSE stores), with the QK^T replaced by the MXFP4
`mfma_scaled` and, for 2DP, the rank-2 term. Softmax scale `1/sqrt(D)` (Hadamard) or `1/(sqrt(D)(1-gamma))`
(2DP). Non-causal, B = 1, 32 Q / 8 KV heads, D = 64 (1B) or 128 (8B). O matches torch (rel L2 2.8e-4 for both
methods). The final kernel for both head dims is [`fa/fmha_d64.py`](fa/fmha_d64.py); at D = 128 it compiles to the
same instructions as the 8B kernel of job 92099. The intermediate kernels of the 8B K-path table below are not in
this repo.

#### Run: full flash attention

| file | content |
|---|---|
| `fa/fmha_d64.py` | the kernel (`run_mx`): MXFP4 QK^T (`corr=0` Hadamard, `corr=1` 2DP: rank-2 term on MFMA with `vp=0`, on `v_dot2` with `vp=4`), online softmax, P.V, O / LSE; knobs `block_m`, `block_n`, `num_warps`, `lazy`, `prio`, `vf` (dot2 fence, D = 64 only), `lrke`, `r2p`, `exs`, `kch` |
| `fa/fmha_roof.py`, `fa/fmha_roof_lds.py`, `fa/fmha_kbank.py` | kernel helpers and the D = 128 code path it reuses |
| `fa/fa_setup.py` | inputs: runs the front ends of `../` (`p1_2dp`, `had_j2`) once per T |
| `fa/bench_d64.py` | benchmark: O / LSE vs torch, interleaved timing (FA kernel and end to end), ISA census; sets `readme1b`, `readme8b` |
| `fa/fmha_fp.py` | the kernel with an fp16 QK^T mode (`QF16`) for the fp16 baseline; same loop as `fmha_d64.py` |
| `fa/bench_fp.py`, `fa/tut_env.py` | fp16 vs MXFP4 benchmark; `tut_env` applies two small edits to the tutorial's `fmha_v3.py` at import time for the parity reference |
| `fa/bench_vdot2.py` | Hadamard vs MFMA 2DP vs `v_dot2` 2DP in the same job (implementations `had`, `mf_p0`, `vd_m`, `vd_mk`, ...) |

The kernel imports `fmha_v3.py` / `common.py` from the public gfx950 Gluon tutorial ("High-Performance Gluon
Kernels on AMD GFX9", `kernels/attention/`, MIT), which is not vendored. Used versions (sha256):
`fmha_v3.py` 641584fc28bc1dbf6748263f72044de5e8c67fd325c0c5c3b999120ea627eb4f,
`common.py` 4406e2f9e19c159a587909d096f198c7411e1428f282e7681ad845f617cd2692.
Triton: the `gfx950-tutorial-v1.0` build, no scheduler plugin.

```bash
export GLUON_TUTORIAL_ATTENTION_DIR=<tutorial checkout>/kernels/attention
cd src/brevitas_examples/papers/2dp/kernels/fa
python bench_d64.py --model 1b --set readme1b --check-ts 2048 16384 --ts 1024 2048 4096 8192 16384 --rounds 5 --e2e
python bench_d64.py --model 8b --set readme8b --check-ts 2048 16384 --ts 4096 8192 16384 --rounds 5 --e2e \
    --u-path U_selfgen_8b_2dp_allrand_s0.pt --u-layer 15
```

Reproduced from this directory (job 92422, one job for both models, 5 interleaved repeats; every variant's O / LSE
checked at T = 2048 and 16384, O rel L2 ≤ 2.9e-4), fastest Hadamard vs fastest 2DP, picked separately for the FA kernel
and for end to end. Times in µs (median of the repeats); ratios: median [min, max] of the per-repeat ratio.
Config names (tile prefix dropped): `lazy` lazy rescale, `prio` `s_setprio`, `ns_scal` no machine sink + scalarize
compile flag, `vf1` / `vf1t` dot2 fence, `e` early K reads, `p0` rank-2 MFMA placement 0; 1B tile 8 warps 256 x 128
unless the name says `w4m128n128`, 8B tile 8 warps 256 x 64 with K through LDS (`L`).

| model | T | FA Had (µs) | FA 2DP (µs) | FA 2DP vs Had | front end Had (µs) | front end 2DP (µs) | e2e Had (µs) | e2e 2DP (µs) | e2e 2DP vs Had | fastest FA config Had / 2DP | fastest e2e config Had / 2DP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1B | 1024 | 21.8 | 22.8 | +5.0 % [+4.0, +5.3] | 54.1 | 52.7 | 72.0 | 71.7 | −0.5 % [−0.6, −0.4] | `p0e_vf1t_lazy` / `p0_vf1t_lazy` | `p0_vf1t_lazy_prio` / `p0_vf1t_lazy` |
| 1B | 2048 | 36.5 | 38.5 | +5.4 % [+4.6, +6.0] | 57.8 | 52.2 | 90.5 | 86.2 | −4.7 % [−5.0, −4.6] | `p0e_vf1t_lazy` / `p0_vf1t_lazy_prio` | `p0e_vf1t_lazy` / `p0_vf1t_lazy_prio` |
| 1B | 4096 | 134.7 | 143.2 | +6.1 % [+5.9, +6.3] | 71.1 | 59.6 | 211.2 | 205.7 | −2.5 % [−2.6, −2.3] | `p0e_vf1t_lazy` / `p0_vf1t_lazy_prio` | `p0e_vf1t_lazy` / `p0_vf1t_lazy_prio` |
| 1B | 8192 | 523.0 | 558.9 | +6.9 % [+6.6, +7.0] | 145.3 | 112.3 | 673.5 | 682.3 | +1.3 % [+1.1, +1.4] | `p0e_vf1t_lazy` / `p0_vf1t_lazy_prio` | `p0e_vf1t_lazy` / `p0_vf1t_lazy_prio` |
| 1B | 16384 | 2105.0 | 2241.5 | +6.4 % [+6.0, +7.3] | 265.8 | 204.4 | 2375.5 | 2461.3 | +3.8 % [+3.0, +4.2] | `p0e_vf1t_lazy` / `p0_vf1t_lazy_prio` | `p0_vf1t_lazy_prio` / `p0_vf1t_lazy_prio` |
| 8B | 4096 | 223.4 | 231.4 | +3.7 % [+3.5, +3.8] | 239.5 | 210.4 | 471.8 | 450.3 | −4.5 % [−4.8, −4.3] | `L_lazy_ns_scal` / `L_lazy_prio` | `L_lazy_ns_scal` / `L_lazy_prio` |
| 8B | 8192 | 880.5 | 918.9 | +4.4 % [+4.3, +4.6] | 435.2 | 380.3 | 1318.7 | 1307.4 | −0.6 % [−1.7, +0.2] | `L_lazy_ns_scal` / `L_lazy_prio` | `L_lazy_ns_scal` / `L_lazy` |
| 8B | 16384 | 3556.2 | 3696.4 | +3.7 % [+3.3, +4.8] | 877.9 | 759.9 | 4423.5 | 4444.5 | +0.6 % [−0.4, +0.9] | `L_lazy_ns_scal` / `L_lazy_prio` | `L_lazy_ns_scal` / `L_lazy_prio` |

Within noise of the original jobs (92366 for 1B, 92099 for 8B); the 1B T = 2048 end-to-end ratio is known to move
by about ±2 points between jobs. The `readme1b` set has only the 256 x 128 tile, which is not the fastest at
T = 1024 (job 92270 used 4 warps, 128 x 128 there); its T = 1024 row is in the table for completeness.

Llama-3.1-8B, T = 8192, FA kernel alone, µs, median of 5 interleaved repeats. Every K-path change is applied
to both methods; the comparison is always fastest Hadamard against fastest 2DP.

| K path | Hadamard | 2DP | 2DP vs Hadamard | job |
|---|---|---|---|---|
| K in registers, global loads in mem1 (first port) | 1256 | 1434 | +14 % | 92059 |
| same, rank-2 as fp16 32x32x16 MFMA with the fp4 accumulator as srcC, scales loaded first | 1166 | 1198 | +2.8 % | 91982 |
| K through LDS, prefetched (`lds3`) | 1027 | 1073 | +4.8 % | 92012 |
| **K through LDS with direct-to-LDS loads, tutorial schedule (`L`)** | **959** | **1005** | **+4.7 %** | 92078 |

Final run (job 92099), 8B, both K through LDS, then each method on its fastest scheduling knobs (Hadamard:
lazy rescale + no machine sink + scalarized packed ops; 2DP: lazy rescale + `s_setprio`), the same knobs for the FA
kernel and end to end. The `first port` rows are the first port in the same job (92099; the first-port row of the
K-path table above is job 92059). Times in µs (median of 5 interleaved
repeats); ratios: median [min, max] of the per-repeat ratio:

| set | T | FA Had (µs) | FA 2DP (µs) | FA 2DP vs Had | front end Had (µs) | front end 2DP (µs) | e2e Had (µs) | e2e 2DP (µs) | e2e 2DP vs Had | fastest FA config Had / 2DP | e2e config Had / 2DP (FA-fastest knobs) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| final | 4096 | 226.2 | 235.6 | +4.2 % [+3.9, +4.5] | 243.0 | 212.1 | 478.1 | 455.8 | −4.6 % [−4.7, −4.3] | `L_lazy_ns_scal` / `L_lazy_prio` | `L_lazy_ns_scal` / `L_lazy_prio` |
| final | 8192 | 903.0 | 949.9 | +4.8 % [+4.0, +6.1] | 436.6 | 380.9 | 1359.3 | 1343.0 | −0.9 % [−2.5, −0.7] | `L_lazy_ns_scal` / `L_lazy_prio` | `L_lazy_ns_scal` / `L_lazy_prio` |
| final | 16384 | 3621.4 | 3776.4 | +4.2 % [+3.1, +4.9] | 881.3 | 764.4 | 4491.5 | 4527.3 | +0.8 % [+0.3, +1.3] | `L_lazy_ns_scal` / `L_lazy_prio` | `L_lazy_ns_scal` / `L_lazy_prio` |
| first port (`base`) | 4096 | 311.6 | 348.1 | +11.6 % [+11.2, +11.9] | 243.0 | 212.1 | 555.7 | 558.6 | +0.5 % [+0.0, +1.1] | `base` / `base` | `base` / `base` |
| first port (`base`) | 8192 | 1237.2 | 1389.9 | +12.3 % [+12.1, +12.3] | 436.6 | 380.9 | 1654.1 | 1767.9 | +6.9 % [+6.8, +7.0] | `base` / `base` | `base` / `base` |
| first port (`base`) | 16384 | 4802.1 | 5539.4 | +15.4 % [+13.6, +16.0] | 881.3 | 764.4 | 5644.4 | 6337.0 | +12.5 % [+11.0, +12.6] | `base` / `base` | `base` / `base` |

With the fastest end-to-end config picked separately, only 8B T = 8192 changes: the fastest 2DP end to end is
`L_lazy` (1341.5 µs), −0.5 % [−1.7, +0.3] vs Hadamard `L_lazy_ns_scal`.

The best Hadamard kernel runs at 1218 TF/s: 61 % of its static issue floor (551 µs) and 41 % MFMA utilization
at the 1.78 GHz clock sampled under load.

Caveats (from two independent reviews of the data):
- The remaining +4–5 % is not established as the cost of the rank-2 term. With identical knobs for both
  methods the FA gap ranges from −1.8 % to +5.6 %, and code placement alone moves the timing by 5–7 % (two
  Hadamard builds whose loops differ only in where one load sits differ by 7 %).
- On the benchmark inputs the lazy rescale is never needed, so its gain is best case; with the rescale needed
  in 52 % of steps it was checked for correctness only.
- O is checked against torch at T = 2048 only.

Llama-3.2-1B (head_dim 64; [`fa/fmha_d64.py`](fa/fmha_d64.py), set `readme1b`). Best tile for both methods: 8 warps,
256 x 128 (BLOCK_N = 128 fits at D = 64). Hadamard: lazy rescale + dot2 fence + early K reads; 2DP: lazy rescale +
`s_setprio` + dot2 fence. Job 92366 (T = 1024: job 92270, 4 warps, 128 x 128), 5 interleaved repeats, O / LSE checked at
T = 2048 and 16384 (92270: T = 2048 and 8192). Fastest picked separately for the FA kernel and end to end; times in µs (median of the repeats);
ratios: median [min, max] of the per-repeat ratio. Config names as in the reproduction table.

| model | T | FA Had (µs) | FA 2DP (µs) | FA 2DP vs Had | front end Had (µs) | front end 2DP (µs) | e2e Had (µs) | e2e 2DP (µs) | e2e 2DP vs Had | fastest FA config Had / 2DP | fastest e2e config Had / 2DP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1B | 1024 | 16.6 | 17.7 | +6.7 % [+5.2, +8.1] | 53.4 | 52.8 | 66.6 | 67.6 | +1.5 % [+1.4, +1.6] | `w4m128n128` / `w4m128n128_lazy_ns_scal` | `w4m128n128` / `w4m128n128_lazy_ns_scal` |
| 1B | 2048 | 37.1 | 39.2 | +5.6 % [+5.5, +5.7] | 56.8 | 52.3 | 89.4 | 87.7 | −1.8 % [−2.0, −1.7] | `p0e_vf1t_lazy` / `p0_vf1t_lazy_prio` | `p0e_vf1t_lazy` / `p0_vf1t_lazy_prio` |
| 1B | 4096 | 137.8 | 146.0 | +6.0 % [+5.9, +6.4] | 71.7 | 59.1 | 215.1 | 208.8 | −2.9 % [−2.9, −2.8] | `p0e_vf1t_lazy` / `p0_vf1t_lazy_prio` | `p0e_vf1t_lazy` / `p0_vf1t_lazy_prio` |
| 1B | 8192 | 540.0 | 578.7 | +7.2 % [+7.0, +8.1] | 144.3 | 114.2 | 696.4 | 702.7 | +0.9 % [+0.8, +1.0] | `p0e_vf1t_lazy` / `p0_vf1t_lazy_prio` | `p0e_vf1t_lazy` / `p0_vf1t_lazy_prio` |
| 1B | 16384 | 2172.2 | 2322.6 | +6.8 % [+6.1, +7.6] | 271.5 | 209.6 | 2454.5 | 2557.4 | +4.2 % [+3.6, +5.0] | `p0e_vf1t_lazy` / `p0_vf1t_lazy_prio` | `p0e_vf1t_lazy` / `p0_vf1t_lazy_prio` |

- The remaining 1B overhead is the rank-2 MFMA itself: at head_dim 64 one fp16 32x32x8 MFMA per 32 x 32 block
  costs as much as the fp4 QK^T MFMA (~32 cycles each). Diagnostic builds without it are +0–1.5 %.
- Part of the end-to-end gain is the fused 2DP front end (one kernel) vs Hadamard's two kernels; with equal front
  ends the end-to-end ratio would be about +2 / +4 / +6 / +6 % at T = 2k–16k (estimate from medians).
- The `v_dot2` form of the rank-2 term: see [MFMA vs `v_dot2`](#rank-2-term-mfma-vs-v_dot2-full-flash-attention).

#### Rank-2 term: MFMA vs `v_dot2` (full flash attention)

The same kernel ([`fa/fmha_d64.py`](fa/fmha_d64.py)) computes the 2DP rank-2 term `a'.b` in one of two ways:

- **MFMA** (`vp=0`, the results above): one fp16 32x32x8 MFMA per 32 x 32 score block, coefficients in a
  zero-padded 8-half slot, fed as the accumulator input of the fp4 QK^T MFMA.
- **`v_dot2`** (`vp=4`; `kch=1` with 128-key tiles): the vector-unit form, one `v_dot2_f32_f16` per logit.
  - `a'` (2 fp16 per query row) is packed in one register per lane for the whole kernel; `b` (2 fp16 per key) is
    one dword per token, copied with each K tile into a 4-deep LDS ring. No padding, no rank-2 MFMA.
  - The term of the *next* tile is computed in the memory stage (`c0 = a'.b`, accumulator 0) and becomes the
    accumulator input of that tile's fp4 QK^T MFMA, so the logits are `R + a'.b` with no extra add, and the
    `v_dot2` instructions stay out of the softmax stage, which is the slowest one.
  - `kch=1`: with 128-key tiles, `b` is read from LDS per 32-column chunk right before its `v_dot2`s (16 live
    registers instead of 64), which avoids spills.

Fastest Hadamard vs fastest of each 2DP form, same job per model (1B: 92415, 8B: 92416), 5 interleaved repeats; every
reported variant's O / LSE checked at T = 2048 and 8192 (O rel L2 ≤ 2.9e-4). The `v_dot2` column is the `vp=4` kernel
above (`vd_mk` = `kch=1`, `vd_m` = `kch=0`); builds whose check failed at any checked T (O rel L2 > 1e-3 or LSE >
1e-4) are excluded from the selection (among the `vp=4` builds: 1B `vf1_lazy_ns_scal` with `vd_m`). The fastest is
picked separately for the FA kernel and for end to end, over the knob configs of the job (MFMA: implementation
`mf_p0`). Times in µs (median of the repeats); ratios: median [min, max] of the per-repeat ratio. The front-end row
gives the front-end times (Hadamard / 2DP; the two 2DP forms share the 2DP front end). Config names: tile prefix
dropped (1B 8 warps 256 x 128, 8B 8 warps 256 x 64), `(none)` = no extra knob, other names as in the reproduction
table.

Llama-3.2-1B (job 92415):

| T | kind | Had (µs) | 2DP-MFMA (µs) | 2DP-`v_dot2` (µs) | MFMA vs Had | `v_dot2` vs Had | `v_dot2` vs MFMA | fastest config Had / MFMA / `v_dot2` |
|---|---|---|---|---|---|---|---|---|
| 2048 | FA kernel | 37.3 | 39.6 | 44.0 | +6.0 % [+5.3, +6.3] | +17.7 % [+17.4, +18.3] | +11.0 % [+10.5, +12.3] | `vf1_lazy` / `vf1_lazy_prio` / `vf1_lazy_ns_scal` vd_mk |
| 2048 | end to end | 89.7 | 86.3 | 91.2 | −3.9 % [−4.0, −3.7] | +1.7 % [+1.6, +2.2] | +5.8 % [+5.6, +6.2] | `vf1_lazy` / `vf1_lazy_prio` / `vf1_lazy_ns_scal` vd_mk |
| 2048 | front end | 57.0 | 52.2 | 52.2 | | | | |
| 4096 | FA kernel | 138.6 | 146.8 | 172.4 | +6.0 % [+5.6, +6.1] | +24.5 % [+24.2, +24.7] | +17.4 % [+17.1, +17.9] | `vf1_lazy_prio` / `vf1_lazy_prio` / `vf1_lazy_ns_scal` vd_mk |
| 4096 | end to end | 216.3 | 209.7 | 235.6 | −3.0 % [−3.1, −2.9] | +8.9 % [+8.9, +9.0] | +12.3 % [+12.2, +12.4] | `vf1_lazy_prio` / `vf1_lazy_prio` / `vf1_lazy_prio` vd_mk |
| 4096 | front end | 71.7 | 59.4 | 59.4 | | | | |
| 8192 | FA kernel | 545.6 | 581.2 | 675.8 | +6.5 % [+6.3, +6.8] | +24.0 % [+23.4, +24.3] | +16.2 % [+16.0, +16.4] | `vf1_lazy_prio` / `vf1_lazy_prio` / `vf1_lazy_prio` vd_mk |
| 8192 | end to end | 701.2 | 709.5 | 797.2 | +1.1 % [+1.0, +1.3] | +13.7 % [+13.3, +14.0] | +12.4 % [+12.2, +12.5] | `vf1_lazy_prio` / `vf1_lazy_prio` / `vf1_lazy_prio` vd_mk |
| 8192 | front end | 147.5 | 116.1 | 116.1 | | | | |
| 16384 | FA kernel | 2193.7 | 2334.5 | 2685.4 | +6.4 % [+5.7, +7.7] | +22.6 % [+21.6, +23.2] | +15.1 % [+14.4, +15.6] | `vf1_lazy` / `vf1_lazy_prio` / `vf1_lazy_ns_scal` vd_mk |
| 16384 | end to end | 2474.0 | 2570.7 | 2900.6 | +3.8 % [+3.3, +4.1] | +17.2 % [+16.7, +17.4] | +12.9 % [+12.7, +13.1] | `vf1_lazy` / `vf1_lazy_prio` / `vf1_lazy` vd_mk |
| 16384 | front end | 274.6 | 211.6 | 211.6 | | | | |

Llama-3.1-8B (job 92416):

| T | kind | Had (µs) | 2DP-MFMA (µs) | 2DP-`v_dot2` (µs) | MFMA vs Had | `v_dot2` vs Had | `v_dot2` vs MFMA | fastest config Had / MFMA / `v_dot2` |
|---|---|---|---|---|---|---|---|---|
| 2048 | FA kernel | 60.2 | 63.1 | 67.7 | +4.8 % [+4.5, +5.5] | +12.4 % [+12.2, +12.7] | +7.2 % [+6.9, +7.6] | `lazy_ns_scal` / `lazy_prio` / `(none)` vd_m |
| 2048 | end to end | 187.7 | 176.2 | 181.4 | −6.1 % [−6.2, −6.1] | −3.4 % [−3.6, −3.2] | +3.0 % [+2.7, +3.1] | `lazy_ns_scal` / `lazy_prio` / `(none)` vd_m |
| 2048 | front end | 124.2 | 108.7 | 108.7 | | | | |
| 4096 | FA kernel | 229.6 | 239.1 | 259.7 | +4.2 % [+3.9, +4.4] | +13.1 % [+12.8, +13.3] | +8.5 % [+8.4, +8.7] | `lazy_ns_scal` / `lazy_prio` / `(none)` vd_m |
| 4096 | end to end | 483.4 | 461.4 | 483.4 | −4.6 % [−4.7, −4.4] | −0.0 % [−0.2, +0.1] | +4.7 % [+4.6, +4.9] | `lazy_ns_scal` / `lazy_prio` / `(none)` vd_m |
| 4096 | front end | 244.7 | 215.7 | 215.7 | | | | |
| 8192 | FA kernel | 920.6 | 959.4 | 1029.8 | +4.2 % [+3.6, +5.5] | +11.6 % [+11.0, +15.0] | +7.2 % [+6.6, +9.6] | `lazy_ns_scal` / `lazy_prio` / `(none)` vd_m |
| 8192 | end to end | 1372.1 | 1358.9 | 1429.9 | −0.8 % [−1.3, −0.6] | +4.2 % [+3.7, +5.6] | +5.0 % [+4.4, +6.3] | `lazy_ns_scal` / `lazy_prio` / `(none)` vd_m |
| 8192 | front end | 440.3 | 386.0 | 386.0 | | | | |
| 16384 | FA kernel | 3644.1 | 3800.1 | 4145.8 | +4.3 % [+3.7, +5.0] | +13.6 % [+13.2, +14.0] | +8.9 % [+8.3, +9.3] | `lazy_ns_scal` / `lazy_prio` / `(none)` vd_m |
| 16384 | end to end | 4522.1 | 4562.8 | 4932.2 | +0.8 % [+0.4, +1.4] | +9.4 % [+8.5, +9.7] | +8.1 % [+7.7, +9.0] | `lazy_ns_scal` / `lazy_prio` / `(none)` vd_m |
| 16384 | front end | 888.0 | 768.9 | 768.9 | | | | |

Why `v_dot2` costs more here than in partial kernel 1 (reviewed; jobs 92405–92416):
- The softmax already keeps the vector unit busy (counted active in ~70 % of cycles in the Hadamard kernel, an upper
  estimate: the counter is in 4-cycle units; counters, job 92409), so extra vector instructions add time directly.
- On gfx950 a `v_dot2_f32_f16` costs ~6 cycles and does not overlap the MFMAs, while a plain `v_fma_f32` does as
  long as the MFMA shadow has room (microbenchmark, job 92411). Adding 32 dummy `v_dot2` per tile to the Hadamard
  kernel reproduces most of the `v_dot2` 2DP slowdown (counters, job 92409).
- The end-to-end `v_dot2` numbers are slightly pessimistic: its front end still writes the padded 8-half coefficient
  slot of the MFMA form, of which the kernel reads only the first dword.
- At head_dim 128 (8B) the kernel does twice the MFMA work per logit for the same `v_dot2` count, consistent with the
  smaller relative cost (+12–14 % vs +18–25 % at 1B); not measured separately.

Why the ~30 % of cycles the vector unit is idle does not absorb the `v_dot2` work (reviewed; 1B, T = 8192):
- The idle cycles are mostly cycles where no wave can issue: the waves spend 36 % of their cycles stalled on LDS data,
  barriers or MFMA results. A SIMD fills an idle cycle only with an instruction from a wave that is ready. This
  kernel runs 2 waves per SIMD, synchronized by `s_barrier` at every stage boundary (the warp-pipelined ping-pong of
  `fmha_v3`), so they tend to stall at the same time. A `v_dot2` sits at a fixed place in its wave's instruction
  stream; it runs when the wave reaches it, not in the other wave's stall.
- Measured: each `v_dot2` added to the Hadamard kernel costs its full ~7 cycles of SIMD time (injection test, jobs
  92406 / 92413). With 32 dummy `v_dot2` per wave-tile in the Hadamard kernel the SIMD time per wave-tile grows by 352
  cycles for 124 more busy vector cycles, and the MFMA / vector overlap drops from 19 % to 6 % of cycles (counters, job
  92409; the 2DP `v_dot2` kernel is in the same table): the extra
  `v_dot2`s also stop MFMA work from running in parallel with the vector work. Why: see
  [Why `v_dot2` does not overlap the MFMAs](#why-v_dot2-does-not-overlap-the-mfmas).
- Removing the barrier lockstep (no warp pipeline) brings each added `v_dot2` down to its ~4-cycle issue slot
  (7.3 cycles with the pipeline), so the ping-pong schedule is what doubles its cost; but that kernel starts about
  19 % slower and is still slower at the 32 `v_dot2` per wave-tile that 2DP needs. With a third wave per SIMD
  (job 92413, injection table below): without the warp pipeline, 32 dummy `v_dot2` cost +18.3 %, as with 2 waves
  (+18.8 %); the pipelined 3-wave build costs less (+13.0 %, and is faster in absolute time: 693.3 vs 732.9 µs) but
  spills 7 registers, so no clean configuration shows a third wave hiding the `v_dot2` work. Hiding the `v_dot2` work would need a schedule that keeps the
  pipeline and places the `v_dot2`s in its stall windows.

Measurements behind these points.

Microbenchmark (job 92411): per iteration every wave issues 8 fp16 `32x32x16` MFMAs (2 accumulator chains; 256 cycles
of MFMA pipe per wave) and NV vector instructions of one kind (8 independent chains). One workgroup per CU; cycles from
`s_memtime`. NV = 0 column: SIMD cycles per iteration; other columns: SIMD cycles added per vector instruction per
wave, (cycles - 256 x waves) / (NV x waves); ~0 = hidden next to the MFMAs.

| instruction | waves/SIMD | NV = 0 (SIMD cyc/iter) | NV = 8 | NV = 16 | NV = 24 | NV = 32 | NV = 48 | NV = 64 |
|---|---|---|---|---|---|---|---|---|
| `v_fma_f32` | 1 | 256 | 0.55 | 0.52 | 0.51 | 0.51 | 0.84 | 1.69 |
| `v_fma_f32` | 2 | 512 | 0.26 | 0.20 | 0.20 | 0.26 | 0.50 | 1.20 |
| `v_dot2_f32_f16` | 1 | 256 | 6.80 | 6.52 | 6.34 | 6.13 | 6.84 | 6.19 |
| `v_dot2_f32_f16` | 2 | 512 | 6.35 | 6.38 | 6.16 | 6.00 | 6.23 | 5.75 |
| `v_exp_f32` | 1 | 256 | 4.54 | 4.52 | 4.51 | 4.51 | 4.76 | 5.57 |
| `v_exp_f32` | 2 | 512 | 3.07 | 3.41 | 3.91 | 3.69 | 4.36 | 5.27 |

Counters (job 92409; 1B, T = 8192, median over the profiled FA dispatches after the first, 6 of 7 per kernel; per SIMD, fractions of SIMD cycles; a wave-tile = one
wave's work on one KV tile). "Hadamard + 32 `v_dot2`" is the Hadamard kernel with 32 dummy `v_dot2` per wave-tile:

| metric | Hadamard | Hadamard + 32 `v_dot2` | 2DP MFMA | 2DP `v_dot2` |
|---|---|---|---|---|
| kernel time under the profiler (µs) | 724.4 | 870.0 | 758.4 | 907.6 |
| SIMD cycles per wave-tile | 1025 | 1377 | 1029 | 1390 |
| VALU-busy cycles per wave-tile | 717 | 840 | 721 | 883 |
| VALU active (fraction of SIMD cycles) | 69.9 % | 61.0 % | 70.1 % | 63.5 % |
| MFMA busy (fraction of SIMD cycles) | 31.2 % | 23.2 % | 37.3 % | 23.0 % |
| MFMA / VALU co-execution (fraction of SIMD cycles) | 18.5 % | 6.0 % | 24.1 % | 7.5 % |
| wave cycles waiting (SQ_WAIT_ANY / wave cycles) | 36.0 % | 40.5 % | 35.1 % | 37.0 % |
| VALU instructions per wave-tile | 142.1 | 173.1 | 143.1 | 183.8 |
| MFMA instructions per wave-tile | 10 | 10 | 12 | 10 |
| LDS instructions per wave-tile | 20.1 | 20.1 | 21.1 | 28.1 |
| LDS bank conflicts per wave-tile | 0.27 | 0.27 | 8.27 | 0.27 |

Injection (job 92413; 1B, T = 8192, Hadamard kernel with k dummy `v_dot2` per wave-tile, per tile config and
occupancy target `wpe`, with and without the warp pipeline). FA time: median of 5 interleaved repeats; vs k = 0: median
[min, max] of the per-repeat ratio against the same config with k = 0. SIMD cycles from `s_memtime` per workgroup / 128
KV tiles; given only for the 8-warp configs with `wpe` 2 (one workgroup per CU, 2 waves per SIMD), where a workgroup's
cycles are its SIMDs' cycles; added cycles per `v_dot2` per wave = Δcycles / k / 2. A spill > 0 marks builds that
spill registers:

| config | `wpe` (waves/SIMD target) | warp pipeline | k | FA (µs) | vs k = 0 | SIMD cycles / KV tile | added SIMD cycles per `v_dot2` per wave | VGPR | spill |
|---|---|---|---|---|---|---|---|---|---|
| `w8m256` | 2 | yes | 0 | 616.4 | — | 2069 | — | 168 | 0 |
| `w8m256` | 2 | yes | 32 | 732.9 | +18.8 % [+18.4, +19.5] | 2538 | 7.3 | 172 | 0 |
| `w8m256` | 2 | yes | 64 | 822.5 | +33.2 % [+32.7, +34.0] | 2994 | 7.2 | 172 | 0 |
| `w8m256` | 2 | no | 0 | 732.9 | — | 2565 | — | 197 | 0 |
| `w8m256` | 2 | no | 32 | 801.6 | +8.8 % [+7.9, +11.3] | 2813 | 3.9 | 201 | 0 |
| `w8m256` | 2 | no | 64 | 858.2 | +16.7 % [+15.6, +20.0] | 3114 | 4.3 | 201 | 0 |
| `w4m128` | 3 | yes | 0 | 611.9 | — | — | — | 168 | 0 |
| `w4m128` | 3 | yes | 32 | 693.3 | +13.0 % [+12.2, +15.2] | — | — | 168 | 7 |
| `w4m128` | 3 | yes | 64 | 794.6 | +29.2 % [+28.5, +31.8] | — | — | 168 | 7 |
| `w4m128` | 3 | no | 0 | 612.5 | — | — | — | 168 | 4 |
| `w4m128` | 3 | no | 32 | 725.4 | +18.3 % [+16.8, +19.2] | — | — | 162 | 0 |
| `w4m128` | 3 | no | 64 | 799.6 | +30.4 % [+28.8, +31.2] | — | — | 162 | 0 |
| `w4m128` | 2 | yes | 0 | 636.1 | — | — | — | 168 | 0 |
| `w4m128` | 2 | yes | 32 | 722.5 | +13.3 % [+13.3, +14.0] | — | — | 180 | 0 |
| `w4m128` | 2 | yes | 64 | 817.3 | +28.7 % [+28.2, +28.9] | — | — | 180 | 0 |
| `w4m128` | 4 | yes | 0 | 640.1 | — | — | — | 128 | 33 |
| `w4m128` | 4 | yes | 32 | 981.3 | +53.0 % [+52.1, +53.8] | — | — | 128 | 74 |
| `w4m128` | 4 | yes | 64 | 1018.3 | +58.9 % [+58.6, +59.4] | — | — | 128 | 76 |
| `w8m256` | 4 | no | 0 | 634.2 | — | — | — | 128 | 49 |
| `w8m256` | 4 | no | 32 | 848.7 | +33.8 % [+33.7, +34.2] | — | — | 128 | 107 |
| `w8m256` | 4 | no | 64 | 917.5 | +44.7 % [+43.9, +45.0] | — | — | 128 | 69 |

#### Why `v_dot2` does not overlap the MFMAs

Controlled microbenchmark (jobs 92434 / 92435): the whole timed loop is one inline-asm block, 16 x (one MFMA, then k
vector instructions of one kind), MFMAs on 4 independent accumulator chains, one workgroup per CU; every variant's ISA
checked (16 MFMAs, 16 k vector instructions, no `s_nop`). Cells: extra SIMD cycles per vector instruction,
(C(k) - C(0)) / (16 k waves). For the co-issuing classes the extra total is about 4 + 4k cycles per loop whatever the
instruction, i.e. a loop-tail cost: their steady-state cost is ~0. One MFMA = 32.25 cycles (fp16 32x32x16) or 32.5
(fp4 32x32x64).

| vector instruction (class) | fp16 MFMA, 1 wave, k = 1 / 2 / 4 / 6 | fp16 MFMA, 2 waves, k = 1 / 2 / 4 / 6 | fp4 MFMA, 1 wave, k = 2 / 4 / 6 | fp4 MFMA, 2 waves, k = 2 / 4 / 6 |
|---|---|---|---|---|
| `v_fma_f32` (VOP3; 3 VGPR, 2 VGPR, 1 VGPR + constants, src2 = dst) | 0.50 / 0.38 / 0.31 / 0.31 | 0.23 / 0.18 / 0.15 / 0.14 | 0.38 / 0.32 / 0.92 | 0.18 / 0.15 / 0.79 |
| `v_fmac_f32`, `v_mul_f32`, `v_add_f32`, `v_mov_b32`, `v_cvt_f32_f16`, `v_mov_b64` (VOP1 / VOP2) | 0.50 / 0.38 / 0.31 / 1.19 | 0.23 / 0.18 / 0.15 / 0.14 | 0.38 / 0.32 / 0.92 | 0.18 / 0.15 / 0.79 |
| `v_fma_mix_f32` (VOP3P, not a dot) | 0.50 / 0.38 / 0.31 / 0.31 | 0.23 / 0.18 / 0.15 / 0.56 | 0.38 / 0.32 / 0.92 | 0.18 / 0.15 / 0.79 |
| `v_pk_fma_f16` (VOP3P, packed fp16) | 0.50 / 0.38 / 0.31 / 0.31 | 0.23 / 0.18 / 0.15 / 0.56 | 0.38 / 0.32 / 0.92 | 0.18 / 0.15 / 0.79 |
| `v_exp_f32` (transcendental) | 0.75 / 0.63 / 2.44 / 4.29 | 0.35 / 0.29 / 2.22 / 4.14 | 0.63 / 3.38 / 4.92 | 0.29 / 3.19 / 4.79 |
| **`v_dot2_f32_f16`** (VOP3P dot; also with 1 VGPR, MFMA-operand or idle-accumulator sources) | **17.75 / 10.88 / 7.44 / 6.29** | **16.50 / 10.25 / 7.12 / 6.08** | **12.75 / 8.38 / 6.92** | **12.13 / 8.06 / 6.70** |
| **`v_dot2c_f32_f16`** (VOP2 dot) | same as `v_dot2` | same | same | same |
| **`v_pk_fma_f32`, `v_pk_mul_f32`, `v_pk_add_f32`, `v_pk_mov_b32`** (packed fp32) | same as `v_dot2` | same | same | same |

Vector unit alone (no MFMA), cycles per instruction per SIMD:

| class | 1 wave | 2 waves | 4 waves |
|---|---|---|---|
| fp32 `fma`, `fmac`, `mul`, `add`, `mov` | 6.0 | 2.25 | 2.2 |
| `v_dot2`, `v_dot2c`, `v_fma_mix_f32`, `v_pk_*`, `v_fma_f64`, `v_cvt_f32_f16`, `v_mov_b64` | 6.0 | 4.2–4.9 | 4.1 |
| `v_exp_f32` | 9.9 | 8.9 | 8.5 |

- **The rule (measured): instruction class.** Dot instructions (`v_dot2`, and the VOP2 `v_dot2c` too) and packed-fp32
  instructions are never issued while an MFMA runs on the SIMD: they wait until it has finished (fitted: D + 4k extra
  cycles per MFMA gap, D = 13.75 / 12.5 cycles for the fp16 MFMA at 1 / 2 waves, 17.5 / 16.25 for the fp4 MFMA), and a
  second wave does not recover this time. It is not the encoding (`v_fma_mix_f32` and `v_pk_fma_f16` are VOP3P and
  co-issue; `v_dot2c` is VOP2 and does not), not the register reads (3-VGPR `v_fma_f32` and 64-bit `v_fma_f64`
  co-issue; `v_dot2` with one VGPR source does not), not dependencies on the MFMA's registers, and not the half-rate
  throughput (`v_fma_mix_f32`, `v_fma_f64`, `v_cvt` are half rate and co-issue).
- **The compiler knows the same class.** LLVM (the Triton build's LLVM, `SIInstrInfo::isNeverCoissue` for
  gfx940/gfx950) lists TRANS, DOT, MFMA and the `v_pk_*` instructions as never co-issued, and a pre-emit pass unpacks
  `v_pk_*_f32` next to MFMAs into scalar ops so they can co-issue; there is no such workaround for dot instructions.
  The measurements differ from that list in two places: `v_pk_fma_f16` co-issues, and `v_exp_f32` co-issues up to
  2 per MFMA. The CDNA3 / CDNA4 ISA guides treat dot instructions as their own class ("DLops") but state no co-issue
  rule. Whether the cause is a shared datapath or an issue rule is not shown.

**Trying a form that co-issues: two `v_fma_mix_f32` per logit.** `v_fma_mix_f32 acc, a', b, acc` with `op_sel` selecting
the low and then the high fp16 half computes the same `acc + a'0 b0 + a'1 b1` from the same packed registers, and it
co-issues. Built into the kernel as the `vp=4` placement (next tile's term in the memory stage; `fa/` does not contain
this variant, code at `/shared/ybouquet/rope_frozen/analysis/vdot2_overlap_fa/`), exact to `v_dot2` within fp32
rounding (logits differ by ≤ 3.8e-6 on logits up to 46), every timed variant's O / LSE checked at T = 2048 and 16384.
Fastest of each (jobs 92444 1B / 92445 8B, 5 interleaved repeats, µs; ratios median [min, max] of per-repeat ratios):

| model | T | timing | Had | 2DP MFMA | 2DP `v_dot2` | 2DP 2x `fma_mix` | MFMA vs Had | `v_dot2` vs Had | 2x `fma_mix` vs Had | 2x `fma_mix` vs `v_dot2` |
|---|---|---|---|---|---|---|---|---|---|---|
| 1B | 2048 | FA | 37.3 | 39.5 | 44.1 | 46.6 | +5.8 % [+5.7, +6.1] | +18.1 % [+17.6, +18.7] | +25.1 % [+24.5, +25.3] | +5.7 % [+5.6, +6.0] |
| 1B | 2048 | e2e | 89.6 | 86.9 | 91.8 | 94.2 | −3.1 % [−3.1, −3.0] | +2.5 % [+2.3, +2.6] | +5.1 % [+4.8, +5.4] | +2.7 % [+2.1, +2.8] |
| 1B | 4096 | FA | 138.6 | 146.8 | 172.2 | 184.3 | +6.1 % [+5.7, +6.3] | +24.2 % [+24.0, +24.5] | +32.8 % [+32.7, +33.2] | +7.0 % [+6.9, +7.1] |
| 1B | 4096 | e2e | 216.0 | 210.6 | 236.2 | 248.5 | −2.6 % [−2.7, −2.5] | +9.4 % [+9.2, +9.5] | +15.0 % [+14.8, +15.1] | +5.2 % [+5.0, +5.4] |
| 1B | 8192 | FA | 545.5 | 581.9 | 674.2 | 715.1 | +6.7 % [+6.4, +7.0] | +23.6 % [+23.5, +23.7] | +31.0 % [+30.8, +31.9] | +6.1 % [+5.8, +6.7] |
| 1B | 8192 | e2e | 701.9 | 708.6 | 796.7 | 834.8 | +0.9 % [+0.7, +1.2] | +13.5 % [+13.3, +13.7] | +19.0 % [+18.6, +19.7] | +4.8 % [+4.3, +5.6] |
| 1B | 16384 | FA | 2198.7 | 2336.1 | 2683.6 | 2833.1 | +6.2 % [+5.7, +7.0] | +22.3 % [+21.5, +22.8] | +28.9 % [+28.4, +29.4] | +5.6 % [+5.4, +5.9] |
| 1B | 16384 | e2e | 2480.2 | 2568.8 | 2902.2 | 3044.5 | +3.7 % [+3.1, +4.5] | +17.0 % [+16.8, +18.6] | +23.1 % [+22.2, +23.6] | +4.8 % [+4.2, +5.3] |
| 8B | 2048 | FA | 62.2 | 64.4 | 69.1 | 72.6 | +3.6 % [+3.4, +3.7] | +11.0 % [+10.8, +11.3] | +16.7 % [+16.3, +17.2] | +5.0 % [+4.8, +5.3] |
| 8B | 2048 | e2e | 190.6 | 177.8 | 183.6 | 185.1 | −6.6 % [−6.9, −6.5] | −3.8 % [−3.9, −3.7] | −2.9 % [−2.9, −2.8] | +1.0 % [+0.8, +1.1] |
| 8B | 4096 | FA | 235.0 | 244.2 | 266.2 | 275.1 | +3.9 % [+3.7, +4.2] | +13.4 % [+13.2, +13.5] | +17.1 % [+17.1, +17.2] | +3.3 % [+3.3, +3.4] |
| 8B | 4096 | e2e | 496.3 | 473.9 | 499.0 | 505.0 | −4.6 % [−4.7, −4.1] | +0.5 % [+0.4, +0.8] | +1.9 % [+1.4, +2.2] | +1.2 % [+1.0, +1.4] |
| 8B | 8192 | FA | 947.5 | 989.1 | 1064.2 | 1099.4 | +4.3 % [+3.3, +4.9] | +12.2 % [+11.4, +13.6] | +16.1 % [+14.9, +16.5] | +3.2 % [+2.6, +3.6] |
| 8B | 8192 | e2e | 1411.9 | 1406.0 | 1484.0 | 1514.0 | −0.5 % [−0.6, −0.4] | +5.1 % [+4.6, +6.1] | +7.4 % [+7.0, +8.1] | +2.2 % [+1.1, +2.4] |
| 8B | 16384 | FA | 3757.9 | 3942.2 | 4302.8 | 4416.1 | +5.0 % [+4.3, +5.6] | +14.6 % [+14.0, +15.1] | +17.7 % [+17.1, +19.3] | +2.7 % [+1.7, +4.6] |
| 8B | 16384 | e2e | 4697.0 | 4732.3 | 5088.7 | 5230.1 | +1.0 % [+0.4, +1.0] | +8.3 % [+8.0, +8.5] | +11.5 % [+11.1, +11.7] | +2.9 % [+2.7, +3.1] |

- The pair co-issues but costs two half-rate vector instructions per logit (~2 x 4.2–4.9 cycles) where `v_dot2`
  costs one plus the MFMA blocking (~7.2 cycles per logit by subtraction); with no free MFMA shadow left in this
  kernel, the pair loses (+3 to +7 % on the FA kernel vs `v_dot2`).
- A full-rate pair (`v_mul_f32` then `v_fmac_f32`, ~2 x 2.25 cycles) would avoid both, but needs `b` in fp32 (more
  registers in kernels already at 254–256 VGPRs) and conflict-free register banks; not tested.
- Correctness note: inline-asm vector ops get no compiler-inserted MFMA wait states. In the `v_dot2` kernel the
  prologue and drain read an MFMA result right after it is produced (static audit); this is the likely cause of the
  variants that fail the correctness check, which are excluded everywhere.

Other `v_dot2` placements (in the QK^T stage, at the head of the softmax, under the P.V MFMAs) were slower; builds
with the scalarize compile flag gave wrong results for some of these other placements, which are not used.

```bash
python bench_vdot2.py --model 1b --only w8m256n128_vf1_lazy w8m256n128_vf1_lazy_prio w8m256n128_vf1_lazy_ns_scal --impls had mf_p0 vd_mk \
    --check-ts 2048 8192 --ts 2048 4096 8192 16384 --rounds 5 --e2e
python bench_vdot2.py --model 8b --only w8m256n64 w8m256n64_lazy_prio w8m256n64_lazy_ns_scal --impls had mf_p0 vd_m \
    --check-ts 2048 8192 --ts 2048 4096 8192 16384 --rounds 5 --e2e --u-path U_selfgen_8b_2dp_allrand_s0.pt --u-layer 15
```

#### MXFP4 vs fp16 flash attention

Baseline: the same kernel with an fp16 QK^T ([`fa/fmha_fp.py`](fa/fmha_fp.py), mode `QF16`): Q fp16 in registers, K
fp16 staged once per workgroup in LDS by direct-to-LDS loads, QK^T on fp16 32x32x16 MFMAs; softmax, P.V and the O /
LSE stores are the MXFP4 kernel's. Variant `f16q` folds the softmax scale into Q (as the tutorial's `SCALE_ON_Q`),
which fp4 cannot do. Equal treatment of fp16 and MXFP4:
- same toolchain, no scheduler plugin (asserted);
- same inputs: fp16 Q / K / V are views of the projection buffer whose V the MXFP4 kernels read (same strides); same
  outputs (O fp16 + LSE);
- same tile / knob / compile-flag search for every method, fastest of each method picked in the same job;
  configurations that spill registers are excluded;
- every timed variant's O / LSE checked against torch at T = 2048 and 16384.

Parity check against the tutorial `fmha_v3` (head_dim 128 only; it does not compile at 64; it is a reference, never
the baseline). At 8B the tutorial was first about 0.4 % faster than our fp16 mode (job 92423): our drain code spilled
registers under the scalarize compile flag. Scheduling fences between the drain steps (`drf`, offered to every method,
MXFP4 included) remove the spill; with them, our fp16 is as fast as or slightly faster than the tutorial (tutorial /
ours 1.003–1.007 with graph timing, T = 2k–16k, job 92429; per-T table below), and the main loops are
instruction-for-instruction the same.

FA kernel alone, 5 interleaved repeats, jobs 92428 (1B) and 92429 (8B). 2DP is the MFMA form. Times in µs (median of
the repeats); speedup = fp16 time / MXFP4 time, median [min, max] of the per-repeat ratios; TF/s from the median time.
fp16 is the faster of `f16` and `f16q` (named in the config column). Strict spill policy: configurations that spill
registers are excluded. Config names: tile prefix dropped (1B 8 warps 256 x 128, 8B 8 warps 256 x 64); `drf` drain
fences, `scal` scalarize compile flag, `nosink` no machine sink, `nosinkscal` both, other names as in the reproduction
table.

Bench timing (`do_bench`):

| model | T | fp16 (µs) | fp16 TF/s | MXFP4 Had (µs) | Had TF/s | MXFP4 2DP (µs) | 2DP TF/s | Had speedup | 2DP speedup | fastest fp16 / Had / 2DP config |
|---|---|---|---|---|---|---|---|---|---|---|
| 1B | 2048 | 49.6 | 693 | 39.0 | 880 | 41.8 | 823 | 1.28x [1.17, 1.33] | 1.19x [1.19, 1.20] | f16q `lazy_drf` / `lazy_vf1_drf` / `lazy_prio_vf1_drf_nosinkscal` |
| 1B | 4096 | 181.2 | 759 | 139.6 | 984 | 148.9 | 923 | 1.30x [1.30, 1.30] | 1.22x [1.22, 1.22] | f16q `lazy_drf` / `lazy_prio_vf1_drf` / `lazy_prio_vf1_drf` |
| 1B | 8192 | 731.3 | 752 | 549.1 | 1001 | 599.2 | 917 | 1.33x [1.32, 1.34] | 1.22x [1.21, 1.25] | f16q `lazy_drf` / `lazy_vf1_e_drf` / `lazy_prio_vf1` |
| 1B | 16384 | 2958.2 | 743 | 2220.3 | 990 | 2382.9 | 923 | 1.33x [1.32, 1.34] | 1.24x [1.23, 1.25] | f16q `lazy_drf` / `lazy_prio_vf1` / `lazy_prio_vf1_drf` |
| 8B | 2048 | 80.0 | 859 | 61.4 | 1119 | 63.0 | 1090 | 1.29x [1.28, 1.32] | 1.27x [1.25, 1.27] | f16q `drf_nosinkscal` / `lazy_drf_scal` / `lazy_prio_e_drf_nosink` |
| 8B | 4096 | 304.0 | 904 | 229.2 | 1199 | 235.6 | 1167 | 1.33x [1.32, 1.34] | 1.29x [1.29, 1.30] | f16q `drf_scal` / `lazy_nosinkscal` / `lazy_prio_e_drf_nosink` |
| 8B | 8192 | 1221.3 | 900 | 913.4 | 1204 | 947.8 | 1160 | 1.34x [1.32, 1.34] | 1.29x [1.28, 1.29] | f16q `drf_scal` / `lazy_drf_nosinkscal` / `lazy_prio_e_drf_nosink` |
| 8B | 16384 | 4831.1 | 910 | 3649.4 | 1205 | 3762.5 | 1169 | 1.32x [1.31, 1.33] | 1.28x [1.28, 1.29] | f16q `drf_nosinkscal` / `lazy_drf_scal` / `lazy_prio_e_drf_nosink` |

CUDA-graph replay (no host launch overhead), same picture:

| model | T | fp16 (µs) | fp16 TF/s | MXFP4 Had (µs) | Had TF/s | MXFP4 2DP (µs) | 2DP TF/s | Had speedup | 2DP speedup | fastest fp16 / Had / 2DP config |
|---|---|---|---|---|---|---|---|---|---|---|
| 1B | 2048 | 50.0 | 687 | 38.6 | 890 | 40.9 | 840 | 1.30x [1.29, 1.30] | 1.22x [1.22, 1.23] | f16q `vf1_drf` / `lazy_vf1_e_nosink` / `lazy_prio_vf1_nosinkscal` |
| 1B | 4096 | 189.7 | 725 | 146.6 | 938 | 156.2 | 880 | 1.29x [1.29, 1.30] | 1.21x [1.21, 1.21] | f16q `lazy_drf` / `lazy_vf1_e_drf` / `lazy_prio_vf1` |
| 1B | 8192 | 748.2 | 735 | 567.9 | 968 | 612.6 | 897 | 1.32x [1.32, 1.32] | 1.22x [1.22, 1.22] | f16q `lazy_drf` / `lazy_vf1_e_drf` / `lazy_prio_vf1_drf` |
| 1B | 16384 | 2971.8 | 740 | 2252.9 | 976 | 2425.8 | 907 | 1.32x [1.32, 1.32] | 1.23x [1.22, 1.23] | f16q `lazy_drf` / `lazy_vf1_e_nosink` / `lazy_prio_vf1` |
| 8B | 2048 | 81.9 | 839 | 62.4 | 1101 | 64.1 | 1072 | 1.31x [1.30, 1.31] | 1.28x [1.26, 1.28] | f16 `drf_scal` / `lazy_nosinkscal` / `lazy_prio_e_drf_nosink` |
| 8B | 4096 | 313.4 | 877 | 235.1 | 1169 | 241.2 | 1140 | 1.33x [1.32, 1.33] | 1.30x [1.28, 1.30] | f16q `drf_scal` / `lazy_nosinkscal` / `lazy_prio_e_drf_nosink` |
| 8B | 8192 | 1226.8 | 896 | 922.8 | 1192 | 950.8 | 1156 | 1.33x [1.33, 1.33] | 1.29x [1.29, 1.29] | f16q `drf_scal` / `lazy_nosinkscal` / `lazy_prio_e_drf_nosink` |
| 8B | 16384 | 4867.0 | 904 | 3643.7 | 1207 | 3760.0 | 1170 | 1.34x [1.33, 1.34] | 1.29x [1.29, 1.30] | f16q `drf_scal` / `lazy_nosinkscal` / `lazy_prio_e_drf_nosink` |

Tutorial parity, 8B (job 92429; strict policy): fastest tutorial `fmha_v3` build (`tut` / `tutq` = `SCALE_ON_Q`) vs our
fastest fp16 build; tutorial / ours > 1 means ours is faster (median [min, max] of the per-repeat ratio):

| T | timing | tutorial best (µs) | our fp16 best (µs) | tutorial / ours | tutorial config / our fp16 config |
|---|---|---|---|---|---|
| 2048 | bench | 79.8 | 80.0 | 0.999 [0.992, 1.016] | tutq `scal` / f16q `drf_nosinkscal` |
| 4096 | bench | 305.2 | 304.0 | 1.002 [0.993, 1.009] | tutq `scal` / f16q `drf_scal` |
| 8192 | bench | 1228.2 | 1221.3 | 1.005 [1.002, 1.011] | tutq `scal` / f16q `drf_scal` |
| 16384 | bench | 4851.5 | 4831.1 | 1.003 [1.001, 1.008] | tutq `scal` / f16q `drf_nosinkscal` |
| 2048 | graph | 82.1 | 81.9 | 1.003 [1.000, 1.016] | tutq `scal` / f16 `drf_scal` |
| 4096 | graph | 315.7 | 313.4 | 1.007 [1.006, 1.009] | tutq `scal` / f16q `drf_scal` |
| 8192 | graph | 1231.9 | 1226.8 | 1.004 [1.003, 1.005] | tutq `scal` / f16q `drf_scal` |
| 16384 | graph | 4881.6 | 4867.0 | 1.003 [1.003, 1.004] | tutq `scal` / f16q `drf_scal` |

Allowing configurations that spill outside the main loop (loose policy) changes nothing at 8B and lowers the 1B
speedups by at most 0.023x (median speedups, strict / loose):

| model | T | timing | fp16 best strict / loose (µs) | Had speedup strict / loose | 2DP speedup strict / loose | delta Had | delta 2DP |
|---|---|---|---|---|---|---|---|
| 1B | 2048 | bench | 49.6 / 49.6 | 1.280 / 1.280 | 1.188 / 1.188 | +0.000 | +0.000 |
| 1B | 4096 | bench | 181.2 / 179.7 | 1.299 / 1.289 | 1.217 / 1.207 | −0.010 | −0.010 |
| 1B | 8192 | bench | 731.3 / 727.6 | 1.331 / 1.324 | 1.218 / 1.214 | −0.006 | −0.004 |
| 1B | 16384 | bench | 2958.2 / 2914.2 | 1.330 / 1.313 | 1.243 / 1.223 | −0.017 | −0.020 |
| 1B | 2048 | graph | 50.0 / 49.7 | 1.297 / 1.287 | 1.223 / 1.214 | −0.010 | −0.010 |
| 1B | 4096 | graph | 189.7 / 189.7 | 1.294 / 1.294 | 1.214 / 1.214 | +0.000 | +0.000 |
| 1B | 8192 | graph | 748.2 / 742.6 | 1.318 / 1.308 | 1.222 / 1.213 | −0.010 | −0.009 |
| 1B | 16384 | graph | 2971.8 / 2922.4 | 1.320 / 1.297 | 1.225 / 1.205 | −0.023 | −0.020 |
| 8B | 2048 | bench | 80.0 / 80.0 | 1.287 / 1.287 | 1.269 / 1.269 | +0.000 | +0.000 |
| 8B | 4096 | bench | 304.0 / 304.0 | 1.326 / 1.326 | 1.290 / 1.290 | +0.000 | +0.000 |
| 8B | 8192 | bench | 1221.3 / 1221.3 | 1.338 / 1.338 | 1.288 / 1.288 | +0.000 | +0.000 |
| 8B | 16384 | bench | 4831.1 / 4831.1 | 1.324 / 1.324 | 1.285 / 1.285 | +0.000 | +0.000 |
| 8B | 2048 | graph | 81.9 / 81.9 | 1.312 / 1.312 | 1.278 / 1.278 | +0.000 | +0.000 |
| 8B | 4096 | graph | 313.4 / 313.4 | 1.333 / 1.333 | 1.299 / 1.299 | +0.000 | +0.000 |
| 8B | 8192 | graph | 1226.8 / 1226.8 | 1.330 / 1.330 | 1.291 / 1.291 | +0.000 | +0.000 |
| 8B | 16384 | graph | 4867.0 / 4867.0 | 1.336 / 1.336 | 1.295 / 1.295 | +0.000 | +0.000 |

Caveats: at 1B nothing outside our kernel confirms that its fp16 mode is competitive (the tutorial does not compile
at head_dim 64); its fp16 Q takes more registers than the MXFP4 Q, and no fp16-specific register saving was tried
(estimated remaining fp16 upside ≤ 2 % at 1B). 2DP's extra rank-2 MFMA is not counted in its TF/s. The `v_dot2` form
of 2DP is not in this comparison.

```bash
python bench_fp.py --model 1b --mode full --ts 2048 4096 8192 16384 --check-ts 2048 16384 --rounds 5 --screen-t 8192
python bench_fp.py --model 8b --mode full --ts 2048 4096 8192 16384 --check-ts 2048 16384 --rounds 5 --screen-t 8192 \
    --u-path U_selfgen_8b_2dp_allrand_s0.pt --u-layer 15
```

Where the overhead came from (switch-off ablations, wrong outputs, diagnostic only; T = 8192):

- **Not the MFMA or VALU throughput.** The kernel runs at 23–25 % MFMA utilization. Removing exp or the
  rescale leaves the 2DP gap at 12–13 %.
- **The K loads.** In the first port every one of the 8 waves loads the whole K tile from global memory with
  lanes = tokens (token stride `n_heads * 64` B): each `global_load_dwordx4` touches 32 cache lines, plus the
  scale loads and, for 2DP, the coefficient loads. Removing the in-loop K loads takes Hadamard from 1256 to
  797 µs and leaves 2DP only +2.7 % slower. Staging K once per workgroup in LDS (`L`) fixes this for both
  methods (Hadamard −24 %).
- **What remains** with `L` (job 92078, same knobs for both) is ~45 µs (+4.7 %), spread over the loop: with K loads removed +3.2 %, with exp
  and rescale removed +3.9 %, with K and V loads removed +2.2 %.

K-tile LDS reads (`L`, checked lane by lane against a measured gfx950 LDS bank model, job 92155):
- **K codes:** each wave reads the whole 4 KB tile with 4 `ds_read_b128`, 1 KB contiguous per instruction:
  conflict-free. Each K tile is written to LDS once per workgroup by direct-to-LDS loads.
- **Scales:** 4 `ds_read_u8`, conflict-free. Reordering the four scale blocks (same order for Q, K codes and
  scales; QK^T unchanged) gives 2 `ds_read_u16`: −0.5 % (Had) / −0.6 % (2DP) at T = 16384, within the
  2–4 µs noise at T = 8192 (job 92168).
- **2DP coefficients:** the only conflict, 2-way on a half-rate `ds_read2st64_b64` (4× the cost of one
  `ds_read_b128`). A lane-major layout read with one `ds_read_b128` removes it but does not speed the kernel up.
- **Global memory:** relaxing the in-loop `vmcnt` waits changes nothing, so the loop does not wait on global
  memory in steady state. Predicating off the loads of the waves whose LDS write is discarded costs ~45 µs.

### Toolchain: scheduler plugin and `fmha_v4`

The tutorial's `fmha_v4` and the LLIR scheduler plugin aborted with `LLVM ERROR: Option 'default' already
exists!`. Cause: `libtriton.so` exports its LLVM with default visibility, and the ROCm SDK `libLLVM.so.23`
(loaded by `libamd_comgr` when HIP initializes) binds its data references to libtriton's copy, so its static
constructors register the same `cl::opt` twice. The plugin is not involved; the same clash explains the
torch-before-triton crash and the `rocprofv3` segfault.

Fix, with Triton tag v2.0 (has `warp_predicate`):
- `TRITON_BACKENDS_IN_TREE=1` (v2 otherwise finds 0 drivers when installed through a `.pth`).
- A copy of the plugin with `patchelf --add-needed libtriton.so --set-rpath <triton>/python/triton/_C`.
- Import triton with `RTLD_LOCAL` first, then torch, then preload the plugin with `RTLD_LOCAL | RTLD_DEEPBIND`.

Tutorial fp16 attention, N = 8192, MI350X, outputs correct: `fmha_v4` 912 → 1023 TF/s with the plugin,
`fmha_v3` 874 → 953 TF/s. On the first MXFP4 port the plugin cut the 2DP vs Hadamard FA gap from +9.1 % to
+6.3 % (job 91978).
