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

Full flash attention, 2DP (rank-2 term on MFMA) vs Hadamard, fastest vs fastest, MI350X:

| model | T | FA kernel 2DP vs Had | end to end 2DP vs Had | job |
|---|---|---|---|---|
| Llama-3.2-1B | 2048 / 4096 / 8192 / 16384 | +5.6 / +6.0 / +7.2 / +6.8 % | −1.8 / −2.9 / +0.9 / +4.2 % | 92366 |
| Llama-3.1-8B | 4096 / 8192 / 16384 | +4.2 / +4.9 / +4.2 % | −4.6 / −0.9 / +0.9 % | 92099 |

With the rank-2 term on the vector unit (`v_dot2`, one per logit) instead: 1B +18 / +25 / +24 / +23 % on the FA
kernel and +2 / +9 / +14 / +17 % end to end (T = 2k–16k, job 92415); 8B +13 / +12 / +14 % and 0 / +4 / +9 %
(T = 4k–16k, job 92416). See [MFMA vs `v_dot2`](#rank-2-term-mfma-vs-v_dot2-full-flash-attention).

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

MI350X, end to end, in µs. Each value is the median over 3 separate jobs; each job is itself the median of 5
interleaved repeats. The last column lists the 2DP vs Hadamard ratio of each of the 3 jobs.

| model | T | 2DP | Hadamard | 2DP vs Hadamard (3 jobs) |
|---|---|---|---|---|
| Llama-3.2-1B | 1024 | **85.8** | 88.0 | −2.6 / −2.2 / −2.6 % |
| Llama-3.2-1B | 2048 | **168.6** | 172.2 | −2.2 / −2.0 / −2.1 % |
| Llama-3.2-1B | 4096 | **497.2** | 500.9 | −0.9 / −0.8 / −0.5 % |
| Llama-3.2-1B | 8192 | 1912.3 | 1909.8 | −0.1 / +0.2 / +0.2 % |
| Llama-3.1-8B | 1024 | **129.2** | 134.0 | −3.8 / −3.8 / −3.2 % |
| Llama-3.1-8B | 2048 | **225.7** | 232.8 | −3.2 / −3.3 / −3.1 % |
| Llama-3.1-8B | 4096 | **657.6** | 673.1 | −2.3 / −2.5 / −2.5 % |
| Llama-3.1-8B | 8192 | **2177.1** | 2187.9 | −0.9 / −0.6 / −0.4 % |

- **Front end.** The 2DP P1 is 3–25 % faster than Hadamard's J1 + J2, because RoPE, the plane and MXFP4 all
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

T = 8192, end to end vs Hadamard, each variant on its best end-to-end tile; median of 5 interleaved repeats
in one job, with the [min, max] of the per-repeat ratio (jobs 91172 / 91173):

| | Hadamard (µs) | MFMA | FMA | `v_dot2` |
|---|---|---|---|---|
| Llama-3.2-1B | 1903.3 | +0.42 % [−3.58, +0.46] | **+0.01 %** [−3.92, +0.18] | +1.00 % [−2.88, +1.11] |
| Llama-3.1-8B | 2176.6 | +0.44 % [+0.36, +0.56] | +1.84 % [+1.49, +1.95] | **+0.07 %** [−0.08, +0.20] |

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
end-to-end tile; then the best chain of each in 5 interleaved repeats. The cells give 2DP vs Hadamard, end to
end (front end + P2), median and [min, max] of the per-repeat ratio; the fastest 2DP variant is in bold.
MI350X, one job per table.

**Partial kernel 1: logit store** (`p2_qk.qk`, T ≤ 8192). Logits rel L2 vs the torch target: 1.12e-2 (1B), 1.17e-2 (8B), the
same for the three variants.

Llama-3.2-1B (job 92106):

| T | Hadamard (µs) | MFMA | FMA | `v_dot2` |
|---|---|---|---|---|
| 1024 | 88.4 | −2.1 % [−2.2, −1.9] | **−2.4 %** [−2.4, −2.3] | −2.3 % [−2.3, −2.2] |
| 2048 | 173.7 | **−2.5 %** [−2.9, −2.3] | −1.9 % [−2.1, −1.7] | −2.5 % [−2.7, −2.0] |
| 4096 | 499.5 | **−0.4 %** [−0.6, −0.3] | +0.8 % [+0.6, +0.9] | +1.4 % [+1.3, +1.6] |
| 8192 | 1953.6 | **−2.5 %** [−2.6, −2.5] | −1.4 % [−1.5, −1.3] | −1.5 % [−1.6, −1.5] |

Llama-3.1-8B (job 92108):

| T | Hadamard (µs) | MFMA | FMA | `v_dot2` |
|---|---|---|---|---|
| 1024 | 134.9 | −3.8 % [−3.9, −3.7] | **−3.9 %** [−4.1, −3.9] | −3.7 % [−3.8, −3.6] |
| 2048 | 232.6 | **−2.0 %** [−2.1, −1.9] | +0.3 % [+0.1, +0.4] | −0.7 % [−0.7, −0.6] |
| 4096 | 671.2 | **−2.1 %** [−2.4, −2.0] | +0.2 % [+0.0, +0.3] | −0.3 % [−0.5, −0.1] |
| 8192 | 2186.2 | **−0.4 %** [−0.4, −0.3] | +1.3 % [+1.2, +1.6] | +0.1 % [−0.0, +0.1] |

**Partial kernel 2: QK^T + online softmax** (`p2_fa_stats.qk_stats`, no P.V, no output).
`m` / `l` vs torch: max abs 1e-6 / max rel 7e-7 for every variant.

Llama-3.2-1B (job 92107):

| T | Hadamard (µs) | MFMA | FMA | `v_dot2` |
|---|---|---|---|---|
| 1024 | 71.4 | +7.7 % [+7.2, +7.8] | +8.5 % [+8.4, +8.9] | **−0.8 %** [−1.1, −0.7] |
| 2048 | 106.5 | +20.5 % [+20.4, +20.6] | +26.0 % [+26.0, +26.1] | **+2.4 %** [+2.3, +2.4] |
| 4096 | 245.8 | +37.0 % [+36.7, +37.3] | +50.5 % [+50.3, +51.0] | **+9.2 %** [+9.0, +9.5] |
| 8192 | 850.7 | +45.8 % [+45.6, +46.2] | +62.2 % [+61.3, +64.0] | **+15.8 %** [+15.0, +16.2] |
| 16384 | 2958.2 | +55.1 % [+48.7, +56.7] | +74.1 % [+66.6, +75.7] | **+20.8 %** [+16.7, +21.4] |

Llama-3.1-8B (job 92109):

| T | Hadamard (µs) | MFMA | FMA | `v_dot2` |
|---|---|---|---|---|
| 1024 | 121.6 | −1.3 % [−2.0, −1.2] | −2.7 % [−2.8, −2.3] | **−6.4 %** [−6.5, −6.2] |
| 2048 | 176.0 | +8.5 % [+8.3, +9.0] | +8.6 % [+8.0, +8.6] | **−3.2 %** [−3.5, −1.6] |
| 4096 | 453.5 | +17.2 % [+14.8, +17.8] | +18.0 % [+17.8, +18.4] | **−0.6 %** [−1.6, +0.5] |
| 8192 | 1332.9 | +31.5 % [+29.7, +32.6] | +34.8 % [+33.2, +35.1] | **+6.1 %** [+5.5, +6.5] |
| 16384 | 4734.7 | +32.4 % [+25.7, +33.3] | +36.2 % [+27.7, +37.5] | **+5.1 %** [+4.9, +5.5] |

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
the real kernel (e.g. 1B T = 16384: +20.8 % here, +4.2 % end to end in the full kernel).*

P2 = `p2_fa_stats.qk_stats`: fp4 QK^T, rank-2 term, exact online softmax (row max and row sum), `m` and `l`
stored. P.V is omitted. Each method is tuned on the end-to-end chain over the rank-2 variants and tiles; median
of 5 interleaved repeats, one job (91651). `m` / `l` match the torch reference (max rel 7e-7).

| model | T | 2DP (µs) | Hadamard (µs) | 2DP vs Hadamard | P2 2DP / Had | front end 2DP / Had |
|---|---|---|---|---|---|---|
| 8B | 1024 | **114.4** | 117.9 | −2.9 % | 27.5 / 25.2 | 91.3 / 97.0 |
| 8B | 2048 | **172.7** | 182.4 | −5.3 % | 72.8 / 67.1 | 109.0 / 124.2 |
| 8B | 4096 | **461.2** | 463.6 | −0.5 % | 259.2 / 236.4 | 216.9 / 248.4 |
| 8B | 8192 | 1447.0 | 1373.5 | +5.4 % | 992.2 / 900.5 | 382.5 / 440.8 |
| 8B | 16384 | 5020.7 | 4772.0 | +5.3 % | 3885.0 / 3513.0 | 770.3 / 885.6 |
| 1B | 1024 | **71.0** | 71.7 | −1.0 % | 22.3 / 21.5 | 52.2 / 53.9 |
| 1B | 2048 | 109.0 | 107.0 | +1.8 % | 61.5 / 53.4 | 51.8 / 57.3 |
| 1B | 4096 | 271.0 | 248.6 | +8.9 % | 220.8 / 181.4 | 58.3 / 71.8 |
| 1B | 8192 | 983.5 | 853.1 | +15.3 % | 849.8 / 681.7 | 112.0 / 144.7 |
| 1B | 16384 | 3601.3 | 3078.9 | +16.1 % | 3337.3 / 2644.3 | 207.2 / 272.5 |

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
checked at T = 2048 and 16384, O rel L2 ≤ 2.9e-4), fastest Hadamard vs fastest 2DP:

| model | T | FA kernel Had / 2DP (µs) | FA 2DP vs Had | end to end 2DP vs Had |
|---|---|---|---|---|
| 1B | 2048 | 36.5 / 38.5 | +5.4 % [+4.6, +6.0] | −4.7 % [−5.0, −4.6] |
| 1B | 4096 | 134.7 / 143.2 | +6.1 % [+5.9, +6.3] | −2.5 % [−2.6, −2.3] |
| 1B | 8192 | 523.0 / 558.9 | +6.9 % [+6.6, +7.0] | +1.4 % [+1.1, +1.4] |
| 1B | 16384 | 2105.0 / 2241.5 | +6.4 % [+6.0, +7.3] | +3.8 % [+3.0, +4.2] |
| 8B | 4096 | 223.4 / 231.4 | +3.7 % [+3.5, +3.8] | −4.5 % [−4.8, −4.3] |
| 8B | 8192 | 880.5 / 918.9 | +4.4 % [+4.3, +4.6] | −0.6 % [−1.7, +0.2] |
| 8B | 16384 | 3556.2 / 3696.4 | +3.7 % [+3.3, +4.8] | +0.6 % [−0.4, +0.9] |

Within noise of the original jobs (92366 for 1B, 92099 for 8B); the 1B T = 2048 end-to-end ratio is known to move
by about ±2 points between jobs. The `readme1b` set has only the 256 x 128 tile, which is not the fastest at
T = 1024 (job 92270 used 4 warps, 128 x 128 there).

Llama-3.1-8B, T = 8192, FA kernel alone, µs, median of 5 interleaved repeats. Every K-path change is applied
to both methods; the comparison is always fastest Hadamard against fastest 2DP.

| K path | Hadamard | 2DP | 2DP vs Hadamard | job |
|---|---|---|---|---|
| K in registers, global loads in mem1 (first port) | 1256 | 1434 | +14 % | 92059 |
| same, rank-2 as fp16 32x32x16 MFMA with the fp4 accumulator as srcC, scales loaded first | 1166 | 1198 | +2.8 % | 91982 |
| K through LDS, prefetched (`lds3`) | 1027 | 1073 | +4.8 % | 92012 |
| **K through LDS with direct-to-LDS loads, tutorial schedule (`L`)** | **959** | **1005** | **+4.7 %** | 92078 |

Final run (job 92099), 8B, both K through LDS, then each method on its fastest scheduling knobs (Hadamard:
lazy rescale + no machine sink + scalarized packed ops; 2DP: lazy rescale + `s_setprio`). Median of 5
interleaved repeats, [min, max] of the per-repeat ratio:

| T | FA kernel Had / 2DP (µs) | FA 2DP vs Had | end to end Had / 2DP (µs) | end to end 2DP vs Had |
|---|---|---|---|---|
| 4096 | 226.2 / 235.6 | +4.2 % [+3.9, +4.5] | 478.1 / 455.8 | **−4.6 %** [−4.7, −4.3] |
| 8192 | 903.0 / 949.9 | +4.9 % [+4.0, +6.1] | 1359.3 / 1343.0 | **−0.9 %** [−2.5, −0.7] |
| 16384 | 3621.4 / 3776.4 | +4.2 % [+3.1, +4.9] | 4491.5 / 4527.3 | +0.9 % [+0.3, +1.3] |

The first port in the same job: FA +11.6 / +12.3 / +15.4 %, end to end +0.5 / +6.9 / +12.5 %.
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
`s_setprio` + dot2 fence. Job 92366 (T = 1024: job 92270), 5 interleaved repeats, O / LSE checked at T = 2048
and 16384:

| T | FA kernel Had / 2DP (µs) | FA 2DP vs Had | end to end 2DP vs Had |
|---|---|---|---|
| 1024 | 16.6 / 17.7 | +6.7 % [+5.2, +8.1] | +1.5 % |
| 2048 | 37.1 / 39.2 | +5.6 % [+5.5, +5.7] | −1.8 % [−2.0, −1.8] |
| 4096 | 137.8 / 146.0 | +6.0 % [+5.9, +6.4] | −2.9 % [−2.9, −2.8] |
| 8192 | 540.0 / 578.7 | +7.2 % [+7.0, +8.1] | +0.9 % [+0.8, +1.0] |
| 16384 | 2172.2 / 2322.6 | +6.8 % [+6.1, +7.6] | +4.2 % [+3.7, +5.0] |

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
reported variant's O / LSE checked at T = 2048 and 8192 (O rel L2 ≤ 2.9e-4):

| model | T | FA kernel Had / MFMA / `v_dot2` (µs) | FA: MFMA vs Had | FA: `v_dot2` vs Had | e2e: MFMA vs Had | e2e: `v_dot2` vs Had |
|---|---|---|---|---|---|---|
| 1B | 2048 | 37.3 / 39.6 / 44.0 | +6.0 % | +17.7 % [+17.4, +18.3] | −3.9 % | +1.7 % |
| 1B | 4096 | 138.6 / 146.8 / 172.4 | +6.0 % | +24.5 % [+24.2, +24.7] | −3.0 % | +8.9 % |
| 1B | 8192 | 545.6 / 581.2 / 675.8 | +6.5 % | +24.0 % [+23.4, +24.3] | +1.1 % | +13.7 % |
| 1B | 16384 | 2193.7 / 2334.5 / 2685.4 | +6.4 % | +22.6 % [+21.6, +23.2] | +3.8 % | +17.2 % |
| 8B | 2048 | 60.2 / 63.1 / 67.7 | +4.8 % | +12.4 % [+12.2, +12.7] | −6.1 % | −3.4 % |
| 8B | 4096 | 229.6 / 239.1 / 259.7 | +4.2 % | +13.1 % [+12.8, +13.3] | −4.6 % | −0.0 % |
| 8B | 8192 | 920.6 / 959.4 / 1029.8 | +4.2 % | +11.6 % [+11.0, +15.0] | −0.8 % | +4.2 % |
| 8B | 16384 | 3644.1 / 3800.1 / 4145.8 | +4.3 % | +13.6 % [+13.2, +14.0] | +0.8 % | +9.4 % |

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
  92406 / 92413). With the 2DP `v_dot2` kernel the SIMD time per wave-tile grows by 352 cycles for 123 more busy
  vector cycles, and the MFMA / vector overlap drops from 19 % to 6 % of cycles (counters, job 92409): the extra
  `v_dot2`s also stop MFMA work from running in parallel with the vector work. Why `v_dot2` breaks that overlap while
  plain fma does not is not measured (3-operand vector ops compete with the MFMAs for register-file read ports, which
  would fit).
- Removing the barrier lockstep (no warp pipeline) brings each added `v_dot2` down to its ~4-cycle issue slot
  (7.3 cycles with the pipeline), so the ping-pong schedule is what doubles its cost; but that kernel starts about
  19 % slower and is still slower at the 32 `v_dot2` per wave-tile that 2DP needs. A third wave per SIMD did not
  help (+18.4 % for 32 dummy `v_dot2` vs +18.9 % with 2 waves, job 92413). Hiding the `v_dot2` work would need a schedule that keeps the pipeline and places the `v_dot2`s in its stall
  windows.

Other `v_dot2` placements (in the QK^T stage, at the head of the softmax, under the P.V MFMAs) were slower; builds
with the scalarize compile flag gave wrong results for some placements and are not used.

```bash
python bench_vdot2.py --model 1b --only w8m256n128_vf1_lazy w8m256n128_vf1_lazy_prio --impls had mf_p0 vd_mk \
    --check-ts 2048 8192 --ts 2048 4096 8192 16384 --rounds 5 --e2e
python bench_vdot2.py --model 8b --only w8m256n64 w8m256n64_lazy_prio --impls had mf_p0 vd_m \
    --check-ts 2048 8192 --ts 2048 4096 8192 16384 --rounds 5 --e2e --u-path U_selfgen_8b_2dp_allrand_s0.pt --u-layer 15
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
