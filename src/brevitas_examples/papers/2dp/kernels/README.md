# 2DP vs Hadamard: MXFP4 QK^T kernels on gfx950

These kernels and the benchmark time the full MXFP4 score path on an AMD MI350X/MI355X (gfx950):
QKV projection → RoPE → transform → MXFP4 → QK^T logits. They compare 2DP with the frozen plane and the
folded unshrink against the standard Hadamard pipeline.

| file | content |
|---|---|
| `p1_2dp.py` | P1 (Gluon): 8-wave ping-pong projection GEMM with the 2DP epilogue: RoPE, plane coordinates `a' = (1-gamma) a`, MXFP4 of the residual. `epi=0` is a plain fp16 store, used as the Hadamard projection |
| `p2_qk.py` | P2 (Gluon): fp4 QK^T (`mfma_scaled`), plus the 2DP rank-2 term `a'.b` on the matrix core or the VALU (`corr=1`); `corr=0` is the bare QK^T |
| `had_j2.py` | Hadamard quantizer (Triton): RoPE + `H_D / sqrt(D)` + MXFP4 |
| `layout_2dp.py` | offline: weight column permutation, RoPE tables, per-tile plane tables |
| `reference.py` | torch fp32 references for the correctness check |
| `bench_2dp_vs_hadamard.py` | the benchmark |

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

- **Tile selection.** For each pipeline, the QK^T variant and tile are picked on the end-to-end chain
  (projection → QK^T), from the same 8 candidates. A QK^T kernel timed alone after the L2 flush does not
  rank the tiles as they run behind the front end. Picking tiles alone flipped the 8B T = 8192 result
  between jobs.
- **Repeats.** The two pipelines and their front ends are then timed in `--repeats` interleaved repeats
  (default 5). The table gives the median and the range of the per-repeat ratio.

## Results

MI350X, end to end, in µs, median of 5 interleaved repeats. The last column gives each 2DP vs Hadamard
ratio with the [min, max] of the per-repeat ratio.

| model | T | 2DP | Hadamard | 2DP vs Hadamard |
|---|---|---|---|---|
| Llama-3.2-1B | 1024 | 87.6 | 87.6 | 0.0 % [−0.1, +0.1] |
| Llama-3.2-1B | 2048 | **168.7** | 173.9 | −3.0 % [−3.4, −2.9] |
| Llama-3.2-1B | 4096 | **515.9** | 523.6 | −1.5 % [−1.6, −1.3] |
| Llama-3.2-1B | 8192 | **1900.3** | 1935.3 | −1.8 % [−1.9, −1.8] |
| Llama-3.1-8B | 1024 | **130.6** | 133.6 | −2.2 % [−2.4, −2.1] |
| Llama-3.1-8B | 2048 | **228.3** | 234.6 | −2.6 % [−2.7, −2.6] |
| Llama-3.1-8B | 4096 | **654.6** | 672.2 | −2.6 % [−2.8, −2.6] |
| Llama-3.1-8B | 8192 | 2223.4 | **2169.6** | +2.4 % [+2.2, +2.6] |

- **Front end.** The 2DP P1 is 3–25 % faster than Hadamard's J1 + J2, because RoPE, the plane and MXFP4 all
  run in the GEMM epilogue.
- **QK^T.** Measured inside the chain, the rank-2 term costs about 0 % at 1B. At 8B it costs up to 6 %
  (1841 vs 1730 µs at T = 8192). That cost grows with T², while the front-end saving grows with T. At
  8B T = 8192 the QK^T cost wins.
- **Large T.** Both pipelines are bound by the fp32 logit store.
