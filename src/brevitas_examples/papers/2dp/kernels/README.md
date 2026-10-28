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

Each pipeline is timed end to end with `triton.testing.do_bench` (median), from the projection to the fp32
logits. Stage times are also reported. Both QK^T kernels are autotuned over the same tile candidates.

## Results

MI350X, end to end, in µs, from this code:

| T | Llama-3.2-1B: 2DP | 1B: Hadamard | Llama-3.1-8B: 2DP | 8B: Hadamard |
|---|---|---|---|---|
| 1024 | 87.5 | 88.1 | **130.3** | 133.9 |
| 2048 | 173.6 | 173.8 | **228.6** | 240.4 |
| 4096 | 521.8 | 524.9 | **671.1** | 684.2 |
| 8192 | 1899.9 | 1933.4 | 2210.6 | 2174.8 |

- **Front end:** the 2DP P1 is 9–25 % faster than Hadamard's J1 + J2 (1B: 52–111 vs 57–148 µs; 8B: 93–382
  vs 105–436 µs), because RoPE, the plane and MXFP4 all run in the GEMM epilogue.
- **QK^T:** the rank-2 term costs −1 % to +4 %.
- **Large T:** both pipelines are bound by the fp32 logit store, which is ~4.8 TB/s at T = 8192 and the same
  for both. Run-to-run noise there is ±2–3 %, which covers the 8B T = 8192 difference.
