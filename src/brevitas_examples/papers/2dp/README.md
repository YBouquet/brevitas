# 2DP: an exact DC plane for low-precision QK^T

This folder holds the configurations of the 2DP experiments on MXFP4 attention.
The method lives in the LLM example:

| file | content |
|---|---|
| `llm/dc_extract_sdpa.py` | `DCExtractQuantScaledDotProductAttention`: 2DP split, Q/K Hadamard baseline, shrinkage corrections, frozen plane |
| `llm/sqnr_decomp.py` | `--sqnr-decomp`: per-layer error decomposition and shrinkage chain |
| `llm/mf_probe.py` | `--mf-probe-gamma`, `--dc-sinkbal`: sink diagnostics |
| `graph/equalize.py` | `had_rand` rotation mode and the `sdpa_value_*` options (R2 policy) |

## Method

After RoPE, the token mean of Q and K concentrates in the slow (low-frequency) half of the head.
For every KV group, 2DP builds an orthonormal plane `U` on that half, spanned by the query mean and the
key mean. The plane coordinates `a = qU`, `b = kU` stay in high precision; only the residuals
`r = q - aU^T`, `s = k - bU^T` are quantized, and the score is computed as two GEMMs:

```
S = Q(r) Q(s)^T / sqrt(d) + a b^T / sqrt(d)
```

which is exact before quantization for any orthonormal `U`. The plane is frozen: it is calibrated once
per layer and KV group on 64 sequences sampled from the float model (`--dc-u-calib selfgen`), so no
calibration data is needed.

Block quantizers shrink the score: `Q(x) ~ (1 - gamma) x`. `--dc-correction unshrink --dc-gamma g`
divides the quantized part of the score by `1 - g`. `gamma` is measured by `--sqnr-decomp`
(`[shrink-xl] ... gamma_attn(pred)`, `gamma_attn = 1 - g_q g_k`).

## Settings

- `quarot/`: Llama-3.1 8B and 70B, MXFP4 W4A4 (e2m1 elements, E8M0 floor-rounded block-32 scales),
  QuaRot rotations with R1 a random Hadamard and R2 (value/output), R3 (Q/K head) and R4 (down_proj)
  deterministic. `*-had.yml` uses the R3 Hadamard on Q/K, `*-2dp.yml` uses 2DP.
- `qk_only/`: only Q and K are quantized (MXFP4); weights, linear inputs, V and softmax stay in bf16,
  no rotation. Llama-3.2 1B/3B, Llama-3.1 8B/70B, Qwen3-4B, Mistral-7B.

Run from the brevitas root:

```bash
# Hadamard
python -m brevitas_examples.llm.main --config src/brevitas_examples/papers/2dp/quarot/llama31-8b-had.yml
# 2DP; the first run calibrates the plane and saves it to --dc-u-path, later runs load it
python -m brevitas_examples.llm.main --config src/brevitas_examples/papers/2dp/quarot/llama31-8b-2dp.yml \
    --dc-u-path U_llama31-8b.pt
# 2DP + unshrink, with the gamma measured by --sqnr-decomp
python -m brevitas_examples.llm.main --config src/brevitas_examples/papers/2dp/quarot/llama31-8b-2dp.yml \
    --dc-u-path U_llama31-8b.pt --dc-correction unshrink --dc-gamma 0.047
```

Useful options (see `python -m brevitas_examples.llm.main --help`, group "2DP"):

| option | effect |
|---|---|
| `--sqnr-decomp 16 [--mf-probe-gamma 0.047]` | print the per-layer decomposition over 16 evaluation chunks before the PPL |
| `--dc-u-basis pca`, `--dc-u-space full` | plane ablations: PCA plane, plane on the full head |
| `--dc-correction mean --dc-abar-mode causal` | subtract the shrinkage of the mean query instead of unshrinking |
| `--dc-sinkbal` | oracle diagnostic on the sink key (uses the float score) |
| `--dc-had-layers 0-3` | hybrid: Q/K Hadamard on these layers, 2DP elsewhere |
| `--dc-tcast-scale e4m3fn_t32` | E4M3 instead of E8M0 block scales for Q/K, via the [tcast](https://github.com/ROCm/tensorcast) emulator |
| `--rotation-sdpa-value-rand [--rotation-sdpa-value-rand-shared]`, `--dc-had-rand [--dc-had-rand-shared]` | random R2 and random R3 |

`--dc-tcast-scale` needs [tcast](https://github.com/ROCm/tensorcast) importable; it is not pip-installable, so
clone it and add the clone's root to `PYTHONPATH`. It replaces only the enabled attention quantizers (Q and K
with `attn_quant_config: qk`).

Llama-3.1-70B needs 4 GPUs (the model is dispatched with accelerate); the other models run on one.

## Reproducibility notes

- R1 (`--rotation-mode had_rand`) draws from the global torch RNG, seeded by `--seed`.
- Evaluation is the full WikiText-2 test set (`dataset_eval_split: test`, `nsamples: -1`).
