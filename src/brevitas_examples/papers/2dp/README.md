# 2DP: an exact DC plane for low-precision QK^T

This folder holds the configurations of the 2DP experiments on MXFP4 attention.
The method lives in the LLM example:

| file | content |
|---|---|
| `llm/dc_extract_sdpa.py` | `DCExtractQuantScaledDotProductAttention`: 2DP split, Q/K Hadamard baseline, shrinkage corrections, frozen plane |
| `llm/sqnr_decomp.py` | `--sqnr-decomp`: per-layer error decomposition and shrinkage chain |
| `llm/mf_probe.py` | `--mf-probe-gamma`, `--dc-sinkbal`: sink diagnostics |
| `graph/equalize.py` | `had_rand` rotation mode and the `sdpa_value_*` options (R2 policy) |

The gfx950 kernels and the 2DP vs Hadamard latency benchmark are in [`kernels/`](kernels/README.md).

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
- `qk_only/`: only Q and K are quantized (`attn_quant_config: qk`), in MXFP4 or, with
  `--dc-tcast-scale e4m3fn_t32`, with E4M3 block scales; weights, linear inputs, V and softmax stay in
  bf16, no rotation. Llama-3.2 1B/3B, Llama-3.1 8B/70B, Qwen3-4B, Mistral-7B.

Run from the brevitas root. Each setting takes three runs: Hadamard, 2DP (which calibrates the plane,
saves it to `--dc-u-path` and prints `gamma`), then 2DP + unshrink with that `gamma`, loading the plane:

```bash
C=src/brevitas_examples/papers/2dp/quarot
# Hadamard
python -m brevitas_examples.llm.main --config $C/llama31-8b-had.yml
# 2DP: calibrate and save the plane, print gamma ([shrink-xl] ... gamma_attn(pred))
python -m brevitas_examples.llm.main --config $C/llama31-8b-2dp.yml \
    --dc-u-path U_llama31-8b_quarot.pt --sqnr-decomp 16
# 2DP + unshrink
python -m brevitas_examples.llm.main --config $C/llama31-8b-2dp.yml \
    --dc-u-path U_llama31-8b_quarot.pt --dc-correction unshrink --dc-gamma 0.047
```

The Q/K-only setting uses the same three runs with the `qk_only/` configs, for example on Llama-3.2-1B:

```bash
C=src/brevitas_examples/papers/2dp/qk_only
python -m brevitas_examples.llm.main --config $C/llama32-1b-had.yml
python -m brevitas_examples.llm.main --config $C/llama32-1b-2dp.yml \
    --dc-u-path U_llama32-1b_qk_mxfp4.pt --sqnr-decomp 16
python -m brevitas_examples.llm.main --config $C/llama32-1b-2dp.yml \
    --dc-u-path U_llama32-1b_qk_mxfp4.pt --dc-correction unshrink --dc-gamma <gamma>
```

For E4M3 block scales, add `--dc-tcast-scale e4m3fn_t32` to all three runs and use a separate plane
file: the plane is calibrated on the quantized model, so it depends on the format. `gamma` also
depends on the model and the format.

## Results

Llama-3.1, `quarot/` setting, full WikiText-2 test set (perplexity, lower is better):

| model | float | Hadamard | 2DP | Hadamard + unshrink | 2DP + unshrink |
|---|---|---|---|---|---|
| Llama-3.1-8B (`gamma` = 0.047) | 5.603 | 9.277 | 9.502 | 9.219 | **8.825** |
| Llama-3.1-70B (`gamma` = 0.046) | 2.486 | 13.325 | | 13.200 | **11.331** |

`gamma` is the shrinkage `1 - g_q g_k` measured by `--sqnr-decomp`. The 2DP numbers load a plane
calibrated in an earlier run (64 self-generated sequences) with `--dc-u-path`; runs that calibrate
the plane themselves have shown differences of up to about 0.08 in perplexity.

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
