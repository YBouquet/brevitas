# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Rank-2 term variants of the 2DP QK^T (MFMA, 2 FMAs, v_dot2_f32_f16) against Hadamard, end to end.

For one (model, T): correctness of each variant against the torch target (first 2048 tokens), an end-to-end
tile sweep (p2_qk.tiles) per QK^T variant, then the best chain of each in 5 interleaved repeats.

    python bench_rank2_variants.py --model 8b --t 8192 --u-path U_selfgen_8b_2dp_allrand_s0.pt --u-layer 15
"""
# isort: off
import triton  # noqa: F401  -- import triton BEFORE torch
# isort: on
import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_2dp_vs_hadamard import build
from bench_2dp_vs_hadamard import load_u
from bench_2dp_vs_hadamard import med_us
from bench_2dp_vs_hadamard import MODELS
from bench_2dp_vs_hadamard import pipes
from p2_qk import qk
from p2_qk import tiles
from reference import mx_quant
from reference import ref_2dp
from reference import ref_logits
from reference import rope_std

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--model", choices=sorted(MODELS), default="1b")
ap.add_argument("--t", type=int, default=8192)
ap.add_argument("--gamma", type=float, default=0.047)
ap.add_argument("--u-path", default=None)
ap.add_argument("--u-layer", type=int, default=15)
args = ap.parse_args()
T, G = args.t, args.gamma
cfg = MODELS[args.model]
D, n_q, n_kv = cfg["hd"], cfg["n_q"], cfg["n_kv"]
U = load_u(cfg, args.u_path, args.u_layer, "cuda")
s = build(cfg, T, U, "cuda")
p1, p2 = pipes(cfg, s, G)
p1["p1_2dp"]()
p1["j1"]()
p1["j2"]()
torch.cuda.synchronize()
a2, _ = p2["qk_2dp_mfma"]
ah, _ = p2["qk_had"]
VAR = {
    "had": (ah, dict(corr=0), "had"),
    "mfma": (a2, dict(corr=1, corr_mfma=1), "2dp"),
    "fma": (a2, dict(corr=1, corr_mfma=0), "2dp"),
    "dot2": (a2, dict(corr=1, corr_mfma=2), "2dp")}
fe = {"2dp": p1["p1_2dp"], "had": lambda: (p1["j1"](), p1["j2"]())}

# correctness (first 2048 tokens of logits vs the torch target, x 1/(1-g))
n_qk, h_tot, gpg = n_q + n_kv, n_q + 2 * n_kv, n_q // n_kv
cut = min(T, 2048)
o = rope_std((s["x"][:cut].float() @ s["W"].float()).view(-1, h_tot, D)[:, :n_qk],
             s["cos"][:cut],
             s["sin"][:cut])
a, rz = ref_2dp(o, U, n_q, gpg)
dref, _, _ = mx_quant(rz)
target = ref_logits(dref[:, :n_q], dref[:, n_q:], gpg, a[:, :n_q], a[:, n_q:], 1 / (1 - G))
outs = {}
for v in ("mfma", "fma", "dot2"):
    args, kw, _ = VAR[v]
    qk(*args, **kw, cfg=(128, 128, 4, 2, 2, 1))
    torch.cuda.synchronize()
    lg = s["out"][:, :cut, :cut] / (1 - G)
    outs[v] = lg.clone()
    print(
        f"{v}: logits rel L2 vs target {((lg - target).norm() / target.norm()).item():.4e}",
        flush=True)
print(
    f"dot2 vs fma max |diff| {(outs['dot2'] - outs['fma']).abs().max().item():.3e}  "
    f"(max |logit| {outs['fma'].abs().max().item():.2f})",
    flush=True)
del outs, target, o, a, rz, dref
torch.cuda.empty_cache()

res = {}
for v, (args, kw, f) in VAR.items():
    for c in tiles(D):
        if T % (c[1] * c[5]):
            continue
        try:
            qk(*args, **kw, cfg=c)
            res[(v, c)] = med_us(lambda: (fe[f](), qk(*args, **kw, cfg=c)))
        except Exception as e:  # noqa: BLE001
            print(
                f"  [{v} {c} failed: {type(e).__name__}: {str(e).splitlines()[0][:160]}]",
                flush=True)
    top = sorted((t, c) for (vv, c), t in res.items() if vv == v)[:5]
    print(f"{v}: top-5 end to end " + "  ".join(f"{c} {t:.1f}" for t, c in top), flush=True)

best = {v: min((t, c) for (vv, c), t in res.items() if vv == v)[1] for v in VAR}
fns = {v: (lambda v=v: (fe[VAR[v][2]](), qk(*VAR[v][0], **VAR[v][1], cfg=best[v]))) for v in VAR}
fns.update(fe_2dp=fe["2dp"], fe_had=fe["had"])
names = list(fns)
t = {n: [] for n in names}
for r in range(5):
    for n in (names if r % 2 == 0 else names[::-1]):
        t[n].append(med_us(fns[n]))
md = {n: statistics.median(x) for n, x in t.items()}
print(f"\n== {cfg['name']} T={T}, best chains, median of 5 interleaved repeats")
for v in VAR:
    fq = md["fe_" + VAR[v][2]]
    vs = "" if v == "had" else f"  vs Had {100 * (md[v] / md['had'] - 1):+.2f} % " \
        f"[{min(100 * (x / y - 1) for x, y in zip(t[v], t['had'])):+.2f}, {max(100 * (x / y - 1) for x, y in zip(t[v], t['had'])):+.2f}]"
    print(f"  {v:<5} {str(best[v]):<26} e2e {md[v]:8.1f}  qk {md[v] - fq:8.1f}{vs}", flush=True)
