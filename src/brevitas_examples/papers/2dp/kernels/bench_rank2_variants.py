# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Rank-2 term variants of the 2DP P2 (MFMA, 2 FMAs, v_dot2_f32_f16) against Hadamard, end to end, over T.

For each T: correctness of each variant at the first T, an end-to-end tile sweep per variant (each variant gets
its own best tile), then the best chain of each variant and of Hadamard in --repeats interleaved repeats.
  --endpoint logits  P2 = p2_qk.qk: fp32 logits stored (T <= 8192: the logit offsets are int32)
  --endpoint fa      P2 = p2_fa_stats.qk_stats: QK^T + online softmax, only m / l stored

    python bench_rank2_variants.py --model 8b --endpoint fa --u-path U_selfgen_8b_2dp_allrand_s0.pt --u-layer 15
"""
# isort: off
import triton  # noqa: F401  -- import triton BEFORE torch
# isort: on
import argparse
import json
import math
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
from p2_fa_stats import qk_stats
from p2_fa_stats import stats_tiles
from p2_qk import qk
from p2_qk import tiles
from reference import mx_quant
from reference import ref_2dp
from reference import ref_logits
from reference import rope_std

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--model", choices=sorted(MODELS), default="1b")
ap.add_argument("--endpoint", choices=("logits", "fa"), default="logits")
ap.add_argument("--ts", type=int, nargs="+", default=None)
ap.add_argument("--gamma", type=float, default=0.047)
ap.add_argument("--u-path", default=None)
ap.add_argument("--u-layer", type=int, default=15)
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--out", default=None)
args = ap.parse_args()
FA = args.endpoint == "fa"
TS = args.ts or ([1024, 2048, 4096, 8192, 16384] if FA else [1024, 2048, 4096, 8192])
assert FA or max(TS) <= 8192, "the logit-store P2 uses int32 offsets: T <= 8192"
G = args.gamma
cfg = MODELS[args.model]
D, n_q, n_kv = cfg["hd"], cfg["n_q"], cfg["n_kv"]
n_qk, h_tot, gpg = n_q + n_kv, n_q + 2 * n_kv, n_q // n_kv
U = load_u(cfg, args.u_path, args.u_layer, "cuda")
sc_had = math.log2(math.e) / math.sqrt(D)
sc_2dp = sc_had / (1.0 - G)
CORR = {"mfma": 1, "fma": 0, "dot2": 2}
result = dict(
    model=cfg, endpoint=args.endpoint, gamma=G, device=torch.cuda.get_device_name(0), runs={})
print(f"{cfg['name']} on {result['device']}, endpoint {args.endpoint}", flush=True)

for ti, T in enumerate(TS):
    s = build(cfg, T, U, "cuda")
    p1, p2 = pipes(cfg, s, G)
    p1["p1_2dp"]()
    p1["j1"]()
    p1["j2"]()
    torch.cuda.synchronize()
    a2, _ = p2["qk_2dp_mfma"]
    ah, _ = p2["qk_had"]
    fe = {"2dp": p1["p1_2dp"], "had": lambda: (p1["j1"](), p1["j2"]())}
    if FA:
        mo = torch.empty(n_q, T, device="cuda", dtype=torch.float32)
        lo = torch.empty_like(mo)
        VAR = {"had": (ah[:6], dict(corr=0, qk_scale=sc_had), "had")}
        VAR.update({v: (a2[:6], dict(corr=1, qk_scale=sc_2dp, corr_mfma=c), "2dp") for v, c in CORR.items()})
        cands = [c for c in stats_tiles(D) if T % c[0] == 0 and T % c[1] == 0]

        def p2f(a, kw, c):
            return lambda: qk_stats(*a, mo, lo, gpg, D, cfg=c, **kw)
    else:
        VAR = {"had": (ah, dict(corr=0), "had")}
        VAR.update({v: (a2, dict(corr=1, corr_mfma=c), "2dp") for v, c in CORR.items()})
        cands = [c for c in tiles(D) if T % (c[1] * c[5]) == 0 and T % c[0] == 0]

        def p2f(a, kw, c):
            return lambda: qk(*a, **kw, cfg=c)

    r = {}
    if ti == 0:
        chk = {}
        if FA:  # m, l vs torch on the logits of the (verified) logit-store kernel
            for f, key, sc in (("2dp", "qk_2dp_mfma", sc_2dp), ("had", "qk_had", sc_had)):
                qk(*p2[key][0], **p2[key][1], cfg=(128, 128, 4, 2, 2, 1))
                torch.cuda.synchronize()
                z = s["out"] * sc
                m_ref = z.amax(-1)
                l_ref = torch.exp2(z - m_ref[..., None]).sum(-1)
                for v, (a, kw, ff) in VAR.items():
                    if ff != f:
                        continue
                    p2f(a, kw, (128, 64, 4, 4, 1))()
                    torch.cuda.synchronize()
                    chk[v] = dict(
                        m_max_abs=(mo - m_ref).abs().max().item(),
                        l_max_rel=((lo - l_ref).abs() / l_ref).max().item())
                del z, m_ref, l_ref
        else:  # first 2048 tokens of the logits vs the torch target, x 1/(1-g)
            cut = min(T, 2048)
            o = rope_std((s["x"][:cut].float() @ s["W"].float()).view(-1, h_tot, D)[:, :n_qk],
                         s["cos"][:cut],
                         s["sin"][:cut])
            a, rz = ref_2dp(o, U, n_q, gpg)
            dref, _, _ = mx_quant(rz)
            target = ref_logits(dref[:, :n_q], dref[:, n_q:], gpg, a[:, :n_q], a[:, n_q:], 1 / (1 - G))
            for v in CORR:
                qk(*VAR[v][0], **VAR[v][1], cfg=(128, 128, 4, 2, 2, 1))
                torch.cuda.synchronize()
                lg = s["out"][:, :cut, :cut] / (1 - G)
                chk[v] = dict(logits_rel_l2=((lg - target).norm() / target.norm()).item())
            del target, o, a, rz, dref
        r["check"] = chk
        print("check:", chk, flush=True)
    if FA:
        del s["out"]
        torch.cuda.empty_cache()

    sweep = {}
    for v, (a, kw, f) in VAR.items():
        for c in cands:
            try:
                p2f(a, kw, c)()
                sweep[(v, c)] = med_us(lambda: (fe[f](), p2f(a, kw, c)()))
            except Exception as e:  # noqa: BLE001
                print(f"  [{v} {c} failed: {type(e).__name__}: {str(e).splitlines()[0][:160]}]", flush=True)
    best = {v: min((t, c) for (vv, c), t in sweep.items() if vv == v)[1] for v in VAR}
    fns = {v: (lambda v=v: (fe[VAR[v][2]](), p2f(*VAR[v][:2], best[v])())) for v in VAR}
    fns.update(fe_2dp=fe["2dp"], fe_had=fe["had"])
    names = list(fns)
    t = {n: [] for n in names}
    for rep in range(args.repeats):
        for n in (names if rep % 2 == 0 else names[::-1]):
            t[n].append(med_us(fns[n]))
    md = {n: statistics.median(x) for n, x in t.items()}
    r.update(best={v: list(c) for v, c in best.items()}, median=md, raw=t, vs_had={})
    print(f"== {cfg['name']} T={T} ({args.endpoint}), best chain per variant, median of {args.repeats} "
          f"interleaved repeats", flush=True)
    for v in VAR:
        p2t = md[v] - md["fe_" + VAR[v][2]]
        line = f"  {v:<5} {str(best[v]):<26} e2e {md[v]:8.1f}  p2 {p2t:8.1f}"
        if v != "had":
            rr = [100 * (x / y - 1) for x, y in zip(t[v], t["had"])]
            r["vs_had"][v] = dict(median=statistics.median(rr), min=min(rr), max=max(rr))
            line += f"  vs Had {statistics.median(rr):+.2f} % [{min(rr):+.2f}, {max(rr):+.2f}]"
        print(line, flush=True)
    result["runs"][T] = r
    del s
    torch.cuda.empty_cache()
if args.out:
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"wrote {args.out}")
