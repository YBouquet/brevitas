# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""2DP vs Hadamard with a flash-attention endpoint: no logits are written.

  2DP       P1 = p1_2dp.front_end_2dp(epi=1)
            P2 = p2_fa_stats.qk_stats(corr=1): fp4 QK^T + a'.b + online softmax (row max, row sum), m / l stored
  Hadamard  J1 + J2 (as in bench_2dp_vs_hadamard.py)
            P2 = p2_fa_stats.qk_stats(corr=0)
The softmax scale is log2(e) / sqrt(D), times 1/(1-g) for 2DP (folded unshrink). P.V is omitted: V is the same
for both methods, so real flash attention adds the same time to both. For each pipeline the rank-2 variant
(2DP) and the tile are picked on the end-to-end chain; then --repeats interleaved repeats, median.

    python bench_fa_proxy.py --model 8b --u-path U_selfgen_8b_2dp_allrand_s0.pt --u-layer 15
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

VARIANTS = {"fma": 0, "mfma": 1, "dot2": 2}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(MODELS), default="1b")
    ap.add_argument("--ts", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--gamma", type=float, default=0.047)
    ap.add_argument("--u-path", default=None)
    ap.add_argument("--u-layer", type=int, default=15)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = MODELS[args.model]
    D, n_q, n_kv = cfg["hd"], cfg["n_q"], cfg["n_kv"]
    gpg = n_q // n_kv
    U = load_u(cfg, args.u_path, args.u_layer, "cuda")
    sc_had = math.log2(math.e) / math.sqrt(D)
    sc_2dp = sc_had / (1.0 - args.gamma)
    result = dict(model=cfg, gamma=args.gamma, device=torch.cuda.get_device_name(0), runs={})
    print(
        f"{cfg['name']} on {result['device']}: flash-attention endpoint (QK^T + online softmax, no logits)"
    )
    for i, T in enumerate(args.ts):
        s = build(cfg, T, U, "cuda")
        p1, p2 = pipes(cfg, s, args.gamma)
        p1["p1_2dp"]()
        p1["j1"]()
        p1["j2"]()
        torch.cuda.synchronize()
        a2 = p2["qk_2dp_mfma"][0][:6]
        ah = p2["qk_had"][0][:6]
        mo = torch.empty(n_q, T, device="cuda", dtype=torch.float32)
        lo = torch.empty_like(mo)
        fe = dict(twodp=p1["p1_2dp"], hadamard=lambda: (p1["j1"](), p1["j2"]()))
        kern = dict(
            twodp={
                v: (a2, dict(corr=1, qk_scale=sc_2dp, corr_mfma=c)) for v, c in VARIANTS.items()},
            hadamard={"bare": (ah, dict(corr=0, qk_scale=sc_had))})

        def run(a, kw, c):
            return lambda: qk_stats(*a, mo, lo, gpg, D, cfg=c, **kw)

        r = dict()
        if i == 0:  # m, l vs torch on the logits of the (verified) storing kernel
            chk = {}
            for f, ks in kern.items():
                key = "qk_2dp_mfma" if f == "twodp" else "qk_had"
                qk(*p2[key][0], **p2[key][1], cfg=(128, 128, 4, 2, 2, 1))
                torch.cuda.synchronize()
                z = s["out"] * (sc_2dp if f == "twodp" else sc_had)
                m_ref = z.amax(-1)
                l_ref = torch.exp2(z - m_ref[..., None]).sum(-1)
                for v, (a, kw) in ks.items():
                    run(a, kw, (128, 64, 4, 4, 1))()
                    torch.cuda.synchronize()
                    chk[f"{f}_{v}"] = dict(
                        m_max_abs=(mo - m_ref).abs().max().item(),
                        l_max_rel=((lo - l_ref).abs() / l_ref).max().item())
                del z, m_ref, l_ref
            r["check"] = chk
            print(
                "check (m max abs err, l max rel err):", {
                    k: f"{v['m_max_abs']:.1e} / {v['l_max_rel']:.1e}" for k, v in chk.items()})
        del s["out"]
        torch.cuda.empty_cache()
        tune = {}
        for f, ks in kern.items():
            cands = []
            for v, (a, kw) in ks.items():
                for c in stats_tiles(D):
                    if T % c[0] or T % c[1]:
                        continue
                    try:
                        run(a, kw, c)()
                        cands.append((med_us(lambda: (fe[f](), run(a, kw, c)())), v, c))
                    except Exception as e:  # noqa: BLE001
                        print(
                            f"  [{f} {v} {c} skipped: {type(e).__name__}: {str(e).splitlines()[0][:120]}]"
                        )
            tune[f] = min(cands)[1:]
        fns = {f: (lambda f=f: (fe[f](), run(*kern[f][tune[f][0]], tune[f][1])())) for f in fe}
        fns.update({f"fe_{f}": fe[f] for f in fe})
        fns.update({f"qk_{f}": run(*kern[f][tune[f][0]], tune[f][1]) for f in fe})
        names = list(fns)
        t = {n: [] for n in names}
        for rep in range(args.repeats):
            for n in (names if rep % 2 == 0 else names[::-1]):
                t[n].append(med_us(fns[n]))
        md = {n: statistics.median(v) for n, v in t.items()}
        ratio = [100 * (x / y - 1) for x, y in zip(t["twodp"], t["hadamard"])]
        r.update(
            e2e=dict(twodp=md["twodp"], hadamard=md["hadamard"]),
            front_end=dict(twodp=md["fe_twodp"], hadamard=md["fe_hadamard"]),
            p2_alone=dict(twodp=md["qk_twodp"], hadamard=md["qk_hadamard"]),
            twodp_vs_had_pct=dict(median=statistics.median(ratio), min=min(ratio), max=max(ratio)),
            pick=dict(
                twodp=[tune["twodp"][0], list(tune["twodp"][1])],
                hadamard=list(tune["hadamard"][1])))
        if i == 0:
            print(
                f"{'T':>6} {'2DP':>9} {'Hadamard':>9} {'2DP vs Had':>10} {'range':>16} | {'P1 2DP':>8} "
                f"{'J1+J2':>8} | {'P2 2DP':>8} {'P2 Had':>8}")
        print(
            f"{T:>6} {md['twodp']:>9.1f} {md['hadamard']:>9.1f} {r['twodp_vs_had_pct']['median']:>+9.1f}% "
            f"[{min(ratio):+5.1f}, {max(ratio):+5.1f}] | {md['fe_twodp']:>8.1f} {md['fe_hadamard']:>8.1f} | "
            f"{md['qk_twodp']:>8.1f} {md['qk_hadamard']:>8.1f}   (2DP {tune['twodp'][0]} {tune['twodp'][1]}, "
            f"Had {tune['hadamard'][1]})",
            flush=True)
        result["runs"][T] = r
        del s
        torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
