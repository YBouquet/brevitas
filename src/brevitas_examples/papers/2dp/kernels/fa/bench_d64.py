# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Portions derived from fmha_v3.py of the gfx950 Gluon tutorial ("High-Performance Gluon Kernels on AMD GFX9",
# Copyright (c) 2026 AMD ROCm Software, MIT License: see LICENSE-gluon-tutorial).

"""fmha_d64 (head_dim 64 port of fmha_kbank) -- Hadamard vs 2DP, every knob / tile applied to BOTH methods.
    python3 -u bench_d64.py --model 1b --set base1b --check-ts 2048 8192 --ts 1024 2048 4096 8192 16384 --e2e
Correctness (every variant, both methods, every --check-ts): O rel L2 and LSE max abs error vs torch on the kernels' own
dequantized inputs (S = Qdq Kdq^T [+ c_q c_k^T], softmax(S * sm_scale) @ V, logsumexp), heads --heads.
Timing: all (variant, method) FA closures (+ e2e = front end + FA, + front ends alone) interleaved per round, order
reversed on odd rounds, do_bench median per call; median over rounds and [min, max] of per-round ratios.
ISA: amdgcn of every variant at --asm-t saved; VGPR/spill/LDS; loop census (outermost loop body, per tile).
--ident: pairs "<x>" (fmha_d64) / "KB:<x>" (fmha_kbank) are compared instruction-for-instruction (kernel name masked).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fmha_roof  # noqa: E402,I001,F401  -- FIRST (imports triton before torch)
import fmha_kbank  # noqa: E402
import fmha_d64  # noqa: E402
import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import traceback  # noqa: E402

import torch  # noqa: E402
import importlib.util  # noqa: E402

from fa_setup import GAMMA, Setup  # noqa: E402  (also puts ../ on sys.path: front ends, reference, med_us)
from bench_2dp_vs_hadamard import med_us  # noqa: E402
from reference import unpack_mx  # noqa: E402

from bench_roof_sets import ENVS, ENV_ORDER, V as _V  # noqa: E402


def V(**kw):
    d = _V(lds=1, mod="d64", block_m=None, block_n=64, ksa=0, r2p=0, pw=8, lrke=0, kpred=0, kcf=0, ablc=0, lth=8.0, vf=0, ftz=0, sched="", rv=0, d1f=0, exs=0)
    d.update(kw)
    return d


KNOBSETS = {   # name suffix -> knobs (the fa_roof / fa_kbank knob sets)
    "": {},
    "_lazy": dict(lazy=1),
    "_prio": dict(prio=1),
    "_ns": dict(env="nosink"),
    "_scal": dict(env="scal"),
    "_lazy_prio": dict(lazy=1, prio=1),
    "_lazy_ns_scal": dict(lazy=1, env="nosink+scal"),
}
TILES = {   # name -> (num_warps, BLOCK_M, BLOCK_N)
    "w8m256n64": (8, 256, 64),     # = L
    "w4m128n64": (4, 128, 64),
    "w8m256n128": (8, 256, 128),
    "w4m128n128": (4, 128, 128),
    "w4m256n64": (4, 256, 64),
    "w4m256n128": (4, 256, 128),
}


def tile(name, **kw):
    nw, bm, bn = TILES[name]
    return V(num_warps=nw, block_m=bm, block_n=bn, **kw)


PLACE_KNOBS = ("", "_lazy", "_prio", "_lazy_prio", "_lazy_ns_scal", "_ns")


def place_set(tiles, places, knobs=PLACE_KNOBS):
    out = {}
    for t in tiles:
        for pn, pk in places.items():
            for k in knobs:
                out[f"{t}_{pn}{k}"] = tile(t, **pk, **KNOBSETS[k])
    return out


PLACES = {"p0": dict(r2p=0), "p1": dict(r2p=1), "p2": dict(r2p=2), "p3": dict(r2p=3),
          "p0x16": dict(r2p=0, pw=16), "p1x16": dict(r2p=1, pw=16), "p1e": dict(r2p=1, lrke=1)}

def p2set():
    t = "w8m256n128"
    out = {}
    for k in ("_lazy", "_lazy_prio", "_lazy_ns_scal", "_prio", "_lazy_scal"):
        out[f"{t}_p3{k}"] = tile(t, r2p=3, **KNOBSETS[k])
        out[f"{t}_p3e{k}"] = tile(t, r2p=3, lrke=1, **KNOBSETS[k])
    for k in ("_lazy_prio", "_lazy_ns_scal", "_lazy"):
        out[f"{t}_p3kp{k}"] = tile(t, r2p=3, kpred=1, **KNOBSETS[k])
        out[f"{t}_p3ekp{k}"] = tile(t, r2p=3, lrke=1, kpred=1, **KNOBSETS[k])
        out[f"{t}_p3kcf{k}"] = tile(t, r2p=3, kcf=1, **KNOBSETS[k])
    for a in (1, 2, 3):
        out[f"{t}_p3_ABLC{a}_lazy_ns_scal"] = tile(t, r2p=3, ablc=a, **KNOBSETS["_lazy_ns_scal"])
        out[f"{t}_p3_ABLC{a}_lazy_prio"] = tile(t, r2p=3, ablc=a, **KNOBSETS["_lazy_prio"])
    out[f"{t}_p0_lazy"] = tile(t, **KNOBSETS["_lazy"])
    out[f"{t}_p0_lazy_ns_scal"] = tile(t, **KNOBSETS["_lazy_ns_scal"])
    return out


KNOBSETS["_lazy_scal"] = dict(lazy=1, env="scal")

def p3set():
    t = "w8m256n128"
    out = {}
    for th_ in (8, ):   # TH 12 / 15 dropped: the reference logits never trigger a lazy rescale (job 92339 resc-stats)
        sfx = "" if th_ == 8 else f"_t{th_}"
        for pl, pk in (("p3", dict(r2p=3)), ("p3e", dict(r2p=3, lrke=1))):
            for k in ("_lazy", "_lazy_prio", "_lazy_ns_scal"):
                out[f"{t}_{pl}{k}{sfx}"] = tile(t, **pk, lth=float(th_), **KNOBSETS[k])
    out[f"{t}_p0_lazy"] = tile(t, **KNOBSETS["_lazy"])
    out[f"{t}_p0_lazy_ns_scal"] = tile(t, **KNOBSETS["_lazy_ns_scal"])
    out[f"{t}_p3_prio"] = tile(t, r2p=3, **KNOBSETS["_prio"])
    return out


def p4set():
    t = "w8m256n128"
    out = {}
    base = {"p3": dict(r2p=3), "p3e": dict(r2p=3, lrke=1)}
    for pl, pk in base.items():
        for k in ("_lazy", "_lazy_prio", "_lazy_ns_scal"):
            out[f"{t}_{pl}{k}"] = tile(t, **pk, **KNOBSETS[k])
            out[f"{t}_{pl}{k}_ftz"] = tile(t, **pk, ftz=1, **KNOBSETS[k])
            out[f"{t}_{pl}_vf1{k}"] = tile(t, **pk, vf=1, **KNOBSETS[k])
            out[f"{t}_{pl}_vf2{k}"] = tile(t, **pk, vf=2, **KNOBSETS[k])
        out[f"{t}_{pl}_vf1_lazy_ftz"] = tile(t, **pk, vf=1, ftz=1, **KNOBSETS["_lazy"])
        out[f"{t}_{pl}_vf1_lazy_prio_ftz"] = tile(t, **pk, vf=1, ftz=1, **KNOBSETS["_lazy_prio"])
        out[f"{t}_{pl}_lazy_milp"] = tile(t, **pk, sched="max-ilp", **KNOBSETS["_lazy"])
    out[f"{t}_p0_lazy"] = tile(t, **KNOBSETS["_lazy"])
    out[f"{t}_p0_lazy_ftz"] = tile(t, ftz=1, **KNOBSETS["_lazy"])
    return out


def p5set():
    t = "w8m256n128"
    out = {}
    pls = {"p3": dict(r2p=3), "p3e": dict(r2p=3, lrke=1), "p0": dict(r2p=0), "p0e": dict(r2p=0, lrke=1)}
    for pl, pk in pls.items():
        for k in ("_lazy", "_lazy_prio", "_lazy_ns_scal"):
            out[f"{t}_{pl}{k}"] = tile(t, **pk, **KNOBSETS[k])                   # no fence (job 92352 reference)
            out[f"{t}_{pl}_vf1t{k}"] = tile(t, **pk, vf=1, rv=1, **KNOBSETS[k])  # fence; rank-2 after it (p3) / dot1 (p0)
        out[f"{t}_{pl}_vf2t_lazy"] = tile(t, **pk, vf=2, rv=1, **KNOBSETS["_lazy"])
        out[f"{t}_{pl}_vf1t_prio"] = tile(t, **pk, vf=1, rv=1, **KNOBSETS["_prio"])
    return out


def p6set():
    t = "w8m256n128"
    out = {}
    pls = {"p0": dict(r2p=0), "p1": dict(r2p=1), "p2": dict(r2p=2), "p0d": dict(r2p=0, d1f=1),
           "p2d": dict(r2p=2, d1f=1), "p3d": dict(r2p=3, rv=1, d1f=1)}
    for pl, pk in pls.items():
        for k in ("_lazy", "_lazy_prio", "_lazy_ns_scal"):
            out[f"{t}_{pl}_vf1t{k}"] = tile(t, **pk, vf=1, **KNOBSETS[k])
    out[f"{t}_p0e_vf1t_lazy"] = tile(t, r2p=0, lrke=1, vf=1, **KNOBSETS["_lazy"])    # fastest Had of job 92360
    out[f"{t}_p0ed_vf1t_lazy"] = tile(t, r2p=0, lrke=1, vf=1, d1f=1, **KNOBSETS["_lazy"])
    return out


def finalset():
    t = "w8m256n128"
    P = dict(p0=dict(r2p=0), p1=dict(r2p=1), p3=dict(r2p=3), p3e=dict(r2p=3, lrke=1), p0e=dict(r2p=0, lrke=1))
    names = [("p0", "", "_lazy"), ("p0", "", "_lazy_prio"), ("p0", "", "_lazy_ns_scal"),
             ("p3", "", "_lazy_prio"), ("p3", "", "_lazy_ns_scal"), ("p3e", "", "_lazy"),
             ("p0", "_vf1t", "_lazy"), ("p0", "_vf1t", "_lazy_prio"), ("p0", "_vf1t", "_lazy_ns_scal"),
             ("p1", "_vf1t", "_lazy_ns_scal"), ("p0e", "_vf1t", "_lazy")]
    out = {}
    for pl, vf, k in names:
        out[f"{t}_{pl}{vf}{k}"] = tile(t, **P[pl], **(dict(vf=1) if vf else {}), **KNOBSETS[k])
    return out


def final8b():
    out = {"L": V(), "KB:L": V(mod="kb"), "L_lazy_prio": V(lazy=1, prio=1), "KB:L_lazy_prio": V(mod="kb", lazy=1, prio=1)}
    for k in ("_lazy", "_lazy_prio", "_lazy_ns_scal"):
        out[f"L_vf1t{k}"] = V(vf=1, **KNOBSETS[k])
    out["L_e_vf1t_lazy"] = V(lrke=1, vf=1, **KNOBSETS["_lazy"])
    out["L_lazy_ns_scal"] = V(**KNOBSETS["_lazy_ns_scal"])
    return out


def comboset():
    """Fence (VF 1) + fmha_v4 slice balancing (EXS e): references = job 92366's fastest pair and fence configs."""
    t = "w8m256n128"
    base = {"p0_vf1t_lazy": dict(r2p=0, vf=1, **KNOBSETS["_lazy"]),
            "p0_vf1t_lazy_prio": dict(r2p=0, vf=1, **KNOBSETS["_lazy_prio"]),
            "p0_vf1t_lazy_ns_scal": dict(r2p=0, vf=1, **KNOBSETS["_lazy_ns_scal"]),
            "p0_vf1t_lazy_scal": dict(r2p=0, vf=1, lazy=1, env="scal"),
            "p0e_vf1t_lazy": dict(r2p=0, lrke=1, vf=1, **KNOBSETS["_lazy"])}
    out = {}
    for n, kw in base.items():
        out[f"{t}_{n}"] = tile(t, **kw)
        for e in (1, 2, 3, 4):
            out[f"{t}_{n}_exs{e}"] = tile(t, exs=e, **kw)
    return out


def readme1b():
    """README table, Llama-3.2-1B (job 92366): the fastest Hadamard / 2DP configs and their neighbours, 256 x 128 tile."""
    t = "w8m256n128"
    return {f"{t}_p0e_vf1t_lazy": tile(t, r2p=0, lrke=1, vf=1, **KNOBSETS["_lazy"]),
            f"{t}_p0_vf1t_lazy": tile(t, r2p=0, vf=1, **KNOBSETS["_lazy"]),
            f"{t}_p0_vf1t_lazy_prio": tile(t, r2p=0, vf=1, **KNOBSETS["_lazy_prio"]),
            f"{t}_p0_vf1t_lazy_ns_scal": tile(t, r2p=0, vf=1, **KNOBSETS["_lazy_ns_scal"]),
            f"{t}_p0_lazy_prio": tile(t, r2p=0, **KNOBSETS["_lazy_prio"])}


def readme8b():
    """README table, Llama-3.1-8B (job 92099): L tile (256 x 64, 8 warps), no dot2 fence at D = 128."""
    return {"L": V(), "L_lazy": V(**KNOBSETS["_lazy"]), "L_lazy_prio": V(**KNOBSETS["_lazy_prio"]),
            "L_lazy_ns_scal": V(**KNOBSETS["_lazy_ns_scal"])}


SETS = {
    "readme1b": readme1b(),
    "readme8b": readme8b(),
    "combo1b": comboset(),
    "final1b": finalset(),
    "final8b": final8b(),
    "place6": p6set(),
    "place5": p5set(),
    "place4": p4set(),
    "place3": p3set(),
    "place2": p2set(),
    "place1": place_set(["w8m256n128"], PLACES),
    "place1L": place_set(["w8m256n64"], {k: PLACES[k] for k in ("p0", "p1", "p2", "p3")}, ("", "_lazy_prio")),
    "ident8b": {
        "L": V(),
        "KB:L": V(mod="kb"),
        "L_lazy_prio": V(lazy=1, prio=1),
        "KB:L_lazy_prio": V(mod="kb", lazy=1, prio=1),
    },
    "base1b": dict(
        [("L" + k, V(**v)) for k, v in KNOBSETS.items()]
        + [("L_ksa1", V(ksa=1))]
        + [(t + k, tile(t, **KNOBSETS[k])) for t in list(TILES)[1:] for k in ("", "_lazy_prio", "_lazy_ns_scal")]),
}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="1b")
ap.add_argument("--set", default="base1b")
ap.add_argument("--only", nargs="*", default=None)
ap.add_argument("--check-ts", type=int, nargs="+", default=[2048, 8192])
ap.add_argument("--heads", type=int, nargs="+", default=[0, 5, 31])
ap.add_argument("--ts", type=int, nargs="+", default=[8192])
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--e2e", action="store_true")
ap.add_argument("--ident", action="store_true")
ap.add_argument("--asm-t", type=int, default=8192)
ap.add_argument("--tag", default="")
ap.add_argument("--hd2", action="store_true", help="also time the Hadamard binary on 2DP data (probe)")
ap.add_argument("--u-path", default=None, help="frozen plane (--dc-u-path); default random")
ap.add_argument("--u-layer", type=int, default=15)
ap.add_argument("--out-dir", default=".")
ap.add_argument("--resc-stats", action="store_true", help="lazy-rescale frequency of the reference logits")
args = ap.parse_args()
variants = SETS[args.set]
if args.only:
    variants = {k: v for k, v in variants.items() if k in args.only}
job = os.environ.get("JOB", "x")
asmdir = os.path.join(args.out_dir, f"asm_{args.set}{args.tag}_{job}")
os.makedirs(asmdir, exist_ok=True)

all_t = sorted(set(args.check_ts + args.ts))
S = {}
for T in all_t:
    st = Setup(args.model, T, args.u_path, args.u_layer)
    s, D, n_q = st.s, st.D, st.n_q
    n_kv = n_q // st.gpg
    n_qk, h_tot = n_q + n_kv, n_q + 2 * n_kv
    v = s["C"].view(T, h_tot, D)[:, n_qk:n_qk + n_kv]
    out = {m: torch.empty(T, n_q, D, device="cuda", dtype=torch.float16) for m in ("had", "2dp")}
    lse = torch.empty(n_q, T, device="cuda", dtype=torch.float32)
    sm = {"had": 1.0 / math.sqrt(D), "2dp": 1.0 / (math.sqrt(D) * (1.0 - GAMMA))}
    S[T] = dict(st=st, v=v, out=out, lse=lse, sm=sm, fe={"had": st.fe_for("had"), "2dp": st.fe_for("2dp", 6), "2dp16": st.fe_for("2dp", 7)})
    print(f"setup T={T} D={D} n_q={n_q} n_kv={n_kv} done", flush=True)

KW = ("num_warps", "wpe", "agpr0", "resc_mem", "prio", "abl", "v1x", "lazy", "block_m", "block_n", "ksa", "r2p",
      "lrke", "kpred", "kcf", "ablc", "lth", "vf", "ftz",
      "sched", "rv", "d1f", "exs")


def fa_closure(T, m, cfg):
    d = S[T]
    st = d["st"]
    kw = {k: cfg[k] for k in KW}
    mod = fmha_kbank if cfg["mod"] == "kb" else fmha_d64
    kw["envtag"] = ENV_ORDER.index(cfg["env"])
    if m == "had":   # the rank-2 placement does not exist without the rank-2 term: identical Hadamard kernel (R2P forced 0)
        kw["r2p"] = 0
        kw["ablc"] = 0
        a = st.ah
        return lambda: mod.run_mx(*a[:2], None, *a[3:5], None, d["v"], d["out"]["had"], d["lse"], d["sm"]["had"], 0,
                                  **kw)
    if m == "hd2":   # probe: the Hadamard binary on the 2DP fp4 codes / scales and 2DP softmax scale (no rank-2 term)
        a = st.a2p[8]
        kw["r2p"] = 0
        kw["ablc"] = 0
        return lambda: mod.run_mx(*a[:2], None, *a[3:5], None, d["v"], d["out"]["had"], d["lse"], d["sm"]["2dp"], 0,
                                  **kw)
    a = st.a2p[cfg["pw"]]
    return lambda: mod.run_mx(*a[:2], a[2], *a[3:5], a[5], d["v"], d["out"]["2dp"], d["lse"], d["sm"]["2dp"], 1, **kw)


def asm_stats(k):
    asm = k.asm["amdgcn"]
    g = lambda pat: int(re.search(pat, asm).group(1))  # noqa: E731
    return dict(vgpr=g(r"\.vgpr_count:\s*(\d+)"), spill=g(r"\.vgpr_spill_count:\s*(\d+)"),
                agpr=g(r"\.agpr_count:\s*(\d+)"), lds=g(r"\.group_segment_fixed_size:\s*(\d+)"))


CLASSES = [   # (name, regex on mnemonic); first match wins
    ("mfma_fp4", r"^v_mfma_scale_f32_32x32x64"), ("mfma_f16_k16", r"^v_mfma_f32_32x32x16_f16"),
    ("mfma_f16_k8", r"^v_mfma_f32_32x32x8_?f16"), ("mfma_other", r"^v_mfma"),
    ("v_exp", r"^v_exp_f32"), ("v_trans_other", r"^v_(log|rcp|rsq|sqrt)_"),
    ("v_permlane", r"^v_permlane"), ("v_pk", r"^v_pk_"), ("v_cvt_pk", r"^v_cvt_pk"),
    ("v_max", r"^v_max"), ("v_fma", r"^v_(fma|fmac|mac|mad)"), ("v_add", r"^v_(add|sub)"), ("v_mul", r"^v_mul"),
    ("v_cndmask", r"^v_cndmask"), ("v_cmp", r"^v_cmp"), ("v_mov", r"^v_(mov|accvgpr)"), ("v_dot", r"^v_dot"),
    ("valu_other", r"^v_"),
    ("ds_read", r"^ds_read"), ("ds_write", r"^ds_write"), ("buf_lds", r"^buffer_load.*"), ("global", r"^global_"),
    ("s_waitcnt", r"^s_waitcnt"), ("s_barrier", r"^s_barrier"), ("s_setprio", r"^s_setprio"),
    ("sched_barrier", r"^s_sched_barrier|^sched_barrier"), ("s_branch", r"^s_c?branch"), ("salu", r"^s_"),
]


def loop_census(asm, tiles_per_iter=2):
    """Outermost loop body (largest backward branch span) instruction census, per tile."""
    lines = [ln.strip() for ln in asm.splitlines()]
    labels = {}
    for i, ln in enumerate(lines):
        m = re.match(r"^(\.LBB\d+_\d+):", ln)
        if m:
            labels[m.group(1)] = i
    best = None
    for j, ln in enumerate(lines):
        m = re.match(r"^s_c?branch\w*\s+(\.LBB\d+_\d+)", ln)
        if m and m.group(1) in labels and labels[m.group(1)] < j:
            span = (labels[m.group(1)], j)
            if best is None or span[1] - span[0] > best[1] - best[0]:
                best = span
    if best is None:
        return {}
    c = collections.Counter()
    for ln in lines[best[0]:best[1] + 1]:
        if not ln or ln.startswith((".", ";")) or ln.endswith(":"):
            continue
        mn = ln.split()[0]
        if mn.endswith(":"):
            continue
        c["total"] += 1
        if "_dpp" in ln or "row_" in ln or "quad_perm" in ln:
            c["dpp_any"] += 1
        for name, pat in CLASSES:
            if re.match(pat, mn):
                c[name] += 1
                break
        else:
            c["other"] += 1
    return {k: v / tiles_per_iter for k, v in sorted(c.items())}


def norm_asm(asm):
    out = []
    for ln in asm.splitlines():
        s_ = ln.strip()
        if not s_ or s_.startswith(";") or s_.startswith(".") and not re.match(r"^\.LBB", s_):
            continue
        s_ = re.sub(r"mx_attn_fwd_\w+", "KNAME", s_)
        out.append(s_)
    return out


res = dict(model=args.model, set=args.set, variants=variants, info={}, census={}, check={}, runs={}, ident={})
ok, fns, asms = {}, {}, {}
for env in ENV_ORDER:
    group = [n for n, c in variants.items() if c["env"] == env]
    if not group:
        continue
    for k_, v_ in ENVS[env].items():
        os.environ[k_] = v_
    if "AMDGCN_SCALARIZE_PACKED_FOPS" not in ENVS[env]:
        os.environ.pop("AMDGCN_SCALARIZE_PACKED_FOPS", None)
    for n in group:
        for m in ("had", "2dp"):
            try:
                for T in all_t:
                    fn = fa_closure(T, m, variants[n])
                    k = fn()
                    torch.cuda.synchronize()
                    if T == args.asm_t:
                        info = asm_stats(k)
                        asm = k.asm["amdgcn"]
                        asms[(n, m)] = asm
                        cen = loop_census(asm)
                        res["info"][f"{n}/{m}"] = info
                        res["census"][f"{n}/{m}"] = cen
                        open(os.path.join(asmdir, f"{n.replace(':', '_')}_{m}.amdgcn"), "w").write(asm)
                        print(f"compiled {n}/{m} env='{env}': {info}", flush=True)
                    fns[(T, n, m)] = fn
                ok[(n, m)] = True
                if args.hd2 and m == "had":
                    for T in all_t:
                        fns[(T, n, "hd2")] = fa_closure(T, "hd2", variants[n])
            except Exception as e:  # noqa: BLE001
                ok[(n, m)] = False
                print(f"FAILED {n}/{m}: {type(e).__name__}: {str(e)[:3000]}", flush=True)
                traceback.print_exc(limit=4)

if args.ident:
    for n in variants:
        if n.startswith("KB:") and (n[3:], "had") in asms:
            for m in ("had", "2dp"):
                a, b = norm_asm(asms[(n, m)]), norm_asm(asms[(n[3:], m)])
                same = a == b
                ndiff = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
                res["ident"][f"{n[3:]}/{m}"] = dict(identical=same, n_kb=len(a), n_d64=len(b), ndiff=ndiff)
                print(f"ISA ident {n[3:]}/{m}: identical={same} lines kb={len(a)} d64={len(b)} differing={ndiff}",
                      flush=True)

# correctness: O rel L2 and LSE abs error vs torch, every check T, every variant and method
for T in args.check_ts:
    d = S[T]
    st, s = d["st"], d["st"].s
    n_q, gpg = st.n_q, st.gpg
    dq = {"had": unpack_mx(s["pk_had"], s["sc_had"])[0], "2dp": unpack_mx(s["pk_2dp"], s["sc_2dp"])[0]}
    refs = {}
    for m in ("had", "2dp"):
        d["fe"][m]()
        torch.cuda.synchronize()
        Vf = d["v"].float()
        for h in args.heads:
            hk = h // gpg
            Lg = dq[m][:, h] @ dq[m][:, n_q + hk].T
            if m == "2dp":
                cf = s["cf"].float()
                Lg = Lg + cf[:, h] @ cf[:, n_q + hk].T
            Lg = Lg * d["sm"][m]
            refs[(m, h)] = (torch.softmax(Lg, dim=-1) @ Vf[:, hk], torch.logsumexp(Lg, dim=-1))
            del Lg
    for n in variants:
        for m in ("had", "2dp"):
            if not ok.get((n, m)):
                continue
            d["fe"]["2dp16" if (m == "2dp" and variants[n]["pw"] == 16) else m]()
            d["out"][m].fill_(float("nan"))
            d["lse"].fill_(float("nan"))
            fns[(T, n, m)]()
            torch.cuda.synchronize()
            eo, el = 0.0, 0.0
            for h in args.heads:
                ro, rl = refs[(m, h)]
                e = ((d["out"][m][:, h].float() - ro).norm() / ro.norm()).item()
                eo = max(eo, e) if e == e else float("inf")
                e = (d["lse"][h] - rl).abs().max().item()
                el = max(el, e) if e == e else float("inf")
            res["check"][f"T{T}/{n}/{m}"] = dict(o_rel_l2=eo, lse_max_abs=el)
            flag = "OK" if (eo < 2e-3 and el < 1e-2) else "BAD"
            print(f"check T={T} {n}/{m}: O max rel L2 {eo:.2e}  LSE max abs {el:.2e}  {flag}", flush=True)
            if flag == "BAD" and not variants[n]["ablc"]:
                ok[(n, m)] = False
    del refs, dq
    torch.cuda.empty_cache()

if args.resc_stats:
    for T in args.check_ts:
        st, s_ = S[T]["st"], S[T]["st"].s
        n_q, gpg = st.n_q, st.gpg
        dq = {"had": unpack_mx(s_["pk_had"], s_["sc_had"])[0], "2dp": unpack_mx(s_["pk_2dp"], s_["sc_2dp"])[0]}
        for m in ("had", "2dp"):
            S[T]["fe"][m]()
            torch.cuda.synchronize()
            for th_ in (8.0, 12.0, 15.0):
                fr, rows = [], []
                for h in args.heads:
                    hk = h // gpg
                    Lg = dq[m][:, h] @ dq[m][:, n_q + hk].T
                    if m == "2dp":
                        cf = s_["cf"].float()
                        Lg = Lg + cf[:, h] @ cf[:, n_q + hk].T
                    Lg = Lg * (S[T]["sm"][m] * 1.44269504089)
                    tm = Lg.view(T, T // 128, 128).amax(-1)          # per-row tile max (log2 units), [T, n_tiles]
                    mr = tm[:, 0].clone()
                    ev_w, ev_r = 0, 0
                    for j in range(1, T // 128):
                        up = tm[:, j] - mr > th_
                        mr = torch.where(up, tm[:, j], mr)
                        ev_r += int(up.sum())
                        ev_w += int(up.view(-1, 32).any(-1).sum())     # a wave (32 rows) takes the rescale branch
                    nt = (T // 128 - 1)
                    fr.append(ev_w / (nt * T // 32))
                    rows.append(ev_r / (nt * T))
                    del Lg, tm
                print(f"resc-stats T={T} {m} TH={th_:4.1f}: wave-tiles rescaling {100 * max(fr):.3f}% (max over heads "
                      f"{args.heads}), row-tiles {100 * max(rows):.4f}%", flush=True)
        del dq
        torch.cuda.empty_cache()

for T in args.ts:
    d = S[T]
    fmap = {}
    for n in variants:
        if not (ok.get((n, "had")) and ok.get((n, "2dp"))):
            continue
        for m in ("had", "2dp") + (("hd2", ) if args.hd2 else ()):
            fmap[f"fa:{n}:{m}"] = fns[(T, n, m)]
            if args.e2e and m != "hd2":
                fek = "2dp16" if (m == "2dp" and variants[n]["pw"] == 16) else m
                fmap[f"e2e:{n}:{m}"] = (lambda f=fns[(T, n, m)], fe=d["fe"][fek]: (fe(), f()))
    if args.e2e:
        for m in ("had", "2dp", "2dp16"):
            fmap[f"fe:{m}"] = d["fe"][m]
    names = list(fmap)
    t = {x: [] for x in names}
    for rr in range(args.rounds):
        for x in (names if rr % 2 == 0 else names[::-1]):
            t[x].append(med_us(fmap[x]))
    flops = 4 * d["st"].n_q * T * T * d["st"].D
    rT = dict(raw=t, med={x: statistics.median(y) for x, y in t.items()})
    print(f"\n== T={T} ({args.rounds} interleaved rounds; median [min, max] us; TFLOPS from median)", flush=True)
    for n in variants:
        if f"fa:{n}:had" not in t:
            continue
        th, t2 = t[f"fa:{n}:had"], t[f"fa:{n}:2dp"]
        rat = [100 * (a / b - 1) for a, b in zip(t2, th)]
        line = (f"{n:22s} FA had {statistics.median(th):8.1f} [{min(th):.1f},{max(th):.1f}] "
                f"({flops / statistics.median(th) / 1e6:5.0f} TF)  2dp {statistics.median(t2):8.1f} "
                f"[{min(t2):.1f},{max(t2):.1f}] ({flops / statistics.median(t2) / 1e6:5.0f} TF)  "
                f"2dp/had {statistics.median(rat):+6.2f}% [{min(rat):+.2f},{max(rat):+.2f}]")
        if args.e2e:
            eh, e2 = t[f"e2e:{n}:had"], t[f"e2e:{n}:2dp"]
            er = [100 * (a / b - 1) for a, b in zip(e2, eh)]
            line += (f" | e2e had {statistics.median(eh):.1f} 2dp {statistics.median(e2):.1f} -> "
                     f"{statistics.median(er):+.2f}% [{min(er):+.2f},{max(er):+.2f}]")
            rT[f"e2e_ratio:{n}"] = [statistics.median(er), min(er), max(er)]
        rT[f"fa_ratio:{n}"] = [statistics.median(rat), min(rat), max(rat)]
        print(line, flush=True)
        if args.hd2:
            tq = t[f"fa:{n}:hd2"]
            r1 = [100 * (a / b - 1) for a, b in zip(tq, th)]
            r2 = [100 * (a / b - 1) for a, b in zip(t2, tq)]
            rT[f"hd2:{n}"] = dict(us=statistics.median(tq), data=[statistics.median(r1), min(r1), max(r1)],
                                  code=[statistics.median(r2), min(r2), max(r2)])
            print(f"{'':22s}   probe: Had binary on 2DP data {statistics.median(tq):8.1f} us -> data effect (hd2/had) "
                  f"{statistics.median(r1):+6.2f}% [{min(r1):+.2f},{max(r1):+.2f}], code effect (2dp/hd2) "
                  f"{statistics.median(r2):+6.2f}% [{min(r2):+.2f},{max(r2):+.2f}]", flush=True)
    # fastest-vs-fastest (per method, best median over variants), with per-round ratio of the two winners
    for kind in (("fa", "e2e") if args.e2e else ("fa",)):
        cand = {m: [n for n in variants if f"{kind}:{n}:{m}" in t and not variants[n]["ablc"]] for m in ("had", "2dp")}
        if not cand["had"]:
            continue
        best = {m: min(cand[m], key=lambda n: statistics.median(t[f"{kind}:{n}:{m}"])) for m in ("had", "2dp")}
        bh, b2 = t[f"{kind}:{best['had']}:had"], t[f"{kind}:{best['2dp']}:2dp"]
        rat = [100 * (a / b - 1) for a, b in zip(b2, bh)]
        rT[f"best_{kind}"] = dict(had=best["had"], had_us=statistics.median(bh), dp=best["2dp"],
                                  dp_us=statistics.median(b2), ratio=[statistics.median(rat), min(rat), max(rat)])
        print(f"FASTEST {kind}: had {best['had']} {statistics.median(bh):.1f} us | 2dp {best['2dp']} "
              f"{statistics.median(b2):.1f} us | 2dp/had {statistics.median(rat):+.2f}% [{min(rat):+.2f},{max(rat):+.2f}]",
              flush=True)
    if args.e2e:
        print(f"front end: had {statistics.median(t['fe:had']):.1f} us, 2dp {statistics.median(t['fe:2dp']):.1f} us, "
              f"2dp(pw16) {statistics.median(t['fe:2dp16']):.1f} us")
    res["runs"][T] = rT
json.dump(res, open(os.path.join(args.out_dir, f"fa_{args.set}{args.tag}_{args.model}_{job}.json"), "w"), indent=1)
print("done", flush=True)
