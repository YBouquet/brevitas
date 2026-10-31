# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Portions derived from fmha_v3.py of the gfx950 Gluon tutorial ("High-Performance Gluon Kernels on AMD GFX9",
# Copyright (c) 2026 AMD ROCm Software, MIT License: see LICENSE-gluon-tutorial).

"""fa_vdot2 study: Hadamard vs MFMA 2DP vs v_dot2 2DP in the real FA kernel (fmha_d64.py, knob VP), same knob grid.
    python3 -u bench_vdot2.py --model 1b --check-ts 2048 8192 --ts 2048 4096 8192 16384 --e2e
Configs = tile x knobs (applied to every implementation). Implementations per config: had; mf_p0 (rank-2 MFMA in dot1,
srcC chain), mf_p3 (next tile's rank-2 MFMA in dot2); vd_a (v_dot2 in dot1 after the fp4 QK^T), vd_b (v_dot2 at the
softmax head in dot2), vd_c (next tile's v_dot2 c0 in dot2 under P.V -> srcC), vd_m (next tile's v_dot2 c0 in mem2);
vd_*k: same placement with b read per 32-column chunk from the LDS ring right before use (KCH 1).
Fold check: max |(s + a'.b) * scale - (s * scale + fp16(a' * scale).b)| over the reference logits (log2 units).
(original fmha_d64 bench docstring:)
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




def V(**kw):
    d = _V(lds=1, mod="d64", block_m=None, block_n=64, ksa=0, r2p=0, pw=8, lrke=0, kpred=0, kcf=0, ablc=0, lth=8.0,
           vf=0, ftz=0, sched="", rv=0, d1f=0, exs=0, vp=0, kch=0)
    d.update(kw)
    return d


KNOBSETS = {
    "": {},
    "_prio": dict(prio=1),
    "_lazy": dict(lazy=1),
    "_lazy_prio": dict(lazy=1, prio=1),
    "_lazy_ns_scal": dict(lazy=1, env="nosink+scal"),
    "_lazy_scal": dict(lazy=1, env="scal"),
}
TILES = {"w8m256n64": (8, 256, 64), "w8m256n128": (8, 256, 128), "w4m128n64": (4, 128, 64)}
# implementation -> kernel knobs (2DP only; Hadamard has no rank-2 term). rv=1: with the dot2 fence (VF 1), a rank-2
# block computed in dot2 goes after the fence (with P.V / exp2), as job 92366's "vf1t" configs.
IMPLS = {
    "had": {},
    "mf_p0": dict(vp=0, r2p=0),
    "mf_p3": dict(vp=0, r2p=3, rv=1),
    "vd_a": dict(vp=1),
    "vd_b": dict(vp=2),
    "vd_c": dict(vp=3, rv=1),
    "vd_m": dict(vp=4),
    # KCH 1: b read from the LDS ring per 32-column chunk right before its v_dot2s (fewer live VGPRs, no LRK-side read)
    "vd_ak": dict(vp=1, kch=1),
    "vd_bk": dict(vp=2, kch=1),
    "vd_ck": dict(vp=3, rv=1, kch=1),
    "vd_mk": dict(vp=4, kch=1),
}
FAM = {"had": "had", "mf": "mf", "vd": "vd"}


def grid(model):
    if model == "1b":
        tiles, knobs, vfs = ("w8m256n64", "w8m256n128"), ("_lazy", "_lazy_prio", "_lazy_ns_scal"), (0, 1)
    else:   # D=128: the fmha_kbank tile family (BLOCK_N 64, BLOCK_M = 32 * num_warps); dot2 fence OFF at 8B
        tiles, knobs, vfs = ("w8m256n64", "w4m128n64"), ("", "_prio", "_lazy", "_lazy_prio", "_lazy_ns_scal"), (0, )
    out = {}
    for t in tiles:
        nw, bm, bn = TILES[t]
        for vf in vfs:
            for k in knobs:
                out[f"{t}{'_vf1' if vf else ''}{k}"] = V(num_warps=nw, block_m=bm, block_n=bn, vf=vf, **KNOBSETS[k])
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="1b")
ap.add_argument("--only", nargs="*", default=None, help="config names")
ap.add_argument("--impls", nargs="*", default=list(IMPLS))
ap.add_argument("--check-ts", type=int, nargs="+", default=[2048, 8192])
ap.add_argument("--heads", type=int, nargs="+", default=[0, 5, 31])
ap.add_argument("--ts", type=int, nargs="+", default=[8192])
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--e2e", action="store_true")
ap.add_argument("--asm-t", type=int, default=8192)
ap.add_argument("--tag", default="")
ap.add_argument("--u-path", default=None, help="frozen plane (--dc-u-path); default random")
ap.add_argument("--u-layer", type=int, default=15)
ap.add_argument("--out-dir", default=".")
args = ap.parse_args()
configs = grid(args.model)
if args.only:
    configs = {k: v for k, v in configs.items() if k in args.only}
impls = [i for i in IMPLS if i in args.impls]
job = os.environ.get("JOB", "x")
asmdir = os.path.join(args.out_dir, f"asm_{args.model}{args.tag}_{job}")
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
    S[T] = dict(st=st, v=v, out=out, lse=lse, sm=sm, fe={"had": st.fe_for("had"), "2dp": st.fe_for("2dp", 6)})
    print(f"setup T={T} D={D} n_q={n_q} n_kv={n_kv} done", flush=True)

KW = ("num_warps", "wpe", "agpr0", "resc_mem", "prio", "abl", "v1x", "lazy", "block_m", "block_n", "ksa", "r2p",
      "lrke", "kpred", "kcf", "ablc", "lth", "vf", "ftz", "sched", "rv", "d1f", "exs", "vp", "kch")


def meth(impl):
    return "had" if impl == "had" else "2dp"


def fa_closure(T, impl, cfg):
    d = S[T]
    st = d["st"]
    c = dict(cfg)
    c.update(IMPLS[impl])
    kw = {k: c[k] for k in KW}
    kw["envtag"] = ENV_ORDER.index(c["env"])
    mod = fmha_d64
    if impl == "had":
        a = st.ah
        return lambda: mod.run_mx(*a[:2], None, *a[3:5], None, d["v"], d["out"]["had"], d["lse"], d["sm"]["had"], 0,
                                  **kw)
    a = st.a2p[8]
    return lambda: mod.run_mx(*a[:2], a[2], *a[3:5], a[5], d["v"], d["out"]["2dp"], d["lse"], d["sm"]["2dp"], 1, **kw)


def asm_stats(k):
    asm = k.asm["amdgcn"]
    g = lambda pat: int(re.search(pat, asm).group(1))  # noqa: E731
    return dict(vgpr=g(r"\.vgpr_count:\s*(\d+)"), spill=g(r"\.vgpr_spill_count:\s*(\d+)"),
                agpr=g(r"\.agpr_count:\s*(\d+)"), lds=g(r"\.group_segment_fixed_size:\s*(\d+)"))


CLASSES = [   # (name, regex on mnemonic); first match wins
    ("mfma_fp4", r"^v_mfma_scale_f32_32x32x64"), ("mfma_f16_k16", r"^v_mfma_f32_32x32x16_f16"),
    ("mfma_f16_k8", r"^v_mfma_f32_32x32x8_?f16"), ("mfma_other", r"^v_mfma"),
    ("v_dot2", r"^v_dot2_f32_f16"),
    ("v_exp", r"^v_exp_f32"), ("v_trans_other", r"^v_(log|rcp|rsq|sqrt)_"),
    ("v_permlane", r"^v_permlane"), ("v_pk", r"^v_pk_"), ("v_cvt_pk", r"^v_cvt_pk"),
    ("v_max", r"^v_max"), ("v_fma", r"^v_(fma|fmac|mac|mad)"), ("v_add", r"^v_(add|sub)"), ("v_mul", r"^v_mul"),
    ("v_cndmask", r"^v_cndmask"), ("v_cmp", r"^v_cmp"), ("v_mov", r"^v_(mov|accvgpr)"), ("v_dot_other", r"^v_dot"),
    ("valu_other", r"^v_"),
    ("ds_read", r"^ds_read"), ("ds_write", r"^ds_write"), ("buf_lds", r"^buffer_load.*"), ("global", r"^global_"),
    ("scratch", r"^scratch_"),
    ("s_waitcnt", r"^s_waitcnt"), ("s_barrier", r"^s_barrier"), ("s_setprio", r"^s_setprio"),
    ("sched_barrier", r"^s_sched_barrier|^sched_barrier"), ("s_branch", r"^s_c?branch"), ("salu", r"^s_"),
]


def loop_census(asm, tiles_per_iter=2):
    """Outermost loop body (largest backward branch span), per tile. valu_issue_c = VALU-port issue cycles of the
    non-MFMA vector instructions (4 per VALU / v_dot2, 8 per transcendental, 20 per permlane; tut README sec. 2),
    mfma_pipe_c = 32 cycles per MFMA (fp4 32x32x64, fp16 32x32x16 and 32x32x8 alike), mfma_issue_c = 8 per MFMA."""
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
        for name, pat in CLASSES:
            if re.match(pat, mn):
                c[name] += 1
                break
        else:
            c["other"] += 1
    nm = sum(v for k, v in c.items() if k.startswith("mfma"))
    nv = sum(v for k, v in c.items() if k.startswith("v_") or k == "valu_other")
    ntr = c["v_exp"] + c["v_trans_other"]
    c["mfma"] = nm
    c["valu"] = nv
    c["valu_issue_c"] = 4 * (nv - ntr - c["v_permlane"]) + 8 * ntr + 20 * c["v_permlane"]
    c["mfma_pipe_c"] = 32 * nm
    c["mfma_issue_c"] = 8 * nm
    return {k: v / tiles_per_iter for k, v in sorted(c.items())}


res = dict(model=args.model, configs=configs, impls={i: IMPLS[i] for i in impls}, info={}, census={}, check={}, runs={},
           fold={})
ok, fns = {}, {}
for env in ENV_ORDER:
    group = [n for n, c in configs.items() if c["env"] == env]
    if not group:
        continue
    for k_, v_ in ENVS[env].items():
        os.environ[k_] = v_
    for k_ in ("AMDGCN_SCALARIZE_PACKED_FOPS", "DISABLE_LLVM_OPT"):
        if k_ not in ENVS[env]:
            os.environ.pop(k_, None)
    for n in group:
        for im in impls:
            try:
                for T in all_t:
                    fn = fa_closure(T, im, configs[n])
                    k = fn()
                    torch.cuda.synchronize()
                    if T == args.asm_t:
                        info = asm_stats(k)
                        asm = k.asm["amdgcn"]
                        cen = loop_census(asm)
                        nw, bm, bn = configs[n]["num_warps"], configs[n]["block_m"], configs[n]["block_n"]
                        info["dot2_expected_per_tile"] = bm * bn // (64 * nw) if im.startswith("vd") else 0
                        res["info"][f"{n}/{im}"] = info
                        res["census"][f"{n}/{im}"] = cen
                        open(os.path.join(asmdir, f"{n}_{im}.amdgcn"), "w").write(asm)
                        print(f"compiled {n}/{im} env='{env}': {info} | loop/tile: v_dot2 {cen.get('v_dot2', 0)} "
                              f"mfma {cen.get('mfma', 0)} valu {cen.get('valu', 0)} valu_issue_c "
                              f"{cen.get('valu_issue_c', 0)} mfma_pipe_c {cen.get('mfma_pipe_c', 0)} total "
                              f"{cen.get('total', 0)}", flush=True)
                    fns[(T, n, im)] = fn
                ok[(n, im)] = True
            except Exception as e:  # noqa: BLE001
                ok[(n, im)] = False
                print(f"FAILED {n}/{im}: {type(e).__name__}: {str(e)[:3000]}", flush=True)
                traceback.print_exc(limit=4)

# correctness: O rel L2 and LSE abs error vs torch, every check T, every config and implementation; + fold check
for T in args.check_ts:
    d = S[T]
    st, s = d["st"], d["st"].s
    n_q, gpg = st.n_q, st.gpg
    dq = {"had": unpack_mx(s["pk_had"], s["sc_had"])[0], "2dp": unpack_mx(s["pk_2dp"], s["sc_2dp"])[0]}
    refs = {}
    fold = dict(max_abs_log2=0.0, max_logit_log2=0.0, max_rank2_log2=0.0)
    for m in ("had", "2dp"):
        d["fe"][m]()
        torch.cuda.synchronize()
        Vf = d["v"].float()
        for h in args.heads:
            hk = h // gpg
            Lg = dq[m][:, h] @ dq[m][:, n_q + hk].T
            if m == "2dp":
                cfh = s["cf"]                                   # fp16 [T, n_qk, 2]: a' (Q heads), b (K heads)
                r2 = cfh[:, h].float() @ cfh[:, n_q + hk].float().T
                sc = d["sm"]["2dp"] * 1.44269504089               # the kernel's qk_scale (log2 units)
                ref_l = (Lg + r2) * sc
                a_f = (cfh[:, h].float() * sc).half().float()     # a' pre-multiplied by the scale, rounded to fp16
                fold_l = Lg * sc + a_f @ cfh[:, n_q + hk].float().T
                fold["max_abs_log2"] = max(fold["max_abs_log2"], (fold_l - ref_l).abs().max().item())
                fold["max_logit_log2"] = max(fold["max_logit_log2"], ref_l.abs().max().item())
                fold["max_rank2_log2"] = max(fold["max_rank2_log2"], (r2 * sc).abs().max().item())
                del r2, ref_l, fold_l
                Lg = Lg + cfh[:, h].float() @ cfh[:, n_q + hk].float().T
            Lg = Lg * d["sm"][m]
            refs[(m, h)] = (torch.softmax(Lg, dim=-1) @ Vf[:, hk], torch.logsumexp(Lg, dim=-1))
            del Lg
    res["fold"][T] = fold
    print(f"fold check T={T}: max |(s+a'b)*scale - (s*scale + fp16(a'*scale).b)| = {fold['max_abs_log2']:.3e} "
          f"(log2 units; max |logit| {fold['max_logit_log2']:.2f}, max |rank-2 term| {fold['max_rank2_log2']:.2f})",
          flush=True)
    for n in configs:
        for im in impls:
            if not ok.get((n, im)):
                continue
            m = meth(im)
            d["fe"][m]()
            d["out"][m].fill_(float("nan"))
            d["lse"].fill_(float("nan"))
            fns[(T, n, im)]()
            torch.cuda.synchronize()
            eo, el = 0.0, 0.0
            for h in args.heads:
                ro, rl = refs[(m, h)]
                e = ((d["out"][m][:, h].float() - ro).norm() / ro.norm()).item()
                eo = max(eo, e) if e == e else float("inf")
                e = (d["lse"][h] - rl).abs().max().item()
                el = max(el, e) if e == e else float("inf")
            res["check"][f"T{T}/{n}/{im}"] = dict(o_rel_l2=eo, lse_max_abs=el)
            flag = "OK" if (eo < 2e-3 and el < 1e-2) else "BAD"
            print(f"check T={T} {n}/{im}: O max rel L2 {eo:.2e}  LSE max abs {el:.2e}  {flag}", flush=True)
            if flag == "BAD":
                ok[(n, im)] = False
    del refs, dq
    torch.cuda.empty_cache()


def fam(im):
    return im.split("_")[0]


def r3(x):
    return [statistics.median(x), min(x), max(x)]


for T in args.ts:
    d = S[T]
    fmap = {}
    for n in configs:
        for im in impls:
            if not ok.get((n, im)):
                continue
            fmap[f"fa:{n}:{im}"] = fns[(T, n, im)]
            if args.e2e:
                fmap[f"e2e:{n}:{im}"] = (lambda f=fns[(T, n, im)], fe=d["fe"][meth(im)]: (fe(), f()))
    if args.e2e:
        for m in ("had", "2dp"):
            fmap[f"fe:{m}"] = d["fe"][m]
    names = list(fmap)
    t = {x: [] for x in names}
    for rr in range(args.rounds):
        for x in (names if rr % 2 == 0 else names[::-1]):
            t[x].append(med_us(fmap[x]))
    rT = dict(raw=t, med={x: statistics.median(y) for x, y in t.items()})
    print(f"\n== T={T} ({args.rounds} interleaved rounds; median us; ratios = median [min, max] of per-round ratios)",
          flush=True)
    kinds = ("fa", "e2e") if args.e2e else ("fa", )
    # same-config: every implementation vs Hadamard of the SAME config
    for kind in kinds:
        print(f"-- same-config {kind}: impl/had - 1 (%)", flush=True)
        for n in configs:
            if f"{kind}:{n}:had" not in t:
                continue
            th = t[f"{kind}:{n}:had"]
            parts = [f"had {statistics.median(th):8.1f}"]
            for im in impls[1:]:
                key = f"{kind}:{n}:{im}"
                if key not in t:
                    parts.append(f"{im} --")
                    continue
                rat = [100 * (a / b - 1) for a, b in zip(t[key], th)]
                rT[f"same_{kind}:{n}:{im}"] = r3(rat)
                parts.append(f"{im} {statistics.median(t[key]):8.1f} {statistics.median(rat):+6.2f}% "
                             f"[{min(rat):+.2f},{max(rat):+.2f}]")
            # best placement per family within this config
            for f_ in ("mf", "vd"):
                cand = [im for im in impls if fam(im) == f_ and f"{kind}:{n}:{im}" in t]
                if cand:
                    b = min(cand, key=lambda im: statistics.median(t[f"{kind}:{n}:{im}"]))
                    rat = [100 * (a / b_ - 1) for a, b_ in zip(t[f"{kind}:{n}:{b}"], th)]
                    rT[f"samebest_{kind}:{n}:{f_}"] = dict(impl=b, ratio=r3(rat))
            print(f"{n:28s} " + " | ".join(parts), flush=True)
    # fastest-vs-fastest: per family, best (config, impl) by median; per-round ratios of the winners
    for kind in kinds:
        best = {}
        for f_ in ("had", "mf", "vd"):
            cand = [(n, im) for n in configs for im in impls if fam(im) == f_ and f"{kind}:{n}:{im}" in t]
            if cand:
                best[f_] = min(cand, key=lambda x: statistics.median(t[f"{kind}:{x[0]}:{x[1]}"]))
        if "had" not in best:
            continue
        bh = t[f"{kind}:{best['had'][0]}:had"]
        line = f"FASTEST {kind}: had {best['had'][0]} {statistics.median(bh):.1f} us"
        rT[f"best_{kind}"] = dict(had=list(best["had"]), had_us=statistics.median(bh))
        for f_ in ("mf", "vd"):
            if f_ not in best:
                continue
            n, im = best[f_]
            b2 = t[f"{kind}:{n}:{im}"]
            rat = [100 * (a / b - 1) for a, b in zip(b2, bh)]
            rT[f"best_{kind}"][f_] = dict(cfg=n, impl=im, us=statistics.median(b2), ratio=r3(rat))
            line += (f" | {f_} {n}/{im} {statistics.median(b2):.1f} us -> {statistics.median(rat):+.2f}% "
                     f"[{min(rat):+.2f},{max(rat):+.2f}]")
        if "mf" in best and "vd" in best:
            a_ = t[f"{kind}:{best['vd'][0]}:{best['vd'][1]}"]
            b_ = t[f"{kind}:{best['mf'][0]}:{best['mf'][1]}"]
            rat = [100 * (x / y - 1) for x, y in zip(a_, b_)]
            rT[f"best_{kind}"]["vd_vs_mf"] = r3(rat)
            line += f" | vd/mf {statistics.median(rat):+.2f}% [{min(rat):+.2f},{max(rat):+.2f}]"
        print(line, flush=True)
    if args.e2e:
        print(f"front end: had {statistics.median(t['fe:had']):.1f} us, 2dp {statistics.median(t['fe:2dp']):.1f} us",
              flush=True)
    res["runs"][T] = rT
    json.dump(res, open(os.path.join(args.out_dir, f"vdot2_{args.model}{args.tag}_{job}.json"), "w"), indent=1)
json.dump(res, open(os.path.join(args.out_dir, f"vdot2_{args.model}{args.tag}_{job}.json"), "w"), indent=1)
print("done", flush=True)
