# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Portions derived from fmha_v3.py of the gfx950 Gluon tutorial ("High-Performance Gluon Kernels on AMD GFX9",
# Copyright (c) 2026 AMD ROCm Software, MIT License: see LICENSE-gluon-tutorial).

"""fp16 flash attention (our kernel's QF16 mode) vs MXFP4 flash attention (Hadamard, 2DP rank-2 on MFMA), FA kernel
alone, B=1, 32 Q / 8 KV heads, non-causal, D = 64 (1B) or 128 (8B). Default toolchain, NO LLIR scheduler plugin.

Methods (one kernel source, fmha_fp.py, for the first four):
  f16   fp16 QK^T, softmax scale applied in the softmax (fma), SQ 0
  f16q  fp16 QK^T, softmax scale * log2(e) folded into Q (SQ 1, the tutorial's SCALE_ON_Q)
  had   MXFP4 QK^T, Hadamard
  mfma  MXFP4 QK^T + rank-2 term on one fp16 32x32x8 MFMA (2DP)
  tut / tutq  tutorial fmha_v3 (tut/fmha_v3_env.py) SCALE_ON_Q False / True: parity reference ONLY (D=128), never a baseline
Inputs: fp16 Q / K / V are views of the fused projection buffer s["C"] = [T, n_q + 2 n_kv, D] (same strides as the MXFP4
kernels' V, which is the same V view). Outputs: O fp16 [T, n_q, D] + LSE fp32 [n_q, T] for every method.
Modes:
  parity  (8B) f16 / f16q / tut / tutq at the plain L config (+ duplicate closures = identical-binary noise floor)
  smoke   compile every fp16 tile at this model, VGPR / spill, correctness
  full    screen (stage 1: tile x knob grid, env ''; stage 2: top-6 per method x compile envs [+ 2DP r2p 3]) at
          --screen-t, then final: top-4 per method + the same-config set, every --ts, correctness at --check-ts,
          interleaved rounds, do_bench (L2 flush) median AND CUDA-graph replay median.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
assert not os.environ.get("LLVM_PASS_PLUGIN_PATH"), "no LLIR scheduler plugin in this comparison"
import fmha_roof  # noqa: E402,I001,F401  -- FIRST (imports triton before torch)
import fmha_fp  # noqa: E402
import tut_env  # noqa: E402
TUT = tut_env.load()   # tutorial fmha_v3 + ENVTAG / preallocated LSE (parity reference only)
from common import MetaData  # noqa: E402
import argparse  # noqa: E402
import collections  # noqa: E402
import importlib.util  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import traceback  # noqa: E402

import torch  # noqa: E402
import triton  # noqa: E402
from triton.testing import do_bench_cudagraph  # noqa: E402

from fa_setup import GAMMA, Setup  # noqa: E402  (also puts ../ on sys.path: front ends, reference, med_us)
from bench_2dp_vs_hadamard import med_us  # noqa: E402
from reference import unpack_mx  # noqa: E402
from bench_roof_sets import ENVS, ENV_ORDER  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="8b")
ap.add_argument("--mode", default="parity", choices=["parity", "smoke", "full"])
ap.add_argument("--ts", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
ap.add_argument("--check-ts", type=int, nargs="+", default=[2048, 16384])
ap.add_argument("--heads", type=int, nargs="+", default=[0, 5, 31])
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--screen-t", type=int, default=8192)
ap.add_argument("--screen-rounds", type=int, default=3)
ap.add_argument("--top1", type=int, default=6)
ap.add_argument("--top2", type=int, default=4)
ap.add_argument("--asm-t", type=int, default=8192)
ap.add_argument("--methods", nargs="+", default=None)
ap.add_argument("--tag", default="")
ap.add_argument("--no-graph", action="store_true")
ap.add_argument("--u-path", default=None, help="frozen plane (--dc-u-path); default random")
ap.add_argument("--u-layer", type=int, default=15)
ap.add_argument("--out-dir", default=".")
args = ap.parse_args()
JOB = os.environ.get("JOB", "x")
print(f"triton {triton.__file__}  LLVM_PASS_PLUGIN_PATH={os.environ.get('LLVM_PASS_PLUGIN_PATH', '<unset>')}", flush=True)
TAG = f"{args.mode}{args.tag}_{args.model}_{JOB}"
ASMDIR = os.path.join(args.out_dir, f"asm_{TAG}")
os.makedirs(ASMDIR, exist_ok=True)

DEF = dict(num_warps=8, block_m=None, block_n=64, lazy=0, prio=0, vf=0, lrke=0, env="", r2p=0, exs=0, drf=0)


def C(**kw):
    d = dict(DEF)
    d.update(kw)
    return d


def cname(c):
    bm = c["block_m"] or 32 * c["num_warps"]
    n = f"w{c['num_warps']}m{bm}n{c['block_n']}"
    for k, s in (("lazy", "_lazy"), ("prio", "_prio"), ("vf", "_vf1"), ("lrke", "_e"), ("drf", "_drf")):
        if c[k]:
            n += s
    if c["r2p"]:
        n += f"_r2p{c['r2p']}"
    if c["exs"]:
        n += f"_exs{c['exs']}"
    if c["env"]:
        n += "_" + c["env"].replace("+", "")
    return n


S = {}


def setup(Ts):
    for T in Ts:
        if T in S:
            continue
        st = Setup(args.model, T, args.u_path, args.u_layer)
        s, D, n_q = st.s, st.D, st.n_q
        n_kv = n_q // st.gpg
        n_qk, h_tot = n_q + n_kv, n_q + 2 * n_kv
        Cv = s["C"].view(T, h_tot, D)
        S[T] = dict(st=st, D=D, n_q=n_q, n_kv=n_kv, q=Cv[:, :n_q], k=Cv[:, n_q:n_qk], v=Cv[:, n_qk:n_qk + n_kv],
                    out=torch.empty(T, n_q, D, device="cuda", dtype=torch.float16),
                    lse=torch.empty(n_q, T, device="cuda", dtype=torch.float32),
                    fe={"had": st.fe_for("had"), "2dp": st.fe_for("2dp", 6)})
        assert S[T]["v"].stride() == S[T]["q"].stride() == S[T]["k"].stride()
        print(f"setup T={T} D={D} n_q={n_q} n_kv={n_kv}; Q/K/V views of C, strides {tuple(S[T]['q'].stride())}",
              flush=True)


KW = ("num_warps", "block_m", "block_n", "lazy", "prio", "vf", "lrke", "exs", "drf")


def closure(T, m, c):
    d = S[T]
    st = d["st"]
    D = d["D"]
    et = ENV_ORDER.index(c["env"])
    kw = {k: c[k] for k in KW}
    if m in ("f16", "f16q"):
        return lambda: fmha_fp.run_fp16(d["q"], d["k"], d["v"], d["out"], d["lse"], 1.0 / math.sqrt(D),
                                        sq=int(m == "f16q"), envtag=et, **kw)
    if m in ("tut", "tutq"):
        md = MetaData(sm_scale=1.0 / math.sqrt(D))
        md.layout, md.max_seqlens_q, md.max_seqlens_k = "bshd", T, T
        q4, k4, v4 = (x.unsqueeze(0) for x in (d["q"], d["k"], d["v"]))
        o4, M = d["out"].unsqueeze(0), d["lse"].unsqueeze(0)
        return lambda: TUT.run_gluon_attention(q4, k4, v4, o4, md, scale_on_q=(m == "tutq"), envtag=et, M=M)
    if m == "had":
        a = st.ah
        return lambda: fmha_fp.run_mx(*a[:2], None, *a[3:5], None, d["v"], d["out"], d["lse"], 1.0 / math.sqrt(D), 0,
                                      envtag=et, r2p=0, **kw)
    a = st.a2p[8]
    return lambda: fmha_fp.run_mx(*a[:2], a[2], *a[3:5], a[5], d["v"], d["out"], d["lse"],
                                  1.0 / (math.sqrt(D) * (1.0 - GAMMA)), 1, envtag=et, r2p=c["r2p"], **kw)


def asm_stats(k):
    asm = k.asm["amdgcn"]
    g = lambda pat: int(re.search(pat, asm).group(1))  # noqa: E731
    return dict(vgpr=g(r"\.vgpr_count:\s*(\d+)"), spill=g(r"\.vgpr_spill_count:\s*(\d+)"),
                agpr=g(r"\.agpr_count:\s*(\d+)"), lds=g(r"\.group_segment_fixed_size:\s*(\d+)"),
                scratch=g(r"\.private_segment_fixed_size:\s*(\d+)"),
                scratch_instr=len(re.findall(r"\bscratch_", asm)),
                scratch_in_loop=sum(1 for x in loop_lines(asm) if x.startswith("scratch_")))


CL = [("mfma_fp4", r"^v_mfma_scale"), ("mfma_k16", r"^v_mfma_f32_32x32x16"), ("mfma_k8", r"^v_mfma_f32_32x32x8"),
      ("trans", r"^v_(exp|log|rcp)_"), ("permlane", r"^v_permlane"), ("v_pk", r"^v_pk_"), ("valu", r"^v_"),
      ("ds_read", r"^ds_read"), ("ds_write", r"^ds_write"), ("ds_other", r"^ds_"), ("buf", r"^buffer_"),
      ("global", r"^global_"), ("waitcnt", r"^s_waitcnt"), ("barrier", r"^s_barrier"), ("setprio", r"^s_setprio"),
      ("salu", r"^s_")]


def loop_lines(asm):
    lines = [x.strip() for x in asm.splitlines()]
    labels = {m.group(1): i for i, x in enumerate(lines) for m in [re.match(r"^(\.LBB\d+_\d+):", x)] if m}
    best = None
    for j, x in enumerate(lines):
        m = re.match(r"^s_c?branch\w*\s+(\.LBB\d+_\d+)", x)
        if m and m.group(1) in labels and labels[m.group(1)] < j:
            if best is None or j - labels[m.group(1)] > best[1] - best[0]:
                best = (labels[m.group(1)], j)
    return lines[best[0]:best[1] + 1] if best else []


def census(asm):
    """Main-loop census per s_barrier-delimited cluster (2 tiles per loop iteration) + waitcnt operands."""
    cl, cur, waits = [], collections.Counter(), []
    for x in loop_lines(asm):
        if not x or x.startswith((";", ".")) or x.split()[0].endswith(":"):
            continue
        mn = x.split()[0]
        if mn == "s_barrier":
            cl.append(dict(cur))
            cur = collections.Counter()
            continue
        if mn == "s_waitcnt":
            waits.append(x[len("s_waitcnt"):].strip())
        cur["total"] += 1
        for k, p in CL:
            if re.match(p, mn):
                cur[k] += 1
                break
    cl.append(dict(cur))
    tot = collections.Counter()
    for c in cl:
        tot.update(c)
    return dict(clusters=cl, total=dict(tot), waits=waits)


RES = dict(model=args.model, mode=args.mode, job=JOB, info={}, census={}, check={}, runs={}, excluded={}, notes=[])


def compile_set(V, Ts):
    """V: {key: (method, cfg)} -> {(key, T): fn} for the variants that compiled at every T."""
    fns = {}
    for env in ENV_ORDER:
        grp = [k for k, (m, c) in V.items() if c["env"] == env]
        if not grp:
            continue
        for k_, v_ in ENVS[env].items():
            os.environ[k_] = v_
        for k_ in ("AMDGCN_SCALARIZE_PACKED_FOPS", "DISABLE_LLVM_OPT"):
            if k_ not in ENVS[env]:
                os.environ.pop(k_, None)
        for key in grp:
            m, c = V[key]
            try:
                got = {}
                for T in Ts:
                    fn = closure(T, m, c)
                    kern = fn()
                    torch.cuda.synchronize()
                    got[(key, T)] = fn
                    info = asm_stats(kern)
                    RES.setdefault("info_T", {})[f"{key}@{T}"] = info
                    old_i = RES["info"].get(key)
                    if old_i is None:
                        RES["info"][key] = dict(info)
                    else:   # worst case over every compiled T (N_CTX is a constexpr: one binary per T)
                        for f_ in ("vgpr", "spill", "scratch", "scratch_instr", "scratch_in_loop"):
                            old_i[f_] = max(old_i[f_], info[f_])
                    if T == args.asm_t or (args.asm_t not in Ts and T == max(Ts)):
                        asm = kern.asm["amdgcn"]
                        RES["census"][key] = census(asm)
                        open(os.path.join(ASMDIR, key.replace("/", "__") + ".amdgcn"), "w").write(asm)
                        print(f"compiled {key} env='{env}' T={T}: {info}", flush=True)
                inf = RES["info"].get(key, {})
                if inf.get("spill", 0) or inf.get("scratch", 0):
                    print(f"SPILL {key}: {inf} (strict selection excludes it; loose selection only if no in-loop "
                          f"scratch)", flush=True)
                fns.update(got)
            except Exception as e:  # noqa: BLE001
                RES["excluded"][key] = f"compile/run failure: {type(e).__name__}: {str(e)[:300]}"
                print(f"FAILED {key}: {type(e).__name__}: {str(e)[:1500]}", flush=True)
                traceback.print_exc(limit=3)
    for k_ in ("AMDGCN_SCALARIZE_PACKED_FOPS", "DISABLE_LLVM_OPT"):
        os.environ.pop(k_, None)
    return fns


def check_set(V, fns, Ts):
    """O rel L2 / LSE max abs vs torch fp32 on each method's own inputs (fp16: the C views; MXFP4: dequantized codes
    [+ rank-2 coefficients]), heads --heads. Failing variants are excluded."""
    for T in Ts:
        d = S[T]
        st, s = d["st"], d["st"].s
        n_q, gpg = st.n_q, st.gpg
        keys = [k for k in V if (k, T) in fns]
        by_fe = collections.defaultdict(list)
        for k in keys:
            m = V[k][0]
            by_fe["2dp" if m == "mfma" else "had"].append(k)
        for fe, ks in by_fe.items():
            d["fe"][fe]()                      # the front end that produced this method's inputs (also rewrites C)
            torch.cuda.synchronize()
            refs = {}
            ms = sorted({V[k][0] for k in ks})
            for m in ms:
                for h in args.heads:
                    hk = h // gpg
                    if m in ("f16", "f16q", "tut", "tutq"):
                        Lg = (d["q"][:, h].float() @ d["k"][:, hk].float().T) / math.sqrt(d["D"])
                    else:
                        meth = "2dp" if m == "mfma" else "had"
                        dq = unpack_mx(s[f"pk_{meth}"], s[f"sc_{meth}"])[0]
                        Lg = dq[:, h] @ dq[:, n_q + hk].T
                        if m == "mfma":
                            cf = s["cf"].float()
                            Lg = Lg + cf[:, h] @ cf[:, n_q + hk].T
                            Lg = Lg / (math.sqrt(d["D"]) * (1.0 - GAMMA))
                        else:
                            Lg = Lg / math.sqrt(d["D"])
                        del dq
                    refs[(m, h)] = (torch.softmax(Lg, dim=-1) @ d["v"][:, hk].float(), torch.logsumexp(Lg, dim=-1))
                    del Lg
            for k in ks:
                m = V[k][0]
                d["out"].fill_(float("nan"))
                d["lse"].fill_(float("nan"))
                fns[(k, T)]()
                torch.cuda.synchronize()
                eo, el = 0.0, 0.0
                for h in args.heads:
                    ro, rl = refs[(m, h)]
                    e = ((d["out"][:, h].float() - ro).norm() / ro.norm()).item()
                    eo = max(eo, e) if e == e else float("inf")
                    e = (d["lse"][h] - rl).abs().max().item()
                    el = max(el, e) if e == e else float("inf")
                ok = eo < 2e-3 and el < 1e-2
                RES["check"][f"T{T}/{k}"] = dict(o_rel_l2=eo, lse_max_abs=el, ok=ok)
                print(f"check T={T} {k}: O max rel L2 {eo:.2e}  LSE max abs {el:.2e}  {'OK' if ok else 'BAD'}",
                      flush=True)
                if not ok:
                    RES["excluded"][k] = f"correctness T={T}"
            del refs
            torch.cuda.empty_cache()
    S[min(Ts)]["fe"]["had"]()   # leave C = the Hadamard front-end output (J1), as at setup
    torch.cuda.synchronize()


def graph_us(fn):
    return float(do_bench_cudagraph(fn, rep=100, return_mode="median")) * 1e3


def time_set(keys, fns, T, rounds, graph):
    tb = {k: [] for k in keys}
    tg = {k: [] for k in keys}
    for r in range(rounds):
        for k in (keys if r % 2 == 0 else keys[::-1]):
            tb[k].append(med_us(fns[(k, T)]))
            if graph:
                tg[k].append(graph_us(fns[(k, T)]))
    return tb, tg


def usable(k):
    return k not in RES["excluded"]


def strict_ok(k):   # no spill / scratch anywhere (the reported selection)
    i = RES["info"].get(k.split("#")[0], {})
    return usable(k) and i.get("spill", 1) == 0 and i.get("scratch", 1) == 0


def loose_ok(k):    # spills allowed outside the main loop only (sensitivity: would a spilling config be faster?)
    i = RES["info"].get(k.split("#")[0], {})
    return usable(k) and i.get("scratch_in_loop", 1) == 0


def flag(k):
    i = RES["info"].get(k, {})
    if i.get("spill", 0) or i.get("scratch", 0):
        return f" [spill {i.get('spill')}, scratch {i.get('scratch')} B, in-loop scratch instr {i.get('scratch_in_loop')}]"
    return ""


def summ(x):
    return dict(median=statistics.median(x), min=min(x), max=max(x))


def ratios(a, b):   # a time / b time, per round
    r = [x / y for x, y in zip(a, b)]
    return dict(median=statistics.median(r), min=min(r), max=max(r))


def norm_asm(asm):
    out = []
    for ln in asm.splitlines():
        s_ = ln.strip()
        if not s_ or s_.startswith(";") or (s_.startswith(".") and not re.match(r"^\.LBB", s_)):
            continue
        out.append(re.sub(r"mx_attn_fwd_\w+", "KNAME", s_))
    return out


def ident_check(T, cfgs):
    """The MXFP4 binaries of fa_fp16/fmha_fp.py (QF16 0) vs the fa_d64_combo/fmha_d64.py original, same knobs."""
    import importlib.util as iu
    sp = iu.spec_from_file_location("fmha_d64_orig", os.path.join(HERE, "fmha_d64.py"))
    orig = iu.module_from_spec(sp)
    sp.loader.exec_module(orig)
    d = S[T]
    st = d["st"]
    for c in cfgs:
        kw = {k: c[k] for k in KW}
        for m in ("had", "mfma"):
            asms = []
            for mod in (orig, fmha_fp):
                if m == "had":
                    a = st.ah
                    k = mod.run_mx(*a[:2], None, *a[3:5], None, d["v"], d["out"], d["lse"], 1.0 / math.sqrt(d["D"]), 0,
                                   r2p=0, **kw)
                else:
                    a = st.a2p[8]
                    k = mod.run_mx(*a[:2], a[2], *a[3:5], a[5], d["v"], d["out"], d["lse"],
                                   1.0 / (math.sqrt(d["D"]) * (1.0 - GAMMA)), 1, r2p=c["r2p"], **kw)
                torch.cuda.synchronize()
                asms.append(norm_asm(k.asm["amdgcn"]))
            same = asms[0] == asms[1]
            nd = sum(1 for x, y in zip(*asms) if x != y) + abs(len(asms[0]) - len(asms[1]))
            RES.setdefault("ident_vs_fmha_d64", {})[f"{m}/{cname(c)}"] = dict(identical=same, lines=len(asms[0]),
                                                                            differing=nd)
            print(f"ISA ident fa_fp16 vs fmha_d64 {m}/{cname(c)}: identical={same} lines={len(asms[0])} "
                  f"differing={nd}", flush=True)


# ------------------------------------------------------------------------------------------------------------------
D_ = 64 if args.model == "1b" else 128
if args.model == "1b":
    TILES = [(8, 256, 64), (8, 256, 128), (4, 128, 64), (4, 128, 128), (4, 256, 64), (4, 256, 128)]
else:   # D=128: BLOCK_N 64 and BLOCK_M = 32 * num_warps only (both MXFP4 and fp16; the V / K layouts of fmha_kbank)
    TILES = [(8, 256, 64), (4, 128, 64)]
VF_SET = (0, 1) if D_ == 64 else (0, )   # dot2 fence: illegal at D=128
METHODS = args.methods or (["f16", "f16q", "had", "mfma"])


def tcfg(t, **kw):
    return C(num_warps=t[0], block_m=t[1], block_n=t[2], **kw)


if args.mode == "parity":
    Ts = args.ts
    setup(Ts)
    V = {}
    for env in ENV_ORDER:   # the same plain L config under every compile env (the tutorial was tuned for nosink)
        L = C(env=env)
        sfx = "" if not env else "_" + env.replace("+", "")
        for m in ("f16", "f16q", "tut", "tutq"):
            V[f"{m}/L{sfx}"] = (m, L)
    fns = compile_set(V, Ts)
    ident_check(args.asm_t, [C(), C(lazy=1, prio=1)])
    for T in Ts:   # identical-binary noise floor: the same compiled closure timed under two keys
        for m in ("f16q", "tutq"):
            if (f"{m}/L", T) in fns:
                fns[(f"{m}/L#2", T)] = fns[(f"{m}/L", T)]
    check_set(V, fns, args.check_ts)
    for T in Ts:
        keys = [k for k in list(V) + ["f16q/L#2", "tutq/L#2"] if (k, T) in fns and usable(k.split("#")[0])]
        tb, tg = time_set(keys, fns, T, args.rounds, not args.no_graph)
        flops = 4 * S[T]["n_q"] * T * T * D_
        run = dict(bench={k: summ(v) for k, v in tb.items()}, graph={k: summ(v) for k, v in tg.items() if v},
                   raw_bench=tb, raw_graph=tg, pairs={})
        print(f"\n== parity {args.model} T={T} ({args.rounds} interleaved rounds; median [min,max] us)", flush=True)
        for k in keys:
            g = f" | graph {statistics.median(tg[k]):8.1f}" if tg[k] else ""
            print(f"   {k:<12} bench {statistics.median(tb[k]):8.1f} [{min(tb[k]):.1f},{max(tb[k]):.1f}] "
                  f"({flops / statistics.median(tb[k]) / 1e6:5.0f} TF/s){g}", flush=True)
        prs = [("f16q/L", "f16q/L#2"), ("tutq/L", "tutq/L#2")]
        for env in ENV_ORDER:
            sfx = "" if not env else "_" + env.replace("+", "")
            prs += [(f"f16/L{sfx}", f"tut/L{sfx}"), (f"f16q/L{sfx}", f"tutq/L{sfx}"), (f"f16q/L{sfx}", f"f16/L{sfx}")]
        for a, b in prs:
            if a in tb and b in tb:
                rb = ratios(tb[a], tb[b])
                line = f"   {a} / {b}: bench {100 * (rb['median'] - 1):+.2f}% [{100 * (rb['min'] - 1):+.2f},{100 * (rb['max'] - 1):+.2f}]"
                pr = dict(bench=rb)
                if tg[a] and tg[b]:
                    rg = ratios(tg[a], tg[b])
                    pr["graph"] = rg
                    line += f" | graph {100 * (rg['median'] - 1):+.2f}% [{100 * (rg['min'] - 1):+.2f},{100 * (rg['max'] - 1):+.2f}]"
                run["pairs"][f"{a} vs {b}"] = pr
                print(line, flush=True)
        for tm, tt in (("bench", tb), ("graph", tg)):
            if not all(tt.get(k) for k in keys):
                continue
            bo = min([k for k in keys if k.startswith("f16") and "#" not in k], key=lambda k: statistics.median(tt[k]))
            bt = min([k for k in keys if k.startswith("tut") and "#" not in k], key=lambda k: statistics.median(tt[k]))
            rr = ratios(tt[bo], tt[bt])
            run["pairs"][f"best ours ({bo}) vs best tutorial ({bt}) [{tm}]"] = rr
            print(f"   [{tm}] best ours {bo} {statistics.median(tt[bo]):.1f} / best tutorial {bt} "
                  f"{statistics.median(tt[bt]):.1f}: {100 * (rr['median'] - 1):+.2f}% "
                  f"[{100 * (rr['min'] - 1):+.2f},{100 * (rr['max'] - 1):+.2f}]", flush=True)
        RES["runs"][T] = run
        json.dump(RES, open(os.path.join(args.out_dir, f"fp_{TAG}.json"), "w"), indent=1, default=str)   # partial

elif args.mode == "smoke":
    Ts = sorted(set(args.check_ts + [args.asm_t]))
    setup(Ts)
    V = {}
    tiles = TILES[:4] if D_ == 64 else TILES
    for t in tiles:
        for m in ("f16", "f16q", "had", "mfma"):
            for kw in (dict(), dict(lazy=1, prio=1), dict(lazy=1, vf=VF_SET[-1]), dict(lazy=1, lrke=1)):
                for drf in (0, 1):
                    if m in ("had", "mfma") and not drf:
                        continue
                    for env in ("", "nosink"):
                        c = tcfg(t, drf=drf, env=env, **kw)
                        V[f"{m}/{cname(c)}"] = (m, c)
    fns = compile_set(V, Ts)
    check_set(V, fns, args.check_ts)
    if D_ == 64:
        ident_check(args.asm_t, [tcfg((8, 256, 128), lazy=1, vf=1, lrke=1), tcfg((8, 256, 128), lazy=1, vf=1, prio=1)])

else:   # full
    setup(sorted(set(args.ts + args.check_ts + [args.screen_t])))
    # ---- stage 1: the same tile x knob grid for every method (env '') ----
    tiles1 = TILES[:4]   # (R=2 tiles w4m256: fp16 spills in-loop at every knob, job 92424; dropped for ALL methods)
    grid = [tcfg(t, lazy=a, prio=b, vf=f, lrke=e, drf=r) for t in tiles1 for a in (0, 1) for b in (0, 1)
            for f in VF_SET for e in (0, 1) for r in (0, 1)]
    V1 = {f"{m}/{cname(c)}": (m, c) for m in METHODS for c in grid}
    print(f"stage 1: {len(grid)} configs x {len(METHODS)} methods", flush=True)
    T0 = args.screen_t
    fns = compile_set(V1, [T0])
    check_set(V1, fns, [T0])
    keys = [k for k in V1 if (k, T0) in fns and loose_ok(k)]
    tb, _ = time_set(keys, fns, T0, args.screen_rounds, False)
    med1 = {k: statistics.median(v) for k, v in tb.items()}
    RES["stage1"] = dict(T=T0, median=med1)

    def topk(V, med, m, n_strict, n_loose):
        ks = sorted([k for k in med if V[k][0] == m], key=lambda k: med[k])
        st = [k for k in ks if strict_ok(k)][:n_strict]
        lo = [k for k in ks if k not in st][:n_loose]
        return st + lo

    top1 = {m: topk(V1, med1, m, args.top1, 2) for m in METHODS}
    for m in METHODS:
        print(f"stage 1 top {m}: " + ", ".join(f"{k.split('/')[1]} {med1[k]:.1f}{' (spill)' if not strict_ok(k) else ''}"
                                               for k in top1[m]), flush=True)
    # ---- stage 2: compile envs on each method's top configs (+ 2DP rank-2 placement r2p 3) ----
    V2 = {}
    for m in METHODS:
        for k in top1[m]:
            c0 = V1[k][1]
            V2[k] = V1[k]
            for env in ENV_ORDER[1:]:
                c = dict(c0, env=env)
                V2[f"{m}/{cname(c)}"] = (m, c)
            if m == "mfma":
                for env in ENV_ORDER:
                    c = dict(c0, env=env, r2p=3)
                    V2[f"{m}/{cname(c)}"] = (m, c)
    new = {k: v for k, v in V2.items() if k not in V1}
    fns.update(compile_set(new, [T0]))
    check_set(new, fns, [T0])
    keys2 = [k for k in V2 if (k, T0) in fns and loose_ok(k)]
    tb, _ = time_set(keys2, fns, T0, args.screen_rounds, False)
    med2 = {k: statistics.median(v) for k, v in tb.items()}
    RES["stage2"] = dict(T=T0, median=med2)
    json.dump(RES, open(os.path.join(args.out_dir, f"fp_{TAG}.json"), "w"), indent=1, default=str)   # partial
    top2 = {m: topk(V2, med2, m, args.top2, 1) for m in METHODS}
    for m in METHODS:
        print(f"stage 2 top {m}: " + ", ".join(f"{k.split('/')[1]} {med2[k]:.1f}{' (spill)' if not strict_ok(k) else ''}"
                                               for k in top2[m]), flush=True)
    # ---- final: each method's top + the same-config set, every T, checks at check-ts, both timing methods ----
    if args.model == "1b":
        t1 = (8, 256, 128)
        same = [tcfg((8, 256, 64)), tcfg((8, 256, 64), drf=1), tcfg(t1), tcfg(t1, drf=1), tcfg(t1, lazy=1, drf=1),
                tcfg(t1, lazy=1, vf=1, drf=1), tcfg(t1, lazy=1, vf=1, prio=1, drf=1),
                tcfg(t1, lazy=1, vf=1, lrke=1), tcfg(t1, lazy=1, vf=1, prio=1)]   # last two: MXFP4 bests (92401)
    else:
        L = (8, 256, 64)
        same = [tcfg(L), tcfg(L, drf=1), tcfg(L, lazy=1, drf=1), tcfg(L, lazy=1, prio=1, drf=1),
                tcfg(L, env="scal", drf=1), tcfg(L, lazy=1, env="nosink+scal"), tcfg(L, lazy=1, prio=1)]
    VF = {}
    for m in METHODS:
        for k in top2[m]:
            VF[k] = V2[k]
        for c in same:
            VF[f"{m}/{cname(c)}"] = (m, c)
    if args.model == "8b":
        for m in ("tut", "tutq"):
            for env in ENV_ORDER:
                c = C(env=env)
                VF[f"{m}/{cname(c)}"] = (m, c)
    RES["final_variants"] = {k: dict(method=m, cfg=c) for k, (m, c) in VF.items()}
    RES["same_config"] = [cname(c) for c in same]
    print(f"final: {len(VF)} variants", flush=True)
    fnsF = compile_set(VF, args.ts)
    check_set(VF, fnsF, args.check_ts)
    ALLM = METHODS + (["tut", "tutq"] if args.model == "8b" else [])
    for T in args.ts:
        keys = [k for k in VF if (k, T) in fnsF and usable(k)]
        tb, tg = time_set(keys, fnsF, T, args.rounds, not args.no_graph)
        flops = 4 * S[T]["n_q"] * T * T * D_
        run = dict(bench={k: summ(v) for k, v in tb.items()}, graph={k: summ(v) for k, v in tg.items() if v},
                   raw_bench=tb, raw_graph=tg, tflops_bench={k: flops / statistics.median(v) / 1e6 for k, v in tb.items()},
                   tflops_graph={k: flops / statistics.median(v) / 1e6 for k, v in tg.items() if v})
        print(f"\n== final {args.model} T={T} ({args.rounds} interleaved rounds; median [min,max] us; TF/s)", flush=True)
        for k in keys:
            g = (f" | graph {statistics.median(tg[k]):8.1f} ({flops / statistics.median(tg[k]) / 1e6:5.0f} TF/s)"
                 if tg[k] else "")
            print(f"   {k:<44} bench {statistics.median(tb[k]):8.1f} [{min(tb[k]):.1f},{max(tb[k]):.1f}] "
                  f"({flops / statistics.median(tb[k]) / 1e6:5.0f} TF/s){g}{flag(k)}", flush=True)
        for tm, tt in (("bench", tb), ("graph", tg)):
            if not any(tt.values()):
                continue
            med = lambda k: statistics.median(tt[k])  # noqa: E731
            for pol, okf in (("strict", strict_ok), ("loose", loose_ok)):
                best = {}
                for m in ALLM:
                    ks = [k for k in keys if VF[k][0] == m and (okf(k) or m in ("tut", "tutq"))]
                    if ks:
                        best[m] = min(ks, key=med)
                f16c = [best[m] for m in ("f16", "f16q") if m in best]
                if not f16c:
                    continue
                f16b = min(f16c, key=med)
                out = dict(best=best, fp16_best=f16b, speedup={})
                print(f"   [{tm}/{pol}] fastest: " + ", ".join(
                    f"{m}={best[m].split('/')[1]} {med(best[m]):.1f} ({flops / med(best[m]) / 1e6:.0f} TF/s)"
                    for m in best), flush=True)
                for m in ("had", "mfma"):
                    if m in best:
                        for ref in ("fp16_best", "f16", "f16q"):
                            rk = f16b if ref == "fp16_best" else best.get(ref)
                            if rk is None:
                                continue
                            rr = ratios(tt[rk], tt[best[m]])
                            out["speedup"][f"{m}_over_{ref}"] = dict(rr, fp16=rk, mx=best[m])
                            print(f"   [{tm}/{pol}] speedup {m} ({best[m].split('/')[1]}) over {ref} ({rk}): "
                                  f"{rr['median']:.3f}x [{rr['min']:.3f}, {rr['max']:.3f}]", flush=True)
                tk = [best[m] for m in ("tut", "tutq") if m in best]
                if tk:
                    tb_ = min(tk, key=med)
                    rr = ratios(tt[tb_], tt[f16b])
                    out["tut_best_over_f16_best"] = dict(rr, tut=tb_, f16=f16b)
                    print(f"   [{tm}/{pol}] parity reference: tutorial best ({tb_}{flag(tb_)}) / our fp16 best ({f16b}): "
                          f"{rr['median']:.3f} [{rr['min']:.3f}, {rr['max']:.3f}] (>1: ours faster)", flush=True)
                run[f"summary_{tm}_{pol}"] = out
            sc = {}
            for c in same:
                n = cname(c)
                row = {}
                for m in METHODS:
                    k = f"{m}/{n}"
                    if k in tt and tt[k]:
                        row[m] = med(k)
                        row[m + "_spill"] = RES["info"].get(k, {}).get("spill")
                for m in ("had", "mfma"):
                    for ref in ("f16", "f16q"):
                        if tt.get(f"{m}/{n}") and tt.get(f"{ref}/{n}"):
                            row[f"{m}_over_{ref}"] = ratios(tt[f"{ref}/{n}"], tt[f"{m}/{n}"])
                sc[n] = row
                print(f"   [{tm}] same-config {n}: " + "  ".join(
                    f"{m} {row[m]:.1f}{'*' if row.get(m + '_spill') else ''}" for m in METHODS if m in row) + "  " +
                    "  ".join(f"{x} {row[x]['median']:.3f}x" for x in row if "_over_" in x), flush=True)
            run[f"same_config_{tm}"] = sc
        RES["runs"][T] = run
        json.dump(RES, open(os.path.join(args.out_dir, f"fp_{TAG}.json"), "w"), indent=1, default=str)   # partial

json.dump(RES, open(os.path.join(args.out_dir, f"fp_{TAG}.json"), "w"), indent=1, default=str)
print("done", flush=True)
