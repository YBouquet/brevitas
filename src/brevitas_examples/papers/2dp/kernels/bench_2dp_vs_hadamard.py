# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""2DP vs Hadamard latency on gfx950: QKV projection -> RoPE -> transform -> MXFP4 -> QK^T logits.

  2DP       P1 = p1_2dp.front_end_2dp(epi=1): GEMM + RoPE + frozen-plane extraction + MXFP4, a' = (1-g) a
            P2 = p2_qk.qk(corr=1): fp4 QK^T + a'.b (rank-2 on the matrix core or the VALU, faster one)
  Hadamard  J1 = p1_2dp.front_end_2dp(epi=0): the same GEMM, fp16 Q/K/V store
            J2 = had_j2.rope_quant_hd: RoPE + H_D / sqrt(D) + MXFP4 (Triton)
            P2 = p2_qk.qk(corr=0): bare fp4 QK^T
Endpoint: fp32 logits [n_q, T, T] of all Q heads, no 1/sqrt(D) (owned by flash attention). The 2DP output
is (1-g) S: 1/(1-g) joins the softmax scale. Both QK^T kernels are autotuned over the same tile candidates.

    python bench_2dp_vs_hadamard.py --model 8b --ts 1024 2048 4096 8192 \
        --u-path U_selfgen_8b_2dp_allrand_s0.pt --u-layer 15 --out bench_8b.json

Needs a gfx950 GPU and a Triton build with Gluon gfx950 support (tag gfx950-tutorial-v1.0). The first T is
also checked against the torch references in reference.py.
"""

# isort: off
import triton  # noqa: F401  -- import triton BEFORE torch (custom libtriton vs torch LLVM symbols)
# isort: on
import argparse
import json
import math
import os
import sys

import torch
from triton.testing import do_bench

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from had_j2 import rope_quant_hd
from layout_2dp import permute_fused_qkv_split_rope_hd
from layout_2dp import rope_tables_hd
from layout_2dp import ublk_constants
from p1_2dp import front_end_2dp
from p2_qk import qk
from p2_qk import tune_qk
from reference import mx_quant
from reference import ref_2dp
from reference import ref_logits
from reference import rope_std
from reference import sylvester
from reference import unpack_mx

MODELS = {
    "1b": dict(name="Llama-3.2-1B", d_model=2048, n_q=32, n_kv=8, hd=64),
    "8b": dict(name="Llama-3.1-8B", d_model=4096, n_q=32, n_kv=8, hd=128),}


def med_us(fn):
    return float(do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8])[0]) * 1e3


def kcontig(w):
    return w.t().contiguous().t()


def load_u(cfg, path, layer, dev):
    """Frozen plane [n_kv, D/2, 2]. Latency does not depend on its values: random orthonormal if no file."""
    if path:
        U = torch.load(path, map_location="cpu").float()
        U = U[layer] if U.dim() == 4 else U
    else:
        g = torch.Generator().manual_seed(0)
        U = torch.linalg.qr(torch.randn(cfg["n_kv"], cfg["hd"] // 2, 2, generator=g))[0]
    return U.to(dev).contiguous()


def build(cfg, T, U, dev):
    D, n_q, n_kv, dm = cfg["hd"], cfg["n_q"], cfg["n_kv"], cfg["d_model"]
    n_qk, h_tot = n_q + n_kv, n_q + 2 * n_kv
    torch.manual_seed(0)
    x = torch.randn(T, dm, device=dev, dtype=torch.float16)
    W = torch.randn(dm, h_tot * D, device=dev, dtype=torch.float16) * dm ** -0.5
    inv = 500000.0 ** (-torch.arange(D // 2, device=dev).float() / (D // 2))
    cos, sin = rope_tables_hd(torch.arange(T, device=dev).float(), inv)
    s = dict(
        x=x,
        W=W,
        cos=cos,
        sin=sin,
        W2d=kcontig(permute_fused_qkv_split_rope_hd(W, n_q, n_kv, D)),
        Wj=kcontig(W),
        UT=ublk_constants(U, n_q, n_kv, D),
        C=torch.empty(T, h_tot * D, device=dev, dtype=torch.float16),
        out=torch.empty(n_q, T, T, device=dev, dtype=torch.float32))
    for k in ("2dp", "had"):
        s[f"pk_{k}"] = torch.empty(T, n_qk, D // 2, device=dev, dtype=torch.uint8)
        s[f"sc_{k}"] = torch.empty(T, n_qk, D // 32, device=dev, dtype=torch.uint8)
    # plane coordinates head-major behind a [T, n_qk, 2] view: a QK^T tile reads one contiguous run
    s["cf"] = torch.empty(n_qk, T, 2, device=dev, dtype=torch.float16).permute(1, 0, 2)
    return s


def pipes(cfg, s, gamma):
    D, n_q, n_kv = cfg["hd"], cfg["n_q"], cfg["n_kv"]
    n_qk, h_tot = n_q + n_kv, n_q + 2 * n_kv
    x, cos, sin, C = s["x"], s["cos"], s["sin"], s["C"]
    p1 = dict(
        p1_2dp=lambda: front_end_2dp(
            x,
            s["W2d"],
            C,
            1,
            D,
            n_q,
            n_kv,
            cos,
            sin,
            s["pk_2dp"],
            s["sc_2dp"],
            s["cf"],
            s["UT"],
            qfac=1.0 - gamma),
        j1=lambda: front_end_2dp(x, s["Wj"], C, 0, D, n_q, n_kv),
        j2=lambda: rope_quant_hd(
            C.view(-1, h_tot, D)[:, :n_qk], cos, sin, s["pk_had"], s["sc_had"]))
    pk, sc, cf = s["pk_2dp"], s["sc_2dp"], s["cf"]
    a2 = (pk[:, :n_q], sc[:, :n_q], cf[:, :n_q], pk[:, n_q:], sc[:, n_q:], cf[:, n_q:], s["out"])
    ph, sh = s["pk_had"], s["sc_had"]
    ah = (ph[:, :n_q], sh[:, :n_q], None, ph[:, n_q:], sh[:, n_q:], None, s["out"])
    gpg = n_q // n_kv
    p2 = dict(
        qk_2dp_mfma=(a2 + (gpg, D), dict(corr=1, corr_mfma=True)),
        qk_2dp_valu=(a2 + (gpg, D), dict(corr=1, corr_mfma=False)),
        qk_had=(ah + (gpg, D), dict(corr=0)))
    return p1, p2


def check(cfg, s, p1, p2, cfgs, U, gamma):
    """fp4 codes / logits of both pipelines vs the torch references."""
    D, n_q, n_kv = cfg["hd"], cfg["n_q"], cfg["n_kv"]
    n_qk, h_tot, gpg = n_q + n_kv, n_q + 2 * n_kv, n_q // n_kv
    o = rope_std((s["x"].float() @ s["W"].float()).view(-1, h_tot, D)[:, :n_qk], s["cos"], s["sin"])
    res = {}

    def logit_err(out, ref):
        return ((out - ref).norm() / ref.norm()).item()

    # 2DP: the kernel returns (1-g) S; x 1/(1-g) it must match R/(1-g) + a.b
    a, rz = ref_2dp(o, U, n_q, gpg)
    dref, cref, _ = mx_quant(rz)
    p1["p1_2dp"]()
    torch.cuda.synchronize()
    _, codes = unpack_mx(s["pk_2dp"], s["sc_2dp"])
    res["2dp_code_mismatch"] = (codes != cref).float().mean().item()
    target = ref_logits(
        dref[:, :n_q], dref[:, n_q:], gpg, a[:, :n_q], a[:, n_q:], 1.0 / (1.0 - gamma))
    for k in ("qk_2dp_mfma", "qk_2dp_valu"):
        args, kw = p2[k]
        qk(*args, **kw, cfg=cfgs[k])
        torch.cuda.synchronize()
        res[f"{k}_logits_rel_l2"] = logit_err(s["out"] / (1.0 - gamma), target)
    del target
    # Hadamard
    dref, cref, _ = mx_quant(o @ (sylvester(D, o.device) / math.sqrt(D)))
    p1["j1"]()
    p1["j2"]()
    torch.cuda.synchronize()
    _, codes = unpack_mx(s["pk_had"], s["sc_had"])
    res["had_code_mismatch"] = (codes != cref).float().mean().item()
    args, kw = p2["qk_had"]
    qk(*args, **kw, cfg=cfgs["qk_had"])
    torch.cuda.synchronize()
    res["qk_had_logits_rel_l2"] = logit_err(s["out"], ref_logits(dref[:, :n_q], dref[:, n_q:], gpg))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(MODELS), default="1b")
    ap.add_argument("--ts", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    ap.add_argument("--gamma", type=float, default=0.047)
    ap.add_argument(
        "--u-path", default=None, help="frozen plane [L, n_kv, D/2, 2] (--dc-u-path file)")
    ap.add_argument("--u-layer", type=int, default=15)
    ap.add_argument("--out", default=None, help="write the results as JSON")
    args = ap.parse_args()
    dev = "cuda"
    cfg = MODELS[args.model]
    U = load_u(cfg, args.u_path, args.u_layer, dev)
    result = dict(model=cfg, gamma=args.gamma, device=torch.cuda.get_device_name(0), runs={})
    print(
        f"{cfg['name']} on {result['device']}, gamma={args.gamma}, U from {args.u_path or 'random'}"
    )
    hdr = f"{'T':>6} {'2DP':>9} {'Hadamard':>9} {'2DP vs Had':>10} | {'P1 2DP':>8} {'J1+J2':>8} | {'P2 2DP':>8} {'P2 Had':>8}"
    for i, T in enumerate(args.ts):
        s = build(cfg, T, U, dev)
        p1, p2 = pipes(cfg, s, args.gamma)
        st, cfgs = {}, {}
        p1["p1_2dp"]()
        p1["j1"]()
        p1["j2"]()  # fill the QK^T inputs before tuning
        torch.cuda.synchronize()
        for k, (a, kw) in p2.items():
            cfgs[k], st[k] = tune_qk(med_us, a, kw)
        for k, fn in p1.items():
            st[k] = med_us(fn)
        k2 = min(("qk_2dp_mfma", "qk_2dp_valu"), key=st.get)
        run = lambda k: (lambda: qk(*p2[k][0], **p2[k][1], cfg=cfgs[k]))
        q2, qh = run(k2), run("qk_had")
        e2e = dict(
            twodp=med_us(lambda: (p1["p1_2dp"](), q2())),
            hadamard=med_us(lambda: (p1["j1"](), p1["j2"](), qh())))
        r = dict(stage=st, e2e=e2e, qk_cfg=cfgs, qk_2dp=k2)
        if i == 0:
            r["check"] = check(cfg, s, p1, p2, cfgs, U, args.gamma)
            print("check:", {k: f"{v:.3e}" for k, v in r["check"].items()})
            print(hdr)
        print(
            f"{T:>6} {e2e['twodp']:>9.1f} {e2e['hadamard']:>9.1f} "
            f"{100 * (e2e['twodp'] / e2e['hadamard'] - 1):>+9.1f}% | {st['p1_2dp']:>8.1f} "
            f"{st['j1'] + st['j2']:>8.1f} | {st[k2]:>8.1f} {st['qk_had']:>8.1f}",
            flush=True)
        result["runs"][T] = r
        del s
        torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
