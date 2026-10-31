# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Inputs of the full flash-attention benchmark, produced by the front ends of ../ (p1_2dp, had_j2).

Setup(model, T) runs, once, both front ends on random fp16 activations / weights:
  ah                 Hadamard MXFP4 Q / K codes + E8M0 scales (J1 + J2), as (qp, qs, None, kp, ks, None)
  a2p[pw]            2DP MXFP4 Q / K codes + scales and the rank-2 coefficients a' (Q) / b (K), written by the
                     UNCHANGED P1 through its strides into zero-padded slots of pw halves per token (pw = 8 for the
                     fp16 32x32x8 rank-2 MFMA, 16 for 32x32x16; the first two halves are the coefficients)
  s["C"]             the fp16 projection buffer [T, n_q + 2 n_kv, D]: V for every method
  fe_for(meth, cm)   the front end of each method (end-to-end timing)
"""
import os
import sys

# isort: off
import triton  # noqa: F401  -- import triton BEFORE torch
# isort: on
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # ../ : the front-end kernels
from bench_2dp_vs_hadamard import kcontig  # noqa: E402
from bench_2dp_vs_hadamard import load_u  # noqa: E402
from bench_2dp_vs_hadamard import MODELS  # noqa: E402
from bench_2dp_vs_hadamard import pipes  # noqa: E402
from layout_2dp import permute_fused_qkv_split_rope_hd  # noqa: E402
from layout_2dp import rope_tables_hd  # noqa: E402
from layout_2dp import ublk_constants  # noqa: E402
from p1_2dp import front_end_2dp  # noqa: E402

GAMMA = 0.047


def build(cfg, T, U, dev="cuda"):
    """bench_2dp_vs_hadamard.build without the [n_q, T, T] logit buffer."""
    D, n_q, n_kv, dm = cfg["hd"], cfg["n_q"], cfg["n_kv"], cfg["d_model"]
    n_qk, h_tot = n_q + n_kv, n_q + 2 * n_kv
    torch.manual_seed(0)
    x = torch.randn(T, dm, device=dev, dtype=torch.float16)
    W = torch.randn(dm, h_tot * D, device=dev, dtype=torch.float16) * dm**-0.5
    inv = 500000.0**(-torch.arange(D // 2, device=dev).float() / (D // 2))
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
        out=torch.empty(1, device=dev, dtype=torch.float32))
    for k in ("2dp", "had"):
        s[f"pk_{k}"] = torch.empty(T, n_qk, D // 2, device=dev, dtype=torch.uint8)
        s[f"sc_{k}"] = torch.empty(T, n_qk, D // 32, device=dev, dtype=torch.uint8)
    s["cf"] = torch.empty(n_qk, T, 2, device=dev, dtype=torch.float16).permute(1, 0, 2)
    return s


class Setup:

    def __init__(self, model, T, u_path=None, u_layer=15, gamma=GAMMA):
        cfg = MODELS[model]
        self.cfg, self.T, self.gamma = cfg, T, gamma
        self.D, n_q, n_kv = cfg["hd"], cfg["n_q"], cfg["n_kv"]
        self.n_q, self.gpg = n_q, n_q // n_kv
        U = load_u(cfg, u_path, u_layer, "cuda")   # None: random orthonormal plane (latency does not depend on it)
        self.s = s = build(cfg, T, U)
        p1, p2 = pipes(cfg, s, gamma)
        self.p1 = p1
        p1["p1_2dp"]()
        p1["j1"]()
        p1["j2"]()
        torch.cuda.synchronize()
        self.ah = p2["qk_had"][0][:6]
        self.fe = dict(twodp=p1["p1_2dp"], hadamard=lambda: (p1["j1"](), p1["j2"]()))
        n_qk = n_q + n_kv
        self.fe2, self.a2p, self.cfp = {2: p1["p1_2dp"]}, {2: p2["qk_2dp_mfma"][0][:6]}, {}
        for pw in (8, 16):
            cfp = torch.zeros(n_qk, T, pw, device="cuda", dtype=torch.float16)
            view = cfp[:, :, :2].permute(1, 0, 2)
            self.cfp[pw] = cfp
            self.fe2[pw] = (
                lambda view=view: front_end_2dp(
                    s["x"],
                    s["W2d"],
                    s["C"],
                    1,
                    self.D,
                    n_q,
                    n_kv,
                    s["cos"],
                    s["sin"],
                    s["pk_2dp"],
                    s["sc_2dp"],
                    view,
                    s["UT"],
                    qfac=1.0 - gamma))
            pk, sc = s["pk_2dp"], s["sc_2dp"]
            self.a2p[pw] = (pk[:, :n_q], sc[:, :n_q], view[:, :n_q], pk[:, n_q:], sc[:, n_q:], view[:, n_q:])
            self.fe2[pw]()
        torch.cuda.synchronize()
        assert torch.equal(self.cfp[8][:, :, :2].permute(1, 0, 2), s["cf"]), "padded P1 coefficients differ"
        assert torch.equal(self.cfp[16][:, :, :2].permute(1, 0, 2), s["cf"])
        assert not self.cfp[8][:, :, 2:].any() and not self.cfp[16][:, :, 2:].any()

    @staticmethod
    def pw(cm):
        return {6: 8, 7: 16, 8: 8}.get(cm, 2)

    def fe_for(self, meth, cm=0):
        """Front end of a method: 'had' -> J1 + J2; '2dp' -> P1 writing the coefficient slot of the rank-2 variant cm
        (6: 8-half slot, 7: 16-half slot)."""
        if meth == "had":
            return self.fe["hadamard"]
        return self.fe2[self.pw(cm)]
