# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""m/f probe: how a score shrinkage reaches the attention output through the sink key.

A shrinkage of the score splits into a shrinkage of its per-key mean over queries (DC, m) and of the
rest (AC, f). This module measures both first-order output errors, their split into a sink-vs-rest
and a within-rest term, and the same split for the actual quantized score error.

Pure tensor functions, no brevitas dependency, so they can be unit-tested on CPU.

Points referenced below: (1) the first-order output error of a score error splits exactly into a
sink-vs-rest and a within-rest term; (2) the signs of the sink contrasts D_dc and D_ac; (3) the identity
D_dc + D_ac = -(H(pi) + ln(p0 / (1 - p0))) relating them to the entropy of the non-sink attention.

Notation (one head; tensors carry leading [N, H] dims):
  ref  [N,H,L,S]  clean pre-softmax score S (no mask added)
  p    [N,H,L,S]  clean causal attention softmax(S + bias); 0 on masked keys
  v    [N,H,S,D]  values (fp16 V, GQA-expanded)
  mask [.,.,L,S]  1 on valid (causal) keys, 0 elsewhere (broadcastable)

  DC  S̄_j    = mean of S_{ij} over the valid queries i >= j   (same convention as be_mean / dSbar)
  AC  S̃_ij   = S_ij − S̄_j                                     (zero valid-query mean per key)
  First-order output error of a score error x:  Cov_{p_i}(x, v) = (J x) v,  J x = p∘x − p (p·x).

Part A  synthetic shrinks on the CLEAN model (no quantization): x_dc = −γ S̄, x_ac = −γ S̃.
        m = ⟨(J x_dc) v⟩, f = ⟨(J x_ac) v⟩ (token means), their split into sink-vs-rest and within-rest
        (Point 1), the signed token-mean mass shift at the sink, D_dc / D_ac (Points 2-3), the Point-3
        identity check, and the share of m_sr / f_sr carried by the common direction ū.
Part B  the same sink-vs-rest split and signed sink shift for the ACTUAL quantized score error dS.
Part C  sink_balance(): oracle intervention that cancels the sink-vs-rest term of dS exactly.
"""

import torch


def _jp(p, x):
    """First-order attention change J·x = p∘x − p·(p·x), row-wise over keys."""
    return p * x - p * (p * x).sum(-1, keepdim=True)


def _tokmean(t):
    """Token mean over batch and queries: [N,H,L,...] -> [H,...] (matches _ben in dc_extract_sdpa)."""
    return t.mean(dim=(0, 2))


def _e(a):
    return float(a.pow(2).sum())


def _dot(a, b):
    return float((a * b).sum())


def _cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


def sink_rest_split(p, x, v):
    """Point 1, per query row. Returns (cov, sr, wr), each [N,H,L,D], with
         cov = Cov_{p_i}(x, v)                                      (= (J x) v)
         sr  = p0 (1−p0) (x_rest − x0) (v_rest − v0)               sink-vs-rest term
         wr  = (1−p0) Cov_{π_i}(x, v)                               within-rest term
       π_ij = p_ij / (1−p0) over keys j >= 1. Rows with p0 ≈ 1 (e.g. query 0) have sr = wr = 0.
       Also returns the per-row pieces (p0, pi, x_rest, v_rest) for reuse."""
    p0 = p[..., 0]  # [N,H,L]
    one_m = (1.0 - p0)
    ok = one_m > 1e-6
    pi = p[..., 1:] / one_m.clamp_min(1e-6).unsqueeze(-1)  # [N,H,L,S-1]
    pi = torch.where(ok.unsqueeze(-1), pi, torch.zeros_like(pi))
    xr = x[..., 1:]
    x_rest = (pi * xr).sum(-1)  # [N,H,L]
    v_r = v[..., 1:, :]  # [N,H,S-1,D]
    v_rest = pi @ v_r  # [N,H,L,D]
    v0 = v[..., 0:1, :]  # [N,H,1,D]
    x0 = x[..., 0]  # [N,H,L]
    cov = _jp(p, x) @ v
    sr = (p0 * one_m * (x_rest - x0)).unsqueeze(-1) * (v_rest - v0)
    e_pxv = (pi * xr) @ v_r  # Σ_j π_j x_j v_j
    wr = one_m.unsqueeze(-1) * (e_pxv - x_rest.unsqueeze(-1) * v_rest)
    wr = torch.where(ok.unsqueeze(-1), wr, torch.zeros_like(wr))
    return cov, sr, wr, (p0, pi, x_rest, v_rest, ok)


def mf_probe(ref, p, v, mask, dS, gamma, lp=None):
    """Parts A and B. Returns a flat dict of floats for one chunk (all heads of one layer).
    Energies (keys ending in 2) and dots are summed over heads and head_dim, like be_* in _decomp,
    so cross-chunk/layer aggregate cosines are dot / sqrt(e1·e2).
    lp = log p (log_softmax; −inf on masked keys). Used for the Point-3 log-odds and entropy so that
    rows with a vanishing sink weight are handled in log space, not through a clamped p0."""
    out = {}
    if lp is None:
        lp = torch.log(p)
    m = mask.to(ref.dtype)
    mq = m.sum(dim=-2, keepdim=True).clamp_min(1)  # valid queries per key
    Sbar = (ref * m).sum(dim=-2, keepdim=True) / mq  # [N,H,1,S] DC
    Stil = (ref - Sbar) * m  # [N,H,L,S] AC (0 on masked)
    Sbar_m = Sbar * m

    # ---- Part A: synthetic shrinks of the clean score
    vecs = {}
    for name, x in (("dc", -gamma * Sbar_m), ("ac", -gamma * Stil)):
        cov, sr, wr, (p0, pi, x_rest, v_rest, ok) = sink_rest_split(p, x, v)
        vecs[name] = (_tokmean(cov), _tokmean(sr), _tokmean(wr))
        dp0 = _jp(p, x)[..., 0]  # first-order Δp at the sink [N,H,L]
        out[f"mf_dp0_{name}"] = float(dp0.mean())  # signed token-mean mass shift at the sink
        out[f"mf_idrel_{name}"] = float((cov - sr - wr).norm() / cov.norm().clamp_min(1e-30))
    (mv, msr, mwr), (fv, fsr, fwr) = vecs["dc"], vecs["ac"]
    out.update({
        "mf_m2": _e(mv),
        "mf_f2": _e(fv),
        "mf_mf2": _e(mv + fv),
        "mf_dot": _dot(mv, fv),
        "mf_msr2": _e(msr),
        "mf_mwr2": _e(mwr),
        "mf_fsr2": _e(fsr),
        "mf_fwr2": _e(fwr),
        "mf_srdot": _dot(msr, fsr),
        "mf_wrdot": _dot(mwr, fwr),
        "mf_msrpar": _dot(msr, mv),
        "mf_fsrpar": _dot(fsr, fv),  # projection of the sr part on the total
        "mf_cos": _cos(mv, fv),
        "mf_srcos": _cos(msr, fsr),
        "mf_wrcos": _cos(mwr, fwr),})

    # ---- Points 2-3: D_dc, D_ac, entropy identity (clean model; rows with p0 < 1 only)
    _, _, _, (p0, pi, _, v_rest, ok) = sink_rest_split(p, Sbar_m, v)
    okf = ok.to(ref.dtype)
    nrow = okf.sum().clamp_min(1)
    D_dc = (pi * Sbar_m[..., 1:]).sum(-1) - Sbar_m[..., 0]  # [N,H,L]
    D_ac = (pi * Stil[..., 1:]).sum(-1) - Stil[..., 0]
    # log space: ln p0, ln(1−p0) = logsumexp over the rest, ln π_j = ln p_j − ln(1−p0)
    lp0 = lp[..., 0]
    l1m = torch.logsumexp(lp[..., 1:], dim=-1)
    v3 = ok & torch.isfinite(lp0) & torch.isfinite(l1m)  # rows where Point 3 is evaluable
    v3f = v3.to(ref.dtype)
    n3 = v3f.sum().clamp_min(1)
    lpi = lp[..., 1:] - l1m.unsqueeze(-1)
    ent = -(torch.where(pi > 0, pi * lpi, torch.zeros_like(pi))).sum(-1)
    logodds = torch.where(v3, lp0 - l1m, torch.zeros_like(lp0))
    rhs = -(ent + logodds)
    lhs = D_dc + D_ac
    alpha = p0 * (1 - p0)
    out.update({
        "mf_Ddc":
            float((D_dc * okf).sum() / nrow),
        "mf_Dac":
            float((D_ac * okf).sum() / nrow),
        "mf_aDdc":
            float((alpha * D_dc * okf).sum() / nrow),
        "mf_aDac":
            float((alpha * D_ac * okf).sum() / nrow),
        "mf_fracDac_pos":
            float(((D_ac > 0).to(ref.dtype) * okf).sum() / nrow),
        "mf_fracDdc_neg":
            float(((D_dc < 0).to(ref.dtype) * okf).sum() / nrow),
        "mf_H":
            float((ent * v3f).sum() / n3),
        "mf_logodds":
            float((logodds * v3f).sum() / n3),
        "mf_fracHgt":
            float(((ent + logodds > 0).to(ref.dtype) * v3f).sum() / n3
                 ),  # Point-3 condition H > ln((1−p0)/p0)
        "mf_frac_p0zero":
            float(((ok & ~v3).to(ref.dtype)).sum() / nrow),  # rows excluded: p0 underflowed
        "mf_p0":
            float(p0.mean()),
        "mf_id3rel":
            float(((lhs - rhs).abs() * v3f).sum() / ((rhs.abs() * v3f).sum()).clamp_min(1e-30)),})
    # How much of m_sr / f_sr lies along the common direction ū (u_i = v_rest,i − v0; ū = token mean).
    # Exact split: tokmean(sr) = part_u + part_d, part_u = ⟨w⟩ū, part_d = ⟨w δ⟩ (not orthogonal in general),
    # so report the projection share (part_u·sr)/‖sr‖² = (eu + cross)/‖sr‖² and cos(part_u, sr).
    u = v_rest - v[..., 0:1, :]  # [N,H,L,D]
    ubar = _tokmean(u * okf.unsqueeze(-1)) / _tokmean(okf).clamp_min(1e-30).unsqueeze(-1)  # [H,D]
    for name, D, srv in (("dc", D_dc, msr), ("ac", D_ac, fsr)):
        w = (-gamma) * alpha * D * okf  # per-row scalar coefficient of u_i
        part_u = _tokmean(w).unsqueeze(-1) * ubar  # ⟨w⟩ ū
        part_d = _tokmean(w.unsqueeze(-1) * (u - ubar.unsqueeze(0).unsqueeze(2)))  # ⟨w δ⟩
        out[f"mf_ue_{name}"] = _e(part_u)
        out[f"mf_upar_{name}"] = _dot(part_u, srv)  # = eu + ⟨part_u, part_d⟩
        out[f"mf_uproj_{name}"] = _dot(part_u, srv) / max(_e(srv), 1e-30)
        out[f"mf_ucos_{name}"] = _cos(part_u, srv)
        out[f"mf_usplit_rel_{name}"] = float(
            (part_u + part_d - srv).norm() / srv.norm().clamp_min(1e-30))

    # ---- Part B: the actual quantized score error dS
    x = dS * m
    cov, sr, wr, _ = sink_rest_split(p, x, v)
    b, bsr, bwr = _tokmean(cov), _tokmean(sr), _tokmean(wr)
    out.update({
        "mfq_be2": _e(b),
        "mfq_sr2": _e(bsr),
        "mfq_wr2": _e(bwr),
        "mfq_srwrdot": _dot(bsr, bwr),
        "mfq_srpar": _dot(bsr, b),  # projection of the sr part on the total
        "mfq_dp0": float(_jp(p, x)[..., 0].mean()),  # signed token-mean first-order Δp at the sink
    })
    return out


def sink_balance(attn_weight, ref, p):
    """Part C (oracle, diagnostic only). Shift each query's sink score so that the sink's score error
    equals the π-average score error over the rest: x0 <- x_rest. This zeroes the sink-vs-rest
    coefficient (x_rest − x0) of Point 1 exactly (first order, clean π), leaving every other key's
    score, hence the within-rest term, unchanged. Uses the clean score `ref` and clean attention `p`.
    attn_weight, ref, p: [N,H,L,S]. Returns the modified attn_weight (same dtype)."""
    ct = torch.promote_types(attn_weight.dtype, torch.float32)  # at least fp32; keeps fp64 in tests
    dS = attn_weight.to(ct) - ref.to(ct)
    p = p.to(ct)
    p0 = p[..., 0]
    one_m = 1.0 - p0
    ok = one_m > 1e-6
    pi = p[..., 1:] / one_m.clamp_min(1e-6).unsqueeze(-1)
    x_rest = (pi * dS[..., 1:]).sum(-1)
    shift = torch.where(ok, x_rest - dS[..., 0], torch.zeros_like(x_rest))
    out = attn_weight.clone()
    out[..., 0] = (attn_weight[..., 0].to(ct) + shift).to(attn_weight.dtype)
    return out
