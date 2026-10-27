# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-layer error decomposition of a quantized Llama forward (``--sqnr-decomp N``).

For each decoder layer, over ``N`` chunks of the evaluation set, measure the LOCAL quantization error
of each operation given its actual (already rotated and corrupted) input, and report
mean/median/min/max across chunks plus cross-layer means:

- linears (q/k/v/o/gate/up/down_proj): SQNR of the weight, the input and the output, and the element
  gains ``g = <Q(t), t> / <t, t>`` (shrinkage ``1 - g``);
- attention (2DP module, :func:`attention_error_stats`): score / softmax / output SQNR, DC/AC split of
  the score error, sink statistics, the bias of the attention-output error and its sources, the
  measured score shrinkage ``gain_gamma``, and the gains ``g_q``, ``g_k`` of the attention quantizers.

The per-layer shrinkage chain (``[shrink]`` lines) predicts the score shrinkage from the factor gains:
``gamma_attn = 1 - g_q g_k`` (attention quantizers), ``gamma_up = 1 - g_out(q_proj) g_out(k_proj)``.

Local by design: the residual stream lives in the rotated (R1) basis, so comparing layer outputs
against a separate fp16 model would mix bases; each operation's error given its real input does not.
"""

import re
import statistics

import torch
import torch.nn.functional as F

from brevitas.nn import QuantLinear
from brevitas_examples.llm.dc_extract_sdpa import dc_modules

_LIN_ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
_LIN_MLP = ["gate_proj", "up_proj", "down_proj"]
_ATTN_KEYS = [
    "pre_softmax",
    "pre_softmax_AC",
    "pre_softmax_DC",
    "pre_softmax_ACp",
    "post_softmax",
    "post_V",
    "post_Vq",
    "out_bias_frac",
    "out_err_rms",
    "bias_frac_qk",
    "rms_qk",
    "bias_frac_v",
    "rms_v",
    "bias_frac_qklin",
    "rms_qklin",
    "bias_frac_qkjen",
    "rms_qkjen",
    "s0_rel_db",
    "s0_cv",
    "s0_logit",
    "s0_p",
    "jdp_kpos0",
    "jdp_kpos_1_7",
    "jdp_kpos_rest",
    "kin_tdc_HF",
    "kin_tdc_LF",
    "gain_gamma",
    "gain_r2",
    "be_gain",
    "be_resid",
    "be_full",
    "gain_cos",
    "be_mean",
    "be_fluc",
    "cov_fluc_cos",
    "rho_signed",
    "rho_abs",
    "xi_PR",
    "xi_offdiag",
    "xi_tr",
    "read_cos",
    "xi_k_cos",
    "cov_slope",
    "be_fpar",
    "be_r2"]


def _sqnr(x, xq):
    x = x.float()
    e = (x - xq.float()).pow(2).sum()
    return float(10 * torch.log10(x.pow(2).sum() / e.clamp_min(1e-20)))


def _dequant(t):
    return t.value if hasattr(t, "value") else t


def _gain(x, xq):
    """Regression slope of the quantized tensor on the reference: Q(t) ~ g t + noise."""
    x = x.float()
    return float((xq.float() * x).sum() / x.pow(2).sum().clamp_min(1e-20))


def attention_error_stats(
        query,
        key,
        value,
        vq,
        attn_weight,
        attn_bias,
        scale_factor,
        qin,
        q_scaled,
        kin,
        k_q,
        mf_gamma=None):
    """Error statistics of one attention call (all heads of one layer).

    Args:
        query, key, value: unquantized Q/K/V in the basis the module computes the score in
            ``[N, H, L|S, D]`` (GQA-expanded). The reference score ``q k^T`` is basis-invariant.
        vq: quantized V. attn_weight: quantized pre-softmax score (no mask). attn_bias: additive mask.
        qin, q_scaled, kin, k_q: inputs and outputs of the Q and K quantizers.
        mf_gamma: if set, add the m/f probe of :func:`~brevitas_examples.llm.mf_probe.mf_probe`.
    """
    ref = (query.float() * scale_factor) @ key.float().transpose(-2, -1)
    aw = attn_weight.float()
    bias = attn_bias.float()
    lref = torch.log_softmax(ref + bias, dim=-1)
    lq = torch.log_softmax(aw + bias, dim=-1)
    p_ref, p_q = lref.exp(), lq.exp()
    v = value.float()
    o_ref = p_ref @ v

    def _dbr(s, e):
        return float(10 * torch.log10(s / e.clamp_min(1e-20)))

    # DC/AC split of the score error per query row: common mode over the valid keys (DC, invisible to
    # softmax) vs the rest (AC); ACp weights the AC part by the clean attention (2nd-order KL driver).
    dS = aw - ref
    m = (bias > -1e30).float()
    nk = m.sum(-1, keepdim=True).clamp_min(1)
    cerr = (dS * m).sum(-1, keepdim=True) / nk
    cref = (ref * m).sum(-1, keepdim=True) / nk
    ac_sig = ((ref - cref) * m).pow(2).sum()
    ac_err = ((dS - cerr) * m).pow(2).sum()
    dc_sig = (cref.pow(2) * nk).sum()
    dc_err = (cerr.pow(2) * nk).sum()
    varE = (p_ref * (dS - (p_ref * dS).sum(-1, keepdim=True)).pow(2)).sum()
    varS = (p_ref * (ref - (p_ref * ref).sum(-1, keepdim=True)).pow(2)).sum()
    # Sink (key 0) logit: level, spread over queries, power relative to the other keys, weight.
    s0 = ref[..., 0]
    s0m = s0.mean(dim=-1)
    s0s = s0.std(dim=-1)
    s0_logit = float(s0m.mean())
    s0_cv = float((s0s / s0m.abs().clamp_min(1e-6)).mean())
    _w = m[..., 1:].expand_as(ref[..., 1:])
    _restsq = (ref[..., 1:].pow(2) * _w).sum() / _w.sum().clamp_min(1)
    s0_rel_db = float(10 * torch.log10(s0.pow(2).mean() / _restsq.clamp_min(1e-20)))
    s0_p = float(p_ref[..., 0].mean())
    # Token-constant (bias) fraction phi^2 and rms of the attention-output error, and its sources:
    # delta_o = dp V (softmax path, fp16 V) + p dV (V-quant path); dp = J dS (1st order) + remainder.
    o_q = p_q @ vq.float()
    do = o_q - o_ref

    def _bf(d):
        b = d.mean(dim=(0, 2))
        pw = d.pow(2).mean(dim=(0, 2))
        return float(b.pow(2).sum() / pw.sum().clamp_min(1e-20)), float(pw.sum().sqrt())

    out_bias_frac, out_err_rms = _bf(do)
    bf_qk, rms_qk = _bf((p_q - p_ref) @ v)
    bf_v, rms_v = _bf(p_ref @ (vq.float() - v))
    dp_lin = p_ref * dS - p_ref * (p_ref * dS).sum(-1, keepdim=True)
    bf_qklin, rms_qklin = _bf(dp_lin @ v)
    bf_qkjen, rms_qkjen = _bf((p_q - p_ref) @ v - dp_lin @ v)
    # Share of the coherent 1st-order dp at key position 0 (sink), 1-7, and 8+.
    ebar = dp_lin.mean(dim=2)
    ke = ebar.pow(2)
    ketot = ke.sum().clamp_min(1e-20)
    jdp_kpos0 = float(ke[..., :1].sum() / ketot)
    jdp_kpos_1_7 = float(ke[..., 1:8].sum() / ketot)
    jdp_kpos_rest = float(ke[..., 8:].sum() / ketot)
    # Token-DC fraction of the quantized K input, HF vs LF half.
    km = kin.float().mean(dim=-2)
    kp = kin.float().pow(2).mean(dim=-2)
    hk = kin.size(-1) // 2
    kin_tdc_HF = float(km[..., :hk].pow(2).sum() / kp[..., :hk].sum().clamp_min(1e-20))
    kin_tdc_LF = float(km[..., hk:].pow(2).sum() / kp[..., hk:].sum().clamp_min(1e-20))
    # Gain split of the AC score error, dS_ac = -gamma S_ac + resid, and the output-bias energy
    # ||beta||^2 carried by each part through J = diag(p) - p p^T.
    Sac = (ref - cref) * m
    dSac = (dS - cerr) * m
    sDS = (dSac * Sac).sum()
    gain_gamma = float(-sDS / ac_sig.clamp_min(1e-20))
    gain_r2 = float(sDS.pow(2) / (ac_sig * ac_err).clamp_min(1e-20))
    JSac = p_ref * Sac - p_ref * (p_ref * Sac).sum(-1, keepdim=True)
    dp_gain = -gain_gamma * JSac
    dp_res = dp_lin - dp_gain

    def _ben(dp):
        return float((dp @ v).mean(dim=(0, 2)).pow(2).sum())

    be_gain = _ben(dp_gain)
    be_resid = _ben(dp_res)
    be_full = _ben(dp_lin)
    _bg = (dp_gain @ v).mean(dim=(0, 2))
    _bfull = (dp_lin @ v).mean(dim=(0, 2))
    gain_cos = float((_bg * _bfull).sum() / (_bg.norm() * _bfull.norm()).clamp_min(1e-20))
    # Mean vs fluctuation split of dS over queries (per key).
    mq = m.sum(dim=-2, keepdim=True).clamp_min(1)
    dSbar = (dS * m).sum(dim=-2, keepdim=True) / mq

    def _jp(x):
        return p_ref * x - p_ref * (p_ref * x).sum(-1, keepdim=True)

    dp_mean = _jp(dSbar * m)
    dp_fluc = _jp((dS - dSbar) * m)
    be_mean = _ben(dp_mean)
    be_fluc = _ben(dp_fluc)
    _bfl = (dp_fluc @ v).mean(dim=(0, 2))
    cov_fluc_cos = float((_bfl * _bfull).sum() / (_bfl.norm() * _bfull.norm()).clamp_min(1e-20))
    # Correlation of the K quantization error with the key direction, and the geometry of its part
    # orthogonal to the key (xi): participation ratio of its covariance, coupling to Cov(q) k.
    drk = k_q.float() - kin.float()
    kd = kin.float()
    _rho = (drk * kd).sum(-1) / (drk.norm(dim=-1) * kd.norm(dim=-1)).clamp_min(1e-20)
    rho_signed = float(_rho.mean())
    rho_abs = float(_rho.abs().mean())
    xi = drk - (drk * kd).sum(
        -1, keepdim=True) / kd.pow(2).sum(
            -1, keepdim=True).clamp_min(1e-12) * kd
    _X = xi.reshape(-1, xi.size(-1))
    _Sig = (_X.transpose(0, 1) @ _X) / max(_X.size(0), 1)
    _trS = _Sig.diagonal().sum()
    _fro2 = _Sig.pow(2).sum()
    xi_PR = float(_trS.pow(2) / _fro2.clamp_min(1e-20))
    xi_offdiag = float((_fro2 - _Sig.diagonal().pow(2).sum()) / _fro2.clamp_min(1e-20))
    xi_tr = float(_trS)
    _Q = query.float()
    _Qc = (_Q - _Q.mean(dim=-2, keepdim=True)).reshape(-1, _Q.size(-1))
    _Sq = (_Qc.transpose(0, 1) @ _Qc) / max(_Qc.size(0), 1)
    _Sqk = torch.einsum('nhsd,de->nhse', kd, _Sq)
    read_cos = float(
        ((_Sqk * xi).sum(-1) / (_Sqk.norm(dim=-1) * xi.norm(dim=-1)).clamp_min(1e-20)).abs().mean())
    xi_k_cos = float(
        ((xi * kd).sum(-1) / (xi.norm(dim=-1) * kd.norm(dim=-1)).clamp_min(1e-20)).abs().mean())
    # Fluctuation split: part proportional to the clean AC score vs the orthogonal remainder.
    _F = (dS - dSbar) * m
    _Stil = (ref - cref) * m
    _slope = (p_ref * _F * _Stil).sum() / (p_ref * _Stil.pow(2)).sum().clamp_min(1e-20)
    cov_slope = float(_slope)
    be_fpar = _ben(_jp(_slope * _Stil))
    be_r2 = _ben(_jp(_F - _slope * _Stil))
    rec = {
        "pre_softmax": _sqnr(ref, aw),
        "pre_softmax_AC": _dbr(ac_sig, ac_err),
        "pre_softmax_DC": _dbr(dc_sig, dc_err),
        "pre_softmax_ACp": _dbr(varS, varE),
        "post_softmax": _sqnr(p_ref, p_q),
        "post_V": _sqnr(o_ref, p_q @ v),
        "post_Vq": _sqnr(o_ref, o_q),
        "out_bias_frac": out_bias_frac,
        "out_err_rms": out_err_rms,
        "bias_frac_qk": bf_qk,
        "rms_qk": rms_qk,
        "bias_frac_v": bf_v,
        "rms_v": rms_v,
        "bias_frac_qklin": bf_qklin,
        "rms_qklin": rms_qklin,
        "bias_frac_qkjen": bf_qkjen,
        "rms_qkjen": rms_qkjen,
        "s0_rel_db": s0_rel_db,
        "s0_cv": s0_cv,
        "s0_logit": s0_logit,
        "s0_p": s0_p,
        "jdp_kpos0": jdp_kpos0,
        "jdp_kpos_1_7": jdp_kpos_1_7,
        "jdp_kpos_rest": jdp_kpos_rest,
        "kin_tdc_HF": kin_tdc_HF,
        "kin_tdc_LF": kin_tdc_LF,
        "gain_gamma": gain_gamma,
        "g_q": _gain(qin, q_scaled),
        "g_k": _gain(kin, k_q),
        "gain_r2": gain_r2,
        "be_gain": be_gain,
        "be_resid": be_resid,
        "be_full": be_full,
        "gain_cos": gain_cos,
        "be_mean": be_mean,
        "be_fluc": be_fluc,
        "cov_fluc_cos": cov_fluc_cos,
        "rho_signed": rho_signed,
        "rho_abs": rho_abs,
        "xi_PR": xi_PR,
        "xi_offdiag": xi_offdiag,
        "xi_tr": xi_tr,
        "read_cos": read_cos,
        "xi_k_cos": xi_k_cos,
        "cov_slope": cov_slope,
        "be_fpar": be_fpar,
        "be_r2": be_r2}
    if mf_gamma is not None:
        from brevitas_examples.llm.mf_probe import mf_probe
        rec.update(mf_probe(ref, p_ref, v, m, dS, mf_gamma, lp=lref))
    return rec


def _mk_lin_hook(store, lid, name):

    def hook(module, args):
        x = args[0]
        with torch.no_grad():
            xq = _dequant(module.input_quant(x))
            W = module.weight
            Wq = _dequant(module.quant_weight())
            y_ref = F.linear(x.float(), W.float())
            y_q = F.linear(xq.float(), Wq.float())
            store.setdefault((lid, name), []).append({
                "weight": _sqnr(W, Wq),
                "act": _sqnr(x, xq),
                "out": _sqnr(y_ref, y_q),
                "g_w": _gain(W, Wq),
                "g_a": _gain(x, xq),
                "g_out": _gain(y_ref, y_q)})

    return hook


def _print_shrinkage(lin_store, attn, nl, tag):
    """Multiplicative shrinkage chain of the attention score, per layer and cross-layer.

    upstream (projections): S scales by g_out(q_proj) g_out(k_proj) -> gamma_up   = 1 - g_qo g_ko
    attention quantizers  : S scales by g_q g_k                     -> gamma_attn = 1 - g_q g_k
    total                 : gamma_tot = 1 - g_qo g_ko g_q g_k (the exact 2DP plane only sees gamma_up).
    gain_gamma, measured on the score, is printed next to gamma_attn as a consistency check."""

    def m(recs, k):
        v = [r[k] for r in recs if k in r]
        return sum(v) / len(v) if v else None

    rows = []
    print(
        f"[shrink] {tag} L | g_a(q) g_w(q) g_out(q) | g_a(k) g_w(k) g_out(k) | g_q g_k | "
        f"gamma_up gamma_attn(pred) gain_gamma(meas) gamma_tot",
        flush=True)
    for i in range(nl):
        q, k, a = lin_store.get((i, "q_proj"), []), lin_store.get((i, "k_proj"), []), attn.get(i, [])
        vals = [
            m(q, "g_a"),
            m(q, "g_w"),
            m(q, "g_out"),
            m(k, "g_a"),
            m(k, "g_w"),
            m(k, "g_out"),
            m(a, "g_q"),
            m(a, "g_k"),
            m(a, "gain_gamma")]
        if any(v is None for v in vals):
            continue
        gaq, gwq, goq, gak, gwk, gok, gq, gk, gg = vals
        up, att = 1 - goq * gok, 1 - gq * gk
        tot = 1 - goq * gok * gq * gk
        rows.append((gaq, gwq, goq, gak, gwk, gok, gq, gk, up, att, gg, tot))
        print(
            f"[shrink] L{i:02d} | {gaq:.4f} {gwq:.4f} {goq:.4f} | {gak:.4f} {gwk:.4f} {gok:.4f} | "
            f"{gq:.4f} {gk:.4f} | {up:.4f} {att:.4f} {gg:.4f} {tot:.4f}",
            flush=True)
    if rows:
        c = [sum(r[j] for r in rows) / len(rows) for j in range(12)]
        print(
            f"[shrink-xl] {tag} g_a(q)={c[0]:.4f} g_w(q)={c[1]:.4f} g_out(q)={c[2]:.4f} "
            f"g_a(k)={c[3]:.4f} g_w(k)={c[4]:.4f} g_out(k)={c[5]:.4f} g_q={c[6]:.4f} g_k={c[7]:.4f}",
            flush=True)
        print(
            f"[shrink-xl] {tag} gamma_up={c[8]:.4f} gamma_attn(pred)={c[9]:.4f} "
            f"gain_gamma(meas)={c[10]:.4f} gamma_tot={c[11]:.4f}",
            flush=True)


def _stats(vals):
    if not vals:
        return None
    return (sum(vals) / len(vals), statistics.median(vals), min(vals), max(vals))


def run_sqnr_decomp(model, loader, n_chunks=16, ctx_len=None, device="cuda", tag="", mf_gamma=None):
    """Run ``n_chunks`` batches of ``loader`` through ``model`` and print the decomposition.

    ``mf_gamma`` enables the m/f probe (:mod:`brevitas_examples.llm.mf_probe`) at that gamma."""
    layers = model.model.layers
    lin_store = {}
    handles = []
    proj_names = _LIN_ATTN + _LIN_MLP
    # Find the QuantLinears by type (down_proj may be wrapped by a rotation module), and parse the
    # layer index and projection name from the module path.
    for name, mod in model.named_modules():
        if isinstance(mod, QuantLinear):
            m = re.search(r"layers\.(\d+)\.", name)
            proj = next((p for p in proj_names if p in name), None)
            if m and proj is not None:
                handles.append(
                    mod.register_forward_pre_hook(_mk_lin_hook(lin_store, int(m.group(1)), proj)))
    attn_mods = dc_modules(model)
    for _, mod in attn_mods:
        mod.decomp_records = []
        mod.mf_gamma = mf_gamma

    print(f"[sqnr-decomp] {tag} running {n_chunks} chunks ...", flush=True)
    with torch.no_grad():
        for _, batch in zip(range(n_chunks), loader):
            ids = batch["input_ids"].to(device)
            if ctx_len:
                ids = ids[:, :ctx_len]
            model(input_ids=ids)
    for h in handles:
        h.remove()

    attn = {}
    for lid, mod in attn_mods:
        attn.setdefault(lid, []).extend(mod.decomp_records)
        mod.decomp_records = None
        mod.mf_gamma = None
    nl = len(layers)

    print(
        f"[sqnr-decomp] {tag} per-layer SQNR (dB) — fmt mean/med/min/max over {n_chunks} chunks",
        flush=True)
    lin_pts = _LIN_ATTN + _LIN_MLP
    for i in range(nl):
        for nm in lin_pts:
            recs = lin_store.get((i, nm), [])
            for q in ("weight", "act", "out"):
                st = _stats([r[q] for r in recs])
                if st:
                    print(
                        f"[sqnr] L{i:02d} {nm:9s} {q:4s} "
                        f"{st[0]:6.2f}/{st[1]:6.2f}/{st[2]:6.2f}/{st[3]:6.2f}",
                        flush=True)
        arecs = attn.get(i, [])
        for q in _ATTN_KEYS:
            st = _stats([r[q] for r in arecs])
            if st:
                print(
                    f"[sqnr] L{i:02d} attn      {q:12s} "
                    f"{st[0]:6.2f}/{st[1]:6.2f}/{st[2]:6.2f}/{st[3]:6.2f}",
                    flush=True)

    # Cross-layer summary: mean over layers of the per-layer means.
    print(f"[sqnr-decomp] {tag} CROSS-LAYER mean of per-layer-mean SQNR (dB):", flush=True)

    def _xl(pts, get):
        acc = {}
        for i in range(nl):
            for nm, q in pts:
                vals = get(i, nm, q)
                if vals:
                    acc.setdefault((nm, q), []).append(sum(vals) / len(vals))
        return acc

    xl_lin = _xl([(nm, q) for nm in lin_pts for q in ("weight", "act", "out")],
                 lambda i,
                 nm,
                 q: [r[q] for r in lin_store.get((i, nm), [])])
    for (nm, q), ms in xl_lin.items():
        print(f"[sqnr-xl] {nm:9s} {q:4s} {sum(ms)/len(ms):6.2f}", flush=True)
    xl_attn = _xl(
        [("attn", q) for q in _ATTN_KEYS], lambda i, nm, q: [r[q] for r in attn.get(i, [])])
    for (nm, q), ms in xl_attn.items():
        print(f"[sqnr-xl] {nm:9s} {q:12s} {sum(ms)/len(ms):6.2f}", flush=True)
    _print_shrinkage(lin_store, attn, nl, tag)
    _print_mf(attn, nl, tag)
    print(f"[sqnr-decomp] {tag} done.", flush=True)


_MF_KEYS = [
    "mf_m2",
    "mf_f2",
    "mf_mf2",
    "mf_cos",
    "mf_msr2",
    "mf_mwr2",
    "mf_fsr2",
    "mf_fwr2",
    "mf_srcos",
    "mf_wrcos",
    "mf_dp0_dc",
    "mf_dp0_ac",
    "mf_idrel_dc",
    "mf_idrel_ac",
    "mf_Ddc",
    "mf_Dac",
    "mf_aDdc",
    "mf_aDac",
    "mf_fracDac_pos",
    "mf_fracDdc_neg",
    "mf_H",
    "mf_logodds",
    "mf_fracHgt",
    "mf_frac_p0zero",
    "mf_p0",
    "mf_id3rel",
    "mf_uproj_dc",
    "mf_uproj_ac",
    "mf_ucos_dc",
    "mf_ucos_ac",
    "mf_usplit_rel_dc",
    "mf_usplit_rel_ac",
    "mfq_be2",
    "mfq_sr2",
    "mfq_wr2",
    "mfq_dp0"]


def _print_mf(attn, nl, tag):
    """m/f probe summary: per-layer means over chunks and cross-layer means (5 significant digits).

    Aggregate cosines are energy-weighted over chunks and layers: sum(dot) / sqrt(sum(e1) sum(e2))."""
    if not any("mf_m2" in r for i in range(nl) for r in attn.get(i, [])):
        return

    def m(recs, k):
        v = [r[k] for r in recs if k in r]
        return sum(v) / len(v) if v else None

    print(f"[mf] {tag} per-layer means over chunks (keys: {' '.join(_MF_KEYS)})", flush=True)
    xl = {k: [] for k in _MF_KEYS}
    agg_keys = {
        "m2": "mf_m2",
        "f2": "mf_f2",
        "dot": "mf_dot",
        "msr2": "mf_msr2",
        "fsr2": "mf_fsr2",
        "srdot": "mf_srdot",
        "mwr2": "mf_mwr2",
        "fwr2": "mf_fwr2",
        "wrdot": "mf_wrdot",
        "qsr2": "mfq_sr2",
        "qwr2": "mfq_wr2",
        "qsrwr": "mfq_srwrdot",
        "upar_dc": "mf_upar_dc",
        "upar_ac": "mf_upar_ac",
        "msrpar": "mf_msrpar",
        "fsrpar": "mf_fsrpar",
        "qbe2": "mfq_be2",
        "qsrpar": "mfq_srpar"}
    agg = {k: 0.0 for k in agg_keys}
    for i in range(nl):
        recs = [r for r in attn.get(i, []) if "mf_m2" in r]
        if not recs:
            continue
        vals = {k: m(recs, k) for k in _MF_KEYS}
        for k, v in vals.items():
            if v is not None:
                xl[k].append(v)
        for r in recs:
            for k, rk in agg_keys.items():
                agg[k] += r[rk]
        print(
            f"[mf] L{i:02d} " +
            " ".join(f"{k}={vals[k]:.5g}" for k in _MF_KEYS if vals[k] is not None),
            flush=True)
    print(f"[mf-xl] {tag} cross-layer mean of per-layer means:", flush=True)
    for k in _MF_KEYS:
        if xl[k]:
            print(f"[mf-xl] {k:16s} {sum(xl[k]) / len(xl[k]):.5g}", flush=True)

    def c(d, a, b):
        return d / max((a * b) ** 0.5, 1e-30)

    print(
        f"[mf-agg] {tag} energy-weighted cos(m,f)={c(agg['dot'], agg['m2'], agg['f2']):.4f} "
        f"cos(m_sr,f_sr)={c(agg['srdot'], agg['msr2'], agg['fsr2']):.4f} "
        f"cos(m_wr,f_wr)={c(agg['wrdot'], agg['mwr2'], agg['fwr2']):.4f} "
        f"cos(q_sr,q_wr)={c(agg['qsrwr'], agg['qsr2'], agg['qwr2']):.4f}",
        flush=True)
    # Projection shares (sr . total) / ||total||^2; the sr and wr shares add up to 1.
    print(
        f"[mf-agg] {tag} sink-vs-rest projection share: m={agg['msrpar'] / max(agg['m2'], 1e-30):.4f} "
        f"f={agg['fsrpar'] / max(agg['f2'], 1e-30):.4f} "
        f"quantized dS={agg['qsrpar'] / max(agg['qbe2'], 1e-30):.4f} | "
        f"u-projection m_sr={agg['upar_dc'] / max(agg['msr2'], 1e-30):.4f} "
        f"f_sr={agg['upar_ac'] / max(agg['fsr2'], 1e-30):.4f}",
        flush=True)
