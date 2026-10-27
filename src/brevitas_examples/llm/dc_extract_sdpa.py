# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""2DP: exact rank-2 DC plane + low-precision residual for the QK^T GEMM.

Drop-in replacement for :class:`~brevitas.nn.QuantScaledDotProductAttention` used by the LLM example
when ``--dc-extract-qk`` is set. Instead of rotating the Q/K head with a Hadamard matrix (QuaRot's R3),
the post-RoPE head is permuted into high-/low-frequency halves ``[HF | LF]`` and, per KV group, a
shared orthonormal plane ``U`` is placed on the LF half, spanned by the query mean and the key mean.
The in-plane coordinates ``a = q U`` and ``b = k U`` are carried exactly; only the residuals
``r = q - a U^T`` and ``s = k - b U^T`` go through the Q/K quantizers. The score is computed as two
separate GEMMs::

    S = Q(r) Q(s)^T / sqrt(d)  +  a b^T / sqrt(d)

which equals ``q k^T / sqrt(d)`` whenever the quantizers are disabled, for any orthonormal ``U``.

``dc_sub_dim`` selects the variant: ``2`` is the 2DP method, ``< 0`` applies a full head-dim
Hadamard to Q and K before quantization (the R3 baseline, optionally randomized), ``0`` quantizes
Q and K as they are.

Optional score corrections (``dc_correction``) compensate the shrinkage of the quantized score,
``Q(x) ~ (1 - gamma) x``:

- ``unshrink``: divide the quantized part of the score by ``1 - gamma``.
- ``mean``: subtract ``gamma * a_bar b^T / sqrt(d)``, with ``a_bar`` the mean in-plane query
  coordinate over the sequence (``dc_abar_mode='full'``) or over positions ``<= i``
  (``dc_abar_mode='causal'``).
- ``scale``: subtract ``gamma * a b^T / sqrt(d)`` (scale the exact plane by ``1 - gamma``).

The plane ``U`` is either estimated per sequence (default) or frozen: a calibration pass
accumulates per-group statistics (:meth:`DCExtractQuantScaledDotProductAttention.start_u_calibration`),
:func:`build_frozen_u` turns them into one plane per layer and KV group, and
:func:`set_frozen_u` installs it.
"""

import math
import re
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
from torch import nn
from torch import Tensor

from brevitas.nn.quant_sdpa import QuantScaledDotProductAttention

_HAD_CACHE = {}
_SIGN_CACHE = {}
_TCAST_DTYPES = {}

DC_CORRECTIONS = ('none', 'unshrink', 'mean', 'scale')
ABAR_MODES = ('full', 'causal')


def hadamard_matrix(dim: int, device, dtype) -> Tensor:
    """Normalized Sylvester Hadamard matrix ``[dim, dim]`` (``dim`` a power of 2), cached."""
    key = (dim, str(device), str(dtype))
    if key not in _HAD_CACHE:
        H = torch.ones(1, 1)
        while H.shape[0] < dim:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _HAD_CACHE[key] = (H / dim ** 0.5).to(device=device, dtype=dtype)
    return _HAD_CACHE[key]


def random_signs(dim: int, device, dtype, seed: int = 0) -> Tensor:
    """Seeded random +-1 vector for the randomized Hadamard ``diag(signs) H``, cached.

    Q and K use the same signs, so the score is unchanged before quantization."""
    key = (dim, str(device), str(dtype), seed)
    if key not in _SIGN_CACHE:
        g = torch.Generator().manual_seed(seed)
        s = torch.randint(0, 2, (dim,), generator=g, dtype=torch.float32) * 2 - 1
        _SIGN_CACHE[key] = s.to(device=device, dtype=dtype)
    return _SIGN_CACHE[key]


def hflf_perm(dim: int) -> Tensor:
    """Channel permutation that groups the RoPE bands of a rotate-half head into ``[HF | LF]``.

    Band ``j`` holds channels ``{j, j + dim/2}``; HF = the faster half of the bands, LF = the slower
    half (where the token-mean of Q and K concentrates)."""
    nb = dim // 2
    h = nb // 2
    hf = list(range(0, h)) + list(range(nb, nb + h))
    lf = list(range(h, nb)) + list(range(nb + h, 2 * nb))
    return torch.tensor(hf + lf, dtype=torch.long)


def mean_subspace(qg: Tensor, kg: Tensor) -> Tensor:
    """Per-sequence plane ``U [N, half, 2] = QR([q_mean, k_mean])``.

    ``qg [N, group, L, half]`` are the group's query heads, ``kg [N, S, half]`` its KV head."""

    def nrm(v):
        return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    kdir = nrm(kg.mean(dim=-2))
    qmean = nrm(qg.mean(dim=(-3, -2)))
    basis = torch.stack([qmean, kdir], dim=-1)
    return torch.linalg.qr(basis)[0]


class DCExtractQuantScaledDotProductAttention(QuantScaledDotProductAttention):
    """QuantScaledDotProductAttention with 2DP (exact DC plane + quantized residual) on Q/K.

    Args (on top of :class:`~brevitas.nn.QuantScaledDotProductAttention`):
        num_key_value_heads (int, optional): number of KV heads, to recover the GQA groups.
            Defaults to the number of query heads.
        dc_sub_dim (int): rank of the exact plane (2 = 2DP). ``< 0``: full head-dim Hadamard on
            Q/K before quantization. ``0``: quantize Q/K unchanged.
        dc_correction (str): shrinkage correction, one of ``none``, ``unshrink``, ``mean``,
            ``scale`` (see module docstring). ``scale`` requires ``dc_sub_dim > 0``.
        dc_gamma (float): shrinkage ``gamma`` used by the correction.
        dc_abar_mode (str): ``full`` or ``causal`` mean query coordinate for ``mean``.
        dc_sinkbal (bool): oracle diagnostic; uses the unquantized score to set the sink-key score
            error equal to the attention-weighted error over the other keys.
        dc_had_rand (bool): with ``dc_sub_dim < 0``, randomize the Hadamard with seeded sign flips.
        dc_had_rand_shared (bool): use one sign pattern for all layers instead of one per layer.
        dc_had_seed (int): base seed of the sign flips.
        tcast_scale (str, optional): replace the attention quantizers' output with a
            `tcast <https://github.com/ROCm/tensorcast>`_ virtual cast to FP4 (e2m1) elements with this
            block-scale spec (e.g. ``e4m3fn_t32``). The brevitas quantizers still run, so that they
            stay calibrated. Requires ``tcast`` to be importable.
    """

    def __init__(
            self,
            num_key_value_heads: Optional[int] = None,
            dc_sub_dim: int = 2,
            dc_correction: str = 'none',
            dc_gamma: float = 0.0,
            dc_abar_mode: str = 'full',
            dc_sinkbal: bool = False,
            dc_had_rand: bool = False,
            dc_had_rand_shared: bool = False,
            dc_had_seed: int = 0,
            tcast_scale: Optional[str] = None,
            **kwargs):
        super().__init__(**kwargs)
        if dc_correction not in DC_CORRECTIONS:
            raise ValueError(f"dc_correction must be one of {DC_CORRECTIONS}, got {dc_correction}")
        if dc_abar_mode not in ABAR_MODES:
            raise ValueError(f"dc_abar_mode must be one of {ABAR_MODES}, got {dc_abar_mode}")
        if dc_correction == 'scale' and dc_sub_dim <= 0:
            raise ValueError("dc_correction='scale' needs an exact plane (dc_sub_dim > 0)")
        if dc_correction != 'none' and not 0.0 <= dc_gamma < 1.0:
            raise ValueError(f"dc_gamma must be in [0, 1), got {dc_gamma}")
        self.num_key_value_heads = num_key_value_heads
        self.dc_sub_dim = dc_sub_dim
        self.dc_correction = dc_correction
        self.dc_gamma = dc_gamma
        self.dc_abar_mode = dc_abar_mode
        self.dc_sinkbal = dc_sinkbal
        self.dc_had_rand = dc_had_rand
        self.dc_had_rand_shared = dc_had_rand_shared
        self.dc_had_seed = dc_had_seed
        self.tcast_scale = tcast_scale
        # Decoder-layer index, set by assign_layer_ids(); seeds the per-layer randomized Hadamard.
        self.layer_idx = -1
        # Frozen plane [n_kv, dim, sd] (dim = head_dim/2 for an LF plane, head_dim for a full one).
        self.u_frozen = None
        # Calibration statistics per KV group while a calibration pass runs (else None).
        self.u_stats = None
        # [sum, count] of the live-vs-frozen plane overlap ||U_live^T U_frozen||_F^2 / sd (else None).
        self.u_overlap = None
        # Diagnostics: list of per-call statistics dicts (sqnr_decomp.run_sqnr_decomp), else None.
        self.decomp_records = None
        # gamma of the m/f probe added to the diagnostics records (else None).
        self.mf_gamma = None

    def start_u_calibration(self):
        self.u_stats = {}

    def stop_u_calibration(self) -> Dict[int, list]:
        stats, self.u_stats = self.u_stats, None
        return stats

    def _quant(self, proxy: nn.Module, x: Tensor) -> Tensor:
        out = proxy(x)
        # The tcast override only replaces quantizers that are enabled (e.g. V stays exact with "qk")
        if self.tcast_scale is None or not proxy.act_quant.is_quant_enabled:
            return out
        import tcast
        dt = _TCAST_DTYPES.get(self.tcast_scale)
        if dt is None:
            dt = _TCAST_DTYPES[self.tcast_scale] = tcast.datatype("e2m1fnuz", self.tcast_scale)
        return tcast.vcast(x.float(), dt).to(x.dtype)

    def _dc_split(self,
                  query: Tensor,
                  key: Tensor,
                  sd: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Split Q/K (post-RoPE, GQA-expanded, ``[N, Hq, L|S, D]``) into residual and plane coordinates.

        Returns ``(res_q, res_k, coords_q, coords_k)`` in the ``[HF | LF]`` permuted basis, computed in
        fp32 and cast back to the input dtype. ``sd`` overrides ``dc_sub_dim``."""
        dt = query.dtype
        D = query.size(-1)
        half = D // 2
        sd = self.dc_sub_dim if sd is None else sd
        n_kv = self.num_key_value_heads or query.size(-3)
        group = query.size(-3) // n_kv

        P = hflf_perm(D).to(query.device)
        qp = query.float().index_select(-1, P)
        kp = key.float().index_select(-1, P)
        res_q, res_k = qp.clone(), kp.clone()
        cq = qp.new_zeros(*qp.shape[:-1], sd)
        ck = kp.new_zeros(*kp.shape[:-1], sd)
        uf_all = None
        if self.u_frozen is not None and self.u_frozen.size(-1) == sd:
            uf_all = self.u_frozen.to(device=qp.device, dtype=qp.dtype)
        for g in range(n_kv):
            sl = slice(g * group, (g + 1) * group)
            qg = qp[..., sl, :, half:]  # [N, group, L, half]
            kg = kp[..., g * group, :, half:]  # [N, S, half]
            if self.u_stats is not None:
                self._accumulate_u_stats(g, qp, kp, sl, g * group)
            if uf_all is not None and uf_all.size(-2) == D:
                # Full head-dim plane: residual = whole head minus its in-plane part.
                Uf = uf_all[g]  # [D, sd]
                qfull = qp[..., sl, :, :]
                kfull = kp[..., g * group, :, :]
                cq[..., sl, :, :] = qfull @ Uf
                ck_g = kfull @ Uf
                ck[..., sl, :, :] = ck_g.unsqueeze(-3).expand(-1, group, -1, -1)
                res_q[..., sl, :, :] = qfull - (qfull @ Uf) @ Uf.transpose(0, 1)
                res_k[..., sl, :, :] = (kfull - ck_g @ Uf.transpose(0, 1)).unsqueeze(-3).expand(
                    -1, group, -1, -1)
                continue
            elif uf_all is not None:
                U = uf_all[g].unsqueeze(0).expand(qg.size(0), -1, -1)  # [N, half, sd]
                if self.u_overlap is not None:
                    with torch.no_grad():
                        ov = (mean_subspace(qg, kg).transpose(-2, -1) @ U).pow(2).sum(
                            dim=(-2, -1)) / sd
                        self.u_overlap[0] += float(ov.sum())
                        self.u_overlap[1] += ov.numel()
            else:
                U = mean_subspace(qg, kg)
            cq[..., sl, :, :] = torch.einsum("nhld,nde->nhle", qg, U)
            ck_g = torch.einsum("nsd,nde->nse", kg, U)
            ck[..., sl, :, :] = ck_g.unsqueeze(-3).expand(-1, group, -1, -1)
            # LF block minus its in-plane part; the HF block is quantized as is.
            Pq = torch.einsum("nhld,nde,nse->nhls", qg, U, U)
            res_q[..., sl, :, half:] = qg - Pq
            Pk = torch.einsum("nsd,nde,nfe->nsf", kg, U, U)
            res_k[..., sl, :, half:] = (kg - Pk).unsqueeze(-3).expand(-1, group, -1, -1)
        return res_q.to(dt), res_k.to(dt), cq.to(dt), ck.to(dt)

    def _accumulate_u_stats(self, g: int, qp: Tensor, kp: Tensor, sl: slice, kv_idx: int):
        """Pool first and second moments of the group's Q/K (permuted basis), skipping position 0."""
        D = qp.size(-1)
        qf = qp[..., sl, 1:, :].double()  # [N, group, L-1, D]
        kf = kp[..., kv_idx, 1:, :].double()  # [N, S-1, D]
        acc = self.u_stats.setdefault(
            g, {
                'q_sum': 0.0, 'q_n': 0, 'k_sum': 0.0, 'k_n': 0, 'q_m2': 0.0, 'k_m2': 0.0})
        acc['q_sum'] = acc['q_sum'] + qf.sum(dim=(0, 1, 2))
        acc['q_n'] += qf.shape[0] * qf.shape[1] * qf.shape[2]
        acc['k_sum'] = acc['k_sum'] + kf.sum(dim=(0, 1))
        acc['k_n'] += kf.shape[0] * kf.shape[1]
        qf2 = qf.reshape(-1, D)
        kf2 = kf.reshape(-1, D)
        acc['q_m2'] = acc['q_m2'] + qf2.transpose(0, 1) @ qf2
        acc['k_m2'] = acc['k_m2'] + kf2.transpose(0, 1) @ kf2

    def _abar(self, cq: Tensor) -> Tensor:
        if self.dc_abar_mode == 'causal':
            n = torch.arange(
                1, cq.size(-2) + 1, device=cq.device, dtype=torch.float32).view(1, 1, -1, 1)
            return (cq.float().cumsum(dim=-2) / n).to(cq.dtype)
        return cq.mean(dim=-2, keepdim=True)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            attn_mask: Optional[Tensor] = None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            scale: Optional[float] = None,
            enable_gqa: bool = False):
        if self.export_mode and self.dc_sub_dim == 0:
            return self.export_handler(query, key, value, attn_mask, scale, is_causal)

        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        if attn_mask is None:
            attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        else:
            attn_bias = torch.zeros(size=attn_mask.shape, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        fully_masked_row = torch.isneginf(attn_bias).all(dim=-1, keepdim=True)
        attn_bias = attn_bias.masked_fill(fully_masked_row, 0.0)

        query, key, value = self.pre_process_q(query), self.pre_process_k(key), self.pre_process_v(value)
        if enable_gqa:
            key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        g = self.dc_gamma if self.dc_correction != 'none' else 0.0
        if self.dc_sub_dim > 0:
            res_q, res_k, cq, ck = self._dc_split(query, key)
            qin = res_q * scale_factor
            q_scaled = self._quant(self.q_scaled_quant, qin)
            # Quantize K with head_dim on the last axis (MX groups run along head_dim), then transpose.
            kin = res_k
            k_q = self._quant(self.k_transposed_quant, res_k)
            attn_weight = q_scaled @ k_q.transpose(-2, -1)
            if g and self.dc_correction == 'unshrink':
                attn_weight = attn_weight / (1.0 - g)
            attn_weight = attn_weight + (cq * scale_factor) @ ck.transpose(-2, -1)
            if g and self.dc_correction in ('mean', 'scale'):
                cq_bar = cq if self.dc_correction == 'scale' else self._abar(cq)
                attn_weight = attn_weight - g * ((cq_bar * scale_factor) @ ck.transpose(-2, -1))
        else:
            if g and self.dc_correction == 'mean':
                # Same correction term as the 2DP arm: plane coordinates of the unrotated Q/K.
                _, _, cq_g, ck_g = self._dc_split(query, key, sd=2)
                Sbar = (self._abar(cq_g) * scale_factor) @ ck_g.transpose(-2, -1)
            if self.dc_sub_dim < 0:
                Hd = hadamard_matrix(query.size(-1), query.device, query.dtype)
                if self.dc_had_rand:
                    # Flip signs before H (flips after H do not change a symmetric block quantizer).
                    seed = 10000 * self.dc_had_seed + (
                        0 if self.dc_had_rand_shared else self.layer_idx)
                    Hd = random_signs(
                        query.size(-1), query.device, query.dtype, seed=seed)[:, None] * Hd
                query = query @ Hd
                key = key @ Hd
            qin = query * scale_factor
            q_scaled = self._quant(self.q_scaled_quant, qin)
            kin = key
            k_q = self._quant(self.k_transposed_quant, key)
            attn_weight = q_scaled @ k_q.transpose(-2, -1)
            if g and self.dc_correction == 'unshrink':
                attn_weight = attn_weight / (1.0 - g)
            elif g and self.dc_correction == 'mean':
                attn_weight = attn_weight - g * Sbar

        if self.dc_sinkbal:
            from brevitas_examples.llm.mf_probe import sink_balance
            with torch.no_grad():
                ref = (query.float() * scale_factor) @ key.float().transpose(-2, -1)
                p = torch.softmax(
                    ref + attn_bias.float(), dim=-1).masked_fill(fully_masked_row, 0.0)
                attn_weight = sink_balance(attn_weight, ref, p)
                del ref, p

        if self.decomp_records is not None:
            from brevitas_examples.llm.sqnr_decomp import attention_error_stats
            with torch.no_grad():
                vq = self._quant(self.v_quant, value)
                self.decomp_records.append(
                    attention_error_stats(
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
                        mf_gamma=self.mf_gamma))

        attn_weight = attn_weight + attn_bias
        attn_weight = self.softmax_input_quant(attn_weight)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = attn_weight.masked_fill(fully_masked_row, 0.0)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        attn_weight = self.attn_output_weights_quant(attn_weight)
        attn_output = attn_weight @ self._quant(self.v_quant, value)
        attn_output = self.sdpa_output_quant(attn_output)
        return attn_output


def dc_modules(model: nn.Module) -> List[Tuple[int, DCExtractQuantScaledDotProductAttention]]:
    """2DP attention modules of ``model`` with their decoder-layer index, sorted by layer."""
    out = []
    for name, m in model.named_modules():
        if isinstance(m, DCExtractQuantScaledDotProductAttention):
            match = re.search(r"layers\.(\d+)\.", name)
            out.append((int(match.group(1)) if match else -1, m))
    return sorted(out, key=lambda x: x[0])


def assign_layer_ids(model: nn.Module):
    for idx, m in dc_modules(model):
        m.layer_idx = idx


def build_frozen_u(
        stats: Dict[int, Dict[int, dict]],
        basis: str = 'mean',
        space: str = 'lf',
        pca_center: bool = False) -> Tuple[Tensor, List[float]]:
    """Build the frozen planes from calibration statistics.

    Args:
        stats: ``{layer: {kv_group: moments}}`` from ``stop_u_calibration``.
        basis: ``mean`` (QR of the query and key means) or ``pca`` (top-2 eigenvectors of
            ``Cq / tr(Cq) + Ck / tr(Ck)``, second moments, centered if ``pca_center``).
        space: ``lf`` (plane on the LF half of the permuted head) or ``full`` (whole head).

    Returns:
        ``U [n_layers, n_kv, dim, 2]`` (fp32, permuted basis) and, for ``pca``, the top-2 energy share
        per (layer, group).
    """
    if basis not in ('mean', 'pca') or space not in ('lf', 'full'):
        raise ValueError(f"unknown plane basis={basis} / space={space}")
    rows, eigfrac = [], []
    for layer in sorted(stats):
        groups = []
        for g in sorted(stats[layer]):
            s = stats[layer][g]
            D = s['k_sum'].numel()
            sel = slice(0, D) if space == 'full' else slice(D // 2, D)
            q_mean = s['q_sum'] / s['q_n']
            k_mean = s['k_sum'] / s['k_n']
            if basis == 'mean':
                qm = q_mean[sel].float()
                km = k_mean[sel].float()
                B = torch.stack([qm / qm.norm().clamp_min(1e-12), km / km.norm().clamp_min(1e-12)],
                                -1)
                groups.append(torch.linalg.qr(B)[0])
            else:
                Cq = (s['q_m2'] / s['q_n'])[sel, sel]
                Ck = (s['k_m2'] / s['k_n'])[sel, sel]
                if pca_center:
                    Cq = Cq - torch.outer(q_mean[sel], q_mean[sel])
                    Ck = Ck - torch.outer(k_mean[sel], k_mean[sel])
                M = Cq / torch.trace(Cq) + Ck / torch.trace(Ck)
                ev, V = torch.linalg.eigh(M)  # ascending, fp64
                groups.append(V[:, -2:].flip(-1).float())
                eigfrac.append(float(ev[-2:].sum() / ev.sum()))
        rows.append(torch.stack(groups))
    return torch.stack(rows), eigfrac


def set_frozen_u(model: nn.Module, U: Optional[Tensor]):
    """Install ``U [n_layers, n_kv, dim, sd]`` (or remove it with ``None``) on the 2DP modules."""
    for idx, m in dc_modules(model):
        if U is None:
            m.u_frozen = None
            continue
        if not 0 <= idx < U.size(0):
            raise ValueError(f"frozen U has {U.size(0)} layers, no entry for layer {idx}")
        m.u_frozen = U[idx]


def u_overlap_logging(model: nn.Module, enable: bool = True) -> Dict[int, float]:
    """Start (``enable=True``) or stop logging the live-vs-frozen plane overlap.

    Returns the per-layer mean overlap collected since the last start (empty when starting)."""
    out = {}
    for idx, m in dc_modules(model):
        if m.u_overlap is not None and m.u_overlap[1] > 0:
            out[idx] = m.u_overlap[0] / m.u_overlap[1]
        m.u_overlap = [0.0, 0] if enable else None
    return out
