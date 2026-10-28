# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Torch fp32 references for the 2DP and Hadamard kernels (correctness checks of the benchmark)."""

import torch

E2M1_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def rope_std(y, cos, sin):
    """HF-transformers rotate_half RoPE on y [T, H, D], cos/sin [T, D/2]."""
    c = torch.cat([cos, cos], -1)[:, None, :]
    s = torch.cat([sin, sin], -1)[:, None, :]
    h = y.shape[-1] // 2
    return y * c + torch.cat([-y[..., h:], y[..., :h]], -1) * s


def hflf_perm(D):
    """Band order [HF_lo | HF_hi | LF_lo | LF_hi], each D/4 (dc_extract_sdpa.hflf_perm)."""
    nb, h = D // 2, D // 4
    return torch.tensor(
        list(range(0, h)) + list(range(nb, nb + h)) + list(range(h, nb)) + list(range(nb + h, D)))


def sylvester(n, device=None):
    """Unnormalized natural-order Hadamard H_n."""
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def mx_quant(z):
    """OCP MXFP4 over blocks of 32 contiguous elements of the last axis, RNE, E8M0 floor scale.
    -> (dequantized, codes u8 (sign bit 8), e8m0 u8 [..., D/32])."""
    sh = z.shape
    x = z.reshape(*sh[:-1], sh[-1] // 32, 32)
    e = torch.floor(torch.log2(x.abs().amax(-1, keepdim=True).clamp_min(1e-30))) - 2
    xs = x / torch.exp2(e)
    ax = xs.abs()
    c = torch.zeros_like(ax, dtype=torch.long)
    for t, incl in [(0.25, 1), (0.75, 0), (1.25, 1), (1.75, 0), (2.5, 1), (3.5, 0), (5.0, 1)]:
        c = c + ((ax > t) if incl else (ax >= t)).long()
    grid = torch.tensor(E2M1_GRID, device=z.device)
    val = torch.sign(xs) * grid[c] * torch.exp2(e)
    code = (c | torch.where(xs < 0, 8, 0)).to(torch.uint8)
    return val.reshape(sh), code.reshape(sh), (e.squeeze(-1) + 127).to(torch.uint8)


def unpack_mx(packed, scales):
    """Kernel output packed [T, H, D/2] (even element in the low nibble), e8m0 [T, H, D/32]
    -> (dequantized [T, H, D], codes [T, H, D])."""
    codes = torch.stack([(packed & 0xF).long(), (packed >> 4).long()], -1)
    codes = codes.reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    grid = torch.tensor(E2M1_GRID, device=packed.device)
    mag = grid[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    return mag * torch.exp2(scales.float() - 127.0).repeat_interleave(32, -1), codes.to(torch.uint8)


def ref_2dp(o, U, n_q, gpg):
    """o [T, n_qk, D] post-RoPE, U [n_kv, D/2, 2] frozen plane on [LF_lo | LF_hi].
    -> (plane coordinates a [T, n_qk, 2], residual [T, n_qk, D] in the hflf order)."""
    D, n_qk = o.shape[-1], o.shape[1]
    op = o[..., hflf_perm(D).to(o.device)]
    grp = torch.tensor([h // gpg if h < n_q else h - n_q for h in range(n_qk)], device=o.device)
    Uh = U[grp]
    lf = op[..., D // 2:]
    a = torch.einsum("thd,hde->the", lf, Uh)
    return a, torch.cat([op[..., :D // 2], lf - torch.einsum("the,hde->thd", a, Uh)], -1)


def ref_logits(dq, dk, gpg, aq=None, ak=None, u=1.0):
    """Logits [n_q, T, T] = u * dq.dk (+ aq.ak); K head = Q head // gpg (GQA). No 1/sqrt(D)."""
    out = u * torch.einsum("thd,shd->hts", dq, dk.repeat_interleave(gpg, 1))
    if aq is not None:
        out = out + torch.einsum("the,she->hts", aq, ak.repeat_interleave(gpg, 1))
    return out
