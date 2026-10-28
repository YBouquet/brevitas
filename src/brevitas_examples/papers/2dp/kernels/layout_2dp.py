# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Offline layout for the 2DP kernels (pure torch): weight column permutation, RoPE tables, U tile tables.

The HF/LF band permutation commutes with RoPE, so it is folded into the weight columns here, together with
the split-rope pairing that puts each RoPE (lo, hi) pair column for column in a tile's two quadrants.
"""

from __future__ import annotations

import torch


def split_rope_heads_per_tile(head_dim: int, quad_cols: int = 128) -> int:
    """Heads per 256-col N-tile in the split-rope layout: one quadrant (128 cols) holds the
    `lo` quarter of HPT heads, so HPT = 128 / (D/4). 8 for D=64, 4 for D=128."""
    return quad_cols // (head_dim // 4)


def permute_fused_qkv_split_rope_hd(
        W_fused: torch.Tensor, n_q: int, n_kv: int, head_dim: int) -> torch.Tensor:
    """`permute_fused_qkv_split_rope` for any head_dim. Column layout:
        [ HF tiles | LF tiles | V raw ],  each 256-col tile = [ HPT heads' lo (128) | same heads' hi (128) ]
    so column c of the left quadrant and column c of the right quadrant are one RoPE pair
    (head = c // QD, band = c % QD) and the rotation in the GEMM epilogue is a register-local
    combine of paired quadrants."""
    d = W_fused.shape[0]
    D, QD = head_dim, head_dim // 4
    NB = D // 2
    h_tot, n_qk = n_q + 2 * n_kv, n_q + n_kv
    hpt = split_rope_heads_per_tile(D)
    assert n_qk % hpt == 0, f"n_qk={n_qk} must be a multiple of heads_per_tile={hpt}"
    w = W_fused.view(d, h_tot, D)
    qk = w[:, :n_qk]
    dev = w.device

    def _band_tiles(lo_cols, hi_cols):
        lo = qk[:, :, torch.tensor(lo_cols, device=dev)]  # [d, n_qk, QD]
        hi = qk[:, :, torch.tensor(hi_cols, device=dev)]
        cols = []
        for t in range(n_qk // hpt):
            hs = slice(t * hpt, (t + 1) * hpt)
            cols.append(lo[:, hs].reshape(d, hpt * QD))
            cols.append(hi[:, hs].reshape(d, hpt * QD))
        return torch.cat(cols, dim=1)

    hf = _band_tiles(list(range(0, QD)), list(range(NB, NB + QD)))
    lf = _band_tiles(list(range(QD, NB)), list(range(NB + QD, D)))
    v = w[:, n_qk:].reshape(d, n_kv * D)
    return torch.cat([hf, lf, v], dim=1).contiguous()


def rope_tables_hd(positions: torch.Tensor, inv_freq: torch.Tensor):
    """-> (cos, sin) [T, NB] for the NB = len(inv_freq) bands. The split-rope kernel indexes
    band = (HF or LF offset) + c % QD; the headsplit kernel indexes band = c % NB."""
    ang = positions[:, None].float() * inv_freq[None, :].float()
    return ang.cos().contiguous(), ang.sin().contiguous()


def ublk_constants(U: torch.Tensor, n_q: int, n_kv: int, head_dim: int) -> torch.Tensor:
    """Frozen plane U [n_kv, D/2, 2] (LF basis [LF_lo | LF_hi]) -> per-LF-tile column tables
    UT [n_tiles, 4, 128] fp32: rows (U_lo e0, U_lo e1, U_hi e0, U_hi e1) for each column of the
    tile's left/right quadrant. Column c of tile t is head h = t*HPT + c // QD, band c % QD;
    Q head h uses group h // gpg, K head n_q + g uses group g. No scaling folded in."""
    D, QD = head_dim, head_dim // 4
    hpt = split_rope_heads_per_tile(D)
    n_qk, gpg = n_q + n_kv, n_q // n_kv
    n_tiles = n_qk // hpt
    UT = torch.empty(n_tiles, 4, 128, dtype=torch.float32, device=U.device)
    for t in range(n_tiles):
        for c in range(128):
            h = t * hpt + c // QD
            g = h // gpg if h < n_q else h - n_q
            b = c % QD
            UT[t, 0, c], UT[t, 1, c] = U[g, b, 0], U[g, b, 1]
            UT[t, 2, c], UT[t, 3, c] = U[g, QD + b, 0], U[g, QD + b, 1]
    return UT.contiguous()
