# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hadamard baseline quantizer J2: RoPE + Walsh-Hadamard + MXFP4 for any head_dim, one Triton kernel.

Hadamard pipeline = projection GEMM (p1_2dp.front_end_2dp, epi=0) -> this kernel -> bare fp4 QK^T
(p2_qk.qk, corr=0). One launch covers all Q and K heads (adjacent in the fused qkv output).

Input  x [T, H, D] fp16, standard (HF-transformers) layout, pre-RoPE; cos/sin [T, D/2] fp32.
Output packed [T, H, D/2] u8 + e8m0 [T, H, D/32]: the standard MX packing of
z = RoPE(x) @ H_D / sqrt(D) (HADAMARD=1) or of RoPE(x) (HADAMARD=0), blocks of 32 contiguous.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwht_stage(a, H: tl.constexpr, NB: tl.constexpr, BLOCK_T: tl.constexpr):
    """One radix-2 butterfly stage (pair stride H) on [BLOCK_T, NB]."""
    G: tl.constexpr = NB // (2 * H)
    x = tl.reshape(a, (BLOCK_T, G, 2, H))
    x = tl.trans(x, (0, 1, 3, 2))
    lo, hi = tl.split(x)
    x = tl.join(lo + hi, lo - hi)
    x = tl.trans(x, (0, 1, 3, 2))
    return tl.reshape(x, (BLOCK_T, NB))


@triton.jit
def _fwht(a, NB: tl.constexpr, LOG_NB: tl.constexpr, BLOCK_T: tl.constexpr):
    """Natural-order (Sylvester) Walsh-Hadamard over the last axis, unnormalized."""
    for i in tl.static_range(LOG_NB):
        a = _fwht_stage(a, 1 << i, NB, BLOCK_T)
    return a


@triton.jit
def _quant_store(
        o,
        PK,
        SC,
        t_off,
        t_mask,
        h,
        stride_pt,
        stride_ph,
        stride_st,
        stride_sh,
        side,
        NB: tl.constexpr,
        BLOCK_T: tl.constexpr):
    """mxfp4 of one head-half o [BLOCK_T, NB] (NB/32 blocks), stored at byte side*NB/2.
    gfx950 v_cvt_scalef32_pk_fp4_f32 (same instruction as the 2DP epilogue)."""
    NBL: tl.constexpr = NB // 32
    x = tl.reshape(o, (BLOCK_T, NBL, 32))
    amax = tl.maximum(tl.max(tl.abs(x), axis=2), 1e-30)
    e = tl.floor(tl.log2(amax)) - 2.0
    a, b2 = tl.split(tl.reshape(x, (BLOCK_T, NBL, 16, 2)))
    sc = tl.exp2(e)[:, :, None] + tl.zeros_like(a)
    r = tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3",
        "=v,v,v,v", [a, b2, sc],
        dtype=tl.int32,
        is_pure=True,
        pack=1)
    byte = (r & 0xFF).to(tl.uint8)  # [BLOCK_T, NBL, 16]
    blk = tl.arange(0, NBL)
    b = tl.arange(0, 16)
    pb = PK + t_off[:, None, None] * stride_pt + h * stride_ph + side * (NB // 2) \
        + blk[None, :, None] * 16 + b[None, None, :]
    tl.store(pb, byte, mask=t_mask[:, None, None])
    sb = SC + t_off[:, None] * stride_st + h * stride_sh + side * NBL + blk[None, :]
    tl.store(sb, (e + 127.0).to(tl.uint8), mask=t_mask[:, None])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": b}, num_warps=w, num_stages=2)
        for b in (32, 64, 128)
        for w in (4, 8)],
    key=["T", "HD", "HADAMARD"],
)
@triton.jit
def rope_quant_hd_kernel(
        X,
        COS,
        SIN,
        PK,
        SC,
        T,
        scale,
        stride_xt,
        stride_xh,
        stride_pt,
        stride_ph,
        stride_st,
        stride_sh,
        stride_ct,
        HD: tl.constexpr,
        LOG_NB: tl.constexpr,
        HADAMARD: tl.constexpr,
        BLOCK_T: tl.constexpr):
    NB: tl.constexpr = HD // 2
    pid_t = tl.program_id(0)
    h = tl.program_id(1)
    t_off = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_off < T
    d = tl.arange(0, NB)
    xb = X + t_off[:, None] * stride_xt + h * stride_xh
    x_lo = tl.load(xb + d[None, :], mask=t_mask[:, None], other=0.0).to(tl.float32)
    x_hi = tl.load(xb + NB + d[None, :], mask=t_mask[:, None], other=0.0).to(tl.float32)
    c = tl.load(COS + t_off[:, None] * stride_ct + d[None, :], mask=t_mask[:, None], other=1.0)
    s = tl.load(SIN + t_off[:, None] * stride_ct + d[None, :], mask=t_mask[:, None], other=0.0)
    o_lo = x_lo * c - x_hi * s
    o_hi = x_hi * c + x_lo * s
    if HADAMARD == 1:
        # H_D = [[A, A], [A, -A]], A = H_{D/2}:  z = [(lo+hi)A, (lo-hi)A] / sqrt(D)
        f_lo = _fwht(o_lo, NB, LOG_NB, BLOCK_T)
        f_hi = _fwht(o_hi, NB, LOG_NB, BLOCK_T)
        o_lo = (f_lo + f_hi) * scale
        o_hi = (f_lo - f_hi) * scale
    _quant_store(
        o_lo, PK, SC, t_off, t_mask, h, stride_pt, stride_ph, stride_st, stride_sh, 0, NB, BLOCK_T)
    _quant_store(
        o_hi, PK, SC, t_off, t_mask, h, stride_pt, stride_ph, stride_st, stride_sh, 1, NB, BLOCK_T)


def rope_quant_hd(x, cos, sin, packed, scales, hadamard=True):
    """x [T, H, D] fp16 (strided view ok, last dim contiguous), cos/sin [T, D/2] fp32,
    packed [T, H, D/2] u8, scales [T, H, D/32] u8 (preallocated)."""
    T, H, D = x.shape
    assert x.stride(2) == 1
    grid = lambda META: (triton.cdiv(T, META["BLOCK_T"]), H)
    rope_quant_hd_kernel[grid](
        x,
        cos,
        sin,
        packed,
        scales,
        T,
        1.0 / (D ** 0.5),
        x.stride(0),
        x.stride(1),
        packed.stride(0),
        packed.stride(1),
        scales.stride(0),
        scales.stride(1),
        cos.stride(0),
        HD=D,
        LOG_NB=(D // 2).bit_length() - 1,
        HADAMARD=1 if hadamard else 0,
    )
    return packed, scales


def sylvester(n, device=None, dtype=torch.float32):
    """Unnormalized natural-order Hadamard H_n (n a power of 2)."""
    H = torch.ones(1, 1, device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H
