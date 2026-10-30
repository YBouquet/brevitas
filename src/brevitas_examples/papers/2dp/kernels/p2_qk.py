# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""P2: fp4 QK^T logits (+ the 2DP rank-2 term), one Gluon kernel, any head_dim (64 / 128).

One program = NL consecutive (BM x BN) logit tiles of one Q head; GQA: Q head hq contracts with K head
hq // GPG. The Q side (fp4 codes, E8M0 scales, a') is loaded once and reused for the NL tiles; the K side of
tile j+1 is loaded before the logits of tile j are stored, so its latency overlaps the store. Per tile, the
whole head (D/2 packed bytes, D/32 E8M0 scales) is one `mfma_scaled` (e2m1 x e2m1) into an explicit
AMDMFMALayout accumulator: no K loop. Operands are fp4 packed bytes with k_width = 16.
CORR (constexpr): 0 = bare fp4 QK^T (Hadamard baseline); 1 = 2DP: R + a'.b with a' = (1-gamma) a stored
by P1 (folded unshrink: the output is (1-gamma) S, and 1/(1-gamma) joins the softmax scale).
CORR_MFMA: how the rank-2 term is computed:
  0  two VALU FMAs per logit on the accumulator's row / column slices (a', b converted to fp32);
  1  one padded fp16 32x32x16 MFMA per 32x32 sub-tile, accumulating into the same accumulator;
  2  one v_dot2_f32_f16 per logit: acc += a'0 b0 + a'1 b1 from the packed fp16 pairs (inline asm).
No 1/sqrt(D): flash attention owns the softmax scale.
Output: fp32 logits [n_q, T, T].
"""

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _k_side(
        KP,
        KS,
        KC,
        hk,
        n0,
        skp_h,
        skp_t,
        sks_h,
        sks_t,
        skc_h,
        skc_t,
        KB: gl.constexpr,
        NS: gl.constexpr,
        BN: gl.constexpr,
        b_lay: gl.constexpr,
        blk_s: gl.constexpr,
        sc_b: gl.constexpr,
        cb: gl.constexpr,
        mfl: gl.constexpr,
        CORR: gl.constexpr,
        CORR_MFMA: gl.constexpr):
    kb_ = gl.arange(0, KB, layout=gl.SliceLayout(1, b_lay))
    nb = gl.arange(0, BN, layout=gl.SliceLayout(0, b_lay)) + n0
    k = gl.load(KP + hk * skp_h + nb[None, :] * skp_t + kb_[:, None])
    msb = gl.arange(0, BN, layout=gl.SliceLayout(1, blk_s)) + n0
    nsa = gl.arange(0, NS, layout=gl.SliceLayout(0, blk_s))
    ks = gl.convert_layout(gl.load(KS + hk * sks_h + msb[:, None] * sks_t + nsa[None, :]), sc_b)
    if CORR != 0 and CORR_MFMA == 1:
        kk = gl.arange(0, 16, layout=gl.SliceLayout(1, cb))
        nk = gl.arange(0, BN, layout=gl.SliceLayout(0, cb)) + n0
        kc0 = gl.load(
            KC + hk * skc_h + nk[None, :] * skc_t + kk[:, None], mask=(kk < 2)[:, None], other=0.0)
        kc1 = kc0
    elif CORR != 0 and CORR_MFMA == 2:
        nc = gl.arange(0, BN, layout=gl.SliceLayout(0, mfl)) + n0
        kc0 = gl.load(
            KC.to(gl.pointer_type(gl.int32)) + (hk * skc_h + nc * skc_t) // 2)  # (b0, b1) fp16
        kc1 = kc0
    elif CORR != 0:
        nc = gl.arange(0, BN, layout=gl.SliceLayout(0, mfl)) + n0
        kc0 = gl.load(KC + hk * skc_h + nc * skc_t).to(gl.float32)
        kc1 = gl.load(KC + hk * skc_h + nc * skc_t + 1).to(gl.float32)
    else:
        kc0 = ks
        kc1 = ks
    return k, ks, kc0, kc1


@gluon.jit
def gl_qk(
        QP,
        QS,
        QC,
        KP,
        KS,
        KC,
        OUT,
        sqp_h,
        sqp_t,
        sqs_h,
        sqs_t,
        sqc_h,
        sqc_t,
        skp_h,
        skp_t,
        sks_h,
        sks_t,
        skc_h,
        skc_t,
        so_h,
        so_m,
        so_n,
        GPG: gl.constexpr,
        HD: gl.constexpr,
        BM: gl.constexpr,
        BN: gl.constexpr,
        WM: gl.constexpr,
        WN: gl.constexpr,
        NL: gl.constexpr,
        CORR: gl.constexpr,
        CORR_MFMA: gl.constexpr):
    KB: gl.constexpr = HD // 2
    NS: gl.constexpr = HD // 32
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    hq = gl.program_id(2)
    hk = hq // GPG
    m0 = pid_m * BM
    nbase = pid_n * (BN * NL)
    mfl: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[32, 32, 64], transposed=False, warps_per_cta=[WM, WN])
    a_lay: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfl, k_width=16)
    b_lay: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfl, k_width=16)
    blk_s: gl.constexpr = gl.BlockedLayout([1, NS], [64, 1], [WM, WN], [1, 0])
    sc_a: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(a_lay, [BM, NS])
    sc_b: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(b_lay, [BN, NS])
    mfl16: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[32, 32, 16], transposed=False, warps_per_cta=[WM, WN])
    ca: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfl16, k_width=8)
    cb: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfl16, k_width=8)

    # Q side: once per program
    ma = gl.arange(0, BM, layout=gl.SliceLayout(1, a_lay)) + m0
    ka = gl.arange(0, KB, layout=gl.SliceLayout(0, a_lay))
    q = gl.load(QP + hq * sqp_h + ma[:, None] * sqp_t + ka[None, :])
    msa = gl.arange(0, BM, layout=gl.SliceLayout(1, blk_s)) + m0
    nsa = gl.arange(0, NS, layout=gl.SliceLayout(0, blk_s))
    qs = gl.convert_layout(gl.load(QS + hq * sqs_h + msa[:, None] * sqs_t + nsa[None, :]), sc_a)
    mc = gl.arange(0, BM, layout=gl.SliceLayout(1, mfl)) + m0
    if CORR != 0 and CORR_MFMA == 1:
        mq = gl.arange(0, BM, layout=gl.SliceLayout(1, ca)) + m0
        kq = gl.arange(0, 16, layout=gl.SliceLayout(0, ca))
        qcm = gl.load(
            QC + hq * sqc_h + mq[:, None] * sqc_t + kq[None, :], mask=(kq < 2)[None, :], other=0.0)
    elif CORR != 0 and CORR_MFMA == 2:
        qc0 = gl.load(
            QC.to(gl.pointer_type(gl.int32)) + (hq * sqc_h + mc * sqc_t) // 2)  # (a'0, a'1) fp16
    elif CORR != 0:
        qc0 = gl.load(QC + hq * sqc_h + mc * sqc_t).to(gl.float32)
        qc1 = gl.load(QC + hq * sqc_h + mc * sqc_t + 1).to(gl.float32)

    k, ks, kc0, kc1 = _k_side(KP, KS, KC, hk, nbase, skp_h, skp_t, sks_h, sks_t, skc_h, skc_t,
                              KB, NS, BN, b_lay, blk_s, sc_b, cb, mfl, CORR, CORR_MFMA)
    for j in gl.static_range(NL):
        n0 = nbase + j * BN
        acc = gl.amd.cdna4.mfma_scaled(
            q, qs, "e2m1", k, ks, "e2m1", gl.zeros([BM, BN], gl.float32, layout=mfl))
        if CORR != 0 and CORR_MFMA == 2:
            qa, _ = gl.broadcast(qc0[:, None], acc)
            kb2, _ = gl.broadcast(kc0[None, :], acc)
            acc = gl.inline_asm_elementwise(
                "v_dot2_f32_f16 $0, $1, $2, $3",
                "=v,v,v,v", [qa, kb2, acc],
                dtype=gl.float32,
                is_pure=True,
                pack=1)
        elif CORR != 0 and CORR_MFMA == 1:
            acc = gl.convert_layout(gl.amd.cdna3.mfma(qcm, kc0, gl.convert_layout(acc, mfl16)), mfl)
        elif CORR != 0:
            ab = qc0[:, None] * kc0[None, :] + qc1[:, None] * kc1[None, :]
            acc = acc + ab
        if j + 1 < NL:  # next tile's K side before this tile's store
            k, ks, kc0, kc1 = _k_side(KP, KS, KC, hk, n0 + BN, skp_h, skp_t, sks_h, sks_t, skc_h, skc_t,
                                      KB, NS, BN, b_lay, blk_s, sc_b, cb, mfl, CORR, CORR_MFMA)
        nc = gl.arange(0, BN, layout=gl.SliceLayout(0, mfl)) + n0
        gl.store(OUT + hq * so_h + mc[:, None] * so_m + nc[None, :] * so_n, acc)


def tiles(head_dim, max_acc=128):
    """(BM, BN, warps, WM, WN, NL): every 32x32-MFMA tiling with <= max_acc accumulator regs per thread."""
    out = []
    for bm in (64, 128, 256):
        for bn in (64, 128, 256):
            for w in (4, 8):
                if bm * bn // (64 * w) > max_acc:
                    continue
                for wm in (1, 2, 4, 8):
                    wn = w // wm
                    if wm * wn != w or bm // wm < 32 or bn // wn < 32:
                        continue
                    for nl in (1, 2, 4):
                        out.append((bm, bn, w, wm, wn, nl))
    return out


def qk(
    qp, qs, qc, kp, ks, kc, out, gpg, head_dim, corr, corr_mfma=False, cfg=(128, 128, 4, 2, 2, 1)):
    """qp [T, n_q, D/2] u8, qs [T, n_q, D/32] u8, qc [T, n_q, 2] fp16 (and kp/ks/kc with n_kv); out [n_q, T, T]
    fp32. Strided views are fine (head = stride(1), token = stride(0)). cfg = (BM, BN, warps, WM, WN, NL).
    corr_mfma: 0 / False = 2 FMAs, 1 / True = MFMA, 2 = v_dot2 (the coefficients must be fp16 pairs at an
    even element offset, as P1 writes them).
    Returns the compiled kernel."""
    T, n_q = qp.shape[0], qp.shape[1]
    bm, bn, warps, wm, wn, nl = cfg
    if qc is None:
        qc, kc = qs, ks
        sqc, skc = (0, 0), (0, 0)
    else:
        sqc, skc = (qc.stride(1), qc.stride(0)), (kc.stride(1), kc.stride(0))
    assert T % (bn * nl) == 0 and T % bm == 0
    grid = (T // bm, T // (bn * nl), n_q)
    return gl_qk[grid](
        qp,
        qs,
        qc,
        kp,
        ks,
        kc,
        out,
        qp.stride(1),
        qp.stride(0),
        qs.stride(1),
        qs.stride(0),
        sqc[0],
        sqc[1],
        kp.stride(1),
        kp.stride(0),
        ks.stride(1),
        ks.stride(0),
        skc[0],
        skc[1],
        out.stride(0),
        out.stride(1),
        out.stride(2),
        GPG=gpg,
        HD=head_dim,
        BM=bm,
        BN=bn,
        WM=wm,
        WN=wn,
        NL=nl,
        CORR=corr,
        CORR_MFMA=int(corr_mfma),
        num_warps=warps)
