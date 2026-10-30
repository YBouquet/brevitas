# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""P2, flash-attention endpoint: fp4 QK^T (+ the 2DP rank-2 term) consumed by an online softmax, no logits stored.

One program = BM query rows of one Q head (GQA: K head = Q head // GPG). It loops over all T/BN K tiles like the
QK^T half of a flash-attention kernel: per tile, one `mfma_scaled` (e2m1 x e2m1) over the whole head, the rank-2
term (CORR=1), then the online-softmax update of the running row max m and row sum l of exp2(s - m), with
s = qk_scale * logits. Only m and l are written ([n_q, T] each). P.V is omitted: V is the same for both methods.
qk_scale folds log2(e) / sqrt(D), and for 2DP the 1/(1-gamma) of the folded unshrink.
CORR_MFMA: 0 = 2 FMAs per logit, 1 = padded fp16 32x32x16 MFMA, 2 = v_dot2_f32_f16 (see p2_qk.py).
The next K tile is loaded before the softmax update of the current one.
"""

from p2_qk import _k_side
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def gl_qk_stats(
        QP,
        QS,
        QC,
        KP,
        KS,
        KC,
        MO,
        LO,
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
        n_kt,
        qk_scale,
        GPG: gl.constexpr,
        HD: gl.constexpr,
        BM: gl.constexpr,
        BN: gl.constexpr,
        WM: gl.constexpr,
        WN: gl.constexpr,
        CORR: gl.constexpr,
        CORR_MFMA: gl.constexpr):
    KB: gl.constexpr = HD // 2
    NS: gl.constexpr = HD // 32
    pid_m = gl.program_id(0)
    hq = gl.program_id(1)
    hk = hq // GPG
    m0 = pid_m * BM
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
    row: gl.constexpr = gl.SliceLayout(1, mfl)

    # Q side: once per program
    ma = gl.arange(0, BM, layout=gl.SliceLayout(1, a_lay)) + m0
    ka = gl.arange(0, KB, layout=gl.SliceLayout(0, a_lay))
    q = gl.load(QP + hq * sqp_h + ma[:, None] * sqp_t + ka[None, :])
    msa = gl.arange(0, BM, layout=gl.SliceLayout(1, blk_s)) + m0
    nsa = gl.arange(0, NS, layout=gl.SliceLayout(0, blk_s))
    qs = gl.convert_layout(gl.load(QS + hq * sqs_h + msa[:, None] * sqs_t + nsa[None, :]), sc_a)
    mc = gl.arange(0, BM, layout=row) + m0
    if CORR != 0 and CORR_MFMA == 1:
        mq = gl.arange(0, BM, layout=gl.SliceLayout(1, ca)) + m0
        kq = gl.arange(0, 16, layout=gl.SliceLayout(0, ca))
        qcm = gl.load(
            QC + hq * sqc_h + mq[:, None] * sqc_t + kq[None, :], mask=(kq < 2)[None, :], other=0.0)
    elif CORR != 0 and CORR_MFMA == 2:
        qc0 = gl.load(QC.to(gl.pointer_type(gl.int32)) + (hq * sqc_h + mc * sqc_t) // 2)
    elif CORR != 0:
        qc0 = gl.load(QC + hq * sqc_h + mc * sqc_t).to(gl.float32)
        qc1 = gl.load(QC + hq * sqc_h + mc * sqc_t + 1).to(gl.float32)

    m_i = gl.full([BM], float("-inf"), gl.float32, row)
    l_i = gl.zeros([BM], gl.float32, row)
    k, ks, kc0, kc1 = _k_side(KP, KS, KC, hk, 0, skp_h, skp_t, sks_h, sks_t, skc_h, skc_t,
                              KB, NS, BN, b_lay, blk_s, sc_b, cb, mfl, CORR, CORR_MFMA)
    for j in range(n_kt):
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
        # next K tile before the softmax update (wraps to tile 0 after the last one: a harmless extra load)
        nn = ((j + 1) % n_kt) * BN
        k, ks, kc0, kc1 = _k_side(KP, KS, KC, hk, nn, skp_h, skp_t, sks_h, sks_t, skc_h, skc_t,
                                  KB, NS, BN, b_lay, blk_s, sc_b, cb, mfl, CORR, CORR_MFMA)
        s = acc * qk_scale
        m_new = gl.maximum(m_i, gl.max(s, axis=1))
        p = gl.exp2(s - m_new[:, None])
        l_i = l_i * gl.exp2(m_i - m_new) + gl.sum(p, axis=1)
        m_i = m_new
    gl.store(MO + hq * so_h + mc, m_i)
    gl.store(LO + hq * so_h + mc, l_i)


def stats_tiles(head_dim, max_acc=128):
    """(BM, BN, warps, WM, WN): every 32x32-MFMA tiling with <= max_acc accumulator regs per thread."""
    out = []
    for bm in (64, 128, 256):
        for bn in (32, 64, 128, 256):
            for w in (4, 8):
                if bm * bn // (64 * w) > max_acc:
                    continue
                for wm in (1, 2, 4, 8):
                    wn = w // wm
                    if wm * wn != w or bm // wm < 32 or bn // wn < 32:
                        continue
                    out.append((bm, bn, w, wm, wn))
    return out


def qk_stats(
    qp,
    qs,
    qc,
    kp,
    ks,
    kc,
    m_out,
    l_out,
    gpg,
    head_dim,
    corr,
    qk_scale,
    corr_mfma=0,
    cfg=(128, 64, 4, 4, 1)):
    """Same operands as p2_qk.qk; m_out, l_out [n_q, T] fp32. Returns the compiled kernel."""
    T, n_q = qp.shape[0], qp.shape[1]
    bm, bn, warps, wm, wn = cfg
    if qc is None:
        qc, kc = qs, ks
        sqc, skc = (0, 0), (0, 0)
    else:
        sqc, skc = (qc.stride(1), qc.stride(0)), (kc.stride(1), kc.stride(0))
    assert T % bm == 0 and T % bn == 0
    return gl_qk_stats[(T // bm, n_q)](
        qp,
        qs,
        qc,
        kp,
        ks,
        kc,
        m_out,
        l_out,
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
        m_out.stride(0),
        T // bn,
        float(qk_scale),
        GPG=gpg,
        HD=head_dim,
        BM=bm,
        BN=bn,
        WM=wm,
        WN=wn,
        CORR=corr,
        CORR_MFMA=int(corr_mfma),
        num_warps=warps)
