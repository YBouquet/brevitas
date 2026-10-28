# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""P2: fp4 QK^T logits (+ the 2DP rank-2 term), one Gluon kernel, any head_dim (64 / 128).

One program = a (BM x BN) logit tile of one Q head; GQA: Q head hq contracts with K head hq // GPG.
The whole head (D/2 packed bytes, D/32 E8M0 scales) is one `mfma_scaled` (e2m1 x e2m1) into an explicit
AMDMFMALayout accumulator: no K loop. Operands are fp4 packed bytes with k_width = 16.
CORR (constexpr): 0 = bare fp4 QK^T (Hadamard baseline); 1 = 2DP: R + a'.b with a' = (1-gamma) a stored
by P1 (folded unshrink: the output is (1-gamma) S, and 1/(1-gamma) joins the softmax scale).
CORR_MFMA: the rank-2 term as one padded fp16 32x32x16 MFMA into the same accumulator (1), or as VALU FMAs
on the accumulator's row / column slices (0). No 1/sqrt(D): flash attention owns the softmax scale.
Output: fp32 logits [n_q, T, T].
"""

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


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
        FUSE_LOAD: gl.constexpr,
        CORR: gl.constexpr,
        CORR_MFMA: gl.constexpr):
    KB: gl.constexpr = HD // 2  # packed bytes per head
    NS: gl.constexpr = HD // 32  # e8m0 scales per head
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    hq = gl.program_id(2)
    hk = hq // GPG
    m0 = pid_m * BM
    n0 = pid_n * BN
    mfl: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[32, 32, 64], transposed=False, warps_per_cta=[WM, WN])
    # fp4 operands are packed bytes: k_width = 16 bytes = the 32 e2m1 each lane feeds one
    # scaled MFMA (gluon test_amd_mfma_scaled). k_width=32, inherited from gluon_e2e.py, is wrong.
    a_lay: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfl, k_width=16)
    b_lay: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfl, k_width=16)
    blk_s: gl.constexpr = gl.BlockedLayout([1, NS], [64, 1], [WM, WN], [1, 0])

    if FUSE_LOAD:
        ma = gl.arange(0, BM, layout=gl.SliceLayout(1, a_lay)) + m0
        ka = gl.arange(0, KB, layout=gl.SliceLayout(0, a_lay))
        q = gl.load(QP + hq * sqp_h + ma[:, None] * sqp_t + ka[None, :])
        kb_ = gl.arange(0, KB, layout=gl.SliceLayout(1, b_lay))
        nb = gl.arange(0, BN, layout=gl.SliceLayout(0, b_lay)) + n0
        k = gl.load(KP + hk * skp_h + nb[None, :] * skp_t + kb_[:, None])
    else:
        blk_a: gl.constexpr = gl.BlockedLayout([1, KB], [64, 1], [WM, WN], [1, 0])
        blk_b: gl.constexpr = gl.BlockedLayout([KB, 1], [1, 64], [WN, WM], [0, 1])
        ma = gl.arange(0, BM, layout=gl.SliceLayout(1, blk_a)) + m0
        ka = gl.arange(0, KB, layout=gl.SliceLayout(0, blk_a))
        q = gl.load(QP + hq * sqp_h + ma[:, None] * sqp_t + ka[None, :])
        kb_ = gl.arange(0, KB, layout=gl.SliceLayout(1, blk_b))
        nb = gl.arange(0, BN, layout=gl.SliceLayout(0, blk_b)) + n0
        k = gl.load(KP + hk * skp_h + nb[None, :] * skp_t + kb_[:, None])
        q = gl.convert_layout(q, a_lay)
        k = gl.convert_layout(k, b_lay)

    msa = gl.arange(0, BM, layout=gl.SliceLayout(1, blk_s)) + m0
    nsa = gl.arange(0, NS, layout=gl.SliceLayout(0, blk_s))
    qs = gl.load(QS + hq * sqs_h + msa[:, None] * sqs_t + nsa[None, :])
    msb = gl.arange(0, BN, layout=gl.SliceLayout(1, blk_s)) + n0
    ks = gl.load(KS + hk * sks_h + msb[:, None] * sks_t + nsa[None, :])
    sc_a: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(a_lay, [BM, NS])
    sc_b: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(b_lay, [BN, NS])
    qs = gl.convert_layout(qs, sc_a)
    ks = gl.convert_layout(ks, sc_b)

    mc = gl.arange(0, BM, layout=gl.SliceLayout(1, mfl)) + m0
    nc = gl.arange(0, BN, layout=gl.SliceLayout(0, mfl)) + n0
    # plane-coordinate loads are issued BEFORE the fp4 MFMA, with the q/k loads, so their
    # latency overlaps instead of sitting between the MFMA and the store of every program
    if CORR != 0 and CORR_MFMA:
        # rank-2 term on the MATRIX CORE: [qc0 qc1 0..0] (K padded 2 -> 16) @ [kc0; kc1; 0..] as one
        # fp16 32x32x16 MFMA accumulating into the same tile. A 32x32 accumulator has the same
        # register mapping whatever the instruction K, so the two MFMA layouts convert trivially.
        mfl16: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[32, 32, 16], transposed=False, warps_per_cta=[WM, WN])
        ca: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfl16, k_width=8)
        cb: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfl16, k_width=8)
        mq = gl.arange(0, BM, layout=gl.SliceLayout(1, ca)) + m0
        kq = gl.arange(0, 16, layout=gl.SliceLayout(0, ca))
        qcm = gl.load(
            QC + hq * sqc_h + mq[:, None] * sqc_t + kq[None, :], mask=(kq < 2)[None, :], other=0.0)
        kk = gl.arange(0, 16, layout=gl.SliceLayout(1, cb))
        nk = gl.arange(0, BN, layout=gl.SliceLayout(0, cb)) + n0
        kcm = gl.load(
            KC + hk * skc_h + nk[None, :] * skc_t + kk[:, None], mask=(kk < 2)[:, None], other=0.0)
    elif CORR != 0:
        # plane coordinates straight into the accumulator's row / column slices
        qc0 = gl.load(QC + hq * sqc_h + mc * sqc_t).to(gl.float32)
        qc1 = gl.load(QC + hq * sqc_h + mc * sqc_t + 1).to(gl.float32)
        kc0 = gl.load(KC + hk * skc_h + nc * skc_t).to(gl.float32)
        kc1 = gl.load(KC + hk * skc_h + nc * skc_t + 1).to(gl.float32)

    acc = gl.zeros([BM, BN], gl.float32, layout=mfl)
    acc = gl.amd.cdna4.mfma_scaled(q, qs, "e2m1", k, ks, "e2m1", acc)

    if CORR != 0 and CORR_MFMA:
        acc = gl.convert_layout(gl.amd.cdna3.mfma(qcm, kcm, gl.convert_layout(acc, mfl16)), mfl)
    elif CORR != 0:
        ab = qc0[:, None] * kc0[None, :] + qc1[:, None] * kc1[None, :]
        acc = acc + ab
    gl.store(OUT + hq * so_h + mc[:, None] * so_m + nc[None, :] * so_n, acc)


# (BM, BN, warps, WM, WN, fuse); all valid tilings, picked per (T, D, CORR) by `tune_qk`
CANDS = [(128, 128, 4, 2, 2, True), (128, 64, 4, 4, 1, False), (64, 128, 4, 2, 2, False),
         (256, 64, 8, 8, 1, True), (128, 64, 4, 2, 2, True), (64, 128, 4, 1, 4, True),
         (256, 128, 4, 4, 1, True), (128, 128, 4, 4, 1, True)]


def qk(qp, qs, qc, kp, ks, kc, out, gpg, head_dim, corr, corr_mfma=False, cfg=CANDS[0]):
    """qp [T, n_q, D/2] u8, qs [T, n_q, D/32] u8, qc [T, n_q, 2] fp16 (and kp/ks/kc with n_kv);
    out [n_q, T, T] fp32. Strided views are fine (head = stride(1), token = stride(0)).
    Returns the compiled kernel (asm / n_regs for the ISA table)."""
    T, n_q = qp.shape[0], qp.shape[1]
    bm, bn, warps, wm, wn, fuse = cfg
    if qc is None:
        qc, kc = qs, ks  # unused when corr == 0
        sqc, skc = (0, 0), (0, 0)
    else:
        sqc, skc = (qc.stride(1), qc.stride(0)), (kc.stride(1), kc.stride(0))
    grid = (triton.cdiv(T, bm), triton.cdiv(T, bn), n_q)
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
        FUSE_LOAD=fuse,
        CORR=corr,
        CORR_MFMA=bool(corr_mfma),
        num_warps=warps)


def tune_qk(bench_fn, args, kwargs, cands=CANDS):
    """Autotune-lite: time every candidate, return (best_cfg, best_us). `bench_fn(callable)`
    returns microseconds. Candidates that fail to compile are skipped."""
    best = (None, float("inf"))
    for cfg in cands:
        try:
            t = bench_fn(lambda: qk(*args, cfg=cfg, **kwargs))
        except Exception as e:  # noqa: BLE001
            print(
                f"    [qk cand {cfg} skipped: {type(e).__name__}: {str(e).splitlines()[0][:120]}]")
            continue
        if t < best[1]:
            best = (cfg, t)
    return best
