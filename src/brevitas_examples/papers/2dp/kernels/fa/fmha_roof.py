# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Portions derived from fmha_v3.py of the gfx950 Gluon tutorial ("High-Performance Gluon Kernels on AMD GFX9",
# Copyright (c) 2026 AMD ROCm Software, MIT License: see LICENSE-gluon-tutorial).

"""fmha_mx (fmha_v3 + MXFP4 QK^T, Hadamard / 2DP) with knobs for pushing both kernels toward their roof.

Same kernel as ../fmha_mx/fmha_mx.py (KLOAD=0, defaults reproduce it), plus constexpr knobs applied to BOTH methods:
  KLOAD    0: original _k_regs (K-scale loaded in a blocked layout and shuffled to the MFMA scale layout in the
              issuing mem1 stage -> that stage waits on its own load).
           1: K codes / scales / 2DP coefficients loaded straight into their MFMA operand layouts with tile-invariant
              offsets precomputed before the loop (no shuffle, no in-stage wait, no per-tile address VALU);
              issue order scale, kc, codes.
           2: as 1 for codes / coefficients, but the scale is loaded as one dword per token (blocked layout) and
              shuffled to the MFMA scale layout inside _qk, i.e. in the consuming dot1 stage.
  PRO_WAIT 1: drain vmcnt at the end of the prologue so the loop-header waitcnt merge does not over-wait.
  RESC_MEM 1: the eager acc/l_i rescale (32 v_pk_mul / tile) moves out of dot1 (4 MFMAs of shadow) into the
              near-empty mem2 stage (fmha_v4's placement, without the lazy skip).
  PRIO     1: s_setprio 1 on the dot clusters, 0 on the mem clusters.
  ABL      timing ablations (wrong results): 1 no exp2, 2 no in-loop K loads, 4 no cluster borders, 8 no acc
           rescale, 16 no in-loop V traffic.
  ENVTAG   unused in the body; distinguishes cache entries compiled under different env (AMDGCN_SCALARIZE_PACKED_FOPS
           is not a cache-invalidating variable).
num_warps 8 (BLOCK_M 256) or 4 (BLOCK_M 128).
"""
import os
import sys

# fmha_v3.py / common.py of the gfx950 Gluon tutorial ("High-Performance Gluon Kernels on AMD GFX9",
# kernels/attention/, MIT): not vendored, point GLUON_TUTORIAL_ATTENTION_DIR at that directory (see README.md).
_TUT = os.environ.get("GLUON_TUTORIAL_ATTENTION_DIR", "")
if not os.path.isfile(os.path.join(_TUT, "fmha_v3.py")):
    raise ImportError("set GLUON_TUTORIAL_ATTENTION_DIR to the tutorial's kernels/attention directory (fmha_v3.py)")
sys.path.insert(0, _TUT)
import triton  # noqa: E402,F401  -- before torch
import fmha_v3 as T3  # noqa: E402

import torch  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.experimental import gluon  # noqa: E402
from triton.experimental.gluon import language as gl  # noqa: E402
from triton.experimental.gluon.language.amd import AMDMFMALayout, warp_pipeline_stage  # noqa: E402
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async  # noqa: E402
from triton.experimental.gluon.language.amd.cdna4 import mfma as mfma_cdna4  # noqa: E402
from triton.experimental.gluon.language._layouts import DotOperandLayout, DistributedLinearLayout, PaddedSharedLayout  # noqa: E402

from common import remap_xcd  # noqa: E402  (tut/common.py)

nan_max = T3.nan_propagating_max


@gluon.jit
def sc_vec1(qk, m_run, qk_scale: gl.constexpr, ABL: gl.constexpr):
    """fmha_v3 VEC1 with SCALE_ON_Q=False (fma + exp2); ABL & 1 drops the exp2 (timing ablation only)."""
    m_ij = nan_max(qk, axis=1) * qk_scale
    m_new = gl.maximum(m_run, m_ij, propagate_nan=tl.PropagateNan.ALL)
    if ABL & 1:
        p = gl.fma(qk, qk_scale, -m_new[:, None])
        alpha = m_run - m_new
    else:
        p = gl.exp2(gl.fma(qk, qk_scale, -m_new[:, None]))
        alpha = gl.exp2(m_run - m_new)
    return m_new, p, alpha


_wps_exit = warp_pipeline_stage.__exit__


def _wps_exit_x(self, exc_type, exc, tb):
    """Labels starting with 'x' emit no cluster border (ABL & 4 timing ablation)."""
    if self.label is not None and str(self.label).startswith("x"):
        return False
    return _wps_exit(self, exc_type, exc, tb)


warp_pipeline_stage.__exit__ = _wps_exit_x


_wps_init = warp_pipeline_stage.__init__


def _wps_init_cx(self, label=None, *, priority=None, **kw):
    """Let warp_pipeline_stage take a constexpr priority (constexpr None = no s_setprio)."""
    from triton.language.core import _unwrap_if_constexpr
    _wps_init(self, label, priority=_unwrap_if_constexpr(priority), **kw)


warp_pipeline_stage.__init__ = _wps_init_cx


@gluon.jit
def _vec2(acc, l_i, p, alpha, p_dot_layout: gl.constexpr, out_dtype: gl.constexpr, RESCALE: gl.constexpr,
          ABL: gl.constexpr):
    """VEC2 (dot1): [acc/l_i rescale if RESCALE] + row sum + p -> fp16 operand."""
    if RESCALE:
        if not (ABL & 8):
            acc = acc * alpha[:, None]
        l_i = l_i * alpha
    l_ij = gl.sum(p, axis=1)
    l_i = l_i + l_ij
    p_dot = gl.convert_layout(p.to(out_dtype), p_dot_layout)
    return acc, l_i, p_dot


@gluon.jit
def _resc(acc, l_i, alpha, ABL: gl.constexpr):
    if ABL & 8:
        return acc, l_i * alpha
    return acc * alpha[:, None], l_i * alpha


@gluon.jit
def _k_regs0(KP, KS, KC, kp_base, ks_base, kc_base, n0, stride_kpm, stride_ksm, stride_kcm,
             KB: gl.constexpr, NS: gl.constexpr, BLOCK_N: gl.constexpr, PW: gl.constexpr,
             k4_dot: gl.constexpr, ks_blk: gl.constexpr, ks_lay: gl.constexpr, cb: gl.constexpr, CORR: gl.constexpr):
    kb = gl.arange(0, KB, layout=gl.SliceLayout(1, k4_dot))
    nb = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, k4_dot)) + n0
    k4 = gl.load(KP + kp_base + nb[None, :] * stride_kpm + kb[:, None])
    ms = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, ks_blk)) + n0
    ns = gl.arange(0, NS, layout=gl.SliceLayout(0, ks_blk))
    ks4 = gl.convert_layout(gl.load(KS + ks_base + ms[:, None] * stride_ksm + ns[None, :]), ks_lay)
    if CORR:
        kk = gl.arange(0, PW, layout=gl.SliceLayout(1, cb))
        nk = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, cb)) + n0
        kc = gl.load(KC + kc_base + nk[None, :] * stride_kcm + kk[:, None])
    else:
        kc = ks4
    return k4, ks4, kc


@gluon.jit
def _k_regs1(KP, KS, KC, kp_base, ks_base, kc_base, n0, stride_kpm, stride_ksm, stride_kcm, kp_off, ks_off, kc_off,
             CORR: gl.constexpr):
    # scale first, then coefficients, then the 4 KB of codes: a consumer of the small loads never waits on the codes
    ks4 = gl.load(KS + (ks_base + n0 * stride_ksm) + ks_off)
    if CORR:
        kc = gl.load(KC + (kc_base + n0 * stride_kcm) + kc_off)
    else:
        kc = ks4
    k4 = gl.load(KP + (kp_base + n0 * stride_kpm) + kp_off)
    return k4, ks4, kc


@gluon.jit
def _k_regs(KP, KS, KC, kp_base, ks_base, kc_base, n0, stride_kpm, stride_ksm, stride_kcm, kp_off, ks_off, kc_off,
            KB: gl.constexpr, NS: gl.constexpr, BLOCK_N: gl.constexpr, PW: gl.constexpr,
            k4_dot: gl.constexpr, ks_blk: gl.constexpr, ks_lay: gl.constexpr, cb: gl.constexpr, CORR: gl.constexpr,
            KLOAD: gl.constexpr):
    if KLOAD == 2:
        k4, ks4, kc = _k_regs1(KP, KS, KC, kp_base, ks_base, kc_base, n0, stride_kpm, stride_ksm, stride_kcm,
                               kp_off, ks_off, kc_off, CORR)   # ks_off in ks_blk here: raw dword per token
    elif KLOAD == 0:
        k4, ks4, kc = _k_regs0(KP, KS, KC, kp_base, ks_base, kc_base, n0, stride_kpm, stride_ksm, stride_kcm,
                               KB, NS, BLOCK_N, PW, k4_dot, ks_blk, ks_lay, cb, CORR)
    else:
        k4, ks4, kc = _k_regs1(KP, KS, KC, kp_base, ks_base, kc_base, n0, stride_kpm, stride_ksm, stride_kcm,
                               kp_off, ks_off, kc_off, CORR)
    return k4, ks4, kc


@gluon.jit
def _qk(q4, qs4, k4, ks4, qcm, kc, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, qk4: gl.constexpr,
        mr2: gl.constexpr, mma_layout: gl.constexpr, CORR: gl.constexpr, ks_lay: gl.constexpr, KS_RAW: gl.constexpr):
    if KS_RAW:   # KLOAD=2: the scale shuffle happens here, in the consuming dot1 stage
        ks4 = gl.convert_layout(ks4, ks_lay)
    if CORR:
        c0 = gl.convert_layout(mfma_cdna4(qcm, kc, gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=mr2)), qk4)
    else:
        c0 = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=qk4)
    s = gl.amd.cdna4.mfma_scaled(q4, qs4, "e2m1", k4, ks4, "e2m1", c0)
    return gl.convert_layout(s, mma_layout)


@gluon.jit
def mx_attn_fwd(QP, QS, QC, KP, KS, KC, V, SM_SCALE: gl.constexpr, L, Out,
                stride_qpm, stride_qph, stride_qsm, stride_qsh, stride_qcm, stride_qch,
                stride_kpm, stride_kph, stride_ksm, stride_ksh, stride_kcm, stride_kch,
                stride_vk, stride_vh, stride_om, stride_oh,
                HQ: gl.constexpr, HK: gl.constexpr, N_CTX: gl.constexpr,
                BLOCK_M: gl.constexpr, BLOCK_DMODEL: gl.constexpr, BLOCK_N: gl.constexpr, CORR: gl.constexpr,
                KLOAD: gl.constexpr, PRO_WAIT: gl.constexpr, RESC_MEM: gl.constexpr, PRIO: gl.constexpr,
                ENVTAG: gl.constexpr, ABL: gl.constexpr):
    num_warps: gl.constexpr = gl.num_warps()
    SCALE_ON_Q: gl.constexpr = False
    KB: gl.constexpr = BLOCK_DMODEL // 2
    NS: gl.constexpr = BLOCK_DMODEL // 32
    PW: gl.constexpr = 8
    DP: gl.constexpr = 1 if PRIO else None   # dot-cluster priority
    MP: gl.constexpr = 0 if PRIO else None
    VEC2_RESC: gl.constexpr = not RESC_MEM
    L_D1: gl.constexpr = "xdot1" if ABL & 4 else "dot1"
    L_M1: gl.constexpr = "xmem1" if ABL & 4 else "mem1"
    L_D2: gl.constexpr = "xdot2" if ABL & 4 else "dot2"
    L_M2: gl.constexpr = "xmem2" if ABL & 4 else "mem2"
    KL_LOOP: gl.constexpr = not (ABL & 2)
    V_LOOP: gl.constexpr = not (ABL & 16)
    off_h_q = gl.program_id(0)
    off_h_q = remap_xcd(off_h_q, HQ)
    start_m = gl.program_id(1)
    off_h_k = off_h_q * HK // HQ

    mma_layout: gl.constexpr = AMDMFMALayout(version=4, instr_shape=[32, 32, 16], transposed=True,
                                             warps_per_cta=[num_warps, 1])
    p_dot_layout: gl.constexpr = DotOperandLayout(operand_index=0, parent=mma_layout, k_width=4)
    v_dot_layout: gl.constexpr = DotOperandLayout(operand_index=1, parent=mma_layout, k_width=4)
    qk4: gl.constexpr = AMDMFMALayout(version=4, instr_shape=[32, 32, 64], transposed=True, warps_per_cta=[num_warps, 1])
    q4_dot: gl.constexpr = DotOperandLayout(operand_index=0, parent=qk4, k_width=16)
    k4_dot: gl.constexpr = DotOperandLayout(operand_index=1, parent=qk4, k_width=16)
    ks_blk: gl.constexpr = gl.BlockedLayout([1, NS], [64, 1], [num_warps, 1], [1, 0])
    qs_lay: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(q4_dot, [BLOCK_M, NS])
    ks_lay: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(k4_dot, [BLOCK_N, NS])
    mr2: gl.constexpr = AMDMFMALayout(version=4, instr_shape=[32, 32, PW], transposed=True, warps_per_cta=[num_warps, 1])
    ca: gl.constexpr = DotOperandLayout(operand_index=0, parent=mr2, k_width=PW // 2)
    cb: gl.constexpr = DotOperandLayout(operand_index=1, parent=mr2, k_width=PW // 2)

    blocked_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 16],
                                                    warps_per_cta=[num_warps, 1], order=[1, 0])
    lse_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[16, 4],
                                                warps_per_cta=[num_warps, 1], order=[1, 0])
    offs_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=blocked_layout)
    offs_m_lse_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=lse_layout)
    mma_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)
    offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=offs_m_layout)
    offs_d = gl.arange(0, BLOCK_DMODEL, layout=offs_d_layout)
    offs_m_lse = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=offs_m_lse_layout)
    qk_scale: gl.constexpr = SM_SCALE * 1.44269504089

    m0 = start_m * BLOCK_M
    qm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, q4_dot)) + m0
    qkb = gl.arange(0, KB, layout=gl.SliceLayout(0, q4_dot))
    q4 = gl.load(QP + off_h_q * stride_qph + qm[:, None] * stride_qpm + qkb[None, :])
    qsm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, ks_blk)) + m0
    qsn = gl.arange(0, NS, layout=gl.SliceLayout(0, ks_blk))
    qs4 = gl.convert_layout(gl.load(QS + off_h_q * stride_qsh + qsm[:, None] * stride_qsm + qsn[None, :]), qs_lay)
    if CORR:
        cm_ = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, ca)) + m0
        ck = gl.arange(0, PW, layout=gl.SliceLayout(0, ca))
        qcm = gl.load(QC + off_h_q * stride_qch + cm_[:, None] * stride_qcm + ck[None, :])
    else:
        qcm = q4
    kp_base = off_h_k * stride_kph
    ks_base = off_h_k * stride_ksh
    kc_base = off_h_k * stride_kch
    # tile-invariant K offsets, already in the operand layouts (KLOAD=1)
    kb_ = gl.arange(0, KB, layout=gl.SliceLayout(1, k4_dot))
    nb_ = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, k4_dot))
    kp_off = nb_[None, :] * stride_kpm + kb_[:, None]
    ms_ = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, ks_blk))
    ns_ = gl.arange(0, NS, layout=gl.SliceLayout(0, ks_blk))
    if KLOAD == 2:
        ks_off = ms_[:, None] * stride_ksm + ns_[None, :]
    else:
        ks_off = gl.convert_layout(ms_[:, None] * stride_ksm + ns_[None, :], ks_lay)
    if CORR:
        kk_ = gl.arange(0, PW, layout=gl.SliceLayout(1, cb))
        nk_ = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, cb))
        kc_off = nk_[None, :] * stride_kcm + kk_[:, None]
    else:
        kc_off = ks_off

    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout)
    acc = gl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=gl.float32, layout=mma_layout)
    n_blocks = (N_CTX + BLOCK_N - 1) // BLOCK_N

    # V: the tutorial's async-copy + LDS path (D = 128, BLOCK_N = 64)
    v_offset_bases: gl.constexpr = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
                                    [16, 0], [32, 0], [1, 0], [2, 0], [4, 0], [8, 0]]
    if num_warps == 8:
        v_async_layout: gl.constexpr = DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
            lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
            warp_bases=[[1, 0], [2, 0], [4, 0]], block_bases=[], shape=[BLOCK_N, BLOCK_DMODEL])
    else:
        v_async_layout: gl.constexpr = DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0], [4, 0]],
            lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
            warp_bases=[[1, 0], [2, 0]], block_bases=[], shape=[BLOCK_N, BLOCK_DMODEL])
    v_async_smem_layout: gl.constexpr = PaddedSharedLayout(interval_padding_pairs=[[512, 32]],
                                                           offset_bases=v_offset_bases, cga_layout=[],
                                                           shape=[BLOCK_N, BLOCK_DMODEL])
    BUF_DEPTH: gl.constexpr = 2
    v_smem = gl.allocate_shared_memory(V.dtype.element_ty, [BUF_DEPTH, BLOCK_N, BLOCK_DMODEL], layout=v_async_smem_layout)
    v_base = V + off_h_k * stride_vh
    v_an: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    v_ad: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_off = (gl.arange(0, BLOCK_N, layout=v_an)[:, None] * stride_vk + gl.arange(0, BLOCK_DMODEL, layout=v_ad)[None, :])
    v_step = BLOCK_N * stride_vk

    block_start = 0
    block_end = n_blocks
    NUM_BLOCKS: gl.constexpr = (N_CTX + BLOCK_N - 1) // BLOCK_N
    ODD_TAIL: gl.constexpr = (NUM_BLOCKS - 3) % 2 == 1
    WAIT_LOOP: gl.constexpr = 2 * BUF_DEPTH - 2

    # -- Prologue ---------------------------------------------------------------------------------------
    cdna4_async.buffer_load_to_shared(v_smem.index(0), v_base, v_off)
    cdna4_async.commit_group()
    cdna4_async.buffer_load_to_shared(v_smem.index(1), v_base + v_step, v_off)
    cdna4_async.commit_group()
    k4, ks4, kc = _k_regs(KP, KS, KC, kp_base, ks_base, kc_base, 0, stride_kpm, stride_ksm, stride_kcm,
                          kp_off, ks_off, kc_off, KB, NS, BLOCK_N, PW, k4_dot, ks_blk, ks_lay, cb, CORR, KLOAD)
    qk = _qk(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, KLOAD == 2)
    m_run, p_c, alpha_c = sc_vec1(qk, m_i, qk_scale, ABL)
    if RESC_MEM:
        acc, l_i = _resc(acc, l_i, alpha_c, ABL)
    k4, ks4, kc = _k_regs(KP, KS, KC, kp_base, ks_base, kc_base, BLOCK_N, stride_kpm, stride_ksm, stride_kcm,
                          kp_off, ks_off, kc_off, KB, NS, BLOCK_N, PW, k4_dot, ks_blk, ks_lay, cb, CORR, KLOAD)
    if PRO_WAIT:
        cdna4_async.wait_group(0)
    if not V_LOOP:   # ABL & 16: one V tile read before the loop and reused (timing ablation only)
        cdna4_async.wait_group(0)
        v_dot = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)

    main_loop_pairs = (block_end - 3 - block_start) // 2
    for pair_idx in tl.range(0, main_loop_pairs):
        block_n = block_start + pair_idx * 2
        # even tile: V slot 0
        with warp_pipeline_stage(L_D1, priority=DP):
            qk = _qk(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, KLOAD == 2)
            acc, l_i, p_dot = _vec2(acc, l_i, p_c, alpha_c, p_dot_layout, gl.float16, VEC2_RESC, ABL)
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage(L_M1, priority=MP):
            if V_LOOP:
                v_dot = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)
            if KL_LOOP:
                k4, ks4, kc = _k_regs(KP, KS, KC, kp_base, ks_base, kc_base, (block_n + 2) * BLOCK_N, stride_kpm,
                                      stride_ksm, stride_kcm, kp_off, ks_off, kc_off, KB, NS, BLOCK_N, PW, k4_dot,
                                      ks_blk, ks_lay, cb, CORR, KLOAD)
        with warp_pipeline_stage(L_D2, priority=DP):
            acc = mfma_cdna4(p_dot, v_dot, acc)
            m_run, p_c, alpha_c = sc_vec1(qk, m_run, qk_scale, ABL)
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage(L_M2, priority=MP):
            if V_LOOP:
                cdna4_async.buffer_load_to_shared(v_smem.index(0), v_base + (block_n + 2) * v_step, v_off)
                cdna4_async.commit_group()
            if RESC_MEM:
                acc, l_i = _resc(acc, l_i, alpha_c, ABL)
        # odd tile: V slot 1
        with warp_pipeline_stage(L_D1, priority=DP):
            qk = _qk(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, KLOAD == 2)
            acc, l_i, p_dot = _vec2(acc, l_i, p_c, alpha_c, p_dot_layout, gl.float16, VEC2_RESC, ABL)
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage(L_M1, priority=MP):
            if V_LOOP:
                v_dot = cdna4_async.load_shared_relaxed(v_smem.index(1), v_dot_layout)
            if KL_LOOP:
                k4, ks4, kc = _k_regs(KP, KS, KC, kp_base, ks_base, kc_base, (block_n + 3) * BLOCK_N, stride_kpm,
                                      stride_ksm, stride_kcm, kp_off, ks_off, kc_off, KB, NS, BLOCK_N, PW, k4_dot,
                                      ks_blk, ks_lay, cb, CORR, KLOAD)
        with warp_pipeline_stage(L_D2, priority=DP):
            acc = mfma_cdna4(p_dot, v_dot, acc)
            m_run, p_c, alpha_c = sc_vec1(qk, m_run, qk_scale, ABL)
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage(L_M2, priority=MP):
            if V_LOOP:
                cdna4_async.buffer_load_to_shared(v_smem.index(1), v_base + (block_n + 3) * v_step, v_off)
                cdna4_async.commit_group()
            if RESC_MEM:
                acc, l_i = _resc(acc, l_i, alpha_c, ABL)

    if ODD_TAIL:
        block_n = block_start + main_loop_pairs * 2
        with warp_pipeline_stage(L_D1, priority=DP):
            qk = _qk(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, KLOAD == 2)
            acc, l_i, p_dot = _vec2(acc, l_i, p_c, alpha_c, p_dot_layout, gl.float16, VEC2_RESC, ABL)
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage(L_M1, priority=MP):
            v_dot = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)
            k4, ks4, kc = _k_regs(KP, KS, KC, kp_base, ks_base, kc_base, (block_n + 2) * BLOCK_N, stride_kpm,
                                  stride_ksm, stride_kcm, kp_off, ks_off, kc_off, KB, NS, BLOCK_N, PW, k4_dot, ks_blk,
                                  ks_lay, cb, CORR, KLOAD)
        with warp_pipeline_stage(L_D2, priority=DP):
            acc = mfma_cdna4(p_dot, v_dot, acc)
            m_run, p_c, alpha_c = sc_vec1(qk, m_run, qk_scale, ABL)
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage(L_M2, priority=MP):
            cdna4_async.buffer_load_to_shared(v_smem.index(0), v_base + (block_n + 2) * v_step, v_off)
            cdna4_async.commit_group()
            if RESC_MEM:
                acc, l_i = _resc(acc, l_i, alpha_c, ABL)

    # -- Drain: output tiles n-3, n-2, n-1 ----------------------------------------------------------------
    # With RESC_MEM the rescale for the carried alpha has already been applied (loop mem2 / prologue), and each
    # sc_vec1 below is followed by its rescale, so _vec2 never rescales in the drain.
    nm3 = block_end - 3
    nm2 = block_end - 2
    nm1 = block_end - 1
    s_nm3 = ((nm3 - block_start) % BUF_DEPTH).to(tl.int32)
    s_nm2 = ((nm2 - block_start) % BUF_DEPTH).to(tl.int32)
    s_nm1 = ((nm1 - block_start) % BUF_DEPTH).to(tl.int32)
    qk = _qk(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, KLOAD == 2)
    cdna4_async.wait_group(1)
    v_dot = cdna4_async.load_shared_relaxed(v_smem.index(s_nm3), v_dot_layout)
    acc, l_i, p_dot = _vec2(acc, l_i, p_c, alpha_c, p_dot_layout, gl.float16, VEC2_RESC, ABL)
    acc = mfma_cdna4(p_dot, v_dot, acc)
    m_run, p_c, alpha_c = sc_vec1(qk, m_run, qk_scale, ABL)
    if RESC_MEM:
        acc, l_i = _resc(acc, l_i, alpha_c, ABL)
    gl.barrier()
    cdna4_async.buffer_load_to_shared(v_smem.index(s_nm1), v_base + nm1 * v_step, v_off)
    cdna4_async.commit_group()
    k4, ks4, kc = _k_regs(KP, KS, KC, kp_base, ks_base, kc_base, nm1 * BLOCK_N, stride_kpm, stride_ksm, stride_kcm,
                          kp_off, ks_off, kc_off, KB, NS, BLOCK_N, PW, k4_dot, ks_blk, ks_lay, cb, CORR, KLOAD)
    qk = _qk(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, KLOAD == 2)
    cdna4_async.wait_group(1)
    v_dot = cdna4_async.load_shared_relaxed(v_smem.index(s_nm2), v_dot_layout)
    acc, l_i, p_dot = _vec2(acc, l_i, p_c, alpha_c, p_dot_layout, gl.float16, VEC2_RESC, ABL)
    acc = mfma_cdna4(p_dot, v_dot, acc)
    m_run, p_c, alpha_c = sc_vec1(qk, m_run, qk_scale, ABL)
    if RESC_MEM:
        acc, l_i = _resc(acc, l_i, alpha_c, ABL)
    cdna4_async.wait_group(0)
    v_dot = cdna4_async.load_shared_relaxed(v_smem.index(s_nm1), v_dot_layout)
    acc, l_i, p_dot = _vec2(acc, l_i, p_c, alpha_c, p_dot_layout, gl.float16, VEC2_RESC, ABL)
    acc = mfma_cdna4(p_dot, v_dot, acc)

    m_i = m_run
    l_recip = 1.0 / l_i
    acc = acc * l_recip[:, None]
    o_ptrs = Out + off_h_q * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :]
    acc_blocked = gl.convert_layout(acc.to(Out.dtype.element_ty), blocked_layout)
    gl.store(o_ptrs, acc_blocked, mask=offs_m[:, None] < N_CTX)
    l_ptrs = L + off_h_q * N_CTX + offs_m_lse
    lse = m_i / 1.44269504089 + gl.log2(l_i) / 1.44269504089
    gl.store(l_ptrs, gl.convert_layout(lse, offs_m_lse_layout), mask=offs_m_lse < N_CTX)


def run_mx(qp, qs, qc, kp, ks, kc, v, out, lse, sm_scale, corr, num_warps=8, wpe=2, agpr0=True, kload=0, pro_wait=0,
           resc_mem=0, prio=0, envtag=0, abl=0, **_ignored):
    """qp [T, n_q, D/2] u8, qs [T, n_q, D/32] u8, kp/ks with n_kv heads; qc/kc [T, n, 8] fp16 zero-padded
    coefficient views (2DP) or None; v [T, n_kv, D] fp16 view; out [T, n_q, D] fp16; lse [n_q, T] fp32."""
    T, n_q, KBv = qp.shape
    n_kv = kp.shape[1]
    D = 2 * KBv
    block_m, block_n = 32 * num_warps, 64
    assert D == 128 and T % block_n == 0 and T % block_m == 0 and v.stride(2) == 1 and out.stride(2) == 1
    if qc is None:
        qc, kc = qs, ks
        sqc, skc = (0, 0), (0, 0)
    else:
        sqc, skc = (qc.stride(0), qc.stride(1)), (kc.stride(0), kc.stride(1))
    grid = (n_q, triton.cdiv(T, block_m), 1)
    kw = dict(llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),)) if agpr0 else {}
    return mx_attn_fwd[grid](qp, qs, qc, kp, ks, kc, v, float(sm_scale), lse, out,
                             qp.stride(0), qp.stride(1), qs.stride(0), qs.stride(1), sqc[0], sqc[1],
                             kp.stride(0), kp.stride(1), ks.stride(0), ks.stride(1), skc[0], skc[1],
                             v.stride(0), v.stride(1), out.stride(0), out.stride(1),
                             HQ=n_q, HK=n_kv, N_CTX=T, BLOCK_M=block_m, BLOCK_DMODEL=D, BLOCK_N=block_n,
                             CORR=bool(corr), KLOAD=kload, PRO_WAIT=bool(pro_wait), RESC_MEM=bool(resc_mem),
                             PRIO=bool(prio), ENVTAG=envtag, ABL=abl, num_warps=num_warps, waves_per_eu=wpe, **kw)
