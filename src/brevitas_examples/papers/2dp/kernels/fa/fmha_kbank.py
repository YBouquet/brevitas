# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Portions derived from fmha_v3.py of the gfx950 Gluon tutorial ("High-Performance Gluon Kernels on AMD GFX9",
# Copyright (c) 2026 AMD ROCm Software, MIT License: see LICENSE-gluon-tutorial).

"""fmha_roof_lds ("L") with K-side LDS / DMA knobs, applied identically to Hadamard and 2DP.

All knobs default to 0 == the unmodified L kernel (fa_roof/fmha_roof_lds.py, snapshot in orig/).
  KSL  1: E8M0 K scales: the head-dim scale blocks are consumed by the MFMA in the order kappa -> swap(kappa)
          (bits 0,1 exchanged; same permutation on Q codes, Q scales and K codes, so QK^T is unchanged). A lane of
          the 32x32x64 scaled MFMA then needs physical scale bytes {2h, 2h+1} of tokens n and n+32 (adjacent in the
          token-major dword) instead of {h, h+2}: LRK reads the scales with 2 x ds_read_u16 instead of 4 x ds_read_u8
          (+ fewer pack ops). The DMA is unchanged (dword per token); the read goes through an aliased smem view.
  KCF  1: 2DP coefficients: LDS layout lane-major for the rank-2 B operand (16 B per lane: token n and n+32), read
          with ONE lane-contiguous ds_read_b128 (was ds_read2st64_b64 at 16 B token stride). DMA becomes one dword
          per lane over 4 waves (the 16 B of a lane come from two tokens).
  KCL  1: K codes: LDS layout token-major in 16-token groups (slot = n%16 + 16*chunk + 64*(n//16)). DMA
          lanes cover 16 whole tokens (16 cache lines / buffer_load instead of 32); LRK ds_read_b128 stays
          conflict-free (a 16-lane phase reads 16 consecutive 16-B slots = 256 B = all 64 banks).
  WQ1, WQ2: in-loop wait_group depth before mem1 / mem2 (L: 1, 1). 2 = exact: only the group being consumed is
          drained (the tutorial's intended depth).
  KPRED 1: the redundant waves of each K-side DMA (broadcast warp bits; L issues their loads with m0 = 0x7fffffff,
          i.e. the global fetch happens and the LDS write is dropped) are skipped with gl.warp_predicate.
  LRKE 1: LRK (K regs for the next dot1) issued in mem1 right after LRV instead of in mem2, so its LDS latency
          overlaps dot2 instead of the mem2 -> dot1 lgkmcnt(0).
KBANK_DUMP=<file> (env, compile time): append the K-side register / shared layouts as JSON lines (for lds_audit.py).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmha_roof as FR  # noqa: E402,I001,F401  (imports triton first; installs the warp_pipeline_stage patches)
from fmha_roof import _qk, _resc  # noqa: E402
from fmha_roof_lds import _v1, _v2, _resc_lazy, lane_major_smem, dma_layout  # noqa: E402

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.runtime.jit import constexpr_function  # noqa: E402
from triton.experimental import gluon  # noqa: E402
from triton.experimental.gluon import language as gl  # noqa: E402
from triton.experimental.gluon.language.amd import AMDMFMALayout, warp_pipeline_stage  # noqa: E402
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async  # noqa: E402
from triton.experimental.gluon.language.amd.cdna4 import mfma as mfma_cdna4  # noqa: E402
from triton.experimental.gluon.language._layouts import DotOperandLayout, DistributedLinearLayout, PaddedSharedLayout  # noqa: E402

from common import remap_xcd  # noqa: E402


def _ser(o):
    if isinstance(o, PaddedSharedLayout):
        return dict(kind="padded", pads=[list(p) for p in o.interval_padding_pairs],
                    bases=[list(b) for b in o.offset_bases], shape=list(o.shape))
    if isinstance(o, DistributedLinearLayout):
        return dict(kind="linear", reg=[list(b) for b in o.reg_bases], lane=[list(b) for b in o.lane_bases],
                    warp=[list(b) for b in o.warp_bases], shape=list(o.shape))
    return repr(o)


@constexpr_function
def _dump(tag, obj, ksl, kcf, kcl, corr):
    p = os.environ.get("KBANK_DUMP")
    if p:
        with open(p, "a") as f:
            f.write(json.dumps(dict(tag=str(tag), meta=dict(ksl=int(ksl), kcf=int(kcf), kcl=int(kcl), corr=int(corr)), obj=_ser(obj))) + "\n")
    return 0


@constexpr_function
def swap_coord_bits(sl, dim, b0, b1):
    """Alias view of the same bytes: every offset base gets coordinate bits b0 <-> b1 of `dim` exchanged, so logical
    index i along dim reads the element the original view calls swap(i)."""
    bases = []
    for b in sl.offset_bases:
        b = list(b)
        v = b[dim]
        x0, x1 = (v >> b0) & 1, (v >> b1) & 1
        v = (v & ~((1 << b0) | (1 << b1))) | (x0 << b1) | (x1 << b0)
        b[dim] = v
        bases.append(b)
    return PaddedSharedLayout([list(p) for p in sl.interval_padding_pairs], bases, [], list(sl.shape))


@constexpr_function
def kswz_smem(lin, KB, BN):
    """K codes [KB, BN] u8, token-major per 16-token group: slot(16 B) = n%16 + 16*chunk + 64*(n//16). A DMA wave
    instruction covers 16 whole tokens (64 B each); a 16-lane read phase (16 tokens, one chunk) covers 256
    contiguous bytes = all 64 banks."""
    assert KB == 64 and BN == 64, (KB, BN)
    lanes = sorted(tuple(b) for b in lin.lane_bases)
    regs = sorted(tuple(b) for b in lin.reg_bases if any(b))
    assert lanes == sorted([(0, 1), (0, 2), (0, 4), (0, 8), (0, 16), (16, 0)]), lanes
    assert regs == sorted([(1, 0), (2, 0), (4, 0), (8, 0), (32, 0), (0, 32)]), regs
    bases = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2], [0, 4], [0, 8], [16, 0], [32, 0], [0, 16], [0, 32]]
    return PaddedSharedLayout([[KB * BN, 16]], bases, [], [KB, BN])


@constexpr_function
def n_owner_warps(dl):
    return 1 << sum(1 for b in dl.warp_bases if any(b))


@gluon.jit
def _dma_body(dummy, dst, ptr, off):
    cdna4_async.buffer_load_to_shared(dst, ptr, off)
    return dummy


@gluon.jit
def _dma(dst, ptr, off, wid, NOWN: gl.constexpr, NW: gl.constexpr, KPRED: gl.constexpr):
    if KPRED and NOWN < NW:
        gl.warp_predicate(wid < NOWN, (wid, ), _dma_body, (dst, ptr, off))
    else:
        cdna4_async.buffer_load_to_shared(dst, ptr, off)


@gluon.jit
def _ack(k_smem, ks_smem, kc_smem, slot, KP, KS, KC, kp_base, ks_base, kc_base, n0, stride_kpm, stride_ksm,
         stride_kcm, kd_off, ksd_off, kcd_off, wid, CORR: gl.constexpr, KPRED: gl.constexpr, NW: gl.constexpr,
         NO_K: gl.constexpr, NO_S: gl.constexpr, NO_C: gl.constexpr):
    """ACK: direct-to-LDS copy of K tile n0 (codes, scales, coefficients) into `slot`. The caller commits."""
    _dma(k_smem.index(slot), KP + kp_base + n0 * stride_kpm, kd_off, wid, NO_K, NW, KPRED)
    _dma(ks_smem.index(slot), KS + ks_base + n0 * stride_ksm, ksd_off, wid, NO_S, NW, KPRED)
    if CORR:
        _dma(kc_smem.index(slot), KC + kc_base + n0 * stride_kcm, kcd_off, wid, NO_C, NW, KPRED)


@gluon.jit
def _lrk(k_smem, ks_smem, kc_smem, slot, k4_dot: gl.constexpr, ks_lay: gl.constexpr, cb: gl.constexpr,
         CORR: gl.constexpr):
    """LRK: LDS -> registers in the operand layouts (k_smem / ks_smem are the read-view aliases)."""
    k4 = cdna4_async.load_shared_relaxed(k_smem.index(slot), k4_dot)
    ks4 = cdna4_async.load_shared_relaxed(ks_smem.index(slot), ks_lay)
    if CORR:
        kc = cdna4_async.load_shared_relaxed(kc_smem.index(slot), cb)
    else:
        kc = ks4
    return k4, ks4, kc


@gluon.jit
def mx_attn_fwd_kb(QP, QS, QC, KP, KS, KC, V, SM_SCALE: gl.constexpr, L, Out,
                   stride_qpm, stride_qph, stride_qsm, stride_qsh, stride_qcm, stride_qch,
                   stride_kpm, stride_kph, stride_ksm, stride_ksh, stride_kcm, stride_kch,
                   stride_vk, stride_vh, stride_om, stride_oh, stride_ksh4, stride_kcm8,
                   HQ: gl.constexpr, HK: gl.constexpr, N_CTX: gl.constexpr,
                   BLOCK_M: gl.constexpr, BLOCK_DMODEL: gl.constexpr, BLOCK_N: gl.constexpr, CORR: gl.constexpr,
                   RESC_MEM: gl.constexpr, PRIO: gl.constexpr, ENVTAG: gl.constexpr, ABL: gl.constexpr,
                   V1X: gl.constexpr, LAZY: gl.constexpr,
                   KSL: gl.constexpr, KCF: gl.constexpr, KCL: gl.constexpr, WQ1: gl.constexpr, WQ2: gl.constexpr,
                   KPRED: gl.constexpr, LRKE: gl.constexpr):
    num_warps: gl.constexpr = gl.num_warps()
    KB: gl.constexpr = BLOCK_DMODEL // 2
    NS: gl.constexpr = BLOCK_DMODEL // 32
    PW: gl.constexpr = 8
    DP: gl.constexpr = 1 if PRIO else None
    MP: gl.constexpr = 0 if PRIO else None
    VEC2_RESC: gl.constexpr = not RESC_MEM
    L_D1: gl.constexpr = "dot1"
    L_M1: gl.constexpr = "mem1"
    L_D2: gl.constexpr = "dot2"
    L_M2: gl.constexpr = "mem2"
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

    # ---- K-side LDS layouts: read views (operand side) and DMA views (global side), aliases of the same bytes ----
    k4_lin: gl.constexpr = gl.to_linear_layout(k4_dot, [KB, BLOCK_N])
    if KCL:
        k_rd: gl.constexpr = kswz_smem(k4_lin, KB, BLOCK_N)
    else:
        k_rd: gl.constexpr = lane_major_smem(k4_lin, 4, 16)                    # L: 16 B per lane chunk
    if KSL:
        k_dv: gl.constexpr = swap_coord_bits(k_rd, 0, 4, 5)                     # logical block kappa <- phys swap
    else:
        k_dv: gl.constexpr = k_rd
    kd_layout: gl.constexpr = dma_layout(k_dv, 4, num_warps)                     # dwordx4 per lane
    ks_dv: gl.constexpr = PaddedSharedLayout.with_identity_for([[BLOCK_N * NS, 4]], [BLOCK_N, NS], [1, 0])
    if KSL:
        ks_rd: gl.constexpr = swap_coord_bits(ks_dv, 1, 0, 1)
    else:
        ks_rd: gl.constexpr = ks_dv
    ksd_layout: gl.constexpr = dma_layout(ks_dv, 2, num_warps)                   # one dword (4 scales) per token
    cb_lin: gl.constexpr = gl.to_linear_layout(cb, [PW, BLOCK_N])
    if KCF:
        kc_dv: gl.constexpr = lane_major_smem(cb_lin, 3, 8)                     # 16 B per lane: tokens n, n+32
        kcd_layout: gl.constexpr = dma_layout(kc_dv, 1, num_warps)               # one dword (2 halves) per lane
    else:
        kc_dv: gl.constexpr = PaddedSharedLayout.with_identity_for([[PW * BLOCK_N, 8]], [PW, BLOCK_N], [0, 1])
        kcd_layout: gl.constexpr = dma_layout(kc_dv, 3, num_warps)               # 16 B (8 halves) per token
    kc_rd: gl.constexpr = kc_dv
    NO_K: gl.constexpr = n_owner_warps(kd_layout)
    NO_S: gl.constexpr = n_owner_warps(ksd_layout)
    NO_C: gl.constexpr = n_owner_warps(kcd_layout)
    #     _d0: gl.constexpr = _dump("k4_dot", k4_lin, KSL, KCF, KCL, CORR)
    #     _d1: gl.constexpr = _dump("k_rd", k_rd, KSL, KCF, KCL, CORR)
    #     _d2: gl.constexpr = _dump("k_dv", k_dv, KSL, KCF, KCL, CORR)
    #     _d3: gl.constexpr = _dump("kd_layout", kd_layout, KSL, KCF, KCL, CORR)
    #     _d4: gl.constexpr = _dump("ks_lay", ks_lay, KSL, KCF, KCL, CORR)
    #     _d5: gl.constexpr = _dump("ks_rd", ks_rd, KSL, KCF, KCL, CORR)
    #     _d6: gl.constexpr = _dump("ks_dv", ks_dv, KSL, KCF, KCL, CORR)
    #     _d7: gl.constexpr = _dump("ksd_layout", ksd_layout, KSL, KCF, KCL, CORR)
    #     _d8: gl.constexpr = _dump("cb", cb_lin, KSL, KCF, KCL, CORR)
    #     _d9: gl.constexpr = _dump("kc_rd", kc_rd, KSL, KCF, KCL, CORR)
    #     _d10: gl.constexpr = _dump("kcd_layout", kcd_layout, KSL, KCF, KCL, CORR)

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
    if KSL:   # Q codes / scales in the same block order as K: logical block kappa reads physical block swap(kappa)
        qkb = (qkb & 15) | ((qkb & 16) << 1) | ((qkb & 32) >> 1)
    q4 = gl.load(QP + off_h_q * stride_qph + qm[:, None] * stride_qpm + qkb[None, :])
    qsm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, ks_blk)) + m0
    qsn = gl.arange(0, NS, layout=gl.SliceLayout(0, ks_blk))
    if KSL:
        qsn = ((qsn & 1) << 1) | ((qsn & 2) >> 1)
    qs4 = gl.convert_layout(gl.load(QS + off_h_q * stride_qsh + qsm[:, None] * stride_qsm + qsn[None, :]), qs_lay)
    if CORR:
        cm_ = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, ca)) + m0
        ck = gl.arange(0, PW, layout=gl.SliceLayout(0, ca))
        qcm = gl.load(QC + off_h_q * stride_qch + cm_[:, None] * stride_qcm + ck[None, :])
    else:
        qcm = q4
    kp_base = off_h_k * stride_kph
    ks_base = (off_h_k * stride_ksh4) * 4
    kc_base = off_h_k * stride_kch
    kd_k = gl.arange(0, KB, layout=gl.SliceLayout(1, kd_layout))
    kd_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, kd_layout))
    kd_off = kd_n[None, :] * stride_kpm + kd_k[:, None]
    ksd_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, ksd_layout))
    ksd_s = gl.arange(0, NS, layout=gl.SliceLayout(0, ksd_layout))
    ksd_off = ksd_n[:, None] * stride_ksm + ksd_s[None, :]
    if CORR:
        kcd_k = gl.arange(0, PW, layout=gl.SliceLayout(1, kcd_layout))
        kcd_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, kcd_layout))
        kcd_off = kcd_n[None, :] * (stride_kcm8 * 8) + kcd_k[:, None]
    else:
        kcd_off = ksd_off
    wid = gl.arange(0, 64 * num_warps, layout=gl.BlockedLayout([1], [64], [num_warps], [0])) // 64

    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout)
    acc = gl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=gl.float32, layout=mma_layout)
    n_blocks = (N_CTX + BLOCK_N - 1) // BLOCK_N

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
    k_smem = gl.allocate_shared_memory(KP.dtype.element_ty, [BUF_DEPTH, KB, BLOCK_N], layout=k_dv)
    ks_smem = gl.allocate_shared_memory(KS.dtype.element_ty, [BUF_DEPTH, BLOCK_N, NS], layout=ks_dv)
    kc_smem = gl.allocate_shared_memory(KC.dtype.element_ty, [BUF_DEPTH, PW, BLOCK_N], layout=kc_dv)
    if KSL:   # read-view aliases (logical block kappa reads physical block swap(kappa))
        k_smem_r = k_smem._reinterpret(layout=k_rd)
        ks_smem_r = ks_smem._reinterpret(layout=ks_rd)
    else:
        k_smem_r = k_smem
        ks_smem_r = ks_smem
    v_base = V + off_h_k * stride_vh
    v_an: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    v_ad: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_off = (gl.arange(0, BLOCK_N, layout=v_an)[:, None] * stride_vk + gl.arange(0, BLOCK_DMODEL, layout=v_ad)[None, :])
    v_step = BLOCK_N * stride_vk

    block_start = 0
    block_end = n_blocks
    NUM_BLOCKS: gl.constexpr = (N_CTX + BLOCK_N - 1) // BLOCK_N
    ODD_TAIL: gl.constexpr = (NUM_BLOCKS - 3) % 2 == 1

    cdna4_async.wait_group(0)
    # -- Prologue (the tutorial's): K0, V0, K1 | LRK0, qk0, VEC1 | barrier, K2 | LRK1, V1 ----------------
    _ack(k_smem, ks_smem, kc_smem, 0, KP, KS, KC, kp_base, ks_base, kc_base, 0, stride_kpm, stride_ksm, stride_kcm,
         kd_off, ksd_off, kcd_off, wid, CORR, False, num_warps, NO_K, NO_S, NO_C)
    cdna4_async.commit_group()                     # ACK[0]
    cdna4_async.buffer_load_to_shared(v_smem.index(0), v_base, v_off)
    cdna4_async.commit_group()                     # ACV[0]
    _ack(k_smem, ks_smem, kc_smem, 1, KP, KS, KC, kp_base, ks_base, kc_base, BLOCK_N, stride_kpm, stride_ksm,
         stride_kcm, kd_off, ksd_off, kcd_off, wid, CORR, False, num_warps, NO_K, NO_S, NO_C)
    cdna4_async.commit_group()                     # ACK[1]
    cdna4_async.wait_group(2)                      # K[0]
    k4, ks4, kc = _lrk(k_smem_r, ks_smem_r, kc_smem, 0, k4_dot, ks_lay, cb, CORR)
    qk = _qk(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, False)
    m_run, p_c, alpha_c, l_c = _v1(qk, m_i, qk_scale, ABL, V1X, p_dot_layout, LAZY)
    if RESC_MEM:
        acc, l_i = _resc(acc, l_i, alpha_c, ABL)
    gl.barrier()                                   # WAR: LRK[0] vs K[2]
    _ack(k_smem, ks_smem, kc_smem, 0, KP, KS, KC, kp_base, ks_base, kc_base, 2 * BLOCK_N, stride_kpm, stride_ksm,
         stride_kcm, kd_off, ksd_off, kcd_off, wid, CORR, False, num_warps, NO_K, NO_S, NO_C)
    cdna4_async.commit_group()                     # ACK[2]
    cdna4_async.wait_group(1)                      # K[1]
    k4, ks4, kc = _lrk(k_smem_r, ks_smem_r, kc_smem, 1, k4_dot, ks_lay, cb, CORR)
    cdna4_async.buffer_load_to_shared(v_smem.index(1), v_base + v_step, v_off)
    cdna4_async.commit_group()                     # ACV[1]

    main_loop_pairs = (block_end - 3 - block_start) // 2
    for pair_idx in tl.range(0, main_loop_pairs):
        block_n = block_start + pair_idx * 2
        # even tile (block_n): LDS slots cur=0, next=1
        with warp_pipeline_stage(L_D1, priority=DP):
            qk = _qk(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, False)
            acc, l_i, p_dot = _v2(acc, l_i, p_c, alpha_c, l_c, p_dot_layout, VEC2_RESC, ABL, V1X)
        cdna4_async.wait_group(WQ1)
        with warp_pipeline_stage(L_M1, priority=MP):
            v_dot = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)                  # LRV
            if LRKE:
                k4, ks4, kc = _lrk(k_smem_r, ks_smem_r, kc_smem, 0, k4_dot, ks_lay, cb, CORR)
            _ack(k_smem, ks_smem, kc_smem, 1, KP, KS, KC, kp_base, ks_base, kc_base, (block_n + 3) * BLOCK_N,
                 stride_kpm, stride_ksm, stride_kcm, kd_off, ksd_off, kcd_off, wid, CORR, KPRED, num_warps,
                 NO_K, NO_S, NO_C)                                                                    # ACK
            cdna4_async.commit_group()
        with warp_pipeline_stage(L_D2, priority=DP):
            acc = mfma_cdna4(p_dot, v_dot, acc)
            m_run, p_c, alpha_c, l_c = _v1(qk, m_run, qk_scale, ABL, V1X, p_dot_layout, LAZY)
        cdna4_async.wait_group(WQ2)
        with warp_pipeline_stage(L_M2, priority=MP):
            if not LRKE:
                k4, ks4, kc = _lrk(k_smem_r, ks_smem_r, kc_smem, 0, k4_dot, ks_lay, cb, CORR)  # LRK
            cdna4_async.buffer_load_to_shared(v_smem.index(0), v_base + (block_n + 2) * v_step, v_off)   # ACV
            cdna4_async.commit_group()
            if LAZY:
                acc, l_i = _resc_lazy(acc, l_i, alpha_c)
            elif RESC_MEM:
                acc, l_i = _resc(acc, l_i, alpha_c, ABL)
        # odd tile (block_n + 1): LDS slots cur=1, next=0
        with warp_pipeline_stage(L_D1, priority=DP):
            qk = _qk(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, False)
            acc, l_i, p_dot = _v2(acc, l_i, p_c, alpha_c, l_c, p_dot_layout, VEC2_RESC, ABL, V1X)
        cdna4_async.wait_group(WQ1)
        with warp_pipeline_stage(L_M1, priority=MP):
            v_dot = cdna4_async.load_shared_relaxed(v_smem.index(1), v_dot_layout)
            if LRKE:
                k4, ks4, kc = _lrk(k_smem_r, ks_smem_r, kc_smem, 1, k4_dot, ks_lay, cb, CORR)
            _ack(k_smem, ks_smem, kc_smem, 0, KP, KS, KC, kp_base, ks_base, kc_base, (block_n + 4) * BLOCK_N,
                 stride_kpm, stride_ksm, stride_kcm, kd_off, ksd_off, kcd_off, wid, CORR, KPRED, num_warps,
                 NO_K, NO_S, NO_C)
            cdna4_async.commit_group()
        with warp_pipeline_stage(L_D2, priority=DP):
            acc = mfma_cdna4(p_dot, v_dot, acc)
            m_run, p_c, alpha_c, l_c = _v1(qk, m_run, qk_scale, ABL, V1X, p_dot_layout, LAZY)
        cdna4_async.wait_group(WQ2)
        with warp_pipeline_stage(L_M2, priority=MP):
            if not LRKE:
                k4, ks4, kc = _lrk(k_smem_r, ks_smem_r, kc_smem, 1, k4_dot, ks_lay, cb, CORR)
            cdna4_async.buffer_load_to_shared(v_smem.index(1), v_base + (block_n + 3) * v_step, v_off)
            cdna4_async.commit_group()
            if LAZY:
                acc, l_i = _resc_lazy(acc, l_i, alpha_c)
            elif RESC_MEM:
                acc, l_i = _resc(acc, l_i, alpha_c, ABL)

    if ODD_TAIL:
        block_n = block_start + main_loop_pairs * 2
        with warp_pipeline_stage(L_D1, priority=DP):
            qk = _qk(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, False)
            acc, l_i, p_dot = _v2(acc, l_i, p_c, alpha_c, l_c, p_dot_layout, VEC2_RESC, ABL, V1X)
        cdna4_async.wait_group(1)
        with warp_pipeline_stage(L_M1, priority=MP):
            v_dot = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)
            _ack(k_smem, ks_smem, kc_smem, 1, KP, KS, KC, kp_base, ks_base, kc_base, (block_n + 3) * BLOCK_N,
                 stride_kpm, stride_ksm, stride_kcm, kd_off, ksd_off, kcd_off, wid, CORR, False, num_warps,
                 NO_K, NO_S, NO_C)
            cdna4_async.commit_group()
        with warp_pipeline_stage(L_D2, priority=DP):
            acc = mfma_cdna4(p_dot, v_dot, acc)
            m_run, p_c, alpha_c, l_c = _v1(qk, m_run, qk_scale, ABL, V1X, p_dot_layout, LAZY)
        cdna4_async.wait_group(1)
        with warp_pipeline_stage(L_M2, priority=MP):
            k4, ks4, kc = _lrk(k_smem_r, ks_smem_r, kc_smem, 0, k4_dot, ks_lay, cb, CORR)
            cdna4_async.buffer_load_to_shared(v_smem.index(0), v_base + (block_n + 2) * v_step, v_off)
            cdna4_async.commit_group()
            if RESC_MEM:
                acc, l_i = _resc(acc, l_i, alpha_c, ABL)

    # -- Drain (the tutorial's): pending {V[n-3], K[n-1], V[n-2]}; K regs hold tile n-2 -------------------
    nm3 = block_end - 3
    nm2 = block_end - 2
    nm1 = block_end - 1
    s_nm3 = ((nm3 - block_start) % BUF_DEPTH).to(tl.int32)
    s_nm2 = ((nm2 - block_start) % BUF_DEPTH).to(tl.int32)
    s_nm1 = ((nm1 - block_start) % BUF_DEPTH).to(tl.int32)
    qk = _qk(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, False)   # qk[n-2]
    cdna4_async.wait_group(2)                                                                # V[n-3]
    v_dot = cdna4_async.load_shared_relaxed(v_smem.index(s_nm3), v_dot_layout)
    acc, l_i, p_dot = _v2(acc, l_i, p_c, alpha_c, l_c, p_dot_layout, VEC2_RESC, ABL, V1X)
    acc = mfma_cdna4(p_dot, v_dot, acc)
    m_run, p_c, alpha_c, l_c = _v1(qk, m_run, qk_scale, ABL, V1X, p_dot_layout, LAZY)
    if RESC_MEM:
        acc, l_i = _resc(acc, l_i, alpha_c, ABL)
    gl.barrier()                                                                             # WAR: LRV[n-3] vs V[n-1]
    cdna4_async.buffer_load_to_shared(v_smem.index(s_nm1), v_base + nm1 * v_step, v_off)
    cdna4_async.commit_group()                                                               # ACV[n-1]
    cdna4_async.wait_group(2)                                                                # K[n-1]
    k4, ks4, kc = _lrk(k_smem_r, ks_smem_r, kc_smem, s_nm1, k4_dot, ks_lay, cb, CORR)
    qk = _qk(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, False)   # qk[n-1]
    cdna4_async.wait_group(1)                                                                # V[n-2]
    v_dot = cdna4_async.load_shared_relaxed(v_smem.index(s_nm2), v_dot_layout)
    acc, l_i, p_dot = _v2(acc, l_i, p_c, alpha_c, l_c, p_dot_layout, VEC2_RESC, ABL, V1X)
    acc = mfma_cdna4(p_dot, v_dot, acc)
    m_run, p_c, alpha_c, l_c = _v1(qk, m_run, qk_scale, ABL, V1X, p_dot_layout, LAZY)
    if RESC_MEM:
        acc, l_i = _resc(acc, l_i, alpha_c, ABL)
    cdna4_async.wait_group(0)                                                                # V[n-1]
    v_dot = cdna4_async.load_shared_relaxed(v_smem.index(s_nm1), v_dot_layout)
    acc, l_i, p_dot = _v2(acc, l_i, p_c, alpha_c, l_c, p_dot_layout, VEC2_RESC, ABL, V1X)
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


def run_mx(qp, qs, qc, kp, ks, kc, v, out, lse, sm_scale, corr, num_warps=8, wpe=2, agpr0=True, resc_mem=0, prio=0,
           envtag=0, abl=0, v1x=0, lazy=0, ksl=0, kcf=0, kcl=0, wq1=1, wq2=1, kpred=0, lrke=0, **_ignored):
    T, n_q, KBv = qp.shape
    n_kv = kp.shape[1]
    D = 2 * KBv
    block_m, block_n = 32 * num_warps, 64
    assert ks.stride(1) % 4 == 0 and ks.stride(0) % 16 == 0 and (qc is None or kc.stride(0) % 8 == 0), \
        "direct-to-LDS dword copies of the K scales need a 16-B token stride"
    assert D == 128 and T % block_n == 0 and T % block_m == 0 and v.stride(2) == 1 and out.stride(2) == 1
    assert abl == 0, "timing ablations are not ported to fmha_kbank"
    if qc is None:   # Hadamard: no coefficients; pass the fp16 V view so KC has a half dtype (never read)
        qc, kc = qs, v
        sqc, skc = (0, 0), (0, 0)
    else:
        sqc, skc = (qc.stride(0), qc.stride(1)), (kc.stride(0), kc.stride(1))
    grid = (n_q, triton.cdiv(T, block_m), 1)
    kw = dict(llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),)) if agpr0 else {}
    return mx_attn_fwd_kb[grid](qp, qs, qc, kp, ks, kc, v, float(sm_scale), lse, out,
                                qp.stride(0), qp.stride(1), qs.stride(0), qs.stride(1), sqc[0], sqc[1],
                                kp.stride(0), kp.stride(1), ks.stride(0), ks.stride(1), skc[0], skc[1],
                                v.stride(0), v.stride(1), out.stride(0), out.stride(1), ks.stride(1) // 4,
                                skc[0] // 8, HQ=n_q, HK=n_kv, N_CTX=T, BLOCK_M=block_m, BLOCK_DMODEL=D, BLOCK_N=block_n,
                                CORR=bool(corr), RESC_MEM=bool(resc_mem or lazy), PRIO=bool(prio), ENVTAG=envtag,
                                ABL=abl, V1X=v1x, LAZY=int(lazy), KSL=int(ksl), KCF=int(kcf), KCL=int(kcl),
                                WQ1=int(wq1), WQ2=int(wq2), KPRED=int(kpred), LRKE=int(lrke),
                                num_warps=num_warps, waves_per_eu=wpe, **kw)
