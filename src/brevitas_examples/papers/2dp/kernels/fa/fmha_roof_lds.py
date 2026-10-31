# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Portions derived from fmha_v3.py of the gfx950 Gluon tutorial ("High-Performance Gluon Kernels on AMD GFX9",
# Copyright (c) 2026 AMD ROCm Software, MIT License: see LICENSE-gluon-tutorial).

"""fmha_roof with the tutorial's K path: K tiles (fp4 codes + E8M0 scales + 2DP coefficients) are copied ONCE per
workgroup into LDS with direct-to-LDS buffer loads (ACK, 3 tiles ahead, mem1) and read back by every wave in mem2
(LRK, for the next dot1), exactly the tutorial fmha_v3 schedule. Same knobs as fmha_roof (RESC_MEM, PRIO, num_warps,
env via ENVTAG) applied to both methods.

Why: in fmha_mx every one of the 8 waves loads the whole K tile from global with lanes = tokens (token stride
n_heads * 64 B), so each global_load_dwordx4 touches 32 different cache lines: per tile per CU 8 waves x 4 instr x 32
lines of TA work (plus 2 x 64 lines for scales and, in 2DP, 2 x 32 for the coefficients per wave). Removing the
in-loop K loads (ABL=2 timing ablation) took Hadamard from 1257 to 797 us at T=8192.

LDS layouts:
  K codes  [KB, BLOCK_N] u8: lane-major copy of the k4_dot register layout (16 B per lane chunk, lanes contiguous),
           so LRK is conflict-free ds_read_b128 and the DMA writes 16 B per lane contiguously (4 warp-instructions).
  scales   [BLOCK_N, NS] u8 token-major (one dword per token = the global layout), read into the MFMA scale layout.
  coeffs   [PW, BLOCK_N] f16 token-major (16 B per token = the global layout), read into the rank-2 operand layout.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmha_roof as FR  # noqa: E402,I001  (imports triton first; installs the warp_pipeline_stage patches)
from fmha_roof import _qk, _resc, _vec2, nan_max, sc_vec1  # noqa: E402

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


@constexpr_function
def lane_major_smem(lin, vec_bits, pad):
    """Shared layout whose offset bits are: the first vec_bits register bases (one contiguous per-lane chunk), then
    the 6 lane bases, then the remaining register bases. Reading `lin` back is then lanes-contiguous (no bank
    conflicts) and each lane's chunk is one vector read. Broadcast (zero) bases are dropped."""
    rank = len(lin.shape)
    zero = [0] * rank
    bases = [list(b) for b in list(lin.reg_bases)[:vec_bits] + list(lin.lane_bases) + list(lin.reg_bases)[vec_bits:]]
    bases = [b for b in bases if b != zero]
    numel = 1
    for s_ in lin.shape:
        numel *= s_
    assert (1 << len(bases)) == numel, (bases, lin.shape)
    return PaddedSharedLayout([[numel, pad]], bases, [], list(lin.shape))


@constexpr_function
def dma_layout(smem, vec_bits, num_warps):
    """Register layout for a direct-to-LDS copy into `smem` (PaddedSharedLayout): lane i writes vec contiguous
    elements at offset i * vec within each warp-instruction; leftover offset bits go to warps, then registers
    (more instructions); missing warp bits are broadcast (the lowering predicates redundant warps off)."""
    bases = [list(b) for b in smem.offset_bases]
    rank = len(bases[0])
    reg = bases[:vec_bits]
    lane = bases[vec_bits:vec_bits + 6]
    rest = bases[vec_bits + 6:]
    wb = num_warps.bit_length() - 1
    warp = rest[:wb] + [[0] * rank] * max(0, wb - len(rest))
    reg = reg + rest[wb:]
    assert len(lane) == 6
    return DistributedLinearLayout(reg, lane, warp, [], list(smem.shape))


LAZY_TH = gl.constexpr(8.0)   # log2 units, fmha_v4's threshold: p <= 2^8, exact (acc, l_i and m share the lagging frame)


@gluon.jit
def _vec1_lazy(qk, m_run, qk_scale: gl.constexpr, TH: gl.constexpr):
    """fmha_v4 VEC1 (branchless lazy max): the running max only advances when a row's tile max exceeds it by more
    than LAZY_TH, so alpha == 1 exactly for the stable rows."""
    m_ij = nan_max(qk, axis=1) * qk_scale
    m_cand = gl.maximum(m_run, m_ij, propagate_nan=tl.PropagateNan.ALL)
    m_new = gl.where(m_cand - m_run > TH, m_cand, m_run)
    p = gl.exp2(gl.fma(qk, qk_scale, -m_new[:, None]))
    alpha = gl.exp2(m_run - m_new)
    return m_new, p, alpha


def _lazy_asm(pack):
    body = "\n".join(f"v_mul_f32 ${i}, ${i}, ${2 * pack + i}" for i in range(pack))
    asm = (f"v_cmp_neq_f32 vcc, 1.0, ${2 * pack}\n"
           "s_cbranch_vccz .Lroof_rs_${:uid}\n" + body + "\n.Lroof_rs_${:uid}:")
    cons = ",".join(["=v"] * pack + [str(i) for i in range(pack)] + ["v"] * pack + ["~{vcc}"])
    return asm, cons


RS_PACK = 16
RS_ASM, RS_CONS = _lazy_asm(RS_PACK)
RS_ASM = gl.constexpr(RS_ASM)
RS_CONS = gl.constexpr(RS_CONS)
RS_PACK = gl.constexpr(RS_PACK)


@gluon.jit
def _resc_lazy(acc, l_i, alpha):
    """acc *= alpha, skipped per wave (s_cbranch_vccz) when every row of the wave has alpha == 1. One alpha per lane
    in this MFMA layout, so a block of RS_PACK accumulator registers of a lane shares one alpha operand."""
    acc = gl.inline_asm_elementwise(RS_ASM, RS_CONS, [acc, alpha[:, None]], dtype=gl.float32, is_pure=True,
                                    pack=RS_PACK)
    return acc, l_i * alpha


@gluon.jit
def _v1(qk, m_run, qk_scale: gl.constexpr, ABL: gl.constexpr, V1X: gl.constexpr, p_dot_layout: gl.constexpr,
        LAZY: gl.constexpr = 0):
    """VEC1 (dot2). V1X=0: max + fma + exp2 (fmha_v3); 1: + the row sum; 2: + the row sum and the fp16 operand cast,
    so VEC2 (dot1, 4 fp4 MFMAs of shadow) keeps only the rescale. LAZY: fmha_v4's lagging max."""
    if LAZY:
        if LAZY == 2:   # timing ablation: the max is frozen after the first tile (rescale branch always skipped)
            m_new, p, alpha = _vec1_lazy(qk, m_run, qk_scale, 1.0e4)
        else:
            m_new, p, alpha = _vec1_lazy(qk, m_run, qk_scale, LAZY_TH)
    else:
        m_new, p, alpha = sc_vec1(qk, m_run, qk_scale, ABL)
    if V1X >= 1:
        l_ij = gl.sum(p, axis=1)
    else:
        l_ij = alpha
    if V1X == 2:
        pc = gl.convert_layout(p.to(gl.float16), p_dot_layout)
    else:
        pc = p
    return m_new, pc, alpha, l_ij


@gluon.jit
def _v2(acc, l_i, pc, alpha, l_ij, p_dot_layout: gl.constexpr, RESCALE: gl.constexpr, ABL: gl.constexpr,
        V1X: gl.constexpr):
    """VEC2 (dot1): [rescale] + whatever VEC1 did not do."""
    if RESCALE:
        if not (ABL & 8):
            acc = acc * alpha[:, None]
        l_i = l_i * alpha
    if V1X == 0:
        l_ij = gl.sum(pc, axis=1)
    l_i = l_i + l_ij
    if V1X == 2:
        p_dot = pc
    else:
        p_dot = gl.convert_layout(pc.to(gl.float16), p_dot_layout)
    return acc, l_i, p_dot


@gluon.jit
def _ack(k_smem, ks_smem, kc_smem, slot, KP, KS, KC, kp_base, ks_base, kc_base, n0, stride_kpm, stride_ksm,
         stride_kcm, kd_off, ksd_off, kcd_off, CORR: gl.constexpr):
    """ACK: direct-to-LDS copy of K tile n0 (codes, scales, coefficients) into `slot`. The caller commits."""
    cdna4_async.buffer_load_to_shared(k_smem.index(slot), KP + kp_base + n0 * stride_kpm, kd_off)
    cdna4_async.buffer_load_to_shared(ks_smem.index(slot), KS + ks_base + n0 * stride_ksm, ksd_off)
    if CORR:
        cdna4_async.buffer_load_to_shared(kc_smem.index(slot), KC + kc_base + n0 * stride_kcm, kcd_off)


@gluon.jit
def _lrk(k_smem, ks_smem, kc_smem, slot, k4_dot: gl.constexpr, ks_lay: gl.constexpr, cb: gl.constexpr,
         CORR: gl.constexpr):
    """LRK: LDS -> registers in the operand layouts."""
    k4 = cdna4_async.load_shared_relaxed(k_smem.index(slot), k4_dot)
    ks4 = cdna4_async.load_shared_relaxed(ks_smem.index(slot), ks_lay)
    if CORR:
        kc = cdna4_async.load_shared_relaxed(kc_smem.index(slot), cb)
    else:
        kc = ks4
    return k4, ks4, kc


@gluon.jit
def mx_attn_fwd_lds(QP, QS, QC, KP, KS, KC, V, SM_SCALE: gl.constexpr, L, Out,
                    stride_qpm, stride_qph, stride_qsm, stride_qsh, stride_qcm, stride_qch,
                    stride_kpm, stride_kph, stride_ksm, stride_ksh, stride_kcm, stride_kch,
                    stride_vk, stride_vh, stride_om, stride_oh, stride_ksh4, stride_kcm8,
                    HQ: gl.constexpr, HK: gl.constexpr, N_CTX: gl.constexpr,
                    BLOCK_M: gl.constexpr, BLOCK_DMODEL: gl.constexpr, BLOCK_N: gl.constexpr, CORR: gl.constexpr,
                    RESC_MEM: gl.constexpr, PRIO: gl.constexpr, ENVTAG: gl.constexpr, ABL: gl.constexpr,
                    V1X: gl.constexpr, LAZY: gl.constexpr):
    num_warps: gl.constexpr = gl.num_warps()
    KB: gl.constexpr = BLOCK_DMODEL // 2
    NS: gl.constexpr = BLOCK_DMODEL // 32
    PW: gl.constexpr = 8
    DP: gl.constexpr = 1 if PRIO else None
    MP: gl.constexpr = 0 if PRIO else None
    VEC2_RESC: gl.constexpr = not RESC_MEM
    L_D1: gl.constexpr = "xdot1" if ABL & 4 else "dot1"
    L_M1: gl.constexpr = "xmem1" if ABL & 4 else "mem1"
    L_D2: gl.constexpr = "xdot2" if ABL & 4 else "dot2"
    L_M2: gl.constexpr = "xmem2" if ABL & 4 else "mem2"
    V_LOOP: gl.constexpr = not (ABL & 16)
    K_LOOP: gl.constexpr = not (ABL & 32)
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

    # K-side LDS layouts and their direct-to-LDS copy layouts
    k4_lin: gl.constexpr = gl.to_linear_layout(k4_dot, [KB, BLOCK_N])
    k_smem_layout: gl.constexpr = lane_major_smem(k4_lin, 4, 16)                 # 16 B per lane chunk
    kd_layout: gl.constexpr = dma_layout(k_smem_layout, 4, num_warps)            # dwordx4 per lane
    ks_smem_layout: gl.constexpr = PaddedSharedLayout.with_identity_for([[BLOCK_N * NS, 4]], [BLOCK_N, NS], [1, 0])
    ksd_layout: gl.constexpr = dma_layout(ks_smem_layout, 2, num_warps)          # one dword (4 scales) per token
    kc_smem_layout: gl.constexpr = PaddedSharedLayout.with_identity_for([[PW * BLOCK_N, 8]], [PW, BLOCK_N], [0, 1])
    kcd_layout: gl.constexpr = dma_layout(kc_smem_layout, 3, num_warps)          # 16 B (8 halves) per token

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
    ks_base = (off_h_k * stride_ksh4) * 4        # == off_h_k * stride_ksh, provably 4-byte aligned (dword DMA)
    kc_base = off_h_k * stride_kch
    # tile-invariant DMA offsets
    kd_k = gl.arange(0, KB, layout=gl.SliceLayout(1, kd_layout))
    kd_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, kd_layout))
    kd_off = kd_n[None, :] * stride_kpm + kd_k[:, None]
    ksd_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, ksd_layout))
    ksd_s = gl.arange(0, NS, layout=gl.SliceLayout(0, ksd_layout))
    ksd_off = ksd_n[:, None] * stride_ksm + ksd_s[None, :]
    if CORR:
        kcd_k = gl.arange(0, PW, layout=gl.SliceLayout(1, kcd_layout))
        kcd_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, kcd_layout))
        kcd_off = kcd_n[None, :] * (stride_kcm8 * 8) + kcd_k[:, None]   # 16 B per token provably contiguous
    else:
        kcd_off = ksd_off

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
    k_smem = gl.allocate_shared_memory(KP.dtype.element_ty, [BUF_DEPTH, KB, BLOCK_N], layout=k_smem_layout)
    ks_smem = gl.allocate_shared_memory(KS.dtype.element_ty, [BUF_DEPTH, BLOCK_N, NS], layout=ks_smem_layout)
    kc_smem = gl.allocate_shared_memory(KC.dtype.element_ty, [BUF_DEPTH, PW, BLOCK_N], layout=kc_smem_layout)
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

    cdna4_async.wait_group(0)
    # -- Prologue (the tutorial's): K0, V0, K1 | LRK0, qk0, VEC1 | barrier, K2 | LRK1, V1 ----------------
    _ack(k_smem, ks_smem, kc_smem, 0, KP, KS, KC, kp_base, ks_base, kc_base, 0, stride_kpm, stride_ksm, stride_kcm,
         kd_off, ksd_off, kcd_off, CORR)
    cdna4_async.commit_group()                     # ACK[0]
    cdna4_async.buffer_load_to_shared(v_smem.index(0), v_base, v_off)
    cdna4_async.commit_group()                     # ACV[0]
    _ack(k_smem, ks_smem, kc_smem, 1, KP, KS, KC, kp_base, ks_base, kc_base, BLOCK_N, stride_kpm, stride_ksm,
         stride_kcm, kd_off, ksd_off, kcd_off, CORR)
    cdna4_async.commit_group()                     # ACK[1]
    cdna4_async.wait_group(2)                      # K[0]
    k4, ks4, kc = _lrk(k_smem, ks_smem, kc_smem, 0, k4_dot, ks_lay, cb, CORR)
    qk = _qk(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, False)
    m_run, p_c, alpha_c, l_c = _v1(qk, m_i, qk_scale, ABL, V1X, p_dot_layout, LAZY)
    if RESC_MEM:
        acc, l_i = _resc(acc, l_i, alpha_c, ABL)
    gl.barrier()                                   # WAR: LRK[0] vs K[2]
    _ack(k_smem, ks_smem, kc_smem, 0, KP, KS, KC, kp_base, ks_base, kc_base, 2 * BLOCK_N, stride_kpm, stride_ksm,
         stride_kcm, kd_off, ksd_off, kcd_off, CORR)
    cdna4_async.commit_group()                     # ACK[2]
    cdna4_async.wait_group(1)                      # K[1]
    k4, ks4, kc = _lrk(k_smem, ks_smem, kc_smem, 1, k4_dot, ks_lay, cb, CORR)
    cdna4_async.buffer_load_to_shared(v_smem.index(1), v_base + v_step, v_off)
    cdna4_async.commit_group()                     # ACV[1]

    if not V_LOOP:   # ABL & 16 (timing only): one V tile read before the loop and reused
        cdna4_async.wait_group(0)
        v_dot = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)
    main_loop_pairs = (block_end - 3 - block_start) // 2
    for pair_idx in tl.range(0, main_loop_pairs):
        block_n = block_start + pair_idx * 2
        # even tile (block_n): LDS slots cur=0, next=1
        with warp_pipeline_stage(L_D1, priority=DP):
            qk = _qk(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, False)
            acc, l_i, p_dot = _v2(acc, l_i, p_c, alpha_c, l_c, p_dot_layout, VEC2_RESC, ABL, V1X)
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage(L_M1, priority=MP):
            if V_LOOP:
                v_dot = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)              # LRV
            if K_LOOP:
                _ack(k_smem, ks_smem, kc_smem, 1, KP, KS, KC, kp_base, ks_base, kc_base, (block_n + 3) * BLOCK_N,
                     stride_kpm, stride_ksm, stride_kcm, kd_off, ksd_off, kcd_off, CORR)            # ACK
            cdna4_async.commit_group()
        with warp_pipeline_stage(L_D2, priority=DP):
            acc = mfma_cdna4(p_dot, v_dot, acc)
            m_run, p_c, alpha_c, l_c = _v1(qk, m_run, qk_scale, ABL, V1X, p_dot_layout, LAZY)
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage(L_M2, priority=MP):
            if K_LOOP:
                k4, ks4, kc = _lrk(k_smem, ks_smem, kc_smem, 0, k4_dot, ks_lay, cb, CORR)        # LRK
            if V_LOOP:
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
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage(L_M1, priority=MP):
            if V_LOOP:
                v_dot = cdna4_async.load_shared_relaxed(v_smem.index(1), v_dot_layout)
            if K_LOOP:
                _ack(k_smem, ks_smem, kc_smem, 0, KP, KS, KC, kp_base, ks_base, kc_base, (block_n + 4) * BLOCK_N,
                     stride_kpm, stride_ksm, stride_kcm, kd_off, ksd_off, kcd_off, CORR)
            cdna4_async.commit_group()
        with warp_pipeline_stage(L_D2, priority=DP):
            acc = mfma_cdna4(p_dot, v_dot, acc)
            m_run, p_c, alpha_c, l_c = _v1(qk, m_run, qk_scale, ABL, V1X, p_dot_layout, LAZY)
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage(L_M2, priority=MP):
            if K_LOOP:
                k4, ks4, kc = _lrk(k_smem, ks_smem, kc_smem, 1, k4_dot, ks_lay, cb, CORR)
            if V_LOOP:
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
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage(L_M1, priority=MP):
            v_dot = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)
            _ack(k_smem, ks_smem, kc_smem, 1, KP, KS, KC, kp_base, ks_base, kc_base, (block_n + 3) * BLOCK_N,
                 stride_kpm, stride_ksm, stride_kcm, kd_off, ksd_off, kcd_off, CORR)
            cdna4_async.commit_group()
        with warp_pipeline_stage(L_D2, priority=DP):
            acc = mfma_cdna4(p_dot, v_dot, acc)
            m_run, p_c, alpha_c, l_c = _v1(qk, m_run, qk_scale, ABL, V1X, p_dot_layout, LAZY)
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage(L_M2, priority=MP):
            k4, ks4, kc = _lrk(k_smem, ks_smem, kc_smem, 0, k4_dot, ks_lay, cb, CORR)
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
    k4, ks4, kc = _lrk(k_smem, ks_smem, kc_smem, s_nm1, k4_dot, ks_lay, cb, CORR)
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
           envtag=0, abl=0, v1x=0, lazy=0, **_ignored):
    T, n_q, KBv = qp.shape
    n_kv = kp.shape[1]
    D = 2 * KBv
    block_m, block_n = 32 * num_warps, 64
    assert ks.stride(1) % 4 == 0 and ks.stride(0) % 16 == 0 and (qc is None or kc.stride(0) % 8 == 0), \
        "direct-to-LDS dword copies of the K scales need a 16-B token stride"
    assert D == 128 and T % block_n == 0 and T % block_m == 0 and v.stride(2) == 1 and out.stride(2) == 1
    if qc is None:   # Hadamard: no coefficients; pass the fp16 V view so KC has a half dtype (never read)
        qc, kc = qs, v
        sqc, skc = (0, 0), (0, 0)
    else:
        sqc, skc = (qc.stride(0), qc.stride(1)), (kc.stride(0), kc.stride(1))
    grid = (n_q, triton.cdiv(T, block_m), 1)
    kw = dict(llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),)) if agpr0 else {}
    return mx_attn_fwd_lds[grid](qp, qs, qc, kp, ks, kc, v, float(sm_scale), lse, out,
                                 qp.stride(0), qp.stride(1), qs.stride(0), qs.stride(1), sqc[0], sqc[1],
                                 kp.stride(0), kp.stride(1), ks.stride(0), ks.stride(1), skc[0], skc[1],
                                 v.stride(0), v.stride(1), out.stride(0), out.stride(1), ks.stride(1) // 4,
                                 skc[0] // 8, HQ=n_q, HK=n_kv, N_CTX=T, BLOCK_M=block_m, BLOCK_DMODEL=D, BLOCK_N=block_n,
                                 CORR=bool(corr), RESC_MEM=bool(resc_mem or lazy), PRIO=bool(prio), ENVTAG=envtag, ABL=abl, V1X=v1x, LAZY=int(lazy),
                                 num_warps=num_warps, waves_per_eu=wpe, **kw)
