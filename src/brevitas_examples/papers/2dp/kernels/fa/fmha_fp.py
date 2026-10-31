# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Portions derived from fmha_v3.py of the gfx950 Gluon tutorial ("High-Performance Gluon Kernels on AMD GFX9",
# Copyright (c) 2026 AMD ROCm Software, MIT License: see LICENSE-gluon-tutorial).

"""fa_fp16: fmha_d64 (MXFP4 Hadamard / 2DP) + an fp16 QK^T mode (QF16 1) in the SAME kernel body.

QF16 1: Q fp16 [T, n_q, D] loaded once into registers (DotOperand of the 32x32x16 MFMA, k_width 8; SQ 1 folds the
        softmax scale * log2(e) into Q in fp32 before the fp16 cast, as the tutorial's SCALE_ON_Q), K fp16 tile
        [D, BLOCK_N] staged once per workgroup in LDS by direct-to-LDS dwordx4 loads (ACK, mem1, 3 tiles ahead), read
        back by every wave (LRK, mem2 or mem1 with LRKE) and QK^T = fp16 v_mfma_f32_32x32x16_f16 (D/16 per 32x32 block).
        Everything else (softmax, lazy, prio, fence, EXS, P.V, O + LSE, prologue / drain, waits) is the code below,
        unchanged. CORR / R2P / KSL / KCF / KCL / KSA are fp4-only knobs (forced off).
        K LDS layout (offset bits, [d, n]): d (log2 D bits, one token row contiguous) | non-phase token bits up to
        1 KB | 16-B pad | token 1, 2, 4, 8 [| rest]. D=128, BLOCK_N=64: exactly the tutorial fmha_v3 kt layout
        (d7 | n16 n32 | pad | n1 n2 n4 n8); D=64, BLOCK_N=128: d6 | n16 n32 n64 | pad | n1 n2 n4 n8; D=64, BLOCK_N=64:
        d6 | n16 n32 (n8^n16) | pad | n1 n2 n4. A 16-lane ds_read_b128 phase (tokens n0..n0+15, one 16-B d chunk) then
        hits 16 distinct 16-B bank groups; each DMA wave-instruction writes 1 KB contiguous = whole token rows.
DRF 1 (any method): scheduling fences between the drain steps (the drain's straight-line code is where the default
        toolchain spills); DRF 0 = the tutorial's drain.
        The DMA register layout is derived from the LDS layout (dma_layout); at D=128 / BLOCK_N 64 / 8 warps it equals
        the tutorial's kt_async_layout (asserted).

fmha_kbank generalized to head_dim 64 (Llama-3.2-1B) and to BLOCK_M / BLOCK_N choices; D=128 with BLOCK_N=64 and
BLOCK_M=32*num_warps takes exactly the fmha_kbank code path (same layouts, same ops).

D=64 port (both methods):
  K codes  KB = D/2 = 32 B per token: one 32x32x64 fp4 MFMA per 32x32 S block (k4_dot / lane_major_smem / dma_layout are
           derived from the operand layout, unchanged code).
  K scales NS = D/32 = 2 B per token. Direct-to-LDS needs dword (or dwordx4) per lane on gfx950 (8/16-bit DMA is
           disabled in Triton), so the LDS scale tile stays [BLOCK_N, 4] B (dword per token, as at D=128) and is read
           through a [BLOCK_N, 2] subslice.
           KSA 0: the dword starts at the head's own 2 scale bytes (byte offset 2*h is only 2-aligned for odd KV heads:
                  the 4-B alignment is asserted to Triton with multiple_of; relies on unaligned buffer loads, checked);
                  bytes 2,3 (the next head's scales) are never read.
           KSA 1: aligned dword holding heads (2j, 2j+1); both 2-B halves are read and the head parity selects.
  coeffs   unchanged (16 B per token slot, PW=8 halves, rank-2 MFMA 32x32x8).
  V        [BLOCK_N, 64] fp16, rows of 128 B. LDS offset bits: d (6) | token 1 | token 16, 32 | token 2, 4, 8 (| 64), 64-B
           pad every 1 KB (= one 64-lane x 16-B DMA instruction, so the direct-to-LDS write stays contiguous). Byte
           offset mod 256 of tokens 4k..4k+3 is {0,128,64,192}+c: the 4 consecutive token rows of a ds_read_b64_tr_b16
           phase land on disjoint 16-bank groups (same rule as the D=128 layout). DMA layout derived from it.
  O store  BlockedLayout [1,8] x [8,8] lanes (64 columns).
BLOCK_N 128 (D=64 only): every K/V layout is derived from BLOCK_N. BLOCK_M = 32 * num_warps * R, R = 1 or 2 (R=2: two
32-row MFMA tiles per wave).

fmha_kbank docstring follows.
fmha_roof_lds ("L") with K-side LDS / DMA knobs, applied identically to Hadamard and 2DP.

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
from fmha_roof_lds import _v1 as _v1_orig, _v2, _resc_lazy, _vec1_lazy, lane_major_smem, dma_layout  # noqa: E402
from fmha_roof import sc_vec1, nan_max  # noqa: E402

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.runtime.jit import constexpr_function  # noqa: E402
from triton.experimental import gluon  # noqa: E402
from triton.experimental.gluon import language as gl  # noqa: E402
from triton.experimental.gluon.language.amd import AMDMFMALayout, warp_pipeline_stage  # noqa: E402
from triton.experimental.gluon.language.amd import slice as amd_slice  # noqa: E402
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
def v_smem_bases(D, BN):
    """V LDS offset bases [token, d] for the generic (non-fmha_kbank) path: d bits, token 1, token 16, 32 (-> 1 KB
    chunk at D=64), then token 2, 4, 8, 64. With a 64-B pad per 1 KB, tokens 4k..4k+3 sit at byte offsets
    {0, 128, 64, 192} + c mod 256."""
    assert D == 64 and BN in (64, 128), (D, BN)
    b = [[0, 1 << i] for i in range(6)] + [[1, 0], [16, 0], [32, 0], [2, 0], [4, 0], [8, 0]]
    if BN == 128:
        b.append([64, 0])
    return b


TUT_KT_BASES = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32], [0, 1], [0, 2], [0, 4],
                [0, 8]]
TUT_KT_ASYNC = dict(reg=[[1, 0], [2, 0], [4, 0], [0, 8]], lane=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
                    warp=[[0, 1], [0, 2], [0, 4]])


@constexpr_function
def kt_smem_layout(D, BN):
    """fp16 K tile [D, BN] LDS layout (see the module docstring); pad 8 halves (16 B) per 512 halves (1 KB)."""
    d = [[1 << i, 0] for i in range(D.bit_length() - 1)]
    if D == 128 and BN == 64:
        b = TUT_KT_BASES
    elif D == 64 and BN == 128:
        b = d + [[0, 16], [0, 32], [0, 64], [0, 1], [0, 2], [0, 4], [0, 8]]
    elif D == 64 and BN == 64:   # no 3rd non-phase token bit below 1 KB: n8 at the 16-B slot bit, 32-B pad
        b = [[1, 0], [2, 0], [4, 0], [0, 8], [8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 1], [0, 2], [0, 4]]
        return PaddedSharedLayout([[512, 16]], b, [], [D, BN])
    else:
        raise AssertionError(("fp16 K layout", D, BN))
    return PaddedSharedLayout([[512, 8]], b, [], [D, BN])


@constexpr_function
def pick(c, a, b):
    return a if c else b


@constexpr_function
def kt16_layout(QF16, D, BN):
    """kt_smem_layout for QF16; any valid placeholder otherwise (D=128 with BLOCK_N 128 never occurs in fp16)."""
    if QF16 or (D, BN) in ((128, 64), (64, 128), (64, 64)):
        return kt_smem_layout(D, BN)
    return kt_smem_layout(64, 64)


@constexpr_function
def check_tut_kt(dl, D, BN, NW):
    if D == 128 and BN == 64 and NW == 8:
        got = dict(reg=[list(x) for x in dl.reg_bases], lane=[list(x) for x in dl.lane_bases],
                   warp=[list(x) for x in dl.warp_bases])
        assert got == TUT_KT_ASYNC, got
    return 0


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
         NO_K: gl.constexpr, NO_S: gl.constexpr, NO_C: gl.constexpr, QF16: gl.constexpr = 0):
    """ACK: direct-to-LDS copy of K tile n0 (codes, scales, coefficients; QF16: the fp16 K tile) into `slot`. The
    caller commits."""
    _dma(k_smem.index(slot), KP + kp_base + n0 * stride_kpm, kd_off, wid, NO_K, NW, KPRED)
    if not QF16:
        _dma(ks_smem.index(slot), KS + ks_base + n0 * stride_ksm, ksd_off, wid, NO_S, NW, KPRED)
    if CORR and not QF16:
        _dma(kc_smem.index(slot), KC + kc_base + n0 * stride_kcm, kcd_off, wid, NO_C, NW, KPRED)


@gluon.jit
def _lrk(k_smem, ks_smem, kc_smem, slot, k4_dot: gl.constexpr, ks_lay: gl.constexpr, cb: gl.constexpr,
         CORR: gl.constexpr, SSL: gl.constexpr = 0, NS: gl.constexpr = 4, kpar=0, QF16: gl.constexpr = 0):
    """LRK: LDS -> registers in the operand layouts (k_smem / ks_smem are the read-view aliases).
    SSL 0: scale tile read whole (D=128); 1: [BLOCK_N, NS] subslice at column 0; 2: both NS-halves, kpar selects.
    QF16: the fp16 K tile only (k4_dot = the fp16 B-operand layout); ks4 / kc are placeholders."""
    k4 = cdna4_async.load_shared_relaxed(k_smem.index(slot), k4_dot)
    if QF16:
        ks4 = k4
    elif SSL == 0:
        ks4 = cdna4_async.load_shared_relaxed(ks_smem.index(slot), ks_lay)
    elif SSL == 1:
        ks4 = cdna4_async.load_shared_relaxed(ks_smem.index(slot).slice(0, NS, dim=1), ks_lay)
    else:
        ks_lo = cdna4_async.load_shared_relaxed(ks_smem.index(slot).slice(0, NS, dim=1), ks_lay)
        ks_hi = cdna4_async.load_shared_relaxed(ks_smem.index(slot).slice(NS, NS, dim=1), ks_lay)
        ks4 = gl.where(kpar, ks_hi, ks_lo)
    if CORR and not QF16:
        kc = cdna4_async.load_shared_relaxed(kc_smem.index(slot), cb)
    else:
        kc = ks4
    return k4, ks4, kc


@gluon.jit
def _qkx(q4, qs4, k4, ks4, qcm, kc, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, qk4: gl.constexpr,
         mr2: gl.constexpr, mma_layout: gl.constexpr, CORR: gl.constexpr, ks_lay: gl.constexpr, QF16: gl.constexpr):
    """dot_qk: QF16 1 = fp16 MFMA 32x32x16 (q4 / k4 are the fp16 operands); else fmha_roof._qk (fp4 [+ rank-2])."""
    if QF16:
        s = mfma_cdna4(q4, k4, gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=mma_layout))
    else:
        s = _qk(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, False)
    return s


@gluon.jit
def _sv1(qk, m_run, qk_scale: gl.constexpr, SQ: gl.constexpr):
    """fmha_roof.sc_vec1 (ABL 0); SQ 1: scale already in Q (the tutorial's SCALE_ON_Q branch: no scale multiply,
    plain subtract)."""
    if SQ:
        m_ij = nan_max(qk, axis=1)
    else:
        m_ij = nan_max(qk, axis=1) * qk_scale
    m_new = gl.maximum(m_run, m_ij, propagate_nan=tl.PropagateNan.ALL)
    if SQ:
        p = gl.exp2(qk - m_new[:, None])
    else:
        p = gl.exp2(gl.fma(qk, qk_scale, -m_new[:, None]))
    alpha = gl.exp2(m_run - m_new)
    return m_new, p, alpha


@gluon.jit
def _sv1_lazy(qk, m_run, qk_scale: gl.constexpr, TH: gl.constexpr, SQ: gl.constexpr):
    """fmha_roof_lds._vec1_lazy; SQ as _sv1."""
    if SQ:
        m_ij = nan_max(qk, axis=1)
    else:
        m_ij = nan_max(qk, axis=1) * qk_scale
    m_cand = gl.maximum(m_run, m_ij, propagate_nan=tl.PropagateNan.ALL)
    m_new = gl.where(m_cand - m_run > TH, m_cand, m_run)
    if SQ:
        p = gl.exp2(qk - m_new[:, None])
    else:
        p = gl.exp2(gl.fma(qk, qk_scale, -m_new[:, None]))
    alpha = gl.exp2(m_run - m_new)
    return m_new, p, alpha


@gluon.jit
def _v1t(qk, m_run, qk_scale: gl.constexpr, LAZY: gl.constexpr, LTH: gl.constexpr, SQ: gl.constexpr = 0):
    """fmha_roof_lds._v1 for V1X=0, ABL=0, with the lazy-max threshold LTH (log2 units) as a knob (LTH 8 = fmha_v4).
    SQ 0: the same ops as fmha_roof.sc_vec1 / fmha_roof_lds._vec1_lazy (local copies with the SQ branch)."""
    if LAZY:
        m_new, p, alpha = _sv1_lazy(qk, m_run, qk_scale, LTH, SQ)
    else:
        m_new, p, alpha = _sv1(qk, m_run, qk_scale, SQ)
    return m_new, p, alpha, alpha


FENCE_ASM = gl.constexpr("; sched fence")
FENCE_CONS = gl.constexpr("=v,0,~{exec}")


@gluon.jit
def _fence(x):
    """Scheduling-region boundary: an empty inline asm that 'clobbers' EXEC (SIInstrInfo::isSchedulingBoundary), so
    the machine scheduler cannot move instructions across it. Emits no instruction (only the asm comment). x is any
    tensor with one element per lane (returned unchanged, tied operand)."""
    return gl.inline_asm_elementwise(FENCE_ASM, FENCE_CONS, [x], dtype=gl.float32, is_pure=False, pack=1)


@gluon.jit
def _v1a(qk, m_run, qk_scale: gl.constexpr, LAZY: gl.constexpr, LTH: gl.constexpr, VF: gl.constexpr,
         SQ: gl.constexpr = 0):
    """VEC1 head (before the fence): row max (+ the fma for VF 1; SQ: subtract)."""
    if SQ:
        m_ij = nan_max(qk, axis=1)
    else:
        m_ij = nan_max(qk, axis=1) * qk_scale
    m_cand = gl.maximum(m_run, m_ij, propagate_nan=tl.PropagateNan.ALL)
    if LAZY:
        m_new = gl.where(m_cand - m_run > LTH, m_cand, m_run)
    else:
        m_new = m_cand
    if VF == 1:
        if SQ:
            x = qk - m_new[:, None]
        else:
            x = gl.fma(qk, qk_scale, -m_new[:, None])
    else:
        x = qk
    return m_new, x


@gluon.jit
def _v1b(x, m_run, m_new, qk_scale: gl.constexpr, VF: gl.constexpr, SQ: gl.constexpr = 0):
    """VEC1 tail (after the fence): [fma] + exp2 + alpha."""
    if VF == 2:
        if SQ:
            x = x - m_new[:, None]
        else:
            x = gl.fma(x, qk_scale, -m_new[:, None])
    p = gl.exp2(x)
    alpha = gl.exp2(m_run - m_new)
    return p, alpha


@gluon.jit
def _split2(x):
    half: gl.constexpr = x.shape[1] // 2
    return amd_slice(x, [x.shape[0], half], [0, 0]), amd_slice(x, [x.shape[0], half], [0, half])


@gluon.jit
def _cat2(x, y):
    layout: gl.constexpr = x.type.layout
    shape: gl.constexpr = [x.shape[0], x.shape[1] + y.shape[1]]
    a = gl.join(x, y).permute(0, 2, 1).reshape(shape)
    return gl.convert_layout(a, layout, assert_trivial=True)


@gluon.jit
def _split8(x):
    a, b = _split2(x)
    a0, a1 = _split2(a)
    b0, b1 = _split2(b)
    s0, s1 = _split2(a0)
    s2, s3 = _split2(a1)
    s4, s5 = _split2(b0)
    s6, s7 = _split2(b1)
    return s0, s1, s2, s3, s4, s5, s6, s7


@gluon.jit
def _cat8(s0, s1, s2, s3, s4, s5, s6, s7):
    return _cat2(_cat2(_cat2(s0, s1), _cat2(s2, s3)), _cat2(_cat2(s4, s5), _cat2(s6, s7)))


@gluon.jit
def _e2(x, DO: gl.constexpr):
    if DO:
        return gl.exp2(x)
    else:
        return x


@gluon.jit
def _v1b_x(x, m_run, m_new, EXS: gl.constexpr):
    """VEC1 tail for VF 1 + EXS e (fmha_v4 balancing on top of the fence): x already holds s*scale - m; exp2 of the
    first 8-e of 8 column slices here (dot2, after the fence), the last e slices stay as x for VEC2 (dot1)."""
    s0, s1, s2, s3, s4, s5, s6, s7 = _split8(x)
    p = _cat8(_e2(s0, EXS < 8), _e2(s1, EXS < 7), _e2(s2, EXS < 6), _e2(s3, EXS < 5),
              _e2(s4, EXS < 4), _e2(s5, EXS < 3), _e2(s6, EXS < 2), _e2(s7, EXS < 1))
    alpha = gl.exp2(m_run - m_new)
    return p, alpha


@gluon.jit
def _v2x(acc, l_i, pc, alpha, p_dot_layout: gl.constexpr, RESCALE: gl.constexpr, EXS: gl.constexpr):
    """VEC2 (dot1) for VF 1 + EXS e: [rescale] + exp2 of the last e slices + row sum + fp16 operand."""
    if RESCALE:
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
    s0, s1, s2, s3, s4, s5, s6, s7 = _split8(pc)
    p = _cat8(_e2(s0, EXS >= 8), _e2(s1, EXS >= 7), _e2(s2, EXS >= 6), _e2(s3, EXS >= 5),
              _e2(s4, EXS >= 4), _e2(s5, EXS >= 3), _e2(s6, EXS >= 2), _e2(s7, EXS >= 1))
    l_i = l_i + gl.sum(p, axis=1)
    p_dot = gl.convert_layout(p.to(gl.float16), p_dot_layout)
    return acc, l_i, p_dot


@gluon.jit
def _r2(qcm, kc, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, mr2: gl.constexpr, qk4: gl.constexpr):
    """Rank-2 term alone (one fp16 32x32xPW MFMA per 32x32 S block), in the fp4 accumulator layout (srcC of dot1)."""
    return gl.convert_layout(mfma_cdna4(qcm, kc, gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=mr2)), qk4)


@gluon.jit
def _qk4(q4, qs4, k4, ks4, c0, mma_layout: gl.constexpr):
    """fp4 QK^T accumulated onto a precomputed rank-2 block c0."""
    s = gl.amd.cdna4.mfma_scaled(q4, qs4, "e2m1", k4, ks4, "e2m1", c0)
    return gl.convert_layout(s, mma_layout)


@gluon.jit
def _lrkc(kc_smem, slot, cb: gl.constexpr):
    return cdna4_async.load_shared_relaxed(kc_smem.index(slot), cb)


@gluon.jit
def _lrk_nc(k_smem, ks_smem, slot, k4_dot: gl.constexpr, ks_lay: gl.constexpr, SSL: gl.constexpr, NS: gl.constexpr,
            kpar):
    """LRK without the coefficients (R2P 2/3 read them in mem1)."""
    k4 = cdna4_async.load_shared_relaxed(k_smem.index(slot), k4_dot)
    if SSL == 0:
        ks4 = cdna4_async.load_shared_relaxed(ks_smem.index(slot), ks_lay)
    elif SSL == 1:
        ks4 = cdna4_async.load_shared_relaxed(ks_smem.index(slot).slice(0, NS, dim=1), ks_lay)
    else:
        ks_lo = cdna4_async.load_shared_relaxed(ks_smem.index(slot).slice(0, NS, dim=1), ks_lay)
        ks_hi = cdna4_async.load_shared_relaxed(ks_smem.index(slot).slice(NS, NS, dim=1), ks_lay)
        ks4 = gl.where(kpar, ks_hi, ks_lo)
    return k4, ks4


@gluon.jit
def _lrk_cf(k_smem, ks_smem, kc_smem, slot, k4_dot: gl.constexpr, ks_lay: gl.constexpr, cb: gl.constexpr,
            SSL: gl.constexpr, NS: gl.constexpr, kpar):
    """LRK with the coefficients read FIRST (LDS returns in order: the rank-2 MFMA waits only for them)."""
    kc = cdna4_async.load_shared_relaxed(kc_smem.index(slot), cb)
    k4, ks4 = _lrk_nc(k_smem, ks_smem, slot, k4_dot, ks_lay, SSL, NS, kpar)
    return k4, ks4, kc


@gluon.jit
def mx_attn_fwd_d64(QP, QS, QC, KP, KS, KC, V, SM_SCALE: gl.constexpr, L, Out,
                   stride_qpm, stride_qph, stride_qsm, stride_qsh, stride_qcm, stride_qch,
                   stride_kpm, stride_kph, stride_ksm, stride_ksh, stride_kcm, stride_kch,
                   stride_vk, stride_vh, stride_om, stride_oh, stride_ksh4, stride_kcm8,
                   HQ: gl.constexpr, HK: gl.constexpr, N_CTX: gl.constexpr,
                   BLOCK_M: gl.constexpr, BLOCK_DMODEL: gl.constexpr, BLOCK_N: gl.constexpr, CORR: gl.constexpr,
                   RESC_MEM: gl.constexpr, PRIO: gl.constexpr, ENVTAG: gl.constexpr, ABL: gl.constexpr,
                   V1X: gl.constexpr, LAZY: gl.constexpr,
                   KSL: gl.constexpr, KCF: gl.constexpr, KCL: gl.constexpr, WQ1: gl.constexpr, WQ2: gl.constexpr,
                   KPRED: gl.constexpr, LRKE: gl.constexpr, KSA: gl.constexpr,
                   R2P_: gl.constexpr, PW: gl.constexpr, ABLC: gl.constexpr,
                   LTH: gl.constexpr, VF: gl.constexpr, RV: gl.constexpr, D1F: gl.constexpr,
                   EXS: gl.constexpr = 0, QF16: gl.constexpr = 0, SQ: gl.constexpr = 0,
                   DRF: gl.constexpr = 0):
    num_warps: gl.constexpr = gl.num_warps()
    gl.static_assert(QF16 == 0 or (CORR == 0 and KSL == 0 and KCF == 0 and KCL == 0), "QF16: fp4-only knobs off")
    gl.static_assert(SQ == 0 or QF16 == 1, "SQ (scale folded into Q) is fp16-only")
    gl.static_assert(EXS == 0 or (VF == 1 and not D1F and V1X == 0), "EXS: VF 1, no D1F, V1X 0")
    gl.static_assert(not D1F or (LAZY and V1X == 0), "D1F: lazy rescale (in mem2) only")
    KB: gl.constexpr = BLOCK_DMODEL // 2
    NS: gl.constexpr = BLOCK_DMODEL // 32
    NSP: gl.constexpr = 4                                  # LDS scale bytes per token (one DMA dword)
    ORIG: gl.constexpr = BLOCK_DMODEL == 128 and BLOCK_N == 64 and BLOCK_M == 32 * num_warps   # fmha_kbank path
    SSL: gl.constexpr = 0 if NS == NSP else (2 if KSA else 1)
    gl.static_assert(NS == NSP or not KSL, "KSL is a D=128 knob")
    R2P: gl.constexpr = R2P_ if CORR else 0     # Hadamard: no rank-2 term -> identical code for every R2P
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
    k4f_dot: gl.constexpr = DotOperandLayout(operand_index=1, parent=qk4, k_width=16)
    ks_blk: gl.constexpr = gl.BlockedLayout([1, NS], [64, 1], [num_warps, 1], [1, 0])
    qs_lay: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(q4_dot, [BLOCK_M, NS])
    ks_lay: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(k4f_dot, [BLOCK_N, NS])
    mr2: gl.constexpr = AMDMFMALayout(version=4, instr_shape=[32, 32, PW], transposed=True, warps_per_cta=[num_warps, 1])
    ca: gl.constexpr = DotOperandLayout(operand_index=0, parent=mr2, k_width=PW // 2)
    cb: gl.constexpr = DotOperandLayout(operand_index=1, parent=mr2, k_width=PW // 2)

    # ---- K-side LDS layouts: read views (operand side) and DMA views (global side), aliases of the same bytes ----
    k4_lin: gl.constexpr = gl.to_linear_layout(k4f_dot, [KB, BLOCK_N])
    if KCL:
        kf_rd: gl.constexpr = kswz_smem(k4_lin, KB, BLOCK_N)
    else:
        kf_rd: gl.constexpr = lane_major_smem(k4_lin, 4, 16)                    # L: 16 B per lane chunk
    if KSL:
        kf_dv: gl.constexpr = swap_coord_bits(kf_rd, 0, 4, 5)                     # logical block kappa <- phys swap
    else:
        kf_dv: gl.constexpr = kf_rd
    kdf_layout: gl.constexpr = dma_layout(kf_dv, 4, num_warps)                     # dwordx4 per lane
    ks_dv: gl.constexpr = PaddedSharedLayout.with_identity_for([[BLOCK_N * NSP, 4]], [BLOCK_N, NSP], [1, 0])
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
    NOF_K: gl.constexpr = n_owner_warps(kdf_layout)
    NO_S: gl.constexpr = n_owner_warps(ksd_layout)
    NO_C: gl.constexpr = n_owner_warps(kcd_layout)
    # fp16 QK^T operands (QF16; the tutorial's: 32x32x16, k_width 8) and the fp16 K LDS / DMA layouts; constexprs
    # cannot be reassigned, so both sets are built and pick() selects (pure compile-time, no code for the other one)
    q16_dot: gl.constexpr = DotOperandLayout(operand_index=0, parent=mma_layout, k_width=8)
    k16_dot: gl.constexpr = DotOperandLayout(operand_index=1, parent=mma_layout, k_width=8)
    k16_dv: gl.constexpr = kt16_layout(QF16, BLOCK_DMODEL, BLOCK_N)
    kd16_layout: gl.constexpr = dma_layout(k16_dv, 3, num_warps)                 # dwordx4 (8 halves) per lane
    _ck: gl.constexpr = check_tut_kt(kd16_layout, BLOCK_DMODEL, BLOCK_N, num_warps)
    k4_dot: gl.constexpr = pick(QF16, k16_dot, k4f_dot)
    k_dv: gl.constexpr = pick(QF16, k16_dv, kf_dv)
    k_rd: gl.constexpr = pick(QF16, k16_dv, kf_rd)
    kd_layout: gl.constexpr = pick(QF16, kd16_layout, kdf_layout)
    NO_K: gl.constexpr = n_owner_warps(kd_layout)
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

    blocked_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 8],
                                                    threads_per_warp=[64 // (BLOCK_DMODEL // 8), BLOCK_DMODEL // 8],
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
    if QF16:   # fp16 Q straight into the A-operand layout (16 B per lane per row chunk); SQ: scale folded in fp32
        qm16 = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, q16_dot)) + m0
        qd16 = gl.arange(0, BLOCK_DMODEL, layout=gl.SliceLayout(0, q16_dot))
        q4 = gl.load(QP + off_h_q * stride_qph + qm16[:, None] * stride_qpm + qd16[None, :])
        if SQ:
            q4 = (q4.to(gl.float32) * qk_scale).to(gl.float16)
        qs4 = q4
    else:
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
    if NS == NSP:
        ks_base = (off_h_k * stride_ksh4) * 4
        kpar = 0
    elif KSA:   # aligned dword {heads 2j, 2j+1}; parity picks the half
        ks_base = ((off_h_k * stride_ksh) // 4) * 4
        kpar = ((off_h_k * stride_ksh) % 4) != 0
    else:       # the head's own dword (2-B aligned for odd heads; alignment asserted)
        ks_base = gl.multiple_of(off_h_k * stride_ksh, 4)
        kpar = 0
    kc_base = off_h_k * stride_kch
    if QF16:
        kd_k = gl.arange(0, BLOCK_DMODEL, layout=gl.SliceLayout(1, kd_layout))
    else:
        kd_k = gl.arange(0, KB, layout=gl.SliceLayout(1, kd_layout))
    kd_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, kd_layout))
    kd_off = kd_n[None, :] * stride_kpm + kd_k[:, None]
    ksd_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, ksd_layout))
    ksd_s = gl.arange(0, NSP, layout=gl.SliceLayout(0, ksd_layout))
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

    if ORIG:
        v_offset_bases: gl.constexpr = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
                                        [16, 0], [32, 0], [1, 0], [2, 0], [4, 0], [8, 0]]
    else:
        v_offset_bases: gl.constexpr = v_smem_bases(BLOCK_DMODEL, BLOCK_N)
    v_async_smem_layout: gl.constexpr = PaddedSharedLayout(interval_padding_pairs=[[512, 32]],
                                                           offset_bases=v_offset_bases, cga_layout=[],
                                                           shape=[BLOCK_N, BLOCK_DMODEL])
    if not ORIG:
        v_async_layout: gl.constexpr = dma_layout(v_async_smem_layout, 3, num_warps)
    elif num_warps == 8:
        v_async_layout: gl.constexpr = DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
            lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
            warp_bases=[[1, 0], [2, 0], [4, 0]], block_bases=[], shape=[BLOCK_N, BLOCK_DMODEL])
    else:
        v_async_layout: gl.constexpr = DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0], [4, 0]],
            lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
            warp_bases=[[1, 0], [2, 0]], block_bases=[], shape=[BLOCK_N, BLOCK_DMODEL])
    BUF_DEPTH: gl.constexpr = 2
    v_smem = gl.allocate_shared_memory(V.dtype.element_ty, [BUF_DEPTH, BLOCK_N, BLOCK_DMODEL], layout=v_async_smem_layout)
    if QF16:   # fp16 K tiles only; ks_smem / kc_smem are never touched (aliases keep the helper signatures)
        k_smem = gl.allocate_shared_memory(KP.dtype.element_ty, [BUF_DEPTH, BLOCK_DMODEL, BLOCK_N], layout=k_dv)
        ks_smem = k_smem
        kc_smem = k_smem
    else:
        k_smem = gl.allocate_shared_memory(KP.dtype.element_ty, [BUF_DEPTH, KB, BLOCK_N], layout=k_dv)
        ks_smem = gl.allocate_shared_memory(KS.dtype.element_ty, [BUF_DEPTH, BLOCK_N, NSP], layout=ks_dv)
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
         kd_off, ksd_off, kcd_off, wid, CORR, False, num_warps, NO_K, NO_S, NO_C, QF16=QF16)
    cdna4_async.commit_group()                     # ACK[0]
    cdna4_async.buffer_load_to_shared(v_smem.index(0), v_base, v_off)
    cdna4_async.commit_group()                     # ACV[0]
    _ack(k_smem, ks_smem, kc_smem, 1, KP, KS, KC, kp_base, ks_base, kc_base, BLOCK_N, stride_kpm, stride_ksm,
         stride_kcm, kd_off, ksd_off, kcd_off, wid, CORR, False, num_warps, NO_K, NO_S, NO_C, QF16=QF16)
    cdna4_async.commit_group()                     # ACK[1]
    cdna4_async.wait_group(2)                      # K[0]
    k4, ks4, kc = _lrk(k_smem_r, ks_smem_r, kc_smem, 0, k4_dot, ks_lay, cb, CORR, SSL, NS, kpar, QF16=QF16)
    qk = _qkx(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, QF16)
    if EXS > 0:   # same partial form as the loop's dot2: the loop's first dot1 finishes the last EXS slices
        m_new0, xq0 = _v1a(qk, m_i, qk_scale, LAZY, LTH, 1, SQ)
        p_c, alpha_c = _v1b_x(xq0, m_i, m_new0, EXS)
        l_c = alpha_c
        m_run = m_new0
    else:
        m_run, p_c, alpha_c, l_c = _v1t(qk, m_i, qk_scale, LAZY, LTH, SQ)
    if RESC_MEM:
        acc, l_i = _resc(acc, l_i, alpha_c, ABL)
    gl.barrier()                                   # WAR: LRK[0] vs K[2]
    _ack(k_smem, ks_smem, kc_smem, 0, KP, KS, KC, kp_base, ks_base, kc_base, 2 * BLOCK_N, stride_kpm, stride_ksm,
         stride_kcm, kd_off, ksd_off, kcd_off, wid, CORR, False, num_warps, NO_K, NO_S, NO_C, QF16=QF16)
    cdna4_async.commit_group()                     # ACK[2]
    cdna4_async.wait_group(1)                      # K[1]
    k4, ks4, kc = _lrk(k_smem_r, ks_smem_r, kc_smem, 1, k4_dot, ks_lay, cb, CORR, SSL, NS, kpar, QF16=QF16)
    if R2P != 0:
        c0 = _r2(qcm, kc, BLOCK_M, BLOCK_N, mr2, qk4)                    # rank-2 of tile 1 (srcC of the loop's first dot1)
    else:
        c0 = qk                                                          # unused placeholder
    cdna4_async.buffer_load_to_shared(v_smem.index(1), v_base + v_step, v_off)
    cdna4_async.commit_group()                     # ACV[1]

    main_loop_pairs = (block_end - 3 - block_start) // 2
    for pair_idx in tl.range(0, main_loop_pairs):
        block_n = block_start + pair_idx * 2
        # even (block_n) tile: LDS slots cur=0, next=1
        with warp_pipeline_stage(L_D1, priority=DP):
            if D1F:   # [rank-2 (R2P 0), row sum] | fence | [fp4 QK^T, fp16 cast]  (lazy / RESC_MEM only)
                if R2P == 0 and CORR:
                    c0 = _r2(qcm, kc, BLOCK_M, BLOCK_N, mr2, qk4)
                l_i = l_i + gl.sum(p_c, axis=1)
                l_i = _fence(l_i)
                if CORR:
                    qk = _qk4(q4, qs4, k4, ks4, c0, mma_layout)
                else:
                    qk = _qkx(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, QF16)
                p_dot = gl.convert_layout(p_c.to(gl.float16), p_dot_layout)
            else:
                if R2P == 0:
                    qk = _qkx(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, QF16)
                else:
                    qk = _qk4(q4, qs4, k4, ks4, c0, mma_layout)
                if EXS > 0:
                    acc, l_i, p_dot = _v2x(acc, l_i, p_c, alpha_c, p_dot_layout, VEC2_RESC, EXS)
                else:
                    acc, l_i, p_dot = _v2(acc, l_i, p_c, alpha_c, l_c, p_dot_layout, VEC2_RESC, ABL, V1X)
        cdna4_async.wait_group(WQ1)
        with warp_pipeline_stage(L_M1, priority=MP):
            if R2P >= 2:
                kc = _lrkc(kc_smem, 0, cb)                                                   # coeffs first
            v_dot = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)                  # LRV
            if LRKE:
                if R2P >= 2:
                    k4, ks4 = _lrk_nc(k_smem_r, ks_smem_r, 0, k4_dot, ks_lay, SSL, NS, kpar)
                else:
                    k4, ks4, kc = _lrk(k_smem_r, ks_smem_r, kc_smem, 0, k4_dot, ks_lay, cb, CORR, SSL, NS, kpar, QF16=QF16)
            _ack(k_smem, ks_smem, kc_smem, 1, KP, KS, KC, kp_base, ks_base, kc_base, (block_n + 3) * BLOCK_N,
                 stride_kpm, stride_ksm, stride_kcm, kd_off, ksd_off, kcd_off, wid, CORR and not (ABLC & 1), KPRED, num_warps,
                 NO_K, NO_S, NO_C, QF16=QF16)                                                                    # ACK
            cdna4_async.commit_group()
        with warp_pipeline_stage(L_D2, priority=DP):
            if VF == 0:
                acc = mfma_cdna4(p_dot, v_dot, acc)
                m_run, p_c, alpha_c, l_c = _v1t(qk, m_run, qk_scale, LAZY, LTH, SQ)
                if R2P == 3:
                    if ABLC & 2:   # timing ablation: no in-loop rank-2 MFMA (wrong results)
                        c0 = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=qk4)
                    else:
                        c0 = _r2(qcm, kc, BLOCK_M, BLOCK_N, mr2, qk4)                           # rank-2 in dot2
            else:   # VF 1: [max, fma] | fence | [P.V, exp2];  VF 2: [max] | fence | [P.V, fma, exp2]
                #       rank-2 (R2P 3): RV 0 before the fence (head), RV 1 after it (with P.V / exp2; tail)
                if R2P == 3 and RV == 0:
                    c0 = _r2(qcm, kc, BLOCK_M, BLOCK_N, mr2, qk4)
                m_new, xq = _v1a(qk, m_run, qk_scale, LAZY, LTH, VF, SQ)
                l_i = _fence(l_i)
                acc = mfma_cdna4(p_dot, v_dot, acc)
                if EXS > 0:
                    p_c, alpha_c = _v1b_x(xq, m_run, m_new, EXS)
                else:
                    p_c, alpha_c = _v1b(xq, m_run, m_new, qk_scale, VF, SQ)
                if R2P == 3 and RV == 1:
                    c0 = _r2(qcm, kc, BLOCK_M, BLOCK_N, mr2, qk4)
                l_c = alpha_c
                m_run = m_new
        cdna4_async.wait_group(WQ2)
        with warp_pipeline_stage(L_M2, priority=MP):
            if not LRKE:
                if R2P >= 2:
                    k4, ks4 = _lrk_nc(k_smem_r, ks_smem_r, 0, k4_dot, ks_lay, SSL, NS, kpar)
                elif R2P == 1:
                    k4, ks4, kc = _lrk_cf(k_smem_r, ks_smem_r, kc_smem, 0, k4_dot, ks_lay, cb, SSL, NS, kpar)
                else:
                    k4, ks4, kc = _lrk(k_smem_r, ks_smem_r, kc_smem, 0, k4_dot, ks_lay, cb, CORR, SSL, NS, kpar, QF16=QF16)  # LRK
            if R2P == 2:
                c0 = _r2(qcm, kc, BLOCK_M, BLOCK_N, mr2, qk4)                                   # rank-2, mem2 head
            cdna4_async.buffer_load_to_shared(v_smem.index(0), v_base + (block_n + 2) * v_step, v_off)   # ACV
            cdna4_async.commit_group()
            if R2P == 1:
                c0 = _r2(qcm, kc, BLOCK_M, BLOCK_N, mr2, qk4)                                   # rank-2, mem2 tail
            if LAZY:
                acc, l_i = _resc_lazy(acc, l_i, alpha_c)
            elif RESC_MEM:
                acc, l_i = _resc(acc, l_i, alpha_c, ABL)
        # odd (block_n + 1) tile: LDS slots cur=1, next=0
        with warp_pipeline_stage(L_D1, priority=DP):
            if D1F:   # [rank-2 (R2P 0), row sum] | fence | [fp4 QK^T, fp16 cast]  (lazy / RESC_MEM only)
                if R2P == 0 and CORR:
                    c0 = _r2(qcm, kc, BLOCK_M, BLOCK_N, mr2, qk4)
                l_i = l_i + gl.sum(p_c, axis=1)
                l_i = _fence(l_i)
                if CORR:
                    qk = _qk4(q4, qs4, k4, ks4, c0, mma_layout)
                else:
                    qk = _qkx(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, QF16)
                p_dot = gl.convert_layout(p_c.to(gl.float16), p_dot_layout)
            else:
                if R2P == 0:
                    qk = _qkx(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, QF16)
                else:
                    qk = _qk4(q4, qs4, k4, ks4, c0, mma_layout)
                if EXS > 0:
                    acc, l_i, p_dot = _v2x(acc, l_i, p_c, alpha_c, p_dot_layout, VEC2_RESC, EXS)
                else:
                    acc, l_i, p_dot = _v2(acc, l_i, p_c, alpha_c, l_c, p_dot_layout, VEC2_RESC, ABL, V1X)
        cdna4_async.wait_group(WQ1)
        with warp_pipeline_stage(L_M1, priority=MP):
            if R2P >= 2:
                kc = _lrkc(kc_smem, 1, cb)                                                   # coeffs first
            v_dot = cdna4_async.load_shared_relaxed(v_smem.index(1), v_dot_layout)                  # LRV
            if LRKE:
                if R2P >= 2:
                    k4, ks4 = _lrk_nc(k_smem_r, ks_smem_r, 1, k4_dot, ks_lay, SSL, NS, kpar)
                else:
                    k4, ks4, kc = _lrk(k_smem_r, ks_smem_r, kc_smem, 1, k4_dot, ks_lay, cb, CORR, SSL, NS, kpar, QF16=QF16)
            _ack(k_smem, ks_smem, kc_smem, 0, KP, KS, KC, kp_base, ks_base, kc_base, (block_n + 4) * BLOCK_N,
                 stride_kpm, stride_ksm, stride_kcm, kd_off, ksd_off, kcd_off, wid, CORR and not (ABLC & 1), KPRED, num_warps,
                 NO_K, NO_S, NO_C, QF16=QF16)                                                                    # ACK
            cdna4_async.commit_group()
        with warp_pipeline_stage(L_D2, priority=DP):
            if VF == 0:
                acc = mfma_cdna4(p_dot, v_dot, acc)
                m_run, p_c, alpha_c, l_c = _v1t(qk, m_run, qk_scale, LAZY, LTH, SQ)
                if R2P == 3:
                    if ABLC & 2:   # timing ablation: no in-loop rank-2 MFMA (wrong results)
                        c0 = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=qk4)
                    else:
                        c0 = _r2(qcm, kc, BLOCK_M, BLOCK_N, mr2, qk4)                           # rank-2 in dot2
            else:   # VF 1: [max, fma] | fence | [P.V, exp2];  VF 2: [max] | fence | [P.V, fma, exp2]
                #       rank-2 (R2P 3): RV 0 before the fence (head), RV 1 after it (with P.V / exp2; tail)
                if R2P == 3 and RV == 0:
                    c0 = _r2(qcm, kc, BLOCK_M, BLOCK_N, mr2, qk4)
                m_new, xq = _v1a(qk, m_run, qk_scale, LAZY, LTH, VF, SQ)
                l_i = _fence(l_i)
                acc = mfma_cdna4(p_dot, v_dot, acc)
                if EXS > 0:
                    p_c, alpha_c = _v1b_x(xq, m_run, m_new, EXS)
                else:
                    p_c, alpha_c = _v1b(xq, m_run, m_new, qk_scale, VF, SQ)
                if R2P == 3 and RV == 1:
                    c0 = _r2(qcm, kc, BLOCK_M, BLOCK_N, mr2, qk4)
                l_c = alpha_c
                m_run = m_new
        cdna4_async.wait_group(WQ2)
        with warp_pipeline_stage(L_M2, priority=MP):
            if not LRKE:
                if R2P >= 2:
                    k4, ks4 = _lrk_nc(k_smem_r, ks_smem_r, 1, k4_dot, ks_lay, SSL, NS, kpar)
                elif R2P == 1:
                    k4, ks4, kc = _lrk_cf(k_smem_r, ks_smem_r, kc_smem, 1, k4_dot, ks_lay, cb, SSL, NS, kpar)
                else:
                    k4, ks4, kc = _lrk(k_smem_r, ks_smem_r, kc_smem, 1, k4_dot, ks_lay, cb, CORR, SSL, NS, kpar, QF16=QF16)  # LRK
            if R2P == 2:
                c0 = _r2(qcm, kc, BLOCK_M, BLOCK_N, mr2, qk4)                                   # rank-2, mem2 head
            cdna4_async.buffer_load_to_shared(v_smem.index(1), v_base + (block_n + 3) * v_step, v_off)   # ACV
            cdna4_async.commit_group()
            if R2P == 1:
                c0 = _r2(qcm, kc, BLOCK_M, BLOCK_N, mr2, qk4)                                   # rank-2, mem2 tail
            if LAZY:
                acc, l_i = _resc_lazy(acc, l_i, alpha_c)
            elif RESC_MEM:
                acc, l_i = _resc(acc, l_i, alpha_c, ABL)

    if ODD_TAIL:
        block_n = block_start + main_loop_pairs * 2
        with warp_pipeline_stage(L_D1, priority=DP):
            if R2P == 0:
                qk = _qkx(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, QF16)
            else:
                qk = _qk4(q4, qs4, k4, ks4, c0, mma_layout)
            if EXS > 0:   # p_c comes from the loop's last dot2 (EXS slices still raw)
                acc, l_i, p_dot = _v2x(acc, l_i, p_c, alpha_c, p_dot_layout, VEC2_RESC, EXS)
            else:
                acc, l_i, p_dot = _v2(acc, l_i, p_c, alpha_c, l_c, p_dot_layout, VEC2_RESC, ABL, V1X)
        cdna4_async.wait_group(1)
        with warp_pipeline_stage(L_M1, priority=MP):
            v_dot = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)
            _ack(k_smem, ks_smem, kc_smem, 1, KP, KS, KC, kp_base, ks_base, kc_base, (block_n + 3) * BLOCK_N,
                 stride_kpm, stride_ksm, stride_kcm, kd_off, ksd_off, kcd_off, wid, CORR, False, num_warps,
                 NO_K, NO_S, NO_C, QF16=QF16)
            cdna4_async.commit_group()
        with warp_pipeline_stage(L_D2, priority=DP):
            acc = mfma_cdna4(p_dot, v_dot, acc)
            m_run, p_c, alpha_c, l_c = _v1t(qk, m_run, qk_scale, LAZY, LTH, SQ)
        cdna4_async.wait_group(1)
        with warp_pipeline_stage(L_M2, priority=MP):
            k4, ks4, kc = _lrk(k_smem_r, ks_smem_r, kc_smem, 0, k4_dot, ks_lay, cb, CORR, SSL, NS, kpar, QF16=QF16)
            if R2P != 0:
                c0 = _r2(qcm, kc, BLOCK_M, BLOCK_N, mr2, qk4)
            cdna4_async.buffer_load_to_shared(v_smem.index(0), v_base + (block_n + 2) * v_step, v_off)
            cdna4_async.commit_group()
            if RESC_MEM:
                acc, l_i = _resc(acc, l_i, alpha_c, ABL)

    # -- Drain (the tutorial's): pending {V[n-3], K[n-1], V[n-2]}; K regs hold tile n-2 (c0: its rank-2 if R2P) ----
    nm3 = block_end - 3
    nm2 = block_end - 2
    nm1 = block_end - 1
    s_nm3 = ((nm3 - block_start) % BUF_DEPTH).to(tl.int32)
    s_nm2 = ((nm2 - block_start) % BUF_DEPTH).to(tl.int32)
    s_nm1 = ((nm1 - block_start) % BUF_DEPTH).to(tl.int32)
    if R2P == 0:
        qk = _qkx(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, QF16)   # qk[n-2]
    else:
        qk = _qk4(q4, qs4, k4, ks4, c0, mma_layout)
    if DRF:   # drain fences: scheduling-region boundaries between the drain steps (register pressure)
        l_i = _fence(l_i)

    cdna4_async.wait_group(2)                                                                # V[n-3]
    v_dot = cdna4_async.load_shared_relaxed(v_smem.index(s_nm3), v_dot_layout)
    if EXS > 0 and not ODD_TAIL:   # p_c from the loop's last dot2 (EXS slices still raw); ODD_TAIL's _v1t is full
        acc, l_i, p_dot = _v2x(acc, l_i, p_c, alpha_c, p_dot_layout, VEC2_RESC, EXS)
    else:
        acc, l_i, p_dot = _v2(acc, l_i, p_c, alpha_c, l_c, p_dot_layout, VEC2_RESC, ABL, V1X)
    acc = mfma_cdna4(p_dot, v_dot, acc)
    m_run, p_c, alpha_c, l_c = _v1t(qk, m_run, qk_scale, LAZY, LTH, SQ)
    if RESC_MEM:
        acc, l_i = _resc(acc, l_i, alpha_c, ABL)
    if DRF:
        l_i = _fence(l_i)
    gl.barrier()                                                                             # WAR: LRV[n-3] vs V[n-1]
    cdna4_async.buffer_load_to_shared(v_smem.index(s_nm1), v_base + nm1 * v_step, v_off)
    cdna4_async.commit_group()                                                               # ACV[n-1]
    cdna4_async.wait_group(2)                                                                # K[n-1]
    k4, ks4, kc = _lrk(k_smem_r, ks_smem_r, kc_smem, s_nm1, k4_dot, ks_lay, cb, CORR, SSL, NS, kpar, QF16=QF16)
    qk = _qkx(q4, qs4, k4, ks4, qcm, kc, BLOCK_M, BLOCK_N, qk4, mr2, mma_layout, CORR, ks_lay, QF16)   # qk[n-1]
    if DRF:
        l_i = _fence(l_i)
    cdna4_async.wait_group(1)                                                                # V[n-2]
    v_dot = cdna4_async.load_shared_relaxed(v_smem.index(s_nm2), v_dot_layout)
    acc, l_i, p_dot = _v2(acc, l_i, p_c, alpha_c, l_c, p_dot_layout, VEC2_RESC, ABL, V1X)
    acc = mfma_cdna4(p_dot, v_dot, acc)
    m_run, p_c, alpha_c, l_c = _v1t(qk, m_run, qk_scale, LAZY, LTH, SQ)
    if RESC_MEM:
        acc, l_i = _resc(acc, l_i, alpha_c, ABL)
    if DRF:
        l_i = _fence(l_i)
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
           envtag=0, abl=0, v1x=0, lazy=0, ksl=0, kcf=0, kcl=0, wq1=1, wq2=1, kpred=0, lrke=0, ksa=0, block_m=None,
           block_n=64, r2p=0, ablc=0, lth=8.0, vf=0, ftz=0, sched="",
           rv=0, d1f=0, exs=0, drf=0, **_ignored):
    T, n_q, KBv = qp.shape
    n_kv = kp.shape[1]
    D = 2 * KBv
    block_m = block_m or 32 * num_warps
    assert block_m in (32 * num_warps, 64 * num_warps) and block_n in (64, 128)
    assert ks.stride(0) % 16 == 0 and (qc is None or kc.stride(0) % 8 == 0), \
        "direct-to-LDS dword copies of the K scales need a 16-B token stride"
    assert (D == 128 and ks.stride(1) % 4 == 0 and block_n == 64) or (D == 64 and ks.stride(1) == 2 and not ksl
                                                                       and not kcl)
    assert T % block_n == 0 and T % block_m == 0 and v.stride(2) == 1 and out.stride(2) == 1
    assert T // block_n >= 4
    assert not exs or (vf == 1 and T // block_n >= 5), "EXS needs VF 1 and >= 1 main-loop pair"
    assert abl == 0 and v1x == 0, "timing ablations / V1X are not ported"
    if qc is None:   # Hadamard: no coefficients; pass the fp16 V view so KC has a half dtype (never read)
        qc, kc = qs, v
        sqc, skc = (0, 0), (0, 0)
        pw = 8
    else:
        pw = kc.stride(0)
        assert pw in (8, 16) and qc.stride(0) == pw
        sqc, skc = (qc.stride(0), qc.stride(1)), (kc.stride(0), kc.stride(1))
    grid = (n_q, triton.cdiv(T, block_m), 1)
    fa = ((("amdgpu-agpr-alloc", "0,0"),) if agpr0 else ())
    if ftz:   # FP32 denormals flushed (MODE register): exp2 underflow and packed-f32 ops never see denormals
        fa += (("denormal-fp-math-f32", "preserve-sign,preserve-sign"),)
    if sched:
        fa += (("amdgpu-sched-strategy", sched),)
    kw = dict(llvm_fn_attrs=fa) if fa else {}
    return mx_attn_fwd_d64[grid](qp, qs, qc, kp, ks, kc, v, float(sm_scale), lse, out,
                                 qp.stride(0), qp.stride(1), qs.stride(0), qs.stride(1), sqc[0], sqc[1],
                                 kp.stride(0), kp.stride(1), ks.stride(0), ks.stride(1), skc[0], skc[1],
                                 v.stride(0), v.stride(1), out.stride(0), out.stride(1), ks.stride(1) // 4,
                                 skc[0] // 8, HQ=n_q, HK=n_kv, N_CTX=T, BLOCK_M=block_m, BLOCK_DMODEL=D,
                                 BLOCK_N=block_n, CORR=bool(corr), RESC_MEM=bool(resc_mem or lazy), PRIO=bool(prio),
                                 ENVTAG=envtag, ABL=abl, V1X=v1x, LAZY=int(lazy), KSL=int(ksl), KCF=int(kcf),
                                 KCL=int(kcl), WQ1=int(wq1), WQ2=int(wq2), KPRED=int(kpred), LRKE=int(lrke),
                                 KSA=int(ksa), R2P_=int(r2p), PW=pw, ABLC=int(ablc), LTH=float(lth), VF=int(vf), RV=int(rv), D1F=int(d1f), EXS=int(exs), DRF=int(drf), num_warps=num_warps, waves_per_eu=wpe, **kw)


def run_fp16(q, k, v, out, lse, sm_scale, sq=0, num_warps=8, wpe=2, agpr0=True, resc_mem=0, prio=0, envtag=0, lazy=0,
             wq1=1, wq2=1, kpred=0, lrke=0, block_m=None, block_n=64, lth=8.0, vf=0, ftz=0, sched="", d1f=0, exs=0,
             drf=0, **_ignored):
    """fp16 QK^T mode of the same kernel. q [T, n_q, D], k / v [T, n_kv, D] fp16 views (head-dim stride 1, any token /
    head strides, e.g. views of the fused projection buffer); out [T, n_q, D] fp16; lse [n_q, T] fp32."""
    T, n_q, D = q.shape
    n_kv = k.shape[1]
    block_m = block_m or 32 * num_warps
    assert q.dtype == k.dtype == v.dtype == torch_f16() and q.stride(2) == k.stride(2) == v.stride(2) == 1
    assert block_m in (32 * num_warps, 64 * num_warps) and block_n in (64, 128)
    assert (D == 128 and block_n == 64) or D == 64
    assert T % block_n == 0 and T % block_m == 0 and out.stride(2) == 1 and T // block_n >= 4
    assert not exs or (vf == 1 and T // block_n >= 5)
    grid = (n_q, triton.cdiv(T, block_m), 1)
    fa = ((("amdgpu-agpr-alloc", "0,0"),) if agpr0 else ())
    if ftz:
        fa += (("denormal-fp-math-f32", "preserve-sign,preserve-sign"),)
    if sched:
        fa += (("amdgpu-sched-strategy", sched),)
    kw = dict(llvm_fn_attrs=fa) if fa else {}
    return mx_attn_fwd_d64[grid](q, q, q, k, k, k, v, float(sm_scale), lse, out,
                                 q.stride(0), q.stride(1), 0, 0, 0, 0,
                                 k.stride(0), k.stride(1), 0, 0, 0, 0,
                                 v.stride(0), v.stride(1), out.stride(0), out.stride(1), 0, 0,
                                 HQ=n_q, HK=n_kv, N_CTX=T, BLOCK_M=block_m, BLOCK_DMODEL=D, BLOCK_N=block_n, CORR=False,
                                 RESC_MEM=bool(resc_mem or lazy), PRIO=bool(prio), ENVTAG=envtag, ABL=0, V1X=0,
                                 LAZY=int(lazy), KSL=0, KCF=0, KCL=0, WQ1=int(wq1), WQ2=int(wq2), KPRED=int(kpred),
                                 LRKE=int(lrke), KSA=0, R2P_=0, PW=8, ABLC=0, LTH=float(lth), VF=int(vf), RV=0,
                                 D1F=int(d1f), EXS=int(exs), QF16=1, SQ=int(sq), DRF=int(drf), num_warps=num_warps,
                                 waves_per_eu=wpe, **kw)


def torch_f16():
    import torch
    return torch.float16
