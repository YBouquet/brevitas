# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Portions derived from fmha_v3.py of the gfx950 Gluon tutorial ("High-Performance Gluon Kernels on AMD GFX9",
# Copyright (c) 2026 AMD ROCm Software, MIT License: see LICENSE-gluon-tutorial).

"""Per-cluster census of the main loop (clusters delimited by s_barrier; loop = .LBB0_3 header .. back-branch).
    python3 clusters.py <file.amdgcn> [...]
Per cluster: MFMA by type, VALU instr, VALU issue cycles (4 / trans 8 / permlane 20), mem-port instr, waitcnt.
Phase model (two waves per SIMD, offset by one stage): phases (D1|M2'), (M1|D1'), (D2|M1'), (M2|D2'); each phase =
max(32*mfma_dot, 8*mfma_dot + valu_dot + valu_mem); per wave-tile = sum(phases) / 2.
"""
import re
import sys

CL = [("fp4", r"^v_mfma_scale"), ("k16", r"^v_mfma_f32_32x32x16"), ("k8", r"^v_mfma_f32_32x32x8"),
      ("exp", r"^v_(exp|log|rcp)_"), ("perm", r"^v_permlane"), ("valu", r"^v_"), ("ds", r"^ds_"),
      ("buf", r"^buffer_"), ("wait", r"^s_waitcnt")]


def loop_lines(asm):
    lines = [x.strip() for x in asm.splitlines()]
    labels = {m.group(1): i for i, x in enumerate(lines) for m in [re.match(r"^(\.LBB\d+_\d+):", x)] if m}
    best = None
    for j, x in enumerate(lines):
        m = re.match(r"^s_c?branch\w*\s+(\.LBB\d+_\d+)", x)
        if m and m.group(1) in labels and labels[m.group(1)] < j:
            if best is None or j - labels[m.group(1)] > best[1] - best[0]:
                best = (labels[m.group(1)], j)
    return lines[best[0]:best[1] + 1]


def clusters(lines):
    out, cur = [], {}
    for x in lines:
        if not x or x.startswith((";", ".")) or x.split()[0].endswith(":"):
            continue
        mn = x.split()[0]
        if mn == "s_barrier":
            out.append(cur)
            cur = {}
            continue
        for k, p in CL:
            if re.match(p, mn):
                cur[k] = cur.get(k, 0) + 1
                break
    out.append(cur)
    return out


def cyc(c):
    nm = c.get("fp4", 0) + c.get("k16", 0) + c.get("k8", 0)
    v = 4 * c.get("valu", 0) + 8 * c.get("exp", 0) + 20 * c.get("perm", 0)
    return nm, v


for f in sys.argv[1:]:
    cs = clusters(loop_lines(open(f).read()))
    # loop body = [head..b1] [b1..b2] ... ; the first and last pieces are one cluster split by the back-edge
    head, body, tail = cs[0], cs[1:-1], cs[-1]
    first = dict(head)
    for k, v in tail.items():
        first[k] = first.get(k, 0) + v
    seq = [first] + body          # 8 clusters: D1 M1 D2 M2 D1 M1 D2 M2 (rotated: head+tail = D1)
    print(f"== {f.split('/')[-1]}  ({len(seq)} clusters)")
    names = ["D1", "M1", "D2", "M2"] * 2
    for n, c in zip(names, seq):
        nm, v = cyc(c)
        print(f"  {n}: mfma fp4={c.get('fp4',0)} k16={c.get('k16',0)} k8={c.get('k8',0)} -> pipe {32*nm:4d}c | valu "
              f"{c.get('valu',0)+c.get('exp',0)+c.get('perm',0):3d} instr ({c.get('exp',0)} trans, {c.get('perm',0)} perm)"
              f" issue {8*nm+v:4d}c | ds {c.get('ds',0)} buf {c.get('buf',0)} wait {c.get('wait',0)}")
    tot = 0
    for t in range(2):
        D1, M1, D2, M2 = seq[4 * t:4 * t + 4]
        ph = []
        for d, m in ((D1, M2), (M1, D1), (D2, M1), (M2, D2)):
            if "fp4" not in d and "k16" not in d:
                d, m = m, d
            nm, vd = cyc(d)
            _, vm = cyc(m)
            ph.append(max(32 * nm, 8 * nm + vd + vm))
        tot += sum(ph) / 2
        print(f"  tile {t}: phases {ph} -> {sum(ph) / 2:.0f} c per wave-tile")
    print(f"  model per wave-tile (avg of 2): {tot / 2:.0f} c")
