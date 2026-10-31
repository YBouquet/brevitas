# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""fmha_v3_env: the tutorial's fmha_v3 (from $GLUON_TUTORIAL_ATTENTION_DIR, not vendored) with three edits applied at
import time, for the fp16 parity reference of bench_fp.py:
  * an unused ENVTAG constexpr, so each compile-time env (nosink / scal) compiles separately;
  * run_gluon_attention(..., envtag=0, M=None): the LSE buffer M can be preallocated (no allocation in the timed call),
    and the compiled kernel is returned (ISA dump).
The kernel body is unchanged."""
import importlib.util
import os
import sys
import tempfile


def load():
    tut = os.environ["GLUON_TUTORIAL_ATTENTION_DIR"]
    src = open(os.path.join(tut, "fmha_v3.py")).read()
    edits = [
        ("                   SCALE_ON_Q: gl.constexpr = True):\n",
         "                   SCALE_ON_Q: gl.constexpr = True, ENVTAG: gl.constexpr = 0):\n"),
        ("def run_gluon_attention(q, k, v, o, metadata: MetaData, scale_on_q: bool = True):",
         "def run_gluon_attention(q, k, v, o, metadata: MetaData, scale_on_q: bool = True, envtag: int = 0, M=None):"),
        ("    M = torch.empty((batch, nheads_q, metadata.max_seqlens_q), device=q.device, dtype=torch.float32)\n",
         "    if M is None:\n        M = torch.empty((batch, nheads_q, metadata.max_seqlens_q), device=q.device, "
         "dtype=torch.float32)\n"),
        ("        SCALE_ON_Q=scale_on_q,\n    )", "        SCALE_ON_Q=scale_on_q,\n        ENVTAG=envtag,\n    )"),
        ("    gluon_attn_fwd[grid](", "    return gluon_attn_fwd[grid]("),   # the compiled kernel, for the ISA dump
    ]
    for a, b in edits:
        assert src.count(a) == 1, f"tutorial fmha_v3.py does not match the expected version: {a[:50]!r}"
        src = src.replace(a, b)
    d = os.path.join(tempfile.gettempdir(), f"fmha_v3_env_{os.getuid()}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "fmha_v3_env.py")   # a real file: Triton needs the kernel source on disk
    open(path, "w").write(src)
    if tut not in sys.path:
        sys.path.insert(0, tut)   # its own `from common import ...`
    spec = importlib.util.spec_from_file_location("fmha_v3_env", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fmha_v3_env"] = mod
    spec.loader.exec_module(mod)
    return mod
