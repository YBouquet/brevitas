# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Portions derived from fmha_v3.py of the gfx950 Gluon tutorial ("High-Performance Gluon Kernels on AMD GFX9",
# Copyright (c) 2026 AMD ROCm Software, MIT License: see LICENSE-gluon-tutorial).

"""Variant sets for bench_roof.py / prof_roof.py."""
BASE = dict(num_warps=8, wpe=2, agpr0=True, kload=0, pro_wait=0, resc_mem=0, prio=0, env="", abl=0, lds=0, v1x=0, lazy=0)
ENVS = {"": {}, "scal": {"AMDGCN_SCALARIZE_PACKED_FOPS": "1"},
        "nosink": {"DISABLE_LLVM_OPT": "disable-machine-sink"},
        "nosink+scal": {"DISABLE_LLVM_OPT": "disable-machine-sink", "AMDGCN_SCALARIZE_PACKED_FOPS": "1"}}
ENV_ORDER = ["", "scal", "nosink", "nosink+scal"]


def V(**kw):
    d = dict(BASE)
    d.update(kw)
    return d


SETS = {
    "sweep1": {
        "base": V(),
        "k1": V(kload=1),
        "k1w": V(kload=1, pro_wait=1),
        "k1w_rm": V(kload=1, pro_wait=1, resc_mem=1),
        "k1w_prio": V(kload=1, pro_wait=1, prio=1),
        "k1w_agpr_wpe1": V(kload=1, pro_wait=1, agpr0=False, wpe=1),
        "k1w_w4": V(kload=1, pro_wait=1, num_warps=4),
        "k1w_scal": V(kload=1, pro_wait=1, env="scal"),
        "k1w_ns": V(kload=1, pro_wait=1, env="nosink"),
        "k1w_rm_ns": V(kload=1, pro_wait=1, resc_mem=1, env="nosink"),
        "k1w_ns_scal": V(kload=1, pro_wait=1, env="nosink+scal"),
    },
    "sweep2": {
        "base": V(),
        "base_w": V(pro_wait=1),
        "base_rm": V(resc_mem=1),
        "base_w4": V(num_warps=4),
        "base_prio": V(prio=1),
        "k1w_rm": V(kload=1, pro_wait=1, resc_mem=1),
        "k2": V(kload=2),
        "k2w": V(kload=2, pro_wait=1),
        "k2w_rm": V(kload=2, pro_wait=1, resc_mem=1),
        "k2_w4": V(kload=2, num_warps=4),
        "base_ns": V(env="nosink"),
        "base_rm_ns": V(resc_mem=1, env="nosink"),
        "k2w_ns": V(kload=2, pro_wait=1, env="nosink"),
        "k2w_rm_ns": V(kload=2, pro_wait=1, resc_mem=1, env="nosink"),
    },
    "sweep3": {
        "base": V(),
        "base_w4": V(num_warps=4),
        "k2w_rm_ns": V(kload=2, pro_wait=1, resc_mem=1, env="nosink"),
        "k2w_rm_ns_w4": V(kload=2, pro_wait=1, resc_mem=1, num_warps=4, env="nosink"),
        "k2_w4_ns": V(kload=2, num_warps=4, env="nosink"),
        "k2w_rm_w4": V(kload=2, pro_wait=1, resc_mem=1, num_warps=4),
        "A_noexp": V(abl=1),
        "A_noK": V(abl=2),
        "A_nobar": V(abl=4),
        "A_noresc": V(abl=8),
        "A_noV": V(abl=16),
        "A_noKV": V(abl=18),
        "A_noexp_noresc": V(abl=9),
        "A_nobar_w4": V(abl=4, num_warps=4),
    },
    "sweep4": {
        "base": V(),
        "base_w4": V(num_warps=4),
        "k2w_rm_w4": V(kload=2, pro_wait=1, resc_mem=1, num_warps=4),
        "A_noK": V(abl=2),
        "L": V(lds=1),
        "L_rm": V(lds=1, resc_mem=1),
        "L_w4": V(lds=1, num_warps=4),
        "L_rm_w4": V(lds=1, resc_mem=1, num_warps=4),
        "L_prio": V(lds=1, prio=1),
        "L_ns": V(lds=1, env="nosink"),
        "L_rm_ns": V(lds=1, resc_mem=1, env="nosink"),
        "L_w4_ns": V(lds=1, num_warps=4, env="nosink"),
        "L_rm_w4_ns": V(lds=1, resc_mem=1, num_warps=4, env="nosink"),
    },
    "sweep5": {
        "base": V(),
        "L": V(lds=1),
        "L_v1": V(lds=1, v1x=1),
        "L_v2": V(lds=1, v1x=2),
        "L_v2_rm": V(lds=1, v1x=2, resc_mem=1),
        "L_v1_ns": V(lds=1, v1x=1, env="nosink"),
        "L_v2_ns": V(lds=1, v1x=2, env="nosink"),
        "L_scal": V(lds=1, env="scal"),
        "LA_noexp": V(lds=1, abl=1),
        "LA_noresc": V(lds=1, abl=8),
        "LA_noexp_noresc": V(lds=1, abl=9),
        "LA_nobar": V(lds=1, abl=4),
        "LA_noV": V(lds=1, abl=16),
        "LA_noK": V(lds=1, abl=32),
        "LA_noKV": V(lds=1, abl=48),
    },
    "sweep6": {
        "base": V(),
        "L": V(lds=1),
        "L_lazy": V(lds=1, lazy=1),
        "L_lazy_v1": V(lds=1, lazy=1, v1x=1),
        "L_lazy_v2": V(lds=1, lazy=1, v1x=2),
        "L_lazy_prio": V(lds=1, lazy=1, prio=1),
        "L_lazy_w4": V(lds=1, lazy=1, num_warps=4),
        "L_lazy_ns": V(lds=1, lazy=1, env="nosink"),
        "L_lazy_scal": V(lds=1, lazy=1, env="scal"),
        "L_lazy_ns_scal": V(lds=1, lazy=1, env="nosink+scal"),
    },
    "sweep7": {
        "L": V(lds=1),
        "L_lazy": V(lds=1, lazy=1),
        "LA_lazy_frozen": V(lds=1, lazy=2),
        "LA_noresc": V(lds=1, abl=8),
        "L_lazy_prio": V(lds=1, lazy=1, prio=1),
        "L_prio_scal": V(lds=1, prio=1, env="scal"),
        "L_lazy_prio_scal": V(lds=1, lazy=1, prio=1, env="scal"),
        "L_lazy_ns_scal": V(lds=1, lazy=1, env="nosink+scal"),
        "L_lazy_prio_ns_scal": V(lds=1, lazy=1, prio=1, env="nosink+scal"),
        "L_ns_scal": V(lds=1, env="nosink+scal"),
    },
    "final": {
        "base": V(),
        "L": V(lds=1),
        "L_lazy": V(lds=1, lazy=1),
        "L_lazy_prio": V(lds=1, lazy=1, prio=1),
        "L_lazy_ns_scal": V(lds=1, lazy=1, env="nosink+scal"),
    },
}
