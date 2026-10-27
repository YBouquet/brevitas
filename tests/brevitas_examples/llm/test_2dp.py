# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import math
from unittest.mock import patch

import pytest
import pytest_cases
import torch
from torch import nn
import torch.nn.functional as F

from brevitas_examples.llm.dc_extract_sdpa import build_frozen_u
from brevitas_examples.llm.dc_extract_sdpa import DCExtractQuantScaledDotProductAttention
from brevitas_examples.llm.dc_extract_sdpa import hflf_perm
from brevitas_examples.llm.dc_extract_sdpa import set_frozen_u
from brevitas_examples.llm.dc_extract_sdpa import u_overlap_logging
from brevitas_examples.llm.llm_args import create_args_parser
from brevitas_examples.llm.main import parse_layer_list
from brevitas_examples.llm.main import quantize_llm
from brevitas_examples.llm.mf_probe import sink_balance
from brevitas_examples.llm.sqnr_decomp import attention_error_stats
from tests.brevitas_examples.common import get_default_args
from tests.brevitas_examples.llm.test_llm import mock_load_raw_dataset

ATOL = 1e-5
N, HQ, HKV, L, D = 2, 4, 2, 12, 16

NO_QUANT = dict(
    q_scaled_quant=None, k_transposed_quant=None, v_quant=None, attn_output_weights_quant=None)


def _qkv(seed=0):
    g = torch.Generator().manual_seed(seed)
    # A common offset per head gives Q/K a non-zero token mean, as after RoPE in an LLM
    q = torch.randn(N, HQ, L, D, generator=g) + 2.0 * torch.randn(1, HQ, 1, D, generator=g)
    k = torch.randn(N, HKV, L, D, generator=g) + 2.0 * torch.randn(1, HKV, 1, D, generator=g)
    v = torch.randn(N, HKV, L, D, generator=g)
    return q, k, v


def _sdpa(**kwargs):
    return DCExtractQuantScaledDotProductAttention(num_key_value_heads=HKV, **NO_QUANT, **kwargs)


class _Layers(nn.Module):
    """Names the attention modules like a decoder (``layers.<i>.attn``)."""

    def __init__(self, n_layers=2, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({'attn': _sdpa(**kwargs)}) for _ in range(n_layers)])

    def forward(self, q, k, v):
        return [l['attn'](q, k, v, is_causal=True, enable_gqa=True) for l in self.layers]


def _random_u(n_layers, dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(n_layers, HKV, dim, 2, generator=g))[0]


def test_hflf_perm_is_permutation():
    for dim in (4, 64, 128):
        assert sorted(hflf_perm(dim).tolist()) == list(range(dim))


@pytest_cases.parametrize('dc_sub_dim', [2, 0, -1])
@pytest_cases.parametrize('had_rand', [False, True])
def test_exact_without_quantization(dc_sub_dim, had_rand):
    # The exact plane + residual split, and the Q/K Hadamard, leave the score unchanged
    if had_rand and dc_sub_dim >= 0:
        pytest.skip("Hadamard randomization only applies to dc_sub_dim < 0")
    q, k, v = _qkv()
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    out = _sdpa(
        dc_sub_dim=dc_sub_dim, dc_had_rand=had_rand)(
            q, k, v, is_causal=True, enable_gqa=True)
    assert torch.allclose(out, ref, atol=ATOL)


@pytest_cases.parametrize('space', ['lf', 'full'])
def test_exact_with_frozen_u(space):
    q, k, v = _qkv()
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    model = _Layers()
    set_frozen_u(model, _random_u(2, D // 2 if space == 'lf' else D))
    u_overlap_logging(model, enable=True)
    for out in model(q, k, v):
        assert torch.allclose(out, ref, atol=ATOL)
    overlap = u_overlap_logging(model, enable=False)
    # The live-vs-frozen overlap is logged for LF planes only
    assert (len(overlap) == 2) == (space == 'lf')
    assert all(0.0 <= o <= 1.0 + 1e-6 for o in overlap.values())


@pytest_cases.parametrize('correction', ['unshrink', 'mean', 'scale'])
@pytest_cases.parametrize('dc_sub_dim', [2, -1])
def test_zero_gamma_is_identity(correction, dc_sub_dim):
    if correction == 'scale' and dc_sub_dim < 0:
        pytest.skip("scale needs an exact plane")
    q, k, v = _qkv()
    ref = _sdpa(dc_sub_dim=dc_sub_dim)(q, k, v, is_causal=True, enable_gqa=True)
    out = _sdpa(
        dc_sub_dim=dc_sub_dim, dc_correction=correction, dc_gamma=0.0)(
            q, k, v, is_causal=True, enable_gqa=True)
    assert torch.equal(out, ref)


def _score_sdpa(score, v):
    L = score.size(-2)
    mask = torch.ones(L, L, dtype=torch.bool).tril()
    p = torch.softmax(score.masked_fill(~mask, float('-inf')), dim=-1)
    return p @ v


@pytest_cases.parametrize('abar_mode', ['full', 'causal'])
def test_corrections(abar_mode):
    # With quantization off, unshrink scales the residual score by 1 / (1 - gamma), mean subtracts
    # gamma * a_bar b^T / sqrt(d), and scale subtracts gamma * a b^T / sqrt(d)
    gamma = 0.1
    q, k, v = _qkv()
    kx, vx = k.repeat_interleave(HQ // HKV, 1), v.repeat_interleave(HQ // HKV, 1)
    m = _sdpa(dc_sub_dim=2)
    res_q, res_k, cq, ck = m._dc_split(q, kx)
    s = 1 / math.sqrt(D)
    s_res = (res_q * s) @ res_k.transpose(-2, -1)
    s_plane = (cq * s) @ ck.transpose(-2, -1)
    if abar_mode == 'causal':
        abar = cq.cumsum(dim=-2) / torch.arange(1, L + 1).view(1, 1, -1, 1)
    else:
        abar = cq.mean(dim=-2, keepdim=True)
    expected = {
        'unshrink': s_res / (1 - gamma) + s_plane,
        'mean': s_res + s_plane - gamma * (abar * s) @ ck.transpose(-2, -1),
        'scale': s_res + (1 - gamma) * s_plane}
    for correction, score in expected.items():
        out = _sdpa(
            dc_sub_dim=2, dc_correction=correction, dc_gamma=gamma, dc_abar_mode=abar_mode)(
                q, k, v, is_causal=True, enable_gqa=True)
        assert torch.allclose(out, _score_sdpa(score, vx), atol=ATOL), correction


def test_causal_abar_uses_past_only():
    m = _sdpa(dc_abar_mode='causal')
    cq = torch.randn(N, HQ, L, 2)
    abar = m._abar(cq)
    cq2 = cq.clone()
    cq2[..., L // 2:, :] += 100.0  # change the future
    assert torch.equal(m._abar(cq2)[..., :L // 2, :], abar[..., :L // 2, :])


def test_invalid_options():
    with pytest.raises(ValueError):
        _sdpa(dc_correction='bogus')
    with pytest.raises(ValueError):
        _sdpa(dc_sub_dim=-1, dc_correction='scale', dc_gamma=0.1)
    with pytest.raises(ValueError):
        _sdpa(dc_correction='unshrink', dc_gamma=1.0)


@pytest_cases.parametrize('basis', ['mean', 'pca'])
@pytest_cases.parametrize('space', ['lf', 'full'])
def test_frozen_u_calibration_and_roundtrip(basis, space, tmp_path):
    q, k, v = _qkv()
    model = _Layers()
    mods = [l['attn'] for l in model.layers]
    for m in mods:
        m.start_u_calibration()
    model(q, k, v)
    stats = {i: m.stop_u_calibration() for i, m in enumerate(mods)}
    U, eigfrac = build_frozen_u(stats, basis=basis, space=space)
    dim = D // 2 if space == 'lf' else D
    assert U.shape == (2, HKV, dim, 2)
    eye = torch.eye(2).expand(2, HKV, 2, 2)
    assert torch.allclose(U.transpose(-2, -1) @ U, eye, atol=1e-5)
    assert (len(eigfrac) > 0) == (basis == 'pca')
    if basis == 'mean' and space == 'lf':
        # The plane spans the pooled LF query and key means (position 0 excluded)
        P = hflf_perm(D)
        kx = k.repeat_interleave(HQ // HKV, 1)
        q_mean = q.index_select(-1, P)[:, :2, 1:, D // 2:].mean(dim=(0, 1, 2))
        k_mean = kx.index_select(-1, P)[:, 0, 1:, D // 2:].mean(dim=(0, 1))
        for vec in (q_mean, k_mean):
            proj = U[0, 0] @ (U[0, 0].transpose(0, 1) @ vec)
            assert torch.allclose(proj, vec, atol=1e-4)

    set_frozen_u(model, U)
    out = model(q, k, v)
    path = tmp_path / "u.pt"
    torch.save(U, path)
    model2 = _Layers()
    set_frozen_u(model2, torch.load(path))
    for a, b in zip(out, model2(q, k, v)):
        assert torch.equal(a, b)


def test_sink_balance():
    g = torch.Generator().manual_seed(0)
    ref = torch.randn(N, HQ, L, L, generator=g)
    p = torch.softmax(ref, dim=-1)
    aw = ref + 0.1 * torch.randn(N, HQ, L, L, generator=g)
    out = sink_balance(aw, ref, p)
    dS = out - ref
    pi = p[..., 1:] / (1 - p[..., :1])
    assert torch.allclose(dS[..., 0], (pi * dS[..., 1:]).sum(-1), atol=1e-5)
    assert torch.equal(out[..., 1:], aw[..., 1:])


def test_attention_error_stats_without_quantization():
    q, k, v = _qkv()
    kx, vx = k.repeat_interleave(HQ // HKV, 1), v.repeat_interleave(HQ // HKV, 1)
    s = 1 / math.sqrt(D)
    score = (q * s) @ kx.transpose(-2, -1)
    bias = torch.zeros(L, L).masked_fill(~torch.ones(L, L, dtype=torch.bool).tril(), float('-inf'))
    rec = attention_error_stats(q, kx, vx, vx, score, bias, s, q * s, q * s, kx, kx, mf_gamma=0.05)
    assert rec['pre_softmax'] > 100.0
    assert abs(rec['gain_gamma']) < 1e-4
    assert rec['g_q'] == pytest.approx(1.0) and rec['g_k'] == pytest.approx(1.0)
    assert 'mf_m2' in rec


def test_parse_layer_list():
    assert parse_layer_list(['0-2', 'last'], 8) == {0, 1, 2, 7}
    assert parse_layer_list(['first,5'], 8) == {0, 5}


@pytest.fixture
def run_args():
    args = get_default_args(create_args_parser())
    args.nsamples = 2
    args.seqlen = 8
    args.model = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    args.dataset = "c4"
    args.eval = True
    args.weight_bit_width = 8
    args.weight_quant_granularity = "per_channel"
    args.input_bit_width = 8
    args.act_calibration = True
    args.dtype = "float32"
    args.quant_sdpa = "eager"
    args.dc_extract_qk = True
    args.nsamples_rot_calibration = 2
    return args


def _run(args):
    with patch('brevitas_examples.llm.llm_quant.data_utils.load_raw_dataset',
               mock_load_raw_dataset):
        return quantize_llm(args)


@pytest.mark.llm
@pytest_cases.parametrize(
    'overrides',
    [
        {},
        {
            'dc_sub_dim': -1, 'dc_had_rand': True},
        {
            'dc_u_mode': 'frozen',
            'dc_u_calib': 'random',
            'dc_correction': 'unshrink',
            'dc_gamma': 0.05},
        {
            'dc_u_mode': 'frozen',
            'dc_u_calib': 'selfgen',
            'dc_selfgen_nsamples': 2,
            'dc_correction': 'mean',
            'dc_gamma': 0.05,
            'dc_abar_mode': 'causal'},
        {
            'dc_had_layers': ['0'], 'dc_sinkbal': True},
        {
            'sqnr_decomp': 1, 'mf_probe_gamma': 0.05},
        {
            'attn_quant_config': 'qk', 'no_quantize_linear': True},
        {
            'replace_rmsnorm': True,
            'rotation': 'fused_no_fx',
            'rotation_orphan_sink': True,
            'rotation_mode': 'had_rand',
            'rotation_sdpa_value_det': True},
        {
            'replace_rmsnorm': True,
            'rotation': 'fused_no_fx',
            'rotation_orphan_sink': True,
            'rotation_mode': 'had',
            'rotation_sdpa_value_rand': True,
            'rotation_sdpa_value_rand_shared': True},],
    ids=[
        'live',
        'had_rand',
        'frozen_random_unshrink',
        'frozen_selfgen_mean_causal',
        'hybrid_sinkbal',
        'sqnr_decomp',
        'qk_only',
        'quarot_value_det',
        'value_rand_shared'])
def test_2dp_run(run_args, overrides):
    for key, val in overrides.items():
        setattr(run_args, key, val)
    results, model = _run(run_args)
    assert math.isfinite(results['quant_ppl'])
    mods = [m for m in model.modules() if isinstance(m, DCExtractQuantScaledDotProductAttention)]
    assert len(mods) == model.config.num_hidden_layers
    assert sorted(m.layer_idx for m in mods) == list(range(len(mods)))
    if run_args.dc_u_mode == 'frozen':
        assert all(m.u_frozen is not None for m in mods)
    if run_args.attn_quant_config == 'qk':
        assert all(m.q_scaled_quant.act_quant.is_quant_enabled for m in mods)
        assert not any(m.v_quant.act_quant.is_quant_enabled for m in mods)


@pytest.mark.llm
def test_2dp_frozen_u_save_load(run_args, tmp_path):
    run_args.dc_u_mode = 'frozen'
    run_args.dc_u_calib = 'random'
    run_args.dc_u_path = str(tmp_path / "u.pt")
    results_calib, _ = _run(copy.deepcopy(run_args))
    results_load, _ = _run(copy.deepcopy(run_args))
    assert results_load['quant_ppl'] == pytest.approx(results_calib['quant_ppl'], rel=1e-5)


def test_tcast_override_skips_disabled_quantizers():
    # A fake tcast that maps every tensor to 0. With only Q and K quantized ("qk"), the scores become 0
    # (uniform causal attention) while V must stay exact.
    import sys
    import types

    from brevitas.quant.scaled_int import Int8ActPerTensorFloat
    fake = types.ModuleType('tcast')
    fake.datatype = lambda *args: None
    fake.vcast = lambda x, dt: torch.zeros_like(x)
    q, k, v = _qkv()
    m = DCExtractQuantScaledDotProductAttention(
        num_key_value_heads=HKV,
        dc_sub_dim=0,
        q_scaled_quant=Int8ActPerTensorFloat,
        k_transposed_quant=Int8ActPerTensorFloat,
        v_quant=None,
        attn_output_weights_quant=None,
        tcast_scale='fake')
    with patch.dict(sys.modules, {'tcast': fake}):
        out = m(q, k, v, is_causal=True, enable_gqa=True)
    vx = v.repeat_interleave(HQ // HKV, 1)
    expected = vx.cumsum(dim=-2) / torch.arange(1, L + 1).view(1, 1, -1, 1)
    assert torch.allclose(out, expected, atol=ATOL)
