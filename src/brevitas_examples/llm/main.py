# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import nullcontext
from copy import deepcopy
import functools
import os
import pprint
import sys
import warnings

import numpy as np
from packaging import version
import torch
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader
from transformers import __version__ as transformers_version
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from brevitas.export.inference.manager import quant_inference_mode
from brevitas.export.onnx.standard.qcdq.manager import StdQCDQONNXManager
from brevitas.graph import load_quant_model_mode
from brevitas.graph.equalize import apply_rewriters
from brevitas.graph.equalize import fuse_parametrizations
from brevitas.graph.equalize import GraphRotationEqualization
from brevitas.graph.equalize import LayerwiseActivationRotation
from brevitas.graph.permute import rotate_permute_mode
from brevitas.graph.quantize import functional_quantization_mode
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph.utils import get_module
from brevitas.graph.utils import remove_weight_orig
from brevitas.nn.quant_sdpa import ScaledDotProductAttention
from brevitas.utils.logging import setup_logger
from brevitas.utils.python_utils import hooked_on_a_function
from brevitas_examples.common.accelerate_utils.accelerate import offload_model
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks
from brevitas_examples.common.accelerate_utils.accelerate import update_internal_dict
from brevitas_examples.common.dynamo_utils import dynamo_export_ctx
from brevitas_examples.common.dynamo_utils import patch_dynamo_export
from brevitas_examples.common.generative.quantize import generate_quant_maps
from brevitas_examples.common.generative.quantize import generate_quantizers
from brevitas_examples.common.generative.quantizers import QUANTIZERS_REGISTRY
from brevitas_examples.common.parse_utils import override_defaults
from brevitas_examples.common.parse_utils import parse_args
from brevitas_examples.llm.dc_extract_sdpa import assign_layer_ids
from brevitas_examples.llm.dc_extract_sdpa import build_frozen_u
from brevitas_examples.llm.dc_extract_sdpa import dc_modules
from brevitas_examples.llm.dc_extract_sdpa import set_frozen_u
from brevitas_examples.llm.dc_extract_sdpa import u_overlap_logging
from brevitas_examples.llm.gguf_export.export import save_quantized_as_gguf
from brevitas_examples.llm.llm_args import create_args_parser
from brevitas_examples.llm.llm_args import fx_required
from brevitas_examples.llm.llm_args import validate
from brevitas_examples.llm.llm_quant.awq.pre_quant import apply_awq
from brevitas_examples.llm.llm_quant.bias_corr import apply_bias_correction
from brevitas_examples.llm.llm_quant.calibrate import apply_calibration
from brevitas_examples.llm.llm_quant.data_utils import get_dataset_for_model
from brevitas_examples.llm.llm_quant.data_utils import llm_collate
from brevitas_examples.llm.llm_quant.equalize import apply_act_equalization
from brevitas_examples.llm.llm_quant.equalize import apply_weight_equalization
from brevitas_examples.llm.llm_quant.eval import compute_float_evaluation_metrics
from brevitas_examples.llm.llm_quant.eval import compute_quantized_evaluation_metrics
from brevitas_examples.llm.llm_quant.export import _get_dataset_props
from brevitas_examples.llm.llm_quant.export import BlockQuantProxyLevelManager
from brevitas_examples.llm.llm_quant.export import convert_hf_hparams_to_gguf
from brevitas_examples.llm.llm_quant.export import gguf_mapping
from brevitas_examples.llm.llm_quant.gpxq import apply_gpfq
from brevitas_examples.llm.llm_quant.gpxq import apply_gptq
from brevitas_examples.llm.llm_quant.gpxq import apply_magr
from brevitas_examples.llm.llm_quant.gpxq import apply_qronos
from brevitas_examples.llm.llm_quant.learned_round_utils import apply_learned_round
from brevitas_examples.llm.llm_quant.ln_affine_merge import apply_layernorm_affine_merge
from brevitas_examples.llm.llm_quant.ln_affine_merge import apply_layernorm_to_rmsnorm
from brevitas_examples.llm.llm_quant.ln_affine_merge import rmsnorm_patch
from brevitas_examples.llm.llm_quant.parse_utils import parse_custom_quantizer
from brevitas_examples.llm.llm_quant.parse_utils import parse_custom_trainer
from brevitas_examples.llm.llm_quant.piso import apply_scale_optimization
from brevitas_examples.llm.llm_quant.piso import build_scale_optimizer
from brevitas_examples.llm.llm_quant.prepare_for_quantize import add_zero_bias_to_linear
from brevitas_examples.llm.llm_quant.prepare_for_quantize import make_dynamo_compatible
from brevitas_examples.llm.llm_quant.prepare_for_quantize import \
    replace_sdpa_with_quantizable_layers
from brevitas_examples.llm.llm_quant.rotation_optimization import apply_fine_tuning
from brevitas_examples.llm.llm_quant.run_utils import fix_rewriter
from brevitas_examples.llm.llm_quant.svd_quant import apply_svd_quant
from brevitas_examples.llm.llm_quant.trainer_utils import TRAINER_REGISTRY

logging = setup_logger(__name__)

try:
    from brevitas.export.shark.manager import SharkManager
except:
    SharkManager = None
    logging.debug("Shark-AI not installed, cannot export to Shark")

# We need to patch this before anything else is executed
patch_dynamo_export()


def filter_results(results, tasks):
    # filter out what we actually want to track
    eval_results = dict()
    for task_name in tasks:
        # log all result metrics we have for this task
        for key, val in results["results"][task_name].items():
            if not isinstance(val, str):
                # for mmlu, we don't log results per subtask, but simply overall results
                name = f"{task_name}_{key}"
                eval_results[name] = val
    return eval_results


def tie_weights(args):
    _should_tie = True
    # Affine normalization merging updates the embedding and output weights independently, breaking
    # the tie between them.
    if args.ln_affine_merge:
        _should_tie = False
    # If any equalization has taken places, the embedding layer and the fully connected one are
    # not tied anymore, and they need to be treated as standalone, separate layers.
    # TODO: Consider conditionally disabling equalization for the output layer.
    if args.act_equalization or args.weight_equalization:
        _should_tie = False
    # Layerwise rotation transforms the output linear layer.
    # TODO: Consider conditionally disabling layerwise rotation for the output layer.
    if args.rotation == 'layerwise':
        _should_tie = False
    # In all other cases we can tie them back so as to reduce memory usage.
    return _should_tie


# Conventional per-task metric of the QuaRot/SpinQuant-style zero-shot average.
ZEROSHOT_METRIC = {
    'arc_easy': 'acc_norm',
    'arc_challenge': 'acc_norm',
    'hellaswag': 'acc_norm',
    'openbookqa': 'acc_norm',
    'piqa': 'acc_norm',
    'winogrande': 'acc',
    'boolq': 'acc'}


def _pick_metric(eval_results, task, metric):
    # filter_results keys look like 'arc_easy_acc_norm,none'
    target = f"{task}_{metric}"
    for key, val in eval_results.items():
        if key.split(',')[0] == target and isinstance(val, (int, float)):
            return val
    return None


def compute_zeroshot_avg(eval_results, tasks):
    """Unweighted mean over tasks of ZEROSHOT_METRIC (acc when not listed or not reported)."""
    vals = []
    for task in tasks:
        preferred = ZEROSHOT_METRIC.get(task, 'acc')
        val = _pick_metric(eval_results, task, preferred)
        if val is None and preferred != 'acc':
            val = _pick_metric(eval_results, task, 'acc')
        if val is not None:
            vals.append(val)
    return sum(vals) / len(vals) if vals else None


def dump_few_shot_samples(results, path):
    """Write lm_eval per-question records (target, correctness, per-candidate log-likelihoods)."""
    import json
    dump = {}
    for task, recs in (results.get("samples") or {}).items():
        dump[task] = [{
            "target":
                r.get("target"),
            "acc":
                r.get("acc"),
            "acc_norm":
                r.get("acc_norm"),
            "resps": [[float(x[0]), bool(x[1])]
                      for x in (r.get("filtered_resps") or [])],
            "conts": [
                str(a[1])
                for a in (r.get("arguments") or [])
                if isinstance(a, (list, tuple)) and len(a) > 1]}
                      for r in recs]
    with open(path, "w") as f:
        json.dump(dump, f)
    print(
        f"Wrote {sum(len(v) for v in dump.values())} few-shot records ({len(dump)} tasks) to {path}"
    )


def sdpa_value_rotation_kwargs(args):
    return dict(
        sdpa_value_only=args.rotation_sdpa_value_only,
        sdpa_value_rand=args.rotation_sdpa_value_rand,
        sdpa_value_rand_seed=args.rotation_sdpa_value_rand_seed,
        sdpa_value_rand_shared=args.rotation_sdpa_value_rand_shared,
        sdpa_value_det=args.rotation_sdpa_value_det)


def print_rotation_summary(args):
    if args.rotation is None:
        return
    if args.rotation_sdpa_value_rand:
        mode = "one shared draw" if args.rotation_sdpa_value_rand_shared else "per-block draws"
        print(
            f"[R2-rand] random R2 ({mode}, seed={args.rotation_sdpa_value_rand_seed}); "
            f"R1/full_rotation_method='{args.rotation_mode}'")
    elif args.rotation_sdpa_value_det:
        print(
            f"[R2-det] deterministic R2 (Sylvester Hadamard) on value regions; "
            f"R1/full_rotation_method='{args.rotation_mode}'")


def parse_layer_list(tokens, n_layers):
    """Layer indices from tokens: integers, inclusive ranges 'A-B', 'first', 'last'."""
    out = set()
    for tok in tokens:
        for t in tok.replace(',', ' ').split():
            if t == 'first':
                out.add(0)
            elif t == 'last':
                out.add(n_layers - 1)
            elif '-' in t:
                a, b = (int(x) for x in t.split('-'))
                out.update(range(a, b + 1))
            else:
                out.add(int(t))
    return out


def generate_selfgen_ids(model, tokenizer, args):
    """Data-free calibration text sampled from the float model.

    Each sequence starts with BOS and one uniformly random token, then is sampled at temperature
    --dc-selfgen-temperature with nucleus --dc-selfgen-top-p. Returns ``[n, seqlen]`` token ids."""
    n = args.dc_selfgen_nsamples
    seqlen = args.dc_u_calib_seqlen or args.seqlen
    bs = args.dc_selfgen_batch_size
    model = offload_model(model)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = torch.Generator().manual_seed(args.seed)
    vmax = min(getattr(tokenizer, "vocab_size", 128000), 128000)
    out = []
    torch.manual_seed(args.seed)
    with torch.no_grad():
        for s in range(0, n, bs):
            b = min(bs, n - s)
            first = torch.randint(0, vmax, (b, 1), generator=g)
            inp = torch.cat([torch.full((b, 1), tokenizer.bos_token_id), first], dim=1).to(dev)
            gen = model.generate(
                input_ids=inp,
                attention_mask=torch.ones_like(inp),
                do_sample=True,
                temperature=args.dc_selfgen_temperature,
                top_p=args.dc_selfgen_top_p,
                top_k=0,
                max_new_tokens=seqlen - 2,
                min_new_tokens=seqlen - 2,
                pad_token_id=tokenizer.eos_token_id)
            out.append(gen[:, :seqlen].cpu())
    remove_hooks(model)
    ids = torch.cat(out)
    sample = tokenizer.decode(ids[0, :80])
    print(
        f"[selfgen] generated {tuple(ids.shape)} tokens from the float model "
        f"(T={args.dc_selfgen_temperature}, top_p={args.dc_selfgen_top_p}, random first token, "
        f"seed {args.seed}); sample: {sample!r}")
    return ids


def setup_frozen_u(model, tokenizer, collate_fn, args, selfgen_ids=None):
    """Load the frozen 2DP plane from --dc-u-path, or calibrate it (and save it if a path is set)."""
    dev = next(model.parameters()).device
    path = args.dc_u_path
    if path and os.path.exists(path):
        U = torch.load(path, map_location=dev)
        set_frozen_u(model, U)
        print(f"[frozen-U] loaded {tuple(U.shape)} from {path}")
        return
    mods = dc_modules(model)
    for _, m in mods:
        m.start_u_calibration()
    if args.dc_u_calib == 'selfgen':
        calib = selfgen_ids
        for r in range(calib.size(0)):
            model(input_ids=calib[r:r + 1].to(dev))
        calname = "self-generated sequences (float model)"
    elif args.dc_u_calib == 'random':
        # Uniform random token ids over the regular vocabulary, BOS first as with
        # --bos-preprocessing sequence.
        gen = torch.Generator().manual_seed(args.seed)
        vmax = min(getattr(tokenizer, "vocab_size", 128000), 128000)
        cl = args.dc_u_calib_seqlen or args.seqlen
        calib = torch.randint(0, vmax, (args.nsamples_rot_calibration, cl), generator=gen)
        if args.bos_preprocessing is not None and tokenizer.bos_token_id is not None:
            calib[:, 0] = tokenizer.bos_token_id
        for r in range(calib.size(0)):
            model(input_ids=calib[r:r + 1].to(dev))
        calname = f"random tokens (uniform over {vmax} ids, calib seqlen {cl})"
    else:
        calib = get_dataset_for_model(
            bos_preprocessing=args.bos_preprocessing,
            dataset_name="wikitext2",
            tokenizer=tokenizer,
            nsamples=args.nsamples_rot_calibration,
            seqlen=args.seqlen,
            split="train",
            seed=args.seed)
        for b in DataLoader(dataset=calib, batch_size=1, collate_fn=collate_fn):
            model(input_ids=b["input_ids"].to(dev))
        calname = "WikiText-2 train windows"
    stats = {idx: m.stop_u_calibration() for idx, m in mods}
    U, eigfrac = build_frozen_u(
        stats, basis=args.dc_u_basis, space=args.dc_u_space, pca_center=args.dc_u_pca_center)
    U = U.to(dev)
    set_frozen_u(model, U)
    msg = (
        f"[frozen-U] calibrated {tuple(U.shape)} on {len(calib)} {calname} (seqlen {args.seqlen}, "
        f"seed {args.seed}), BOS position skipped; basis={args.dc_u_basis} space={args.dc_u_space}")
    if eigfrac:
        msg += (
            f" centered={args.dc_u_pca_center} top-2 energy share "
            f"mean {sum(eigfrac) / len(eigfrac):.4f} min {min(eigfrac):.4f}")
    print(msg)
    if path:
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(U.cpu(), path)
        print(f"[frozen-U] saved to {path}")
    for ref_path in args.dc_u_ref:
        R = torch.load(ref_path, map_location=dev).float()
        if R.shape != U.shape:
            print(
                f"[frozen-U] overlap vs {ref_path} skipped: shape {tuple(R.shape)} != {tuple(U.shape)}"
            )
            continue
        ov = (U.float().transpose(-2, -1) @ R).pow(2).sum(dim=(-2, -1)) / R.size(-1)
        per_layer = ov.mean(dim=1)
        print(
            f"[frozen-U] overlap vs reference {ref_path}: mean {float(ov.mean()):.4f} "
            f"min {float(ov.min()):.4f} (layer {int(ov.min(dim=1).values.argmin())})")
        print(
            "[frozen-U] vs-ref per-layer: " +
            " ".join(f"L{l}:{float(v):.3f}" for l, v in enumerate(per_layer)))


def print_u_overlap(model, tag):
    """Print and reset the live-vs-frozen plane overlap ||U_live^T U_frozen||_F^2 / 2 per layer."""
    per = u_overlap_logging(model, enable=True)
    if not per:
        return
    vals = list(per.values())
    print(
        f"[frozen-U] overlap during {tag}: mean {sum(vals)/len(vals):.4f} min {min(vals):.4f} "
        f"(layer {min(per, key=per.get)})")
    print("[frozen-U] per-layer: " + " ".join(f"L{l}:{v:.3f}" for l, v in per.items()))


def fused_rotation_no_fx(model, calibration_loader, args):
    with torch.no_grad(), rmsnorm_patch(model, model.config) as patcher:
        rmsnorm_classes = patcher.rmsnorm_classes
        with make_dynamo_compatible(model) as dynamo_comp, dynamo_export_ctx():
            fx_model, guards = torch._dynamo.export(dynamo_comp.model)(**next(iter(calibration_loader)))
    if hasattr(model, str(torch.nn.functional.scaled_dot_product_attention)):
        m_to_add = getattr(model, str(torch.nn.functional.scaled_dot_product_attention))
        fx_model.add_module(str(torch.nn.functional.scaled_dot_product_attention), m_to_add)

    layers_to_expand = []
    if args.rotation is not None:
        for name, _ in fx_model.named_modules():
            if any(map(lambda x: x in name, args.rotation_layers_to_expand)):
                layers_to_expand.append(name)

    if args.ln_affine_merge:
        warnings.warn(
            "Applying LN affine merge. Note that this call breaks ties between the the lm_head and the embedding layer"
        )
        apply_layernorm_affine_merge(fx_model, rmsnorm_classes)
    fx_model, rewriters = apply_layernorm_to_rmsnorm(fx_model, return_rewriters=True)
    rewriters = fix_rewriter(rewriters, model, 'weight')

    for r in rewriters:
        r.apply(model)
    fx_model = offload_model(fx_model)
    # Since we apply the rewriters to a different, non-fx model, we need only to compute them
    # And apply them in a second moment on the non-fx model
    delay_rewriters = True
    return_rewriters = True

    extra_state_kwargs = {'equivariant_layers': rmsnorm_classes}
    eq = GraphRotationEqualization(
        orphan_sink=args.rotation_orphan_sink,
        full_rotation_method=args.rotation_mode,
        return_rewriters=return_rewriters,
        sdpa_regions=args.rotation_sdpa_regions,
        **sdpa_value_rotation_kwargs(args),
        use_parametrized_rotations=args.fine_tune,
        delay_rewriters=delay_rewriters,
        expansion_step=args.expansion_step,
        layers_to_expand=layers_to_expand,
        rotation_block_size=args.rotation_block_size,
        disable_block_rotation_for_fused=args.disable_block_rotation_for_fused,
        extra_state_kwargs=extra_state_kwargs)

    if args.permute_fn is not None:
        print("Applying permutations...")
        with rotate_permute_mode(fx_model,
                                 rotation=eq,
                                 permute_fn=args.permute_fn,
                                 block_size=args.rotation_block_size,
                                 disable_for_fused_rotations=args.disable_block_rotation_for_fused,
                                 extra_state_kwargs=extra_state_kwargs) as rpm:

            # Get fx_model from the context manager
            fx_model = rpm.model
            # Run calibration on fx_model to collect activation statistics
            with torch.no_grad():
                fx_model(**next(iter(calibration_loader)))
            # Get rewriters from the context manager
            rewriters = rpm.rewriters
    else:
        # Only rotation enabled
        fx_model, rewriters = eq.apply(fx_model)

    # fused_rotation_no_fx() may be called either if args.rotation == 'fused_no_fx' or args.permute_fn is not None,
    # so if args.rotation == 'layerwise', we need to skip applying the rewriters here to do it later
    if args.rotation != 'layerwise':
        model = offload_model(model)
        rewriters = fix_rewriter(rewriters, model, 'weight')
        model = apply_rewriters(model, rewriters, delay_rewriters=False)

    remove_hooks(model)


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def model_export(model, tokenizer, ref_input, args, config=None):
    # Use a float32 container by default; preserve the dtype for GGUF export.
    # Observed issues with llama.cpp and Qwen3 when upcasting to float32.
    if 'gguf' not in args.export_target:
        model = model.to(dtype=torch.float32)

    if args.export_target == 'onnx_qcdq':
        export_device = torch.device('cpu')
        model = model.to(export_device)
        ref_input = tree_map(
            lambda value: value.to(export_device) if isinstance(value, torch.Tensor) else value,
            ref_input)
        from optimum.exporters.onnx import onnx_export_from_model

        if args.weight_quant_granularity == 'per_group':
            export_manager = BlockQuantProxyLevelManager
        else:
            export_manager = StdQCDQONNXManager
            export_manager.change_weight_export(export_weight_q_node=True)

        print(f"Exporting the model in ./{args.export_prefix}")
        model.eval()
        with torch.no_grad(), quant_inference_mode(model, export_manager=export_manager):
            model(**ref_input)
            onnx_export_from_model(
                model,
                f"./{args.export_prefix}",
                task="text-generation-with-past",
                do_validation=False,
                device=str(export_device))
    elif 'gguf' in args.export_target:
        save_quantized_as_gguf(model, tokenizer, args.export_target, export_path=args.export_path)
    elif args.export_target == 'vllm':
        from brevitas.export.inference.vLLM.manager import vLLMExportManager

        model = offload_model(model)
        with quant_inference_mode(model, export_manager=vLLMExportManager) as export_mode:
            model(**ref_input)
            remove_hooks(model)
            export_mode.export_manager.export(model, tokenizer, args.export_prefix)
    elif args.export_target == 'shark':
        assert SharkManager is not None, "Please install shark-ai to export to Shark"
        from sharktank.types import Theta

        if args.export_prefix is None:
            export_path = "./dataset.irpa"
        else:
            os.makedirs(f"./{args.export_prefix}", exist_ok=True)
            export_path = f"./{args.export_prefix}/dataset.irpa"

        print(f"Exporting the model in {export_path}")

        export = SharkManager(config=config)
        with torch.no_grad():
            model = offload_model(model)
            ds = export.export(model, **ref_input)
        properties = ds.properties
        root_theta = ds.root_theta.flatten()

        # Export is always upcast to float32, and if we don't update the properties, it will fail
        properties['torch_dtype'] = 'float32'
        root_theta = Theta(gguf_mapping(root_theta, config))
        properties = convert_hf_hparams_to_gguf(_get_dataset_props(properties))
        root_theta.rename_tensors_to_paths()
        ds.properties = properties
        ds.root_theta = root_theta
        ds.save(export_path, io_report_callback=None)


# Recursive function to unwrap equalized layers
def find_equalized_layer(layer):
    if hasattr(layer, 'layer'):
        return find_equalized_layer(layer.layer)
    return layer


def quantize_llm(args, extra_args=None):
    validate(args, extra_args)
    set_seed(args.seed)
    if args.export_prefix is None:
        args.export_prefix = f"{args.model.replace('/', '--')}"

    # Whether to quantize SDPA with FX

    # `torch_dtype` renamed to `dtype` in transformers 4.56, removed in 5.x.
    dtype_kwarg = "dtype" if version.parse(transformers_version) >= version.parse(
        "4.56.0") else "torch_dtype"
    kwargs = {dtype_kwarg: args.dtype}
    if args.quant_sdpa:
        kwargs["attn_implementation"] = "sdpa"

    print("Model loading...")
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    dtype = next(model.parameters()).dtype
    config = model.config
    print("Model loaded.")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    float_probabilities = None
    float_ppl = None
    quant_ppl = None
    quant_ear = None
    quant_kld = None

    require_fx = fx_required(args)
    if require_fx and args.calibration_batch_size > 1:
        warnings.warn(
            f"The provided configuration requires fx and has a batch size of {args.calibration_batch_size}.\nErrors may occur when using fx and batch_size > 1.\nIf you experience any issues try chaning the configuration to avoid using fx or to set the batch_size to 1."
        )

    collate_fn = llm_collate(
        model_name_or_path=args.model, require_fx=require_fx and args.export_target is not None)

    # Load the data for calibration and evaluation.
    calibration_dataset = get_dataset_for_model(
        bos_preprocessing=args.bos_preprocessing,
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        split="train",
        seed=args.seed)

    # Batched data loader to accelerate GPXQ algorithms
    calibration_loader = DataLoader(
        dataset=calibration_dataset, batch_size=args.calibration_batch_size, collate_fn=collate_fn)

    validation_dataset = get_dataset_for_model(
        bos_preprocessing=args.bos_preprocessing,
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        split=args.dataset_eval_split,
        seed=args.seed)

    validation_loader = DataLoader(dataset=validation_dataset, batch_size=1, collate_fn=collate_fn)

    if args.fine_tune:
        # Load the data for training
        finetune_dataset = get_dataset_for_model(
            bos_preprocessing=args.bos_preprocessing,
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            nsamples=args.nsamples_rot_calibration,
            seqlen=args.seqlen,
            split="train",
            seed=args.seed)

    device = next(iter(model.parameters())).device
    print("Data loaded.")

    if args.eval:
        print("Float model eval...")
        model = offload_model(model)
        float_metrics = compute_float_evaluation_metrics(
            model=model,
            data=validation_loader,
            context_length=args.seqlen // 2,
            tokenizer=tokenizer,
            seed=args.seed,
            top_k=args.eval_top_k)
        float_ppl = float_metrics.ppl
        float_probabilities = float_metrics.probabilities
        remove_hooks(model)

    # Self-generated calibration text for the frozen 2DP plane, sampled before any transformation.
    selfgen_ids = None
    if (args.dc_extract_qk and args.dc_u_mode == 'frozen' and args.dc_u_calib == 'selfgen' and
            not (args.dc_u_path and os.path.exists(args.dc_u_path))):
        selfgen_ids = generate_selfgen_ids(model, tokenizer, args)

    rmsnorm_classes = ()
    if require_fx:
        with torch.no_grad(), rmsnorm_patch(model, model.config, enabled=args.replace_rmsnorm) as patcher:
            rmsnorm_classes = patcher.rmsnorm_classes
            with make_dynamo_compatible(model) as dynamo_comp, dynamo_export_ctx():
                model, guards = torch._dynamo.export(model)(**next(iter(calibration_loader)))
        # Blockwise optimization does not work with FX at the moment
        args.gpxq_block_name = None
    model.eval()

    # Apply LN affine merging before inserting MHA layers
    # since currently there is support only for merging into Linear
    # Delay affine merging to fused_rotation_no_fx() for fused rotations and permutations.
    if args.ln_affine_merge and args.rotation != 'fused_no_fx' and args.permute_fn is None:
        print("Apply LN affine merge...")
        apply_layernorm_affine_merge(model, rmsnorm_classes)
        print("LN affine merge applied.")

    if args.convert_layernorm_to_rmsnorm:
        print("Convert LayerNorm to RMSNorm...")
        apply_layernorm_to_rmsnorm(model)
        print("Layernorm To RMSNorm applied.")

    if args.dc_extract_qk and args.rotation_sdpa_regions:
        print("[dc-extract-qk] --rotation-sdpa-regions forced off: 2DP handles the Q/K head")
        args.rotation_sdpa_regions = False

    # Insert standard MHA layers when performing fx based weight/act equalization to avoid dealing
    # with all the variability in HF implementations
    if args.quant_sdpa == 'fx':
        print("Replace `F.scaled_dot_product_attention` with QuantSDPA...")
        model = replace_sdpa_with_quantizable_layers(model)
        print("Replacing done.")
    elif args.quant_sdpa == 'functional':
        print("Inserting SDPA quantizable module")
        model = offload_model(model)
        with torch.no_grad(), functional_quantization_mode(model, {torch.nn.functional.scaled_dot_product_attention: ScaledDotProductAttention}):
            model(**next(iter(calibration_loader)))
        remove_hooks(model)
    elif args.quant_sdpa == 'eager':
        model = replace_sdpa_with_quantizable_layers(
            model, is_fx=False, eager_quant_sdpa_class=args.eager_quant_sdpa_class)

    layers_to_expand = []
    if args.rotation is not None:
        for name, _ in model.named_modules():
            if any(map(lambda x: x in name, args.rotation_layers_to_expand)):
                layers_to_expand.append(name)

    print_rotation_summary(args)
    if args.rotation == 'fx':
        model = offload_model(model)
        eq = GraphRotationEqualization(
            orphan_sink=args.rotation_orphan_sink,
            full_rotation_method=args.rotation_mode,
            sdpa_regions=args.rotation_sdpa_regions,
            **sdpa_value_rotation_kwargs(args),
            use_parametrized_rotations=args.fine_tune,
            expansion_step=args.expansion_step,
            layers_to_expand=layers_to_expand,
            rotation_block_size=args.rotation_block_size,
            disable_block_rotation_for_fused=args.disable_block_rotation_for_fused,
            extra_state_kwargs={'equivariant_layers': rmsnorm_classes})
        model = eq.apply(model)
        remove_hooks(model)

    # Permutations are always fused. So, if we are applying them, then we go through
    # the 'fused_no_fx' path to get the permutation-equivariant regions in the graph.
    # If args.rotation == 'layerwise', then the rotations will not be applied in
    # fused_rotation_no_fx(). Rotations will be added in the layerwise block below.
    if args.rotation == 'fused_no_fx' or args.permute_fn is not None:
        fused_rotation_no_fx(model, calibration_loader, args)

    if args.rotation == 'layerwise':
        model = offload_model(model)
        eq = LayerwiseActivationRotation(
            layers_to_expand=layers_to_expand,
            expansion_step=args.expansion_step,
            rotation_block_size=args.rotation_block_size)
        model = eq.apply(model)
        remove_hooks(model)

    # make_dynamo_compatible (fused_no_fx rotation) resets the attention implementation; restore the
    # quantizable eager attention, otherwise the QK^T quantizers are bypassed.
    if args.quant_sdpa == 'eager':
        model.config._attn_implementation = 'quant_sdpa'

    if args.weight_equalization:
        print("Apply weight equalization...")
        # In case of float16 model, we need to offload to account for missing ops
        model = offload_model(model)
        apply_weight_equalization(model)
        remove_hooks(model)
        print("Weight equalization applied.")

    if args.act_equalization is not None:
        offload_model(model)
        print(f"Apply act equalization (SmoothQuant) with alpha {args.act_equalization_alpha}")
        if args.load_checkpoint:
            loader = [next(iter(calibration_loader))]
        else:
            loader = calibration_loader
        apply_act_equalization(
            model, args.act_equalization, loader, alpha=args.act_equalization_alpha)
        print("Act equalization applied.")
        remove_hooks(model)

    if args.magr and not args.load_checkpoint:
        print("Applying MagR...")
        model = offload_model(model)
        apply_magr(
            model,
            calibration_loader,
            create_weight_orig=not args.disable_create_weight_orig,
            alpha=args.magr_alpha,
            buffer_device=args.gpxq_buffer_device)
        remove_hooks(model)
        print(f"MagR applied.")

    if not args.no_quantize:
        name_blacklist = []
        custom_quantizer = None
        print("Applying model quantization...")
        # When AWQ is enabled, the scaling_impl_type for the weights needs to be 'stats', as the
        # scaling factor that multiplies the weights is optimized
        weight_scaling_impl_type = 'stats' if (
            args.awq_scale or args.awq_clip) else 'parameter_from_stats'
        weight_kwargs = {}
        if args.weight_quant_rescaling_init is not None:
            weight_kwargs = {
                **weight_kwargs,
                **{
                    'scaling_affine_rescaling_init': args.weight_quant_rescaling_init,
                    'zero_point_affine_rescaling_init': args.weight_quant_rescaling_init}}
        if args.weight_narrow_range:
            weight_kwargs = {**weight_kwargs, **{'narrow_range': args.weight_narrow_range}}
        input_kwargs = {}
        if args.input_narrow_range:
            input_kwargs = {**input_kwargs, **{'narrow_range': args.input_narrow_range}}
        quantizers_dict = generate_quantizers(
            weight_bit_width=args.weight_bit_width,
            weight_param_method=args.weight_param_method,
            weight_scale_precision=args.weight_scale_precision,
            weight_quant_type=args.weight_quant_type,
            weight_quant_granularity=args.weight_quant_granularity,
            weight_group_size=args.weight_group_size,
            weight_group_dim=args.weight_group_dim,
            weight_scaling_impl_type=weight_scaling_impl_type,
            quantize_weight_zero_point=args.quantize_weight_zero_point,
            weight_quant_format=args.weight_quant_format,
            input_bit_width=args.input_bit_width,
            input_quant_format=args.input_quant_format,
            input_scale_precision=args.input_scale_precision,
            input_scale_type=args.input_scale_type,
            input_param_method=args.input_param_method,
            input_quant_type=args.input_quant_type,
            input_quant_granularity=args.input_quant_granularity,
            input_group_size=args.input_group_size,
            attn_quant_config=args.attn_quant_config,
            attn_bit_width=args.attn_bit_width,
            attn_quant_format=args.attn_quant_format,
            attn_scale_precision=args.attn_scale_precision,
            attn_scale_type=args.attn_scale_type,
            attn_param_method=args.attn_param_method,
            attn_quant_type=args.attn_quant_type,
            attn_quant_granularity=args.attn_quant_granularity,
            attn_group_size=args.attn_group_size,
            quantize_input_zero_point=args.quantize_input_zero_point,
            scale_rounding_func_type=args.scale_rounding_func_type,
            quant_attn_mode='sdpa',
            scaling_min_val=args.scaling_min_val,
            weight_kwargs=weight_kwargs,
            input_kwargs=input_kwargs)
        if args.no_quantize_linear:
            # Attention quantizers only: they are built from the input_* options, the linear ones
            # built alongside are dropped here.
            assert quantizers_dict['k_transposed_quant'] is not None, \
                "--no-quantize-linear needs attention quantizers: set input_bit_width (and attn_*)"
            quantizers_dict['linear_input_quant'] = None
            quantizers_dict['input_quant'] = None
            quantizers_dict['weight_quant'] = None
            print(
                "[qk-only] linear weights and inputs NOT quantized; attention quantizers kept "
                f"(attn_quant_config={args.attn_quant_config})")
        if args.custom_quantizer is not None:
            quantizer_name = parse_custom_quantizer(args.custom_quantizer)
            custom_quantizer = QUANTIZERS_REGISTRY.get(quantizer_name)
            quantizers_dict = custom_quantizer.override_quantizers_dict(quantizers_dict)
        # Quantize the embedding only when first/last-layer quantization is requested.
        # The embedding uses the same weight quantizer as the linear layers.
        # A custom quantizer that branches on module name or type still applies.
        dc_extract_kwargs = None
        if args.dc_extract_qk:
            dc_extract_kwargs = dict(
                num_key_value_heads=getattr(config, 'num_key_value_heads', None),
                dc_sub_dim=args.dc_sub_dim,
                dc_correction=args.dc_correction,
                dc_gamma=args.dc_gamma,
                dc_abar_mode=args.dc_abar_mode,
                dc_sinkbal=args.dc_sinkbal,
                dc_had_rand=args.dc_had_rand,
                dc_had_rand_shared=args.dc_had_rand_shared,
                dc_had_seed=args.dc_had_seed,
                tcast_scale=args.dc_tcast_scale)
        layer_map = generate_quant_maps(
            **quantizers_dict,
            dtype=dtype,
            device=device,
            quantize_embedding=args.quantize_first_last_layer,
            dc_extract_kwargs=dc_extract_kwargs)
        if not args.quantize_first_last_layer:
            # Dynamo tracing changes the name of the modules, thus we need this workaround to pick
            # up the last module.
            if require_fx:
                last_node = [node for node in model.graph.nodes if node.op == 'call_module'][-1]
                last_module = get_module(model, last_node.target)
                # In case we have layerwise rotation/equalization, we need to pick the wrapped module
                last_module = find_equalized_layer(last_module)
                last_layer_kwargs = layer_map[type(last_module)][1]
                prev_weight_quant = deepcopy(last_layer_kwargs['weight_quant'])
                prev_input_quant = deepcopy(last_layer_kwargs['input_quant'])
                # Force the last layer to be unquantized, overriding any custom
                # quantizer definition for it.
                weight_quant = lambda module: prev_weight_quant if id(module) != id(
                    last_module) else None
                input_quant = lambda module: prev_input_quant if id(module) != id(
                    last_module) else None
                last_layer_kwargs['weight_quant'] = weight_quant
                last_layer_kwargs['input_quant'] = input_quant
            else:
                name_blacklist += ["lm_head", "embed_out"]

        model = layerwise_quantize(
            model=model, compute_layer_map=layer_map, name_blacklist=name_blacklist)

        if args.dc_extract_qk:
            assign_layer_ids(model)
            if args.dc_had_layers:
                mods = dc_modules(model)
                had_layers = parse_layer_list(args.dc_had_layers, len(mods))
                for idx, m in mods:
                    if idx in had_layers:
                        m.dc_sub_dim = -1
                print(
                    f"[dc-hybrid] Q/K Hadamard (dc_sub_dim=-1) at layers {sorted(had_layers)}, "
                    f"2DP (dc_sub_dim={args.dc_sub_dim}) elsewhere")

        if custom_quantizer is not None:
            model = custom_quantizer.post_process_quant_model(model)

        # Just to be sure
        model.eval()
        model = model.to(dtype)

        # Tie back first/last layer weights in case they got untied
        print("Model quantization applied.")

    if args.awq_scale or args.awq_clip:
        apply_awq(
            model=model,
            tokenizer=tokenizer,
            calibration_loader=calibration_loader,
            args=args,
            auto_scale=args.awq_scale,
            mse_range=args.awq_clip,
        )

    if not require_fx and tie_weights(args):
        model.tie_weights()

    if args.bias_corr:
        model = add_zero_bias_to_linear(model)

    model = offload_model(model)

    dict_hooks = dict()

    # When offloading to CPU + GPU, the CPU scale factors must be updated
    # before we move them back to the meta device.
    # If we don't, we lose the new value but the internal flag "init_done" is True, thus we will use the wrong scale.
    # To do this, we attach a "hook" to the post_forward function, called before the post_forward
    # The function will update the dict with the initialized scales
    for m in model.modules():
        if hasattr(m, '_hf_hook'):
            if m._hf_hook.weights_map is not None:
                # We store the original function to be restored later
                dict_hooks[m] = m._hf_hook.post_forward
                new_funct = functools.partial(update_internal_dict, m)
                m._hf_hook.post_forward = hooked_on_a_function(m._hf_hook.post_forward, new_funct)

    # If we are doing functional SDPA quantization, we create the correct context manager,
    # otherwise nullcontext. We would love to avoid the extra indentation level but it doesn't seem easy.
    if args.quant_sdpa == "functional":
        quantization_cm = functional_quantization_mode(
            model, {torch.nn.functional.scaled_dot_product_attention: ScaledDotProductAttention})
    else:
        quantization_cm = nullcontext()

    with quantization_cm:
        # We initialize weights scale factor
        with torch.no_grad():
            model(**next(iter(calibration_loader)))

        if args.compile_ptq:
            for m in model.modules():
                if hasattr(m, 'compile_quant'):
                    m.compile_quant()

        if args.sopt_optimize and not args.sopt_optimize_in_gpxq:
            print("Applying Scale Opt...")
            apply_scale_optimization(model, calibration_loader, args)
            print("Scale Optimization applied.")

        if args.act_calibration and not args.load_checkpoint:
            print("Apply act calibration...")
            apply_calibration(model, calibration_loader)
            print("Act calibration applied.")

        if args.fine_tune:
            # Load custom training plugin if specified. The registered Trainer class
            # carries its own ``training_args_cls`` class attribute (which in turn
            # defines the optimizer setup via ``optimizer_scheduler_args``), consumed
            # inside apply_fine_tuning.
            custom_trainer_cls = None

            if args.custom_trainer is not None:
                custom_trainer_config_name = parse_custom_trainer(args.custom_trainer)
                custom_trainer_cls = TRAINER_REGISTRY.get(custom_trainer_config_name)

            fine_tune_extra_args = extra_args if extra_args is not None else []
            if args.load_checkpoint:
                # Skip training when loading from a checkpoint by forcing
                # max_steps to 0 through the training arguments. Appended last so
                # that it overrides any user-provided --max_steps (the argument
                # parser keeps the last value for a repeated flag).
                fine_tune_extra_args += ["--max_steps", "0"]
            apply_fine_tuning(
                model=model,
                tokenizer=tokenizer,
                train_dataset=finetune_dataset,
                collate_fn=collate_fn,
                trainer_cls=custom_trainer_cls,
                extra_args=fine_tune_extra_args)
            # Remove hooks from training
            remove_hooks(model)
            model = offload_model(model)
            # Fuse rotation parametrizations with weights when rotations
            # were used (the function is a no-op when there are none).
            if args.rotation is not None:
                model = fuse_parametrizations(model)

        if args.svd_quant:
            print("Apply SVDQuant...")
            remove_hooks(model)
            model = apply_svd_quant(
                model,
                blacklist=None,
                rank=args.svd_quant_rank,
                iters=args.svd_quant_iters,
                dtype=torch.float32)
            model = offload_model(model)
            with torch.no_grad():
                model(**next(iter(calibration_loader)))
            print("SVDQuant applied.")

        if args.learned_round:
            print("Applying learned round...")
            if args.load_checkpoint:
                args.learned_round_iters = 1
                loader = DataLoader(
                    dataset=[calibration_dataset[0]], batch_size=1, collate_fn=collate_fn)
            else:
                loader = calibration_loader
            remove_hooks(model)
            apply_learned_round(model, loader, args)
            print("Learned round applied.")
            model = offload_model(model)

        if args.load_checkpoint:
            print(f"Loading checkpoint from {args.checkpoint_name}...")
            remove_hooks(model)
            with load_quant_model_mode(model):
                model.load_state_dict(torch.load(args.checkpoint_name, map_location='cpu'))
            model = offload_model(model)
            print("Checkpoint loaded.")

        if args.gptq and not args.load_checkpoint:
            print("Applying GPTQ...")
            scale_optimizer = build_scale_optimizer(
                args, cross_act_objective=False) if args.sopt_optimize_in_gpxq else None
            apply_gptq(
                model,
                calibration_loader,
                act_order=args.gpxq_act_order,
                use_quant_activations=args.gpxq_use_quant_activations,
                create_weight_orig=not args.disable_create_weight_orig,
                block_name=args.gpxq_block_name,
                buffer_device=args.gpxq_buffer_device,
                max_accumulator_bit_width=args.gpxq_max_accumulator_bit_width,
                max_accumulator_tile_size=args.gpxq_max_accumulator_tile_size,
                scale_optimizer=scale_optimizer)

            print("GPTQ applied.")

        if args.gpfq and not args.load_checkpoint:
            print("Applying GPFQ...")
            apply_gpfq(
                model,
                calibration_loader,
                act_order=args.gpxq_act_order,
                block_name=args.gpxq_block_name,
                buffer_device=args.gpxq_buffer_device,
                max_accumulator_bit_width=args.gpxq_max_accumulator_bit_width,
                max_accumulator_tile_size=args.gpxq_max_accumulator_tile_size)
            print("GPFQ applied.")

        if args.qronos and not args.load_checkpoint:
            print("Applying Qronos...")
            # Interleaved Qronos always uses the cross-activation objective
            # (enforced in validate()).
            scale_optimizer = build_scale_optimizer(
                args, cross_act_objective=True) if args.sopt_optimize_in_gpxq else None
            apply_qronos(
                model,
                calibration_loader,
                alpha=args.qronos_alpha,
                act_order=args.gpxq_act_order,
                block_name=args.gpxq_block_name,
                buffer_device=args.gpxq_buffer_device,
                scale_optimizer=scale_optimizer)
            print("Qronos applied.")

        if args.bias_corr and not args.load_checkpoint:
            print("Applying bias correction...")
            apply_bias_correction(model, calibration_loader)
            print("Bias correction applied.")

        # We restore the original behaviour of the post-forward.
        for k, v in dict_hooks.items():
            k._hf_hook.post_forward = v

        # create_weight_orig=True creates a copy of the weights for the model to use when disabling weight
        # quantization so that any downstream optimization can optimize w.r.t. the original reference model.
        # However, it also creates additional tensors that are stored on the CPU, but are cast to the GPU
        # by the zero shot evaluation libraries (e.g., LightEvel), so we remove the `weight_orig` tensors
        # here, if they exist, to save memory.
        remove_weight_orig(model)
        frozen_u_pending = args.dc_extract_qk and args.dc_u_mode == 'frozen' and not args.no_quantize
        if args.eval and not args.no_quantize:
            print("Model eval...")
            with torch.no_grad(), quant_inference_mode(model, compile=args.compile_eval):
                model(**next(iter(calibration_loader)))
                if frozen_u_pending:
                    setup_frozen_u(model, tokenizer, collate_fn, args, selfgen_ids)
                    u_overlap_logging(model, enable=True)
                    frozen_u_pending = False
                if args.sqnr_decomp > 0:
                    from brevitas_examples.llm.sqnr_decomp import run_sqnr_decomp
                    run_sqnr_decomp(
                        model,
                        validation_loader,
                        n_chunks=args.sqnr_decomp,
                        ctx_len=args.seqlen // 2,
                        device=next(model.parameters()).device,
                        tag=f"dc_sub_dim={args.dc_sub_dim}",
                        mf_gamma=args.mf_probe_gamma)
                quant_metrics = compute_quantized_evaluation_metrics(
                    model=model,
                    data=validation_loader,
                    context_length=args.seqlen // 2,
                    tokenizer=tokenizer,
                    reference_probabilities=float_probabilities,
                    seed=args.seed)
                print_u_overlap(model, "ppl")
            quant_ppl = quant_metrics.ppl
            # NOTE: EAR and KLD are by default normalized by reference top-K probability mass.
            quant_ear = quant_metrics.ear
            quant_kld = quant_metrics.kld
            print(f"Quantized perplexity ({args.dataset}): {quant_ppl:.3f}")
            print(
                f"Quantized expected acceptance rate @ top-{args.eval_top_k} "
                f"({args.dataset}): {quant_ear:.6f}")
            print(
                f"Quantized KL divergence @ top-{args.eval_top_k} "
                f"({args.dataset}): {quant_kld:.6f}")
        few_shot_eval_results = dict()
        if args.few_shot_eval == 'lm_eval':
            from lm_eval import evaluator
            from lm_eval.models.huggingface import HFLM
            with torch.no_grad(), quant_inference_mode(model, compile=args.compile_eval):
                model(**next(iter(calibration_loader)))
                if frozen_u_pending:
                    setup_frozen_u(model, tokenizer, collate_fn, args, selfgen_ids)
                    u_overlap_logging(model, enable=True)
                    frozen_u_pending = False
                batch_size = 'auto' if args.few_shot_override_batch_size is None else args.few_shot_override_batch_size
                wrapped_model = HFLM(
                    pretrained=model,
                    add_bos_token=args.bos_preprocessing is not None,
                    batch_size=batch_size)  # need to wrap for LLM eval
                few_shot_eval_results = evaluator.simple_evaluate(
                    model=wrapped_model,
                    model_args=None,
                    tasks=list(args.few_shot_tasks),
                    device='cuda:0',
                    limit=args.few_shot_limit,
                    num_fewshot=0 if args.few_shot_zeroshot else None,
                    log_samples=args.few_shot_samples_out is not None,
                    batch_size=None,
                    verbosity="ERROR")
            if args.few_shot_samples_out is not None:
                dump_few_shot_samples(few_shot_eval_results, args.few_shot_samples_out)
            few_shot_eval_results = filter_results(few_shot_eval_results, args.few_shot_tasks)
            print("Few shot eval results")
            pprint.pprint(few_shot_eval_results)
            zeroshot_avg = compute_zeroshot_avg(few_shot_eval_results, args.few_shot_tasks)
            if zeroshot_avg is not None:
                few_shot_eval_results['zeroshot_avg'] = zeroshot_avg
                print(f"Zero-shot avg ({len(args.few_shot_tasks)} tasks): {zeroshot_avg:.4f}")
            print_u_overlap(model, "zero-shot")
        elif args.few_shot_eval == 'lighteval':

            with torch.no_grad(), quant_inference_mode(model, compile=args.compile_eval):
                model(**next(iter(calibration_loader)))
                remove_hooks(model)

                from brevitas_examples.llm.eval_lighteval import run_lighteval
                few_shot_eval_results = run_lighteval(
                    model_name=args.model,
                    model=model,
                    tasks=args.few_shot_tasks,
                    dtype=args.dtype,
                    batch_size=args.few_shot_override_batch_size,
                    max_samples=args.few_shot_limit,
                )
            # Print nicely formatted results
            pprint.pprint(few_shot_eval_results)
        remove_hooks(model)
        if args.checkpoint_name is not None and not args.load_checkpoint:
            print(f"Saving checkpoint to {args.checkpoint_name}")
            torch.save(model.state_dict(), args.checkpoint_name)

        if args.export_target:
            print(f"Export to {args.export_target}")
            model_export(model, tokenizer, next(iter(calibration_loader)), args, config)

    eval_results = {
        "float_ppl": float_ppl,
        "quant_ppl": quant_ppl,
        "quant_ear": quant_ear,
        "quant_kld": quant_kld}
    return {**eval_results, **few_shot_eval_results}, model


def main():
    overrides = override_defaults(sys.argv[1:])
    parser = create_args_parser()
    args, extra_args = parse_args(parser, sys.argv[1:], override_defaults=overrides)
    quantize_llm(args, extra_args)


if __name__ == '__main__':
    main()
