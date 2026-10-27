# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser
from argparse import ArgumentTypeError
from argparse import Namespace
from typing import List
from typing import Optional
from warnings import warn

from brevitas_examples.common.parse_utils import add_bool_arg
from brevitas_examples.common.parse_utils import create_entrypoint_args_parser
from brevitas_examples.common.parse_utils import quant_format_validator
from brevitas_examples.llm.llm_quant.piso_args import ScaleOptimizerArgumentParser

# Export targets that do not require the (optional) `gguf` dependency
BASE_EXPORT_TARGETS = [
    None,
    'vllm',
    'shark',
    'onnx_qcdq',
    'sharded_torchmlir_group_weight',
    'sharded_packed_torchmlir_group_weight']


def lazy_validate_export_target(value: str):
    """Validate --export-target; lazily importing GGUF export targets"""
    if value in BASE_EXPORT_TARGETS:
        return value
    if value.startswith('gguf:'):
        from brevitas_examples.llm.gguf_export.targets import GGUF_EXPORT_TARGETS
        if value in GGUF_EXPORT_TARGETS:
            return value
        raise ArgumentTypeError(
            f"invalid --export-target {value!r}; expected one of {GGUF_EXPORT_TARGETS}")
    raise ArgumentTypeError(
        f"invalid --export-target {value!r}; expected one of "
        f"{sorted(t for t in BASE_EXPORT_TARGETS if t is not None)} or 'gguf:<ftype>'")


def create_args_parser() -> ArgumentParser:
    parser = create_entrypoint_args_parser(description="LLM quantization")
    parser.add_argument(
        '--model',
        type=str,
        default="facebook/opt-125m",
        help='HF model name. Default: facebook/opt-125m.')
    parser.add_argument(
        '--custom-quantizer',
        type=str,
        default=None,
        help=
        'Override the quantization list and/or post-process the quantized model with a user-defined quantization plugin. The plugin can be a registered name or a .py file path followed by :plugin_name. Default: None.'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help='Data type for model. Default: %(default)s')
    parser.add_argument(
        '--seed', type=int, default=0, help='Seed for sampling the calibration data. Default: 0.')
    parser.add_argument(
        '--nsamples',
        type=int,
        default=128,
        help=
        'Number of calibration data samples. Set to -1 to load the entire dataset. Default: 128.')
    parser.add_argument(
        '--nsamples-rot-calibration',
        type=int,
        default=800,
        help='Number of calibration data samples for rotation. Default: %(default)d.')
    parser.add_argument('--seqlen', type=int, default=2048, help='Sequence length. Default: 2048.')
    parser.add_argument(
        '--eval', action='store_true', help='Eval model PPL, EAR, and KLD on the chosen dataset.')
    parser.add_argument(
        '--eval-top-k',
        type=int,
        default=10,
        help='Number of tokens to evaluate EAR and KLD on. Default: 10.')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['wikitext2', 'c4', 'pile', 'fineweb'],
        default='wikitext2',
        help='Dataset to use for quantization (default: %(default)s)')
    parser.add_argument(
        '--dataset-eval-split',
        type=str,
        choices=['validation', 'test'],
        default='validation',
        help='Specify which split to use for the evaluation dataset (default: %(default)s)')
    parser.add_argument(
        '--gpxq-block-name',
        '--block-name',
        type=str,
        default=None,
        help=
        'Attribute for model blocks. Used for faster GPxQ optimization (if FX is not needed) and learned round (default: %(default)s)'
    )
    parser.add_argument(
        '--gpxq-buffer-device',
        type=str,
        choices=['cpu', 'same'],
        default='cpu',
        help=
        'Device for GPxQ buffers. "same" means using the same device for the buffer as the layer weights. Default %(default)s'
    )
    parser.add_argument(
        '--weight-bit-width', type=int, default=8, help='Weight bit width. Default: 8.')
    parser.add_argument(
        '--weight-param-method',
        type=str,
        default='stats',
        choices=['stats', 'mse', 'hqo'],
        help='How scales/zero-point are determined. Default: stats.')
    parser.add_argument(
        '--weight-scale-precision',
        type=str,
        default='float_scale',
        choices=['signed_float_scale', 'float_scale', 'po2_scale'],
        help='Whether scale is a float value or a po2. Default: %(default)s.')
    parser.add_argument(
        '--weight-quant-rescaling-init',
        type=float,
        default=None,
        help='Value to rescale weight quantization grid. Default: None (no rescaling).')
    parser.add_argument(
        '--weight-narrow-range',
        action="store_true",
        help='Use narrow range for weight quantization. Default: False.')
    parser.add_argument(
        '--weight-quant-type',
        type=str,
        default='sym',
        choices=['sym', 'asym'],
        help='Weight quantization type. Default: asym.')
    parser.add_argument(
        '--weight-quant-format',
        type=quant_format_validator,
        default='int',
        help=
        'Weight quantization type. Either int or eXmY, with X+Y==weight_bit_width-1. It\'s possible to add float_ocp_ or float_fnuz_ before the exponent/mantissa bitwidth. Default: int.'
    )
    parser.add_argument(
        '--weight-quant-granularity',
        type=str,
        default='per_group',
        choices=['per_channel', 'per_tensor', 'per_group'],
        help='Granularity for scales/zero-point of weights. Default: per_group.')
    parser.add_argument(
        '--scale-rounding-func-type',
        type=str,
        default=None,
        choices=['round', 'ceil', 'floor', 'midmax'],
        help='Rounding function to use with Po2 scale. Default: None.')
    parser.add_argument(
        '--weight-group-dim',
        type=int,
        default=None,
        choices=[1, 0],
        help='Override default group_dim for groupsize quantization. Default: layer-dependant')
    parser.add_argument(
        '--weight-group-size',
        type=int,
        default=128,
        help='Group size for per_group weight quantization. Default: 128.')
    parser.add_argument(
        '--quantize-weight-zero-point', action='store_true', help='Quantize weight zero-point.')
    parser.add_argument(
        '--input-bit-width',
        type=int,
        default=None,
        help='Input bit width. Default: None (disables input quantization).')
    parser.add_argument(
        '--input-quant-format',
        type=quant_format_validator,
        default='int',
        help=
        'Input quantization type. Either int or eXmY, with X+Y==weight_bit_width-1. It\'s possible to add float_ocp_ or float_fnuz_ before the exponent/mantissa bitwidth. Default: int.'
    )
    parser.add_argument(
        '--input-param-method',
        type=str,
        default='stats',
        choices=['stats', 'mse'],
        help=
        'How scales/zero-point are determined. Default: stats (percentile for static, absmax or minmax for dynamic).'
    )
    parser.add_argument(
        '--input-scale-precision',
        type=str,
        default='float_scale',
        choices=['signed_float_scale', 'float_scale', 'po2_scale'],
        help='Whether input scale is a float value or a po2. Default: float.')
    parser.add_argument(
        '--input-scale-type',
        type=str,
        default='static',
        choices=['static', 'dynamic', 'no_scale'],
        help='Whether input scale is a static value or a dynamic value.')
    parser.add_argument(
        '--input-quant-type',
        type=str,
        default='asym',
        choices=['sym', 'asym'],
        help='Input quantization type. Default: asym.')
    parser.add_argument(
        '--input-narrow-range',
        action="store_true",
        help='Use narrow range for input quantization. Default: False.')
    parser.add_argument(
        '--input-quant-granularity',
        type=str,
        default='per_tensor',
        choices=['per_tensor', 'per_row', 'per_group'],
        help='Granularity for scales/zero-point of inputs. Default: per_tensor.')
    parser.add_argument(
        '--input-group-size',
        type=int,
        default=64,
        help='Group size for per_group input quantization. Default: 64.')
    parser.add_argument(
        '--attn-quant-config',
        type=str,
        default='qkvs',
        choices=['qkvs', 'qkv', 'kv', 'qk'],
        help=
        'Decide which parts of attention should be quantized. "kv" will only quantize KV, "qk" only the operands of QK^T, "qkvs" will quantize all MatMuls in attention (QKV & Softmax output). Note: --quant-sdpa needs be set for this to have an effect. Default: %(default)s'
    )
    parser.add_argument(
        '--attn-bit-width',
        type=int,
        default=None,
        help='Attention bit width. Default: None (same as input).')
    parser.add_argument(
        '--attn-quant-format',
        type=attn_quant_format_validator,
        default=None,
        help=
        'Attention quantization type. Either int or eXmY, with X+Y==weight_bit_width-1. It\'s possible to add float_ocp_ or float_fnuz_ before the exponent/mantissa bitwidth. Default: None (same as input).'
    )
    parser.add_argument(
        '--attn-param-method',
        type=str,
        default=None,
        choices=['stats', 'mse'],
        help='How scales/zero-point are determined. Default: None (same as input).')
    parser.add_argument(
        '--attn-scale-precision',
        type=str,
        default=None,
        choices=['signed_float_scale', 'float_scale', 'po2_scale'],
        help='Whether input scale is a float value or a po2. Default: (same as input).')
    parser.add_argument(
        '--attn-scale-type',
        type=None,
        default=None,
        choices=['static', 'dynamic', 'no_scale'],
        help='Whether input scale is a static value or a dynamic value. Default: (same as input).')
    parser.add_argument(
        '--attn-quant-type',
        type=str,
        default=None,
        choices=['sym', 'asym'],
        help='Input quantization type. Default: (same as input).')
    parser.add_argument(
        '--attn-quant-granularity',
        type=str,
        default=None,
        choices=['per_tensor', 'per_row', 'per_group'],
        help='Granularity for scales/zero-point of inputs. Default: (same as input).')
    parser.add_argument(
        '--attn-group-size',
        type=int,
        default=None,
        help='Group size for per_group input quantization. Default: (same as input).')
    parser.add_argument(
        '--learned-round-lr',
        type=float,
        default=5e-3,
        help='Learning rate for learned round parameter optimization. Default: %(default)s')
    parser.add_argument(
        '--learned-round-scale-lr',
        type=float,
        default=1e-2,
        help='Learning rate for scale optimization during round learning. Default: %(default)s')
    parser.add_argument(
        '--learned-round-scale-momentum',
        type=float,
        default=0.9,
        help='Learning rate for scale optimization during round learning. Default: %(default)s')
    parser.add_argument(
        '--learned-round-iters',
        type=int,
        default=200,
        help='Number of iterations for learned round. Default: 200.')
    parser.add_argument(
        '--learned-round-scale',
        action='store_true',
        help='Learned scale factor together with round.')
    parser.add_argument(
        '--quantize-input-zero-point', action='store_true', help='Quantize input zero-point.')
    parser.add_argument(
        '--quantize-first-last-layer',
        action='store_true',
        help=
        'Quantize the first (embedding) and last nn.Linear layer. When not set, the embedding is '
        'left unquantized and the last layer weight quantizer is forced to None, overriding any '
        'definition from a custom quantizer.')
    parser.add_argument('--magr', action='store_true', help='Apply MagR.')
    parser.add_argument(
        '--magr-alpha', type=float, default=0.01, help='Alpha for MagR. Default: 0.01.')
    parser.add_argument('--qronos', action='store_true', help='Apply Qronos.')
    parser.add_argument(
        '--qronos-alpha', default=1e-6, type=float, help='Alpha for Qronos. Default: 1e-6')
    parser.add_argument('--gptq', action='store_true', help='Apply GPTQ.')
    parser.add_argument('--gpfq', action='store_true', help='Apply GPFQ.')
    parser.add_argument(
        '--gpxq-act-order', action='store_true', help='Apply GPxQ activation ordering.')
    parser.add_argument(
        '--gpxq-use-quant-activations',
        action='store_true',
        help='Use quantized activations in GPxQ.')
    parser.add_argument(
        '--disable-create-weight-orig',
        action='store_true',
        help='Disable maintaining original weights for non-quant forward pass. Default: false')
    parser.add_argument(
        '--gpxq-max-accumulator-bit-width',
        type=int,
        default=None,
        help='Maximum accumulator bit width for GPxQ using AXE.')
    parser.add_argument(
        '--gpxq-max-accumulator-tile-size',
        type=int,
        default=None,
        help='Maximum accumulator tile size for GPxQ using AXE.')
    parser.add_argument(
        '--act-calibration', action='store_true', help='Apply activation calibration.')
    parser.add_argument('--bias-corr', action='store_true', help='Apply bias correction.')
    parser.add_argument('--ln-affine-merge', action='store_true', help='Merge LN affine params.')
    parser.add_argument(
        '--convert-layernorm-to-rmsnorm', action='store_true', help='Merge LN affine params.')
    parser.add_argument(
        '--replace-rmsnorm', action='store_true', help='Replace HF RMSNorms with Torch one.')
    parser.add_argument('--no-quantize', action='store_true', help='Disable quantization.')
    parser.add_argument(
        '--no-quantize-linear',
        action='store_true',
        help='Keep the weights and inputs of linear layers in full precision and quantize only '
        'attention (per --attn-quant-config). The input_* options must still be set: they are the '
        'defaults of the attn_* options.')
    parser.add_argument(
        '--scaling-min-val',
        type=float,
        default=1e-4,
        help='Minimum value to clamp scale to when using bf16 or fp16 quantization.')
    parser.add_argument(
        '--quant-sdpa',
        type=str,
        choices=['eager', 'functional', 'fx'],
        default=None,
        help='Define how to quantize SDPA. (default: %(default)s)')
    parser.add_argument(
        '--eager-quant-sdpa-class',
        type=str,
        default='auto',
        help=
        'If quant_sdpa is eager, specify the name of the attention class. (default: %(default)s)')
    parser.add_argument(
        '--weight-equalization',
        action='store_true',
        help='Apply weight equalization. Relevant to ReLU based models (e.g. OPT).')
    parser.add_argument(
        '--rotation',
        type=str,
        default=None,
        choices=['fx', 'layerwise', 'fused_no_fx'],
        help='Apply graph rotation equalization')
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        default=False,
        help="Enable the training / fine-tuning pipeline. When used with "
        "--rotation, rotation matrices are parametrized and optimized. "
        "When used without --rotation, a custom trainer plugin can "
        "fine-tune all model weights (default: %(default)s).",
    )
    parser.add_argument(
        "--optimize-rotations",
        action="store_true",
        default=False,
        dest="optimize_rotations",
        help="Deprecated: use --fine-tune instead. Implies --fine-tune "
        "(default: %(default)s).",
    )
    parser.add_argument(
        '--custom-trainer',
        type=str,
        default=None,
        help='Path to a custom training plugin, in the format '
        '"path/to/plugin.py:config_name". The plugin .py file should register a '
        'custom Trainer class into the TRAINER_REGISTRY. The Trainer class may '
        'expose a "training_args_cls" class attribute to customise the training '
        'arguments (including the optimizer setup via "optimizer_scheduler_args"); '
        'otherwise the LLM example defaults are used. Implies --fine-tune. '
        'Default: None.',
    )
    parser.add_argument(
        '--rotation-mode',
        default='had',
        choices=['had', 'ort', 'had_rand'],
        help=
        'If GraphRotation is enabled, decide how to compute the random rotation matrix that is fully fused. Online or partial rotation will always be Hadamard. had_rand: Hadamard with random sign flips'
    )
    parser.add_argument(
        '--rotation-orphan-sink',
        action="store_true",
        help=
        'If GraphRotation is enabled, decide wheter to add standalone hadamard matrices for the unfused layers'
    )
    parser.add_argument(
        '--rotation-sdpa-regions',
        action="store_true",
        help='If GraphRotation is enabled, decide wheter to equalize across SDPA')
    parser.add_argument(
        '--rotation-sdpa-value-only',
        action="store_true",
        help=
        'If GraphRotation is enabled, rotate only the SDPA value region (R2, fused into the value '
        'and output projections), not the Q/K head. Ignored with --rotation-sdpa-regions.')
    parser.add_argument(
        '--rotation-sdpa-value-det',
        action="store_true",
        help='Like --rotation-sdpa-value-only, with a deterministic Hadamard R2 even when '
        '--rotation-mode is had_rand (QuaRot setting: random R1, deterministic R2).')
    parser.add_argument(
        '--rotation-sdpa-value-rand',
        action="store_true",
        help='Like --rotation-sdpa-value-only, with an independent random Hadamard R2 per block, '
        'whatever --rotation-mode.')
    parser.add_argument(
        '--rotation-sdpa-value-rand-seed',
        default=0,
        type=int,
        help='Seed of the random R2 draws. Default: %(default)s')
    parser.add_argument(
        '--rotation-sdpa-value-rand-shared',
        action="store_true",
        help='With --rotation-sdpa-value-rand, use one random R2 for all blocks.')
    parser.add_argument(
        '--rotation-layers-to-expand',
        type=str,
        default=[],
        nargs='*',
        help='A list of module names to expand with hadamard rotation. Default: %(default)s')
    parser.add_argument(
        '--expansion-step',
        type=int,
        default=1,
        help=
        'When layer expansion is set, decide how much to increase the layer sizes. Default: %(default)s'
    )
    parser.add_argument(
        '--rotation-block-size',
        type=int,
        default=None,
        help='Perform blockwise rotations when possible. Default: %(default)s')
    parser.add_argument(
        '--disable-block-rotation-for-fused',
        action='store_true',
        help='Disable block rotations when using fused rotations. Default: %(default)s')
    parser.add_argument('--svd-quant', action='store_true', help='Apply SVDQuant.')
    parser.add_argument(
        '--svd-quant-rank',
        type=int,
        default=32,
        help='Rank to use for SVDQuant (default: %(default)s).')
    parser.add_argument(
        '--svd-quant-iters',
        type=int,
        default=1,
        help='Number of iterations to use for SVDQuant (default: %(default)s).')
    parser.add_argument(
        '--act-equalization',
        default=None,
        choices=[None, 'layerwise', 'fx'],
        help='Apply activation equalization (SmoothQuant). Layerwise introduces standalone mul nodes,'
        'while fx merges them whenever possible into previous tensors, which is possible on ReLU based models (e.g. OPT).'
    )
    parser.add_argument(
        '--act-equalization-alpha',
        default=0.5,
        type=float,
        help='If activation equalization is enabled, decide what alpha to use')
    parser.add_argument(
        '--permute-fn',
        choices=['absmax', 'massdiff', 'zigzag', 'random'],
        default=None,
        help=
        'Permutation function to use. If None, no permutation is applied. Works with block rotations when both are enabled.'
    )
    parser.add_argument(
        '--export-target',
        default=None,
        type=lazy_validate_export_target,
        help="Model export. One of 'vllm', 'shark', 'onnx_qcdq', 'sharded_torchmlir_group_weight', "
        "'sharded_packed_torchmlir_group_weight', or 'gguf:<ftype>' (e.g. 'gguf:q4_k_s'; see "
        "brevitas_examples.llm.gguf_export.targets.GGUF_EXPORT_TARGETS for the full list of "
        "supported ftypes). Default: %(default)s")
    parser.add_argument(
        '--export-prefix',
        type=str,
        default=None,
        help=
        "Path prefix to use for the various export flows. If None, a path will be derived from the model name (default: %(default)s)"
    )
    parser.add_argument(
        '--export-path',
        type=str,
        default=None,
        help="Output path for GGUF export. If it ends in '.gguf', treated as the exact file to write "
        "(parent directories created as needed); otherwise treated as a directory to write into "
        "using gguf-py's auto-derived filename. If None, writes to the current working directory. "
        "Default: %(default)s")
    parser.add_argument(
        '--checkpoint-name',
        type=str,
        default=None,
        help="Filename to save checkpoint. If `None`, no checkpoint is saved (default: %(default)s)"
    )
    parser.add_argument(
        '--load-checkpoint',
        action="store_true",
        help='Boolean flag to load_checkpoint, uses checkpoint_name. Default %(default)s)')
    parser.add_argument(
        '--learned-round',
        default=None,
        choices=[None, 'identity'],
        help='Whether to use learned round. If `None`, RTN is used (default: %(default)s)')
    parser.add_argument(
        '--learned-round-fast-update',
        default=False,
        action="store_true",
        help='Whether to use fast update with learned round. Prototype (default: %(default)s)')
    parser.add_argument(
        '--few-shot-eval',
        type=str,
        default=None,
        choices=['lm_eval', 'lighteval'],
        help='Perform zero_shot evaluation with lm_eval or lighteval. Default %(default)s)')
    parser.add_argument('--few-shot-override-batch-size', type=int, default=None)
    parser.add_argument(
        '--compile-ptq',
        default=False,
        action="store_true",
        help='Compile for PTQ algorithms. Default %(default)s)')
    parser.add_argument(
        '--compile-eval',
        default=False,
        action="store_true",
        help='Compile during evaluation. Default %(default)s)')
    parser.add_argument(
        '--few-shot-zeroshot',
        default=False,
        action="store_true",
        help='Whether to do zero or few shot eval. Default %(default)s)')
    parser.add_argument(
        '--bos-preprocessing',
        type=str,
        default=None,
        choices=[None, 'document', 'sequence'],
        help=
        'Type of BOS token pre-processing for training and evaluation datasets. Default %(default)s)'
    )
    parser.add_argument(
        '--few-shot-limit', type=int, default=None, help='Few shot limit. Default %(default)s)')
    parser.add_argument(
        '--few-shot-samples-out',
        type=str,
        default=None,
        help='lm_eval only: write the per-candidate log-likelihoods of every question to this JSON '
        'file, for paired comparisons between runs. Default %(default)s)')
    parser.add_argument(
        '--few-shot-tasks',
        default=['arc_challenge', 'arc_easy', 'winogrande', 'piqa'],
        type=str,
        nargs='*',
        help='A list of tasks for zero_shot evaluation. Default: %(default)s')
    parser.add_argument(
        "--awq-scale",
        action="store_true",
        help="Whether to apply AWQ scaling (default: %(default)s).",
    )
    parser.add_argument(
        "--awq-clip",
        action="store_true",
        help="Whether to apply AWQ clipping (default: %(default)s).")

    parser.add_argument(
        '--calibration-batch-size',
        type=int,
        default=1,
        help='Batch size for calibration data loader. (default: %(default)s).')
    add_dc_extract_args(parser)

    # Adds Scale Optimization (--sopt-*) flags (PiSO being the current solver instance behind them).
    ScaleOptimizerArgumentParser.add_arguments(parser)

    return parser


def add_dc_extract_args(parser: ArgumentParser):
    """Options of 2DP (exact DC plane + quantized residual for QK^T), see dc_extract_sdpa.py."""
    group = parser.add_argument_group('2DP (QK^T with an exact DC plane)')
    group.add_argument(
        '--dc-extract-qk',
        action="store_true",
        help='Compute QK^T with the 2DP attention module. Requires --quant-sdpa. Forces '
        '--rotation-sdpa-regions off: the Q/K head is handled by the module (--dc-sub-dim).')
    group.add_argument(
        '--dc-sub-dim',
        type=int,
        default=2,
        help='Rank of the exact plane (2 = 2DP). < 0: full head-dim Hadamard on Q/K before '
        'quantization (QuaRot R3). 0: quantize Q/K unchanged. Default: %(default)s')
    group.add_argument(
        '--dc-correction',
        type=str,
        default='none',
        choices=['none', 'unshrink', 'mean', 'scale'],
        help='Score shrinkage correction: unshrink = divide the quantized score by 1 - gamma; mean = '
        'subtract gamma times the score of the mean query plane coordinate; scale = scale the exact '
        'plane by 1 - gamma. Default: %(default)s')
    group.add_argument(
        '--dc-gamma',
        type=float,
        default=0.0,
        help='gamma of --dc-correction, e.g. the gamma_attn(pred) printed by --sqnr-decomp. '
        'Default: %(default)s')
    group.add_argument(
        '--dc-abar-mode',
        type=str,
        default='full',
        choices=['full', 'causal'],
        help='Mean query coordinate of --dc-correction mean: over the whole sequence or over '
        'positions <= i. Default: %(default)s')
    group.add_argument(
        '--dc-sinkbal',
        action="store_true",
        help='Oracle diagnostic (uses the unquantized score): set the sink-key score error to the '
        'attention-weighted error of the other keys.')
    group.add_argument(
        '--dc-had-rand',
        action="store_true",
        help='With --dc-sub-dim < 0, randomize the Q/K Hadamard with sign flips (one per layer).')
    group.add_argument(
        '--dc-had-rand-shared',
        action="store_true",
        help='With --dc-had-rand, use the same sign flips in every layer.')
    group.add_argument(
        '--dc-had-seed', type=int, default=0, help='Seed of --dc-had-rand. Default: %(default)s')
    group.add_argument(
        '--dc-had-layers',
        type=str,
        nargs='*',
        default=[],
        help='Decoder layers that use the Q/K Hadamard (--dc-sub-dim -1) instead of 2DP: indices, '
        'inclusive ranges A-B, "first" or "last". Default: %(default)s')
    group.add_argument(
        '--dc-tcast-scale',
        type=str,
        default=None,
        help='Replace the output of the attention quantizers by a tcast virtual cast to e2m1 '
        'elements with this block-scale spec (e.g. e4m3fn_t32 for an E4M3 block-32 scale). '
        'Requires the tcast package. Default: %(default)s')
    group.add_argument(
        '--dc-u-mode',
        type=str,
        default='live',
        choices=['live', 'frozen'],
        help='live: estimate the plane per sequence. frozen: one plane per layer and KV group, '
        'loaded from --dc-u-path if it exists, else calibrated (and saved to --dc-u-path if set). '
        'Default: %(default)s')
    group.add_argument(
        '--dc-u-path',
        type=str,
        default=None,
        help='File of the frozen plane. Default: %(default)s')
    group.add_argument(
        '--dc-u-calib',
        type=str,
        default='wikitext2',
        choices=['wikitext2', 'random', 'selfgen'],
        help='Calibration data of the frozen plane: --nsamples-rot-calibration WikiText-2 train '
        'windows, as many uniform random token sequences, or --dc-selfgen-nsamples sequences sampled '
        'from the float model. Default: %(default)s')
    group.add_argument(
        '--dc-u-calib-seqlen',
        type=int,
        default=None,
        help='Sequence length of random/selfgen calibration sequences. Default: --seqlen')
    group.add_argument(
        '--dc-u-basis',
        type=str,
        default='mean',
        choices=['mean', 'pca'],
        help='Frozen plane: QR of the query and key means, or top-2 eigenvectors of the normalized '
        'Q and K second moments. Default: %(default)s')
    group.add_argument(
        '--dc-u-space',
        type=str,
        default='lf',
        choices=['lf', 'full'],
        help='Frozen plane on the low-frequency half of the head or on the full head. '
        'Default: %(default)s')
    group.add_argument(
        '--dc-u-pca-center',
        action="store_true",
        help='With --dc-u-basis pca, use centered second moments (covariances).')
    group.add_argument(
        '--dc-u-ref',
        type=str,
        nargs='*',
        default=[],
        help='Frozen-plane files to compare a newly calibrated plane with (subspace overlap).')
    group.add_argument(
        '--dc-selfgen-nsamples',
        type=int,
        default=64,
        help='Number of self-generated calibration sequences. Default: %(default)s')
    group.add_argument(
        '--dc-selfgen-temperature',
        type=float,
        default=1.0,
        help='Sampling temperature of the self-generated sequences. Default: %(default)s')
    group.add_argument(
        '--dc-selfgen-top-p',
        type=float,
        default=0.95,
        help='Nucleus (top-p) of the self-generated sequences. Default: %(default)s')
    group.add_argument(
        '--dc-selfgen-batch-size',
        type=int,
        default=16,
        help='Generation batch size of the self-generated sequences. Default: %(default)s')
    group.add_argument(
        '--sqnr-decomp',
        type=int,
        default=0,
        help='Before the quantized PPL, print the per-layer error decomposition over this many '
        'evaluation chunks (0: off). Needs --dc-extract-qk for the attention part. '
        'Default: %(default)s')
    group.add_argument(
        '--mf-probe-gamma',
        type=float,
        default=None,
        help='Add the m/f probe at this gamma to --sqnr-decomp. Default: %(default)s')


def fx_required(args: Namespace):
    _is_required = False
    if args.act_equalization == 'fx':
        _is_required = True
    if args.rotation == 'fx':
        _is_required = True
    if args.quant_sdpa == 'fx':
        _is_required = True
    # Weight equalization requires FX to identify sources and sinks.
    if args.weight_equalization:
        _is_required = True
    # LayerNorm conversion requires FX to identify the source layers that it must update.
    if args.convert_layernorm_to_rmsnorm:
        _is_required = True
    # Affine normalization merging requires FX to identify all sinks.
    # The fused_no_fx and permutation paths create their own FX graphs.
    if args.ln_affine_merge and args.rotation != 'fused_no_fx' and args.permute_fn is None:
        _is_required = True
    return _is_required


def dynamo_required(args: Namespace):
    return fx_required(args) or args.rotation == 'fused_no_fx' or args.permute_fn is not None


def validate(args: Namespace, extra_args: Optional[List[str]] = None) -> None:
    assert args.eval_top_k > 0, 'Error: eval_top_k must be positive.'
    # --optimize-rotations is a deprecated alias for --fine-tune
    if args.optimize_rotations:
        warn(
            "--optimize-rotations is deprecated, use --fine-tune instead.",
            DeprecationWarning,
            stacklevel=2)
        args.fine_tune = True
    if not args.fine_tune:
        assert extra_args is None or len(extra_args) == 0, f"The following unknown arguments were passed: {[extra_arg for extra_arg in extra_args if extra_arg.startswith('--')]}"
    if args.export_target == 'onnx_qcdq':
        try:
            from optimum.exporters.onnx import onnx_export_from_model
        except ImportError as e:
            raise ImportError(
                "ONNX QCDQ export of LLMs requires `optimum`, which is an optional "
                "dependency. Install it with `pip install \"brevitas[llm,export,llm_onnx_export]\"`."
            ) from e
        del onnx_export_from_model
    if args.rotation == 'fx':
        assert args.ln_affine_merge, 'Graph rotation requires to merge LN/RMS norm affine parameters'
        assert args.replace_rmsnorm, 'Graph rotation requires to replace HF RMSNorm with PyTorch ones (torch 2.4+ require)'
        assert args.convert_layernorm_to_rmsnorm, 'Graph rotation requires to replace LayerNorm with RMSNorm'
        # FX is not compatible with few-shot evaluation
        assert args.few_shot_eval is None, "FX is not compatible with few shot evaluation, use fused_no_fx"
    # Otherwise we might end up tracing through dynamo twice and other weird errors.
    # Fused_no_fx takes care of all the rotations-related transformations
    if fx_required(args):
        assert args.rotation != "fused_no_fx", "fused_no_fx is incompatible with any option that requires FX tracing"
    elif args.rotation == 'fused_no_fx':
        assert not args.convert_layernorm_to_rmsnorm, 'LayerNorm is automatically replaced with RMSNorm when running with --rotation=fused_no_fx. Remove the flag --convert-layernorm-to-rmsnorm'
        assert args.replace_rmsnorm, 'Graph rotation requires to replace HF RMSNorm with PyTorch ones (torch 2.4+ require)'
    # Re-validate for callers that build `args` directly without going through argparse (e.g. benchmark sweeps).
    if args.export_target is not None:
        lazy_validate_export_target(args.export_target)
    if not args.no_quantize:
        if args.weight_quant_rescaling_init is not None:
            assert args.weight_quant_rescaling_init > 0, \
                'Error: weight_quant_rescaling_init must be positive.'
        assert (int(args.gptq) + int(args.gpfq) + int(args.qronos)) <= 1, "GPTQ, GPFQ, and/or Qronos are enabled together."
        if (args.gpfq or args.qronos):
            # create_weight_orig=True creates a copy of the weights for the model to use
            # when disabling weight quantization so that any downstream optimization can
            # optimize w.r.t. the original reference model. This is required for GPFQ and
            # Qronos as it would otherwise consistently degrade performance.
            assert not args.disable_create_weight_orig, \
                'Error: create_weight_orig is required with GPFQ and Qronos'
        if args.gpxq_max_accumulator_bit_width is not None:
            assert args.weight_quant_format == 'int', "AXE only supports integer formats."
            assert args.input_quant_format == 'int', "AXE only supports integer formats."
            assert args.input_bit_width is not None, \
                "Specify input bit width; activation quantization is required to guarantee accumulator bounds."
            if not (args.gptq or args.gpfq):
                warn("Max accumulator bit width is specified, but no GPxQ is enabled.")
            if args.gpxq_max_accumulator_tile_size is not None:
                if args.weight_quant_granularity == 'per_group':
                    assert args.gpxq_max_accumulator_tile_size == args.weight_group_size, \
                        "Group size must be equal to tile size with per_group quantization."
                if args.input_quant_granularity == 'per_group':
                    assert args.gpxq_max_accumulator_tile_size == args.input_group_size, \
                        "Group size must be equal to tile size with per_group quantization."

        if args.export_target == 'sharded_torchmlir_group_weight':
            assert args.weight_quant_granularity == 'per_group', "Sharded torch group export requires per group weight quant."
            assert args.input_bit_width is None, "Sharded torch group weight export doesn't support input quant."
            assert not args.quantize_weight_zero_point, "Quantized weight zero point not supported."
        if args.export_target == 'sharded_packed_torchmlir_group_weight':
            assert args.weight_quant_granularity == 'per_group', "Sharded torch group export requires per group weight quant."
            assert args.input_bit_width is None, "Sharded packed torch group weight export doesn't support input quant."
            assert not args.quantize_weight_zero_point, "Quantized weight zero point not supported."
        if args.export_target == 'onnx_qcdq':
            if args.weight_quant_granularity == 'per_group':
                assert args.input_bit_width is None, "ONNX QCDQ per_group quantization requires no input quantization"
            if args.weight_quant_type == 'asym':
                assert args.quantize_weight_zero_point, "Quantized weight zero point required."
            if args.input_bit_width is not None and args.input_quant_type == 'asym':
                assert args.quantize_input_zero_point, "Quantized input zero point required."
        if args.input_bit_width and args.input_scale_type == 'static':
            assert args.act_calibration, "Static input quantization is being applied without activation calibration. Set --act-calibration."
        if args.no_quantize_linear:
            assert args.input_bit_width is not None and args.quant_sdpa is not None, \
                "--no-quantize-linear keeps only the attention quantizers: set --quant-sdpa and input_bit_width"
    assert not (args.rotation_sdpa_value_det and args.rotation_sdpa_value_rand), \
        "--rotation-sdpa-value-det and --rotation-sdpa-value-rand are exclusive"
    validate_dc_extract(args)

    # Validate Scale Optimization (--sopt-*) flags
    ScaleOptimizerArgumentParser.validate(args, extra_args)


def validate_dc_extract(args: Namespace) -> None:
    if not args.dc_extract_qk:
        dc_defaults = {
            'dc_correction': 'none',
            'dc_sinkbal': False,
            'dc_had_rand': False,
            'dc_had_layers': [],
            'dc_tcast_scale': None,
            'dc_u_mode': 'live',
            'sqnr_decomp': 0}
        changed = [k for k, v in dc_defaults.items() if getattr(args, k) != v]
        assert not changed, f"{changed} require --dc-extract-qk"
        return
    assert args.quant_sdpa is not None, "--dc-extract-qk requires --quant-sdpa"
    if args.dc_correction != 'none':
        assert 0.0 < args.dc_gamma < 1.0, "--dc-correction needs --dc-gamma in (0, 1)"
    if args.dc_correction == 'scale':
        assert args.dc_sub_dim > 0, "--dc-correction scale needs an exact plane (--dc-sub-dim > 0)"
    assert not args.dc_had_rand_shared or args.dc_had_rand, "--dc-had-rand-shared needs --dc-had-rand"
    if args.mf_probe_gamma is not None:
        assert args.sqnr_decomp > 0, "--mf-probe-gamma runs inside --sqnr-decomp"


def attn_quant_format_validator(value):
    if value is None:
        return True
    else:
        return quant_format_validator(value)
