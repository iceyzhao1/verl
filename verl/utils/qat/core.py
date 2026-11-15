# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QAT (Quantization-Aware Training) utilities for verl FSDP training."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import torch.nn as nn

from verl.base_config import BaseConfig
from verl.utils.qat.mxfp_debug import warn_once

logger = logging.getLogger(__name__)


@dataclass
class QATConfig(BaseConfig):
    """Unified configuration for QAT (Quantization-Aware Training)."""

    enable: bool = False
    mode: str = "w4a16"
    group_size: int = 16
    ignore_patterns: list[str] = field(default_factory=lambda: ["lm_head", "embed_tokens", "re:.*mlp.gate$"])
    activation_observer: str = "static_minmax"
    quantization_config_path: Optional[str] = None
    quant_backend: str = "npu"
    rounding_mode: str = "rint"
    mxfp4_scale_alg: str = "standard"
    fake_quant_targets: list[str] = field(default_factory=lambda: ["Fprop"])
    backward_reduction_axis_fake_quant: bool = False
    activation_residual_fprop: bool = False
    # Compensate only the input A operand of Wgrad, using the same grad_output as the main path.
    activation_residual_bprop: bool = False
    activation_residual_fprop_alpha: float = 1.0
    activation_residual_bprop_alpha: float = 1.0
    rotation_enable: bool = False
    rotation_targets: list[str] = field(default_factory=lambda: ["Fprop"])
    rotation_block_size: int = 32
    rotation_seed: int = 0
    probe_quant_error: bool = False
    probe_quant_error_output_path: Optional[str] = None
    probe_quant_error_use_high_precision: bool = True
    fallback_layers: Any = None
    experts: dict[str, Any] = field(default_factory=dict)


def load_quantization_config(qat_config: QATConfig) -> dict[str, Any]:
    """Load quantization config JSON file from QATConfig."""
    if not qat_config.quantization_config_path:
        raise ValueError("quantization_config_path is required when QAT is enabled")

    logger.info(f"Loading QAT quantization config from: {qat_config.quantization_config_path}")

    with open(qat_config.quantization_config_path) as f:
        quant_config = json.load(f)

    if qat_config.ignore_patterns:
        original_ignore = quant_config.get("ignore", [])
        quant_config["ignore"] = qat_config.ignore_patterns
        if original_ignore != qat_config.ignore_patterns:
            logger.info(f"Overriding JSON 'ignore' field: {original_ignore} -> {qat_config.ignore_patterns}")

    logger.info("Successfully loaded QAT quantization config")
    return quant_config


def _should_quantize(name: str, module: nn.Module, config: QATConfig) -> bool:
    """Check if a module should be quantized."""
    if not isinstance(module, nn.Linear):
        return False

    for pattern in config.ignore_patterns:
        if pattern.startswith("re:"):
            regex = pattern[3:]
            if re.match(regex, name):
                logger.debug(f"Ignoring {name} due to regex pattern: {regex}")
                return False
        else:
            if pattern in name:
                logger.debug(f"Ignoring {name} due to pattern: {pattern}")
                return False

    if module.in_features % config.group_size != 0:
        logger.warning(
            f"Skipping {name}: in_features={module.in_features} not divisible by group_size={config.group_size}"
        )
        return False

    return True


def apply_qat(
    model: nn.Module,
    config: QATConfig | dict[str, Any],
) -> nn.Module:
    """Apply QAT to a model by replacing nn.Linear with QATLinear."""
    if not isinstance(config, QATConfig):
        config = QATConfig(**config)

    if not config.enable:
        logger.info("QAT is disabled, returning original model")
        return model

    from verl.utils.qat.mxfp_quant import normalize_mxfp_mode

    try:
        mxfp_mode = normalize_mxfp_mode(config.mode)
    except ValueError:
        mxfp_mode = None

    if mxfp_mode is not None:
        if config.probe_quant_error:
            if not config.probe_quant_error_output_path:
                raise ValueError("probe_quant_error_output_path is required when probe_quant_error=True")
            from verl.utils.qat.mxfp_probe import configure_quant_error_probe

            configure_quant_error_probe(config.probe_quant_error_output_path)
            warn_once(
                ("probe", config.probe_quant_error_output_path, config.probe_quant_error_use_high_precision),
                "MXFP QDQ error probe enabled: output=%s high_precision_matmul=%s; each key is sampled once per step",
                config.probe_quant_error_output_path,
                config.probe_quant_error_use_high_precision,
            )
        from verl.utils.qat.mxfp_linear import MXFPQATConfig, MXFPQATLinear
        from verl.utils.qat.mxfp_moe import attach_mxfp_expert_qat

        mxfp_config = MXFPQATConfig(
            mode=mxfp_mode.value,
            group_size=config.group_size,
            quant_backend=config.quant_backend,
            rounding_mode=config.rounding_mode,
            mxfp4_scale_alg=config.mxfp4_scale_alg,
            fake_quant_targets=config.fake_quant_targets,
            backward_reduction_axis_fake_quant=config.backward_reduction_axis_fake_quant,
            activation_residual_fprop=config.activation_residual_fprop,
            activation_residual_bprop=config.activation_residual_bprop,
            activation_residual_fprop_alpha=config.activation_residual_fprop_alpha,
            activation_residual_bprop_alpha=config.activation_residual_bprop_alpha,
            rotation_enable=config.rotation_enable,
            rotation_targets=config.rotation_targets,
            rotation_block_size=config.rotation_block_size,
            rotation_seed=config.rotation_seed,
            probe_quant_error=config.probe_quant_error,
            probe_quant_error_output_path=config.probe_quant_error_output_path,
            probe_quant_error_use_high_precision=config.probe_quant_error_use_high_precision,
        )
        experts_config = config.experts
        if isinstance(experts_config, dict):
            experts_enabled = bool(experts_config.get("enable", False))
        else:
            experts_enabled = bool(getattr(experts_config, "enable", False))
        moe_count = attach_mxfp_expert_qat(model, mxfp_config) if experts_enabled else 0
        modules_to_replace = [
            (name, module)
            for name, module in model.named_modules()
            if name
            and isinstance(module, nn.Linear)
            and not isinstance(module, MXFPQATLinear)
            and not _matches_ignore_patterns(name, config.ignore_patterns)
        ]
        for name, module in modules_to_replace:
            layer_config = MXFPQATConfig(**{**mxfp_config.__dict__, "layer_name": name})
            _set_module(model, name, MXFPQATLinear.from_linear(module, layer_config))
        logger.info(
            "Applied %s QAT: dense_linear=%d, packed_moe_blocks=%d, experts_enabled=%s",
            mxfp_mode.value,
            len(modules_to_replace),
            moe_count,
            experts_enabled,
        )
        if experts_enabled and moe_count == 0:
            warn_once(
                ("moe-missing", mxfp_mode.value),
                "MXFP expert QAT is enabled but no packed MoE blocks were found; expert path is inactive",
            )
        warn_once(
            (
                "apply",
                mxfp_mode.value,
                config.quant_backend,
                tuple(config.fake_quant_targets),
                tuple(config.rotation_targets),
                config.rotation_enable,
                config.mxfp4_scale_alg,
                experts_enabled,
                config.activation_residual_fprop,
                config.activation_residual_bprop,
                config.activation_residual_fprop_alpha,
                config.activation_residual_bprop_alpha,
            ),
            "FSDP2 MXFP QAT configured: mode=%s dense_linear=%d moe_blocks=%d experts=%s "
            "fprop=%s dgrad=%s wgrad=%s rotation=%s rotation_targets=%s mxfp4_scale_alg=%s "
            "backward_axis_cache=%s activation_residual_fprop=%s activation_residual_bprop=%s "
            "activation_residual_fprop_alpha=%s activation_residual_bprop_alpha=%s",
            mxfp_mode.value,
            len(modules_to_replace),
            moe_count,
            experts_enabled,
            "fprop" in mxfp_config.fake_quant_targets,
            "dgrad" in mxfp_config.fake_quant_targets,
            "wgrad" in mxfp_config.fake_quant_targets,
            mxfp_config.rotation.enable,
            mxfp_config.rotation.targets,
            mxfp_config.mxfp4_scale_alg,
            mxfp_config.backward_reduction_axis_fake_quant,
            mxfp_config.activation_residual_fprop,
            mxfp_config.activation_residual_bprop,
            mxfp_config.activation_residual_fprop_alpha,
            mxfp_config.activation_residual_bprop_alpha,
        )
        return model

    from verl.utils.qat.linear import QATLinear, QATMode

    mode = QATMode(config.mode.lower())
    logger.info(f"Applying QAT with mode={mode.value}, group_size={config.group_size}")

    modules_to_replace = []
    for name, module in model.named_modules():
        if _should_quantize(name, module, config):
            modules_to_replace.append((name, module))

    logger.info(f"Found {len(modules_to_replace)} Linear layers to convert to QAT")

    converted_count = 0
    for name, module in modules_to_replace:
        if isinstance(module, QATLinear):
            continue

        fake_quant_module = QATLinear.from_linear(
            module,
            mode=mode,
            group_size=config.group_size,
            activation_observer=config.activation_observer,
        )

        _set_module(model, name, fake_quant_module)
        converted_count += 1

    logger.info(f"Successfully applied QAT to {converted_count} layers")

    return model


def _matches_ignore_patterns(name: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern.startswith("re:"):
            if re.match(pattern[3:], name):
                logger.debug("Ignoring %s due to regex pattern %s", name, pattern[3:])
                return True
        elif pattern in name:
            logger.debug("Ignoring %s due to pattern %s", name, pattern)
            return True
    return False


def _set_module(model: nn.Module, name: str, new_module: nn.Module):
    """Set a module in the model by its full name."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


FUSION_PATTERNS = {
    "qkv": ["q_proj", "k_proj", "v_proj"],
    "gate_up": ["gate_proj", "up_proj"],
}


def setup_fusion_siblings(model: nn.Module):
    """Setup fusion siblings for QKV and GateUp layers."""
    import weakref

    from verl.utils.qat.linear import QATLinear

    qat_modules = {name: m for name, m in model.named_modules() if isinstance(m, QATLinear)}

    counts = {}
    for group_name, suffixes in FUSION_PATTERNS.items():
        groups: dict[str, dict[str, nn.Module]] = {}
        for name, module in qat_modules.items():
            for suffix in suffixes:
                if name.endswith(suffix):
                    parent = name.rsplit(".", 1)[0]
                    groups.setdefault(parent, {})[suffix] = module

        count = 0
        for parent, projs in groups.items():
            if len(projs) >= 2:
                modules = list(projs.values())
                for i, m in enumerate(modules):
                    siblings = modules[:i] + modules[i + 1 :]
                    m._fusion_siblings_ref = [weakref.ref(s) for s in siblings]
                count += 1
        counts[group_name] = count

    logger.info(f"[QAT Fuse] Setup fusion siblings: {counts}")
    return counts


def enable_qat_fuse(model: nn.Module):
    """Enable QAT fuse mode: sets up fusion siblings for weight scale fusion."""
    setup_fusion_siblings(model)
    model._qat_fuse_enabled = True
    logger.info("[QAT Fuse] Enabled QAT fuse mode")


def invalidate_all_scales(model: nn.Module):
    """Clear all cached weight scales after optimizer.step()."""
    from verl.utils.qat.linear import QATLinear

    count = 0
    for module in model.modules():
        if isinstance(module, QATLinear):
            module._weight_blockwise_scale = None
            module._weight_global_scale = None
            module._cached_weight_amax = None
            count += 1

    logger.debug(f"[QAT Fuse] Invalidated scales for {count} QATLinear layers")
