# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

"""Configuration bridge for vLLM-Ascend MXFP rollout fake quantization."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from verl.utils.qat.mxfp_linear import MXFPQATConfig
from verl.utils.qat.mxfp_quant import MXFPMode, normalize_mxfp_mode, normalize_rounding_mode

_VLLM_MODE = {
    MXFPMode.W8A8: "w8a8_mxfp8",
    MXFPMode.W4A8: "w4a8_mxfp4",
    MXFPMode.W4A4: "w4a4_mxfp4",
}


def _as_list(value: Any, default: list | None = None) -> list:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable):
        return [_plain_config_value(item) for item in value]
    return [value]


def _plain_config_value(value: Any) -> Any:
    if isinstance(value, Mapping) or hasattr(value, "items"):
        return {key: _plain_config_value(item) for key, item in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, str):
        return [_plain_config_value(item) for item in value]
    return value


def _as_dict(value: Any) -> Mapping:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {key: _plain_config_value(item) for key, item in value.items()}
    if hasattr(value, "items"):
        return {key: _plain_config_value(item) for key, item in value.items()}
    raise TypeError(f"Expected a mapping, got {type(value).__name__}")


def mxfp_qat_config_to_vllm_ascend_env(config: Mapping[str, Any]) -> dict[str, str]:
    """Translate verl's MXFP QAT settings to vLLM-Ascend's environment contract."""
    config = _as_dict(config)
    mode = normalize_mxfp_mode(config.get("mode", MXFPMode.W8A8.value))
    group_size = int(config.get("group_size", 32))
    quant_backend = str(config.get("quant_backend", "npu")).lower()
    rounding_mode = normalize_rounding_mode(config.get("rounding_mode", "rint"))
    mxfp4_scale_alg = str(config.get("mxfp4_scale_alg", "standard"))
    if quant_backend == "npu" and rounding_mode in {"random", "hash"}:
        raise ValueError("MXFP8 stochastic rounding requires quant_backend='torch'")
    fake_quant_targets = _as_list(config.get("fake_quant_targets"), ["Fprop"])
    rotation_targets = _as_list(config.get("rotation_targets"), ["Fprop"])
    rotation_enable = bool(config.get("rotation_enable", False))
    rotation_block_size = int(config.get("rotation_block_size", group_size))
    rotation_seed = int(config.get("rotation_seed", 0))

    # Reuse the training config validation so rollout rejects settings the actor cannot run.
    qat_config = MXFPQATConfig(
        mode=mode.value,
        group_size=group_size,
        quant_backend=quant_backend,
        rounding_mode=rounding_mode,
        mxfp4_scale_alg=mxfp4_scale_alg,
        fake_quant_targets=tuple(fake_quant_targets),
        activation_residual_fprop=bool(config.get("activation_residual_fprop", False)),
        activation_residual_bprop=bool(config.get("activation_residual_bprop", False)),
        activation_residual_fprop_alpha=config.get("activation_residual_fprop_alpha", 1.0),
        activation_residual_bprop_alpha=config.get("activation_residual_bprop_alpha", 1.0),
        rotation_enable=rotation_enable,
        rotation_targets=tuple(rotation_targets),
        rotation_block_size=rotation_block_size,
        rotation_seed=rotation_seed,
    )

    ignore_patterns = _as_list(config.get("ignore_patterns"))
    experts_config = _as_dict(config.get("experts"))
    if not bool(experts_config.get("enable", False)):
        ignore_patterns.extend(
            [
                r"re:.*mlp\.experts(?:\.|$)",
                r"re:.*experts\.[^.]+\.(gate_proj|up_proj|down_proj)(?:\.weight)?$",
                r"re:.*shared_expert(?:\.|$)",
            ]
        )
    fallback_layers = config.get("fallback_layers")
    if fallback_layers is None:
        fallback_layers_env = ""
    elif isinstance(fallback_layers, str):
        fallback_layers_env = fallback_layers
    else:
        fallback_layers_env = json.dumps(_as_list(fallback_layers))

    return {
        "VLLM_ASCEND_QAT_FAKE_QUANT": "1",
        "VLLM_ASCEND_QAT_FAKE_QUANT_MODE": _VLLM_MODE[mode],
        "VLLM_ASCEND_QAT_MXFP8_QUANT_BACKEND": qat_config.quant_backend,
        "VLLM_ASCEND_QAT_MXFP8_ROUNDING_MODE": qat_config.rounding_mode,
        "VLLM_ASCEND_QAT_MXFP4_SCALE_ALG": qat_config.mxfp4_scale_alg,
        "VLLM_ASCEND_QAT_MXFP8_GROUP_SIZE": str(qat_config.group_size),
        "VLLM_ASCEND_QAT_MXFP8_ACTIVATION_RESIDUAL_FPROP": str(qat_config.activation_residual_fprop).lower(),
        "VLLM_ASCEND_QAT_MXFP8_ACTIVATION_RESIDUAL_FPROP_ALPHA": str(qat_config.activation_residual_fprop_alpha),
        "VLLM_ASCEND_QAT_MXFP8_FAKE_QUANT_TARGETS": json.dumps(
            [target for target in ("fprop", "dgrad", "wgrad") if target in qat_config.fake_quant_targets]
        ),
        "VLLM_ASCEND_QAT_MXFP8_ROTATION_ENABLE": str(qat_config.rotation.enable).lower(),
        "VLLM_ASCEND_QAT_MXFP8_ROTATION_KIND": "block_hadamard_sign",
        "VLLM_ASCEND_QAT_MXFP8_ROTATION_BLOCK_SIZE": str(qat_config.rotation.block_size),
        "VLLM_ASCEND_QAT_MXFP8_ROTATION_SEED": str(qat_config.rotation.seed),
        "VLLM_ASCEND_QAT_MXFP8_ROTATION_TARGETS": json.dumps(
            [target for target in ("fprop", "dgrad", "wgrad") if target in qat_config.rotation.targets]
        ),
        "VLLM_ASCEND_QAT_MXFP8_IGNORE_PATTERNS": json.dumps(ignore_patterns),
        "VLLM_ASCEND_QAT_MXFP8_FALLBACK_LAYERS": fallback_layers_env,
    }
