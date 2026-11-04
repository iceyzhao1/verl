# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

"""Describe and export the effective training forward weights for Ascend.

Collect metadata after training precision has been applied. Collection also
works after FSDP2 wrapping: DTensor shape/dtype describe global parameters and
no values are gathered. This module neither changes training precision nor
performs a distributed collective. The caller supplies stable full parameters.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from verl.utils.qat.mxfp_quant import MXFPMode, normalize_mxfp_mode, quantize_mxfp8
from verl.utils.qat.mxfp_rotation import RotationConfig, normalize_targets, rotate_blocks


@dataclass(frozen=True)
class AscendTensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    module_name: str
    kind: str = "other"
    qat_config: Any = None
    enabled: bool = True
    is_buffer: bool = False


@dataclass(frozen=True)
class _ForwardRecipe:
    backend: str
    rotation: RotationConfig


@dataclass(frozen=True)
class _ExportSpec:
    source: str
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    recipe: _ForwardRecipe | None
    expert: int | None = None
    half: int | None = None


@dataclass(frozen=True)
class AscendWeightSyncPlan:
    quant_description: dict[str, Any]
    manifest: list[dict[str, Any]]
    source_specs: dict[str, AscendTensorSpec]
    exports: dict[str, tuple[_ExportSpec, ...]]


def _get(config: Any, name: str, default: Any = None) -> Any:
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def collect_ascend_module_specs(model: nn.Module) -> dict[str, AscendTensorSpec]:
    """Collect all parameters (including tied aliases) and persistent buffers.

    Effective QAT configs come from the training modules, not a second module
    selection policy. Packed experts inherit their containing MoE config.
    Names use the same checkpoint conversion as FSDP's export entry point.
    FSDP2 can retain its sharded DTensors here; only global metadata is read.
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    tensors = dict(model.named_parameters(remove_duplicate=False))
    buffers = set()
    for module_name, module in modules.items():
        for name, tensor in module._buffers.items():
            if tensor is not None and name not in module._non_persistent_buffers_set:
                full_name = f"{module_name}.{name}" if module_name else name
                tensors[full_name] = tensor
                buffers.add(full_name)

    specs = {}
    for name, tensor in tensors.items():
        module_name, _, local_name = name.rpartition(".")
        owner = modules[module_name]
        kind, config = "other", None
        enabled = bool(getattr(owner, "fake_quant_enabled", True))
        if local_name == "weight" and isinstance(owner, nn.Linear):
            kind, config = "linear", getattr(owner, "qat_config", None)
        elif tensor.ndim == 3 and name.endswith((".experts.gate_up_proj", ".experts.down_proj")):
            kind = "packed_gate_up" if local_name == "gate_up_proj" else "packed_down"
            ancestor = module_name
            while True:
                block = modules[ancestor]
                config = getattr(block, "_mxfp_expert_qat_runtime_config", None)
                if config is None:
                    config = getattr(block, "_mxfp_expert_qat_config", None)
                if config is not None:
                    break
                if not ancestor:
                    break
                ancestor = ancestor.rpartition(".")[0]
        specs[name] = AscendTensorSpec(
            name, tuple(tensor.shape), tensor.dtype, module_name, kind, config, enabled, name in buffers
        )

    if getattr(model, "_checkpoint_conversion_mapping", None):
        from verl.utils.model import convert_weight_keys

        converted = convert_weight_keys(specs, model)
        if len(converted) != len(specs):
            raise ValueError("Checkpoint name conversion produced duplicate weight names")
        specs = {name: replace(spec, name=name) for name, spec in converted.items()}
    return specs


def _ignored(module_name: str, patterns: Iterable[str]) -> bool:
    return any(re.match(p[3:], module_name) if p.startswith("re:") else p in module_name for p in patterns)


def _fallback_ranges(value: Any) -> tuple[tuple[int, int], ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, int):
        value = [value]
    ranges = []
    for item in value:
        if isinstance(item, int):
            start = end = item
        elif isinstance(item, str) and re.fullmatch(r"\d+(?:-\d+)?", item):
            parts = item.split("-")
            start, end = int(parts[0]), int(parts[-1])
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            start, end = map(int, item)
        else:
            raise ValueError(f"Invalid fallback_layers range: {item!r}")
        if start < 0 or end < start:
            raise ValueError(f"Invalid fallback_layers range: {item!r}")
        ranges.append((start, end))
    return tuple(ranges)


def _recipe(spec: AscendTensorSpec, config: Any, fallback: tuple[tuple[int, int], ...]) -> _ForwardRecipe | None:
    if spec.kind == "other":
        return None
    effective = spec.qat_config
    active = (
        effective is not None
        and spec.enabled
        and "fprop" in normalize_targets(_get(effective, "fake_quant_targets", ("fprop",)))
    )
    ignored = _ignored(spec.module_name, _get(config, "ignore_patterns", ()))
    layer = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", spec.module_name)
    falls_back = layer is not None and any(start <= int(layer[1]) <= end for start, end in fallback)
    if active and (ignored or falls_back):
        field = "fallback_layers" if falls_back else "ignore_patterns"
        raise ValueError(
            f"{spec.name}: {field} conflicts with the effective training module; fix training selection first"
        )
    if effective is None:
        expected = bool(_get(config, "enable", False)) and not ignored and not falls_back
        if spec.kind.startswith("packed_"):
            expected = expected and bool(_get(_get(config, "experts", {}), "enable", False))
        expected = expected and "fprop" in normalize_targets(_get(config, "fake_quant_targets", ("fprop",)))
        if expected:
            raise ValueError(f"{spec.name}: training module does not expose its effective QAT configuration")
        return None
    if not active:
        if (
            spec.enabled
            and _get(effective, "rotation_enable", False)
            and "fprop" in normalize_targets(_get(effective, "rotation_targets", ("fprop",)))
        ):
            raise NotImplementedError(f"{spec.name}: rotation without forward quantization is not supported")
        return None
    if not bool(_get(config, "enable", False)):
        raise ValueError(f"{spec.name}: effective training QAT conflicts with qat.enable=False")
    mode = normalize_mxfp_mode(_get(effective, "mode"))
    if mode != MXFPMode.W8A8:
        raise NotImplementedError(f"{spec.name}: {mode.value} needs its own packed exporter and reload support")
    if _get(effective, "group_size", 32) != 32:
        raise ValueError(f"{spec.name}: MXFP8 requires group_size=32")
    if _get(effective, "activation_residual_fprop", False) or _get(effective, "activation_residual_bprop", False):
        raise NotImplementedError(f"{spec.name}: real activation residual execution is not supported by this path")
    if _get(effective, "probe_quant_error", False) and _get(effective, "probe_quant_error_use_high_precision", True):
        raise ValueError(
            f"{spec.name}: the training error probe is executing high precision instead of the requested recipe"
        )
    if str(_get(effective, "rounding_mode", "rint")).lower() != "rint":
        raise NotImplementedError(f"{spec.name}: real activation rounding compatibility is only validated for rint")
    backend = _get(effective, "quant_backend", "npu")
    if backend not in {"torch", "npu"}:
        raise ValueError(f"{spec.name}: unsupported quant_backend {backend!r}")
    rotate = bool(_get(effective, "rotation_enable", False)) and "fprop" in normalize_targets(
        _get(effective, "rotation_targets", ("fprop",))
    )
    block_size = _get(effective, "rotation_block_size", 32)
    if rotate and block_size != 32:
        raise ValueError(f"{spec.name}: MXFP8 rotation requires rotation_block_size=32")
    rotation = (
        RotationConfig(True, ("fprop",), block_size, _get(effective, "rotation_seed", 0))
        if rotate
        else RotationConfig()
    )
    return _ForwardRecipe(backend, rotation)


def _expand(spec: AscendTensorSpec, recipe: _ForwardRecipe | None) -> tuple[_ExportSpec, ...]:
    if not spec.kind.startswith("packed_"):
        return (_ExportSpec(spec.name, spec.name, spec.shape, spec.dtype, recipe),)
    experts, rows, columns = spec.shape
    base = spec.name.rsplit(".", 1)[0]
    if spec.kind == "packed_gate_up" and rows % 2:
        raise ValueError(f"{spec.name}: gate_up_proj output dimension must be even")
    return tuple(
        _ExportSpec(
            spec.name,
            f"{base}.{expert}.{projection}.weight",
            (rows // 2 if half is not None else rows, columns),
            spec.dtype,
            recipe,
            expert,
            half,
        )
        for expert in range(experts)
        for projection, half in (
            (("gate_proj", 0), ("up_proj", 1)) if spec.kind == "packed_gate_up" else (("down_proj", None),)
        )
    )


def build_ascend_weight_sync_plan(
    qat_config: Any, module_specs: Mapping[str, AscendTensorSpec], *, param_dtype: torch.dtype | None = None
) -> AscendWeightSyncPlan:
    """Resolve the actual forward configuration, failing before engine startup.

    ``param_dtype`` is the engine's effective forward parameter dtype, before
    rotation and quantization. Master parameter dtypes remain the source
    contract; FSDP's parameter precision policy does not cast buffers.
    """
    if param_dtype is not None and param_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(f"Unsupported forward parameter dtype: {param_dtype}")
    fallback = _fallback_ranges(_get(qat_config, "fallback_layers"))
    exports, entries, recipes = {}, [], []
    for source, spec in module_specs.items():
        if source != spec.name:
            raise ValueError(f"Tensor spec key {source!r} does not match {spec.name!r}")
        recipe = _recipe(spec, qat_config, fallback)
        forward_spec = spec
        if param_dtype is not None and not spec.is_buffer and spec.dtype.is_floating_point:
            forward_spec = replace(spec, dtype=param_dtype)
        exports[source] = _expand(forward_spec, recipe)
        entries.extend(exports[source])
        if recipe is not None:
            recipes.append(recipe)
    rotation = recipes[0].rotation if recipes else RotationConfig()
    if any(recipe.rotation != rotation for recipe in recipes):
        raise NotImplementedError(
            "Different per-module rotations cannot be represented by the current Ascend description"
        )
    description = {
        "quant_method": "ascend",
        "version": "1.0.0",
        "group_size": 32,
        "model_quant_type": "W8A8_MXFP8" if recipes else "FLOAT",
        "rotation_enable": rotation.enable,
        "rotation_kind": "block_hadamard_sign",
        "rotation_block_size": rotation.block_size,
        "rotation_seed": rotation.seed,
        "rotation_targets": ["fprop"] if rotation.enable else [],
    }
    manifest, groups = [], {}
    for entry in entries:
        if entry.name in description:
            raise ValueError(f"Duplicate canonical tensor name: {entry.name}")
        if entry.recipe is not None:
            if len(entry.shape) != 2 or entry.shape[-1] % 32:
                raise ValueError(f"{entry.name}: MXFP8 requires a 2D weight with input dimension divisible by 32")
            if entry.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
                raise ValueError(f"{entry.name}: unsupported quantization source dtype {entry.dtype}")
            if entry.name + "_scale" in description:
                raise ValueError(f"Duplicate canonical tensor name: {entry.name}_scale")
            description[entry.name] = description[entry.name + "_scale"] = "W8A8_MXFP8"
            manifest.extend(
                [
                    {"name": entry.name, "shape": list(entry.shape), "dtype": "float8_e4m3fn"},
                    {"name": entry.name + "_scale", "shape": [entry.shape[0], entry.shape[1] // 32], "dtype": "uint8"},
                ]
            )
        else:
            description[entry.name] = "FLOAT"
            manifest.append(
                {"name": entry.name, "shape": list(entry.shape), "dtype": str(entry.dtype).removeprefix("torch.")}
            )
        match = re.match(r"(.*)\.(q_proj|k_proj|v_proj|gate_proj|up_proj)\.weight$", entry.name)
        if match:
            family = "qkv" if match[2] in {"q_proj", "k_proj", "v_proj"} else "gate_up"
            groups.setdefault((match[1], family), set()).add(entry.recipe)
        if ".experts." in entry.name and entry.name.endswith(".weight"):
            groups.setdefault((entry.name.split(".experts.")[0], "experts"), set()).add(entry.recipe)
    if any(len(values) > 1 for values in groups.values()):
        raise ValueError("Fused QKV, gate/up, or expert members have incompatible effective forward configurations")
    return AscendWeightSyncPlan(description, manifest, dict(module_specs), exports)


@torch.no_grad()
def _quantize_weight(weight: torch.Tensor, entry: _ExportSpec) -> tuple[torch.Tensor, torch.Tensor]:
    recipe = entry.recipe
    if recipe.rotation.enable:
        weight = rotate_blocks(weight, recipe.rotation)
    payload, scale = quantize_mxfp8(weight.contiguous(), quant_backend=recipe.backend, rounding_mode="rint")
    expected_scale = (entry.shape[0], entry.shape[1] // 32)
    if payload.dtype != torch.float8_e4m3fn or tuple(payload.shape) != entry.shape:
        raise ValueError(
            f"{entry.name}: quantizer did not produce real FP8 payload; fake payloads cannot be transmitted"
        )
    if scale.dtype != torch.uint8 or tuple(scale.shape) != expected_scale:
        raise ValueError(f"{entry.name}: quantizer produced an incompatible MXFP8 scale")
    return payload.contiguous(), scale.contiguous()


def iter_ascend_weight_sync_params(
    params: Iterable[tuple[str, torch.Tensor]], plan: AscendWeightSyncPlan
) -> Iterable[tuple[str, torch.Tensor]]:
    """Export a stable, complete stream of full tensors without altering masters."""
    seen = set()
    for name, tensor in params:
        if name not in plan.source_specs or name in seen:
            raise ValueError(f"Unexpected or duplicate source tensor: {name}")
        seen.add(name)
        spec = plan.source_specs[name]
        if tuple(tensor.shape) != spec.shape or tensor.dtype != spec.dtype:
            raise ValueError(f"{name}: source shape/dtype differs from the collected training snapshot")
        for entry in plan.exports[name]:
            weight = tensor if entry.expert is None else tensor[entry.expert]
            if entry.half is not None:
                start = entry.half * entry.shape[0]
                weight = weight[start : start + entry.shape[0]]
            weight = weight.to(entry.dtype)
            if entry.recipe is None:
                yield entry.name, weight.detach().contiguous()
            else:
                payload, scale = _quantize_weight(weight, entry)
                yield entry.name, payload
                yield entry.name + "_scale", scale
    missing = set(plan.source_specs) - seen
    if missing:
        raise ValueError(f"Missing source tensors: {sorted(missing)}")
