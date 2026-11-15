# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

"""Dense MXFP QAT linear with independent fprop, dgrad and wgrad transforms."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.utils.qat.mxfp_debug import warn_once
from verl.utils.qat.mxfp_probe import quantize_with_probe
from verl.utils.qat.mxfp_quant import MXFPMode, normalize_mxfp4_scale_alg, normalize_mxfp_mode
from verl.utils.qat.mxfp_rotation import RotationConfig, rotate_blocks

_TARGETS = ("fprop", "wgrad", "dgrad")
_MXFP_MODE_KEYS = {
    "mxfp_w8a8",
    "w8a8_mxfp8",
    "mxfp_w4a8",
    "w4a8_mxfp4",
    "mxfp_w4a4",
    "w4a4_mxfp4",
}


def _normalize_targets(targets: Any) -> tuple[str, ...]:
    if targets is None:
        return ("fprop",)
    if isinstance(targets, str):
        targets = [part.strip() for part in targets.split(",") if part.strip()]
    if not isinstance(targets, (list, tuple, set)):
        raise TypeError("MXFP fake_quant_targets must be a list or comma-separated string")
    normalized = {str(target).lower() for target in targets}
    aliases = {"forward": "fprop", "fprop": "fprop", "wgrad": "wgrad", "dgrad": "dgrad"}
    try:
        normalized = {aliases[target] for target in normalized}
    except KeyError as exc:
        raise ValueError(f"Unsupported MXFP fake quant target {exc.args[0]!r}; expected {_TARGETS}") from exc
    return tuple(target for target in _TARGETS if target in normalized)


def is_mxfp_mode(mode: str) -> bool:
    return str(mode).lower() in _MXFP_MODE_KEYS


def validate_activation_residual_alpha(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite real number in [0, 1], got {value!r}")


@dataclass(frozen=True)
class MXFPQATConfig:
    mode: str = MXFPMode.W8A8.value
    group_size: int = 32
    quant_backend: str = "npu"
    rounding_mode: str = "rint"
    mxfp4_scale_alg: str = "standard"
    fake_quant_targets: tuple[str, ...] = ("fprop",)
    activation_residual_fprop: bool = False
    activation_residual_bprop: bool = False
    activation_residual_fprop_alpha: float = 1.0
    activation_residual_bprop_alpha: float = 1.0
    backward_reduction_axis_fake_quant: bool = True
    rotation_enable: bool = False
    rotation_targets: tuple[str, ...] = ("fprop",)
    rotation_block_size: int = 32
    rotation_seed: int = 0
    layer_name: str | None = None
    probe_quant_error: bool = False
    probe_quant_error_output_path: str | None = None
    probe_quant_error_use_high_precision: bool = True

    def __post_init__(self):
        validate_activation_residual_alpha(self.activation_residual_fprop_alpha, "activation_residual_fprop_alpha")
        validate_activation_residual_alpha(self.activation_residual_bprop_alpha, "activation_residual_bprop_alpha")
        mode = normalize_mxfp_mode(self.mode)
        object.__setattr__(self, "mode", mode.value)
        object.__setattr__(self, "quant_backend", str(self.quant_backend).lower())
        object.__setattr__(self, "mxfp4_scale_alg", normalize_mxfp4_scale_alg(self.mxfp4_scale_alg))
        if self.group_size != 32:
            raise ValueError(f"MXFP QAT requires group_size=32, got {self.group_size}")
        if mode in (MXFPMode.W4A8, MXFPMode.W4A4) and self.quant_backend != "torch":
            raise ValueError("MXFP4 QAT requires quant_backend='torch'")
        if self.quant_backend not in ("torch", "npu"):
            raise ValueError(f"Unsupported MXFP quant backend {self.quant_backend!r}")
        object.__setattr__(self, "fake_quant_targets", _normalize_targets(self.fake_quant_targets))
        object.__setattr__(self, "rotation_targets", _normalize_targets(self.rotation_targets))
        if self.rotation_enable and self.rotation_block_size != self.group_size:
            raise ValueError("rotation_block_size must equal group_size")

    @property
    def activation_residual_fprop_enabled(self) -> bool:
        return self.activation_residual_fprop and self.activation_residual_fprop_alpha > 0

    @property
    def activation_residual_bprop_enabled(self) -> bool:
        return self.activation_residual_bprop and self.activation_residual_bprop_alpha > 0

    @property
    def weight_format(self) -> str:
        return "mxfp4" if self.mode in (MXFPMode.W4A8.value, MXFPMode.W4A4.value) else "mxfp8"

    @property
    def activation_format(self) -> str:
        return "mxfp4" if self.mode == MXFPMode.W4A4.value else "mxfp8"

    @property
    def rotation(self) -> RotationConfig:
        return RotationConfig(
            enable=self.rotation_enable,
            targets=self.rotation_targets,
            block_size=self.rotation_block_size,
            seed=self.rotation_seed,
        )


def _target_tensor(
    tensor: torch.Tensor,
    config: MXFPQATConfig,
    target: str,
    fmt: str,
    *,
    pad_rotation: bool = False,
    probe_operand: str = "activation",
    probe_module_name: str | None = None,
) -> torch.Tensor:
    rotated = tensor
    if config.rotation.enable and target in config.rotation.targets:
        rotated = rotate_blocks(rotated, config.rotation, pad_to_block=pad_rotation)
    if target in config.fake_quant_targets:
        return quantize_with_probe(
            rotated,
            fmt,
            config,
            target=target,
            operand=probe_operand,
            module_name=probe_module_name,
            valid_shape=tuple(tensor.shape) if pad_rotation and tuple(rotated.shape) != tuple(tensor.shape) else None,
        )
    return rotated


def _prepare_dgrad_weight(weight: torch.Tensor, config: MXFPQATConfig) -> torch.Tensor:
    with torch.no_grad():
        return _target_tensor(
            weight.transpose(0, 1),
            config,
            "dgrad",
            config.weight_format,
            probe_operand="weight",
        ).transpose(0, 1)


def _prepare_wgrad_input(input: torch.Tensor, config: MXFPQATConfig) -> torch.Tensor:
    input_2d = input.reshape(-1, input.shape[-1])
    with torch.no_grad():
        return _target_tensor(input_2d.transpose(0, 1), config, "wgrad", config.activation_format, pad_rotation=True)


def _prepare_wgrad_reference_input(
    original_input: torch.Tensor, config: MXFPQATConfig, wgrad_enabled: bool
) -> torch.Tensor:
    """Map original A to the main wgrad coordinates, without either A QDQ."""
    values = original_input
    if config.rotation.enable and "fprop" in config.rotation.targets:
        values = rotate_blocks(values, config.rotation)
    values_t = values.reshape(-1, values.shape[-1]).transpose(0, 1)
    if wgrad_enabled and config.rotation.enable and "wgrad" in config.rotation.targets:
        values_t = rotate_blocks(values_t, config.rotation, pad_to_block=True)
    return values_t


def _add_fprop_activation_residual(
    output: torch.Tensor,
    input: torch.Tensor,
    quantized_input: torch.Tensor,
    quantized_weight: torch.Tensor,
    config: MXFPQATConfig,
) -> torch.Tensor:
    if not config.activation_residual_fprop_enabled:
        return output
    # This forward correction must not add an implicit weight-gradient path:
    # dgrad/wgrad are controlled independently by the existing backward policy.
    with torch.no_grad():
        values = input
        if config.rotation.enable and "fprop" in config.rotation.targets:
            values = rotate_blocks(values, config.rotation)
        residual = values - quantized_input
        correction = F.linear(residual, quantized_weight)
        if config.activation_residual_fprop_alpha != 1.0:
            correction = correction * config.activation_residual_fprop_alpha
    return output + correction


class _MXFPLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input, weight, bias, dgrad_weight, wgrad_input_t, config, dgrad_enabled, wgrad_enabled, original_input
    ):
        ctx.has_saved_dgrad_weight = dgrad_weight is not None
        ctx.has_saved_wgrad_input = wgrad_input_t is not None
        ctx.dgrad_enabled = dgrad_enabled
        ctx.wgrad_enabled = wgrad_enabled
        ctx.config = config
        ctx.bias_is_none = bias is None
        ctx.save_for_backward(
            input,
            weight,
            dgrad_weight if dgrad_weight is not None else weight,
            wgrad_input_t if wgrad_input_t is not None else input,
            original_input if config.activation_residual_bprop_enabled and ctx.needs_input_grad[1] else None,
        )
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, saved_dgrad_weight, saved_wgrad_input_t, original_input = ctx.saved_tensors
        config = ctx.config
        input_2d = input.reshape(-1, input.shape[-1])
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        need_input_grad, need_weight_grad = ctx.needs_input_grad[:2]
        # Return gradients in the transformed operand coordinates; outer autograd handles QDQ and rotation.

        grad_input = None
        if need_input_grad:
            if ctx.dgrad_enabled:
                warn_once(
                    (
                        "dense-dgrad",
                        config.mode,
                        config.quant_backend,
                        config.fake_quant_targets,
                        config.rotation.targets,
                    ),
                    "dense dgrad executed: fake_quant=%s rotation=%s reduction_axis=output cached=%s",
                    "dgrad" in config.fake_quant_targets,
                    config.rotation.enable and "dgrad" in config.rotation.targets,
                    ctx.has_saved_dgrad_weight,
                )
            if ctx.dgrad_enabled:
                grad_output_t = _target_tensor(
                    grad_output_2d,
                    config,
                    "dgrad",
                    config.activation_format,
                    probe_operand="grad_output",
                )
                weight_dgrad = (
                    saved_dgrad_weight if ctx.has_saved_dgrad_weight else _prepare_dgrad_weight(weight, config)
                )
                # _prepare_dgrad_weight restores F.linear's [out_features, in_features] layout.
                weight_t = weight_dgrad
            else:
                grad_output_t = grad_output_2d
                weight_t = weight
            grad_input_2d = grad_output_t.matmul(weight_t)
            grad_input = grad_input_2d.reshape_as(input)

        grad_weight = None
        if need_weight_grad:
            if ctx.wgrad_enabled:
                warn_once(
                    (
                        "dense-wgrad",
                        config.mode,
                        config.quant_backend,
                        config.fake_quant_targets,
                        config.rotation.targets,
                    ),
                    "dense wgrad executed: fake_quant=%s rotation=%s reduction_axis=token cached=%s",
                    "wgrad" in config.fake_quant_targets,
                    config.rotation.enable and "wgrad" in config.rotation.targets,
                    ctx.has_saved_wgrad_input,
                )
            if ctx.wgrad_enabled:
                input_t = saved_wgrad_input_t if ctx.has_saved_wgrad_input else _prepare_wgrad_input(input, config)
                grad_output_t = _target_tensor(
                    grad_output_2d.transpose(0, 1),
                    config,
                    "wgrad",
                    config.activation_format,
                    pad_rotation=True,
                    probe_operand="grad_output",
                )
            else:
                input_t = input_2d.transpose(0, 1)
                grad_output_t = grad_output_2d.transpose(0, 1)
            grad_weight = grad_output_t.matmul(input_t.transpose(0, 1))
            if config.activation_residual_bprop_enabled:
                # Dense wgrad starts from fprop Aq, then may quantize A again
                # along tokens. Reference original A so both errors are covered.
                reference_input_t = _prepare_wgrad_reference_input(original_input, config, ctx.wgrad_enabled)
                residual_t = reference_input_t - input_t
                # Reuse the exact main-branch G. Neither G nor W is compensated.
                correction = grad_output_t.matmul(residual_t.transpose(0, 1))
                if config.activation_residual_bprop_alpha != 1.0:
                    correction = correction * config.activation_residual_bprop_alpha
                grad_weight = grad_weight + correction

        grad_bias = grad_output_2d.sum(dim=0) if not ctx.bias_is_none and ctx.needs_input_grad[2] else None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class MXFPQATLinear(nn.Linear):
    """An nn.Linear replacement; parameters are reused to preserve FSDP identity."""

    _is_verl_qat_linear = True
    supports_qat_fusion = False

    def __init__(self, in_features: int, out_features: int, bias: bool, config: MXFPQATConfig, **factory_kwargs):
        super().__init__(in_features, out_features, bias=bias, **factory_kwargs)
        self.qat_config = config
        self.fake_quant_enabled = True

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: MXFPQATConfig) -> MXFPQATLinear:
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.in_features = linear.in_features
        result.out_features = linear.out_features
        result._parameters["weight"] = linear._parameters["weight"]
        result._parameters["bias"] = linear._parameters["bias"]
        result.qat_config = config
        result.fake_quant_enabled = True
        result.training = linear.training
        return result

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        config = self.qat_config
        warn_once(
            (
                "dense-fprop",
                config.mode,
                config.quant_backend,
                config.fake_quant_targets,
                config.rotation.targets,
            ),
            "dense fprop executed: mode=%s fake_quant=%s rotation=%s layer=%s",
            config.mode,
            "fprop" in config.fake_quant_targets,
            config.rotation.enable and "fprop" in config.rotation.targets,
            config.layer_name or "<first-dense>",
        )
        if not self.fake_quant_enabled:
            return F.linear(input, self.weight, self.bias)

        weight = _target_tensor(
            self.weight,
            config,
            "fprop",
            config.weight_format,
            probe_operand="weight",
        )
        values = _target_tensor(
            input,
            config,
            "fprop",
            config.activation_format,
            probe_operand="activation",
        )

        if not input.requires_grad and not self.weight.requires_grad:
            return _add_fprop_activation_residual(F.linear(values, weight, self.bias), input, values, weight, config)

        needs_custom_backward = config.activation_residual_bprop_enabled or (
            config.rotation.enable and "fprop" in config.rotation.targets
        )
        needs_custom_backward = needs_custom_backward or any(
            target in config.fake_quant_targets or (config.rotation.enable and target in config.rotation.targets)
            for target in ("dgrad", "wgrad")
        )
        if not needs_custom_backward:
            return _add_fprop_activation_residual(F.linear(values, weight, self.bias), input, values, weight, config)

        dgrad_weight = None
        wgrad_input_t = None
        dgrad_enabled = "dgrad" in config.fake_quant_targets or (
            config.rotation.enable and "dgrad" in config.rotation.targets
        )
        wgrad_enabled = "wgrad" in config.fake_quant_targets or (
            config.rotation.enable and "wgrad" in config.rotation.targets
        )
        if config.backward_reduction_axis_fake_quant:
            if dgrad_enabled and input.requires_grad:
                dgrad_weight = _prepare_dgrad_weight(weight, config)
            if wgrad_enabled and self.weight.requires_grad:
                wgrad_input_t = _prepare_wgrad_input(values, config)
        output = _MXFPLinear.apply(
            values,
            weight,
            self.bias,
            dgrad_weight,
            wgrad_input_t,
            config,
            dgrad_enabled,
            wgrad_enabled,
            input.detach() if config.activation_residual_bprop_enabled and self.weight.requires_grad else None,
        )
        return _add_fprop_activation_residual(output, input, values, weight, config)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
            f"mode={self.qat_config.mode}, targets={self.qat_config.fake_quant_targets}, "
            f"quant_backend={self.qat_config.quant_backend}"
        )
