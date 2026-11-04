# Copyright 2026 verl contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Load training-exported MXFP8/FLOAT tensors without inference-side quantization."""

from contextlib import contextmanager

import torch

_MXFP8_SCHEMES = {
    "AscendW8A8MXFP8DynamicLinearMethod",
    "AscendW8A8MXFP8DynamicFusedMoEMethod",
}


def _restore_mxfp8_checkpoint_layout(module, scheme):
    original_shapes = getattr(module, "_mxfp8_original_shapes", {})
    scale_shape = original_shapes.get("weight_scale")
    if type(scheme).__name__ == "AscendW8A8MXFP8DynamicLinearMethod" and scale_shape and scale_shape[-1] % 2:
        # The Linear kernel packs scales in pairs, padding an odd last group.
        # Ascend's restore hook reshapes directly to the unpadded shape, which
        # fails in that case. Invert the packing and remove only the added pad.
        module.weight.data = module.weight.data.transpose(0, 1)
        unpacked_scale = module.weight_scale.data.transpose(0, 1).reshape(scale_shape[0], -1)
        module.weight_scale.data = unpacked_scale[:, : scale_shape[1]].contiguous()
        module._mxfp8_transformed = False
    else:
        scheme.restore_weights_for_rl_loading(module)


def bootstrap_config_from_load_config(load_config):
    extra_config = load_config.model_loader_extra_config or {}
    config = extra_config.get("verl_bootstrap")
    if not isinstance(config, dict):
        raise ValueError("Prequantized loading requires model_loader_extra_config['verl_bootstrap']")
    if not isinstance(config.get("manifest"), list) or not config["manifest"]:
        raise ValueError("Prequantized loading requires a nonempty exported tensor manifest")
    return config


@contextmanager
def _parameter_subclasses(model):
    # vLLM checks these classes to select packed/sharded parameter loaders.
    changed = []
    try:
        for parameter in model.parameters():
            subclass = getattr(parameter, "subclass_type", None)
            if subclass is not None and type(parameter) is not subclass:
                changed.append((parameter, type(parameter)))
                parameter.__class__ = subclass
        yield
    finally:
        for parameter, original_type in changed:
            parameter.__class__ = original_type


class PrequantizedWeightLoad:
    """One complete streamed update, with transforms outside bucket callbacks.

    The manifest describes canonical, unsharded HF tensors. The model's existing
    weight loaders remain responsible for fusion and local TP/expert sharding.
    Initial loading leaves common postprocessing to BaseModelLoader. Hot reload
    reapplies MXFP8/FLOAT transforms and retains the original runtime storage,
    including pointers already captured by accelerator graphs.
    """

    def __init__(self, model, manifest, *, initial_load=False):
        self.model = model
        self.initial_load = initial_load
        self._expected = {}
        for item in manifest:
            name = item["name"]
            if not isinstance(name, str) or not name or name in self._expected:
                raise ValueError(f"Invalid or duplicate manifest tensor: {name}")
            shape = tuple(item["shape"])
            if any(not isinstance(size, int) or size < 0 for size in shape):
                raise ValueError(f"Invalid manifest shape for {name}: {shape}")
            self._expected[name] = (shape, str(item["dtype"]).removeprefix("torch."))
        if not self._expected:
            raise ValueError("Exported tensor manifest must not be empty")
        self._seen = set()
        self._finished = False
        self._reload_modules = []
        self._float_modules = []
        for module_name, module in model.named_modules():
            method = getattr(module, "quant_method", None)
            if method is None:
                continue
            scheme = getattr(method, "quant_method", None)
            if scheme is None and "Unquantized" in type(method).__name__:
                if not initial_load and type(method).__name__ in {
                    "AscendUnquantizedLinearMethod",
                    "AscendUnquantizedFusedMoEMethod",
                }:
                    if getattr(module, "_mxfp8_fake_quant_enabled", False):
                        raise ValueError("Prequantized loading requires inference-side MXFP8 fake quant to be disabled")
                    if hasattr(module, "w13_weight_list") or hasattr(module, "w2_weight_list"):
                        raise ValueError("Prequantized loading does not support split FLOAT MoE runtime weight lists")
                    self._float_modules.append((module, method))
                continue
            if type(scheme).__name__ not in _MXFP8_SCHEMES:
                raise ValueError(
                    "Prequantized weight sync supports only ModelSlim W8A8_MXFP8/FLOAT; "
                    f"found {type(scheme if scheme is not None else method).__name__}"
                )
            if initial_load and type(scheme).__name__ == "AscendW8A8MXFP8DynamicFusedMoEMethod":
                for name in ("w13_weight_scale", "w2_weight_scale"):
                    scale = getattr(module, name)
                    if scale.shape[-1] % 2:
                        raise ValueError(
                            f"MXFP8 MoE layer {module_name or '<root>'}.{name} has shape {tuple(scale.shape)}; "
                            "the current Ascend MoE kernel requires an even number of scale groups"
                        )
            if not initial_load:
                if not getattr(module, "_mxfp8_transformed", False):
                    raise RuntimeError("MXFP8 hot reload requires the initial postprocessing to be complete")
                if not callable(getattr(scheme, "restore_weights_for_rl_loading", None)):
                    raise RuntimeError("This vllm-ascend MXFP8 scheme has no weight reload hook")
                self._reload_modules.append((module, method, scheme))

        self._runtime_storage = []
        if not initial_load:
            # detach() preserves the original shape/stride/storage even when the
            # scheme later assigns parameter.data to a newly allocated tensor.
            transformed_modules = [module for module, _method, _scheme in self._reload_modules]
            transformed_modules.extend(module for module, _method in self._float_modules)
            for module in transformed_modules:
                for name, parameter in module.named_parameters(recurse=False):
                    self._runtime_storage.append((module, name, parameter, parameter.detach()))
            # Some FLOAT router paths use this derived tensor directly. Keep its
            # captured storage too, and refresh values after postprocessing.
            self._runtime_fp32_weights = [
                (module, module.weight_fp32, module.weight_fp32.detach())
                for module, _method in self._float_modules
                if hasattr(module, "weight_fp32")
            ]
            for module, _method, scheme in self._reload_modules:
                _restore_mxfp8_checkpoint_layout(module, scheme)

    @torch.no_grad()
    def load_bucket(self, weights):
        if self._finished:
            raise RuntimeError("Cannot load a bucket after finishing a weight update")
        weights = list(weights)
        bucket_names = set()
        for name, tensor in weights:
            if name not in self._expected:
                raise ValueError(f"Unexpected exported tensor: {name}")
            if name in self._seen or name in bucket_names:
                raise ValueError(f"Duplicate exported tensor: {name}")
            expected_shape, expected_dtype = self._expected[name]
            if tuple(tensor.shape) != expected_shape or str(tensor.dtype).removeprefix("torch.") != expected_dtype:
                raise ValueError(
                    f"Exported tensor {name} has {tuple(tensor.shape)}/{tensor.dtype}; "
                    f"expected {expected_shape}/{expected_dtype}"
                )
            bucket_names.add(name)
        with _parameter_subclasses(self.model):
            self.model.load_weights(weights)
        self._seen.update(bucket_names)

    @torch.no_grad()
    def finish(self):
        if self._finished:
            raise RuntimeError("Weight update has already finished")
        missing = self._expected.keys() - self._seen
        if missing:
            raise ValueError(f"Missing exported tensors: {', '.join(sorted(missing))}")
        if not self.initial_load:
            for module, method, _scheme in self._reload_modules:
                method.process_weights_after_loading(module)
            # The existing Ascend MoE loader patch restores FLOAT checkpoint
            # layout while loading. Reapply its runtime transform once per round.
            for module, method in self._float_modules:
                method.process_weights_after_loading(module)
            for module, name, parameter, runtime_data in self._runtime_storage:
                new_data = getattr(module, name).detach()
                if new_data.shape != runtime_data.shape or new_data.dtype != runtime_data.dtype:
                    raise RuntimeError(f"Runtime layout changed during prequantized reload: {name}")
                if new_data.data_ptr() != runtime_data.data_ptr() or new_data.stride() != runtime_data.stride():
                    # restore/reshape can leave overlapping views of the same
                    # storage. Only that case needs an extra allocation.
                    if new_data.untyped_storage().data_ptr() == runtime_data.untyped_storage().data_ptr():
                        new_data = new_data.clone()
                    runtime_data.copy_(new_data)
                parameter.data = runtime_data
                module.register_parameter(name, parameter)
            for module, original, runtime_data in self._runtime_fp32_weights:
                runtime_data.copy_(module.weight_fp32)
                module.weight_fp32 = original
        self._finished = True
