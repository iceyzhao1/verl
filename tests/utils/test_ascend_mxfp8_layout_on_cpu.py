# Copyright 2026 verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise real Ascend MXFP8/FLOAT layout methods without the NPU runtime.

Set VLLM_ASCEND_SOURCE_DIR to the target vllm-ascend checkout. An installed
package or a sibling checkout is also detected. Only the two CPU-compatible
layout methods are compiled from that source; NPU kernels and model construction
are deliberately outside this test's coverage.
"""

import ast
import importlib.util
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from verl.utils.vllm.prequantized_weights import PrequantizedWeightLoad


def _ascend_layout_source():
    relative_path = Path("vllm_ascend/quantization/methods/w8a8_mxfp8.py")
    configured = os.environ.get("VLLM_ASCEND_SOURCE_DIR")
    if configured:
        path = Path(configured) / relative_path
        if not path.is_file():
            pytest.fail(f"VLLM_ASCEND_SOURCE_DIR does not contain {relative_path}: {configured}")
        return path
    sibling = Path(__file__).resolve().parents[2].parent / "vllm-ascend" / relative_path
    if sibling.is_file():
        return sibling
    spec = importlib.util.find_spec("vllm_ascend")
    if spec is not None and spec.submodule_search_locations:
        installed = Path(next(iter(spec.submodule_search_locations))) / "quantization/methods/w8a8_mxfp8.py"
        if installed.is_file():
            return installed
    pytest.skip("Set VLLM_ASCEND_SOURCE_DIR to run the real Ascend MXFP8 layout regression")


def _real_scheme(source, class_name):
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method_names = {"process_weights_after_loading", "restore_weights_for_rl_loading"}
    methods = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in method_names]
    assert {method.name for method in methods} == method_names, f"Missing MXFP8 reload/layout hook in {source}"
    scheme_class = ast.ClassDef(name=class_name, bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[scheme_class], type_ignores=[]))
    namespace = {"torch": torch, "F": F, "logger": logging.getLogger(__name__)}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[class_name]()


class LayoutModel(torch.nn.Module):
    def __init__(self, scheme, shapes):
        super().__init__()
        for name, shape in shapes.items():
            dtype = torch.uint8 if "scale" in name else torch.float8_e4m3fn
            self.register_parameter(name, torch.nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False))
        self.quant_method = SimpleNamespace(
            quant_method=scheme, process_weights_after_loading=scheme.process_weights_after_loading
        )

    def load_weights(self, weights):
        for name, tensor in weights:
            getattr(self, name).data.copy_(tensor)


def _exported(shapes, offset):
    weights = []
    for name, shape in shapes.items():
        values = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
        values = (values % 97) + offset
        dtype = torch.uint8 if "scale" in name else torch.float8_e4m3fn
        weights.append((name, values.to(dtype)))
    return weights


@pytest.mark.parametrize(
    "class_name,shapes",
    [
        (
            "AscendW8A8MXFP8DynamicLinearMethod",
            {"weight": (3, 128), "weight_scale": (3, 4)},
        ),
        (
            "AscendW8A8MXFP8DynamicLinearMethod",
            {"weight": (3, 96), "weight_scale": (3, 3)},
        ),
        (
            "AscendW8A8MXFP8DynamicFusedMoEMethod",
            {
                "w13_weight": (2, 128, 64),
                "w13_weight_scale": (2, 128, 2),
                "w2_weight": (2, 64, 64),
                "w2_weight_scale": (2, 64, 2),
            },
        ),
    ],
    ids=["linear-even-groups", "linear-odd-groups", "moe"],
)
@torch.no_grad()
def test_real_ascend_layout_two_updates_preserve_captured_storage(class_name, shapes):
    source = _ascend_layout_source()
    model = LayoutModel(_real_scheme(source, class_name), shapes)
    model.quant_method.process_weights_after_loading(model)
    original = {
        name: (param, param.detach(), param.data_ptr(), param.stride()) for name, param in model.named_parameters()
    }
    for offset in (2, 5):
        weights = _exported(shapes, offset)
        manifest = [{"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)} for name, tensor in weights]
        update = PrequantizedWeightLoad(model, manifest)
        for item in weights:
            update.load_bucket([item])
        update.finish()

        # A fresh first load provides the expected runtime values/layout using
        # the same real Ascend method, without going through restore/reload.
        reference = LayoutModel(_real_scheme(source, class_name), shapes)
        reference.load_weights(weights)
        reference.quant_method.process_weights_after_loading(reference)
        for name, parameter in model.named_parameters():
            original_parameter, captured_tensor, ptr, stride = original[name]
            assert parameter is original_parameter
            assert parameter.data_ptr() == ptr
            assert parameter.stride() == stride
            expected = getattr(reference, name)
            torch.testing.assert_close(parameter.float(), expected.float(), rtol=0, atol=0)
            torch.testing.assert_close(captured_tensor.float(), expected.float(), rtol=0, atol=0)


def test_real_ascend_moe_odd_scale_groups_fail_before_first_bucket():
    class_name = "AscendW8A8MXFP8DynamicFusedMoEMethod"
    shapes = {
        "w13_weight": (2, 128, 96),
        "w13_weight_scale": (2, 128, 3),
        "w2_weight": (2, 96, 64),
        "w2_weight_scale": (2, 96, 2),
    }
    model = torch.nn.Module()
    model.experts = LayoutModel(_real_scheme(_ascend_layout_source(), class_name), shapes)
    weights = _exported(shapes, 1)
    manifest = [{"name": name, "shape": list(t.shape), "dtype": str(t.dtype)} for name, t in weights]
    with pytest.raises(ValueError, match=r"experts.w13_weight_scale.*\(2, 128, 3\).*even"):
        PrequantizedWeightLoad(model, manifest, initial_load=True)


def _real_float_method(moe):
    """Use actual FLOAT postprocessing; stub only base/device-format operations."""
    root = _ascend_layout_source().parents[3]
    source = root / ("vllm_ascend/ops/fused_moe/fused_moe.py" if moe else "vllm_ascend/ops/linear.py")
    class_name = "AscendUnquantizedFusedMoEMethod" if moe else "AscendUnquantizedLinearMethod"
    base_name = "UnquantizedFusedMoEMethod" if moe else "UnquantizedLinearMethod"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(
        node
        for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name == "process_weights_after_loading"
    )

    class Base:
        def process_weights_after_loading(self, layer):
            pass

        def _maybe_pad_weight(self, tensor):
            return tensor

    base_class = type(base_name, (Base,), {})
    scheme_class = ast.ClassDef(
        name=class_name, bases=[ast.Name(id=base_name, ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
    )
    module = ast.fix_missing_locations(ast.Module(body=[scheme_class], type_ignores=[]))
    namespace = {
        "torch": torch,
        base_name: base_class,
        "is_mxfp8_fake_quant_enabled": lambda: False,
        "get_ascend_config": lambda: SimpleNamespace(enable_fused_mc2=0),
        "_should_keep_nd_for_310p_weight": lambda _tensor: False,
        # A new CPU tensor stands in for an NZ format conversion allocation.
        # This tests pointer preservation, not the NPU's actual NZ format.
        "maybe_trans_nz": lambda tensor: tensor.clone(),
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[class_name]()


@torch.no_grad()
def test_real_float_moe_postprocess_preserves_captured_storage_across_expert_buckets():
    class FloatMoe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w13_weight = torch.nn.Parameter(torch.zeros(2, 6, 4, dtype=torch.bfloat16), requires_grad=False)
            self.w2_weight = torch.nn.Parameter(torch.zeros(2, 4, 3, dtype=torch.bfloat16), requires_grad=False)
            self.quant_method = _real_float_method(moe=True)

        def load_weights(self, weights):
            for name, value in weights:
                expert_id, projection = name.split(".")
                param = self.w2_weight if projection == "down" else self.w13_weight
                # The existing Ascend wrapper restores the checkpoint orientation
                # before calling the upstream expert/shard loader.
                if (projection == "down" and param.shape[2] == 4) or (projection != "down" and param.shape[1] == 4):
                    param.data = param.data.transpose(1, 2)
                destination = param.data[int(expert_id)]
                if projection != "down":
                    destination = destination[:3] if projection == "gate" else destination[3:]
                destination.copy_(value)

    model = FloatMoe()
    model.quant_method.process_weights_after_loading(model)
    original = {name: (p, p.detach(), p.data_ptr()) for name, p in model.named_parameters()}
    for offset in (2, 5):
        weights = [
            (f"{expert}.{projection}", torch.full(shape, offset + expert, dtype=torch.bfloat16))
            for expert in range(2)
            for projection, shape in (("gate", (3, 4)), ("up", (3, 4)), ("down", (4, 3)))
        ]
        manifest = [{"name": name, "shape": list(t.shape), "dtype": str(t.dtype)} for name, t in weights]
        update = PrequantizedWeightLoad(model, manifest)
        for item in weights:
            update.load_bucket([item])
        update.finish()
        for name, param in model.named_parameters():
            saved, captured, ptr = original[name]
            assert param is saved and param.data_ptr() == ptr
            assert param.shape == ((2, 4, 6) if name == "w13_weight" else (2, 3, 4))
            for expert in range(2):
                torch.testing.assert_close(captured[expert], torch.full_like(captured[expert], offset + expert))


@torch.no_grad()
def test_real_float_router_postprocess_refreshes_captured_fp32_cache():
    class Router(torch.nn.Module):
        prefix = "router"
        precast_fp32_weight = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(3, 4, dtype=torch.bfloat16), requires_grad=False)
            self.quant_method = _real_float_method(moe=False)

        def load_weights(self, weights):
            for name, value in weights:
                getattr(self, name).copy_(value)

    model = Router()
    model.quant_method.process_weights_after_loading(model)
    original_weight, weight_ptr = model.weight, model.weight.data_ptr()
    original_cache, cache_ptr = model.weight_fp32, model.weight_fp32.data_ptr()
    captured_cache = model.weight_fp32.detach()
    for offset in (2, 5):
        weight = torch.full((3, 4), offset, dtype=torch.bfloat16)
        update = PrequantizedWeightLoad(model, [{"name": "weight", "shape": [3, 4], "dtype": "bfloat16"}])
        update.load_bucket([("weight", weight)])
        update.finish()
        assert model.weight is original_weight and model.weight.data_ptr() == weight_ptr
        assert model.weight_fp32 is original_cache and model.weight_fp32.data_ptr() == cache_ptr
        torch.testing.assert_close(captured_cache, weight.float())
