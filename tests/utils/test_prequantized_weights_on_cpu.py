# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace

import pytest
import torch

from verl.utils.vllm.prequantized_weights import PrequantizedWeightLoad


class AscendW8A8MXFP8DynamicLinearMethod:
    """CPU layout stand-in, including the scale reshape that allocates storage."""

    def __init__(self):
        self.calls = 0

    def process_weights_after_loading(self, layer):
        self.calls += 1
        layer.weight.data = layer.weight.data.t().contiguous()
        layer.weight_scale.data = layer.weight_scale.data.reshape(3, 2, 2).transpose(0, 1).contiguous()
        layer._mxfp8_transformed = True

    def restore_weights_for_rl_loading(self, layer):
        layer.weight.data = layer.weight.data.t()
        layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1).reshape(3, 4)
        layer._mxfp8_transformed = False


class QuantizedLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(3, 128, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(torch.zeros(3, 4, dtype=torch.uint8), requires_grad=False)
        scheme = AscendW8A8MXFP8DynamicLinearMethod()
        self.quant_method = SimpleNamespace(
            quant_method=scheme, process_weights_after_loading=scheme.process_weights_after_loading
        )


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = QuantizedLayer()
        self.norm = torch.nn.Parameter(torch.zeros(3), requires_grad=False)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, tensor in weights:
            params[name].data.copy_(tensor)


def exported_weights(offset):
    return [
        ("proj.weight", (torch.arange(384, dtype=torch.float32).reshape(3, 128) + offset).to(torch.float8_e4m3fn)),
        ("proj.weight_scale", (torch.arange(12, dtype=torch.uint8) + offset).reshape(3, 4)),
        ("norm", torch.full((3,), float(offset))),
    ]


def manifest_for(weights):
    return [{"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)} for name, tensor in weights]


def test_first_load_leaves_postprocessing_to_base_loader():
    model = Model()
    weights = exported_weights(1)
    load = PrequantizedWeightLoad(model, manifest_for(weights), initial_load=True)
    load.load_bucket(weights[:1])
    load.load_bucket(weights[1:])
    load.finish()
    assert model.proj.quant_method.quant_method.calls == 0
    torch.testing.assert_close(model.proj.weight.float(), weights[0][1].float())


def test_repeated_hot_updates_keep_graph_storage_and_transform_once_per_round():
    model = Model()
    model.proj.quant_method.process_weights_after_loading(model.proj)
    original = {name: (param, param.data_ptr(), param.stride()) for name, param in model.named_parameters()}
    # The transpose+reshape on runtime scale really does allocate a new buffer.
    scale = model.proj.weight_scale
    assert scale.transpose(0, 1).reshape(3, 4).data_ptr() != scale.data_ptr()
    for offset in (2, 7):
        weights = exported_weights(offset)
        load = PrequantizedWeightLoad(model, manifest_for(weights))
        load.load_bucket(weights[:1])
        # No postprocessing between buckets: scale remains in checkpoint layout.
        assert model.proj.weight_scale.shape == (3, 4)
        load.load_bucket(weights[1:])
        calls_before_finish = model.proj.quant_method.quant_method.calls
        load.finish()
        assert model.proj.quant_method.quant_method.calls == calls_before_finish + 1
        for name, param in model.named_parameters():
            old_param, ptr, stride = original[name]
            assert param is old_param
            assert param.data_ptr() == ptr
            assert param.stride() == stride
        torch.testing.assert_close(model.proj.weight.float(), weights[0][1].t().contiguous().float())
        assert torch.equal(model.proj.weight_scale, weights[1][1].reshape(3, 2, 2).transpose(0, 1))
        torch.testing.assert_close(model.norm, weights[2][1])


@pytest.mark.parametrize("bad", ["duplicate", "unknown", "shape", "dtype", "missing"])
def test_manifest_rejects_corrupt_or_incomplete_stream(bad):
    model = Model()
    weights = exported_weights(1)
    load = PrequantizedWeightLoad(model, manifest_for(weights), initial_load=True)
    if bad == "duplicate":
        with pytest.raises(ValueError, match="Duplicate"):
            load.load_bucket([weights[0], weights[0]])
    elif bad == "unknown":
        with pytest.raises(ValueError, match="Unexpected"):
            load.load_bucket([("unknown", weights[0][1])])
    elif bad == "shape":
        with pytest.raises(ValueError, match="expected"):
            load.load_bucket([("proj.weight", weights[0][1].reshape(-1))])
    elif bad == "dtype":
        with pytest.raises(ValueError, match="expected"):
            load.load_bucket([("proj.weight_scale", weights[1][1].float())])
    else:
        load.load_bucket(weights[:1])
        with pytest.raises(ValueError, match="Missing.*norm"):
            load.finish()


def test_other_quantization_formats_fail_before_loading():
    model = Model()
    model.proj.quant_method = SimpleNamespace(quant_method=object())
    with pytest.raises(ValueError, match="only ModelSlim W8A8_MXFP8/FLOAT"):
        PrequantizedWeightLoad(model, manifest_for(exported_weights(1)), initial_load=True)


def test_parameter_subclass_restored_after_loader_error():
    class LoaderParameter(torch.nn.Parameter):
        pass

    model = Model()
    model.norm.subclass_type = LoaderParameter

    def fail(_weights):
        assert isinstance(model.norm, LoaderParameter)
        raise RuntimeError("loader failed")

    model.load_weights = fail
    load = PrequantizedWeightLoad(model, manifest_for(exported_weights(1)), initial_load=True)
    with pytest.raises(RuntimeError, match="loader failed"):
        load.load_bucket(exported_weights(1))
    assert type(model.norm) is torch.nn.Parameter


def test_float_moe_postprocess_replaces_parameters_but_graph_still_sees_new_values():
    class AscendUnquantizedFusedMoEMethod:
        def process_weights_after_loading(self, layer):
            for name in ("w13_weight", "w2_weight"):
                data = getattr(layer, name).data.transpose(1, 2).contiguous()
                layer.register_parameter(name, torch.nn.Parameter(data, requires_grad=False))

    class FloatMoe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w13_weight = torch.nn.Parameter(torch.zeros(1, 6, 4), requires_grad=False)
            self.w2_weight = torch.nn.Parameter(torch.zeros(1, 4, 3), requires_grad=False)
            self.quant_method = AscendUnquantizedFusedMoEMethod()

        def load_weights(self, weights):
            for name, value in weights:
                param = getattr(self, name)
                # Same checkpoint-layout restore as the existing Ascend loader
                # wrapper, performed only on the first shard of each parameter.
                if param.shape != value.shape:
                    param.data = param.data.transpose(1, 2)
                param.data.copy_(value)

    model = FloatMoe()
    model.quant_method.process_weights_after_loading(model)
    original = {name: (param, param.data_ptr()) for name, param in model.named_parameters()}
    for offset in (4, 9):
        weights = [
            ("w13_weight", torch.arange(24, dtype=torch.float32).reshape(1, 6, 4) + offset),
            ("w2_weight", torch.arange(12, dtype=torch.float32).reshape(1, 4, 3) + offset),
        ]
        load = PrequantizedWeightLoad(model, manifest_for(weights))
        load.load_bucket(weights[:1])
        load.load_bucket(weights[1:])
        load.finish()
        for name, value in weights:
            param = getattr(model, name)
            assert param is original[name][0]
            assert param.data_ptr() == original[name][1]
            torch.testing.assert_close(param, value.transpose(1, 2))
