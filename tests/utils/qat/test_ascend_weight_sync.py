# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

"""CPU contracts for describing and exporting effective training forward weights."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from verl.utils.qat.ascend_weight_sync import (
    AscendTensorSpec,
    build_ascend_weight_sync_plan,
    collect_ascend_module_specs,
    iter_ascend_weight_sync_params,
)
from verl.utils.qat.mxfp_linear import MXFPQATConfig, MXFPQATLinear
from verl.utils.qat.mxfp_quant import quantize_mxfp8
from verl.utils.qat.mxfp_rotation import rotate_blocks


def _config(**kwargs):
    return {"enable": True, "ignore_patterns": [], "experts": {"enable": True}, **kwargs}


def _linear(config=None, *, inputs=32):
    return MXFPQATLinear.from_linear(
        nn.Linear(inputs, 32, bias=False, dtype=torch.bfloat16),
        config or MXFPQATConfig(quant_backend="torch"),
    )


def _plan(model, **kwargs):
    return build_ascend_weight_sync_plan(_config(**kwargs), collect_ascend_module_specs(model))


def _export(model, plan):
    return dict(iter_ascend_weight_sync_params(model.state_dict().items(), plan))


def _assert_same_payload(actual, expected):
    # The uint8 view compares the encoded FP8 bits rather than tolerating drift.
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_real_fp8_payload_and_e8m0_scale_have_known_cpu_values():
    model = nn.Module()
    model.proj = _linear()
    with torch.no_grad():
        model.proj.weight[0].fill_(1.0)
        model.proj.weight[1].fill_(2.0)
        model.proj.weight[2].fill_(-0.5)
        model.proj.weight[3:].zero_()
    original = model.proj.weight.detach().clone()
    plan = _plan(model)
    exported = _export(model, plan)

    assert exported["proj.weight"].dtype == torch.float8_e4m3fn
    assert exported["proj.weight_scale"].dtype == torch.uint8
    assert exported["proj.weight_scale"].shape == (32, 1)
    # OCP MXFP8 shares floor(log2(amax))-8 per block, encoded with bias 127.
    assert exported["proj.weight_scale"][:3, 0].tolist() == [119, 120, 118]
    assert exported["proj.weight"].float()[:3, 0].tolist() == [256.0, 256.0, -256.0]
    scale = torch.pow(2.0, exported["proj.weight_scale"].float() - 127)
    reconstructed = exported["proj.weight"].float() * scale.repeat_interleave(32, dim=-1)
    torch.testing.assert_close(reconstructed, original.float(), rtol=0, atol=0)
    assert torch.equal(model.proj.weight, original)
    assert plan.quant_description["proj.weight"] == "W8A8_MXFP8"
    assert plan.manifest == [
        {"name": "proj.weight", "shape": [32, 32], "dtype": "float8_e4m3fn"},
        {"name": "proj.weight_scale", "shape": [32, 1], "dtype": "uint8"},
    ]


def test_unsharded_scale_groups_survive_bucketed_loading_into_local_tp_shapes():
    from verl.utils.vllm.prequantized_weights import PrequantizedWeightLoad

    model = nn.Module()
    model.proj = _linear(inputs=128)
    with torch.no_grad():
        model.proj.weight.copy_(torch.tensor([1.0, 2.0, 4.0, 8.0]).repeat_interleave(32).expand(32, -1))
    plan = _plan(model)
    exported = _export(model, plan)
    assert plan.manifest[1] == {"name": "proj.weight_scale", "shape": [32, 4], "dtype": "uint8"}

    class LocalTarget(nn.Module):
        """CPU stand-in for a row-parallel loader, before Ascend postprocessing."""

        def __init__(self, rank):
            super().__init__()
            self.rank = rank
            self.register_buffer("weight", torch.empty(32, 64, dtype=torch.float8_e4m3fn))
            self.register_buffer("scale", torch.empty(32, 2, dtype=torch.uint8))

        def load_weights(self, weights):
            for name, value in weights:
                local = value.chunk(2, dim=1)[self.rank]
                (self.scale if name.endswith("_scale") else self.weight).copy_(local)

    for rank in (0, 1):
        target = LocalTarget(rank)
        load = PrequantizedWeightLoad(target, plan.manifest, initial_load=True)
        load.load_bucket([("proj.weight", exported["proj.weight"])])
        load.load_bucket([("proj.weight_scale", exported["proj.weight_scale"])])
        load.finish()
        expected_scales = torch.tensor([119 + 2 * rank, 120 + 2 * rank], dtype=torch.uint8)
        assert torch.equal(target.scale, expected_scales.expand(32, -1))
        assert torch.equal(target.weight.float(), torch.full((32, 64), 256.0))


def test_rotation_matches_training_forward_transform_without_mutating_master():
    model = nn.Module()
    config = MXFPQATConfig(quant_backend="torch", rotation_enable=True, rotation_seed=7)
    model.proj = _linear(config)
    original = model.proj.weight.detach().clone()
    plan = _plan(model)
    exported = _export(model, plan)
    expected, expected_scale = quantize_mxfp8(rotate_blocks(original, config.rotation), quant_backend="torch")

    _assert_same_payload(exported["proj.weight"], expected)
    assert torch.equal(exported["proj.weight_scale"], expected_scale)
    assert torch.equal(model.proj.weight, original)
    assert plan.quant_description["rotation_enable"] is True
    assert plan.quant_description["rotation_seed"] == 7
    assert plan.quant_description["rotation_targets"] == ["fprop"]


@pytest.mark.parametrize("rotation", [False, True])
def test_forward_parameter_cast_precedes_quantization_and_preserves_master_and_buffers(rotation):
    model = nn.Module()
    config = MXFPQATConfig(quant_backend="torch", rotation_enable=rotation)
    model.proj = _linear(config).float()
    model.norm = nn.LayerNorm(32)
    model.register_buffer("running_bias", torch.ones(32, dtype=torch.float32))
    with torch.no_grad():
        model.proj.weight.fill_(1.064)
    original = model.proj.weight.detach().clone()
    plan = build_ascend_weight_sync_plan(_config(), collect_ascend_module_specs(model), param_dtype=torch.bfloat16)
    exported = _export(model, plan)
    expected, scale = quantize_mxfp8(rotate_blocks(original.bfloat16(), config.rotation), quant_backend="torch")
    from_master, _ = quantize_mxfp8(rotate_blocks(original, config.rotation), quant_backend="torch")

    _assert_same_payload(exported["proj.weight"], expected)
    assert not torch.equal(expected.view(torch.uint8), from_master.view(torch.uint8))
    assert torch.equal(exported["proj.weight_scale"], scale)
    assert torch.equal(model.proj.weight, original)
    assert model.proj.weight.dtype == torch.float32
    assert exported["norm.weight"].dtype == torch.bfloat16
    assert exported["running_bias"].dtype == torch.float32
    manifest = {item["name"]: item for item in plan.manifest}
    assert manifest["norm.weight"]["dtype"] == "bfloat16"
    assert manifest["running_bias"]["dtype"] == "float32"


def test_each_sync_quantizes_the_latest_master_instead_of_a_training_cache():
    model = nn.Module()
    model.proj = _linear()
    plan = _plan(model)
    with torch.no_grad():
        model.proj.weight.fill_(1.0)
    before = _export(model, plan)
    with torch.no_grad():
        model.proj.weight.fill_(2.0)
    after = _export(model, plan)
    assert before["proj.weight_scale"].unique().item() == 119
    assert after["proj.weight_scale"].unique().item() == 120
    assert model.proj.weight.unique().item() == 2.0


def test_float_parameters_tied_aliases_and_persistent_buffers_are_complete():
    model = nn.Module()
    model.proj = _linear()
    model.embed_tokens = nn.Embedding(10, 32, dtype=torch.bfloat16)
    model.lm_head = nn.Linear(32, 10, bias=False, dtype=torch.bfloat16)
    model.lm_head.weight = model.embed_tokens.weight
    model.norm = nn.LayerNorm(32, dtype=torch.bfloat16)
    model.register_buffer("token_count", torch.tensor(3))
    model.register_buffer("scratch", torch.ones(1), persistent=False)
    plan = _plan(model, ignore_patterns=["lm_head"])
    exported = _export(model, plan)
    manifest = {item["name"]: item for item in plan.manifest}

    assert "scratch" not in manifest
    for name in ("embed_tokens.weight", "lm_head.weight", "norm.weight", "norm.bias", "token_count"):
        assert plan.quant_description[name] == "FLOAT"
        assert torch.equal(exported[name], model.state_dict()[name])
        assert manifest[name]["dtype"] == str(exported[name].dtype).removeprefix("torch.")
    assert set(exported) == set(manifest)


def _packed_model(config):
    model = nn.Module()
    model.model = nn.Module()
    block = nn.Module()
    model.model.layers = nn.ModuleList([block])
    block.mlp = nn.Module()
    block.mlp._mxfp_expert_qat_runtime_config = config
    block.mlp.experts = nn.Module()
    block.mlp.experts.gate_up_proj = nn.Parameter(torch.randn(2, 64, 32, dtype=torch.bfloat16))
    block.mlp.experts.down_proj = nn.Parameter(torch.randn(2, 32, 32, dtype=torch.bfloat16))
    return model


@pytest.mark.parametrize("rotation", [False, True])
def test_packed_experts_expand_before_quantization_and_pair_every_scale(rotation):
    config = MXFPQATConfig(quant_backend="torch", rotation_enable=rotation, rotation_seed=5)
    model = _packed_model(config)
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    plan = _plan(model)
    exported = _export(model, plan)
    base = "model.layers.0.mlp.experts"
    for expert in range(2):
        gate_up = original[base + ".gate_up_proj"][expert]
        for projection, weight in (
            ("gate_proj", gate_up[:32]),
            ("up_proj", gate_up[32:]),
            ("down_proj", original[base + ".down_proj"][expert]),
        ):
            name = f"{base}.{expert}.{projection}.weight"
            expected, scale = quantize_mxfp8(rotate_blocks(weight, config.rotation), quant_backend="torch")
            _assert_same_payload(exported[name], expected)
            assert torch.equal(exported[name + "_scale"], scale)
            assert plan.quant_description[name] == "W8A8_MXFP8"
    assert len(plan.manifest) == 12
    assert set(exported) == {item["name"] for item in plan.manifest}
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in original.items())


def test_float_packed_experts_expand_without_inventing_scales():
    model = _packed_model(None)
    plan = _plan(model, experts={"enable": False})
    exported = _export(model, plan)
    assert len(exported) == 6
    assert all(value.dtype == torch.bfloat16 for value in exported.values())
    assert all(plan.quant_description[name] == "FLOAT" for name in exported)
    assert all(not name.endswith("_scale") for name in exported)


def test_fallback_conflict_is_rejected_before_any_export():
    model = _packed_model(MXFPQATConfig(quant_backend="torch"))
    with pytest.raises(ValueError, match="fallback_layers conflicts with the effective training"):
        _plan(model, fallback_layers="0-3")


def test_ignore_pattern_conflict_is_not_silently_downgraded():
    model = nn.Module()
    model.proj = _linear()
    with pytest.raises(ValueError, match="ignore_patterns conflicts"):
        _plan(model, ignore_patterns=["re:^proj$"])


@pytest.mark.parametrize("mode", ["mxfp_w4a8", "mxfp_w4a4"])
def test_unsupported_packed_modes_require_their_own_export_and_reload(mode):
    model = nn.Module()
    model.proj = _linear(MXFPQATConfig(mode=mode, quant_backend="torch"))
    with pytest.raises(NotImplementedError, match="packed exporter and reload support"):
        _plan(model)


def test_unknown_effective_mode_is_not_reinterpreted_as_mxfp8():
    model = nn.Module()
    model.proj = _linear()
    model.proj.qat_config = {"mode": "w8a16"}
    with pytest.raises(ValueError, match="Unsupported MXFP QAT mode"):
        _plan(model)


def test_missing_effective_config_is_not_guessed_from_global_mode():
    model = nn.Module()
    model.proj = nn.Linear(32, 32, bias=False)
    with pytest.raises(ValueError, match="does not expose its effective QAT configuration"):
        _plan(model)


def test_backward_only_quantization_exports_float_forward_weights():
    model = nn.Module()
    model.proj = _linear(MXFPQATConfig(quant_backend="torch", fake_quant_targets=("dgrad", "wgrad")))
    plan = _plan(model)
    exported = _export(model, plan)
    assert plan.quant_description["proj.weight"] == "FLOAT"
    assert torch.equal(exported["proj.weight"], model.proj.weight)
    assert set(exported) == {"proj.weight"}


@pytest.mark.parametrize(
    "members",
    [("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj"), ("experts.0.gate_proj", "experts.1.gate_proj")],
)
def test_fused_members_cannot_mix_float_and_quantized_recipes(members):
    model = nn.Module()
    model.group = nn.Module()
    for index, name in enumerate(members):
        module = model.group
        *parents, child = name.split(".")
        for parent in parents:
            if not hasattr(module, parent):
                module.add_module(parent, nn.Module())
            module = getattr(module, parent)
        module.add_module(child, nn.Linear(32, 32, bias=False) if index == 1 else _linear())
    with pytest.raises(ValueError, match="incompatible effective forward configurations"):
        _plan(model, ignore_patterns=[f"group.{members[1]}"])


def test_disabled_qat_module_exports_float_even_if_config_has_quantization_and_rotation():
    model = nn.Module()
    model.proj = _linear(MXFPQATConfig(mode="mxfp_w4a4", quant_backend="torch", rotation_enable=True))
    model.proj.fake_quant_enabled = False
    plan = _plan(model)
    exported = _export(model, plan)
    assert plan.quant_description["proj.weight"] == "FLOAT"
    assert plan.quant_description["rotation_enable"] is False
    assert set(exported) == {"proj.weight"}
    assert torch.equal(exported["proj.weight"], model.proj.weight)


def test_unrepresentable_per_layer_rotation_fails_early():
    model = nn.Module()
    model.a = _linear(MXFPQATConfig(quant_backend="torch", rotation_enable=True, rotation_seed=1))
    model.b = _linear(MXFPQATConfig(quant_backend="torch", rotation_enable=True, rotation_seed=2))
    with pytest.raises(NotImplementedError, match="Different per-module rotations"):
        _plan(model)


def test_unaligned_weight_is_rejected_before_engine_startup():
    model = nn.Module()
    model.proj = _linear(inputs=31)
    with pytest.raises(ValueError, match="input dimension divisible by 32"):
        _plan(model)


def test_canonical_names_use_the_same_checkpoint_conversion_as_engine_export():
    from verl.utils.model import convert_weight_keys

    model = nn.Module()
    model.proj = _linear()
    model._checkpoint_conversion_mapping = {"^legacy": "proj"}
    plan = _plan(model)
    params = convert_weight_keys(model.state_dict(), model)
    exported = dict(iter_ascend_weight_sync_params(params.items(), plan))
    assert set(exported) == {"legacy.weight", "legacy.weight_scale"}
    assert plan.source_specs["legacy.weight"].module_name == "proj"


@pytest.mark.parametrize("case", ["missing", "duplicate", "unexpected", "shape", "dtype"])
def test_source_snapshot_validation(case):
    model = nn.Module()
    model.proj = _linear()
    plan = _plan(model)
    weight = model.proj.weight
    params = {
        "missing": [],
        "duplicate": [("proj.weight", weight), ("proj.weight", weight)],
        "unexpected": [("other.weight", weight)],
        "shape": [("proj.weight", weight[:1])],
        "dtype": [("proj.weight", weight.float())],
    }[case]
    with pytest.raises(ValueError):
        list(iter_ascend_weight_sync_params(params, plan))


def test_fake_float_payload_cannot_escape_as_real_fp8(monkeypatch):
    model = nn.Module()
    model.proj = _linear()
    plan = _plan(model)
    monkeypatch.setattr(
        "verl.utils.qat.ascend_weight_sync.quantize_mxfp8",
        lambda weight, **kwargs: (weight, torch.ones((32, 1), dtype=torch.uint8)),
    )
    with pytest.raises(ValueError, match="fake payloads cannot be transmitted"):
        _export(model, plan)


def test_scale_name_collision_is_rejected_in_both_source_orders():
    model = nn.Module()
    model.proj = _linear()
    specs = collect_ascend_module_specs(model)
    scale = AscendTensorSpec("proj.weight_scale", (32, 1), torch.uint8, "proj")
    for ordered in ({**specs, scale.name: scale}, {scale.name: scale, **specs}):
        with pytest.raises(ValueError, match="Duplicate canonical tensor name"):
            build_ascend_weight_sync_plan(_config(), ordered)


def test_disabled_rotation_parameters_do_not_create_fake_recipe_conflicts():
    model = nn.Module()
    model.a = _linear()
    model.b = _linear(replace(MXFPQATConfig(quant_backend="torch"), rotation_seed=99))
    assert _plan(model).quant_description["rotation_enable"] is False


def test_startup_metadata_plan_does_not_read_or_quantize_tensor_values(monkeypatch):
    model = nn.Module()
    model.proj = _linear().to("meta")

    def unexpected_quantization(*args, **kwargs):
        raise AssertionError("startup planning must only inspect metadata")

    monkeypatch.setattr("verl.utils.qat.ascend_weight_sync.quantize_mxfp8", unexpected_quantization)
    plan = _plan(model)
    assert plan.manifest[0]["shape"] == [32, 32]


def test_real_qwen3_moe_metadata_and_export_names_match_training_layout():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    from verl.utils.model import convert_weight_keys
    from verl.utils.qat.core import apply_qat

    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=64,
    )
    model = Qwen3MoeForCausalLM(config)
    qat = _config(
        mode="mxfp_w8a8",
        group_size=32,
        quant_backend="torch",
        rotation_enable=True,
        ignore_patterns=["lm_head", "embed_tokens", "re:.*mlp.gate$"],
    )
    model = apply_qat(model, qat)
    specs = collect_ascend_module_specs(model)
    plan = build_ascend_weight_sync_plan(qat, specs, param_dtype=torch.bfloat16)
    exported = dict(iter_ascend_weight_sync_params(convert_weight_keys(model.state_dict(), model).items(), plan))
    base = "model.layers.0.mlp.experts"
    for expert in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"{base}.{expert}.{projection}.weight"
            assert plan.quant_description[name] == "W8A8_MXFP8"
            assert exported[name].dtype == torch.float8_e4m3fn
            assert exported[name + "_scale"].dtype == torch.uint8
    assert set(exported) == {item["name"] for item in plan.manifest}
    assert plan.quant_description["model.layers.0.mlp.gate.weight"] == "FLOAT"
