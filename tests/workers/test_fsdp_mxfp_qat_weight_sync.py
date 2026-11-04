from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("peft")

from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


@pytest.mark.parametrize("mode", ["mxfp_w8a8", "mxfp_w4a8", "mxfp_w4a4"])
def test_mxfp_weight_sync_keeps_high_precision_weights(mode, monkeypatch):
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.module = torch.nn.Linear(32, 32)
    engine.model_config = SimpleNamespace(lora={})
    engine.engine_config = SimpleNamespace(mixed_precision=None)
    engine._qat_enabled = True
    engine._qat_config = SimpleNamespace(mode=mode, group_size=32, ignore_patterns=[])
    engine._is_offload_param = False
    engine._uses_fsdp2_cpu_offload_policy = True
    engine.rank = 0

    monkeypatch.setattr(transformer_impl, "log_gpu_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: torch.device("cpu"))

    original_params = engine.module.state_dict()
    params, peft_config = engine.get_per_tensor_param()
    synced_params = dict(params)

    assert peft_config is None
    assert synced_params.keys() == original_params.keys()
    for name, tensor in original_params.items():
        torch.testing.assert_close(synced_params[name], tensor)


@pytest.fixture
def cpu_fsdp_mesh(tmp_path):
    """Use real DTensors and collectives without requiring an accelerator."""
    from torch.distributed.device_mesh import init_device_mesh

    if not torch.distributed.is_gloo_available():
        pytest.skip("The CPU FSDP2 integration test requires Gloo")
    owns_group = not torch.distributed.is_initialized()
    if not owns_group and (torch.distributed.get_world_size() != 1 or torch.distributed.get_backend() != "gloo"):
        pytest.skip("The CPU FSDP2 integration test requires an isolated single-rank Gloo group")
    if owns_group:
        store = torch.distributed.FileStore(str(tmp_path / "rendezvous"), 1)
        torch.distributed.init_process_group("gloo", store=store, rank=0, world_size=1, timeout=timedelta(seconds=30))
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        if owns_group:
            torch.distributed.destroy_process_group()


@pytest.mark.parametrize(
    ("master_dtype", "forward_dtype"),
    [
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
    ],
)
def test_prequantized_fsdp2_export_uses_forward_dtype_and_latest_master(
    master_dtype, forward_dtype, cpu_fsdp_mesh, monkeypatch
):
    from torch.distributed.tensor import DTensor

    from verl.utils.fsdp_utils import MixedPrecisionPolicy, fully_shard
    from verl.utils.qat.core import QATConfig, apply_qat
    from verl.utils.qat.mxfp_quant import quantize_mxfp8
    from verl.utils.qat.mxfp_rotation import rotate_blocks

    model = torch.nn.Module()
    model.proj = torch.nn.Linear(32, 32, bias=True, dtype=master_dtype)
    model.norm = torch.nn.LayerNorm(32, dtype=master_dtype)
    model.lm_head = torch.nn.Linear(32, 8, bias=False, dtype=master_dtype)
    model.register_buffer("running_bias", torch.ones(32, dtype=torch.float32))
    with torch.no_grad():
        model.proj.weight.fill_(1.064)
        model.norm.weight.fill_(1.064)
    qat_config = QATConfig(
        enable=True,
        mode="mxfp_w8a8",
        group_size=32,
        quant_backend="torch",
        ignore_patterns=["lm_head"],
        rotation_enable=True,
        rotation_seed=7,
    )
    model = apply_qat(model, qat_config)
    fully_shard(model, mesh=cpu_fsdp_mesh, mp_policy=MixedPrecisionPolicy(param_dtype=forward_dtype))
    assert isinstance(model.proj.weight, DTensor)

    engine = object.__new__(FSDPEngineWithLMHead)
    engine.module = model
    engine.model_config = SimpleNamespace(lora={})
    engine.engine_config = SimpleNamespace(strategy="fsdp2")
    engine._qat_enabled = True
    engine._qat_config = qat_config
    engine._is_lora = False
    engine._is_offload_param = False
    engine._uses_fsdp2_cpu_offload_policy = True
    engine._autocast_dtype = forward_dtype
    engine.rank = 0
    monkeypatch.setattr(transformer_impl, "log_gpu_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: torch.device("cpu"))

    def masters():
        return {
            name: (value.full_tensor() if isinstance(value, DTensor) else value).clone()
            for name, value in model.state_dict().items()
        }

    initial = masters()
    first_params, peft_config = engine.get_per_tensor_param(prequantized=True)
    first = dict(first_params)
    plan = engine.get_ascend_weight_sync_plan()
    assert peft_config is None
    assert set(first) == set(initial) | {"proj.weight_scale"}
    assert set(first) == {item["name"] for item in plan.manifest}
    assert plan.quant_description["proj.weight"] == "W8A8_MXFP8"

    for name, value in initial.items():
        if name == "proj.weight":
            continue
        expected = value if name == "running_bias" else value.to(forward_dtype)
        assert first[name].dtype == expected.dtype
        torch.testing.assert_close(first[name], expected, rtol=0, atol=0)
        assert plan.quant_description[name] == "FLOAT"
    expected, scale = quantize_mxfp8(
        rotate_blocks(initial["proj.weight"].to(forward_dtype), model.proj.qat_config.rotation), quant_backend="torch"
    )
    assert first["proj.weight"].dtype == torch.float8_e4m3fn
    assert torch.equal(first["proj.weight"].view(torch.uint8), expected.view(torch.uint8))
    assert first["proj.weight_scale"].dtype == torch.uint8
    assert torch.equal(first["proj.weight_scale"], scale)
    for name, value in masters().items():
        assert torch.equal(value, initial[name])
        assert value.dtype == initial[name].dtype

    with torch.no_grad():
        model.proj.weight.mul_(2)
        model.norm.weight.add_(0.5)
        model.running_bias.add_(1)
    latest = masters()
    second_params, peft_config = engine.get_per_tensor_param(prequantized=True)
    second = dict(second_params)
    assert peft_config is None
    assert engine.get_ascend_weight_sync_plan() is plan
    expected, scale = quantize_mxfp8(
        rotate_blocks(latest["proj.weight"].to(forward_dtype), model.proj.qat_config.rotation), quant_backend="torch"
    )
    assert torch.equal(second["proj.weight"].view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(second["proj.weight_scale"], scale)
    assert torch.equal(second["proj.weight_scale"].int(), first["proj.weight_scale"].int() + 1)
    torch.testing.assert_close(second["norm.weight"], latest["norm.weight"].to(forward_dtype), rtol=0, atol=0)
    assert torch.equal(second["running_bias"], latest["running_bias"])
    for name, value in masters().items():
        assert torch.equal(value, latest[name])
        assert value.dtype == latest[name].dtype
