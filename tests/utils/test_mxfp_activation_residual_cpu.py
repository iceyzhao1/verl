"""Full activation-residual numerics with real QDQ, rotation, and autograd.

Only the Ascend grouped-matmul hardware boundary is replaced by CPU matmuls.
"""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from verl.utils.qat.mxfp_linear import MXFPQATConfig, MXFPQATLinear, _target_tensor
from verl.utils.qat.mxfp_moe import (
    _pad_expert_rows,
    _transform_tokens,
    _transform_weight,
    mxfp_grouped_matmul,
    prepare_mxfp_fprop_input,
)
from verl.utils.qat.mxfp_rotation import rotate_blocks


def _config(**kwargs):
    return MXFPQATConfig(mode="mxfp_w4a8", quant_backend="torch", **kwargs)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5", None])
@pytest.mark.parametrize("field", ["activation_residual_fprop_alpha", "activation_residual_bprop_alpha"])
def test_residual_config_rejects_invalid_alpha(field, value):
    with pytest.raises(ValueError, match=field):
        _config(**{field: value})


def _grouped_matmul(x, weight, group_list, group_list_type, *, split_item=2, group_type=0):
    counts = group_list.tolist()
    if group_list_type == 0:
        counts = [count - (counts[index - 1] if index else 0) for index, count in enumerate(counts)]
    offset = 0
    outputs = []
    for expert, count in enumerate(counts):
        if split_item == 2:
            outputs.append(x[offset : offset + count] @ weight[expert])
        else:
            outputs.append(x[:, offset : offset + count] @ weight[offset : offset + count])
        offset += count
    return torch.cat(outputs, dim=0) if split_item == 2 else torch.stack(outputs)


def test_activation_residual_switches_default_off():
    config = _config()
    assert config.activation_residual_fprop is False
    assert config.activation_residual_bprop is False


@pytest.mark.parametrize("rotation", [False, True])
@pytest.mark.parametrize("grad_path", ["no_grad", "autograd", "custom"])
@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75, 1.0])
def test_dense_full_activation_residual_uses_two_gemms_and_preserves_weight_error(
    monkeypatch, rotation, grad_path, alpha
):
    torch.manual_seed(51)
    config = _config(
        activation_residual_fprop=True,
        activation_residual_fprop_alpha=alpha,
        fake_quant_targets=("fprop", "dgrad", "wgrad") if grad_path == "custom" else ("fprop",),
        rotation_enable=rotation,
        rotation_targets=("fprop", "dgrad", "wgrad"),
        rotation_seed=3,
    )
    qat = MXFPQATLinear(32, 64, True, config)
    x = torch.randn(2, 5, 32, requires_grad=grad_path != "no_grad")
    linear_calls = []
    linear = F.linear

    def counted_linear(input, weight, bias=None):
        linear_calls.append((input, weight, bias))
        return linear(input, weight, bias)

    monkeypatch.setattr(F, "linear", counted_linear)
    with torch.set_grad_enabled(grad_path != "no_grad"):
        output = qat(x)
    assert len(linear_calls) == 2
    assert linear_calls[0][1] is linear_calls[1][1]
    assert linear_calls[0][2] is qat.bias
    assert linear_calls[1][2] is None
    values = rotate_blocks(x, config.rotation) if rotation else x
    weight = _target_tensor(qat.weight, config, "fprop", config.weight_format, probe_operand="weight")
    aq = _target_tensor(x, config, "fprop", config.activation_format)
    expected = linear(aq + alpha * (values - aq), weight, qat.bias)
    torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)
    assert not torch.allclose(output, linear(x, qat.weight, qat.bias), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("targets", [("fprop",), ("fprop", "dgrad", "wgrad")])
@pytest.mark.parametrize("rotation", [False, True])
@pytest.mark.parametrize("cached_backward", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_dense_fprop_switch_preserves_existing_backward_with_fixed_output_gradient(
    targets, rotation, cached_backward, alpha
):
    torch.manual_seed(53)
    baseline = MXFPQATLinear(
        32,
        64,
        True,
        _config(
            fake_quant_targets=targets,
            backward_reduction_axis_fake_quant=cached_backward,
            rotation_enable=rotation,
            rotation_targets=("fprop", "dgrad", "wgrad"),
            rotation_seed=9,
        ),
    )
    corrected = MXFPQATLinear(
        32,
        64,
        True,
        replace(baseline.qat_config, activation_residual_fprop=True, activation_residual_fprop_alpha=alpha),
    )
    corrected.load_state_dict(baseline.state_dict())
    x = torch.randn(5, 32, requires_grad=True)
    x_corrected = x.detach().clone().requires_grad_()
    grad_output = torch.randn(5, 64)
    baseline(x).backward(grad_output)
    corrected(x_corrected).backward(grad_output)
    torch.testing.assert_close(x.grad, x_corrected.grad, rtol=0, atol=0)
    torch.testing.assert_close(baseline.weight.grad, corrected.weight.grad, rtol=0, atol=0)
    torch.testing.assert_close(baseline.bias.grad, corrected.bias.grad, rtol=0, atol=0)


def test_dense_disabled_switch_is_original_one_gemm_and_parameter_identity(monkeypatch):
    linear = torch.nn.Linear(32, 64)
    qat = MXFPQATLinear.from_linear(linear, _config())
    assert qat.weight is linear.weight
    assert qat.bias is linear.bias
    x = torch.randn(3, 32)
    weight = _target_tensor(qat.weight, qat.qat_config, "fprop", "mxfp4", probe_operand="weight")
    values = _target_tensor(x, qat.qat_config, "fprop", "mxfp8")
    expected = F.linear(values, weight, qat.bias)
    calls = []
    original = F.linear

    def counted_linear(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(F, "linear", counted_linear)
    assert torch.equal(qat(x), expected)
    assert len(calls) == 1


@pytest.mark.parametrize("bias", [False, True])
def test_dense_frozen_input_and_weight_branch_keeps_bias_gradient(bias):
    torch.manual_seed(57)
    config = _config(activation_residual_fprop=True)
    qat = MXFPQATLinear(32, 64, bias, config)
    qat.weight.requires_grad_(False)
    x = torch.randn(3, 32)
    output = qat(x)
    weight_q = _target_tensor(qat.weight, config, "fprop", config.weight_format, probe_operand="weight")
    torch.testing.assert_close(output, F.linear(x, weight_q, qat.bias), rtol=2e-5, atol=2e-6)
    if bias:
        output.sum().backward()
        torch.testing.assert_close(qat.bias.grad, torch.full_like(qat.bias, 3))


def test_dense_fake_quant_disabled_bypasses_residual(monkeypatch):
    config = _config(activation_residual_fprop=True)
    qat = MXFPQATLinear(32, 64, True, config)
    qat.fake_quant_enabled = False
    x = torch.randn(3, 32)
    calls = []
    original = F.linear

    def counted_linear(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(F, "linear", counted_linear)
    output = qat(x)
    assert len(calls) == 1
    torch.testing.assert_close(output, original(x, qat.weight, qat.bias), rtol=0, atol=0)


@pytest.mark.parametrize("rotation", [False, True])
@pytest.mark.parametrize("shared_input", [False, True])
@pytest.mark.parametrize("group_list_type", [0, 1])
def test_moe_fprop_full_residual_uses_same_weight_groups_and_empty_experts(
    monkeypatch, rotation, shared_input, group_list_type
):
    torch.manual_seed(59)
    calls = []

    def counted_gmm(*args, **kwargs):
        calls.append((args, kwargs))
        return _grouped_matmul(*args, **kwargs)

    monkeypatch.setattr("verl.utils.qat.mxfp_moe._call_grouped_matmul", counted_gmm)
    config = _config(activation_residual_fprop=True, rotation_enable=rotation, rotation_seed=5)
    x = torch.randn(7, 32, requires_grad=True)
    weight = torch.randn(3, 32, 64, requires_grad=True)
    groups = torch.tensor([3, 3, 7] if group_list_type == 0 else [3, 0, 4])
    shared = prepare_mxfp_fprop_input(x, config) if shared_input else None
    weight_q = _transform_weight(weight, config, "fprop", config.weight_format, reduction_dim=1)
    output = mxfp_grouped_matmul(
        x,
        weight,
        groups,
        config,
        name="gate_proj",
        group_list_type=group_list_type,
        fprop_x=shared,
        fprop_weight=weight_q,
    )
    assert len(calls) == 2
    assert calls[0][0][1] is calls[1][0][1] is weight_q
    assert calls[0][0][2] is calls[1][0][2] is groups
    if shared_input:
        assert calls[0][0][0] is shared
    values = rotate_blocks(x, config.rotation) if rotation else x
    expected = _grouped_matmul(values, weight_q, groups, group_list_type)
    torch.testing.assert_close(output, expected, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize("targets", [("fprop",), ("fprop", "dgrad", "wgrad")])
@pytest.mark.parametrize("rotation", [False, True])
def test_moe_fprop_switch_preserves_existing_backward_with_fixed_output_gradient(monkeypatch, targets, rotation):
    monkeypatch.setattr("verl.utils.qat.mxfp_moe._call_grouped_matmul", _grouped_matmul)
    torch.manual_seed(61)
    config = _config(
        fake_quant_targets=targets,
        rotation_enable=rotation,
        rotation_targets=("fprop", "dgrad", "wgrad"),
        rotation_seed=11,
    )
    x = torch.randn(7, 32, requires_grad=True)
    weight = torch.randn(3, 32, 64, requires_grad=True)
    x_corrected = x.detach().clone().requires_grad_()
    weight_corrected = weight.detach().clone().requires_grad_()
    groups = torch.tensor([3, 0, 4])
    grad_output = torch.randn(7, 64)
    mxfp_grouped_matmul(x, weight, groups, config, name="up_proj").backward(grad_output)
    mxfp_grouped_matmul(
        x_corrected,
        weight_corrected,
        groups,
        replace(config, activation_residual_fprop=True),
        name="up_proj",
        fprop_x=prepare_mxfp_fprop_input(x_corrected, config),
    ).backward(grad_output)
    torch.testing.assert_close(x.grad, x_corrected.grad, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, weight_corrected.grad, rtol=0, atol=0)


def test_zero_activation_residual_still_runs_second_gmm(monkeypatch):
    calls = []

    def counted_gmm(*args, **kwargs):
        calls.append(args)
        return _grouped_matmul(*args, **kwargs)

    monkeypatch.setattr("verl.utils.qat.mxfp_moe._call_grouped_matmul", counted_gmm)
    x = torch.randn(4, 32)
    weight = torch.randn(1, 32, 64)
    config = _config(fake_quant_targets=(), activation_residual_fprop=True)
    result = mxfp_grouped_matmul(x, weight, torch.tensor([4]), config, name="zero-residual")
    assert len(calls) == 2
    assert torch.count_nonzero(calls[1][0]) == 0
    torch.testing.assert_close(result, x @ weight[0])


_ROTATION_CASES = [(), ("fprop",), ("dgrad",), ("wgrad",), ("fprop", "dgrad", "wgrad")]


def _rotate(tensor, config, target, *, transpose=False, pad_to_block=False):
    if config.rotation.enable and target in config.rotation.targets:
        return rotate_blocks(tensor, config.rotation, transpose=transpose, pad_to_block=pad_to_block)
    return tensor


@pytest.mark.parametrize("targets", [("fprop",), ("wgrad",), ("fprop", "dgrad", "wgrad")])
@pytest.mark.parametrize("rotation_targets", _ROTATION_CASES)
@pytest.mark.parametrize("cached_backward", [False, True])
@pytest.mark.parametrize("fprop", [False, True])
def test_dense_backward_compensates_original_a_and_retains_main_g(targets, rotation_targets, cached_backward, fprop):
    torch.manual_seed(71)
    config = _config(
        fake_quant_targets=targets,
        activation_residual_fprop=fprop,
        backward_reduction_axis_fake_quant=cached_backward,
        rotation_enable=bool(rotation_targets),
        rotation_targets=rotation_targets,
        rotation_seed=13,
    )
    baseline = MXFPQATLinear(32, 64, True, config)
    corrected = MXFPQATLinear(32, 64, True, replace(config, activation_residual_bprop=True))
    corrected.load_state_dict(baseline.state_dict())
    x = torch.randn(5, 7, 32, requires_grad=True)
    x_corrected = x.detach().clone().requires_grad_()
    grad_output = torch.randn(5, 7, 64)
    output = baseline(x)
    output_corrected = corrected(x_corrected)
    # bprop is independent of the forward switch and does not alter Y.
    torch.testing.assert_close(output_corrected, output, rtol=0, atol=0)
    output.backward(grad_output)
    output_corrected.backward(grad_output)
    torch.testing.assert_close(x_corrected.grad, x.grad, rtol=0, atol=0)
    torch.testing.assert_close(corrected.bias.grad, baseline.bias.grad, rtol=0, atol=0)

    a_reference_t = _rotate(x.detach(), config, "fprop").reshape(-1, 32).T
    wgrad_enabled = "wgrad" in targets or "wgrad" in rotation_targets
    if wgrad_enabled:
        a_reference_t = _rotate(a_reference_t, config, "wgrad", pad_to_block=True)
        g_main_t = _target_tensor(
            grad_output.reshape(-1, 64).T,
            config,
            "wgrad",
            config.activation_format,
            pad_rotation=True,
            probe_operand="grad_output",
        )
    else:
        g_main_t = grad_output.reshape(-1, 64).T
    expected = _rotate(g_main_t @ a_reference_t.T, config, "fprop", transpose=True)
    torch.testing.assert_close(corrected.weight.grad, expected, rtol=2e-5, atol=2e-5)
    assert not torch.allclose(corrected.weight.grad, baseline.weight.grad, rtol=1e-4, atol=1e-4)
    if "wgrad" in targets:
        g_reference_t = _rotate(grad_output.reshape(-1, 64).T, config, "wgrad", pad_to_block=True)
        g_compensated = _rotate(g_reference_t @ a_reference_t.T, config, "fprop", transpose=True)
        # The remaining G quantization error must not be silently compensated.
        assert not torch.allclose(corrected.weight.grad, g_compensated, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("need_input_grad,need_weight_grad", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("bprop", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_dense_backward_only_saves_original_a_when_compensated_weight_grad_is_needed(
    monkeypatch, need_input_grad, need_weight_grad, bprop, alpha
):
    torch.manual_seed(73)
    config = _config(
        fake_quant_targets=("fprop", "dgrad", "wgrad"),
        activation_residual_bprop=bprop,
        activation_residual_bprop_alpha=alpha,
    )
    qat = MXFPQATLinear(32, 64, False, config)
    qat.weight.requires_grad_(need_weight_grad)
    x = torch.randn(35, 32, requires_grad=need_input_grad)
    saved_raw_a = []

    def pack(tensor):
        # QDQ's existing autograd may save views of raw A as well. Count the
        # detached, original-shaped operand added specifically for compensation.
        if tensor.data_ptr() == x.data_ptr() and tensor.shape == x.shape and not tensor.requires_grad:
            saved_raw_a.append(tensor)
        return tensor

    matmuls = []
    original_matmul = torch.Tensor.matmul

    def counted_matmul(self, other):
        matmuls.append((self, other))
        return original_matmul(self, other)

    monkeypatch.setattr(torch.Tensor, "matmul", counted_matmul)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = qat(x)
    matmuls.clear()
    output.backward(torch.randn_like(output))
    active = bprop and alpha > 0
    assert len(saved_raw_a) == int(active and need_weight_grad)
    assert len(matmuls) == int(need_input_grad) + int(need_weight_grad) * (2 if active else 1)
    if active and need_weight_grad:
        assert matmuls[-1][0] is matmuls[-2][0]
    assert (x.grad is not None) == need_input_grad
    assert (qat.weight.grad is not None) == need_weight_grad


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("rotation", [False, True])
def test_backward_alpha_interpolates_wgrad_with_fixed_g(monkeypatch, alpha, moe, rotation):
    torch.manual_seed(83)
    monkeypatch.setattr("verl.utils.qat.mxfp_moe._call_grouped_matmul", _grouped_matmul)
    config = _config(
        fake_quant_targets=("fprop", "dgrad", "wgrad"),
        activation_residual_fprop=True,
        activation_residual_fprop_alpha=0.5,
        rotation_enable=rotation,
        rotation_targets=("fprop", "dgrad", "wgrad"),
    )
    a = torch.randn(36, 32)
    w = torch.randn(3, 32, 64) if moe else torch.randn(64, 32)
    g = torch.randn(36, 64)
    results = []
    for enabled, strength in ((False, 1.0), (True, 1.0), (True, alpha)):
        current = replace(config, activation_residual_bprop=enabled, activation_residual_bprop_alpha=strength)
        x = a.clone().requires_grad_()
        if moe:
            weight = w.clone().requires_grad_()
            output = mxfp_grouped_matmul(x, weight, torch.tensor([3, 0, 33]), current, name="alpha")
        else:
            layer = MXFPQATLinear(32, 64, False, current)
            with torch.no_grad():
                layer.weight.copy_(w)
            weight = layer.weight
            output = layer(x)
        output.backward(g)
        results.append((output.detach(), x.grad, weight.grad))
    base, full, actual = results
    torch.testing.assert_close(actual[0], base[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], base[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], base[2] + alpha * (full[2] - base[2]), rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("rotation", [False, True])
def test_moe_forward_alpha_and_zero_path(monkeypatch, alpha, rotation):
    torch.manual_seed(89)
    calls = []

    def counted(*args, **kwargs):
        calls.append(args)
        return _grouped_matmul(*args, **kwargs)

    monkeypatch.setattr("verl.utils.qat.mxfp_moe._call_grouped_matmul", counted)
    config = _config(
        fake_quant_targets=("fprop",),
        activation_residual_fprop=True,
        activation_residual_fprop_alpha=alpha,
        rotation_enable=rotation,
        rotation_targets=("fprop",),
    )
    x, w, groups = torch.randn(7, 32), torch.randn(3, 32, 64), torch.tensor([3, 0, 4])
    output = mxfp_grouped_matmul(x, w, groups, config, name="alpha")
    assert len(calls) == (1 if alpha == 0 else 2)
    aq = prepare_mxfp_fprop_input(x, config)
    ar = rotate_blocks(x, config.rotation) if rotation else x
    wq = _transform_weight(w, config, "fprop", config.weight_format, reduction_dim=1)
    expected = _grouped_matmul(aq + alpha * (ar - aq), wq, groups, 1)
    torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-5)


def test_dense_zero_alpha_uses_disabled_forward_and_backward_path(monkeypatch):
    config = _config(
        fake_quant_targets=("fprop",),
        activation_residual_fprop=True,
        activation_residual_bprop=True,
        activation_residual_fprop_alpha=0.0,
        activation_residual_bprop_alpha=0.0,
    )
    layer = MXFPQATLinear(32, 64, True, config)
    baseline = MXFPQATLinear(
        32, 64, True, replace(config, activation_residual_fprop=False, activation_residual_bprop=False)
    )
    baseline.load_state_dict(layer.state_dict())
    x = torch.randn(7, 32, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    with monkeypatch.context() as context:
        context.setattr(
            "verl.utils.qat.mxfp_linear._MXFPLinear.apply", lambda *a: pytest.fail("unexpected custom backward")
        )
        y = layer(x)
    expected = baseline(other)
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    g = torch.randn_like(y)
    y.backward(g)
    expected.backward(g)
    torch.testing.assert_close(x.grad, other.grad, rtol=0, atol=0)
    torch.testing.assert_close(layer.weight.grad, baseline.weight.grad, rtol=0, atol=0)


@pytest.mark.parametrize("targets", [("fprop",), ("wgrad",), ("fprop", "dgrad", "wgrad")])
@pytest.mark.parametrize("rotation_targets", _ROTATION_CASES)
@pytest.mark.parametrize("group_list_type", [0, 1])
@pytest.mark.parametrize("fprop", [False, True])
def test_moe_backward_compensates_original_a_and_retains_main_g(
    monkeypatch, targets, rotation_targets, group_list_type, fprop
):
    torch.manual_seed(79)
    calls = []

    def counted_gmm(*args, **kwargs):
        calls.append((args, kwargs))
        return _grouped_matmul(*args, **kwargs)

    monkeypatch.setattr("verl.utils.qat.mxfp_moe._call_grouped_matmul", counted_gmm)
    config = _config(
        fake_quant_targets=targets,
        activation_residual_fprop=fprop,
        rotation_enable=bool(rotation_targets),
        rotation_targets=rotation_targets,
        rotation_seed=17,
    )
    x = torch.randn(36, 32, requires_grad=True)
    weight = torch.randn(3, 32, 64, requires_grad=True)
    x_corrected = x.detach().clone().requires_grad_()
    weight_corrected = weight.detach().clone().requires_grad_()
    groups = torch.tensor([3, 3, 36] if group_list_type == 0 else [3, 0, 33])
    grad_output = torch.randn(36, 64)
    baseline = mxfp_grouped_matmul(x, weight, groups, config, name="up_proj", group_list_type=group_list_type)
    corrected = mxfp_grouped_matmul(
        x_corrected,
        weight_corrected,
        groups,
        replace(config, activation_residual_bprop=True),
        name="up_proj",
        group_list_type=group_list_type,
        fprop_x=prepare_mxfp_fprop_input(x_corrected, config),
    )
    torch.testing.assert_close(corrected, baseline, rtol=0, atol=0)
    baseline.backward(grad_output)
    calls.clear()
    corrected.backward(grad_output)
    torch.testing.assert_close(x_corrected.grad, x.grad, rtol=0, atol=0)
    assert len(calls) == 3  # one dgrad and two separate wgrad grouped matmuls
    assert calls[-1][0][1] is calls[-2][0][1]  # same G, without G compensation
    assert calls[-1][0][2] is calls[-2][0][2]  # same padded expert groups
    assert calls[-1][1] == calls[-2][1] == {"split_item": 3, "group_type": 2}

    wgrad_enabled = "wgrad" in targets or "wgrad" in rotation_targets
    if wgrad_enabled:
        a_reference, g_padded, wgrad_groups, _ = _pad_expert_rows(
            x.detach(), grad_output, groups, group_list_type, config.group_size
        )
        a_reference = _rotate(a_reference.T, config, "wgrad").T.contiguous()
        g_main = _transform_tokens(g_padded, config, "wgrad", config.activation_format, operand="grad_output")
    else:
        a_reference = _rotate(x.detach(), config, "fprop")
        g_main = grad_output
        wgrad_groups = groups
    expected = _grouped_matmul(a_reference.T, g_main, wgrad_groups, group_list_type, split_item=3, group_type=2)
    if not wgrad_enabled:
        expected = _rotate(expected.movedim(1, -1), config, "fprop", transpose=True).movedim(-1, 1)
    torch.testing.assert_close(weight_corrected.grad, expected, rtol=2e-5, atol=2e-5)
    if wgrad_enabled and "wgrad" not in targets:
        # This existing MoE path already starts from raw A and only rotates it.
        # Its zero residual must still execute the compensation grouped matmul.
        torch.testing.assert_close(weight_corrected.grad, weight.grad, rtol=0, atol=0)
    else:
        assert not torch.allclose(weight_corrected.grad, weight.grad, rtol=1e-4, atol=1e-4)
    if "wgrad" in targets:
        g_reference = _rotate(g_padded.T, config, "wgrad").T.contiguous()
        g_compensated = _grouped_matmul(
            a_reference.T, g_reference, wgrad_groups, group_list_type, split_item=3, group_type=2
        )
        assert not torch.allclose(weight_corrected.grad, g_compensated, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("need_input_grad,need_weight_grad", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_moe_backward_compensation_respects_frozen_gradients(monkeypatch, need_input_grad, need_weight_grad, alpha):
    calls = []

    def counted_gmm(*args, **kwargs):
        calls.append((args, kwargs))
        return _grouped_matmul(*args, **kwargs)

    monkeypatch.setattr("verl.utils.qat.mxfp_moe._call_grouped_matmul", counted_gmm)
    config = _config(
        fake_quant_targets=("fprop", "dgrad", "wgrad"),
        activation_residual_bprop=True,
        activation_residual_bprop_alpha=alpha,
    )
    x = torch.randn(7, 32, requires_grad=need_input_grad)
    weight = torch.randn(3, 32, 64, requires_grad=need_weight_grad)
    output = mxfp_grouped_matmul(x, weight, torch.tensor([3, 0, 4]), config, name="down_proj")
    calls.clear()
    output.sum().backward()
    assert len(calls) == int(need_input_grad) + (1 + int(alpha > 0)) * int(need_weight_grad)
    assert (x.grad is not None) == need_input_grad
    assert (weight.grad is not None) == need_weight_grad


def test_zero_backward_activation_residual_still_runs_second_gemm_and_gmm(monkeypatch):
    config = _config(fake_quant_targets=(), activation_residual_bprop=True)
    qat = MXFPQATLinear(32, 64, False, config)
    x = torch.randn(7, 32)
    matmuls = []
    original_matmul = torch.Tensor.matmul

    def counted_matmul(self, other):
        matmuls.append((self, other))
        return original_matmul(self, other)

    monkeypatch.setattr(torch.Tensor, "matmul", counted_matmul)
    qat(x).sum().backward()
    assert len(matmuls) == 2
    assert torch.count_nonzero(matmuls[1][1]) == 0
    gmm_calls = []

    def counted_gmm(*args, **kwargs):
        gmm_calls.append((args, kwargs))
        return _grouped_matmul(*args, **kwargs)

    monkeypatch.setattr("verl.utils.qat.mxfp_moe._call_grouped_matmul", counted_gmm)
    weight = torch.randn(3, 32, 64, requires_grad=True)
    output = mxfp_grouped_matmul(x, weight, torch.tensor([3, 0, 4]), config, name="zero-residual")
    gmm_calls.clear()
    output.sum().backward()
    assert len(gmm_calls) == 2
    assert torch.count_nonzero(gmm_calls[1][0][0]) == 0
