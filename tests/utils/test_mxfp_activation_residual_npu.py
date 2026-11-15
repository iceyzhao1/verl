"""Native Ascend smoke checks for A-only Wgrad compensation.

Run in the installed verl/NPU environment. Unlike the CPU suite, these tests do
not replace grouped matmul. They do not cover FSDP, TP, graph capture, or RL.
"""

from dataclasses import replace

import pytest
import torch

from verl.utils.qat.mxfp_linear import MXFPQATConfig, MXFPQATLinear, _target_tensor
from verl.utils.qat.mxfp_moe import mxfp_grouped_matmul

pytest.importorskip("torch_npu", reason="Native Ascend test requires torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is unavailable")


def _config(fprop, alpha):
    return MXFPQATConfig(
        mode="mxfp_w4a8",
        quant_backend="torch",
        fake_quant_targets=("fprop", "dgrad", "wgrad"),
        activation_residual_fprop=fprop,
        activation_residual_bprop=True,
        activation_residual_fprop_alpha=alpha,
        activation_residual_bprop_alpha=alpha,
        rotation_enable=False,
        probe_quant_error=False,
    )


def _check_bf16_weight_grad(actual, reference):
    # Smoke-test tolerance for two BF16 products and their addition. This is not
    # a model-quality threshold; FP32 CPU tests check the algebra more tightly.
    actual = actual.detach().float().cpu()
    relative_l2 = (actual - reference).norm() / reference.norm().clamp_min(1e-12)
    assert relative_l2.item() < 0.02


@pytest.mark.parametrize("fprop", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_dense_original_a_wgrad_on_native_npu(fprop, alpha):
    torch.manual_seed(101)
    config = _config(fprop, alpha)
    baseline = MXFPQATLinear(
        32, 64, False, replace(config, activation_residual_bprop=False), device="npu", dtype=torch.bfloat16
    )
    corrected = MXFPQATLinear(32, 64, False, config, device="npu", dtype=torch.bfloat16)
    corrected.load_state_dict(baseline.state_dict())
    a = torch.randn(64, 32, device="npu", dtype=torch.bfloat16, requires_grad=True)
    a_corrected = a.detach().clone().requires_grad_()
    g = torch.randn(64, 64, device="npu", dtype=torch.bfloat16)
    baseline_output = baseline(a)
    corrected_output = corrected(a_corrected)
    torch.testing.assert_close(corrected_output, baseline_output, rtol=0, atol=0)
    baseline_output.backward(g)
    corrected_output.backward(g)
    torch.testing.assert_close(a_corrected.grad, a.grad, rtol=0, atol=0)
    g_main_t = _target_tensor(g.T, config, "wgrad", config.activation_format, probe_operand="grad_output")
    aq = _target_tensor(a.detach(), config, "fprop", config.activation_format)
    aq_b = _target_tensor(aq.T, config, "wgrad", config.activation_format).T
    blended_a = (1 - alpha) * aq_b.float().cpu() + alpha * a.detach().float().cpu()
    reference = g_main_t.float().cpu() @ blended_a
    _check_bf16_weight_grad(corrected.weight.grad, reference)


@pytest.mark.parametrize("fprop", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_moe_original_a_wgrad_on_native_npu_with_empty_expert(fprop, alpha):
    torch.manual_seed(103)
    config = _config(fprop, alpha)
    a = torch.randn(64, 32, device="npu", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(3, 32, 64, device="npu", dtype=torch.bfloat16, requires_grad=True)
    a_corrected = a.detach().clone().requires_grad_()
    weight_corrected = weight.detach().clone().requires_grad_()
    g = torch.randn(64, 64, device="npu", dtype=torch.bfloat16)
    groups = torch.tensor([32, 0, 32], device="npu", dtype=torch.int64)
    baseline_output = mxfp_grouped_matmul(
        a, weight, groups, replace(config, activation_residual_bprop=False), name="native-smoke"
    )
    corrected_output = mxfp_grouped_matmul(a_corrected, weight_corrected, groups, config, name="native-smoke")
    torch.testing.assert_close(corrected_output, baseline_output, rtol=0, atol=0)
    baseline_output.backward(g)
    corrected_output.backward(g)
    torch.testing.assert_close(a_corrected.grad, a.grad, rtol=0, atol=0)
    # Each nonempty expert has exactly one 32-token quantization block, so a
    # direct per-expert reference needs no copy of production padding logic.
    g_main = _target_tensor(g.T, config, "wgrad", config.activation_format, probe_operand="grad_output").T
    aq_b = _target_tensor(a.detach().T, config, "wgrad", config.activation_format).T
    a_cpu = (1 - alpha) * aq_b.float().cpu() + alpha * a.detach().float().cpu()
    g_cpu = g_main.float().cpu()
    reference = torch.stack([a_cpu[:32].T @ g_cpu[:32], torch.zeros(32, 64), a_cpu[32:].T @ g_cpu[32:]])
    _check_bf16_weight_grad(weight_corrected.grad, reference)
