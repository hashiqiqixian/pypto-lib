# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MoE golden checks derived from the released quantized Expert.forward contract."""

import pytest
import pypto
import importlib
from types import SimpleNamespace

torch = pytest.importorskip("torch")
F = torch.nn.functional

if getattr(pypto, "__pypto_stub__", False):
    pytest.skip("MoE modules require the PyPTO frontend", allow_module_level=True)

from models.deepseek_v4_1_flash.moe import _golden_expert, golden_moe, tp_token_owners
from models.deepseek_v4_1_flash.quantization import pack_mx_b_scale


def _reference_linear(value, weight):
    """Independent group-32 activation round trip before a quantized GEMM."""
    groups = value.float().reshape(value.shape[0], -1, 32)
    peak = groups.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = (peak / 448.0).log2().ceil().exp2()
    rounded = (groups / scale).to(torch.float8_e4m3fn).float() * scale
    return F.linear(rounded.reshape_as(value), weight.float()).to(value.dtype)


def _reference_expert(value, w1, w2, w3, route_weight=None):
    gate = _reference_linear(value, w1).float().clamp(max=10.0)
    up = _reference_linear(value, w3).float().clamp(-10.0, 10.0)
    hidden = F.silu(gate) * up
    if route_weight is not None:
        hidden = hidden * route_weight[:, None]
    return _reference_linear(hidden.to(value.dtype), w2).float()


def test_route_weight_precedes_down_projection_activation_rounding():
    generator = torch.Generator().manual_seed(240)
    value = torch.randn(3, 64, generator=generator).to(torch.bfloat16)
    matrices = [torch.randn(64, 64, generator=generator) * 0.125 for _ in range(3)]
    route_weight = torch.tensor([0.073, 0.217, 0.413])
    expected = _reference_expert(value, *matrices, route_weight)
    actual = _golden_expert(value, *matrices, route_weight)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    wrong_order = _reference_expert(value, *matrices) * route_weight[:, None]
    assert not torch.equal(actual, wrong_order)


def _inputs():
    generator = torch.Generator().manual_seed(41)
    value = torch.randn(5, 64, generator=generator).to(torch.bfloat16)
    payload = torch.randint(0, 256, (8, 64, 32), generator=generator, dtype=torch.uint8)
    scales = torch.full((8, 64, 2), 123, dtype=torch.uint8)
    shared = torch.randn(64, 64, generator=generator).mul(0.125).to(torch.float8_e4m3fn)
    shared_scale = pack_mx_b_scale(torch.full((2, 64), 127, dtype=torch.uint8))
    return dict(
        x=value, norm_weight=torch.ones(64, dtype=torch.bfloat16),
        gate_weight=torch.randn(8, 64, generator=generator), correction_bias=torch.linspace(-1, 1, 8),
        routed_w1=payload, routed_w1_scale=scales,
        routed_w2=payload, routed_w2_scale=scales,
        routed_w3=payload, routed_w3_scale=scales,
        shared_w1=shared, shared_w1_scale=shared_scale,
        shared_w2=shared, shared_w2_scale=shared_scale,
        shared_w3=shared, shared_w3_scale=shared_scale,
    )


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("active", [0, 1, 5])
def test_moe_tp_owners_and_inactive_rows(tp, active):
    inputs = _inputs()
    expected = golden_moe(**inputs, num_tokens=active)
    actual = golden_moe(**inputs, token_owners=tp_token_owners(5, tp), tp_size=tp, num_tokens=active)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[active:], inputs["x"][active:], rtol=0, atol=0)
    assert torch.isfinite(actual).all()


def test_moe_rejects_duplicate_tp_ownership():
    with pytest.raises(ValueError, match="exactly one TP rank"):
        golden_moe(**_inputs(), token_owners=torch.zeros(5, dtype=torch.int32), tp_size=2)


def test_native_fixture_scales_keep_exponent_bytes_and_fp8_runtime_dtype(monkeypatch):
    module = importlib.import_module("models.deepseek_v4_1_flash.moe")
    monkeypatch.setattr(module, "C", SimpleNamespace(
        EP_SIZE=2, TP_SIZE=2, DP_SIZE=1, N_LOCAL_EXPERTS=4, N_EXPERTS=8,
        D=64, MOE_INTER=64, TOPK=6, PREFILL_MAX_TOKENS=8192,
    ))
    specs = module.build_specs(3, 3)
    scales = [spec for spec in specs if spec.name.endswith("_scale")]
    assert len(scales) == 6
    for spec in scales:
        assert spec.dtype == torch.float8_e8m0fnu
        value = spec.init_value()
        assert value.dtype == torch.float8_e8m0fnu
        assert list(value.shape) == spec.shape
        expected_code = 121 if spec.name.startswith("routed_") else 127
        assert torch.equal(value.view(torch.uint8), torch.full(spec.shape, expected_code, dtype=torch.uint8))
