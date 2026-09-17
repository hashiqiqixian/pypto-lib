# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the real routed projection against independent packed E2M1 values."""

import os

import pypto
import pytest

torch = pytest.importorskip("torch")


def _case():
    # Distinct low/high patterns expose nibble order, row pitch, and both N panels.
    row = torch.arange(64, dtype=torch.int32)[:, None]
    column = torch.arange(256, dtype=torch.int32)[None, :]
    low = (row + column) % 16
    high = (3 * row + column + 7) % 16
    packed = (low | (high << 4)).to(torch.uint8)
    scale_codes = ((row + torch.arange(16)[None, :]) % 7 + 121).to(torch.uint8)
    scale = scale_codes.view(torch.float8_e8m0fnu)

    # Both nibbles in every group, including the second K256 iteration, are read.
    x = torch.zeros(32, 512, dtype=torch.bfloat16)
    tokens = torch.arange(32)
    positions = (tokens // 2) * 32 + tokens % 2
    x[tokens, positions] = 1.0
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    )
    unpacked = torch.empty(64, 512)
    unpacked[:, 0::2] = values[low.long()]
    unpacked[:, 1::2] = values[high.long()]
    unpacked *= torch.exp2(scale_codes.float() - 127).repeat_interleave(32, dim=1)
    # One-hot activations are exact under group-32 FP8 quantization.
    expected = torch.nn.functional.linear(x.float(), unpacked).to(torch.bfloat16)
    return x, packed, scale, expected


def _entry(active):
    import pypto.language as pl

    from models.deepseek_v4_1_flash.moe import make_routed_projection

    project = make_routed_projection(512, 64)

    @pl.jit
    def entry(
        x: pl.Tensor[[32, 512], pl.BF16],
        weight: pl.Tensor[[64, 256], pl.UINT8],
        scale: pl.Tensor[[64, 16], pl.FP8E8M0],
        output: pl.InOut[pl.Tensor[[32, 64], pl.BF16]],
    ):
        output = project(x, weight, scale, output, active)
        return output

    return entry


def _run(*, compile_only, active):
    from golden import TensorSpec, run

    x, packed, scale, expected = _case()
    expected[active:] = 13.0

    def reference(values):
        values["output"].copy_(expected)

    result = run(
        fn=_entry(active),
        specs=[
            TensorSpec("x", list(x.shape), x.dtype, init_value=x),
            TensorSpec("weight", list(packed.shape), packed.dtype, init_value=packed),
            TensorSpec("scale", list(scale.shape), scale.dtype, init_value=scale),
            TensorSpec("output", list(expected.shape), expected.dtype, init_value=13.0),
        ],
        golden_fn=reference,
        compile_only=compile_only,
        rtol=0,
        atol=0,
        config={
            "platform": "a5" if compile_only else os.environ["PYPTO_TEST_PLATFORM"],
            "device_id": int(os.getenv("PYPTO_TEST_DEVICE", "0")),
        },
    )
    assert result.passed


@pytest.mark.skipif(getattr(pypto, "__pypto_stub__", False), reason="requires the real PyPTO frontend")
def test_routed_projection_compiles_packed_fp4():
    _run(compile_only=True, active=17)


@pytest.mark.skipif(not os.getenv("PYPTO_TEST_PLATFORM"), reason="requires an explicitly selected A5 device")
@pytest.mark.parametrize("active", [17, 32])
def test_routed_projection_npu_keeps_nibbles_groups_and_tail(active):
    _run(compile_only=False, active=active)
