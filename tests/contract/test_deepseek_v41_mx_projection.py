# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real MX projection compilation and opt-in A5 packed-panel addressing checks.

Set PYPTO_TEST_PLATFORM=a5 and PYPTO_TEST_DEVICE to run the device cases.
"""

import importlib
import os

import pytest
import torch

import pypto


def _case(output_dtype):
    rows, inner, columns = 17, 512, 256
    groups = inner // 32
    x = torch.empty(rows, inner, dtype=torch.bfloat16)
    weight = torch.zeros(inner, columns)
    logical_codes = torch.empty(groups, columns, dtype=torch.uint8)
    expected = torch.zeros(rows, columns)
    column_ids = torch.arange(columns)
    coefficients = (1 + (column_ids % 4).float() / 4) * torch.where(column_ids % 3 == 0, -1.0, 1.0)
    for group in range(groups):
        # Every activation is exactly representable after the MXFP8 activation
        # round trip; each K32 weight block has one nonzero per output column.
        for row in range(rows):
            x[row, group * 32:(group + 1) * 32] = (1 + (row % 4) / 4) * 2.0 ** ((row // 4 + group) % 3 - 1)
        weight[group * 32 + (column_ids // 16 + group * 3) % 32, column_ids] = coefficients
        exponents = (group * 5 + column_ids * 3 + column_ids // 16) % 9 - 4
        logical_codes[group] = (exponents + 127).to(torch.uint8)
        expected += x[:, group * 32, None].float() * coefficients[None, :] * (2.0 ** exponents)[None, :]
    # Independent physical order: output block, K-group pair, output lane,
    # pair lane. Do not use the production packer to generate test input bytes.
    packed = torch.tensor([
        int(logical_codes[pair * 2 + pair_lane, block * 16 + lane])
        for block in range(columns // 16)
        for pair in range(groups // 2)
        for lane in range(16)
        for pair_lane in range(2)
    ], dtype=torch.uint8).reshape(groups, columns)
    return x, weight.to(torch.float8_e4m3fn), packed.view(torch.float8_e8m0fnu), expected.to(output_dtype)


def _entry(family, output_dtype):
    import pypto.language as pl

    module_name = "prefill_c1a_common" if family == "c1a" else "decode_swa"
    module = importlib.import_module(f"models.deepseek_v4_1_flash.{module_name}")
    dtype = pl.FP32 if output_dtype == torch.float32 else pl.BF16
    project = module.make_projection(512, 256, output_dtype=dtype)

    @pl.jit
    def entry(
        x: pl.Tensor[[17, 512], pl.BF16],
        weight: pl.Tensor[[512, 256], pl.FP8E4M3FN],
        scale: pl.Tensor[[16, 256], pl.FP8E8M0, pl.MX_B_NN],
        output: pl.Out[pl.Tensor[[17, 256], dtype]],
    ):
        output = project(x, weight, scale, output, 17)
        return output

    return entry


def _run(family, output_dtype, *, compile_only):
    from golden import TensorSpec, run

    x, weight, scale, expected = _case(output_dtype)

    def reference(values):
        values["output"].copy_(expected)

    result = run(
        fn=_entry(family, output_dtype),
        specs=[
            TensorSpec("x", list(x.shape), x.dtype, init_value=x),
            TensorSpec("weight", list(weight.shape), weight.dtype, init_value=weight),
            TensorSpec("scale", list(scale.shape), scale.dtype, init_value=scale),
            TensorSpec("output", list(expected.shape), expected.dtype),
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
@pytest.mark.parametrize("family", ["c1a", "swa"])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.bfloat16])
def test_mx_projection_compiles_real_packed_panels(family, output_dtype):
    _run(family, output_dtype, compile_only=True)


@pytest.mark.skipif(not os.getenv("PYPTO_TEST_PLATFORM"), reason="requires an explicitly selected A5 device")
@pytest.mark.parametrize("family", ["c1a", "swa"])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.bfloat16])
def test_mx_projection_npu_keeps_both_n_panels_and_k_groups(family, output_dtype):
    _run(family, output_dtype, compile_only=False)
