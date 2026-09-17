# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""E2M1 boundary and packed-cache publisher checks against fixed format examples.

Set PYPTO_TEST_PLATFORM=a5 (or a5sim) to also execute the four native publishers.
PYPTO_TEST_DEVICE selects the device and defaults to zero.
"""

import math
import os
import struct

import pytest
import torch

from models.deepseek_v4_1_flash.quantization import (
    dequantize_mxfp4_cache,
    quantize_mxfp4_cache,
    quantize_mxfp4_weight,
)


def _rounding_cases(dtype):
    # The low E2M1 significand bit selects these codes at the seven midpoints.
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=dtype)
    below = torch.nextafter(midpoints, torch.full_like(midpoints, -float("inf")))
    above = torch.nextafter(midpoints, torch.full_like(midpoints, float("inf")))
    positive = torch.stack((below, midpoints, above), dim=1).flatten()
    codes = torch.stack(
        (torch.arange(7), torch.tensor([0, 2, 2, 4, 4, 6, 6]), torch.arange(1, 8)), dim=1
    ).flatten().to(torch.uint8)
    probes = torch.cat((positive, -positive, torch.tensor([0.0, -0.0], dtype=dtype)))
    expected = torch.cat((codes, codes | 8, torch.tensor([0, 8], dtype=torch.uint8)))
    return probes, expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("group_size,scale_format", [(16, "e4m3"), (32, "e8m0")])
def test_cache_midpoints_neighbors_and_signed_zero(dtype, group_size, scale_format):
    probes, codes = _rounding_cases(dtype)
    rows = torch.zeros(probes.numel(), group_size, dtype=dtype)
    rows[:, 0], rows[:, 1] = probes, 6.0  # Fix every group's scale to exactly one.
    payload, scales = quantize_mxfp4_cache(rows, group_size, scale_format)
    expected = torch.zeros_like(payload)
    expected[:, 0] = codes | 0x70
    assert torch.equal(payload, expected)
    scale_byte = 0x38 if scale_format == "e4m3" else 127
    assert torch.equal(scales.view(torch.uint8), torch.full_like(scales.view(torch.uint8), scale_byte))


def test_checkpoint_fp4_uses_the_same_rounding_contract():
    probes, codes = _rounding_cases(torch.float32)
    rows = torch.zeros(probes.numel(), 32)
    rows[:, 0], rows[:, 1] = probes, 6.0
    payload, scales = quantize_mxfp4_weight(rows)
    expected = torch.zeros_like(payload)
    expected[:, 0] = codes | 0x70
    assert torch.equal(payload, expected)
    assert torch.equal(scales, torch.full_like(scales, 127))


def test_e8m0_preserves_small_groups_and_has_its_own_scale_floor():
    rows = torch.zeros(4, 32)
    rows[:, 0] = torch.tensor([2.0**-12, 0.0, 6 * 2.0**-126, 2.0**-127])
    payload, scales = quantize_mxfp4_cache(rows, 32, "e8m0")
    expected = torch.zeros_like(payload)
    expected[:, 0] = torch.tensor([6, 0, 7, 1], dtype=torch.uint8)
    assert torch.equal(payload, expected)
    assert torch.equal(scales[:, 0], torch.tensor([113, 1, 1, 1], dtype=torch.uint8))
    assert torch.equal(dequantize_mxfp4_cache(payload, scales, 32, "e8m0"), rows)


def test_e8m0_scale_boundaries_match_scalar_fp32_reference():
    def fp32(value):
        return struct.unpack("<f", struct.pack("<f", value))[0]

    # Independent oracle: round the reference multiply to FP32, then use the
    # exact binary decomposition from frexp rather than tensor log2 or bit-ceil.
    samples, expected = [], []
    for power in (-120, -24, -1, 0, 1, 24, 120):
        center_bits = struct.unpack("<I", struct.pack("<f", 6.0 * 2.0**power))[0]
        for offset in (-2, -1, 0, 1, 2):
            value = struct.unpack("<f", struct.pack("<I", center_bits + offset))[0]
            required = fp32(max(value, 6.0 * 2.0**-126) * fp32(1.0 / 6.0))
            significand, exponent = math.frexp(required)
            code = exponent + 127 - int(significand == 0.5)
            samples.append(value)
            expected.append(code)
    rows = torch.zeros(len(samples), 32)
    rows[:, 0] = torch.tensor(samples)
    _, scales = quantize_mxfp4_cache(rows, 32, "e8m0")
    assert torch.equal(scales[:, 0], torch.tensor(expected, dtype=torch.uint8))


def test_e4m3_keeps_its_nonzero_minimum_scale():
    rows = torch.zeros(2, 16)
    rows[0, 0] = 2.0**-12
    payload, scales = quantize_mxfp4_cache(rows, 16, "e4m3")
    assert torch.equal(payload, torch.zeros_like(payload))
    assert torch.equal(scales.view(torch.uint8), torch.ones_like(scales.view(torch.uint8)))


def _publisher_case(width, group_size, scale_format):
    probes, codes = _rounding_cases(torch.bfloat16)
    count = probes.numel() + 4
    groups = width // group_size
    rows = torch.zeros(count, groups, group_size, dtype=torch.bfloat16)
    rows[:-4, :, 0] = probes[:, None]
    rows[:-4, :, 1] = 6.0
    rows[-4:, :, 0] = torch.tensor([2.0**-12, 0.0, 2.0**-127, 6 * 2.0**-126])[:, None]
    payload = torch.zeros(count, groups, group_size // 2, dtype=torch.uint8)
    payload[:-4, :, 0] = codes[:, None] | 0x70
    scale_byte = 0x38 if scale_format == "e4m3" else 127
    scales = torch.full((count, groups), scale_byte, dtype=torch.uint8)
    scales[-4:] = 1
    if scale_format == "e8m0":
        payload[-4:, :, 0] = torch.tensor([6, 0, 1, 7], dtype=torch.uint8)[:, None]
        scales[-4] = 113
    slots = torch.arange(count, dtype=torch.int64)
    slots[:4] = torch.tensor([127, 128, 255, -1])
    return rows.flatten(1), slots, payload.flatten(1), scales


def _publisher_entry(family, scale_format, count, width, group_size):
    import pypto.language as pl

    from models.deepseek_v4_1_flash.decode_c2a_full import publish_compressed, publish_index_key
    from models.deepseek_v4_1_flash.prefill_c1a_common import publish_compressed_cache, publish_index_cache

    c1a = publish_compressed_cache if scale_format == "e4m3" else publish_index_cache
    c2a = publish_compressed if scale_format == "e4m3" else publish_index_key
    scale_dtype = pl.FP8E4M3FN if scale_format == "e4m3" else pl.FP8E8M0
    use_c2a = family == "c2a"

    @pl.jit
    def publish(
        value: pl.Tensor[[count, width], pl.BF16],
        slots: pl.Tensor[[count], pl.INT64],
        cache: pl.InOut[pl.Tensor[[2, 128, 1, width // 2], pl.UINT8]],
        scales: pl.InOut[pl.Tensor[[2, 128, 1, width // group_size], scale_dtype]],
    ):
        if use_c2a:
            # Supply C2A's existing input-ready dependency with a real producer.
            staged = pl.create_tensor([count, width], dtype=pl.BF16)
            with pl.spmd(count, name_hint="cache_test_input") as ready:
                row = pl.tile.get_block_idx()
                pl.store(pl.load(value, [row, 0], [1, width]), [row, 0], staged)
            c2a(staged, slots, cache, scales, count, ready)
        else:
            c1a(value, slots, cache, scales, count)
        return cache, scales

    return publish


@pytest.mark.skipif(
    not os.getenv("PYPTO_TEST_PLATFORM"), reason="requires an explicitly selected NPU platform"
)
@pytest.mark.parametrize("family", ["c1a", "c2a"])
@pytest.mark.parametrize("width,group_size,scale_format", [(512, 16, "e4m3"), (128, 32, "e8m0")])
def test_native_publishers_rounding_scales_and_page_isolation(family, width, group_size, scale_format):
    from golden import TensorSpec, run

    rows, slots, expected_payload, expected_scales = _publisher_case(width, group_size, scale_format)
    scale_dtype = torch.float8_e4m3fn if scale_format == "e4m3" else torch.float8_e8m0fnu
    cache = torch.full((2, 128, 1, width // 2), 0xA5, dtype=torch.uint8)
    scale_bytes = torch.full((2, 128, 1, width // group_size), 0x38, dtype=torch.uint8)
    expected_cache = cache.clone().flatten(0, 2)
    expected_scale_bytes = scale_bytes.clone().flatten(0, 2)
    valid = slots >= 0
    expected_cache[slots[valid]] = expected_payload[valid]
    expected_scale_bytes[slots[valid]] = expected_scales[valid]

    def golden(values):
        values["cache"].copy_(expected_cache.reshape_as(cache))
        values["scales"].view(torch.uint8).copy_(expected_scale_bytes.reshape_as(scale_bytes))

    def exact_bytes(actual, expected, **_kwargs):
        return torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), "cache bytes differ"

    result = run(
        fn=_publisher_entry(family, scale_format, len(rows), width, group_size),
        specs=[
            TensorSpec("value", list(rows.shape), rows.dtype, init_value=rows),
            TensorSpec("slots", list(slots.shape), slots.dtype, init_value=slots),
            TensorSpec("cache", list(cache.shape), cache.dtype, init_value=cache),
            TensorSpec(
                "scales", list(scale_bytes.shape), scale_dtype, init_value=scale_bytes.view(scale_dtype)
            ),
        ],
        golden_fn=golden,
        compare_fn=exact_bytes,
        config={
            "platform": os.environ["PYPTO_TEST_PLATFORM"],
            "device_id": int(os.getenv("PYPTO_TEST_DEVICE", "0")),
        },
    )
    assert result.passed
