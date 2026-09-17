# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Small numerical contracts complementing the native C1A decode CLI harnesses."""

import importlib
import inspect

import pytest
import torch

import pypto

if getattr(pypto, "__pypto_stub__", False):
    pytest.skip("C1A modules require the PyPTO frontend", allow_module_level=True)

from models.deepseek_v4_1_flash._golden_smoke import _attention_values


def _golden(mode, values):
    module = importlib.import_module(f"models.deepseek_v4_1_flash.decode_c1a_{mode}")
    function = getattr(module, f"golden_decode_c1a_{mode}")
    return function(**{name: values[name] for name in inspect.signature(function).parameters})


def _two_requests():
    values = _attention_values()
    for name in (
        "window_cache", "window_cache_scale", "compressed_cache", "compressed_cache_scale",
        "index_cache", "index_cache_scale",
    ):
        values[name] = values[name].repeat(4, 1, 1, 1)
    values["request_ids"] = torch.tensor([0, 1], dtype=torch.int32)
    values["index_block_table"] = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    values["compressed_lens"] = torch.tensor([128, 129], dtype=torch.int32)
    values["window_slots"] = torch.tensor([383, 128], dtype=torch.int64)
    values["compressed_slots"] = values["window_slots"].clone()
    values["compressor_wkv"] = values["compressor_wkv"].to(torch.bfloat16)
    values["window_indices"] = torch.stack((
        torch.arange(256, 384, dtype=torch.int32),
        torch.cat((torch.arange(385, 512, dtype=torch.int32), torch.tensor([128], dtype=torch.int32))),
    ))
    values["candidate_mask"] = torch.ones(2, 256, dtype=torch.uint8)
    return values


def _feedback(values, result):
    updated = dict(values)
    for name in (
        "window_cache", "window_cache_scale", "compressed_cache", "compressed_cache_scale",
        "index_cache", "index_cache_scale",
    ):
        updated[name] = getattr(result, name)
    updated["candidate_mask"] = result.candidate_mask
    updated["compressed_indices"] = result.topk_indices
    return updated


def test_decode_full_reindex_reuse_share_published_rows():
    values = _two_requests()
    full = _golden("full", values)
    assert bool(torch.isfinite(full.output).all())
    assert bool(full.output.ne(0).any())
    feedback = _feedback(values, full)
    reindex = _golden("reindex", feedback)
    reuse = _golden("reuse", feedback)
    torch.testing.assert_close(reindex.output, full.output, rtol=0, atol=0)
    torch.testing.assert_close(reuse.output, full.output, rtol=0, atol=0)
    for result in (reindex, reuse):
        assert torch.equal(result.compressed_cache, full.compressed_cache)
        assert torch.equal(
            result.compressed_cache_scale.view(torch.uint8),
            full.compressed_cache_scale.view(torch.uint8),
        )
    assert torch.equal(reindex.index_cache, full.index_cache)
    assert torch.equal(reindex.index_cache_scale, full.index_cache_scale)


def test_decode_owner_writes_only_supplied_physical_slots():
    values = _two_requests()
    full = _golden("full", values)
    untouched = torch.ones(4 * 128, dtype=torch.bool)
    untouched[values["compressed_slots"]] = False
    for name in (
        "window_cache", "window_cache_scale", "compressed_cache", "compressed_cache_scale",
        "index_cache", "index_cache_scale",
    ):
        before = values[name].view(torch.uint8).reshape(4 * 128, -1)
        after = getattr(full, name).view(torch.uint8).reshape(4 * 128, -1)
        assert torch.equal(after[untouched], before[untouched]), name


def test_decode_request_history_isolation():
    values = _two_requests()
    expected = _golden("full", values)
    other = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in values.items()}
    other["x"][1].neg_()
    for name in ("window_cache", "compressed_cache", "index_cache"):
        storage = other[name].view(torch.uint8)
        storage[1].zero_()
        storage[3].zero_()
    actual = _golden("full", other)
    torch.testing.assert_close(actual.output[0], expected.output[0], rtol=0, atol=0)
    assert torch.equal(actual.topk_indices[0], expected.topk_indices[0])
