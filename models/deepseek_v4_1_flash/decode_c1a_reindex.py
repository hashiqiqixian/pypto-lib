# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Continuous-batch decode C1A reindex attention."""

import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# A5-only; intentionally excluded from the A2/A3 device sweep.
# ci: no-sim
# ci: a5

_SCRIPT_ENTRY_POINT = "__" + "main__"
if __name__ == _SCRIPT_ENTRY_POINT:
    if not any(arg == "--tp" or arg.startswith("--tp=") for arg in sys.argv):
        sys.argv.extend(["--tp", "2"])

import torch

from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.attention_common import AttentionGoldenResult, golden_compressed_attention
from models.deepseek_v4_1_flash.config import AttentionMode
from models.deepseek_v4_1_flash.attention_tp import decode_tp_output_all_reduce
from models.deepseek_v4_1_flash.prefill_c1a_reindex import (
    build_tensor_specs,
    make_c1a_reindex_test,
    make_prefill_c1a_reindex,
    paged_indexer,
)
from models.deepseek_v4_1_flash.prefill_c1a_test_utils import apply_distributed_golden, run_decode_c1a


def golden_decode_c1a_reindex(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_cache_scale: torch.Tensor,
    request_ids: torch.Tensor,
    compressed_lens: torch.Tensor,
    index_cache: torch.Tensor,
    index_cache_scale: torch.Tensor,
    index_block_table: torch.Tensor,
    candidate_mask: torch.Tensor,
    index_wq_b: torch.Tensor,
    index_wq_b_scale: torch.Tensor,
    index_weights_proj: torch.Tensor,
) -> AttentionGoldenResult:
    return golden_compressed_attention(
        mode=AttentionMode.REINDEX,
        ratio=1,
        x=x,
        wq_a=wq_a,
        wq_a_scale=wq_a_scale,
        q_norm_weight=q_norm_weight,
        wq_b=wq_b,
        wq_b_scale=wq_b_scale,
        wkv=wkv,
        wkv_scale=wkv_scale,
        kv_norm_weight=kv_norm_weight,
        attn_sink=attn_sink,
        wo_a=wo_a,
        wo_b=wo_b,
        wo_b_scale=wo_b_scale,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        window_slots=window_slots,
        window_indices=window_indices,
        window_cache=window_cache,
        window_cache_scale=window_cache_scale,
        compressed_cache=compressed_cache,
        compressed_cache_scale=compressed_cache_scale,
        compressed_indices=None,
        compressor_wkv=None,
        compressor_wgate=None,
        compressor_norm_weight=None,
        compressor_state_rows=None,
        compressor_state=None,
        compressed_slots=None,
        position_ids=None,
        compressed_lens=compressed_lens,
        compressed_rope_cos=None,
        compressed_rope_sin=None,
        index_wk=None,
        index_norm_weight=None,
        index_wq_b=index_wq_b,
        index_wq_b_scale=index_wq_b_scale,
        index_weights_proj=index_weights_proj,
        index_cache=index_cache,
        index_cache_scale=index_cache_scale,
        index_block_table=index_block_table,
        request_ids=request_ids,
        candidate_mask=candidate_mask,
    )


# Ratio-1 prefill and decode share row-wise mathematics and packed UINT8 caches.
# The stage-specific reducer retains the bounded decode window and epoch protocol.
decode_c1a_reindex = make_prefill_c1a_reindex(
    paged_indexer,
    output_reduce=decode_tp_output_all_reduce,
    output_window_tokens=C.DECODE_MAX_TOKENS,
)
decode_c1a_reindex_test, l3_decode_c1a_reindex_test = make_c1a_reindex_test(
    decode_c1a_reindex, C.DECODE_MAX_TOKENS,
)


def golden_decode_c1a_reindex_case(tensors):
    apply_distributed_golden("reindex", golden_decode_c1a_reindex, tensors)


__all__ = ["golden_decode_c1a_reindex", "decode_c1a_reindex"]


if __name__ == _SCRIPT_ENTRY_POINT:
    run_decode_c1a(
        "reindex", l3_decode_c1a_reindex_test, build_tensor_specs, golden_decode_c1a_reindex_case,
    )
