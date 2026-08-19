# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek_v4_flash_dspark"
sys.path.insert(0, str(_MODEL_DIR))

from config import (  # noqa: E402
    BLOCK_SIZE,
    DECODE_SEQ,
)
from dspark_attention import S as ATTENTION_QUERY_WIDTH  # noqa: E402
from dspark_context_kv import DSPARK_QUERY_TOKENS as CONTEXT_QUERY_TOKENS  # noqa: E402
from dspark_drafter import (  # noqa: E402
    DSPARK_DRAFT_LAYERS,
    DSPARK_MOE_TOKENS,
    DSPARK_NOISE_TOKEN_ID,
    DSPARK_QUERY_TOKENS,
    DSPARK_QUERY_WIDTH,
    DSPARK_SUPPORTED_BATCHES,
    DSPARK_SWA_INDEX_WIDTH,
    HC_DIM,
    MIX_HC,
    N_RANKS,
    ORI_BLOCK_NUM,
    _anchor_position_set,
    _block_tables,
    build_tensor_specs as build_backbone_tensor_specs,
)
from dspark_markov import (  # noqa: E402
    DSPARK_MARKOV_RANK,
    DSPARK_MOE_TOKENS as MARKOV_MOE_TOKENS,
    DSPARK_QUERY_WIDTH as MARKOV_QUERY_WIDTH,
    build_tensor_specs as build_markov_tensor_specs,
)
from markov_head import MARKOV_RANK as HEAD_MARKOV_RANK  # noqa: E402


def rms_norm_reference(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    hidden_fp32 = hidden.float()
    inv_rms = torch.rsqrt(hidden_fp32.square().mean(dim=-1, keepdim=True) + eps)
    return hidden_fp32 * inv_rms * weight.float()


def prepare_reference(
    target_hidden: torch.Tensor,
    main_proj_weight: torch.Tensor,
    main_norm_weight: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    embedding_weight: torch.Tensor,
    *,
    noise_token_id: int,
    query_width: int,
    query_pad: int,
    max_batch: int,
    hc_mult: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project target states and build the padded anchor/noise query block."""
    _, main_width = target_hidden.shape
    hidden_size = main_proj_weight.shape[0]
    assert main_width == 3 * hidden_size
    main_hidden = target_hidden.float().matmul(main_proj_weight.float().t())
    main_hidden = rms_norm_reference(main_hidden, main_norm_weight, eps)

    batch = anchor_token_ids.shape[0]
    query_ids = torch.zeros(max_batch * query_pad, dtype=torch.int64)
    active_ids = query_ids[: batch * query_width].view(batch, query_width)
    active_ids[:, 0] = anchor_token_ids.to(torch.int64)
    active_ids[:, 1:] = noise_token_id
    query_hidden = embedding_weight.float().index_select(0, query_ids)
    query_hc = query_hidden.unsqueeze(-2).expand(max_batch * query_pad, hc_mult, hidden_size)
    return main_hidden, query_ids, query_hc.contiguous()


def noncausal_metadata_reference(
    context_positions: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    block_size: int,
    window_size: int,
    query_width: int,
    index_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-layer cache slots and block-shared noncausal SWA rows."""
    layers, batch, _ = block_tables.shape
    query_slots = torch.full((layers, batch, query_width), -1, dtype=torch.int64)
    context_slots = torch.full((layers, batch), -1, dtype=torch.int64)
    indices = torch.full((layers, batch, index_width), -1, dtype=torch.int32)
    lengths = torch.zeros((layers, batch), dtype=torch.int32)

    def physical_slot(layer: int, request: int, position: int) -> int:
        logical_block = position // block_size
        physical_block = int(block_tables[layer, request, logical_block])
        return physical_block * block_size + position % block_size

    for layer in range(layers):
        for request in range(batch):
            context_position = int(context_positions[request])
            context_slots[layer, request] = physical_slot(
                layer,
                request,
                context_position,
            )
            prefix_len = context_position + 1
            start_position = max(prefix_len - window_size, 0)
            end_position = prefix_len + query_width
            visible_positions = range(start_position, end_position)
            visible_slots = [physical_slot(layer, request, position) for position in visible_positions]
            lengths[layer, request] = len(visible_slots)
            indices[layer, request, : len(visible_slots)] = torch.tensor(visible_slots, dtype=torch.int32)
            for query_offset in range(query_width):
                query_position = prefix_len + query_offset
                query_slots[layer, request, query_offset] = physical_slot(layer, request, query_position)
    return context_slots, query_slots, indices, lengths


def noncausal_attention_reference(
    query: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Dense reference for metadata-selected single-head SWA attention."""
    output = torch.zeros_like(query, dtype=torch.float32)
    for row in range(query.shape[0]):
        visible = int(lengths[row])
        if visible == 0:
            continue
        selected = cache.index_select(0, indices[row, :visible].long()).float()
        scores = query[row].float().matmul(selected.t()) * scale
        output[row] = torch.softmax(scores, dim=-1).matmul(selected)
    return output


def markov_sample_reference(
    head_hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute base logits and the seven left-to-right Markov samples."""
    normalized = rms_norm_reference(head_hidden, norm_weight, eps)
    base_logits = normalized.matmul(lm_head_weight.float().t())
    draft_ids = torch.empty(
        head_hidden.shape[:2],
        dtype=torch.int64,
        device=head_hidden.device,
    )
    markov_biases = torch.empty_like(base_logits)
    previous = anchor_token_ids.to(torch.int64)
    for step in range(head_hidden.shape[1]):
        markov_embedding = markov_w1.float().index_select(0, previous)
        bias = markov_embedding.matmul(markov_w2.float().t())
        markov_biases[:, step] = bias
        previous = torch.argmax(base_logits[:, step] + bias, dim=-1)
        draft_ids[:, step] = previous
    return base_logits, markov_biases, draft_ids


def _metadata_inputs(
    context_position: int,
    *,
    layers: int = 3,
    batch: int = 1,
    logical_blocks: int = 16,
    physical_stride: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    context_positions = torch.full((batch,), context_position, dtype=torch.int32)
    block_tables = torch.empty(layers, batch, logical_blocks, dtype=torch.int32)
    for layer in range(layers):
        logical = torch.arange(logical_blocks, dtype=torch.int32)
        block_tables[layer] = logical + layer * physical_stride
    return context_positions, block_tables


def _build_metadata(
    context_position: int,
    *,
    block_size: int = 4,
    window_size: int = 5,
    index_width: int = 16,
    block_tables: torch.Tensor | None = None,
):
    if block_tables is None:
        context_positions, block_tables = _metadata_inputs(context_position)
    else:
        context_positions = torch.tensor([context_position], dtype=torch.int32)
    return noncausal_metadata_reference(
        context_positions,
        block_tables,
        block_size=block_size,
        window_size=window_size,
        query_width=7,
        index_width=index_width,
    )


def test_config_matches_issue_model_contract() -> None:
    assert DSPARK_DRAFT_LAYERS == 3
    assert DSPARK_QUERY_WIDTH == 7
    assert DSPARK_NOISE_TOKEN_ID == 128799
    assert DSPARK_MARKOV_RANK == 256
    assert DSPARK_QUERY_TOKENS == 16 * 7
    assert DSPARK_MOE_TOKENS == 16 * 8
    assert DSPARK_SWA_INDEX_WIDTH == 192
    assert ATTENTION_QUERY_WIDTH == DSPARK_QUERY_WIDTH == MARKOV_QUERY_WIDTH
    assert CONTEXT_QUERY_TOKENS == DSPARK_QUERY_TOKENS
    assert MARKOV_MOE_TOKENS == DSPARK_MOE_TOKENS
    assert HEAD_MARKOV_RANK == DSPARK_MARKOV_RANK


def test_prepare_projects_three_states_and_builds_anchor_noise_layout() -> None:
    torch.manual_seed(3)
    batch, hidden_size, vocab = 2, 4, 12
    target_hidden = torch.randn(batch * 2, 3 * hidden_size)
    projection = torch.randn(hidden_size, 3 * hidden_size)
    norm_weight = torch.randn(hidden_size)
    anchors = torch.tensor([2, 5])
    embedding = torch.randn(vocab, hidden_size)
    main_hidden, query_ids, query_hc = prepare_reference(
        target_hidden, projection, norm_weight, anchors, embedding,
        noise_token_id=7, query_width=7, query_pad=8, max_batch=16, hc_mult=4, eps=1e-6,
    )
    manual = target_hidden.matmul(projection.t())
    manual = manual * torch.rsqrt(manual.square().mean(-1, keepdim=True) + 1e-6)
    manual = manual * norm_weight
    torch.testing.assert_close(main_hidden, manual)
    assert query_ids[: batch * 7].view(batch, 7)[:, 0].tolist() == anchors.tolist()
    assert torch.equal(query_ids[: batch * 7].view(batch, 7)[:, 1:], torch.full((batch, 6), 7))
    assert query_ids[batch * 7 :].eq(0).all()
    assert query_ids.shape == (16 * 8,)
    assert query_hc.shape == (16 * 8, 4, hidden_size)
    assert query_hc[batch * 7 :].eq(embedding[0].view(1, 1, hidden_size)).all()


def test_noncausal_rows_include_every_query_kv() -> None:
    _, _, indices, lengths = _build_metadata(2)
    active_row = indices[0, 0]
    assert int(lengths[0, 0]) == 10
    future_slot = int(active_row[int(lengths[0, 0]) - 1])
    assert future_slot in active_row.tolist()

    cache = torch.zeros(32, 2)
    cache[future_slot] = torch.tensor([20.0, 5.0])
    query = torch.tensor([[1.0, 0.0]]).expand(7, -1).contiguous()
    before = noncausal_attention_reference(
        query, cache, active_row.expand(7, -1), lengths[0, 0].expand(7), 1.0,
    )
    cache[future_slot] = 0
    after = noncausal_attention_reference(
        query, cache, active_row.expand(7, -1), lengths[0, 0].expand(7), 1.0,
    )
    assert not torch.equal(before[0], after[0])


def test_short_context_keeps_all_available_history() -> None:
    _, _, indices, lengths = _build_metadata(1)
    assert int(lengths[0, 0]) == 9
    assert indices[0, 0, :9].tolist() == list(range(9))


def test_full_window_keeps_window_plus_full_query_block() -> None:
    _, _, indices, lengths = _build_metadata(9)
    assert int(lengths[0, 0]) == 12
    assert indices[0, 0, :12].tolist() == list(range(5, 17))


def test_page_boundary_maps_each_visible_position() -> None:
    context_slots, query_slots, indices, _ = _build_metadata(3)
    assert int(context_slots[0, 0]) == 3
    assert query_slots[0, 0, :7].tolist() == list(range(4, 11))
    assert indices[0, 0, :11].tolist() == list(range(11))


def test_ring_wrap_uses_block_table_physical_mapping() -> None:
    block_table = (
        torch.tensor(
            [[[7, 2, 7, 2, 7, 2, 7, 2, 7, 2, 7, 2, 7, 2, 7, 2]]],
            dtype=torch.int32,
        )
        .expand(3, -1, -1)
        .clone()
    )
    context_slots, query_slots, indices, lengths = _build_metadata(
        9,
        block_tables=block_table,
    )
    assert int(context_slots[0, 0]) == 7 * 4 + 1
    assert int(query_slots[0, 0, 0]) == 7 * 4 + 2
    assert int(lengths[0, 0]) == 12
    assert indices[0, 0, :12].tolist() == [9, 10, 11, 28, 29, 30, 31, 8, 9, 10, 11, 28]


def test_three_layers_use_isolated_cache_tables() -> None:
    context_slots, query_slots, indices, _ = _build_metadata(5)
    assert len(set(context_slots[:, 0].tolist())) == 3
    assert len(set(query_slots[:, 0, 0].tolist())) == 3
    assert not torch.equal(indices[0], indices[1])
    assert not torch.equal(indices[1], indices[2])


def test_markov_reference_matches_base_bias_and_final_ids() -> None:
    torch.manual_seed(4)
    batch, width, hidden_size, vocab, rank = 3, 7, 4, 9, 2
    head_hidden = torch.randn(batch, width, hidden_size)
    norm_weight = torch.randn(hidden_size)
    lm_head = torch.randn(vocab, hidden_size)
    anchors = torch.tensor([0, 1, 2])
    markov_w1 = torch.randn(vocab, rank)
    markov_w2 = torch.randn(vocab, rank)
    base_logits, biases, draft_ids = markov_sample_reference(
        head_hidden, norm_weight, lm_head, anchors, markov_w1, markov_w2, 1e-6,
    )
    previous = anchors
    for step in range(width):
        expected_bias = markov_w1.index_select(0, previous).matmul(markov_w2.t())
        torch.testing.assert_close(biases[:, step], expected_bias)
        previous = torch.argmax(base_logits[:, step] + expected_bias, dim=-1)
        assert torch.equal(draft_ids[:, step], previous)


def test_markov_causality_changes_later_ids_without_changing_base_logits() -> None:
    head_hidden = torch.zeros(1, 7, 2)
    norm_weight = torch.ones(2)
    lm_head = torch.zeros(3, 2)
    markov_w1 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    markov_w2 = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    base_a, _, ids_a = markov_sample_reference(
        head_hidden, norm_weight, lm_head, torch.tensor([0]), markov_w1, markov_w2, 1e-6,
    )
    base_b, _, ids_b = markov_sample_reference(
        head_hidden, norm_weight, lm_head, torch.tensor([1]), markov_w1, markov_w2, 1e-6,
    )
    torch.testing.assert_close(base_a, base_b)
    assert int(ids_a[0, 0]) != int(ids_b[0, 0])
    assert not torch.equal(ids_a[0, 1:], ids_b[0, 1:])


def test_supported_dynamic_batches_preserve_public_shapes() -> None:
    assert DSPARK_SUPPORTED_BATCHES == (4, 8, 12, 16)
    for batch in DSPARK_SUPPORTED_BATCHES:
        head_hidden = torch.zeros(batch, 7, 2)
        _, _, draft_ids = markov_sample_reference(
            head_hidden, torch.ones(2), torch.zeros(5, 2),
            torch.zeros(batch, dtype=torch.int64), torch.zeros(5, 2), torch.zeros(5, 2), 1e-6,
        )
        assert draft_ids.shape == (batch, 7)


def test_markov_runtime_specs_cover_minimum_hardware_batches() -> None:
    for batch in (4, 16):
        specs = {spec.name: spec for spec in build_markov_tensor_specs(batch)}
        assert specs["head_hidden"].shape == [batch, 7, 4096]
        assert specs["draft_token_ids"].shape == [batch, 7]
        assert specs["draft_token_ids"].is_output


def test_backbone_runtime_specs_cover_multirank_public_outputs() -> None:
    for batch in (4, 16):
        specs = {spec.name: spec for spec in build_backbone_tensor_specs(batch)}
        assert specs["target_hidden"].shape == [N_RANKS, batch * DECODE_SEQ, 3 * 4096]
        assert specs["context_position_ids"].shape == [N_RANKS, DSPARK_QUERY_TOKENS]
        assert specs["context_slot_mapping"].shape == [
            N_RANKS,
            DSPARK_DRAFT_LAYERS,
            DSPARK_QUERY_TOKENS,
        ]
        assert specs["head_hidden"].shape == [N_RANKS, batch, 7, 4096]
        assert specs["head_hidden"].is_output
        assert specs["initial_hidden"].shape == [N_RANKS, 128, 4, 4096]
        assert specs["initial_hidden"].is_output
        assert specs["intermediate_hidden"].shape == [
            N_RANKS,
            DSPARK_DRAFT_LAYERS,
            128,
            4,
            4096,
        ]
        assert specs["intermediate_hidden"].is_output
        assert specs["hc_attn_fn"].shape == [N_RANKS, DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM]
        assert specs["wkv"].shape == [N_RANKS, DSPARK_DRAFT_LAYERS * 4096, 512]
        cache = specs["kv_caches"]
        assert cache.shape == [
            N_RANKS,
            DSPARK_DRAFT_LAYERS,
            ORI_BLOCK_NUM,
            BLOCK_SIZE,
            1,
            512,
        ]
        assert cache.is_output
        assert cache.init_value is not None


def test_backbone_harness_covers_context_and_cache_boundaries() -> None:
    positions = _anchor_position_set(16)
    assert int(positions.min()) < 128
    assert int(positions.max()) >= 128
    assert any(int(position) % 64 == 63 for position in positions)

    tables = _block_tables(16)
    assert tables.shape[:3] == (N_RANKS, 3, 16)
    assert not torch.equal(tables[:, 0], tables[:, 1])
    assert not torch.equal(tables[:, 1], tables[:, 2])
    assert int(tables.max()) < 512
