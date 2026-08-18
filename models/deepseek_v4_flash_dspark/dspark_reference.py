# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Small PyTorch references for the DeepSeek-V4-Flash DSpark drafter."""

from __future__ import annotations

import torch


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
