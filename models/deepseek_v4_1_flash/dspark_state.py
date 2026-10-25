# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Request-owned state for V4.1 DSpark bootstrap and recurrent decode."""

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from config import DECODE_ROWS_PER_REQUEST, DSPARK_QUERY_WIDTH, EP_SIZE, MAX_BATCH_PER_DP, TP_SIZE


BATCH = MAX_BATCH_PER_DP
ROWS = DECODE_ROWS_PER_REQUEST
STATE_VALID = 0
STATE_GENERATION = 1
STATE_ANCHOR = 2
STATE_COMMITTED = 3
STATE_DRAFT_COUNT = 4
STATE_LIMIT = 5
STATE_META_WIDTH = 6
STATE_TOKEN_WIDTH = ROWS
PREFILL_SAMPLE_ROWS = 16
SAMPLED_STRIDE = 8

if TP_SIZE != 1:
    raise ValueError("DSpark request state currently requires TP1")


def golden_bootstrap(
    slot_ids: torch.Tensor,
    generations: torch.Tensor,
    prompt_lengths: torch.Tensor,
    position_limits: torch.Tensor,
    sampled_rows: torch.Tensor,
    sampled_ids: torch.Tensor,
    draft_ids: torch.Tensor,
    state_tokens: torch.Tensor,
    state_meta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Publish one terminal prefill token and its five speculative successors."""
    tokens = state_tokens.clone()
    meta = state_meta.clone()
    for request in range(slot_ids.numel()):
        slot = int(slot_ids[request])
        row = int(sampled_rows[request])
        if slot < 0 or slot >= meta.shape[0] or row < 0 or row >= sampled_ids.shape[0]:
            continue
        generation = int(generations[request])
        prompt = int(prompt_lengths[request])
        limit = int(position_limits[request])
        if prompt <= 0 or prompt >= limit:
            continue
        if int(meta[slot, STATE_GENERATION]) != generation or int(meta[slot, STATE_VALID]) != 0:
            continue
        tokens[slot, 0] = sampled_ids[row, 0]
        meta[slot, STATE_VALID] = 1
        meta[slot, STATE_ANCHOR] = prompt
        meta[slot, STATE_COMMITTED] = prompt
        meta[slot, STATE_LIMIT] = limit
        count = DSPARK_QUERY_WIDTH if prompt + DSPARK_QUERY_WIDTH < limit else 0
        meta[slot, STATE_DRAFT_COUNT] = count
        if count:
            tokens[slot, 1:ROWS] = draft_ids[request]
    return tokens, meta


@pl.jit
def bootstrap_inputs_rank(
    slot_ids: pl.Tensor[[BATCH], pl.INT32],
    generations: pl.Tensor[[BATCH], pl.INT32],
    prompt_lengths: pl.Tensor[[BATCH], pl.INT32],
    position_limits: pl.Tensor[[BATCH], pl.INT32],
    sampled_rows: pl.Tensor[[BATCH], pl.INT32],
    sampled_ids: pl.Tensor[[PREFILL_SAMPLE_ROWS, SAMPLED_STRIDE], pl.INT32],
    state_meta: pl.Tensor[[BATCH, STATE_META_WIDTH], pl.INT32],
    next_prefill_tokens: pl.Out[pl.Tensor[[BATCH], pl.INT64]],
    first_query_positions: pl.Out[pl.Tensor[[BATCH], pl.INT32]],
):
    for _core in pl.spmd(1, name_hint="dspark_bootstrap_inputs"):
        for request in pl.range(BATCH):
            pl.write(next_prefill_tokens, [request], pl.cast(0, pl.INT64))
            pl.write(first_query_positions, [request], pl.cast(0, pl.INT32))
            slot = pl.read(slot_ids, [request])
            row = pl.read(sampled_rows, [request])
            prompt = pl.read(prompt_lengths, [request])
            limit = pl.read(position_limits, [request])
            if slot >= 0 and slot < BATCH and row >= 0 and row < PREFILL_SAMPLE_ROWS:
                valid = pl.read(state_meta, [slot, STATE_VALID])
                generation = pl.read(state_meta, [slot, STATE_GENERATION])
                expected = pl.read(generations, [request])
                if valid == 0 and generation == expected and prompt > 0 and prompt < limit:
                    token = pl.read(sampled_ids, [row, 0])
                    pl.write(next_prefill_tokens, [request], pl.cast(token, pl.INT64))
                    pl.write(first_query_positions, [request], prompt)
    return next_prefill_tokens, first_query_positions


@pl.jit.host
def l3_bootstrap_inputs(
    slot_ids: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    generations: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    prompt_lengths: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    position_limits: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    sampled_rows: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    sampled_ids: pl.Tensor[[EP_SIZE, PREFILL_SAMPLE_ROWS, SAMPLED_STRIDE], pl.INT32],
    state_meta: pl.Tensor[[EP_SIZE, BATCH, STATE_META_WIDTH], pl.INT32],
    next_prefill_tokens: pl.Out[pl.Tensor[[EP_SIZE, BATCH], pl.INT64]],
    first_query_positions: pl.Out[pl.Tensor[[EP_SIZE, BATCH], pl.INT32]],
):
    for rank in pl.range(pld.world_size()):
        bootstrap_inputs_rank(
            slot_ids[rank], generations[rank], prompt_lengths[rank], position_limits[rank], sampled_rows[rank],
            sampled_ids[rank], state_meta[rank], next_prefill_tokens[rank], first_query_positions[rank],
            device=rank,
        )


@pl.jit
def publish_bootstrap_rank(
    slot_ids: pl.Tensor[[BATCH], pl.INT32],
    generations: pl.Tensor[[BATCH], pl.INT32],
    prompt_lengths: pl.Tensor[[BATCH], pl.INT32],
    position_limits: pl.Tensor[[BATCH], pl.INT32],
    sampled_rows: pl.Tensor[[BATCH], pl.INT32],
    sampled_ids: pl.Tensor[[PREFILL_SAMPLE_ROWS, SAMPLED_STRIDE], pl.INT32],
    draft_ids: pl.Tensor[[BATCH, DSPARK_QUERY_WIDTH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[BATCH, STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[BATCH, STATE_META_WIDTH], pl.INT32]],
):
    for _core in pl.spmd(1, name_hint="dspark_bootstrap_publish"):
        for request in pl.range(BATCH):
            slot = pl.read(slot_ids, [request])
            row = pl.read(sampled_rows, [request])
            prompt = pl.read(prompt_lengths, [request])
            limit = pl.read(position_limits, [request])
            if slot >= 0 and slot < BATCH and row >= 0 and row < PREFILL_SAMPLE_ROWS:
                if prompt > 0 and prompt < limit:
                    generation = pl.read(state_meta, [slot, STATE_GENERATION])
                    expected = pl.read(generations, [request])
                    valid = pl.read(state_meta, [slot, STATE_VALID])
                    if generation == expected and valid == 0:
                        token = pl.read(sampled_ids, [row, 0])
                        pl.write(state_tokens, [slot, 0], pl.cast(token, pl.INT64))
                        pl.write(state_meta, [slot, STATE_ANCHOR], prompt)
                        pl.write(state_meta, [slot, STATE_COMMITTED], prompt)
                        pl.write(state_meta, [slot, STATE_LIMIT], limit)
                        draft_count = pl.cast(0, pl.INT32)
                        if prompt + DSPARK_QUERY_WIDTH < limit:
                            draft_count = pl.cast(DSPARK_QUERY_WIDTH, pl.INT32)
                            for offset in pl.range(DSPARK_QUERY_WIDTH):
                                draft = pl.read(draft_ids, [request, offset])
                                pl.write(state_tokens, [slot, offset + 1], pl.cast(draft, pl.INT64))
                        pl.write(state_meta, [slot, STATE_DRAFT_COUNT], draft_count)
                        pl.write(state_meta, [slot, STATE_VALID], pl.cast(1, pl.INT32))
    return state_tokens, state_meta


@pl.jit.host
def l3_publish_bootstrap(
    slot_ids: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    generations: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    prompt_lengths: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    position_limits: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    sampled_rows: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    sampled_ids: pl.Tensor[[EP_SIZE, PREFILL_SAMPLE_ROWS, SAMPLED_STRIDE], pl.INT32],
    draft_ids: pl.Tensor[[EP_SIZE, BATCH, DSPARK_QUERY_WIDTH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[EP_SIZE, BATCH, STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[EP_SIZE, BATCH, STATE_META_WIDTH], pl.INT32]],
):
    for rank in pl.range(pld.world_size()):
        publish_bootstrap_rank(
            slot_ids[rank], generations[rank], prompt_lengths[rank], position_limits[rank],
            sampled_rows[rank], sampled_ids[rank], draft_ids[rank], state_tokens[rank], state_meta[rank],
            device=rank,
        )


def golden_prepare_target(
    slot_ids: torch.Tensor,
    generations: torch.Tensor,
    state_tokens: torch.Tensor,
    state_meta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve one six-row verification slab per request."""
    ids = torch.zeros((slot_ids.numel(), ROWS), dtype=torch.int64, device=slot_ids.device)
    positions = torch.zeros_like(ids, dtype=torch.int32)
    widths = torch.zeros_like(slot_ids, dtype=torch.int32)
    for request in range(slot_ids.numel()):
        slot = int(slot_ids[request])
        if slot < 0 or slot >= state_meta.shape[0]:
            continue
        if int(state_meta[slot, STATE_VALID]) != 1 or int(state_meta[slot, STATE_GENERATION]) != int(generations[request]):
            continue
        anchor = int(state_meta[slot, STATE_ANCHOR])
        limit = int(state_meta[slot, STATE_LIMIT])
        if anchor >= limit:
            continue
        drafts = min(max(int(state_meta[slot, STATE_DRAFT_COUNT]), 0), DSPARK_QUERY_WIDTH)
        width = drafts + 1 if anchor + drafts < limit else 1
        widths[request] = width
        ids[request, :width] = state_tokens[slot, :width]
        positions[request] = torch.arange(anchor, anchor + ROWS, device=slot_ids.device).clamp_max(limit - 1)
    return ids, positions, widths


@pl.jit
def prepare_target_rank(
    slot_ids: pl.Tensor[[BATCH], pl.INT32],
    generations: pl.Tensor[[BATCH], pl.INT32],
    state_tokens: pl.Tensor[[BATCH, STATE_TOKEN_WIDTH], pl.INT64],
    state_meta: pl.Tensor[[BATCH, STATE_META_WIDTH], pl.INT32],
    input_ids: pl.Out[pl.Tensor[[BATCH, ROWS], pl.INT64]],
    position_ids: pl.Out[pl.Tensor[[BATCH, ROWS], pl.INT32]],
    active_widths: pl.Out[pl.Tensor[[BATCH], pl.INT32]],
):
    for _core in pl.spmd(1, name_hint="dspark_prepare_target"):
        for request in pl.range(BATCH):
            width = pl.cast(0, pl.INT32)
            anchor = pl.cast(0, pl.INT32)
            limit = pl.cast(1, pl.INT32)
            slot = pl.read(slot_ids, [request])
            if slot >= 0 and slot < BATCH:
                valid = pl.read(state_meta, [slot, STATE_VALID])
                generation = pl.read(state_meta, [slot, STATE_GENERATION])
                expected = pl.read(generations, [request])
                anchor = pl.read(state_meta, [slot, STATE_ANCHOR])
                limit = pl.read(state_meta, [slot, STATE_LIMIT])
                if valid == 1 and generation == expected and anchor < limit:
                    count = pl.read(state_meta, [slot, STATE_DRAFT_COUNT])
                    count = pl.max(pl.min(count, DSPARK_QUERY_WIDTH), 0)
                    width = pl.cast(1, pl.INT32)
                    if anchor + count < limit:
                        width = pl.cast(count + 1, pl.INT32)
            pl.write(active_widths, [request], width)
            for offset in pl.range(ROWS):
                token = pl.cast(0, pl.INT64)
                position = pl.cast(0, pl.INT32)
                if width > 0:
                    position = pl.cast(pl.min(anchor + offset, limit - 1), pl.INT32)
                    if offset < width:
                        token = pl.read(state_tokens, [slot, offset])
                pl.write(input_ids, [request, offset], token)
                pl.write(position_ids, [request, offset], position)
    return input_ids, position_ids, active_widths


@pl.jit.host
def l3_prepare_target(
    slot_ids: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    generations: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    state_tokens: pl.Tensor[[EP_SIZE, BATCH, STATE_TOKEN_WIDTH], pl.INT64],
    state_meta: pl.Tensor[[EP_SIZE, BATCH, STATE_META_WIDTH], pl.INT32],
    input_ids: pl.Out[pl.Tensor[[EP_SIZE, BATCH, ROWS], pl.INT64]],
    position_ids: pl.Out[pl.Tensor[[EP_SIZE, BATCH, ROWS], pl.INT32]],
    active_widths: pl.Out[pl.Tensor[[EP_SIZE, BATCH], pl.INT32]],
):
    for rank in pl.range(pld.world_size()):
        prepare_target_rank(
            slot_ids[rank], generations[rank], state_tokens[rank], state_meta[rank],
            input_ids[rank], position_ids[rank], active_widths[rank], device=rank,
        )


def golden_accept_target(
    slot_ids: torch.Tensor,
    generations: torch.Tensor,
    active_widths: torch.Tensor,
    sampled_ids: torch.Tensor,
    state_tokens: torch.Tensor,
    state_meta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accept the longest matching draft prefix and retain the next target token."""
    tokens = state_tokens.clone()
    meta = state_meta.clone()
    accepted_ids = torch.full((slot_ids.numel(), ROWS), -1, dtype=torch.int32, device=slot_ids.device)
    accepted_counts = torch.zeros_like(slot_ids, dtype=torch.int32)
    for request in range(slot_ids.numel()):
        slot = int(slot_ids[request])
        width = int(active_widths[request])
        if slot < 0 or slot >= meta.shape[0] or width < 1 or width > ROWS:
            continue
        if int(meta[slot, STATE_VALID]) != 1 or int(meta[slot, STATE_GENERATION]) != int(generations[request]):
            continue
        count = min(max(int(meta[slot, STATE_DRAFT_COUNT]), 0), width - 1)
        matched = 0
        for offset in range(count):
            if int(tokens[slot, offset + 1]) != int(sampled_ids[request * ROWS + offset, 0]):
                break
            matched += 1
        accepted = matched + 1
        accepted_counts[request] = accepted
        accepted_ids[request, :accepted] = sampled_ids[request * ROWS : request * ROWS + accepted, 0]
        tokens[slot, 0] = sampled_ids[request * ROWS + matched, 0]
        meta[slot, STATE_ANCHOR] += accepted
        meta[slot, STATE_COMMITTED] += accepted
        meta[slot, STATE_DRAFT_COUNT] = 0
    return tokens, meta, accepted_ids, accepted_counts


@pl.jit
def accept_target_rank(
    slot_ids: pl.Tensor[[BATCH], pl.INT32],
    generations: pl.Tensor[[BATCH], pl.INT32],
    active_widths: pl.Tensor[[BATCH], pl.INT32],
    sampled_ids: pl.Tensor[[BATCH * ROWS, SAMPLED_STRIDE], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[BATCH, STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[BATCH, STATE_META_WIDTH], pl.INT32]],
    accepted_ids: pl.Out[pl.Tensor[[BATCH, ROWS], pl.INT32]],
    accepted_counts: pl.Out[pl.Tensor[[BATCH], pl.INT32]],
    context_valid: pl.Out[pl.Tensor[[BATCH, ROWS], pl.INT32]],
    context_positions: pl.Out[pl.Tensor[[BATCH, ROWS], pl.INT32]],
    num_sampled: pl.Out[pl.Tensor[[BATCH], pl.INT32]],
    last_sampled: pl.Out[pl.Tensor[[BATCH], pl.INT64]],
    first_query_positions: pl.Out[pl.Tensor[[BATCH], pl.INT32]],
):
    for _core in pl.spmd(1, name_hint="dspark_accept_target"):
        for request in pl.range(BATCH):
            accepted = pl.cast(0, pl.INT32)
            anchor = pl.cast(0, pl.INT32)
            can_draft = pl.cast(0, pl.INT32)
            next_token = pl.cast(0, pl.INT64)
            slot = pl.read(slot_ids, [request])
            width = pl.read(active_widths, [request])
            if slot >= 0 and slot < BATCH and width > 0 and width <= ROWS:
                valid = pl.read(state_meta, [slot, STATE_VALID])
                generation = pl.read(state_meta, [slot, STATE_GENERATION])
                expected = pl.read(generations, [request])
                if valid == 1 and generation == expected:
                    anchor = pl.read(state_meta, [slot, STATE_ANCHOR])
                    limit = pl.read(state_meta, [slot, STATE_LIMIT])
                    count = pl.read(state_meta, [slot, STATE_DRAFT_COUNT])
                    count = pl.max(pl.min(count, width - 1), 0)
                    matched = pl.cast(0, pl.INT32)
                    still_matching = pl.cast(1, pl.INT32)
                    for offset in pl.range(DSPARK_QUERY_WIDTH):
                        if offset < count and still_matching == 1:
                            draft = pl.read(state_tokens, [slot, offset + 1])
                            predicted = pl.read(sampled_ids, [request * ROWS + offset, 0])
                            if draft == pl.cast(predicted, pl.INT64):
                                matched = pl.cast(offset + 1, pl.INT32)
                            else:
                                still_matching = pl.cast(0, pl.INT32)
                    accepted = pl.cast(matched + 1, pl.INT32)
                    next_token = pl.cast(pl.read(sampled_ids, [request * ROWS + matched, 0]), pl.INT64)
                    pl.write(state_tokens, [slot, 0], next_token)
                    pl.write(state_meta, [slot, STATE_ANCHOR], pl.cast(anchor + accepted, pl.INT32))
                    committed = pl.read(state_meta, [slot, STATE_COMMITTED])
                    pl.write(state_meta, [slot, STATE_COMMITTED], pl.cast(committed + accepted, pl.INT32))
                    pl.write(state_meta, [slot, STATE_DRAFT_COUNT], pl.cast(0, pl.INT32))
                    if anchor + accepted + DSPARK_QUERY_WIDTH < limit:
                        can_draft = pl.cast(1, pl.INT32)
            pl.write(accepted_counts, [request], accepted)
            pl.write(num_sampled, [request], pl.cast(accepted * can_draft, pl.INT32))
            pl.write(last_sampled, [request], next_token)
            pl.write(first_query_positions, [request], pl.cast((anchor + accepted) * can_draft, pl.INT32))
            for offset in pl.range(ROWS):
                token = pl.cast(-1, pl.INT32)
                valid = pl.cast(0, pl.INT32)
                position = pl.cast(0, pl.INT32)
                if offset < accepted:
                    token = pl.read(sampled_ids, [request * ROWS + offset, 0])
                    position = pl.cast(anchor + offset, pl.INT32)
                    valid = can_draft
                pl.write(accepted_ids, [request, offset], token)
                pl.write(context_valid, [request, offset], valid)
                pl.write(context_positions, [request, offset], position)
    return state_tokens, state_meta, accepted_ids, accepted_counts


@pl.jit.host
def l3_accept_target(
    slot_ids: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    generations: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    active_widths: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    sampled_ids: pl.Tensor[[EP_SIZE, BATCH * ROWS, SAMPLED_STRIDE], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[EP_SIZE, BATCH, STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[EP_SIZE, BATCH, STATE_META_WIDTH], pl.INT32]],
    accepted_ids: pl.Out[pl.Tensor[[EP_SIZE, BATCH, ROWS], pl.INT32]],
    accepted_counts: pl.Out[pl.Tensor[[EP_SIZE, BATCH], pl.INT32]],
    context_valid: pl.Out[pl.Tensor[[EP_SIZE, BATCH, ROWS], pl.INT32]],
    context_positions: pl.Out[pl.Tensor[[EP_SIZE, BATCH, ROWS], pl.INT32]],
    num_sampled: pl.Out[pl.Tensor[[EP_SIZE, BATCH], pl.INT32]],
    last_sampled: pl.Out[pl.Tensor[[EP_SIZE, BATCH], pl.INT64]],
    first_query_positions: pl.Out[pl.Tensor[[EP_SIZE, BATCH], pl.INT32]],
):
    for rank in pl.range(pld.world_size()):
        accept_target_rank(
            slot_ids[rank], generations[rank], active_widths[rank], sampled_ids[rank],
            state_tokens[rank], state_meta[rank], accepted_ids[rank], accepted_counts[rank],
            context_valid[rank], context_positions[rank], num_sampled[rank],
            last_sampled[rank], first_query_positions[rank], device=rank,
        )
