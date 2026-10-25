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
