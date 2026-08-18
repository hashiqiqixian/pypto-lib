# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build layer-isolated noncausal SWA metadata for the seven-token DSpark block."""

import pypto.language as pl

from config import (
    BLOCK_SIZE,
    DSPARK_DRAFT_LAYERS,
    DSPARK_MAX_BATCH,
    DSPARK_QUERY_TOKENS,
    DSPARK_QUERY_WIDTH,
    DSPARK_SWA_INDEX_WIDTH,
    FLASH as M,
    KV_ORI_MAX_BLOCKS,
)


# Dynamic shape variables.
B_DYN = pl.dynamic("DSPARK_METADATA_B_DYN")

# model config
WIN = M.sliding_window
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS


@pl.jit.inline
def build_dspark_metadata(
    anchor_positions: pl.Tensor[[B_DYN], pl.INT32],
    block_tables: pl.Tensor[[DSPARK_DRAFT_LAYERS, B_DYN, ORI_MAX_BLOCKS], pl.INT32],
    query_slot_mapping: pl.Tensor[[DSPARK_DRAFT_LAYERS, DSPARK_QUERY_TOKENS], pl.INT64],
    swa_indices: pl.Tensor[[DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    swa_lens: pl.Tensor[[DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH], pl.INT32],
    query_positions: pl.Tensor[[DSPARK_QUERY_TOKENS], pl.INT32],
):
    batch = pl.tensor.dim(anchor_positions, 0)
    active_tokens = batch * DSPARK_QUERY_WIDTH
    with pl.spmd(1, name_hint="dspark_query_metadata") as query_metadata_tid:
        metadata_core = pl.tile.get_block_idx()
        for token in pl.range(metadata_core, DSPARK_QUERY_TOKENS):
            pl.write(query_positions, [token], pl.cast(0, pl.INT32))
            for layer in pl.range(DSPARK_DRAFT_LAYERS):
                pl.write(query_slot_mapping, [layer, token], pl.cast(-1, pl.INT64))
            if token < active_tokens:
                request = token // DSPARK_QUERY_WIDTH
                query_offset = token % DSPARK_QUERY_WIDTH
                anchor_position = pl.read(anchor_positions, [request])
                query_position = anchor_position + 1 + query_offset
                pl.write(query_positions, [token], pl.cast(query_position, pl.INT32))
                for layer in pl.range(DSPARK_DRAFT_LAYERS):
                    logical_block = query_position // BLOCK_SIZE
                    block_offset = query_position % BLOCK_SIZE
                    physical_block = pl.read(
                        block_tables, [layer, request, pl.cast(logical_block, pl.INDEX)]
                    )
                    query_slot = physical_block * BLOCK_SIZE + block_offset
                    pl.write(query_slot_mapping, [layer, token], pl.cast(query_slot, pl.INT64))

    with pl.spmd(DSPARK_MAX_BATCH, name_hint="dspark_visible_metadata") as visible_metadata_tid:
        request = pl.tile.get_block_idx()
        start_position = pl.cast(0, pl.INT32)
        visible_len = pl.cast(0, pl.INT32)
        if request < batch:
            anchor_position = pl.read(anchor_positions, [request])
            prefix_len = anchor_position + 1
            start_position = pl.cast(pl.max(prefix_len - WIN, 0), pl.INT32)
            visible_len = pl.cast(
                prefix_len + DSPARK_QUERY_WIDTH - start_position,
                pl.INT32,
            )
        for layer in pl.range(DSPARK_DRAFT_LAYERS):
            pl.write(swa_lens, [layer, request], visible_len)
            for visible_offset in pl.range(DSPARK_SWA_INDEX_WIDTH):
                visible_slot = pl.cast(-1, pl.INT32)
                if visible_offset < visible_len:
                    visible_position = start_position + visible_offset
                    logical_block = visible_position // BLOCK_SIZE
                    block_offset = visible_position % BLOCK_SIZE
                    physical_block = pl.read(
                        block_tables, [layer, request, pl.cast(logical_block, pl.INDEX)]
                    )
                    visible_slot = pl.cast(
                        physical_block * BLOCK_SIZE + block_offset,
                        pl.INT32,
                    )
                pl.write(swa_indices, [layer, request, visible_offset], visible_slot)
    return (
        query_slot_mapping,
        swa_indices,
        swa_lens,
        query_positions,
        query_metadata_tid,
        visible_metadata_tid,
    )
