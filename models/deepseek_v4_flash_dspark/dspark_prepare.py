# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Project target hidden rows and prepare the padded anchor-first DSpark query block."""

import pypto.language as pl

from config import (
    DSPARK_DRAFT_LAYERS,
    DSPARK_MOE_TOKENS,
    DSPARK_NOISE_TOKEN_ID,
    DSPARK_QUERY_TOKENS,
    DSPARK_QUERY_WIDTH,
    FLASH as M,
)
from dspark_proj import dspark_proj
from lookup_embedding import lookup_embedding


# Dynamic shape variables.
T_MAIN_DYN = pl.dynamic("DSPARK_PREPARE_T_MAIN_DYN")
B_DYN = pl.dynamic("DSPARK_PREPARE_B_DYN")

# model config
D = M.hidden_size
VOCAB = M.vocab_size
HC_MULT = M.hc_mult
MAIN_IN = DSPARK_DRAFT_LAYERS * D


@pl.jit.inline
def prepare_dspark_inputs(
    target_hidden: pl.Tensor[[T_MAIN_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[D, MAIN_IN], pl.BF16],
    main_norm_weight: pl.Tensor[[D], pl.BF16],
    anchor_token_ids: pl.Tensor[[B_DYN], pl.INT64],
    embedding_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    main_x: pl.Tensor[[T_MAIN_DYN, D], pl.BF16],
    query_token_ids: pl.Tensor[[DSPARK_MOE_TOKENS], pl.INT64],
    query_hidden: pl.Tensor[[DSPARK_MOE_TOKENS, D], pl.BF16],
    query_hc_flat: pl.Tensor[[DSPARK_MOE_TOKENS, HC_MULT * D], pl.FP32],
):
    dspark_proj(target_hidden, main_proj_weight, main_norm_weight, main_x)

    batch = pl.tensor.dim(anchor_token_ids, 0)
    active_tokens = batch * DSPARK_QUERY_WIDTH
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_query_ids"):
        for token in pl.range(DSPARK_MOE_TOKENS):
            token_id = pl.cast(0, pl.INT64)
            if token < active_tokens:
                request = token // DSPARK_QUERY_WIDTH
                query_offset = token % DSPARK_QUERY_WIDTH
                token_id = pl.cast(DSPARK_NOISE_TOKEN_ID, pl.INT64)
                if query_offset == 0:
                    token_id = pl.read(anchor_token_ids, [request])
            pl.write(query_token_ids, [token], token_id)

    query_hidden, query_hc_flat, lookup_tid = lookup_embedding(
        query_token_ids, embedding_weight, query_hidden, query_hc_flat
    )
    return main_x, query_token_ids, query_hidden, query_hc_flat, lookup_tid


@pl.jit
def dspark_prepare(
    target_hidden: pl.Tensor[[T_MAIN_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[D, MAIN_IN], pl.BF16],
    main_norm_weight: pl.Tensor[[D], pl.BF16],
    anchor_token_ids: pl.Tensor[[B_DYN], pl.INT64],
    embedding_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    main_x: pl.Out[pl.Tensor[[T_MAIN_DYN, D], pl.BF16]],
    query_token_ids: pl.Out[pl.Tensor[[DSPARK_MOE_TOKENS], pl.INT64]],
    query_hidden: pl.Out[pl.Tensor[[DSPARK_MOE_TOKENS, D], pl.BF16]],
    query_hc: pl.Out[pl.Tensor[[DSPARK_MOE_TOKENS, HC_MULT, D], pl.FP32]],
):
    target_hidden.bind_dynamic(0, T_MAIN_DYN)
    main_x.bind_dynamic(0, T_MAIN_DYN)
    anchor_token_ids.bind_dynamic(0, B_DYN)
    query_hc_flat = pl.reshape(query_hc, [DSPARK_MOE_TOKENS, HC_MULT * D])
    main_x, query_token_ids, query_hidden, query_hc_flat, _ = prepare_dspark_inputs(
        target_hidden,
        main_proj_weight,
        main_norm_weight,
        anchor_token_ids,
        embedding_weight,
        main_x,
        query_token_ids,
        query_hidden,
        query_hc_flat,
    )
    return main_x, query_token_ids, query_hidden, query_hc


assert DSPARK_QUERY_TOKENS < DSPARK_MOE_TOKENS
