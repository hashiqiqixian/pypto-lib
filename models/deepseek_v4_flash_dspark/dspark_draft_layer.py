# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compose one DSpark draft layer from HC, attention, and MoE operators."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    BLOCK_SIZE,
    DSPARK_MAX_BATCH,
    DSPARK_MOE_TOKENS,
    DSPARK_QUERY_TOKENS,
    DSPARK_SWA_INDEX_WIDTH,
    FLASH as M,
    KV_ORI_BLOCK_NUM,
)
from dspark_attention import dspark_attention
from hc_post import hc_post_prefill
from hc_pre import hc_pre_with_start_dep
from moe import (
    AUX_PAD,
    IDX_PAD,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    N_ROUTES,
    RECV_MAX,
    TOPK,
    VOCAB,
    moe,
)
from rmsnorm import rms_norm


# model config
T = DSPARK_MOE_TOKENS
T_QUERY = DSPARK_QUERY_TOKENS
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
MAX_SEQ_LEN = M.max_position_embeddings
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

# tiling
PAD_D_TILE = 512


@pl.jit.inline(auto_scope=False)
def dspark_draft_layer(
    query_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    query_positions: pl.Tensor[[T_QUERY], pl.INT32],
    kv_cache: pl.Tensor[[KV_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    context_cache_ready_tid: pl.Scalar[pl.TASK_ID],
    metadata_ready_tid: pl.Scalar[pl.TASK_ID],
    query_slot_mapping: pl.Tensor[[T_QUERY], pl.INT64],
    swa_indices: pl.Tensor[[DSPARK_MAX_BATCH, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    swa_lens: pl.Tensor[[DSPARK_MAX_BATCH], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    query_token_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    output_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    active_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    start_tid: pl.Scalar[pl.TASK_ID],
):
    query_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    combine = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre_with_start_dep(
        query_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        query_mixed,
        post,
        combine,
        start_tid,
    )

    query_normed = pl.create_tensor([T_QUERY, D], dtype=pl.BF16)
    query_mixed_active: pl.Tensor[[T_QUERY, D], pl.BF16] = pl.slice(
        query_mixed,
        [T_QUERY, D],
        [0, 0],
    )
    query_ready_tid = rms_norm(query_mixed_active, attn_norm_w, query_normed)
    attention_output = pl.create_tensor([T_QUERY, D], dtype=pl.BF16)
    dspark_attention(
        query_normed,
        wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, query_positions,
        kv_cache, query_ready_tid, context_cache_ready_tid, metadata_ready_tid,
        query_slot_mapping, swa_indices, swa_lens,
        attn_sink, wo_a, wo_b, wo_b_scale,
        attention_output,
    )

    padded_attention = pl.create_tensor([T, D], dtype=pl.BF16)
    for pad_idx in pl.spmd(T * (D // PAD_D_TILE), name_hint="dspark_attention_pad"):
        pad_token = pad_idx // (D // PAD_D_TILE)
        pad_col = (pad_idx % (D // PAD_D_TILE)) * PAD_D_TILE
        output_tile = pl.full([1, PAD_D_TILE], dtype=pl.BF16, value=0.0)
        if pad_token < T_QUERY:
            output_tile = attention_output[pad_token : pad_token + 1, pad_col : pad_col + PAD_D_TILE]
        padded_attention[pad_token : pad_token + 1, pad_col : pad_col + PAD_D_TILE] = output_tile

    attention_hc = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
    hc_post_prefill(
        padded_attention,
        query_hc,
        post,
        combine,
        attention_hc,
        active_tokens,
    )

    moe(
        attention_hc,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        ffn_norm_w, gate_w, gate_bias, tid2eid, query_token_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale, routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale,
        output_hc,
        recv_meta, recv_x, recv_aux, recv_route,
        arrived, data_arrived, routed_y_buf, combine_arrived,
        layer_id, active_tokens, my_rank, moe_epoch,
    )
    return output_hc
