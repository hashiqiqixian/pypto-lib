# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Three-layer DeepSeek-V4-Flash DSpark draft backbone."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    BLOCK_SIZE,
    DECODE_SEQ,
    DSPARK_DRAFT_LAYERS,
    DSPARK_MAX_BATCH,
    DSPARK_MOE_TOKENS,
    DSPARK_QUERY_TOKENS,
    DSPARK_QUERY_WIDTH,
    DSPARK_SWA_INDEX_WIDTH,
    FLASH as M,
    KV_ORI_BLOCK_NUM,
    KV_ORI_MAX_BLOCKS,
)
from dspark_context_kv import dspark_context_kv_query
from dspark_draft_layer import dspark_draft_layer
from dspark_metadata import build_dspark_metadata
from dspark_prepare import prepare_dspark_inputs
from hc_head import hc_head
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
    clear_moe_signals,
)
from qkv_proj_rope import T_DYN as QKV_Q_T_DYN


T_MAIN_DYN = pl.dynamic("DSPARK_BACKBONE_T_MAIN_DYN")
B_DYN = pl.dynamic("DSPARK_BACKBONE_B_DYN")

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
ORI_BLOCK_NUM = KV_ORI_BLOCK_NUM
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
MAIN_IN = DSPARK_DRAFT_LAYERS * D


@pl.jit.inline
def rebase_moe_signals(
    completion_anchor: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    completed_epoch: pl.Scalar[pl.INT32],
):
    """Clear stale credits while preserving the monotonic epoch baseline."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_moe_rebase") as rebase_tid:
        _completion_anchor = pl.read(completion_anchor, [0, 0, 0])
        data_epoch = pl.cast(completed_epoch * N_LOCAL, pl.INT32)
        for source in pl.range(N_RANKS):
            pl.write(arrived, [source, 0], completed_epoch)
            pl.write(data_arrived, [source, 0], data_epoch)
            pl.write(combine_arrived, [source, 0], data_epoch)
    return rebase_tid


@pl.jit.inline
def dspark_moe_barrier(
    signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    expected: pl.Scalar[pl.INT32],
    rebase_tid: pl.Scalar[pl.TASK_ID],
):
    """Keep a faster rank from publishing the next epoch before all rebases."""
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dspark_moe_epoch_barrier",
        deps=[rebase_tid],
    ) as barrier_tid:
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(
                    target=signal,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
        for source in pl.range(N_RANKS):
            if source != my_rank:
                pld.system.wait(
                    signal=signal,
                    offsets=[source, 0],
                    expected=expected,
                    cmp=pld.WaitCmp.Ge,
                )
    return barrier_tid


@pl.jit.inline
def clear_dspark_moe_barrier(
    completion_anchor: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    """Reset the barrier window after the final draft layer completes."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_moe_barrier_clear"):
        _completion_anchor = pl.read(completion_anchor, [0, 0, 0])
        zero = pl.cast(0, pl.INT32)
        for source in pl.range(N_RANKS):
            pl.write(signal, [source, 0], zero)


@pl.jit
def draft_backbone(
    target_hidden: pl.Tensor[[T_MAIN_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[D, MAIN_IN], pl.BF16],
    main_norm_weight: pl.Tensor[[D], pl.BF16],
    anchor_token_ids: pl.Tensor[[B_DYN], pl.INT64],
    embedding_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    context_position_ids: pl.Tensor[[QKV_Q_T_DYN], pl.INT32],
    context_slot_mapping: pl.Tensor[[DSPARK_DRAFT_LAYERS, QKV_Q_T_DYN], pl.INT64],
    anchor_positions: pl.Tensor[[B_DYN], pl.INT32],
    block_tables: pl.Tensor[
        [DSPARK_DRAFT_LAYERS, B_DYN, ORI_MAX_BLOCKS],
        pl.INT32,
    ],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    hc_attn_fn_0: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale_0: pl.Tensor[[3], pl.FP32],
    hc_attn_base_0: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w_0: pl.Tensor[[D], pl.BF16],
    wq_a_0: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b_0: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale_0: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv_0: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq_0: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv_0: pl.Tensor[[HEAD_DIM], pl.BF16],
    kv_caches: pl.InOut[
        pl.Tensor[[DSPARK_DRAFT_LAYERS * ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    attn_sink_0: pl.Tensor[[H], pl.FP32],
    wo_a_0: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_0: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale_0: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn_0: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale_0: pl.Tensor[[3], pl.FP32],
    hc_ffn_base_0: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_w_0: pl.Tensor[[D], pl.BF16],
    gate_w_0: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias_0: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid_0: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    routed_w1_0: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale_0: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3_0: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale_0: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2_0: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale_0: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1_0: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale_0: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3_0: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale_0: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2_0: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale_0: pl.Tensor[[D], pl.FP32],
    hc_attn_fn_1: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale_1: pl.Tensor[[3], pl.FP32],
    hc_attn_base_1: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w_1: pl.Tensor[[D], pl.BF16],
    wq_a_1: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b_1: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale_1: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv_1: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq_1: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv_1: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink_1: pl.Tensor[[H], pl.FP32],
    wo_a_1: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_1: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale_1: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn_1: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale_1: pl.Tensor[[3], pl.FP32],
    hc_ffn_base_1: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_w_1: pl.Tensor[[D], pl.BF16],
    gate_w_1: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias_1: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid_1: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    routed_w1_1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale_1: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3_1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale_1: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2_1: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale_1: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1_1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale_1: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3_1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale_1: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2_1: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale_1: pl.Tensor[[D], pl.FP32],
    hc_attn_fn_2: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale_2: pl.Tensor[[3], pl.FP32],
    hc_attn_base_2: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w_2: pl.Tensor[[D], pl.BF16],
    wq_a_2: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b_2: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale_2: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv_2: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq_2: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv_2: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink_2: pl.Tensor[[H], pl.FP32],
    wo_a_2: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_2: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale_2: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn_2: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale_2: pl.Tensor[[3], pl.FP32],
    hc_ffn_base_2: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_w_2: pl.Tensor[[D], pl.BF16],
    gate_w_2: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias_2: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid_2: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    routed_w1_2: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale_2: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3_2: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale_2: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2_2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale_2: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1_2: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale_2: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3_2: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale_2: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2_2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale_2: pl.Tensor[[D], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    initial_hidden: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    intermediate_hidden: pl.Out[
        pl.Tensor[[DSPARK_DRAFT_LAYERS, T, HC_MULT, D], pl.FP32]
    ],
    head_hidden: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    moe_barrier_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    target_hidden.bind_dynamic(0, T_MAIN_DYN)
    context_position_ids.bind_dynamic(0, QKV_Q_T_DYN)
    context_slot_mapping.bind_dynamic(1, QKV_Q_T_DYN)
    anchor_token_ids.bind_dynamic(0, B_DYN)
    anchor_positions.bind_dynamic(0, B_DYN)
    block_tables.bind_dynamic(1, B_DYN)
    head_hidden.bind_dynamic(0, B_DYN)
    batch = pl.tensor.dim(anchor_token_ids, 0)
    target_tokens = pl.tensor.dim(target_hidden, 0)
    active_tokens = batch * DSPARK_QUERY_WIDTH

    kv_cache_0 = pl.slice(
        kv_caches,
        [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
        [0, 0, 0, 0],
    )
    kv_cache_1 = pl.slice(
        kv_caches,
        [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
        [ORI_BLOCK_NUM, 0, 0, 0],
    )
    kv_cache_2 = pl.slice(
        kv_caches,
        [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
        [2 * ORI_BLOCK_NUM, 0, 0, 0],
    )

    main_x = pl.create_tensor([target_tokens, D], dtype=pl.BF16)
    query_token_ids = pl.create_tensor([T], dtype=pl.INT64)
    hidden_0_flat = pl.reshape(initial_hidden, [T, HC_MULT * D])
    main_x, query_token_ids, hidden_0_flat, lookup_tid = prepare_dspark_inputs(
        target_hidden,
        main_proj_weight,
        main_norm_weight,
        anchor_token_ids,
        embedding_weight,
        main_x,
        query_token_ids,
        hidden_0_flat,
    )

    context_main_x = pl.create_tensor([T_QUERY, D], dtype=pl.BF16)
    context_positions = pl.create_tensor([T_QUERY], dtype=pl.INT32)
    context_slots_0 = pl.create_tensor([T_QUERY], dtype=pl.INT64)
    context_slots_1 = pl.create_tensor([T_QUERY], dtype=pl.INT64)
    context_slots_2 = pl.create_tensor([T_QUERY], dtype=pl.INT64)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_context_pad"):
        for token in pl.range(T_QUERY):
            position = pl.cast(0, pl.INT32)
            if token < batch:
                position = pl.read(context_position_ids, [token])
            pl.write(context_positions, [token], position)
            slot_0 = pl.cast(-1, pl.INT64)
            slot_1 = pl.cast(-1, pl.INT64)
            slot_2 = pl.cast(-1, pl.INT64)
            if token < batch:
                slot_0 = pl.read(context_slot_mapping, [0, token])
                slot_1 = pl.read(context_slot_mapping, [1, token])
                slot_2 = pl.read(context_slot_mapping, [2, token])
            pl.write(context_slots_0, [token], slot_0)
            pl.write(context_slots_1, [token], slot_1)
            pl.write(context_slots_2, [token], slot_2)
        for token_d in pl.range(T_QUERY * (D // 512)):
            token = token_d // (D // 512)
            d0 = (token_d % (D // 512)) * 512
            value = pl.full([1, 512], dtype=pl.BF16, value=0.0)
            if token < batch:
                context_row = (token + 1) * DECODE_SEQ - 1
                value = main_x[context_row : context_row + 1, d0 : d0 + 512]
            context_main_x[token : token + 1, d0 : d0 + 512] = value

    context_kv_0_tid = dspark_context_kv_query(
        context_main_x, wkv_0, gamma_ckv_0, freqs_cos, freqs_sin,
        context_positions, context_slots_0, kv_cache_0, lookup_tid,
    )
    context_kv_1_tid = dspark_context_kv_query(
        context_main_x, wkv_1, gamma_ckv_1, freqs_cos, freqs_sin,
        context_positions, context_slots_1, kv_cache_1, context_kv_0_tid,
    )
    context_kv_2_tid = dspark_context_kv_query(
        context_main_x, wkv_2, gamma_ckv_2, freqs_cos, freqs_sin,
        context_positions, context_slots_2, kv_cache_2, context_kv_1_tid,
    )

    query_slot_mapping = pl.create_tensor([DSPARK_DRAFT_LAYERS, T_QUERY], dtype=pl.INT64)
    swa_indices = pl.create_tensor(
        [DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH, DSPARK_SWA_INDEX_WIDTH],
        dtype=pl.INT32,
    )
    swa_lens = pl.create_tensor([DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH], dtype=pl.INT32)
    query_positions = pl.create_tensor([T_QUERY], dtype=pl.INT32)
    (
        query_slot_mapping,
        swa_indices,
        swa_lens,
        query_positions,
        query_metadata_tid,
        visible_metadata_tid,
    ) = build_dspark_metadata(
        anchor_positions,
        block_tables,
        query_slot_mapping,
        swa_indices,
        swa_lens,
        query_positions,
    )
    query_slot_mapping_0 = query_slot_mapping[0]
    query_slot_mapping_1 = query_slot_mapping[1]
    query_slot_mapping_2 = query_slot_mapping[2]
    swa_indices_0 = swa_indices[0]
    swa_indices_1 = swa_indices[1]
    swa_indices_2 = swa_indices[2]
    swa_lens_0 = swa_lens[0]
    swa_lens_1 = swa_lens[1]
    swa_lens_2 = swa_lens[2]
    metadata_ready_tid = pl.system.task_dummy(
        deps=[query_metadata_tid, visible_metadata_tid]
    )
    hidden_1 = intermediate_hidden[0]
    dspark_draft_layer(
        initial_hidden,
        hc_attn_fn_0, hc_attn_scale_0, hc_attn_base_0,
        attn_norm_w_0, wq_a_0, wq_b_0, wq_b_scale_0, wkv_0, gamma_cq_0, gamma_ckv_0,
        freqs_cos, freqs_sin, query_positions,
        kv_cache_0, context_kv_0_tid, metadata_ready_tid,
        query_slot_mapping_0, swa_indices_0, swa_lens_0,
        attn_sink_0, wo_a_0, wo_b_0, wo_b_scale_0,
        hc_ffn_fn_0, hc_ffn_scale_0, hc_ffn_base_0, ffn_norm_w_0,
        gate_w_0, gate_bias_0, tid2eid_0, query_token_ids,
        routed_w1_0, routed_w1_scale_0, routed_w3_0, routed_w3_scale_0,
        routed_w2_0, routed_w2_scale_0,
        shared_w1_0, shared_w1_scale_0, shared_w3_0, shared_w3_scale_0,
        shared_w2_0, shared_w2_scale_0,
        hidden_1,
        recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived, routed_y_buf, combine_arrived,
        pl.const(40, pl.INT32), pl.cast(active_tokens, pl.INT32), my_rank, pl.const(1, pl.INT32), lookup_tid,
    )
    rebase_1_tid = rebase_moe_signals(hidden_1, arrived, data_arrived, combine_arrived, pl.const(1, pl.INT32))
    barrier_1_tid = dspark_moe_barrier(moe_barrier_signal, my_rank, pl.const(1, pl.INT32), rebase_1_tid)

    hidden_2 = intermediate_hidden[1]
    dspark_draft_layer(
        hidden_1,
        hc_attn_fn_1, hc_attn_scale_1, hc_attn_base_1,
        attn_norm_w_1, wq_a_1, wq_b_1, wq_b_scale_1, wkv_1, gamma_cq_1, gamma_ckv_1,
        freqs_cos, freqs_sin, query_positions,
        kv_cache_1, context_kv_1_tid, metadata_ready_tid,
        query_slot_mapping_1, swa_indices_1, swa_lens_1,
        attn_sink_1, wo_a_1, wo_b_1, wo_b_scale_1,
        hc_ffn_fn_1, hc_ffn_scale_1, hc_ffn_base_1, ffn_norm_w_1,
        gate_w_1, gate_bias_1, tid2eid_1, query_token_ids,
        routed_w1_1, routed_w1_scale_1, routed_w3_1, routed_w3_scale_1,
        routed_w2_1, routed_w2_scale_1,
        shared_w1_1, shared_w1_scale_1, shared_w3_1, shared_w3_scale_1,
        shared_w2_1, shared_w2_scale_1,
        hidden_2,
        recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived, routed_y_buf, combine_arrived,
        pl.const(41, pl.INT32), pl.cast(active_tokens, pl.INT32), my_rank, pl.const(2, pl.INT32), barrier_1_tid,
    )
    rebase_2_tid = rebase_moe_signals(hidden_2, arrived, data_arrived, combine_arrived, pl.const(2, pl.INT32))
    barrier_2_tid = dspark_moe_barrier(moe_barrier_signal, my_rank, pl.const(2, pl.INT32), rebase_2_tid)

    hidden_3 = intermediate_hidden[2]
    dspark_draft_layer(
        hidden_2,
        hc_attn_fn_2, hc_attn_scale_2, hc_attn_base_2,
        attn_norm_w_2, wq_a_2, wq_b_2, wq_b_scale_2, wkv_2, gamma_cq_2, gamma_ckv_2,
        freqs_cos, freqs_sin, query_positions,
        kv_cache_2, context_kv_2_tid, metadata_ready_tid,
        query_slot_mapping_2, swa_indices_2, swa_lens_2,
        attn_sink_2, wo_a_2, wo_b_2, wo_b_scale_2,
        hc_ffn_fn_2, hc_ffn_scale_2, hc_ffn_base_2, ffn_norm_w_2,
        gate_w_2, gate_bias_2, tid2eid_2, query_token_ids,
        routed_w1_2, routed_w1_scale_2, routed_w3_2, routed_w3_scale_2,
        routed_w2_2, routed_w2_scale_2,
        shared_w1_2, shared_w1_scale_2, shared_w3_2, shared_w3_scale_2,
        shared_w2_2, shared_w2_scale_2,
        hidden_3,
        recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived, routed_y_buf, combine_arrived,
        pl.const(42, pl.INT32), pl.cast(active_tokens, pl.INT32), my_rank, pl.const(3, pl.INT32), barrier_2_tid,
    )
    clear_moe_signals(hidden_3, arrived, data_arrived, combine_arrived)
    clear_dspark_moe_barrier(hidden_3, moe_barrier_signal)

    padded_head_hidden = pl.create_tensor([T, D], dtype=pl.BF16)
    hc_head(hidden_3, hc_head_fn, hc_head_scale, hc_head_base, padded_head_hidden)
    head_hidden_flat = pl.reshape(head_hidden, [batch * DSPARK_QUERY_WIDTH, D])
    for token in pl.spmd(T, name_hint="dspark_head_unpad"):
        if token < active_tokens:
            head_hidden_flat[token : token + 1, :] = padded_head_hidden[
                token : token + 1,
                :,
            ]
    return head_hidden
