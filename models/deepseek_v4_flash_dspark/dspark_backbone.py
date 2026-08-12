# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""Multi-rank synthetic validation harness for the three-layer DSpark draft backbone."""

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from config import (
    BLOCK_SIZE,
    DECODE_SEQ,
    DSPARK_DRAFT_LAYERS,
    DSPARK_MAX_BATCH,
    DSPARK_NOISE_TOKEN_ID,
    DSPARK_QUERY_TOKENS,
    DSPARK_QUERY_WIDTH,
    DSPARK_SUPPORTED_BATCHES,
    FLASH as M,
)
from draft_backbone import (
    AUX_PAD,
    B_DYN,
    D,
    HEAD_DIM,
    HC_DIM,
    HC_MULT,
    H,
    IDX_PAD,
    MAIN_IN,
    MAX_SEQ_LEN,
    MIX_HC,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    N_ROUTES,
    ORI_BLOCK_NUM,
    ORI_MAX_BLOCKS,
    O_GROUP_IN,
    O_GROUPS,
    O_LORA,
    Q_LORA,
    RECV_MAX,
    ROPE_DIM,
    T_MAIN_DYN,
    TOPK,
    VOCAB,
    draft_backbone,
)
from qkv_proj_rope import T_DYN as QKV_Q_T_DYN


# Three draft layers plus their MoE communication graph exceed the runtime's
# default per-ring heap. Match the established large-model harness allocation.
_DSPARK_RING_HEAP = (4 * 1024 * 1024 * 1024,) * 4


@pl.jit.host
def l3_draft_backbone(
    target_hidden: pl.Tensor[[N_RANKS, T_MAIN_DYN, MAIN_IN], pl.BF16],
    initial_hidden: pl.Out[
        pl.Tensor[[N_RANKS, DSPARK_MAX_BATCH * 8, HC_MULT, D], pl.FP32]
    ],
    hc_attn_inv_rms: pl.Out[
        pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH * 8, 1], pl.FP32]
    ],
    main_proj_weight: pl.Tensor[[N_RANKS, D, MAIN_IN], pl.BF16],
    main_norm_weight: pl.Tensor[[N_RANKS, D], pl.BF16],
    anchor_token_ids: pl.Tensor[[N_RANKS, B_DYN], pl.INT64],
    embedding_weight: pl.Tensor[[N_RANKS, VOCAB, D], pl.BF16],
    context_position_ids: pl.Tensor[[N_RANKS, QKV_Q_T_DYN], pl.INT32],
    context_slot_mapping: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS, QKV_Q_T_DYN], pl.INT64],
    anchor_positions: pl.Tensor[[N_RANKS, B_DYN], pl.INT32],
    block_tables: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS, B_DYN, ORI_MAX_BLOCKS], pl.INT32],
    freqs_cos: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    hc_attn_fn_0: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale_0: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_attn_base_0: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    attn_norm_w_0: pl.Tensor[[N_RANKS, D], pl.BF16],
    wq_a_0: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16],
    wq_b_0: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale_0: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32],
    wkv_0: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16],
    gamma_cq_0: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16],
    gamma_ckv_0: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    kv_caches_0: pl.InOut[pl.Tensor[[N_RANKS, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink_0: pl.Tensor[[N_RANKS, H], pl.FP32],
    wo_a_0: pl.Tensor[[N_RANKS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_0: pl.Tensor[[N_RANKS, D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale_0: pl.Tensor[[N_RANKS, D], pl.FP32],
    hc_ffn_fn_0: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale_0: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base_0: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    ffn_norm_w_0: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w_0: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias_0: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid_0: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    routed_w1_0: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale_0: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3_0: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale_0: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2_0: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale_0: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1_0: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale_0: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3_0: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale_0: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2_0: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale_0: pl.Tensor[[N_RANKS, D], pl.FP32],
    hc_attn_fn_1: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale_1: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_attn_base_1: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    attn_norm_w_1: pl.Tensor[[N_RANKS, D], pl.BF16],
    wq_a_1: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16],
    wq_b_1: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale_1: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32],
    wkv_1: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16],
    gamma_cq_1: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16],
    gamma_ckv_1: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    kv_caches_1: pl.InOut[pl.Tensor[[N_RANKS, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink_1: pl.Tensor[[N_RANKS, H], pl.FP32],
    wo_a_1: pl.Tensor[[N_RANKS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_1: pl.Tensor[[N_RANKS, D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale_1: pl.Tensor[[N_RANKS, D], pl.FP32],
    hc_ffn_fn_1: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale_1: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base_1: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    ffn_norm_w_1: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w_1: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias_1: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid_1: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    routed_w1_1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale_1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3_1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale_1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2_1: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale_1: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1_1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale_1: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3_1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale_1: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2_1: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale_1: pl.Tensor[[N_RANKS, D], pl.FP32],
    hc_attn_fn_2: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale_2: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_attn_base_2: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    attn_norm_w_2: pl.Tensor[[N_RANKS, D], pl.BF16],
    wq_a_2: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16],
    wq_b_2: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale_2: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32],
    wkv_2: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16],
    gamma_cq_2: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16],
    gamma_ckv_2: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    kv_caches_2: pl.InOut[pl.Tensor[[N_RANKS, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink_2: pl.Tensor[[N_RANKS, H], pl.FP32],
    wo_a_2: pl.Tensor[[N_RANKS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_2: pl.Tensor[[N_RANKS, D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale_2: pl.Tensor[[N_RANKS, D], pl.FP32],
    hc_ffn_fn_2: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale_2: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base_2: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    ffn_norm_w_2: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w_2: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias_2: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid_2: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    routed_w1_2: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale_2: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3_2: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale_2: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2_2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale_2: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1_2: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale_2: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3_2: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale_2: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2_2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale_2: pl.Tensor[[N_RANKS, D], pl.FP32],
    hc_head_fn: pl.Tensor[[N_RANKS, HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[N_RANKS, 1], pl.FP32],
    hc_head_base: pl.Tensor[[N_RANKS, HC_MULT], pl.FP32],
    head_hidden: pl.Out[pl.Tensor[[N_RANKS, B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16]],
):
    target_hidden.bind_dynamic(1, T_MAIN_DYN)
    context_position_ids.bind_dynamic(1, QKV_Q_T_DYN)
    context_slot_mapping.bind_dynamic(2, QKV_Q_T_DYN)
    anchor_token_ids.bind_dynamic(1, B_DYN)
    anchor_positions.bind_dynamic(1, B_DYN)
    block_tables.bind_dynamic(2, B_DYN)
    head_hidden.bind_dynamic(1, B_DYN)

    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    moe_barrier_signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        moe_barrier_signal = pld.window(moe_barrier_signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        draft_backbone(
            target_hidden[rank], main_proj_weight[rank], main_norm_weight[rank],
            anchor_token_ids[rank], embedding_weight[rank],
            context_position_ids[rank], context_slot_mapping[rank], anchor_positions[rank], block_tables[rank],
            freqs_cos[rank], freqs_sin[rank],
            hc_attn_fn_0[rank], hc_attn_scale_0[rank], hc_attn_base_0[rank], attn_norm_w_0[rank],
            wq_a_0[rank], wq_b_0[rank], wq_b_scale_0[rank], wkv_0[rank], gamma_cq_0[rank], gamma_ckv_0[rank],
            kv_caches_0[rank], attn_sink_0[rank], wo_a_0[rank], wo_b_0[rank], wo_b_scale_0[rank],
            hc_ffn_fn_0[rank], hc_ffn_scale_0[rank], hc_ffn_base_0[rank], ffn_norm_w_0[rank],
            gate_w_0[rank], gate_bias_0[rank], tid2eid_0[rank],
            routed_w1_0[rank], routed_w1_scale_0[rank], routed_w3_0[rank], routed_w3_scale_0[rank],
            routed_w2_0[rank], routed_w2_scale_0[rank],
            shared_w1_0[rank], shared_w1_scale_0[rank], shared_w3_0[rank], shared_w3_scale_0[rank],
            shared_w2_0[rank], shared_w2_scale_0[rank],
            hc_attn_fn_1[rank], hc_attn_scale_1[rank], hc_attn_base_1[rank], attn_norm_w_1[rank],
            wq_a_1[rank], wq_b_1[rank], wq_b_scale_1[rank], wkv_1[rank], gamma_cq_1[rank], gamma_ckv_1[rank],
            kv_caches_1[rank], attn_sink_1[rank], wo_a_1[rank], wo_b_1[rank], wo_b_scale_1[rank],
            hc_ffn_fn_1[rank], hc_ffn_scale_1[rank], hc_ffn_base_1[rank], ffn_norm_w_1[rank],
            gate_w_1[rank], gate_bias_1[rank], tid2eid_1[rank],
            routed_w1_1[rank], routed_w1_scale_1[rank], routed_w3_1[rank], routed_w3_scale_1[rank],
            routed_w2_1[rank], routed_w2_scale_1[rank],
            shared_w1_1[rank], shared_w1_scale_1[rank], shared_w3_1[rank], shared_w3_scale_1[rank],
            shared_w2_1[rank], shared_w2_scale_1[rank],
            hc_attn_fn_2[rank], hc_attn_scale_2[rank], hc_attn_base_2[rank], attn_norm_w_2[rank],
            wq_a_2[rank], wq_b_2[rank], wq_b_scale_2[rank], wkv_2[rank], gamma_cq_2[rank], gamma_ckv_2[rank],
            kv_caches_2[rank], attn_sink_2[rank], wo_a_2[rank], wo_b_2[rank], wo_b_scale_2[rank],
            hc_ffn_fn_2[rank], hc_ffn_scale_2[rank], hc_ffn_base_2[rank], ffn_norm_w_2[rank],
            gate_w_2[rank], gate_bias_2[rank], tid2eid_2[rank],
            routed_w1_2[rank], routed_w1_scale_2[rank], routed_w3_2[rank], routed_w3_scale_2[rank],
            routed_w2_2[rank], routed_w2_scale_2[rank],
            shared_w1_2[rank], shared_w1_scale_2[rank], shared_w3_2[rank], shared_w3_scale_2[rank],
            shared_w2_2[rank], shared_w2_scale_2[rank],
            hc_head_fn[rank], hc_head_scale[rank], hc_head_base[rank], initial_hidden[rank],
            hc_attn_inv_rms[rank, 0], hc_attn_inv_rms[rank, 1], hc_attn_inv_rms[rank, 2], head_hidden[rank],
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived, routed_y_buf, combine_arrived,
            moe_barrier_signal, rank,
            device=rank,
        )


def _anchor_position_set(batch):
    import torch

    cases = torch.tensor(
        [
            1,
            BLOCK_SIZE - 1,
            M.sliding_window - 1,
            M.sliding_window + BLOCK_SIZE,
            2 * BLOCK_SIZE - 1,
            4 * BLOCK_SIZE + 3,
            8 * BLOCK_SIZE - 1,
            MAX_SEQ_LEN - DSPARK_QUERY_WIDTH - 1,
        ],
        dtype=torch.int32,
    )
    repeats = (batch + cases.numel() - 1) // cases.numel()
    return cases.repeat(repeats)[:batch].contiguous()


def _block_tables(batch):
    import torch

    logical = torch.arange(ORI_MAX_BLOCKS, dtype=torch.int32)
    tables = torch.empty(N_RANKS, DSPARK_DRAFT_LAYERS, batch, ORI_MAX_BLOCKS, dtype=torch.int32)
    for rank in range(N_RANKS):
        for layer in range(DSPARK_DRAFT_LAYERS):
            for request in range(batch):
                request_base = request * (ORI_BLOCK_NUM // DSPARK_MAX_BATCH)
                ring_offset = rank * 3 + layer * 7
                request_block = (logical + ring_offset) % (ORI_BLOCK_NUM // DSPARK_MAX_BATCH)
                tables[rank, layer, request] = request_base + request_block
    return tables


def _context_slots(tables, positions):
    import torch

    slots = torch.empty(N_RANKS, DSPARK_DRAFT_LAYERS, positions.shape[1], dtype=torch.int64)
    for rank in range(N_RANKS):
        for layer in range(DSPARK_DRAFT_LAYERS):
            for request in range(positions.shape[1]):
                position = int(positions[rank, request])
                physical_block = int(tables[rank, layer, request, position // BLOCK_SIZE])
                slots[rank, layer, request] = physical_block * BLOCK_SIZE + position % BLOCK_SIZE
    return slots


def _balanced_routes():
    import torch

    token_ids = torch.arange(VOCAB, dtype=torch.int64).unsqueeze(1)
    route_ids = torch.arange(TOPK, dtype=torch.int64).unsqueeze(0)
    routes = ((token_ids * TOPK + route_ids) % N_EXPERTS_GLOBAL).to(torch.int32)
    return routes.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()


def build_tensor_specs(batch):
    import torch
    from golden import TensorSpec

    if batch not in DSPARK_SUPPORTED_BATCHES:
        raise ValueError(f"unsupported DSpark batch {batch}; expected one of {DSPARK_SUPPORTED_BATCHES}")

    positions = _anchor_position_set(batch).unsqueeze(0).expand(N_RANKS, -1).contiguous()
    tables = _block_tables(batch)
    context_slots = _context_slots(tables, positions)
    context_positions_padded = torch.zeros(N_RANKS, DSPARK_QUERY_TOKENS, dtype=torch.int32)
    context_positions_padded[:, :batch] = positions
    context_slots_padded = torch.full(
        (N_RANKS, DSPARK_DRAFT_LAYERS, DSPARK_QUERY_TOKENS),
        -1,
        dtype=torch.int64,
    )
    context_slots_padded[:, :, :batch] = context_slots
    anchors = torch.arange(batch, dtype=torch.int64).unsqueeze(0).expand(N_RANKS, -1).contiguous()
    anchors = (anchors + 1) % VOCAB
    routes = _balanced_routes()

    def init_target_hidden():
        values = torch.zeros(N_RANKS, batch * DECODE_SEQ, MAIN_IN, dtype=torch.bfloat16)
        columns = torch.arange(D, dtype=torch.float32)
        base = ((columns % 31) - 15) * 0.002
        for rank in range(N_RANKS):
            for request in range(batch):
                for offset in range(DECODE_SEQ):
                    row = request * DECODE_SEQ + offset
                    values[rank, row, :D] = (
                        base + 0.01 * (rank + request + 1) + 0.001 * offset
                    ).to(torch.bfloat16)
        return values

    def init_main_proj_weight():
        weight = torch.zeros(N_RANKS, D, MAIN_IN, dtype=torch.bfloat16)
        diagonal = torch.arange(D)
        weight[:, diagonal, diagonal] = 1
        return weight

    def init_embedding_weight():
        weight = torch.zeros(N_RANKS, VOCAB, D, dtype=torch.bfloat16)
        columns = torch.arange(D, dtype=torch.float32)
        base = ((columns % 29) - 14) * 0.002
        for rank in range(N_RANKS):
            weight[rank, 0] = (base + 0.005 * rank).to(torch.bfloat16)
            weight[rank, DSPARK_NOISE_TOKEN_ID] = (base + 0.03 + 0.005 * rank).to(torch.bfloat16)
            for token in range(1, batch + 1):
                weight[rank, token] = (base + 0.01 * token + 0.005 * rank).to(torch.bfloat16)
        return weight

    def init_wkv():
        weight = torch.zeros(N_RANKS, D, HEAD_DIM, dtype=torch.bfloat16)
        diagonal = torch.arange(HEAD_DIM)
        weight[:, diagonal, diagonal] = 1
        return weight

    def ranked(name, shape, dtype, init_value=0, *, output=False, resident=False):
        spec = TensorSpec(name, [N_RANKS, *shape], dtype, init_value=init_value, is_output=output)
        if resident:
            spec.resident = "stacked"
        return spec

    def cache_init(layer):
        def init():
            cache = torch.empty(N_RANKS, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
            for rank in range(N_RANKS):
                cache[rank].fill_(layer + 1 + rank * 0.25)
            return cache

        return init

    specs = [
        ranked("target_hidden", [batch * DECODE_SEQ, MAIN_IN], torch.bfloat16, init_value=init_target_hidden),
        TensorSpec(
            "initial_hidden",
            [N_RANKS, DSPARK_MAX_BATCH * 8, HC_MULT, D],
            torch.float32,
            is_output=True,
        ),
        TensorSpec(
            "hc_attn_inv_rms",
            [N_RANKS, DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH * 8, 1],
            torch.float32,
            is_output=True,
        ),
        ranked("main_proj_weight", [D, MAIN_IN], torch.bfloat16, init_value=init_main_proj_weight, resident=True),
        ranked("main_norm_weight", [D], torch.bfloat16, init_value=1, resident=True),
        ranked("anchor_token_ids", [batch], torch.int64, init_value=lambda: anchors),
        ranked("embedding_weight", [VOCAB, D], torch.bfloat16, init_value=init_embedding_weight, resident=True),
        ranked(
            "context_position_ids",
            [DSPARK_QUERY_TOKENS],
            torch.int32,
            init_value=lambda: context_positions_padded,
        ),
        ranked(
            "context_slot_mapping",
            [DSPARK_DRAFT_LAYERS, DSPARK_QUERY_TOKENS],
            torch.int64,
            init_value=lambda: context_slots_padded,
        ),
        ranked("anchor_positions", [batch], torch.int32, init_value=lambda: positions),
        ranked("block_tables", [DSPARK_DRAFT_LAYERS, batch, ORI_MAX_BLOCKS], torch.int32, init_value=lambda: tables),
        ranked("freqs_cos", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, init_value=1, resident=True),
        ranked("freqs_sin", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, resident=True),
    ]

    for layer in range(DSPARK_DRAFT_LAYERS):
        suffix = f"_{layer}"
        specs.extend(
            [
                ranked(f"hc_attn_fn{suffix}", [MIX_HC, HC_DIM], torch.float32, resident=True),
                ranked(f"hc_attn_scale{suffix}", [3], torch.float32, resident=True),
                ranked(f"hc_attn_base{suffix}", [MIX_HC], torch.float32, resident=True),
                ranked(f"attn_norm_w{suffix}", [D], torch.bfloat16, init_value=1, resident=True),
                ranked(f"wq_a{suffix}", [D, Q_LORA], torch.bfloat16, resident=True),
                ranked(f"wq_b{suffix}", [Q_LORA, H * HEAD_DIM], torch.int8, resident=True),
                ranked(f"wq_b_scale{suffix}", [H * HEAD_DIM], torch.float32, resident=True),
                ranked(f"wkv{suffix}", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv, resident=True),
                ranked(f"gamma_cq{suffix}", [Q_LORA], torch.bfloat16, init_value=1, resident=True),
                ranked(f"gamma_ckv{suffix}", [HEAD_DIM], torch.bfloat16, init_value=1, resident=True),
                ranked(
                    f"kv_caches{suffix}",
                    [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
                    torch.bfloat16,
                    init_value=cache_init(layer),
                    output=True,
                ),
                ranked(f"attn_sink{suffix}", [H], torch.float32, resident=True),
                ranked(f"wo_a{suffix}", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, resident=True),
                ranked(f"wo_b{suffix}", [D, O_GROUPS * O_LORA], torch.int8, resident=True),
                ranked(f"wo_b_scale{suffix}", [D], torch.float32, resident=True),
                ranked(f"hc_ffn_fn{suffix}", [MIX_HC, HC_DIM], torch.float32, resident=True),
                ranked(f"hc_ffn_scale{suffix}", [3], torch.float32, resident=True),
                ranked(f"hc_ffn_base{suffix}", [MIX_HC], torch.float32, resident=True),
                ranked(f"ffn_norm_w{suffix}", [D], torch.bfloat16, init_value=1, resident=True),
                ranked(f"gate_w{suffix}", [N_EXPERTS_GLOBAL, D], torch.float32, resident=True),
                ranked(f"gate_bias{suffix}", [N_EXPERTS_GLOBAL], torch.float32, resident=True),
                ranked(f"tid2eid{suffix}", [VOCAB, TOPK], torch.int32, init_value=lambda: routes, resident=True),
                ranked(f"routed_w1{suffix}", [N_LOCAL, MOE_INTER, D], torch.int8, resident=True),
                ranked(f"routed_w1_scale{suffix}", [N_LOCAL, MOE_INTER], torch.float32, resident=True),
                ranked(f"routed_w3{suffix}", [N_LOCAL, MOE_INTER, D], torch.int8, resident=True),
                ranked(f"routed_w3_scale{suffix}", [N_LOCAL, MOE_INTER], torch.float32, resident=True),
                ranked(f"routed_w2{suffix}", [N_LOCAL, D, MOE_INTER], torch.int8, resident=True),
                ranked(f"routed_w2_scale{suffix}", [N_LOCAL, D], torch.float32, resident=True),
                ranked(f"shared_w1{suffix}", [MOE_INTER, D], torch.int8, resident=True),
                ranked(f"shared_w1_scale{suffix}", [MOE_INTER], torch.float32, resident=True),
                ranked(f"shared_w3{suffix}", [MOE_INTER, D], torch.int8, resident=True),
                ranked(f"shared_w3_scale{suffix}", [MOE_INTER], torch.float32, resident=True),
                ranked(f"shared_w2{suffix}", [D, MOE_INTER], torch.int8, resident=True),
                ranked(f"shared_w2_scale{suffix}", [D], torch.float32, resident=True),
            ]
        )

    specs.extend(
        [
            ranked("hc_head_fn", [HC_MULT, HC_DIM], torch.float32, resident=True),
            ranked("hc_head_scale", [1], torch.float32, resident=True),
            ranked("hc_head_base", [HC_MULT], torch.float32, resident=True),
            TensorSpec(
                "head_hidden",
                [N_RANKS, batch, DSPARK_QUERY_WIDTH, D],
                torch.bfloat16,
                is_output=True,
            ),
        ]
    )
    return specs


def golden_draft_backbone(tensors):
    import torch

    def rms_norm(hidden):
        hidden_fp32 = hidden.float()
        inv_rms = torch.rsqrt(hidden_fp32.square().mean(dim=-1, keepdim=True) + M.rms_norm_eps)
        return (hidden_fp32 * inv_rms).to(torch.bfloat16)

    def project_kv(hidden):
        projected = hidden.float()[..., :HEAD_DIM]
        inv_rms = torch.rsqrt(projected.square().mean(dim=-1, keepdim=True) + M.rms_norm_eps)
        return (projected * inv_rms).to(torch.bfloat16)

    def hc_coefficients(token_count):
        pre = torch.full((token_count, HC_MULT), 0.5 + M.hc_eps, dtype=torch.float32)
        post = torch.ones(token_count, HC_MULT, dtype=torch.float32)
        combine = torch.full((token_count, HC_MULT, HC_MULT), 0.25 + M.hc_eps, dtype=torch.float32)
        combine = combine / (combine.sum(-2, keepdim=True) + M.hc_eps)
        for _ in range(M.hc_sinkhorn_iters - 1):
            combine = combine / (combine.sum(-1, keepdim=True) + M.hc_eps)
            combine = combine / (combine.sum(-2, keepdim=True) + M.hc_eps)
        return pre, post, combine

    def hc_pre_zero_function(hidden):
        pre, post, combine = hc_coefficients(hidden.shape[0])
        mixed = hidden[:, 0] * pre[:, 0:1]
        for lane in range(1, HC_MULT):
            mixed = mixed + hidden[:, lane] * pre[:, lane : lane + 1]
        return mixed.to(torch.bfloat16), post, combine

    def hc_post_zero_output(residual, post, combine):
        output = torch.empty_like(residual)
        zero = torch.zeros(residual.shape[0], D, dtype=torch.float32)
        for output_lane in range(HC_MULT):
            row = zero * post[:, output_lane : output_lane + 1]
            for input_lane in range(HC_MULT):
                row = row + residual[:, input_lane] * combine[:, input_lane, output_lane : output_lane + 1]
            output[:, output_lane] = row
        return output

    tensors["head_hidden"].zero_()
    tensors["initial_hidden"].zero_()
    tensors["hc_attn_inv_rms"].zero_()
    positions = tensors["anchor_positions"]
    tables = tensors["block_tables"]
    context_slots = tensors["context_slot_mapping"]
    for rank in range(N_RANKS):
        main_x = rms_norm(tensors["target_hidden"][rank, :, :D])
        query_ids = torch.zeros(DSPARK_MAX_BATCH * 8, dtype=torch.int64)
        for request in range(positions.shape[1]):
            row = request * DSPARK_QUERY_WIDTH
            query_ids[row] = tensors["anchor_token_ids"][rank, request]
            query_ids[row + 1 : row + DSPARK_QUERY_WIDTH] = DSPARK_NOISE_TOKEN_ID
        query_hidden = tensors["embedding_weight"][rank].index_select(0, query_ids)
        hidden = query_hidden.float().unsqueeze(1).expand(-1, HC_MULT, -1).contiguous()
        tensors["initial_hidden"][rank] = hidden

        for layer in range(DSPARK_DRAFT_LAYERS):
            cache = tensors[f"kv_caches_{layer}"][rank].view(-1, HEAD_DIM)
            layer_context_slots = context_slots[rank, layer]
            valid_context_slots = layer_context_slots[layer_context_slots >= 0].long()
            context_x = main_x[DECODE_SEQ - 1 :: DECODE_SEQ]
            cache[valid_context_slots] = project_kv(context_x[: valid_context_slots.numel()])
            query_slots = []
            for request in range(positions.shape[1]):
                anchor = int(positions[rank, request])
                for offset in range(1, DSPARK_QUERY_WIDTH + 1):
                    position = anchor + offset
                    physical_block = int(tables[rank, layer, request, position // BLOCK_SIZE])
                    query_slots.append(physical_block * BLOCK_SIZE + position % BLOCK_SIZE)
            tensors["hc_attn_inv_rms"][rank, layer, :, 0] = torch.rsqrt(
                hidden.reshape(hidden.shape[0], -1).square().mean(dim=-1) + M.rms_norm_eps
            )
            mixed, post, combine = hc_pre_zero_function(hidden)
            query_normed = rms_norm(mixed[: DSPARK_MAX_BATCH * DSPARK_QUERY_WIDTH])
            active_tokens = positions.shape[1] * DSPARK_QUERY_WIDTH
            cache[torch.tensor(query_slots, dtype=torch.int64)] = project_kv(query_normed[:active_tokens])
            attention_hidden = hc_post_zero_output(
                hidden[: DSPARK_MAX_BATCH * DSPARK_QUERY_WIDTH],
                post[: DSPARK_MAX_BATCH * DSPARK_QUERY_WIDTH],
                combine[: DSPARK_MAX_BATCH * DSPARK_QUERY_WIDTH],
            )
            padded_attention = torch.zeros_like(hidden)
            padded_attention[: positions.shape[1] * DSPARK_QUERY_WIDTH] = attention_hidden[
                : positions.shape[1] * DSPARK_QUERY_WIDTH
            ]
            _, moe_post, moe_combine = hc_pre_zero_function(padded_attention)
            hidden = hc_post_zero_output(padded_attention, moe_post, moe_combine)

        active = hidden[: positions.shape[1] * DSPARK_QUERY_WIDTH]
        head_pre = torch.full((active.shape[0], HC_MULT), 0.5 + M.hc_eps, dtype=torch.float32)
        head = active[:, 0] * head_pre[:, 0:1]
        head = head + active[:, 1] * head_pre[:, 1:2]
        tail = active[:, 2] * head_pre[:, 2:3]
        tail = tail + active[:, 3] * head_pre[:, 3:4]
        head = (head + tail).to(torch.bfloat16)
        tensors["head_hidden"][rank] = head.view(positions.shape[1], DSPARK_QUERY_WIDTH, D)


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser(description="Validate the multi-rank DeepSeek V4 DSpark draft backbone.")
    parser.add_argument("--batch", type=int, choices=DSPARK_SUPPORTED_BATCHES, default=4)
    parser.add_argument("--ep", type=int, choices=(2, 4, 8, 16), default=2)
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)))
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--dump-args", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--enable-scope-stats", action="store_true")
    args = parser.parse_args()

    device_ids = [int(device) for device in args.device.split(",")]
    assert args.ep == N_RANKS
    assert len(device_ids) >= N_RANKS
    result = run_jit(
        fn=l3_draft_backbone,
        specs=build_tensor_specs(args.batch),
        golden_fn=golden_draft_backbone,
        compile_only=args.compile_only,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids[:N_RANKS], num_sub_workers=0),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            ring_heap=_DSPARK_RING_HEAP,
            enable_dep_gen=True,
            enable_dump_args=args.dump_args,
            enable_scope_stats=args.enable_scope_stats,
        ),
        rtol=1e-3,
        atol=1e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
