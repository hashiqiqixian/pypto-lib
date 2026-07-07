# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F401,F403,F405,F821
"""DeepSeek-V4 MTP common decode tails.

Projection now lives in ``mtp_projection.py`` and the packed prefill path lives
in ``prefill_mtp.py``.  This module keeps the decode-side pieces shared by MTP
entrypoints:

    projected_hidden -> SWA decode attention -> MoE -> pre-hc hidden
    pre-hc hidden -> hc_head -> shared-head RMSNorm -> local logits

The pre-hc hidden layout follows the current MTP projection/prefill contract:
``[T, HC_MULT, D]``.
"""

import pypto.language as pl
import pypto.language.distributed as pld

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE
from decode_attention_swa import (
    HEAD_DIM,
    H,
    MAX_SEQ_LEN,
    O_GROUP_IN,
    O_GROUPS,
    O_LORA,
    ORI_MAX_BLOCKS,
    Q_LORA,
    ROPE_HEAD_DIM,
    SPARSE_CMP_MAX_BLOCKS,
    attention_swa,
)
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
    TOPK as MOE_TOPK,
    VOCAB as MOE_VOCAB,
    moe,
)


B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
EPS = M.rms_norm_eps
D_INV = 1.0 / D
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
MTP_LAYER_ID = M.num_hidden_layers
MTP_MOE_EPOCH = 1

T_TILE = 8
MATMUL_T_TILE = 16
VOCAB_SHARD = 512
LM_HEAD_K_CHUNK = 128
LM_HEAD_K_BLOCKS = D // LM_HEAD_K_CHUNK
RMS_K_CHUNK = 128
RMS_K_BLOCKS = D // RMS_K_CHUNK

assert D % LM_HEAD_K_CHUNK == 0
assert D % RMS_K_CHUNK == 0
assert T % T_TILE == 0
assert T <= MATMUL_T_TILE


@pl.jit.inline
def mtp_local_logits(
    mtp_hidden: pl.Tensor[[B, S, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_SHARD, D], pl.BF16],
    candidate_logits: pl.Tensor[[T, VOCAB_SHARD], pl.FP32],
) -> pl.Tensor[[T, VOCAB_SHARD], pl.FP32]:
    hidden_flat = pl.reshape(mtp_hidden, [T, D])
    hidden_pad = pl.create_tensor([MATMUL_T_TILE, D], dtype=pl.BF16)
    logits_pad = pl.create_tensor([MATMUL_T_TILE, VOCAB_SHARD], dtype=pl.FP32)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_logits_hidden_pad"):
        for k0 in pl.pipeline(0, D, LM_HEAD_K_CHUNK, stage=2):
            for t0 in pl.range(0, T, T_TILE):
                hidden_pad[t0 : t0 + T_TILE, k0 : k0 + LM_HEAD_K_CHUNK] = (
                    hidden_flat[t0 : t0 + T_TILE, k0 : k0 + LM_HEAD_K_CHUNK]
                )
            if MATMUL_T_TILE > T:
                hidden_pad[T:MATMUL_T_TILE, k0 : k0 + LM_HEAD_K_CHUNK] = pl.full(
                    [MATMUL_T_TILE - T, LM_HEAD_K_CHUNK],
                    dtype=pl.BF16,
                    value=0.0,
                )

    for t0 in pl.parallel(0, MATMUL_T_TILE, MATMUL_T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_logits_lm_head_shard"):
            for kb in pl.pipeline(LM_HEAD_K_BLOCKS, stage=2):
                k0 = kb * LM_HEAD_K_CHUNK
                hidden_chunk = hidden_pad[t0 : t0 + MATMUL_T_TILE, k0 : k0 + LM_HEAD_K_CHUNK]
                weight_chunk = lm_head_weight[:, k0 : k0 + LM_HEAD_K_CHUNK]
                if kb == 0:
                    acc = pl.matmul(hidden_chunk, weight_chunk, b_trans=True, out_dtype=pl.FP32)
                else:
                    acc = pl.matmul_acc(acc, hidden_chunk, weight_chunk, b_trans=True)
            logits_pad[t0 : t0 + MATMUL_T_TILE, 0:VOCAB_SHARD] = acc

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_logits_output"):
        for t0 in pl.range(0, T, T_TILE):
            candidate_logits[t0 : t0 + T_TILE, 0:VOCAB_SHARD] = (
                logits_pad[t0 : t0 + T_TILE, 0:VOCAB_SHARD]
            )

    return candidate_logits


@pl.jit.inline
def mtp_shared_head_norm(
    mtp_hidden: pl.Tensor[[B, S, D], pl.BF16],
    shared_head_norm_w: pl.Tensor[[D], pl.FP32],
    normed_hidden: pl.Tensor[[B, S, D], pl.BF16],
) -> pl.Tensor[[B, S, D], pl.BF16]:
    hidden_flat = pl.reshape(mtp_hidden, [T, D])
    normed_flat = pl.reshape(normed_hidden, [T, D])

    for t0 in pl.parallel(0, T, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_shared_head_norm_rms"):
            sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(RMS_K_BLOCKS, stage=2):
                k0 = kb * RMS_K_CHUNK
                hidden_chunk = pl.cast(
                    hidden_flat[t0 : t0 + T_TILE, k0 : k0 + RMS_K_CHUNK],
                    target_type=pl.FP32,
                )
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(hidden_chunk, hidden_chunk)), [1, T_TILE]),
                )
            inv = pl.reshape(pl.rsqrt(pl.add(pl.mul(sq_sum, D_INV), EPS)), [T_TILE, 1])
            for kb in pl.range(RMS_K_BLOCKS):
                k0 = kb * RMS_K_CHUNK
                hidden_chunk = pl.cast(
                    hidden_flat[t0 : t0 + T_TILE, k0 : k0 + RMS_K_CHUNK],
                    target_type=pl.FP32,
                )
                weight = pl.reshape(shared_head_norm_w[k0 : k0 + RMS_K_CHUNK], [1, RMS_K_CHUNK])
                normed = pl.col_expand_mul(pl.row_expand_mul(hidden_chunk, inv), weight)
                normed_flat[t0 : t0 + T_TILE, k0 : k0 + RMS_K_CHUNK] = pl.cast(
                    normed,
                    target_type=pl.BF16,
                    mode="rint",
                )

    normed_hidden = pl.reshape(normed_flat, [B, S, D])
    return normed_hidden


@pl.jit.inline
def mtp_decoder_layer_tail(
    projected_hidden: pl.Tensor[[T, HC_MULT, D], pl.BF16],
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
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.Tensor[[B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    cmp_kv: pl.Tensor[[B * SPARSE_CMP_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[MOE_VOCAB, MOE_TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
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
    pre_hc_hidden: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, HC_MULT, D], pl.BF16]:
    x_attn = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    x_attn = attention_swa(
        projected_hidden,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        gamma_cq,
        gamma_ckv,
        freqs_cos,
        freqs_sin,
        kv_cache,
        block_table,
        ori_slot_mapping,
        position_ids,
        cmp_kv,
        cmp_block_table,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        x_attn,
    )
    pre_hc_hidden = moe(
        x_attn,
        hc_ffn_fn,
        hc_ffn_scale,
        hc_ffn_base,
        norm_w,
        gate_w,
        gate_bias,
        tid2eid,
        input_ids,
        routed_w1,
        routed_w1_scale,
        routed_w3,
        routed_w3_scale,
        routed_w2,
        routed_w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w3,
        shared_w3_scale,
        shared_w2,
        shared_w2_scale,
        pre_hc_hidden,
        recv_meta,
        recv_x,
        recv_aux,
        recv_route,
        arrived,
        routed_y_buf,
        combine_arrived,
        pl.cast(MTP_LAYER_ID, pl.INT32),
        num_tokens,
        my_rank,
        pl.cast(MTP_MOE_EPOCH, pl.INT32),
    )
    return pre_hc_hidden


@pl.jit.inline
def mtp_compute_logits_tail(
    pre_hc_hidden: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    shared_head_norm_w: pl.Tensor[[D], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB_SHARD, D], pl.BF16],
    candidate_logits: pl.Tensor[[T, VOCAB_SHARD], pl.FP32],
) -> pl.Tensor[[T, VOCAB_SHARD], pl.FP32]:
    dense_hidden_flat = pl.create_tensor([T, D], dtype=pl.BF16)
    dense_hidden_flat = hc_head(
        pre_hc_hidden,
        hc_head_fn,
        hc_head_scale,
        hc_head_base,
        dense_hidden_flat,
    )
    dense_hidden = pl.reshape(dense_hidden_flat, [B, S, D])
    normed_hidden = pl.create_tensor([B, S, D], dtype=pl.BF16)
    normed_hidden = mtp_shared_head_norm(dense_hidden, shared_head_norm_w, normed_hidden)
    candidate_logits = mtp_local_logits(normed_hidden, lm_head_weight, candidate_logits)
    return candidate_logits
