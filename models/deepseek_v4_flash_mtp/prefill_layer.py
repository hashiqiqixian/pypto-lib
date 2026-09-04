# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2 # CI: 2-card run; borrows 2 cards via task-submit --device-num
"""DeepSeek-V4 Flash one-layer numerical harness with MoE EP2.

The default remains the original fresh ``B=1, S=128`` path.  Passing
``--dispatch-batch 4`` specializes the same one-layer program to four
request-isolated streams sharing the production cache/state and MoE windows.
This executable harness is not a serving ABI; tensor ranks specialize at import time while its CLI names and default B1 numerical path stay unchanged.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

# The prefill path routes PREFILL_TOKENS tokens. Set MOE_TOKENS before importing
# moe (which freezes recv shapes and derives RECV_MAX = EP * MOE_TOKENS at import).
import config
config.MOE_TOKENS = config.PREFILL_TOKENS
# Import moe first. It applies the EP2 FLASH override before dependent
# modules bake config-derived MoE shapes.
from moe import (
    AUX_PAD,
    D,
    HC_DIM,
    HC_MULT,
    IDX_PAD,
    MIX_HC,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    N_ROUTES,
    RECV_MAX,
    SIGNAL_PAD,
    T,
    TOPK,
    VOCAB,
    build_tensor_specs as build_moe_tensor_specs,
    clear_prefill_moe_signals,
    golden_moe,
    prefill_moe,
)
from config import FLASH as MODEL_CONFIG, PREFILL_BATCH, PREFILL_DISPATCH_BATCH, PREFILL_SEQ
from prefill_swa import (
    BLOCK_NUM as SWA_ORI_BLOCK_NUM,
    BLOCK_SIZE as SWA_BLOCK_SIZE,
    build_tensor_specs as build_swa_attention_tensor_specs,
    golden_prefill_attention_swa,
    prefill_attention_swa,
)
from prefill_hca import (
    COMPRESS_RATIO as HCA_COMPRESS_RATIO,
    CMP_STORAGE_BLOCK_SIZE as HCA_CMP_STORAGE_BLOCK_SIZE,
    HCA_CMP_BLOCK_NUM,
    HCA_ORI_BLOCK_NUM,
    HCA_STATE_BLOCK_NUM,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as HCA_MAIN_OUT_DIM,
    build_tensor_specs as build_hca_attention_tensor_specs,
    golden_prefill_attention_hca,
    prefill_attention_hca,
)
from prefill_csa import (
    BLOCK_SIZE,
    COMPRESS_RATIO as CSA_COMPRESS_RATIO,
    CMP_STORAGE_BLOCK_SIZE as CSA_CMP_STORAGE_BLOCK_SIZE,
    CSA_CMP_BLOCK_NUM,
    CSA_ORI_BLOCK_NUM,
    CSA_STATE_BLOCK_NUM,
    CSA_STATE_BLOCK_SIZE,
    CSA_STATE_MAX_BLOCKS,
    H,
    HEAD_DIM,
    IDX_CACHE_MAX_BLOCKS,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    INNER_OUT_DIM,
    INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as CSA_MAIN_OUT_DIM,
    MAX_SEQ_LEN,
    O_GROUPS,
    O_GROUP_IN,
    O_LORA,
    PREFILL_IDX_BLOCK_NUM,
    Q_LORA,
    ROPE_HEAD_DIM,
    SPARSE_CMP_MAX_BLOCKS,
    SPARSE_ORI_MAX_BLOCKS,
    build_tensor_specs as build_csa_attention_tensor_specs,
    golden_prefill_attention_csa,
    prefill_attention_csa,
)
assert SWA_BLOCK_SIZE == BLOCK_SIZE, "SWA/HCA/CSA must share the PyPTO block size"
assert SWA_ORI_BLOCK_NUM == HCA_ORI_BLOCK_NUM == CSA_ORI_BLOCK_NUM

# The dispatch batch is import-time specialized because it is part of tensor shapes.
TOK_TILE = T
_DISPATCH_BATCH_CHOICES = (1, PREFILL_DISPATCH_BATCH)


def _parse_int_argv(name, default):
    for index, token in enumerate(sys.argv):
        if token == name and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if token.startswith(f"{name}="):
            return int(token.split("=", 1)[1])
    return default


USER_BATCH = _parse_int_argv("--dispatch-batch", 1)
LAYER_ID = _parse_int_argv("--layer-id", 2)

assert PREFILL_BATCH == 1, "prefill_layer leaf programs require B=1"
assert USER_BATCH in _DISPATCH_BATCH_CHOICES, (
    f"--dispatch-batch must be one of {_DISPATCH_BATCH_CHOICES} (got {USER_BATCH})"
)
assert PREFILL_SEQ == TOK_TILE == 128, "prefill_layer requires S=128"

# Fixed shared cache/state/table capacities for the selected dispatch batch.
ORI_CACHE_BLOCKS = CSA_ORI_BLOCK_NUM
HCA_CMP_CACHE_BLOCKS = HCA_CMP_BLOCK_NUM
CSA_CMP_CACHE_BLOCKS = CSA_CMP_BLOCK_NUM
IDX_CACHE_BLOCKS = PREFILL_IDX_BLOCK_NUM
ORI_TABLE_BLOCKS = SPARSE_ORI_MAX_BLOCKS
CMP_TABLE_BLOCKS = SPARSE_CMP_MAX_BLOCKS
IDX_TABLE_BLOCKS = IDX_CACHE_MAX_BLOCKS
HCA_STATE_POOL_BLOCKS = max(
    HCA_STATE_BLOCK_NUM,
    USER_BATCH
    * (
        (T + HCA_COMPRESS_RATIO - 1 + HCA_STATE_BLOCK_SIZE - 1)
        // HCA_STATE_BLOCK_SIZE
    ),
)
CSA_STATE_POOL_BLOCKS = max(
    CSA_STATE_BLOCK_NUM,
    USER_BATCH * ((T + CSA_STATE_BLOCK_SIZE - 1) // CSA_STATE_BLOCK_SIZE),
)
INNER_STATE_POOL_BLOCKS = max(
    INNER_STATE_BLOCK_NUM,
    USER_BATCH * ((T + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE),
)
assert (
    HCA_STATE_POOL_BLOCKS // USER_BATCH * HCA_STATE_BLOCK_SIZE
    >= T + HCA_COMPRESS_RATIO - 1
)
assert CSA_STATE_POOL_BLOCKS // USER_BATCH * CSA_STATE_BLOCK_SIZE >= T
assert INNER_STATE_POOL_BLOCKS // USER_BATCH * INNER_STATE_BLOCK_SIZE >= T

# Per-ring runtime output heap, 1 GiB on each of the 4 rings. The 256 MiB
# compile-time default deadlocks the ring allocator on this layer.
PREFILL_RING_HEAP = (1024 * 1024 * 1024,) * 4


# JIT dependency discovery scans every syntactic branch before IR simplification.
# Replace only the unselected sparse-attention entry with an arity-compatible
# no-op so HCA (storage width 1) and CSA (storage width 32) are never bound to
# the same dynamic symbol in one executable-harness specialization.
if not (LAYER_ID >= 2 and LAYER_ID % 2 == 1):

    @pl.jit.inline(auto_scope=False)
    def prefill_attention_hca(
        x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
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
        cmp_wkv: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
        cmp_wgate: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
        cmp_ape: pl.Tensor[[HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
        cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
        compress_state: pl.Tensor[
            [HCA_STATE_POOL_BLOCKS, HCA_STATE_BLOCK_SIZE, 2 * HCA_MAIN_OUT_DIM], pl.FP32
        ],
        compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
        kv_cache: pl.Tensor[[ORI_CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
        ori_slot_mapping: pl.Tensor[[T], pl.INT64],
        ori_block_table: pl.Tensor[[ORI_TABLE_BLOCKS], pl.INT32],
        cmp_kv: pl.Tensor[
            [HCA_CMP_CACHE_BLOCKS, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
        ],
        cmp_block_table: pl.Tensor[[CMP_TABLE_BLOCKS], pl.INT32],
        position_ids: pl.Tensor[[T], pl.INT32],
        cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
        state_slot_mapping: pl.Tensor[[T], pl.INT64],
        attn_sink: pl.Tensor[[H], pl.FP32],
        wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
        wo_b_scale: pl.Tensor[[D], pl.FP32],
        x_out: pl.Tensor[[T, HC_MULT, D], pl.FP32],
        num_tokens: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[T, HC_MULT, D], pl.FP32]:
        return x_out


if not (LAYER_ID >= 2 and LAYER_ID % 2 == 0):

    @pl.jit.inline(auto_scope=False)
    def prefill_attention_csa(
        x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
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
        cmp_wkv: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
        cmp_wgate: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
        cmp_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
        cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
        compress_state: pl.Tensor[
            [CSA_STATE_POOL_BLOCKS, CSA_STATE_BLOCK_SIZE, 2 * CSA_MAIN_OUT_DIM], pl.FP32
        ],
        compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
        hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
        idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
        idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
        idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
        inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
        inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
        inner_ape: pl.Tensor[[CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
        inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
        inner_compress_state: pl.Tensor[
            [INNER_STATE_POOL_BLOCKS, INNER_STATE_BLOCK_SIZE, 2 * INNER_OUT_DIM], pl.FP32
        ],
        inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
        kv_cache: pl.Tensor[[ORI_CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
        ori_block_table: pl.Tensor[[ORI_TABLE_BLOCKS], pl.INT32],
        ori_slot_mapping: pl.Tensor[[T], pl.INT64],
        cmp_kv: pl.Tensor[
            [CSA_CMP_CACHE_BLOCKS, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
        ],
        cmp_block_table: pl.Tensor[[CMP_TABLE_BLOCKS], pl.INT32],
        idx_kv_cache: pl.Tensor[
            [IDX_CACHE_BLOCKS, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8
        ],
        idx_kv_scale: pl.Tensor[
            [IDX_CACHE_BLOCKS, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32
        ],
        idx_block_table: pl.Tensor[[IDX_TABLE_BLOCKS], pl.INT32],
        position_ids: pl.Tensor[[T], pl.INT32],
        cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
        idx_slot_mapping: pl.Tensor[[T], pl.INT64],
        state_slot_mapping: pl.Tensor[[T], pl.INT64],
        inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
        attn_sink: pl.Tensor[[H], pl.FP32],
        wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
        wo_b_scale: pl.Tensor[[D], pl.FP32],
        x_out: pl.Tensor[[T, HC_MULT, D], pl.FP32],
        num_tokens: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[T, HC_MULT, D], pl.FP32]:
        return x_out


@pl.jit.inline(auto_scope=False)
def _prefill_layer_tile(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
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
    hca_cmp_wkv: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    hca_compress_state: pl.Tensor[
        [HCA_STATE_POOL_BLOCKS, HCA_STATE_BLOCK_SIZE, 2 * HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    csa_compress_state: pl.Tensor[
        [CSA_STATE_POOL_BLOCKS, CSA_STATE_BLOCK_SIZE, 2 * CSA_MAIN_OUT_DIM], pl.FP32
    ],
    csa_compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.Tensor[
        [INNER_STATE_POOL_BLOCKS, INNER_STATE_BLOCK_SIZE, 2 * INNER_OUT_DIM], pl.FP32
    ],
    csa_inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[ORI_CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[ORI_TABLE_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    hca_cmp_kv: pl.Tensor[
        [HCA_CMP_CACHE_BLOCKS, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
    ],
    csa_cmp_kv: pl.Tensor[
        [CSA_CMP_CACHE_BLOCKS, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
    ],
    hca_cmp_block_table: pl.Tensor[[CMP_TABLE_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[CMP_TABLE_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Tensor[
        [IDX_CACHE_BLOCKS, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8
    ],
    idx_kv_scale: pl.Tensor[
        [IDX_CACHE_BLOCKS, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32
    ],
    idx_block_table: pl.Tensor[[IDX_TABLE_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
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
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
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
    x_next: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, N_LOCAL, SIGNAL_PAD], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pl.InOut[pld.DistributedTensor[[N_RANKS, N_LOCAL, SIGNAL_PAD], pl.INT32]],
    consumed: pl.InOut[pld.DistributedTensor[[N_RANKS, SIGNAL_PAD], pl.INT32]],
    layer_id: pl.Scalar[pl.INT32],
    valid_n: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, HC_MULT, D], pl.FP32]:
    x_attn = pl.create_tensor([TOK_TILE, HC_MULT, D], dtype=pl.FP32)
    with pl.scope():
        if LAYER_ID < 2:
            prefill_attention_swa(
                x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
                freqs_cos, freqs_sin,
                kv_cache, ori_block_table, ori_slot_mapping,
                position_ids,
                attn_sink, wo_a, wo_b, wo_b_scale,
                x_attn, valid_n,
            )
        elif LAYER_ID % 2 == 1:
            prefill_attention_hca(
                x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
                freqs_cos, freqs_sin,
                hca_cmp_wkv, hca_cmp_wgate, hca_cmp_ape, hca_cmp_norm_w,
                hca_compress_state, hca_compress_state_block_table,
                kv_cache, ori_slot_mapping, ori_block_table,
                hca_cmp_kv, hca_cmp_block_table,
                position_ids, hca_cmp_slot_mapping, hca_state_slot_mapping,
                attn_sink, wo_a, wo_b, wo_b_scale,
                x_attn, valid_n,
            )
        else:
            prefill_attention_csa(
                x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
                freqs_cos, freqs_sin,
                csa_cmp_wkv, csa_cmp_wgate, csa_cmp_ape, csa_cmp_norm_w,
                csa_compress_state, csa_compress_state_block_table,
                csa_hadamard_idx,
                csa_idx_wq_b, csa_idx_wq_b_scale, csa_weights_proj,
                csa_inner_wkv, csa_inner_wgate, csa_inner_ape, csa_inner_norm_w,
                csa_inner_compress_state, csa_inner_compress_state_block_table,
                kv_cache, ori_block_table, ori_slot_mapping,
                csa_cmp_kv, csa_cmp_block_table,
                idx_kv_cache, idx_kv_scale, idx_block_table,
                position_ids, csa_cmp_slot_mapping, csa_idx_slot_mapping,
                csa_state_slot_mapping, csa_inner_state_slot_mapping,
                attn_sink, wo_a, wo_b, wo_b_scale,
                x_attn, valid_n,
            )

    with pl.scope():
        prefill_moe(
            x_attn,
            hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
            norm_w, gate_w, gate_bias, tid2eid, input_ids,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale,
            x_next,
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived, consumed,
            layer_id, valid_n, my_rank, moe_epoch,
        )
    return x_next


@pl.jit(auto_scope=False)
def prefill_layer_core(
    x_hc: pl.Tensor[[USER_BATCH, T, HC_MULT, D], pl.FP32],
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
    hca_cmp_wkv: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[pl.Tensor[
        [HCA_STATE_POOL_BLOCKS, HCA_STATE_BLOCK_SIZE, 2 * HCA_MAIN_OUT_DIM], pl.FP32
    ]],
    hca_compress_state_block_table: pl.Tensor[
        [USER_BATCH, HCA_STATE_MAX_BLOCKS], pl.INT32
    ],
    csa_cmp_wkv: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[pl.Tensor[
        [CSA_STATE_POOL_BLOCKS, CSA_STATE_BLOCK_SIZE, 2 * CSA_MAIN_OUT_DIM], pl.FP32
    ]],
    csa_compress_state_block_table: pl.Tensor[
        [USER_BATCH, CSA_STATE_MAX_BLOCKS], pl.INT32
    ],
    csa_hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[pl.Tensor[
        [INNER_STATE_POOL_BLOCKS, INNER_STATE_BLOCK_SIZE, 2 * INNER_OUT_DIM], pl.FP32
    ]],
    csa_inner_compress_state_block_table: pl.Tensor[
        [USER_BATCH, INNER_STATE_MAX_BLOCKS], pl.INT32
    ],
    kv_cache: pl.InOut[pl.Tensor[[ORI_CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[USER_BATCH, ORI_TABLE_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[USER_BATCH, T], pl.INT64],
    hca_cmp_kv: pl.InOut[pl.Tensor[
        [HCA_CMP_CACHE_BLOCKS, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
    ]],
    csa_cmp_kv: pl.InOut[pl.Tensor[
        [CSA_CMP_CACHE_BLOCKS, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
    ]],
    hca_cmp_block_table: pl.Tensor[[USER_BATCH, CMP_TABLE_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[USER_BATCH, CMP_TABLE_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[
        [IDX_CACHE_BLOCKS, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8
    ]],
    idx_kv_scale: pl.InOut[pl.Tensor[
        [IDX_CACHE_BLOCKS, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32
    ]],
    idx_block_table: pl.Tensor[[USER_BATCH, IDX_TABLE_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[USER_BATCH, T], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[USER_BATCH, T], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[USER_BATCH, T], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[USER_BATCH, T], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[USER_BATCH, T], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[USER_BATCH, T], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[USER_BATCH, T], pl.INT64],
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
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[USER_BATCH, T], pl.INT64],
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
    x_next: pl.Out[pl.Tensor[[USER_BATCH, T, HC_MULT, D], pl.FP32]],
    num_tokens_per_owner: pl.Tensor[[USER_BATCH, N_RANKS], pl.INT32],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, N_LOCAL, SIGNAL_PAD], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pl.InOut[pld.DistributedTensor[[N_RANKS, N_LOCAL, SIGNAL_PAD], pl.INT32]],
    consumed: pl.InOut[pld.DistributedTensor[[N_RANKS, SIGNAL_PAD], pl.INT32]],
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[USER_BATCH, T, HC_MULT, D], pl.FP32]:
    moe_epoch: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
    for request_id in pl.range(USER_BATCH):
        request_index = pl.cast(request_id, pl.INDEX)
        valid_n: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
        for owner_rank in pl.range(N_RANKS):
            valid_n = pl.max(
                valid_n,
                pl.read(num_tokens_per_owner, [request_id, owner_rank]),
            )
        if valid_n > 0:
            moe_epoch = moe_epoch + pl.cast(1, pl.INT32)

            x_hc_profile = pl.slice(
                x_hc, [1, T, HC_MULT, D], [request_index, 0, 0, 0]
            )
            x_hc_request = pl.reshape(x_hc_profile, [T, HC_MULT, D])
            ori_slot_profile = pl.slice(
                ori_slot_mapping, [1, T], [request_index, 0]
            )
            ori_slot_request = pl.reshape(ori_slot_profile, [T])
            position_profile = pl.slice(
                position_ids, [1, T], [request_index, 0]
            )
            position_request = pl.reshape(position_profile, [T])
            input_ids_profile = pl.slice(
                input_ids, [1, T], [request_index, 0]
            )
            input_ids_request = pl.reshape(input_ids_profile, [T])
            hca_cmp_slot_profile = pl.slice(
                hca_cmp_slot_mapping, [1, T], [request_index, 0]
            )
            hca_cmp_slot_request = pl.reshape(hca_cmp_slot_profile, [T])
            hca_state_slot_profile = pl.slice(
                hca_state_slot_mapping, [1, T], [request_index, 0]
            )
            hca_state_slot_request = pl.reshape(hca_state_slot_profile, [T])
            csa_cmp_slot_profile = pl.slice(
                csa_cmp_slot_mapping, [1, T], [request_index, 0]
            )
            csa_cmp_slot_request = pl.reshape(csa_cmp_slot_profile, [T])
            csa_idx_slot_profile = pl.slice(
                csa_idx_slot_mapping, [1, T], [request_index, 0]
            )
            csa_idx_slot_request = pl.reshape(csa_idx_slot_profile, [T])
            csa_state_slot_profile = pl.slice(
                csa_state_slot_mapping, [1, T], [request_index, 0]
            )
            csa_state_slot_request = pl.reshape(csa_state_slot_profile, [T])
            inner_state_slot_profile = pl.slice(
                csa_inner_state_slot_mapping, [1, T], [request_index, 0]
            )
            inner_state_slot_request = pl.reshape(inner_state_slot_profile, [T])

            ori_tables = pl.reshape(ori_block_table, [USER_BATCH * ORI_TABLE_BLOCKS])
            ori_table = pl.slice(
                ori_tables, [ORI_TABLE_BLOCKS], [request_index * ORI_TABLE_BLOCKS]
            )
            hca_cmp_tables = pl.reshape(
                hca_cmp_block_table, [USER_BATCH * CMP_TABLE_BLOCKS]
            )
            hca_cmp_table = pl.slice(
                hca_cmp_tables, [CMP_TABLE_BLOCKS], [request_index * CMP_TABLE_BLOCKS]
            )
            csa_cmp_tables = pl.reshape(
                csa_cmp_block_table, [USER_BATCH * CMP_TABLE_BLOCKS]
            )
            csa_cmp_table = pl.slice(
                csa_cmp_tables, [CMP_TABLE_BLOCKS], [request_index * CMP_TABLE_BLOCKS]
            )
            idx_tables = pl.reshape(idx_block_table, [USER_BATCH * IDX_TABLE_BLOCKS])
            idx_table = pl.slice(
                idx_tables, [IDX_TABLE_BLOCKS], [request_index * IDX_TABLE_BLOCKS]
            )
            hca_state_tables = pl.reshape(
                hca_compress_state_block_table,
                [USER_BATCH * HCA_STATE_MAX_BLOCKS],
            )
            hca_state_table = pl.slice(
                hca_state_tables,
                [HCA_STATE_MAX_BLOCKS],
                [request_index * HCA_STATE_MAX_BLOCKS],
            )
            csa_state_tables = pl.reshape(
                csa_compress_state_block_table,
                [USER_BATCH * CSA_STATE_MAX_BLOCKS],
            )
            csa_state_table = pl.slice(
                csa_state_tables,
                [CSA_STATE_MAX_BLOCKS],
                [request_index * CSA_STATE_MAX_BLOCKS],
            )
            inner_state_tables = pl.reshape(
                csa_inner_compress_state_block_table,
                [USER_BATCH * INNER_STATE_MAX_BLOCKS],
            )
            inner_state_table = pl.slice(
                inner_state_tables,
                [INNER_STATE_MAX_BLOCKS],
                [request_index * INNER_STATE_MAX_BLOCKS],
            )
            x_next_profile = pl.slice(
                x_next,
                [1, T, HC_MULT, D],
                [request_index, 0, 0, 0],
            )
            x_next_request = pl.reshape(x_next_profile, [T, HC_MULT, D])

            _prefill_layer_tile(
                x_hc_request,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_w, wq_a, wq_b, wq_b_scale,
                wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin,
                hca_cmp_wkv, hca_cmp_wgate, hca_cmp_ape, hca_cmp_norm_w,
                hca_compress_state, hca_state_table,
                csa_cmp_wkv, csa_cmp_wgate, csa_cmp_ape, csa_cmp_norm_w,
                csa_compress_state, csa_state_table,
                csa_hadamard_idx, csa_idx_wq_b, csa_idx_wq_b_scale,
                csa_weights_proj, csa_inner_wkv, csa_inner_wgate,
                csa_inner_ape, csa_inner_norm_w,
                csa_inner_compress_state, inner_state_table,
                kv_cache, ori_table,
                ori_slot_request,
                hca_cmp_kv, csa_cmp_kv, hca_cmp_table, csa_cmp_table,
                idx_kv_cache, idx_kv_scale, idx_table,
                position_request,
                hca_cmp_slot_request, hca_state_slot_request,
                csa_cmp_slot_request, csa_idx_slot_request,
                csa_state_slot_request, inner_state_slot_request,
                attn_sink, wo_a, wo_b, wo_b_scale,
                hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
                norm_w, gate_w, gate_bias, tid2eid,
                input_ids_request,
                routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                routed_w2, routed_w2_scale,
                shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
                shared_w2, shared_w2_scale,
                x_next_request,
                recv_meta, recv_x, recv_aux, recv_route,
                arrived, data_arrived, routed_y_buf, combine_arrived, consumed,
                layer_id, valid_n, my_rank, moe_epoch,
            )

    if moe_epoch > 0:
        final_request_profile = pl.slice(
            x_next,
            [1, T, HC_MULT, D],
            [USER_BATCH - 1, 0, 0, 0],
        )
        final_request = pl.reshape(final_request_profile, [T, HC_MULT, D])
        clear_prefill_moe_signals(
            final_request,
            arrived,
            data_arrived,
            combine_arrived,
            consumed,
            my_rank,
            moe_epoch,
        )
    return x_next


@pl.jit.host
def l3_prefill_layer(
    x_hc: pl.Tensor[[N_RANKS, USER_BATCH, T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_wkv: pl.Tensor[[N_RANKS, HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[N_RANKS, HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[N_RANKS, HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[pl.Tensor[
        [N_RANKS, HCA_STATE_POOL_BLOCKS, HCA_STATE_BLOCK_SIZE, 2 * HCA_MAIN_OUT_DIM],
        pl.FP32,
    ]],
    hca_compress_state_block_table: pl.Tensor[
        [N_RANKS, USER_BATCH, HCA_STATE_MAX_BLOCKS], pl.INT32
    ],
    csa_cmp_wkv: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[
        pl.Tensor[[N_RANKS, CSA_STATE_POOL_BLOCKS, CSA_STATE_BLOCK_SIZE, 2 * CSA_MAIN_OUT_DIM], pl.FP32]
    ],
    csa_compress_state_block_table: pl.Tensor[
        [N_RANKS, USER_BATCH, CSA_STATE_MAX_BLOCKS], pl.INT32
    ],
    csa_hadamard_idx: pl.Tensor[[N_RANKS, IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[N_RANKS, Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[N_RANKS, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[N_RANKS, D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[N_RANKS, INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[N_RANKS, INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[N_RANKS, IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[
        pl.Tensor[[N_RANKS, INNER_STATE_POOL_BLOCKS, INNER_STATE_BLOCK_SIZE, 2 * INNER_OUT_DIM], pl.FP32]
    ],
    csa_inner_compress_state_block_table: pl.Tensor[
        [N_RANKS, USER_BATCH, INNER_STATE_MAX_BLOCKS], pl.INT32
    ],
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, ORI_CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[N_RANKS, USER_BATCH, ORI_TABLE_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[N_RANKS, USER_BATCH, T], pl.INT64],
    hca_cmp_kv: pl.InOut[
        pl.Tensor[
            [N_RANKS, HCA_CMP_CACHE_BLOCKS, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
        ]
    ],
    csa_cmp_kv: pl.InOut[
        pl.Tensor[
            [N_RANKS, CSA_CMP_CACHE_BLOCKS, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
        ]
    ],
    hca_cmp_block_table: pl.Tensor[[N_RANKS, USER_BATCH, CMP_TABLE_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[N_RANKS, USER_BATCH, CMP_TABLE_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[
        pl.Tensor[
            [N_RANKS, IDX_CACHE_BLOCKS, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8
        ]
    ],
    idx_kv_scale: pl.InOut[
        pl.Tensor[[N_RANKS, IDX_CACHE_BLOCKS, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]
    ],
    idx_block_table: pl.Tensor[[N_RANKS, USER_BATCH, IDX_TABLE_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[N_RANKS, USER_BATCH, T], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[N_RANKS, USER_BATCH, T], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[N_RANKS, USER_BATCH, T], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[N_RANKS, USER_BATCH, T], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[N_RANKS, USER_BATCH, T], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[N_RANKS, USER_BATCH, T], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[N_RANKS, USER_BATCH, T], pl.INT64],
    attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, USER_BATCH, T], pl.INT64],
    routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    x_next: pl.Out[pl.Tensor[[N_RANKS, USER_BATCH, T, HC_MULT, D], pl.FP32]],
    num_tokens_per_owner: pl.Tensor[[USER_BATCH, N_RANKS], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
):
    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, SIGNAL_PAD], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer(
        [N_RANKS, N_LOCAL, SIGNAL_PAD], dtype=pl.INT32
    )
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer(
        [N_RANKS, N_LOCAL, SIGNAL_PAD], dtype=pl.INT32
    )
    consumed_buf = pld.alloc_window_buffer([N_RANKS, SIGNAL_PAD], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, SIGNAL_PAD], dtype=pl.INT32)
        data_arrived = pld.window(
            data_arrived_buf, [N_RANKS, N_LOCAL, SIGNAL_PAD], dtype=pl.INT32
        )
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(
            combine_arrived_buf, [N_RANKS, N_LOCAL, SIGNAL_PAD], dtype=pl.INT32
        )
        consumed = pld.window(consumed_buf, [N_RANKS, SIGNAL_PAD], dtype=pl.INT32)
        prefill_layer_core(
            x_hc[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank], wq_a[rank], wq_b[rank], wq_b_scale[rank],
            wkv[rank], gamma_cq[rank], gamma_ckv[rank], freqs_cos[rank], freqs_sin[rank],
            hca_cmp_wkv[rank], hca_cmp_wgate[rank], hca_cmp_ape[rank], hca_cmp_norm_w[rank],
            hca_compress_state[rank], hca_compress_state_block_table[rank],
            csa_cmp_wkv[rank], csa_cmp_wgate[rank], csa_cmp_ape[rank], csa_cmp_norm_w[rank],
            csa_compress_state[rank], csa_compress_state_block_table[rank],
            csa_hadamard_idx[rank],
            csa_idx_wq_b[rank], csa_idx_wq_b_scale[rank], csa_weights_proj[rank],
            csa_inner_wkv[rank], csa_inner_wgate[rank], csa_inner_ape[rank], csa_inner_norm_w[rank],
            csa_inner_compress_state[rank],
            csa_inner_compress_state_block_table[rank],
            kv_cache[rank], ori_block_table[rank], ori_slot_mapping[rank],
            hca_cmp_kv[rank], csa_cmp_kv[rank],
            hca_cmp_block_table[rank], csa_cmp_block_table[rank],
            idx_kv_cache[rank], idx_kv_scale[rank], idx_block_table[rank],
            position_ids[rank],
            hca_cmp_slot_mapping[rank], hca_state_slot_mapping[rank],
            csa_cmp_slot_mapping[rank], csa_idx_slot_mapping[rank],
            csa_state_slot_mapping[rank], csa_inner_state_slot_mapping[rank],
            attn_sink[rank], wo_a[rank], wo_b[rank], wo_b_scale[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            norm_w[rank], gate_w[rank], gate_bias[rank], tid2eid[rank], input_ids[rank],
            routed_w1[rank], routed_w1_scale[rank], routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank], shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            x_next[rank],
            num_tokens_per_owner,
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived, consumed,
            layer_id, rank,
            device=rank,
        )


HOST_TENSOR_ORDER = (
    "x_hc",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm_w",
    "wq_a",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "gamma_cq",
    "gamma_ckv",
    "freqs_cos",
    "freqs_sin",
    "hca_cmp_wkv",
    "hca_cmp_wgate",
    "hca_cmp_ape",
    "hca_cmp_norm_w",
    "hca_compress_state",
    "hca_compress_state_block_table",
    "csa_cmp_wkv",
    "csa_cmp_wgate",
    "csa_cmp_ape",
    "csa_cmp_norm_w",
    "csa_compress_state",
    "csa_compress_state_block_table",
    "csa_hadamard_idx",
    "csa_idx_wq_b",
    "csa_idx_wq_b_scale",
    "csa_weights_proj",
    "csa_inner_wkv",
    "csa_inner_wgate",
    "csa_inner_ape",
    "csa_inner_norm_w",
    "csa_inner_compress_state",
    "csa_inner_compress_state_block_table",
    "kv_cache",
    "ori_block_table",
    "ori_slot_mapping",
    "hca_cmp_kv",
    "csa_cmp_kv",
    "hca_cmp_block_table",
    "csa_cmp_block_table",
    "idx_kv_cache",
    "idx_kv_scale",
    "idx_block_table",
    "position_ids",
    "hca_cmp_slot_mapping",
    "hca_state_slot_mapping",
    "csa_cmp_slot_mapping",
    "csa_idx_slot_mapping",
    "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_w",
    "gate_w",
    "gate_bias",
    "tid2eid",
    "input_ids",
    "routed_w1",
    "routed_w1_scale",
    "routed_w3",
    "routed_w3_scale",
    "routed_w2",
    "routed_w2_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w3",
    "shared_w3_scale",
    "shared_w2",
    "shared_w2_scale",
    "x_next",
    "num_tokens_per_owner",
)


# ---------------------------------------------------------------------------
# Host-side fixed metadata builder, tensor specs, and golden reference.
# ---------------------------------------------------------------------------

_KIND_BUILDER = {
    "swa": build_swa_attention_tensor_specs,
    "hca": build_hca_attention_tensor_specs,
    "csa": build_csa_attention_tensor_specs,
}

# Child-local token-metadata tensors gathered for the fixed tile.
_TOKEN_META_NAMES = {
    "position_ids", "ori_slot_mapping",
    "cmp_slot_mapping", "state_slot_mapping", "idx_slot_mapping", "inner_state_slot_mapping",
}
# Child cache/state pools plus request-local table views (persist across tiles).
_CACHE_STATE_NAMES = {
    "kv_cache", "block_table", "ori_block_table", "cmp_kv", "cmp_block_table",
    "idx_kv_cache", "idx_kv_scale", "idx_block_table",
    "compress_state", "compress_state_block_table",
    "inner_compress_state", "inner_compress_state_block_table",
}

# Global cache/state pools and child-local mappings.
_PACKED_CACHE_SPECS = {
    "kv_cache": "kv_cache",
    "ori_block_table": "ori_block_table",
    "hca_cmp_kv": ("hca", "cmp_kv"),
    "csa_cmp_kv": ("csa", "cmp_kv"),
    "hca_cmp_block_table": ("hca", "cmp_block_table"),
    "csa_cmp_block_table": ("csa", "cmp_block_table"),
    "idx_kv_cache": "idx_kv_cache",
    "idx_kv_scale": "idx_kv_scale",
    "idx_block_table": "idx_block_table",
    "hca_compress_state": ("hca", "compress_state"),
    "hca_compress_state_block_table": ("hca", "compress_state_block_table"),
    "csa_compress_state": ("csa", "compress_state"),
    "csa_compress_state_block_table": ("csa", "compress_state_block_table"),
    "csa_inner_compress_state": ("csa", "inner_compress_state"),
    "csa_inner_compress_state_block_table": ("csa", "inner_compress_state_block_table"),
}

_HISTORY_CACHE_NAMES = {
    "kv_cache", "hca_cmp_kv", "csa_cmp_kv", "idx_kv_cache", "idx_kv_scale",
    "hca_compress_state", "csa_compress_state", "csa_inner_compress_state",
}


def _child_to_packed(kind, child_name):
    """Map a child-local cache/state name to the layer-level tensor name."""
    if child_name in ("block_table", "ori_block_table"):
        return "ori_block_table"
    if child_name == "cmp_block_table":
        return ("hca_" if kind == "hca" else "csa_") + child_name
    if child_name == "cmp_kv":
        return ("hca_" if kind == "hca" else "csa_") + child_name
    if child_name in ("kv_cache", "idx_kv_cache", "idx_kv_scale", "idx_block_table"):
        return child_name
    prefix = "hca_" if kind == "hca" else "csa_"
    return prefix + child_name


def _spec_value(spec, torch):
    init_value = getattr(spec, "init_value", None)
    if callable(init_value):
        return init_value()
    if init_value is not None:
        return init_value.clone() if hasattr(init_value, "clone") else init_value
    return torch.zeros(spec.shape, dtype=spec.dtype)


def _attention_kind_for_layer(layer_id):
    ratio = MODEL_CONFIG.compress_ratios[layer_id]
    if ratio == 0:
        return "swa"
    if ratio == 128:
        return "hca"
    if ratio == 4:
        return "csa"
    raise ValueError(f"unsupported DeepSeek V4 attention compress ratio {ratio} at layer {layer_id}")


def _request_fixture(start_pos, valid_tokens, request_id, torch):
    """Production-partitioned token mappings and tables for one request."""
    pos = torch.arange(start_pos, start_pos + T, dtype=torch.int64)
    active = torch.arange(T) < valid_tokens

    def partition(physical_blocks):
        begin = request_id * physical_blocks // USER_BATCH
        end = (request_id + 1) * physical_blocks // USER_BATCH
        if end <= begin:
            raise ValueError(
                f"pool with {physical_blocks} blocks cannot isolate B{USER_BATCH}"
            )
        return begin, end - begin

    def paged_rows(logical_rows, physical_blocks, storage_block_size):
        block_base, block_count = partition(physical_blocks)
        logical_blocks = logical_rows // storage_block_size
        return (
            (block_base + logical_blocks % block_count) * storage_block_size
            + logical_rows % storage_block_size
        )

    def slots(logical_rows, physical_blocks, storage_block_size, mask):
        rows = paged_rows(logical_rows, physical_blocks, storage_block_size)
        return torch.where(mask, rows, torch.full_like(rows, -1))

    def compressed_slots(ratio, physical_blocks, storage_block_size):
        result = torch.full((T,), -1, dtype=torch.int64)
        mask = active & (((pos + 1) % ratio) == 0)
        logical_rows = ((pos[mask] + 1) // ratio) - 1
        result[mask] = paged_rows(
            logical_rows, physical_blocks, storage_block_size
        )
        return result

    def block_table(capacity, physical_blocks):
        block_base, block_count = partition(physical_blocks)
        return (
            torch.arange(capacity, dtype=torch.int32) % block_count + block_base
        )

    addressable = active & (pos >= 0) & (pos < MAX_SEQ_LEN)
    return {
        "position_ids": pos.to(torch.int32),
        "ori_slot_mapping": slots(
            pos, ORI_CACHE_BLOCKS, BLOCK_SIZE, addressable
        ),
        "hca_cmp_slot_mapping": compressed_slots(
            HCA_COMPRESS_RATIO, HCA_CMP_CACHE_BLOCKS, HCA_CMP_STORAGE_BLOCK_SIZE
        ),
        "hca_state_slot_mapping": slots(
            pos,
            HCA_STATE_POOL_BLOCKS,
            HCA_STATE_BLOCK_SIZE,
            addressable & ((pos // HCA_STATE_BLOCK_SIZE) < HCA_STATE_MAX_BLOCKS),
        ),
        "csa_cmp_slot_mapping": compressed_slots(
            CSA_COMPRESS_RATIO, CSA_CMP_CACHE_BLOCKS, CSA_CMP_STORAGE_BLOCK_SIZE
        ),
        "csa_idx_slot_mapping": compressed_slots(
            CSA_COMPRESS_RATIO, IDX_CACHE_BLOCKS, CSA_CMP_STORAGE_BLOCK_SIZE
        ),
        "csa_state_slot_mapping": slots(
            pos,
            CSA_STATE_POOL_BLOCKS,
            CSA_STATE_BLOCK_SIZE,
            addressable & ((pos // CSA_STATE_BLOCK_SIZE) < CSA_STATE_MAX_BLOCKS),
        ),
        "csa_inner_state_slot_mapping": slots(
            pos,
            INNER_STATE_POOL_BLOCKS,
            INNER_STATE_BLOCK_SIZE,
            addressable & ((pos // INNER_STATE_BLOCK_SIZE) < INNER_STATE_MAX_BLOCKS),
        ),
        "ori_block_table": block_table(ORI_TABLE_BLOCKS, ORI_CACHE_BLOCKS),
        "hca_cmp_block_table": block_table(
            CMP_TABLE_BLOCKS, HCA_CMP_CACHE_BLOCKS
        ),
        "csa_cmp_block_table": block_table(
            CMP_TABLE_BLOCKS, CSA_CMP_CACHE_BLOCKS
        ),
        "idx_block_table": block_table(IDX_TABLE_BLOCKS, IDX_CACHE_BLOCKS),
        "hca_compress_state_block_table": block_table(
            HCA_STATE_MAX_BLOCKS, HCA_STATE_POOL_BLOCKS
        ),
        "csa_compress_state_block_table": block_table(
            CSA_STATE_MAX_BLOCKS, CSA_STATE_POOL_BLOCKS
        ),
        "csa_inner_compress_state_block_table": block_table(
            INNER_STATE_MAX_BLOCKS, INNER_STATE_POOL_BLOCKS
        ),
    }


def build_tensor_specs(layer_id=2, start_pos=0, num_tokens=T):
    """Build one production-partitioned B1 or B4 single-layer fixture."""
    import torch
    from golden import ScalarSpec, TensorSpec

    if not 1 <= num_tokens <= T:
        raise ValueError(f"num_tokens must be in [1, {T}], got {num_tokens}")
    kind = _attention_kind_for_layer(layer_id)
    total_tokens = T

    def kind_specs(build_fn):
        return {
            s.name: s
            for s in build_fn(start_pos=start_pos, num_tokens=num_tokens)
            if isinstance(s, TensorSpec)
        }

    swa = kind_specs(build_swa_attention_tensor_specs)
    hca = kind_specs(build_hca_attention_tensor_specs)
    csa = kind_specs(build_csa_attention_tensor_specs)
    active = {"swa": swa, "hca": hca, "csa": csa}[kind]
    src_by_kind = {"swa": swa, "hca": hca, "csa": csa}

    def ranked_init(src):
        def init():
            return torch.stack([_spec_value(src, torch) for _ in range(N_RANKS)], dim=0).contiguous()
        return init

    def replicate(values):
        def init():
            return torch.stack([values.clone() for _ in range(N_RANKS)], dim=0).contiguous()
        return init

    # Per-rank weight tensors (same selection as prefill_layer.py minus token +
    # cache/state tensors, which are rebuilt for the standalone layer below).
    weight_specs = [
        ("hc_attn_fn", active["hc_attn_fn"]),
        ("hc_attn_scale", active["hc_attn_scale"]),
        ("hc_attn_base", active["hc_attn_base"]),
        ("attn_norm_w", active["attn_norm_w"]),
        ("wq_a", active["wq_a"]),
        ("wq_b", active["wq_b"]),
        ("wq_b_scale", active["wq_b_scale"]),
        ("wkv", active["wkv"]),
        ("gamma_cq", active["gamma_cq"]),
        ("gamma_ckv", active["gamma_ckv"]),
        ("freqs_cos", active["freqs_cos"]),
        ("freqs_sin", active["freqs_sin"]),
        ("hca_cmp_wkv", hca["cmp_wkv"]),
        ("hca_cmp_wgate", hca["cmp_wgate"]),
        ("hca_cmp_ape", hca["cmp_ape"]),
        ("hca_cmp_norm_w", hca["cmp_norm_w"]),
        ("csa_cmp_wkv", csa["cmp_wkv"]),
        ("csa_cmp_wgate", csa["cmp_wgate"]),
        ("csa_cmp_ape", csa["cmp_ape"]),
        ("csa_cmp_norm_w", csa["cmp_norm_w"]),
        ("csa_hadamard_idx", csa["hadamard_idx"]),
        ("csa_idx_wq_b", csa["idx_wq_b"]),
        ("csa_idx_wq_b_scale", csa["idx_wq_b_scale"]),
        ("csa_weights_proj", csa["idx_weights_proj"]),
        ("csa_inner_wkv", csa["inner_wkv"]),
        ("csa_inner_wgate", csa["inner_wgate"]),
        ("csa_inner_ape", csa["inner_ape"]),
        ("csa_inner_norm_w", csa["inner_norm_w"]),
        ("attn_sink", active["attn_sink"]),
        ("wo_a", active["wo_a"]),
        ("wo_b", active["wo_b"]),
        ("wo_b_scale", active["wo_b_scale"]),
    ]

    tensor_specs = [TensorSpec(name, [N_RANKS, *src.shape], src.dtype, init_value=ranked_init(src))
                    for name, src in weight_specs]

    requests = [
        _request_fixture(start_pos, num_tokens, request_id, torch)
        for request_id in range(USER_BATCH)
    ]
    meta = {
        name: torch.stack([request[name] for request in requests], dim=0)
        for name in requests[0]
    }
    hca_mapping_names = ("hca_cmp_slot_mapping", "hca_state_slot_mapping")
    csa_mapping_names = (
        "csa_cmp_slot_mapping",
        "csa_idx_slot_mapping",
        "csa_state_slot_mapping",
        "csa_inner_state_slot_mapping",
    )
    if kind != "hca":
        for name in hca_mapping_names:
            meta[name].fill_(-1)
    if kind != "csa":
        for name in csa_mapping_names:
            meta[name].fill_(-1)

    def init_x_hc():
        values = (
            torch.rand(
                N_RANKS, USER_BATCH, T, HC_MULT, D, dtype=torch.float32
            )
            - 0.5
        ) / 10.0
        request_offsets = torch.arange(USER_BATCH, dtype=torch.float32).view(
            1, USER_BATCH, 1, 1, 1
        )
        return (values + request_offsets / 100.0).contiguous()

    def init_input_ids():
        ids = torch.empty(N_RANKS, USER_BATCH, T, dtype=torch.int64)
        for rank in range(N_RANKS):
            for request_id in range(USER_BATCH):
                ids[rank, request_id] = (
                    torch.arange(T, dtype=torch.int64)
                    + rank
                    + request_id * (T + 1)
                ) % VOCAB
        if num_tokens < T:
            ids[:, :, num_tokens:] = -1
        return ids.contiguous()

    tensor_specs.append(TensorSpec("x_hc", [N_RANKS, USER_BATCH, total_tokens, HC_MULT, D], torch.float32, init_value=init_x_hc))
    tensor_specs.append(TensorSpec("input_ids", [N_RANKS, USER_BATCH, total_tokens], torch.int64, init_value=init_input_ids))
    tensor_specs.append(TensorSpec("position_ids", [N_RANKS, USER_BATCH, total_tokens], torch.int32,
                                   init_value=replicate(meta["position_ids"])))
    tensor_specs.append(TensorSpec("ori_slot_mapping", [N_RANKS, USER_BATCH, total_tokens], torch.int64,
                                   init_value=replicate(meta["ori_slot_mapping"])))
    for name in ("hca_cmp_slot_mapping", "hca_state_slot_mapping", "csa_cmp_slot_mapping",
                 "csa_idx_slot_mapping", "csa_state_slot_mapping", "csa_inner_state_slot_mapping"):
        tensor_specs.append(TensorSpec(name, [N_RANKS, USER_BATCH, total_tokens], torch.int64, init_value=replicate(meta[name])))

    def resolve_cache_src(packed_name, info):
        """Resolve (source spec, source kind, child-local name) for a layer cache."""
        if isinstance(info, tuple):
            sk, cn = info
            return src_by_kind[sk][cn], sk, cn
        cn = info
        if cn == "ori_block_table":
            return (active.get("ori_block_table") or swa["block_table"]), kind, cn
        if cn in ("cmp_kv", "cmp_block_table"):
            return (active.get(cn) or csa[cn]), kind, cn
        if cn in ("idx_kv_cache", "idx_kv_scale", "idx_block_table"):
            return csa[cn], kind, cn
        return active[cn], kind, cn  # kv_cache

    # Fixed-capacity cache/state pools and request-partitioned tables.
    for packed_name, info in _PACKED_CACHE_SPECS.items():
        src, _, _ = resolve_cache_src(packed_name, info)
        if packed_name not in _HISTORY_CACHE_NAMES:
            value = meta[packed_name]
        else:
            value = _spec_value(src, torch)
            expanded_blocks = {
                "hca_compress_state": HCA_STATE_POOL_BLOCKS,
                "csa_compress_state": CSA_STATE_POOL_BLOCKS,
                "csa_inner_compress_state": INNER_STATE_POOL_BLOCKS,
            }.get(packed_name)
            if expanded_blocks is not None and value.shape[0] < expanded_blocks:
                expanded = torch.zeros((expanded_blocks, *value.shape[1:]), dtype=value.dtype)
                expanded[: value.shape[0]].copy_(value)
                value = expanded

        def make_init(value=value):
            def init():
                return torch.stack([value.clone() for _ in range(N_RANKS)], dim=0).contiguous()

            return init

        tensor_specs.append(
            TensorSpec(
                packed_name,
                [N_RANKS, *value.shape],
                src.dtype,
                init_value=make_init(),
            )
        )

    # MoE weight tensors (per rank). tid2eid keeps its hash-table init.
    for spec in build_moe_tensor_specs(layer_id=layer_id):
        if not isinstance(spec, TensorSpec) or spec.name in {"x_hc", "x_next", "input_ids"}:
            continue
        if spec.name == "tid2eid":
            def init_tid2eid(spec=spec):
                _, vocab, topk = spec.shape
                ids = torch.arange(vocab, dtype=torch.int64).view(vocab, 1)
                ks = torch.arange(topk, dtype=torch.int64).view(1, topk)
                table = ((ids * topk + ks) % N_EXPERTS_GLOBAL).to(dtype=spec.dtype)
                return table.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

            tensor_specs.append(TensorSpec(spec.name, spec.shape, spec.dtype, init_value=init_tid2eid))
        else:
            tensor_specs.append(spec)

    tensor_specs.append(TensorSpec(
        "x_next", [N_RANKS, USER_BATCH, total_tokens, HC_MULT, D], torch.float32
    ))

    def init_num_tokens_per_owner():
        return torch.full(
            (USER_BATCH, N_RANKS), num_tokens, dtype=torch.int32
        )

    tensor_specs.append(TensorSpec(
        "num_tokens_per_owner",
        [USER_BATCH, N_RANKS],
        torch.int32,
        init_value=init_num_tokens_per_owner,
    ))

    # Keep static weight parameters device-resident (child_memory), sharded per
    # rank. Cache/state/table tensors remain host tensors for output validation.
    RESIDENT_WEIGHT_NAMES = frozenset([
        # Attention core weights + RoPE tables
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
        "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
        "freqs_cos", "freqs_sin",
        # HCA / CSA compressor + indexer weights (states/block tables excluded)
        "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w",
        "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape", "csa_cmp_norm_w",
        "csa_hadamard_idx", "csa_idx_wq_b", "csa_idx_wq_b_scale", "csa_weights_proj",
        "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
        # Attention output projection
        "attn_sink", "wo_a", "wo_b", "wo_b_scale",
        # MoE FFN / gate / experts + static route table
        "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
        "gate_w", "gate_bias", "tid2eid",
        "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
        "routed_w2", "routed_w2_scale",
        "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
        "shared_w2", "shared_w2_scale",
    ])
    for spec in tensor_specs:
        if spec.name in RESIDENT_WEIGHT_NAMES:
            spec.resident = "stacked"

    tensor_by_name = {spec.name: spec for spec in tensor_specs}
    missing = [name for name in HOST_TENSOR_ORDER if name not in tensor_by_name]
    if missing:
        raise ValueError(f"missing prefill layer tensor specs: {missing}")
    return [tensor_by_name[name] for name in HOST_TENSOR_ORDER] + [
        ScalarSpec("layer_id", torch.int32, layer_id),
    ]


def golden_prefill_layer(tensors):
    """Torch reference for every serialized request and mutable shared pool."""
    import torch
    from golden import TensorSpec

    layer_id = int(tensors["layer_id"])
    kind = _attention_kind_for_layer(layer_id)
    valid_per_request = tensors["num_tokens_per_owner"].max(dim=1).values
    start_pos = int(tensors["position_ids"][0, 0, 0])

    # Map child-local attention tensor names to layer-level names.
    mapped = dict(tensors)
    if kind == "swa":
        mapped["block_table"] = tensors["ori_block_table"]
        attention_golden = golden_prefill_attention_swa
    elif kind == "hca":
        mapped.update({
            "cmp_wkv": tensors["hca_cmp_wkv"], "cmp_wgate": tensors["hca_cmp_wgate"],
            "cmp_ape": tensors["hca_cmp_ape"], "cmp_norm_w": tensors["hca_cmp_norm_w"],
            "compress_state": tensors["hca_compress_state"],
            "compress_state_block_table": tensors["hca_compress_state_block_table"],
            "cmp_slot_mapping": tensors["hca_cmp_slot_mapping"], "state_slot_mapping": tensors["hca_state_slot_mapping"],
        })
        attention_golden = golden_prefill_attention_hca
    else:
        mapped.update({
            "cmp_wkv": tensors["csa_cmp_wkv"], "cmp_wgate": tensors["csa_cmp_wgate"],
            "cmp_ape": tensors["csa_cmp_ape"], "cmp_norm_w": tensors["csa_cmp_norm_w"],
            "compress_state": tensors["csa_compress_state"],
            "compress_state_block_table": tensors["csa_compress_state_block_table"],
            "hadamard_idx": tensors["csa_hadamard_idx"], "idx_wq_b": tensors["csa_idx_wq_b"],
            "idx_wq_b_scale": tensors["csa_idx_wq_b_scale"], "idx_weights_proj": tensors["csa_weights_proj"],
            "inner_wkv": tensors["csa_inner_wkv"], "inner_wgate": tensors["csa_inner_wgate"],
            "inner_ape": tensors["csa_inner_ape"], "inner_norm_w": tensors["csa_inner_norm_w"],
            "inner_compress_state": tensors["csa_inner_compress_state"],
            "inner_compress_state_block_table": tensors["csa_inner_compress_state_block_table"],
            "cmp_slot_mapping": tensors["csa_cmp_slot_mapping"], "idx_slot_mapping": tensors["csa_idx_slot_mapping"],
            "state_slot_mapping": tensors["csa_state_slot_mapping"],
            "inner_state_slot_mapping": tensors["csa_inner_state_slot_mapping"],
        })
        attention_golden = golden_prefill_attention_csa

    attn_specs = _KIND_BUILDER[kind](
        start_pos=start_pos,
        num_tokens=int(valid_per_request[0]),
    )
    x_next = tensors["x_next"]

    for request_id in range(USER_BATCH):
        valid = int(valid_per_request[request_id])
        x_attn = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
        for rank in range(N_RANKS):
            attn_tensors = {}
            for spec in attn_specs:
                if not isinstance(spec, TensorSpec):
                    continue
                name = spec.name
                if name == "x_out":
                    value = x_attn[rank]
                elif name == "x_hc":
                    value = tensors["x_hc"][rank, request_id]
                elif name in _TOKEN_META_NAMES:
                    value = mapped[name][rank, request_id]
                elif name in _CACHE_STATE_NAMES:
                    packed_name = _child_to_packed(kind, name)
                    packed = tensors[packed_name]
                    value = (
                        packed[rank]
                        if packed_name in _HISTORY_CACHE_NAMES
                        else packed[rank, request_id]
                    )
                else:
                    value = mapped[name][rank]
                attn_tensors[name] = value
            attn_tensors["num_tokens"] = valid
            attention_golden(attn_tensors)

        moe_tensors = dict(tensors)
        moe_tensors["x_hc"] = x_attn
        moe_tensors["input_ids"] = tensors["input_ids"][:, request_id]
        moe_tensors["num_tokens"] = valid
        x_next_request = torch.zeros(
            N_RANKS, T, HC_MULT, D, dtype=torch.float32
        )
        moe_tensors["x_next"] = x_next_request
        golden_moe(moe_tensors)
        x_next[:, request_id].copy_(x_next_request)


if __name__ == "__main__":
    import argparse

    from golden import mapped_pool_ratio_allclose, ratio_reldiff, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--ep", type=int, default=N_RANKS, choices=[2, 4, 8],
                        help="EP world size / rank count (parsed at import by moe)")
    parser.add_argument(
        "--dispatch-batch",
        type=int,
        default=USER_BATCH,
        choices=_DISPATCH_BATCH_CHOICES,
        help="Import-time request dispatch batch; default preserves B1.",
    )
    parser.add_argument("-d", "--device", type=str,
                        default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids; need at least {N_RANKS}")
    parser.add_argument("--layer-id", type=int, default=2,
                        help="Layer id selects attention by MODEL_CONFIG.compress_ratios[layer_id].")
    parser.add_argument("--start-pos", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=T)
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) >= N_RANKS, f"need at least {N_RANKS} devices, got {device_ids}"
    assert args.dispatch_batch == USER_BATCH, (
        "import-time dispatch batch must match parsed CLI value, got "
        f"{USER_BATCH} vs {args.dispatch_batch}"
    )
    assert args.layer_id == LAYER_ID, (
        "import-time layer id must match parsed CLI value, got "
        f"{LAYER_ID} vs {args.layer_id}"
    )

    result = run(
        fn=l3_prefill_layer,
        specs=build_tensor_specs(
            layer_id=args.layer_id,
            start_pos=args.start_pos,
            num_tokens=args.num_tokens,
        ),
        golden_fn=golden_prefill_layer,
        compile_only=args.compile_only,
        config=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:N_RANKS],
                num_sub_workers=0,
            ),
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
            ring_heap=PREFILL_RING_HEAP,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "x_next": ratio_reldiff(
                diff_thd=0.01,
                pct_thd=0.05,
                valid_rows=args.num_tokens,
                valid_axis=2,
            ),
            "kv_cache": mapped_pool_ratio_allclose(
                "ori_slot_mapping",
                mapping_shape=(N_RANKS, USER_BATCH, T),
                block_size=BLOCK_SIZE,
                leading_rank_axis=True,
                atol=1e-4,
                rtol=1.0 / 128,
            ),
            # CSA runs preserve the standalone child-kernel precision
            # contracts: sparse BF16 cache rows may differ by one ULP, C8 rows
            # by one LSB, and only a small fraction of recurrent-state values
            # cross the strict pointwise FP32 threshold.
            "csa_compress_state": mapped_pool_ratio_allclose(
                "csa_state_slot_mapping",
                mapping_shape=(N_RANKS, USER_BATCH, T),
                block_size=CSA_STATE_BLOCK_SIZE,
                leading_rank_axis=True,
                atol=1e-3,
                rtol=1e-3,
                max_error_ratio=0.005,
            ),
            "csa_inner_compress_state": mapped_pool_ratio_allclose(
                "csa_inner_state_slot_mapping",
                mapping_shape=(N_RANKS, USER_BATCH, T),
                block_size=INNER_STATE_BLOCK_SIZE,
                leading_rank_axis=True,
                atol=1e-3,
                rtol=1e-3,
                max_error_ratio=0.005,
            ),
            "hca_compress_state": mapped_pool_ratio_allclose(
                "hca_state_slot_mapping",
                mapping_shape=(N_RANKS, USER_BATCH, T),
                block_size=HCA_STATE_BLOCK_SIZE,
                leading_rank_axis=True,
                atol=1e-3,
                rtol=1e-3,
                max_error_ratio=0.005,
            ),
            "hca_cmp_kv": mapped_pool_ratio_allclose(
                "hca_cmp_slot_mapping",
                mapping_shape=(N_RANKS, USER_BATCH, T),
                block_size=HCA_CMP_STORAGE_BLOCK_SIZE,
                leading_rank_axis=True,
                atol=1e-4,
                rtol=1.0 / 128,
                max_error_ratio=0.005,
            ),
            "csa_cmp_kv": mapped_pool_ratio_allclose(
                "csa_cmp_slot_mapping",
                mapping_shape=(N_RANKS, USER_BATCH, T),
                block_size=CSA_CMP_STORAGE_BLOCK_SIZE,
                leading_rank_axis=True,
                atol=1e-4,
                rtol=1.0 / 128,
                max_error_ratio=0.005,
            ),
            "idx_kv_cache": mapped_pool_ratio_allclose(
                "csa_idx_slot_mapping",
                mapping_shape=(N_RANKS, USER_BATCH, T),
                block_size=CSA_CMP_STORAGE_BLOCK_SIZE,
                leading_rank_axis=True,
                atol=1,
                rtol=0,
                max_error_ratio=0.01,
            ),
            "idx_kv_scale": mapped_pool_ratio_allclose(
                "csa_idx_slot_mapping",
                mapping_shape=(N_RANKS, USER_BATCH, T),
                block_size=CSA_CMP_STORAGE_BLOCK_SIZE,
                leading_rank_axis=True,
                atol=1e-3,
                rtol=1e-3,
                max_error_ratio=0.01,
            ),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
