# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F401,F403,F405,F821
"""DeepSeek-V4 MTP kernels.

This file keeps the MTP tail in one reviewable module while preserving the
runtime boundaries as separate functions:

    hidden_states + prev_hidden_states
        -> mtp_projection_impl
        -> SWA decoder tail
        -> hc_head + shared-head norm
        -> local logits
        -> candidate_logits

``mtp_projection_impl`` mirrors the MTP-only prolog in the official
implementation:
``e_proj(enorm(hidden_states)) + h_proj(hnorm(prev_hidden_states))``.
The public validation entry is ``mtp_full_chain``; the other MTP functions are
kept as named boundaries so serving integration can wire forward and logits
separately later.
"""

import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, INT8_AMAX_EPS, INT8_SCALE_MAX
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
    build_tensor_specs as build_swa_tensor_specs,
    golden_attention_swa,
)
from hc_head import build_tensor_specs as build_hc_head_tensor_specs, golden_hc_head, hc_head
from moe import (
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    TOPK as MOE_TOPK,
    VOCAB as MOE_VOCAB,
    build_tensor_specs_ep1 as build_moe_ep1_tensor_specs,
    golden_moe_ep1,
    moe_ep1,
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

T_TILE = 8
LINEAR_T_TILE = 16
T_PAD = ((T + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
D_CHUNK = 128
OUT_CHUNK = 128
D_BLOCKS = D // D_CHUNK
OUT_BLOCKS = D // OUT_CHUNK
QUANT_CHUNK = 128
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
def _mtp_dense_projection_impl(
    hidden_states: pl.Tensor[[B, S, D], pl.BF16],
    prev_hidden_states: pl.Tensor[[B, S, D], pl.BF16],
    enorm_w: pl.Tensor[[D], pl.FP32],
    hnorm_w: pl.Tensor[[D], pl.FP32],
    e_proj_w: pl.Tensor[[D, D], pl.INT8],
    e_proj_w_scale: pl.Tensor[[D], pl.FP32],
    e_proj_smooth: pl.Tensor[[D], pl.FP32],
    h_proj_w: pl.Tensor[[D, D], pl.INT8],
    h_proj_w_scale: pl.Tensor[[D], pl.FP32],
    h_proj_smooth: pl.Tensor[[D], pl.FP32],
    hidden_states_out: pl.Out[pl.Tensor[[B, S, D], pl.BF16]],
):
    hidden_flat = pl.reshape(hidden_states, [T, D])
    prev_flat = pl.reshape(prev_hidden_states, [T, D])
    out_flat = pl.reshape(hidden_states_out, [T, D])
    hidden_norm = pl.create_tensor([T, D], dtype=pl.BF16)
    prev_norm = pl.create_tensor([T, D], dtype=pl.BF16)
    hidden_i8 = pl.create_tensor([T_PAD, D], dtype=pl.INT8)
    prev_i8 = pl.create_tensor([T_PAD, D], dtype=pl.INT8)
    hidden_inv_rms = pl.create_tensor([T, 1], dtype=pl.FP32)
    prev_inv_rms = pl.create_tensor([T, 1], dtype=pl.FP32)
    hidden_amax_parts = pl.create_tensor([D_BLOCKS, T], dtype=pl.FP32)
    prev_amax_parts = pl.create_tensor([D_BLOCKS, T], dtype=pl.FP32)
    hidden_scale_dq = pl.create_tensor([T_PAD, 1], dtype=pl.FP32)
    prev_scale_dq = pl.create_tensor([T_PAD, 1], dtype=pl.FP32)
    out_pad = pl.create_tensor([T_PAD, D], dtype=pl.BF16)

    for t0 in pl.parallel(0, T, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_projection_rms"):
            hidden_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            prev_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(D_BLOCKS, stage=2):
                k0 = kb * D_CHUNK
                hidden_chunk = pl.cast(hidden_flat[t0 : t0 + T_TILE, k0 : k0 + D_CHUNK], target_type=pl.FP32)
                prev_chunk = pl.cast(prev_flat[t0 : t0 + T_TILE, k0 : k0 + D_CHUNK], target_type=pl.FP32)
                hidden_sq_sum = pl.add(
                    hidden_sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(hidden_chunk, hidden_chunk)), [1, T_TILE]),
                )
                prev_sq_sum = pl.add(
                    prev_sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(prev_chunk, prev_chunk)), [1, T_TILE]),
                )
            hidden_inv = pl.reshape(pl.rsqrt(pl.add(pl.mul(hidden_sq_sum, D_INV), EPS)), [T_TILE, 1])
            prev_inv = pl.reshape(pl.rsqrt(pl.add(pl.mul(prev_sq_sum, D_INV), EPS)), [T_TILE, 1])
            hidden_inv_rms = pl.assemble(hidden_inv_rms, hidden_inv, [t0, 0])
            prev_inv_rms = pl.assemble(prev_inv_rms, prev_inv, [t0, 0])

    for t0 in pl.parallel(0, T, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_projection_norm"):
            hidden_inv = hidden_inv_rms[t0 : t0 + T_TILE, 0:1]
            prev_inv = prev_inv_rms[t0 : t0 + T_TILE, 0:1]
            for kb in pl.range(D_BLOCKS):
                k0 = kb * D_CHUNK
                hidden_chunk = pl.cast(hidden_flat[t0 : t0 + T_TILE, k0 : k0 + D_CHUNK], target_type=pl.FP32)
                prev_chunk = pl.cast(prev_flat[t0 : t0 + T_TILE, k0 : k0 + D_CHUNK], target_type=pl.FP32)
                enorm = pl.reshape(enorm_w[k0 : k0 + D_CHUNK], [1, D_CHUNK])
                hnorm = pl.reshape(hnorm_w[k0 : k0 + D_CHUNK], [1, D_CHUNK])
                e_smooth = pl.reshape(e_proj_smooth[k0 : k0 + D_CHUNK], [1, D_CHUNK])
                h_smooth = pl.reshape(h_proj_smooth[k0 : k0 + D_CHUNK], [1, D_CHUNK])
                hidden_norm_tile = pl.col_expand_mul(
                    pl.col_expand_mul(pl.row_expand_mul(hidden_chunk, hidden_inv), enorm),
                    e_smooth,
                )
                prev_norm_tile = pl.col_expand_mul(
                    pl.col_expand_mul(pl.row_expand_mul(prev_chunk, prev_inv), hnorm),
                    h_smooth,
                )
                hidden_norm_bf16 = pl.cast(hidden_norm_tile, target_type=pl.BF16, mode="rint")
                prev_norm_bf16 = pl.cast(prev_norm_tile, target_type=pl.BF16, mode="rint")
                hidden_norm = pl.assemble(hidden_norm, hidden_norm_bf16, [t0, k0])
                prev_norm = pl.assemble(prev_norm, prev_norm_bf16, [t0, k0])
                hidden_abs = pl.maximum(pl.cast(hidden_norm_bf16, target_type=pl.FP32), pl.neg(pl.cast(hidden_norm_bf16, target_type=pl.FP32)))
                prev_abs = pl.maximum(pl.cast(prev_norm_bf16, target_type=pl.FP32), pl.neg(pl.cast(prev_norm_bf16, target_type=pl.FP32)))
                hidden_amax_parts = pl.assemble(hidden_amax_parts, pl.reshape(pl.row_max(hidden_abs), [1, T_TILE]), [kb, t0])
                prev_amax_parts = pl.assemble(prev_amax_parts, pl.reshape(pl.row_max(prev_abs), [1, T_TILE]), [kb, t0])

    for t0 in pl.parallel(0, T, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_projection_quant"):
            hidden_amax = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            prev_amax = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for ab in pl.range(D_BLOCKS):
                hidden_amax = pl.maximum(hidden_amax, hidden_amax_parts[ab : ab + 1, t0 : t0 + T_TILE])
                prev_amax = pl.maximum(prev_amax, prev_amax_parts[ab : ab + 1, t0 : t0 + T_TILE])
            hidden_sq_row = pl.div(pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), hidden_amax)
            prev_sq_row = pl.div(pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), prev_amax)
            hidden_scale_dq = pl.assemble(hidden_scale_dq, pl.reshape(pl.recip(hidden_sq_row), [T_TILE, 1]), [t0, 0])
            prev_scale_dq = pl.assemble(prev_scale_dq, pl.reshape(pl.recip(prev_sq_row), [T_TILE, 1]), [t0, 0])
            hidden_sq_col = pl.reshape(hidden_sq_row, [T_TILE, 1])
            prev_sq_col = pl.reshape(prev_sq_row, [T_TILE, 1])
            for k0 in pl.range(0, D, QUANT_CHUNK):
                hidden_q_f32 = pl.cast(hidden_norm[t0 : t0 + T_TILE, k0 : k0 + QUANT_CHUNK], target_type=pl.FP32)
                prev_q_f32 = pl.cast(prev_norm[t0 : t0 + T_TILE, k0 : k0 + QUANT_CHUNK], target_type=pl.FP32)
                hidden_q_i32 = pl.cast(pl.row_expand_mul(hidden_q_f32, hidden_sq_col), target_type=pl.INT32, mode="rint")
                prev_q_i32 = pl.cast(pl.row_expand_mul(prev_q_f32, prev_sq_col), target_type=pl.INT32, mode="rint")
                hidden_q_half = pl.cast(hidden_q_i32, target_type=pl.FP16, mode="round")
                prev_q_half = pl.cast(prev_q_i32, target_type=pl.FP16, mode="round")
                hidden_i8 = pl.assemble(hidden_i8, pl.cast(hidden_q_half, target_type=pl.INT8, mode="trunc"), [t0, k0])
                prev_i8 = pl.assemble(prev_i8, pl.cast(prev_q_half, target_type=pl.INT8, mode="trunc"), [t0, k0])
    for t0 in pl.parallel(0, T_PAD, LINEAR_T_TILE):
        for nb in pl.parallel(0, OUT_BLOCKS, 1):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_projection_linear"):
                n0 = nb * OUT_CHUNK
                hidden_a0 = hidden_i8[t0 : t0 + LINEAR_T_TILE, 0:D_CHUNK]
                prev_a0 = prev_i8[t0 : t0 + LINEAR_T_TILE, 0:D_CHUNK]
                e_w0 = e_proj_w[n0 : n0 + OUT_CHUNK, 0:D_CHUNK]
                h_w0 = h_proj_w[n0 : n0 + OUT_CHUNK, 0:D_CHUNK]
                hidden_acc = pl.matmul(hidden_a0, e_w0, b_trans=True, out_dtype=pl.INT32)
                prev_acc = pl.matmul(prev_a0, h_w0, b_trans=True, out_dtype=pl.INT32)
                for kb in pl.pipeline(1, D_BLOCKS, stage=2):
                    k0 = kb * D_CHUNK
                    hidden_a = hidden_i8[t0 : t0 + LINEAR_T_TILE, k0 : k0 + D_CHUNK]
                    prev_a = prev_i8[t0 : t0 + LINEAR_T_TILE, k0 : k0 + D_CHUNK]
                    e_w = e_proj_w[n0 : n0 + OUT_CHUNK, k0 : k0 + D_CHUNK]
                    h_w = h_proj_w[n0 : n0 + OUT_CHUNK, k0 : k0 + D_CHUNK]
                    hidden_acc = pl.matmul_acc(hidden_acc, hidden_a, e_w, b_trans=True)
                    prev_acc = pl.matmul_acc(prev_acc, prev_a, h_w, b_trans=True)
                e_scale = pl.reshape(e_proj_w_scale[n0 : n0 + OUT_CHUNK], [1, OUT_CHUNK])
                h_scale = pl.reshape(h_proj_w_scale[n0 : n0 + OUT_CHUNK], [1, OUT_CHUNK])
                hidden_deq = pl.col_expand_mul(
                    pl.row_expand_mul(pl.cast(hidden_acc, target_type=pl.FP32, mode="none"), hidden_scale_dq[t0 : t0 + LINEAR_T_TILE, 0:1]),
                    e_scale,
                )
                prev_deq = pl.col_expand_mul(
                    pl.row_expand_mul(pl.cast(prev_acc, target_type=pl.FP32, mode="none"), prev_scale_dq[t0 : t0 + LINEAR_T_TILE, 0:1]),
                    h_scale,
                )
                acc = pl.add(hidden_deq, prev_deq)
                out_pad = pl.assemble(out_pad, pl.cast(acc, target_type=pl.BF16, mode="rint"), [t0, n0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_projection_output"):
        for n0 in pl.pipeline(0, D, OUT_CHUNK, stage=2):
            out_flat[:, n0:n0 + OUT_CHUNK] = out_pad[0:T, n0:n0 + OUT_CHUNK]

    hidden_states_out = pl.reshape(out_flat, [B, S, D])
    return hidden_states_out


@pl.jit.inline
def _mask_mtp_input_positions(
    hidden_states: pl.Tensor[[B, S, D], pl.BF16],
    position_ids: pl.Tensor[[T], pl.INT32],
    hidden_states_masked: pl.Out[pl.Tensor[[B, S, D], pl.BF16]],
) -> pl.Tensor[[B, S, D], pl.BF16]:
    hidden_flat = pl.reshape(hidden_states, [T, D])
    masked_flat = pl.reshape(hidden_states_masked, [T, D])
    for t0 in pl.parallel(0, T, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_position0_mask"):
            for t in pl.range(t0, t0 + T_TILE):
                pos = pl.read(position_ids, [t])
                for k0 in pl.pipeline(0, D, D_CHUNK, stage=2):
                    if pos == 0:
                        masked_flat[t:t + 1, k0:k0 + D_CHUNK] = pl.full(
                            [1, D_CHUNK],
                            dtype=pl.BF16,
                            value=0.0,
                        )
                    else:
                        masked_flat[t:t + 1, k0:k0 + D_CHUNK] = hidden_flat[t:t + 1, k0:k0 + D_CHUNK]
    hidden_states_masked = pl.reshape(masked_flat, [B, S, D])
    return hidden_states_masked


@pl.jit.inline
def _copy_hc_lane_to_dense(
    prev_hidden_states: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    lane: pl.Scalar[pl.INT32],
    prev_lane: pl.Out[pl.Tensor[[B, S, D], pl.BF16]],
) -> pl.Tensor[[B, S, D], pl.BF16]:
    prev_flat = pl.reshape(prev_lane, [T, D])
    for t0 in pl.parallel(0, T, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_prev_hc_lane"):
            for k0 in pl.pipeline(0, D, D_CHUNK, stage=2):
                prev_flat[t0:t0 + T_TILE, k0:k0 + D_CHUNK] = (
                    prev_hidden_states[t0:t0 + T_TILE, lane, k0:k0 + D_CHUNK]
                )
    prev_lane = pl.reshape(prev_flat, [B, S, D])
    return prev_lane


@pl.jit.inline
def _copy_dense_to_hc_lane(
    dense_lane: pl.Tensor[[B, S, D], pl.BF16],
    lane: pl.Scalar[pl.INT32],
    projected_hidden: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
) -> pl.Tensor[[T, HC_MULT, D], pl.BF16]:
    dense_flat = pl.reshape(dense_lane, [T, D])
    for t0 in pl.parallel(0, T, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_projected_hc_lane"):
            for k0 in pl.pipeline(0, D, D_CHUNK, stage=2):
                projected_hidden[t0:t0 + T_TILE, lane, k0:k0 + D_CHUNK] = (
                    dense_flat[t0:t0 + T_TILE, k0:k0 + D_CHUNK]
                )
    return projected_hidden


@pl.jit.inline
def mtp_projection_impl(
    hidden_states: pl.Tensor[[B, S, D], pl.BF16],
    prev_hidden_states: pl.Tensor[[T, HC_DIM], pl.BF16],
    position_ids: pl.Tensor[[T], pl.INT32],
    enorm_w: pl.Tensor[[D], pl.FP32],
    hnorm_w: pl.Tensor[[D], pl.FP32],
    e_proj_w: pl.Tensor[[D, D], pl.INT8],
    e_proj_w_scale: pl.Tensor[[D], pl.FP32],
    e_proj_smooth: pl.Tensor[[D], pl.FP32],
    h_proj_w: pl.Tensor[[D, D], pl.INT8],
    h_proj_w_scale: pl.Tensor[[D], pl.FP32],
    h_proj_smooth: pl.Tensor[[D], pl.FP32],
    projected_hidden: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
) -> pl.Tensor[[T, HC_MULT, D], pl.BF16]:
    hidden_masked = pl.create_tensor([B, S, D], dtype=pl.BF16)
    hidden_masked = _mask_mtp_input_positions(hidden_states, position_ids, hidden_masked)
    prev_hc = pl.reshape(prev_hidden_states, [T, HC_MULT, D])
    for lane in pl.range(HC_MULT):
        prev_lane = pl.create_tensor([B, S, D], dtype=pl.BF16)
        prev_lane = _copy_hc_lane_to_dense(prev_hc, lane, prev_lane)
        projected_lane = pl.create_tensor([B, S, D], dtype=pl.BF16)
        projected_lane = _mtp_dense_projection_impl(
            hidden_masked,
            prev_lane,
            enorm_w,
            hnorm_w,
            e_proj_w,
            e_proj_w_scale,
            e_proj_smooth,
            h_proj_w,
            h_proj_w_scale,
            h_proj_smooth,
            projected_lane,
        )
        projected_hidden = _copy_dense_to_hc_lane(projected_lane, lane, projected_hidden)
    return projected_hidden


@pl.jit
def mtp_projection(
    hidden_states: pl.Tensor[[B, S, D], pl.BF16],
    prev_hidden_states: pl.Tensor[[T, HC_DIM], pl.BF16],
    position_ids: pl.Tensor[[T], pl.INT32],
    enorm_w: pl.Tensor[[D], pl.FP32],
    hnorm_w: pl.Tensor[[D], pl.FP32],
    e_proj_w: pl.Tensor[[D, D], pl.INT8],
    e_proj_w_scale: pl.Tensor[[D], pl.FP32],
    e_proj_smooth: pl.Tensor[[D], pl.FP32],
    h_proj_w: pl.Tensor[[D, D], pl.INT8],
    h_proj_w_scale: pl.Tensor[[D], pl.FP32],
    h_proj_smooth: pl.Tensor[[D], pl.FP32],
    projected_hidden: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
):
    projected_hidden = mtp_projection_impl(
        hidden_states,
        prev_hidden_states,
        position_ids,
        enorm_w,
        hnorm_w,
        e_proj_w,
        e_proj_w_scale,
        e_proj_smooth,
        h_proj_w,
        h_proj_w_scale,
        h_proj_smooth,
        projected_hidden,
    )
    return projected_hidden


@pl.jit.inline
def mtp_local_logits(
    mtp_hidden: pl.Tensor[[B, S, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_SHARD, D], pl.BF16],
    candidate_logits: pl.Out[pl.Tensor[[T, VOCAB_SHARD], pl.FP32]],
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
    normed_hidden: pl.Out[pl.Tensor[[B, S, D], pl.BF16]],
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
    pre_hc_residual: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
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
    pre_hc_residual = moe_ep1(
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
        pre_hc_residual,
        pl.const(0, pl.INT32),
    )
    return pre_hc_residual


@pl.jit.inline
def mtp_compute_logits_tail(
    pre_hc_residual: pl.Tensor[[T, HC_DIM], pl.BF16],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    shared_head_norm_w: pl.Tensor[[D], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB_SHARD, D], pl.BF16],
    candidate_logits: pl.Out[pl.Tensor[[T, VOCAB_SHARD], pl.FP32]],
) -> pl.Tensor[[T, VOCAB_SHARD], pl.FP32]:
    pre_hc_stack = pl.reshape(pre_hc_residual, [T, HC_MULT, D])
    dense_hidden_flat = pl.create_tensor([T, D], dtype=pl.BF16)
    dense_hidden_flat = hc_head(
        pre_hc_stack,
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


@pl.jit.inline
def mtp_forward_tail(
    hidden_states: pl.Tensor[[B, S, D], pl.BF16],
    prev_hidden_states: pl.Tensor[[T, HC_DIM], pl.BF16],
    position_ids: pl.Tensor[[T], pl.INT32],
    enorm_w: pl.Tensor[[D], pl.FP32],
    hnorm_w: pl.Tensor[[D], pl.FP32],
    e_proj_w: pl.Tensor[[D, D], pl.INT8],
    e_proj_w_scale: pl.Tensor[[D], pl.FP32],
    e_proj_smooth: pl.Tensor[[D], pl.FP32],
    h_proj_w: pl.Tensor[[D, D], pl.INT8],
    h_proj_w_scale: pl.Tensor[[D], pl.FP32],
    h_proj_smooth: pl.Tensor[[D], pl.FP32],
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
    pre_hc_residual: pl.Out[pl.Tensor[[T, HC_DIM], pl.BF16]],
) -> pl.Tensor[[T, HC_DIM], pl.BF16]:
    projected_hidden = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    projected_hidden = mtp_projection_impl(
        hidden_states,
        prev_hidden_states,
        position_ids,
        enorm_w,
        hnorm_w,
        e_proj_w,
        e_proj_w_scale,
        e_proj_smooth,
        h_proj_w,
        h_proj_w_scale,
        h_proj_smooth,
        projected_hidden,
    )
    pre_hc_stack = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    pre_hc_stack = mtp_decoder_layer_tail(
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
        pre_hc_stack,
    )
    pre_hc_flat = pl.reshape(pre_hc_residual, [T, HC_DIM])
    pre_hc_stack_flat = pl.reshape(pre_hc_stack, [T, HC_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_forward_flatten"):
        for k0 in pl.pipeline(0, HC_DIM, D_CHUNK, stage=2):
            pre_hc_flat[:, k0:k0 + D_CHUNK] = pre_hc_stack_flat[:, k0:k0 + D_CHUNK]
    return pre_hc_residual


@pl.jit
def mtp_full_chain(
    hidden_states: pl.Tensor[[B, S, D], pl.BF16],
    prev_hidden_states: pl.Tensor[[T, HC_DIM], pl.BF16],
    position_ids: pl.Tensor[[T], pl.INT32],
    enorm_w: pl.Tensor[[D], pl.FP32],
    hnorm_w: pl.Tensor[[D], pl.FP32],
    e_proj_w: pl.Tensor[[D, D], pl.INT8],
    e_proj_w_scale: pl.Tensor[[D], pl.FP32],
    e_proj_smooth: pl.Tensor[[D], pl.FP32],
    h_proj_w: pl.Tensor[[D, D], pl.INT8],
    h_proj_w_scale: pl.Tensor[[D], pl.FP32],
    h_proj_smooth: pl.Tensor[[D], pl.FP32],
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
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    shared_head_norm_w: pl.Tensor[[D], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB_SHARD, D], pl.BF16],
    candidate_logits: pl.Out[pl.Tensor[[T, VOCAB_SHARD], pl.FP32]],
):
    pre_hc_residual = pl.create_tensor([T, HC_DIM], dtype=pl.BF16)
    pre_hc_residual = mtp_forward_tail(
        hidden_states,
        prev_hidden_states,
        position_ids,
        enorm_w,
        hnorm_w,
        e_proj_w,
        e_proj_w_scale,
        e_proj_smooth,
        h_proj_w,
        h_proj_w_scale,
        h_proj_smooth,
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
        cmp_kv,
        cmp_block_table,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
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
        pre_hc_residual,
    )
    candidate_logits = mtp_compute_logits_tail(
        pre_hc_residual,
        hc_head_fn,
        hc_head_scale,
        hc_head_base,
        shared_head_norm_w,
        lm_head_weight,
        candidate_logits,
    )
    return candidate_logits


def _rms_norm(x, weight):
    import torch

    shape = x.shape
    x_2d = x.reshape(T, D).float()
    sq_sum = torch.zeros(T, 1, dtype=torch.float32)
    for k0 in range(0, D, D_CHUNK):
        x_chunk = x_2d[:, k0:k0 + D_CHUNK]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    inv = torch.rsqrt(sq_sum * D_INV + EPS)
    return (x_2d * inv * weight.float().view(1, D)).reshape(shape)


def golden_mtp_projection(tensors):
    import torch

    hidden_states = tensors["hidden_states"].reshape(T, D).clone()
    hidden_states[tensors["position_ids"].reshape(T) == 0] = 0
    hidden_states = (_rms_norm(hidden_states, tensors["enorm_w"]) * tensors["e_proj_smooth"].float()).to(torch.bfloat16)
    prev_hidden_states = tensors["prev_hidden_states"].reshape(T, HC_MULT, D)
    hidden_i8, hidden_scale = _quantize_rows(hidden_states.float())
    hidden_e = hidden_i8.to(torch.int32).matmul(tensors["e_proj_w"].to(torch.int32).t()).float()
    hidden_e = hidden_e * hidden_scale * tensors["e_proj_w_scale"].float().view(1, D)

    lanes = []
    for h in range(HC_MULT):
        prev_lane = (
            _rms_norm(prev_hidden_states[:, h, :], tensors["hnorm_w"])
            * tensors["h_proj_smooth"].float()
        ).to(torch.bfloat16)
        prev_i8, prev_scale = _quantize_rows(prev_lane.float())
        hidden_h = prev_i8.to(torch.int32).matmul(tensors["h_proj_w"].to(torch.int32).t()).float()
        hidden_h = hidden_h * prev_scale * tensors["h_proj_w_scale"].float().view(1, D)
        lanes.append((hidden_e + hidden_h).to(torch.bfloat16).reshape(T, 1, D))
    tensors["projected_hidden"][:] = torch.cat(lanes, dim=1)


def _local_logits(mtp_hidden, lm_head_weight):
    return mtp_hidden.reshape(T, D).float().matmul(lm_head_weight.float().t())


def golden_mtp_full_chain(tensors):
    import torch
    from mtp_inputs import validate_mtp_input_full_chain

    tensors["projected_hidden"] = torch.empty(T, HC_MULT, D, dtype=torch.bfloat16)
    golden_mtp_projection(tensors)

    tensors["x_hc"] = tensors["projected_hidden"]
    tensors["x_out"] = torch.empty(T, HC_MULT, D, dtype=torch.bfloat16)
    golden_attention_swa(tensors)

    tensors["x_hc"] = tensors["x_out"]
    tensors["x_next"] = torch.empty(T, HC_MULT, D, dtype=torch.bfloat16)
    golden_moe_ep1(tensors)

    pre_hc_residual = tensors["x_next"].flatten(1)
    assert pre_hc_residual.reshape(T, HC_MULT, D).equal(tensors["x_next"])
    step1_tensors = dict(tensors)
    step1_tensors["prev_hidden_states"] = pre_hc_residual
    step1_tensors["projected_hidden"] = torch.empty(T, HC_MULT, D, dtype=torch.bfloat16)
    golden_mtp_projection(step1_tensors)
    assert tuple(step1_tensors["projected_hidden"].shape) == (T, HC_MULT, D)

    tensors["x_hc"] = tensors["x_next"]
    tensors["y"] = torch.empty(T, D, dtype=torch.bfloat16)
    golden_hc_head(tensors)

    normed_hidden = _rms_norm(
        tensors["y"].reshape(B, S, D),
        tensors["shared_head_norm_w"],
    ).to(torch.bfloat16)
    tensors["candidate_logits"][:] = _local_logits(
        normed_hidden,
        tensors["lm_head_weight"],
    )
    _check_logits_contract(tensors)
    validate_mtp_input_full_chain(tensors["candidate_logits"])


def _candidate_position_rows():
    return [b * S + s for b in range(B) for s in range(S)]


def _check_logits_contract(tensors):
    logits = tensors["candidate_logits"]
    assert tuple(logits.shape) == (T, VOCAB_SHARD), tuple(logits.shape)

    rows = _candidate_position_rows()
    view = logits.reshape(B, S, VOCAB_SHARD)
    for b in range(B):
        for s in range(S):
            row = rows[b * S + s]
            assert bool((view[b, s] == logits[row]).all()), (b, s, row)

    values, indices = logits.float().max(dim=-1)
    assert tuple(values.shape) == (T,)
    assert tuple(indices.shape) == (T,)
    assert bool((indices >= 0).all())
    assert bool((indices < VOCAB_SHARD).all())


def _quantize_rows(x):
    import torch

    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    x_i32 = torch.round(x * scale_quant).to(torch.int32)
    return x_i32.to(torch.float16).to(torch.int8), 1.0 / scale_quant


def _quantize_weight_per_out(w):
    import torch

    amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    w_i32 = torch.round(w.float() * scale_quant.view(-1, 1)).to(torch.int32)
    return w_i32.to(torch.float16).to(torch.int8), 1.0 / scale_quant


def build_projection_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_proj_pair():
        w = (torch.rand(D, D)/ D ** 0.5).to(torch.bfloat16)
        return _quantize_weight_per_out(w)

    e_proj_cache = None
    h_proj_cache = None

    def init_e_proj_w():
        nonlocal e_proj_cache
        e_proj_cache = init_proj_pair()
        return e_proj_cache[0]

    def init_e_proj_w_scale():
        nonlocal e_proj_cache
        if e_proj_cache is None:
            e_proj_cache = init_proj_pair()
        return e_proj_cache[1].float()

    def init_h_proj_w():
        nonlocal h_proj_cache
        h_proj_cache = init_proj_pair()
        return h_proj_cache[0]

    def init_h_proj_w_scale():
        nonlocal h_proj_cache
        if h_proj_cache is None:
            h_proj_cache = init_proj_pair()
        return h_proj_cache[1].float()

    return [
        TensorSpec("hidden_states", [B, S, D], torch.bfloat16, init_value=lambda: torch.randn(B, S, D)),
        TensorSpec(
            "prev_hidden_states",
            [T, HC_DIM],
            torch.bfloat16,
            init_value=lambda: torch.randn(T, HC_DIM),
        ),
        TensorSpec("position_ids", [T], torch.int32, init_value=lambda: torch.arange(T, dtype=torch.int32)),
        TensorSpec("enorm_w", [D], torch.float32, init_value=lambda: torch.ones(D)),
        TensorSpec("hnorm_w", [D], torch.float32, init_value=lambda: torch.ones(D)),
        TensorSpec("e_proj_w", [D, D], torch.int8, init_value=init_e_proj_w),
        TensorSpec("e_proj_w_scale", [D], torch.float32, init_value=init_e_proj_w_scale),
        TensorSpec("e_proj_smooth", [D], torch.float32, init_value=lambda: torch.ones(D)),
        TensorSpec("h_proj_w", [D, D], torch.int8, init_value=init_h_proj_w),
        TensorSpec("h_proj_w_scale", [D], torch.float32, init_value=init_h_proj_w_scale),
        TensorSpec("h_proj_smooth", [D], torch.float32, init_value=lambda: torch.ones(D)),
        TensorSpec("projected_hidden", [T, HC_MULT, D], torch.bfloat16, is_output=True),
    ]


def _extend_specs(specs, candidates, skip_names):
    names = {getattr(spec, "name", None) for spec in specs}
    for spec in candidates:
        name = getattr(spec, "name", None)
        if name in skip_names or name in names:
            continue
        specs.append(spec)
        names.add(name)
    return specs


def build_swa_tail_tensor_specs():
    import torch
    from golden import TensorSpec

    specs = []
    _extend_specs(
        specs,
        build_projection_tensor_specs(),
        {
            "projected_hidden",
        },
    )
    _extend_specs(
        specs,
        build_swa_tensor_specs(),
        {
            "x_hc",
            "x_out",
        },
    )
    _extend_specs(
        specs,
        build_moe_ep1_tensor_specs(layer_id=0),
        {
            "x_hc",
            "x_next",
            "layer_id",
        },
    )
    _extend_specs(
        specs,
        build_hc_head_tensor_specs(),
        {
            "x_hc",
            "y",
        },
    )
    specs.extend([
        TensorSpec(
            "shared_head_norm_w",
            [D],
            torch.float32,
            init_value=lambda: torch.ones(D),
        ),
        TensorSpec(
            "lm_head_weight",
            [VOCAB_SHARD, D],
            torch.bfloat16,
            init_value=lambda: (torch.randn(VOCAB_SHARD, D) / D ** 0.5).to(torch.bfloat16),
        ),
        TensorSpec("candidate_logits", [T, VOCAB_SHARD], torch.float32, is_output=True),
    ])
    return specs


def build_full_chain_tensor_specs():
    return build_swa_tail_tensor_specs()


def build_tensor_specs():
    return build_projection_tensor_specs()


CASES = {
    "full-chain": (
        mtp_full_chain,
        build_full_chain_tensor_specs,
        golden_mtp_full_chain,
        {
            "candidate_logits": "logits",
        },
    ),
}


if __name__ == "__main__":
    import argparse
    import torch
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--case", type=str, default="full-chain", choices=sorted(CASES))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    fn, spec_builder, golden_fn, compare_outputs = CASES[args.case]

    compare_fn = {}
    if compare_outputs.get("candidate_logits") == "logits":
        compare_fn["candidate_logits"] = ratio_allclose(atol=1e-2, rtol=1e-2, max_error_ratio=0.02)

    result = run_jit(
        fn=fn,
        specs=spec_builder(),
        golden_fn=golden_fn,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn=compare_fn,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
