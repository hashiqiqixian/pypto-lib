# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Q/KV LoRA + RoPE (dynamic shape): projects token-major
attention-normalized inputs for both decode and prefill attention paths."""


import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ, INT8_SCALE_MAX, INT8_AMAX_EPS
from dynamic_shapes import TOKENS_DYN as T_DYN


# model config
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_DIM_SCALE = float(ROPE_DIM)
ROPE_HALF = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
EPS = M.rms_norm_eps
MAX_SEQ_LEN = M.max_position_embeddings

# tiling
Q_PROJ_TILE = 128       # qproj K-tile (Q_LORA reduction)
QPROJ_MM_N_TILE = 512   # 64 KiB INT8 Right tile on A2/A3
Q_LORA_TILE = 256       # qr rms-norm / quant N granularity (decoupled from qr_proj matmul)
KV_TILE = 64            # kv rms-norm / rope / NOPE N granularity (decoupled from kv_proj matmul)
QUANT_TILE = 256
T_TILE = 8
MATMUL_T_TILE = 16
T_MAX = max(DECODE_BATCH * DECODE_SEQ, PREFILL_BATCH * PREFILL_SEQ)

# Per-projection matmul tiles. Decoupled so each projection's M/N/K can be tuned
# independently of one another AND of the downstream rms/rope granularity above
# (e.g. the matmul N-tile is no longer chained to KV_TILE / Q_LORA_TILE, which the
# NOPE_DIM=448 constraint caps at <=64).
QR_M_TILE = MATMUL_T_TILE  # qr_proj token (M) tile; cube rows must be a 16-row boxed tile
QR_N_TILE = 128         # qr_proj Q_LORA (N) per matmul
QR_K_TILE = 256         # qr_proj D (K) reduction tile    | divides QR_K_SLICE
QR_OK = 2               # qr_proj split-K factor          | D//QR_OK cores share each N-group
QR_K_SLICE = D // QR_OK # qr_proj K per split (=2048)     | QR_K_SLICE//QR_K_TILE inner chunks
KV_M_TILE = MATMUL_T_TILE  # kv_proj token (M) tile; decode pads from 8 real rows to 16
KV_N_TILE = 128         # kv_proj HEAD_DIM (N) per matmul
KV_K_TILE = 256         # kv_proj D (K) reduction tile    | divides KV_K_SLICE
KV_OK = 4               # kv_proj split-K factor          | D//KV_OK cores share each N-group
KV_K_SLICE = D // KV_OK # kv_proj K per split (=1024)     | KV_K_SLICE//KV_K_TILE inner chunks
QPROJ_M_TILE = MATMUL_T_TILE  # qproj token (M) tile; decode pads from 8 real rows to 16
KV_RMS_T_TILE = 8       # kv rms-norm + rope fused token (T) tile
Q_ROPE_T_TILE = 8
Q_ROPE_H_TILE = 4       # heads per fused qproj dequant/rms/rope task; cos/sin build amortizes over them
assert H % Q_ROPE_H_TILE == 0
assert (DECODE_BATCH * DECODE_SEQ) % T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % T_TILE == 0
assert DECODE_BATCH * DECODE_SEQ <= MATMUL_T_TILE
for _m_tile in (QR_M_TILE, KV_M_TILE, QPROJ_M_TILE):
    assert (PREFILL_BATCH * PREFILL_SEQ) % _m_tile == 0
assert Q_LORA % QR_N_TILE == 0 and D % QR_OK == 0 and QR_K_SLICE % QR_K_TILE == 0
assert HEAD_DIM % KV_N_TILE == 0 and D % KV_OK == 0 and KV_K_SLICE % KV_K_TILE == 0
assert (H * HEAD_DIM) % QPROJ_MM_N_TILE == 0 and ((H * HEAD_DIM) // QPROJ_MM_N_TILE) % 4 == 0
assert Q_LORA % Q_PROJ_TILE == 0 and QPROJ_MM_N_TILE * QPROJ_M_TILE * 4 <= 128 * 1024  # L0C Acc cap
assert (DECODE_BATCH * DECODE_SEQ) % KV_RMS_T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % KV_RMS_T_TILE == 0
assert (DECODE_BATCH * DECODE_SEQ) % Q_ROPE_T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % Q_ROPE_T_TILE == 0


@pl.jit.inline
def materialize_rope_rows(
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    rope_cos_t: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin_t: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
):
    t_dim = pl.tensor.dim(position_ids, 0)
    rope_tiles = (t_dim + KV_RMS_T_TILE - 1) // KV_RMS_T_TILE
    for rope_t0 in pl.spmd(rope_tiles, name_hint="qkv_rope_rows"):
        t0 = rope_t0 * KV_RMS_T_TILE
        for rope_dt in pl.range(KV_RMS_T_TILE):
            rope_t = t0 + rope_dt
            if rope_t < t_dim:
                rope_pos = pl.cast(pl.read(position_ids, [rope_t]), pl.INDEX)
                rope_cos_t[rope_t : rope_t + 1, 0:ROPE_DIM] = freqs_cos[rope_pos : rope_pos + 1, 0:ROPE_DIM]
                rope_sin_t[rope_t : rope_t + 1, 0:ROPE_DIM] = freqs_sin[rope_pos : rope_pos + 1, 0:ROPE_DIM]

@pl.jit.inline
def qkv_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    t_dim = pl.tensor.dim(x, 0)
    x_view = pl.reshape(x, [t_dim, D])
    rope_cos_view = pl.reshape(rope_cos, [t_dim, ROPE_DIM])
    rope_sin_view = pl.reshape(rope_sin, [t_dim, ROPE_DIM])
    kv_view = pl.reshape(kv, [t_dim, HEAD_DIM])
    qr_view = pl.reshape(qr, [t_dim, Q_LORA])
    qr_scale_view = pl.reshape(qr_scale, [t_dim, 1])
    t_matmul = ((t_dim + MATMUL_T_TILE - 1) // MATMUL_T_TILE) * MATMUL_T_TILE

    x_matmul = pl.create_tensor([T_MAX, D], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qkv_dynamic_pad_x"):
        for pad_t in pl.range(T_MAX):
            x_row = pl.tile.full([1, D], dtype=pl.BF16, value=0.0)
            if pad_t < t_dim:
                x_row = pl.load(x_view, [pad_t, 0], [1, D], target_memory=pl.MemorySpace.Vec)
            pl.store(x_row, [pad_t, 0], x_matmul)

    # RoPE indices and interleaved cos/signed-sin rows are head-invariant.
    # Prepare them once per token tile so the 16 Q head-group tasks do not each
    # rebuild the same arange/cast/gather chain on their critical AIV path.
    q_rope_cos_il = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    q_rope_sin_signed = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    q_rope_swap_idx = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.INT32)
    for qrp_idx in pl.spmd(
        (t_dim + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE,
        name_hint="q_rope_prepare",
        allow_early_resolve=True,
    ):
        qrp_t0 = qrp_idx * Q_ROPE_T_TILE
        qrp_valid_rows = pl.min(Q_ROPE_T_TILE, t_dim - qrp_t0)
        qrp_ones = pl.tile.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        qrp_idx_i32 = pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32)
        qrp_idx_fp32 = pl.cast(qrp_idx_i32, target_type=pl.FP32)
        qrp_col = pl.col_expand_mul(qrp_ones, qrp_idx_fp32)
        qrp_half = pl.mul(qrp_col, 0.5)
        qrp_dup_i32 = pl.cast(qrp_half, target_type=pl.INT32, mode="trunc")
        qrp_dup_f = pl.cast(qrp_dup_i32, target_type=pl.FP32)
        qrp_lane = pl.sub(qrp_col, pl.mul(qrp_dup_f, 2.0))
        qrp_next_col = pl.add(qrp_col, 1.0)
        qrp_lane_offset = pl.mul(qrp_lane, 2.0)
        qrp_swap_f = pl.sub(qrp_next_col, qrp_lane_offset)
        qrp_row_seed = pl.cast(
            pl.mul(
                pl.cast(
                    pl.tile.arange(0, [1, Q_ROPE_T_TILE], dtype=pl.INT32),
                    target_type=pl.FP32,
                ),
                ROPE_DIM_SCALE,
            ),
            target_type=pl.INT32,
        )
        qrp_row_grid = pl.col_expand_mul(
            pl.tile.full([ROPE_DIM, Q_ROPE_T_TILE], dtype=pl.INT32, value=1),
            qrp_row_seed,
        )
        qrp_row_offset = pl.transpose(
            qrp_row_grid,
            axis1=0,
            axis2=1,
        )
        qrp_dup_idx = pl.add(pl.cast(qrp_dup_f, target_type=pl.INT32), qrp_row_offset)
        qrp_swap_idx = pl.add(pl.cast(qrp_swap_f, target_type=pl.INT32), qrp_row_offset)
        qrp_sign = pl.sub(pl.mul(qrp_lane, 2.0), 1.0)
        qrp_cos_rows = pl.load(
            rope_cos_view,
            [qrp_t0, 0],
            [Q_ROPE_T_TILE, ROPE_DIM],
            valid_shapes=[qrp_valid_rows, ROPE_DIM],
            target_memory=pl.MemorySpace.Vec,
        )
        qrp_sin_rows = pl.load(
            rope_sin_view,
            [qrp_t0, 0],
            [Q_ROPE_T_TILE, ROPE_DIM],
            valid_shapes=[qrp_valid_rows, ROPE_DIM],
            target_memory=pl.MemorySpace.Vec,
        )
        qrp_cos = pl.cast(qrp_cos_rows, target_type=pl.FP32)
        qrp_sin = pl.cast(qrp_sin_rows, target_type=pl.FP32)
        qrp_gather_tmp = pl.create_tile(
            [Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec
        )
        qrp_cos_il = pl.tile.gather(qrp_cos, qrp_dup_idx, qrp_gather_tmp)
        qrp_sin_il = pl.tile.gather(qrp_sin, qrp_dup_idx, qrp_gather_tmp)
        qrp_sin_signed = pl.mul(qrp_sin_il, qrp_sign)
        pl.store(
            pl.set_validshape(qrp_cos_il, qrp_valid_rows, ROPE_DIM),
            [qrp_t0, 0],
            q_rope_cos_il,
        )
        pl.store(
            pl.set_validshape(qrp_sin_signed, qrp_valid_rows, ROPE_DIM),
            [qrp_t0, 0],
            q_rope_sin_signed,
        )
        pl.store(
            pl.set_validshape(qrp_swap_idx, qrp_valid_rows, ROPE_DIM),
            [qrp_t0, 0],
            q_rope_swap_idx,
        )

    # Split-K qr_proj (M=t_dim, K=D=4096, N=Q_LORA=1024). QR_N_TILE=128 gives
    # eight N-groups; QR_OK=2 expands them to 16 cube blocks and atomic-adds the
    # K partials into a zero-seeded output. Auto-dep on qr_fp32 orders the seed
    # before every atomic RMW.
    qr_fp32 = pl.create_tensor([T_MAX, Q_LORA], dtype=pl.FP32)
    qr_i8_matmul = pl.create_tensor([T_MAX, Q_LORA], dtype=pl.INT8)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_proj_seed"):
        for tc in pl.range(t_matmul // QR_M_TILE):
            ts0 = tc * QR_M_TILE
            for nb in pl.range(Q_LORA // QR_N_TILE):
                nseed0 = nb * QR_N_TILE
                qr_fp32[ts0 : ts0 + QR_M_TILE, nseed0 : nseed0 + QR_N_TILE] = pl.full(
                    [QR_M_TILE, QR_N_TILE], dtype=pl.FP32, value=0.0
                )
                qr_i8_matmul[ts0 : ts0 + QR_M_TILE, nseed0 : nseed0 + QR_N_TILE] = pl.cast(
                    pl.full([QR_M_TILE, QR_N_TILE], dtype=pl.FP16, value=0.0),
                    target_type=pl.INT8,
                    mode="trunc",
                )
    for qbg_idx in pl.spmd((Q_LORA // QR_N_TILE) * QR_OK, name_hint="qr_proj_matmul", allow_early_resolve=True):
        q_a_col0 = (qbg_idx // QR_OK) * QR_N_TILE
        qr_k_base = (qbg_idx % QR_OK) * QR_K_SLICE
        for tc in pl.range(t_matmul // QR_M_TILE):
            t0 = tc * QR_M_TILE
            q_acc = pl.create_tile(
                [QR_M_TILE, QR_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc
            )
            for db in pl.pipeline(QR_K_SLICE // QR_K_TILE, stage=2):
                qr_d0 = qr_k_base + db * QR_K_TILE
                q_x_chunk_bf16 = pl.load(
                    x_matmul,
                    [t0, qr_d0],
                    [QR_M_TILE, QR_K_TILE],
                )
                w_chunk = pl.load(
                    wq_a,
                    [qr_d0, q_a_col0],
                    [QR_K_TILE, QR_N_TILE],
                )
                if db == 0:
                    q_acc = pl.matmul(q_x_chunk_bf16, w_chunk, out_dtype=pl.FP32)
                else:
                    q_acc = pl.matmul_acc(q_acc, q_x_chunk_bf16, w_chunk)
            qr_fp32 = pl.store(q_acc, [t0, q_a_col0], qr_fp32, atomic=pl.AtomicType.Add)

    # Two passes per block: pass 1 computes amax; pass 2 recomputes norm and quantizes.
    for tg_idx in pl.spmd((t_dim + T_TILE - 1) // T_TILE, name_hint="qr_rms_norm_quant", allow_early_resolve=True):
        tg = tg_idx * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - tg)
        qr_sum_tmp = pl.create_tile(
            [T_TILE, Q_LORA_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        qr_max_tmp = pl.create_tile(
            [T_TILE, Q_LORA_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        # row_sum/row_max use an explicit Vec scratch tile. Keep these four
        # reductions sequential so adjacent software-pipeline stages cannot
        # overwrite the shared scratch before its result is consumed.
        qr_rms_chunk0 = pl.load(
            qr_fp32,
            [tg, 0],
            [T_TILE, Q_LORA_TILE],
            valid_shapes=[valid_rows, Q_LORA_TILE],
            target_memory=pl.MemorySpace.Vec,
        )
        qr_sq_sum = pl.row_sum(pl.mul(qr_rms_chunk0, qr_rms_chunk0), qr_sum_tmp)
        gamma_rms_chunk0 = pl.reshape(
            pl.cast(
                pl.load(gamma_cq, [0], [Q_LORA_TILE], target_memory=pl.MemorySpace.Vec),
                target_type=pl.FP32,
            ),
            [1, Q_LORA_TILE],
        )
        qr_g0 = pl.col_expand_mul(qr_rms_chunk0, gamma_rms_chunk0)
        qr_amax_g = pl.row_max(pl.abs(qr_g0), qr_max_tmp)
        for qr_rms_qb in pl.range(1, Q_LORA // Q_LORA_TILE):
            qr_rms_col0 = qr_rms_qb * Q_LORA_TILE
            qr_rms_chunk = pl.load(
                qr_fp32,
                [tg, qr_rms_col0],
                [T_TILE, Q_LORA_TILE],
                valid_shapes=[valid_rows, Q_LORA_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            qr_sq_part = pl.row_sum(pl.mul(qr_rms_chunk, qr_rms_chunk), qr_sum_tmp)
            gamma_rms_input = pl.load(
                gamma_cq,
                [qr_rms_col0],
                [Q_LORA_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            gamma_rms_cast = pl.cast(gamma_rms_input, target_type=pl.FP32)
            gamma_rms_chunk = pl.reshape(gamma_rms_cast, [1, Q_LORA_TILE])
            qr_g = pl.col_expand_mul(qr_rms_chunk, gamma_rms_chunk)
            qr_g_abs = pl.abs(qr_g)
            qr_amax_part = pl.row_max(qr_g_abs, qr_max_tmp)
            qr_sq_sum = pl.add(qr_sq_sum, qr_sq_part)
            qr_amax_g = pl.maximum(qr_amax_g, qr_amax_part)
        qr_inv_rms = pl.rsqrt(pl.add(pl.mul(qr_sq_sum, 1.0 / Q_LORA), EPS), high_precision=True)
        qr_amax_normed = pl.mul(qr_inv_rms, qr_amax_g)
        qr_tile_amax = pl.maximum(qr_amax_normed, INT8_AMAX_EPS)

        qr_scale_quant = pl.mul(pl.recip(qr_tile_amax), INT8_SCALE_MAX)
        qr_tile_scale_dq = pl.mul(qr_tile_amax, 1.0 / INT8_SCALE_MAX)
        pl.store(pl.set_validshape(qr_tile_scale_dq, valid_rows, 1), [tg, 0], qr_scale_view)

        for qa in pl.pipeline(0, Q_LORA, QUANT_TILE, stage=2):
            qr_chunk = pl.load(
                qr_fp32,
                [tg, qa],
                [T_TILE, QUANT_TILE],
                valid_shapes=[valid_rows, QUANT_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            gamma_q_input = pl.load(
                gamma_cq,
                [qa],
                [QUANT_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            gamma_q_cast = pl.cast(gamma_q_input, target_type=pl.FP32)
            gamma_q_chunk = pl.reshape(gamma_q_cast, [1, QUANT_TILE])
            qr_q_normed = pl.col_expand_mul(pl.row_expand_mul(qr_chunk, qr_inv_rms), gamma_q_chunk)
            qr_q_scaled = pl.row_expand_mul(qr_q_normed, qr_scale_quant)
            qr_q_i32 = pl.cast(qr_q_scaled, target_type=pl.INT32, mode="rint")
            qr_q_half = pl.cast(qr_q_i32, target_type=pl.FP16, mode="round")
            qr_q_i8 = pl.cast(qr_q_half, target_type=pl.INT8, mode="trunc")
            qr_q_out = pl.set_validshape(qr_q_i8, valid_rows, QUANT_TILE)
            pl.store(qr_q_out, [tg, qa], qr_view)
            pl.store(qr_q_out, [tg, qa], qr_i8_matmul)

    # UN-MIXED qproj: keep the pure-matmul scope (cube, INT32 -> GM) separate from
    # downstream vector work. This lets the scheduler defer q dequant until AIV is free
    # instead of pinning it next to qproj and competing with the critical qr_proj AIV work.
    q_proj_i32 = pl.create_tensor([T_MAX, H * HEAD_DIM], dtype=pl.INT32)
    for hg_idx in pl.spmd(
        (H * HEAD_DIM) // QPROJ_MM_N_TILE,
        name_hint="qproj_matmul",
        allow_early_resolve=True,
    ):
        w_col0 = hg_idx * QPROJ_MM_N_TILE
        for tc in pl.range(t_matmul // QPROJ_M_TILE):
            t0 = tc * QPROJ_M_TILE
            col_acc = pl.create_tile(
                [QPROJ_M_TILE, QPROJ_MM_N_TILE], dtype=pl.INT32, target_memory=pl.MemorySpace.Acc
            )
            for qb in pl.pipeline(0, Q_LORA // Q_PROJ_TILE, stage=1):
                qr_proj_col0 = qb * Q_PROJ_TILE
                qr_i8_chunk = pl.load(
                    qr_i8_matmul,
                    [t0, qr_proj_col0],
                    [QPROJ_M_TILE, Q_PROJ_TILE],
                )
                wq_chunk = pl.load(
                    wq_b,
                    [qr_proj_col0, w_col0],
                    [Q_PROJ_TILE, QPROJ_MM_N_TILE],
                )
                if qr_proj_col0 == 0:
                    col_acc = pl.matmul(qr_i8_chunk, wq_chunk, out_dtype=pl.INT32)
                else:
                    col_acc = pl.matmul_acc(col_acc, qr_i8_chunk, wq_chunk)
            pl.store(col_acc, [t0, w_col0], q_proj_i32)

    # Fuse qproj dequant, per-head RMSNorm, and NOPE writeback. Q RoPE input is
    # materialized through GM before gather because slicing a HEAD_DIM-wide Tile
    # preserves its parent row stride instead of producing a dense ROPE_DIM Tile.
    # A full [token, head] tile fits in Vec UB, so dequantize each head once and
    # retain it across the RMS reduction instead of rereading/recomputing NOPE.
    # RoPE: out[j] = inv_rms * (x[j] * cos[j] + x[j^1] * sign[j] * sin[j]).
    q_flat = pl.reshape(q, [t_dim, H * HEAD_DIM])
    q_rope_dq = pl.create_tensor([t_dim, H * ROPE_DIM], dtype=pl.FP32)
    # Expand the row scalar to ROPE_DIM before the GM round-trip. A2/A3 cannot
    # transpose the ColMajor row-reduction result or load a strided [T_TILE, 1]
    # ND slice, while a dense 64-column Tile is naturally aligned.
    q_head_inv_rms_buf = pl.create_tensor([t_dim, H * ROPE_DIM], dtype=pl.FP32)
    for hg_idx in pl.spmd(H // Q_ROPE_H_TILE, name_hint="qproj_dequant_rms_nope_rope", allow_early_resolve=True):
        hg = hg_idx * Q_ROPE_H_TILE
        for tg_idx in pl.range((t_dim + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE):
            tg = tg_idx * Q_ROPE_T_TILE
            valid_rows = pl.min(Q_ROPE_T_TILE, t_dim - tg)
            qr_scale_dq_t = pl.load(
                qr_scale_view,
                [tg, 0],
                [Q_ROPE_T_TILE, 1],
                valid_shapes=[valid_rows, 1],
                target_memory=pl.MemorySpace.Vec,
            )
            q_head_reduce_tmp = pl.create_tile(
                [Q_ROPE_T_TILE, HEAD_DIM], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            # row_sum uses a shared explicit scratch tile, so process adjacent
            # heads sequentially to avoid overlapping scratch writes.
            for h_inner in pl.range(Q_ROPE_H_TILE):
                h = hg + h_inner
                h0 = h * HEAD_DIM
                q_head_acc = pl.load(
                    q_proj_i32,
                    [tg, h0],
                    [Q_ROPE_T_TILE, HEAD_DIM],
                    valid_shapes=[valid_rows, HEAD_DIM],
                    target_memory=pl.MemorySpace.Vec,
                )
                q_head_scale_input = pl.load(
                    wq_b_scale,
                    [h0],
                    [HEAD_DIM],
                    target_memory=pl.MemorySpace.Vec,
                )
                q_head_scale = pl.reshape(q_head_scale_input, [1, HEAD_DIM])
                q_head_acc_fp32 = pl.cast(q_head_acc, target_type=pl.FP32, mode="none")
                q_head_row_scaled = pl.row_expand_mul(q_head_acc_fp32, qr_scale_dq_t)
                q_head_dq = pl.col_expand_mul(q_head_row_scaled, q_head_scale)
                q_head_sq = pl.mul(q_head_dq, q_head_dq)
                q_head_sq_sum = pl.row_sum(q_head_sq, q_head_reduce_tmp)
                q_head_sq_mean = pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM)
                q_head_var = pl.add(q_head_sq_mean, EPS)
                q_head_inv_rms = pl.rsqrt(q_head_var, high_precision=True)
                q_head_inv_rms_seed = pl.tile.full(
                    [Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0
                )
                q_head_inv_rms_expanded = pl.row_expand_mul(q_head_inv_rms_seed, q_head_inv_rms)
                pl.store(q_head_inv_rms_expanded, [tg, h * ROPE_DIM], q_head_inv_rms_buf)

                q_nope_normed = pl.row_expand_mul(q_head_dq[:, 0:NOPE_DIM], q_head_inv_rms)
                q_nope_bf16 = pl.cast(q_nope_normed, target_type=pl.BF16, mode="rint")
                pl.store(
                    pl.set_validshape(q_nope_bf16, valid_rows, NOPE_DIM),
                    [tg, h0],
                    q_flat,
                )

                # Preserve the original FP32 RoPE-before-RMS ordering. A GM
                # round-trip turns this HEAD_DIM slice view into a dense
                # ROPE_DIM Tile for the gather stage below.
                q_rope_chunk = q_head_dq[:, NOPE_DIM:HEAD_DIM]
                pl.store(q_rope_chunk, [tg, h * ROPE_DIM], q_rope_dq)

    for hg_idx in pl.spmd(H // Q_ROPE_H_TILE, name_hint="q_rope_dense_gather", allow_early_resolve=True):
        hg = hg_idx * Q_ROPE_H_TILE
        for tg_idx in pl.range((t_dim + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE):
            tg = tg_idx * Q_ROPE_T_TILE
            valid_rows = pl.min(Q_ROPE_T_TILE, t_dim - tg)
            q_cos_il = pl.load(
                q_rope_cos_il,
                [tg, 0],
                [Q_ROPE_T_TILE, ROPE_DIM],
                valid_shapes=[valid_rows, ROPE_DIM],
                target_memory=pl.MemorySpace.Vec,
            )
            q_sin_signed = pl.load(
                q_rope_sin_signed,
                [tg, 0],
                [Q_ROPE_T_TILE, ROPE_DIM],
                valid_shapes=[valid_rows, ROPE_DIM],
                target_memory=pl.MemorySpace.Vec,
            )
            q_swap_idx = pl.load(
                q_rope_swap_idx,
                [tg, 0],
                [Q_ROPE_T_TILE, ROPE_DIM],
                valid_shapes=[valid_rows, ROPE_DIM],
                target_memory=pl.MemorySpace.Vec,
            )
            q_gather_tmp = pl.create_tile(
                [Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec
            )
            for h_inner in pl.range(Q_ROPE_H_TILE):
                h = hg + h_inner
                h0 = h * HEAD_DIM
                q_rope_chunk = pl.load(
                    q_rope_dq,
                    [tg, h * ROPE_DIM],
                    [Q_ROPE_T_TILE, ROPE_DIM],
                    valid_shapes=[valid_rows, ROPE_DIM],
                    target_memory=pl.MemorySpace.Vec,
                )
                q_head_inv_rms_loaded = pl.load(
                    q_head_inv_rms_buf,
                    [tg, h * ROPE_DIM],
                    [Q_ROPE_T_TILE, ROPE_DIM],
                    valid_shapes=[valid_rows, ROPE_DIM],
                    target_memory=pl.MemorySpace.Vec,
                )
                q_rope_swapped = pl.tile.gather(q_rope_chunk, q_swap_idx, q_gather_tmp)
                q_rope_base = pl.mul(q_rope_chunk, q_cos_il)
                q_rope_delta = pl.mul(q_rope_swapped, q_sin_signed)
                q_rope_rot = pl.add(q_rope_base, q_rope_delta)
                q_rope_normed = pl.mul(q_rope_rot, q_head_inv_rms_loaded)
                q_rope_bf16 = pl.cast(q_rope_normed, target_type=pl.BF16, mode="rint")
                pl.store(pl.set_validshape(q_rope_bf16, valid_rows, ROPE_DIM), [tg, h0 + NOPE_DIM], q_flat)

    # Split-K kv_proj uses four 128-column N-groups and KV_OK=4, again producing
    # 16 cube blocks. KV is off the critical path, so more K splits only add atomic
    # contention without shortening decode.
    kv_fp32 = pl.create_tensor([T_MAX, HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj_seed"):
        for tc in pl.range(t_matmul // KV_M_TILE):
            kts0 = tc * KV_M_TILE
            for nb in pl.range(HEAD_DIM // KV_N_TILE):
                kvseed0 = nb * KV_N_TILE
                kv_fp32[kts0 : kts0 + KV_M_TILE, kvseed0 : kvseed0 + KV_N_TILE] = pl.full(
                    [KV_M_TILE, KV_N_TILE], dtype=pl.FP32, value=0.0
                )
    # `late_dep` is a dummy barrier hung off the rms_norm TaskId: kv_proj is off the
    # critical path, so it resolves one hop after rms_norm and lets qr_proj_matmul
    # take the cores first.
    with pl.spmd((HEAD_DIM // KV_N_TILE) * KV_OK, name_hint="kv_proj_matmul", deps=[late_dep]) as _kv_tid:
        kbg = pl.tile.get_block_idx()
        kv_col0 = (kbg // KV_OK) * KV_N_TILE
        kv_k_base = (kbg % KV_OK) * KV_K_SLICE
        for tc in pl.range(t_matmul // KV_M_TILE):
            t0 = tc * KV_M_TILE
            kv_acc = pl.create_tile(
                [KV_M_TILE, KV_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc
            )
            for db in pl.pipeline(KV_K_SLICE // KV_K_TILE, stage=2):
                d0 = kv_k_base + db * KV_K_TILE
                kv_x_chunk_bf16 = pl.load(
                    x_matmul,
                    [t0, d0],
                    [KV_M_TILE, KV_K_TILE],
                )
                wkv_chunk = pl.load(
                    wkv,
                    [d0, kv_col0],
                    [KV_K_TILE, KV_N_TILE],
                )
                if db == 0:
                    kv_acc = pl.matmul(kv_x_chunk_bf16, wkv_chunk, out_dtype=pl.FP32)
                else:
                    kv_acc = pl.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk)
            kv_fp32 = pl.store(kv_acc, [t0, kv_col0], kv_fp32, atomic=pl.AtomicType.Add)

    # Fused KV RMSNorm + interleaved (CANN A3) RoPE. One spmd task per [KV_RMS_T_TILE, HEAD_DIM]
    # row block computes the per-row inv_rms once (pass 1) and consumes it locally for
    # BOTH the NOPE writeback and the rope rotation -- so inv_rms no longer round-trips
    # through GM (the old kv_inv_rms_tensor) and the two passes collapse into a single
    # dispatch. NOPE columns [0:NOPE_DIM) and rope columns [NOPE_DIM:HEAD_DIM) are
    # disjoint, so each task writes a clean, conflict-free row block of kv. Vec UB stays
    # well under the 192 KB cap (chunks are at most [KV_RMS_T_TILE, KV_TILE] fp32).
    for tg_idx in pl.spmd((t_dim + KV_RMS_T_TILE - 1) // KV_RMS_T_TILE, name_hint="kv_rms_norm_rope"):
        tg = tg_idx * KV_RMS_T_TILE
        valid_rows = pl.min(KV_RMS_T_TILE, t_dim - tg)
        kv_reduce_tmp = pl.create_tile(
            [KV_RMS_T_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        kv_gather_tmp = pl.create_tile(
            [KV_RMS_T_TILE, ROPE_DIM], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec
        )
        # Pass 1: per-row sum of squares over the full HEAD_DIM -> inv_rms.
        kv_sq_sum = pl.tile.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HEAD_DIM // KV_TILE, stage=2):
            kv_sq_col0 = kb * KV_TILE
            kv_chunk = pl.load(
                kv_fp32,
                [tg, kv_sq_col0],
                [KV_RMS_T_TILE, KV_TILE],
                valid_shapes=[valid_rows, KV_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            kv_sq_sum = pl.add(
                kv_sq_sum,
                pl.reshape(pl.row_sum(pl.mul(kv_chunk, kv_chunk), kv_reduce_tmp), [1, KV_RMS_T_TILE]),
            )
        kv_inv_rms = pl.rsqrt(pl.add(pl.mul(kv_sq_sum, 1.0 / HEAD_DIM), EPS), high_precision=True)
        kv_inv_rms_t = pl.reshape(kv_inv_rms, [KV_RMS_T_TILE, 1])

        # NOPE writeback: rms-normalize columns [0:NOPE_DIM) with per-column gamma.
        for nb in pl.pipeline(NOPE_DIM // KV_TILE, stage=2):
            n0 = nb * KV_TILE
            kv_chunk = pl.load(
                kv_fp32,
                [tg, n0],
                [KV_RMS_T_TILE, KV_TILE],
                valid_shapes=[valid_rows, KV_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            gamma_kv_input = pl.load(
                gamma_ckv,
                [n0],
                [KV_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            gamma_kv_cast = pl.cast(gamma_kv_input, target_type=pl.FP32)
            gamma_kv_chunk = pl.reshape(gamma_kv_cast, [1, KV_TILE])
            kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk)
            kv_nope_out = pl.cast(kv_normed, target_type=pl.BF16, mode="rint")
            pl.store(pl.set_validshape(kv_nope_out, valid_rows, KV_TILE), [tg, n0], kv_view)

        # RoPE writeback on columns [NOPE_DIM:HEAD_DIM), interleaved (CANN A3) swap-gather
        # (same form as qproj_dequant_rms_nope_rope), built in-kernel. inv_rms (per-row, the same
        # factor used for NOPE above) and gamma (per-column, full ROPE_DIM) are folded into
        # kv_rope_norm_chunk BEFORE the swap so the swapped lane n[j^1] carries gamma[j^1]
        # (gamma does NOT commute with the rotation; inv_rms does).
        #   out[j] = n[j]*cos_il[j] + n[j^1]*sign[j]*sin_il[j]
        gamma_rope_input = pl.load(
            gamma_ckv,
            [NOPE_DIM],
            [ROPE_DIM],
            target_memory=pl.MemorySpace.Vec,
        )
        gamma_rope_cast = pl.cast(gamma_rope_input, target_type=pl.FP32)
        gamma_rope = pl.reshape(gamma_rope_cast, [1, ROPE_DIM])
        kv_rope_chunk = pl.load(
            kv_fp32,
            [tg, NOPE_DIM],
            [KV_RMS_T_TILE, ROPE_DIM],
            valid_shapes=[valid_rows, ROPE_DIM],
            target_memory=pl.MemorySpace.Vec,
        )
        kv_rope_norm_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_rope_chunk, kv_inv_rms_t), gamma_rope)
        kv_ones = pl.tile.full([KV_RMS_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        kv_col = pl.col_expand_mul(
            kv_ones,
            pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
        )
        kv_dup_f = pl.cast(pl.cast(pl.mul(kv_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        kv_lane = pl.sub(kv_col, pl.mul(kv_dup_f, 2.0))                                            # j%2
        kv_swap_f = pl.sub(pl.add(kv_col, 1.0), pl.mul(kv_lane, 2.0))                              # j^1
        kv_row_seed = pl.cast(
            pl.mul(
                pl.cast(
                    pl.tile.arange(0, [1, KV_RMS_T_TILE], dtype=pl.INT32),
                    target_type=pl.FP32,
                ),
                ROPE_DIM_SCALE,
            ),
            target_type=pl.INT32,
        )
        kv_row_grid = pl.col_expand_mul(
            pl.tile.full([ROPE_DIM, KV_RMS_T_TILE], dtype=pl.INT32, value=1),
            kv_row_seed,
        )
        kv_row_offset = pl.transpose(
            kv_row_grid,
            axis1=0,
            axis2=1,
        )
        kv_dup_idx = pl.add(pl.cast(kv_dup_f, target_type=pl.INT32), kv_row_offset)
        kv_swap_idx = pl.add(pl.cast(kv_swap_f, target_type=pl.INT32), kv_row_offset)
        kv_sign = pl.sub(pl.mul(kv_lane, 2.0), 1.0)                                                # [-1,+1,...]
        kv_cos_rows = pl.load(
            rope_cos_view,
            [tg, 0],
            [KV_RMS_T_TILE, ROPE_DIM],
            valid_shapes=[valid_rows, ROPE_DIM],
            target_memory=pl.MemorySpace.Vec,
        )
        kv_sin_rows = pl.load(
            rope_sin_view,
            [tg, 0],
            [KV_RMS_T_TILE, ROPE_DIM],
            valid_shapes=[valid_rows, ROPE_DIM],
            target_memory=pl.MemorySpace.Vec,
        )
        kv_cos_il = pl.tile.gather(pl.cast(kv_cos_rows, target_type=pl.FP32), kv_dup_idx, kv_gather_tmp)
        kv_sin_il = pl.tile.gather(pl.cast(kv_sin_rows, target_type=pl.FP32), kv_dup_idx, kv_gather_tmp)
        kv_swapped = pl.tile.gather(kv_rope_norm_chunk, kv_swap_idx, kv_gather_tmp)
        kv_rope_rot = pl.add(pl.mul(kv_rope_norm_chunk, kv_cos_il), pl.mul(pl.mul(kv_swapped, kv_sign), kv_sin_il))
        kv_rope_i16 = pl.cast(kv_rope_rot, target_type=pl.BF16, mode="rint")
        pl.store(pl.set_validshape(kv_rope_i16, valid_rows, ROPE_DIM), [tg, NOPE_DIM], kv_view)

    return q


@pl.jit
def qr_pipeline_diagnostics_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    qr_proj_diag: pl.Out[pl.Tensor[[T_DYN, Q_LORA], pl.FP32]],
    qr_sq_mean_diag: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32]],
    qr_inv_rms_diag: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32]],
    qr_amax_raw_diag: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32]],
    qr_amax_norm_diag: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32]],
    qr_scale_diag: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32]],
    qr_diag: pl.Out[pl.Tensor[[T_DYN, Q_LORA], pl.INT8]],
):
    """Standalone checkpoints for bisecting the QR projection/quant pipeline."""
    x.bind_dynamic(0, T_DYN)
    qr_proj_diag.bind_dynamic(0, T_DYN)
    qr_sq_mean_diag.bind_dynamic(0, T_DYN)
    qr_inv_rms_diag.bind_dynamic(0, T_DYN)
    qr_amax_raw_diag.bind_dynamic(0, T_DYN)
    qr_amax_norm_diag.bind_dynamic(0, T_DYN)
    qr_scale_diag.bind_dynamic(0, T_DYN)
    qr_diag.bind_dynamic(0, T_DYN)

    t_dim = pl.tensor.dim(x, 0)
    t_matmul = ((t_dim + MATMUL_T_TILE - 1) // MATMUL_T_TILE) * MATMUL_T_TILE
    x_matmul = pl.create_tensor([T_MAX, D], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_diag_pad_x"):
        for pad_t in pl.range(T_MAX):
            x_row = pl.tile.full([1, D], dtype=pl.BF16, value=0.0)
            if pad_t < t_dim:
                x_row = pl.load(x, [pad_t, 0], [1, D], target_memory=pl.MemorySpace.Vec)
            pl.store(x_row, [pad_t, 0], x_matmul)

    qr_fp32 = pl.create_tensor([T_MAX, Q_LORA], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_diag_proj_seed"):
        for tc in pl.range(t_matmul // QR_M_TILE):
            ts0 = tc * QR_M_TILE
            for nb in pl.range(Q_LORA // QR_N_TILE):
                nseed0 = nb * QR_N_TILE
                qr_fp32[ts0 : ts0 + QR_M_TILE, nseed0 : nseed0 + QR_N_TILE] = pl.full(
                    [QR_M_TILE, QR_N_TILE], dtype=pl.FP32, value=0.0
                )

    for qbg_idx in pl.spmd(
        (Q_LORA // QR_N_TILE) * QR_OK,
        name_hint="qr_diag_proj_matmul",
        allow_early_resolve=True,
    ):
        q_a_col0 = (qbg_idx // QR_OK) * QR_N_TILE
        qr_k_base = (qbg_idx % QR_OK) * QR_K_SLICE
        for tc in pl.range(t_matmul // QR_M_TILE):
            t0 = tc * QR_M_TILE
            q_acc = pl.create_tile(
                [QR_M_TILE, QR_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc
            )
            for db in pl.pipeline(QR_K_SLICE // QR_K_TILE, stage=2):
                qr_d0 = qr_k_base + db * QR_K_TILE
                q_x_chunk = pl.load(x_matmul, [t0, qr_d0], [QR_M_TILE, QR_K_TILE])
                w_chunk = pl.load(wq_a, [qr_d0, q_a_col0], [QR_K_TILE, QR_N_TILE])
                if db == 0:
                    q_acc = pl.matmul(q_x_chunk, w_chunk, out_dtype=pl.FP32)
                else:
                    q_acc = pl.matmul_acc(q_acc, q_x_chunk, w_chunk)
            qr_fp32 = pl.store(q_acc, [t0, q_a_col0], qr_fp32, atomic=pl.AtomicType.Add)

    qr_snapshot_tiles = (t_dim + T_TILE - 1) // T_TILE
    for snapshot_idx in pl.spmd(
        qr_snapshot_tiles * (Q_LORA // Q_LORA_TILE),
        name_hint="qr_diag_proj_snapshot",
        allow_early_resolve=True,
    ):
        snapshot_tg_idx = snapshot_idx // (Q_LORA // Q_LORA_TILE)
        snapshot_qb = snapshot_idx % (Q_LORA // Q_LORA_TILE)
        snapshot_tg = snapshot_tg_idx * T_TILE
        snapshot_col0 = snapshot_qb * Q_LORA_TILE
        snapshot_rows = pl.min(T_TILE, t_dim - snapshot_tg)
        snapshot_chunk = pl.load(
            qr_fp32,
            [snapshot_tg, snapshot_col0],
            [T_TILE, Q_LORA_TILE],
            valid_shapes=[snapshot_rows, Q_LORA_TILE],
            target_memory=pl.MemorySpace.Vec,
        )
        pl.store(
            pl.set_validshape(snapshot_chunk, snapshot_rows, Q_LORA_TILE),
            [snapshot_tg, snapshot_col0],
            qr_proj_diag,
        )

    for tg_idx in pl.spmd(
        (t_dim + T_TILE - 1) // T_TILE,
        name_hint="qr_diag_reduce_quant",
        allow_early_resolve=True,
    ):
        tg = tg_idx * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - tg)
        sum_tmp = pl.create_tile([T_TILE, Q_LORA_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        max_tmp = pl.create_tile([T_TILE, Q_LORA_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        proj_chunk0 = pl.load(
            qr_fp32,
            [tg, 0],
            [T_TILE, Q_LORA_TILE],
            valid_shapes=[valid_rows, Q_LORA_TILE],
            target_memory=pl.MemorySpace.Vec,
        )
        sq_sum = pl.row_sum(pl.mul(proj_chunk0, proj_chunk0), sum_tmp)
        gamma_chunk0 = pl.reshape(
            pl.cast(
                pl.load(gamma_cq, [0], [Q_LORA_TILE], target_memory=pl.MemorySpace.Vec),
                target_type=pl.FP32,
            ),
            [1, Q_LORA_TILE],
        )
        amax_raw = pl.row_max(pl.abs(pl.col_expand_mul(proj_chunk0, gamma_chunk0)), max_tmp)
        for qb in pl.range(1, Q_LORA // Q_LORA_TILE):
            col0 = qb * Q_LORA_TILE
            proj_chunk = pl.load(
                qr_fp32,
                [tg, col0],
                [T_TILE, Q_LORA_TILE],
                valid_shapes=[valid_rows, Q_LORA_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            sq_part = pl.row_sum(pl.mul(proj_chunk, proj_chunk), sum_tmp)
            gamma_chunk = pl.reshape(
                pl.cast(
                    pl.load(gamma_cq, [col0], [Q_LORA_TILE], target_memory=pl.MemorySpace.Vec),
                    target_type=pl.FP32,
                ),
                [1, Q_LORA_TILE],
            )
            proj_gamma = pl.col_expand_mul(proj_chunk, gamma_chunk)
            amax_part = pl.row_max(pl.abs(proj_gamma), max_tmp)
            sq_sum = pl.add(sq_sum, sq_part)
            amax_raw = pl.maximum(amax_raw, amax_part)

        sq_mean = pl.mul(sq_sum, 1.0 / Q_LORA)
        inv_rms = pl.rsqrt(pl.add(sq_mean, EPS), high_precision=True)
        amax_norm = pl.mul(inv_rms, amax_raw)
        tile_amax = pl.maximum(amax_norm, INT8_AMAX_EPS)
        scale_quant = pl.mul(pl.recip(tile_amax), INT8_SCALE_MAX)
        scale_dq = pl.mul(tile_amax, 1.0 / INT8_SCALE_MAX)

        pl.store(pl.set_validshape(sq_mean, valid_rows, 1), [tg, 0], qr_sq_mean_diag)
        pl.store(pl.set_validshape(inv_rms, valid_rows, 1), [tg, 0], qr_inv_rms_diag)
        pl.store(pl.set_validshape(amax_raw, valid_rows, 1), [tg, 0], qr_amax_raw_diag)
        pl.store(pl.set_validshape(amax_norm, valid_rows, 1), [tg, 0], qr_amax_norm_diag)
        pl.store(pl.set_validshape(scale_dq, valid_rows, 1), [tg, 0], qr_scale_diag)

        for qa in pl.pipeline(0, Q_LORA, QUANT_TILE, stage=2):
            proj_chunk = pl.load(
                qr_fp32,
                [tg, qa],
                [T_TILE, QUANT_TILE],
                valid_shapes=[valid_rows, QUANT_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            gamma_chunk = pl.reshape(
                pl.cast(
                    pl.load(gamma_cq, [qa], [QUANT_TILE], target_memory=pl.MemorySpace.Vec),
                    target_type=pl.FP32,
                ),
                [1, QUANT_TILE],
            )
            normed = pl.col_expand_mul(pl.row_expand_mul(proj_chunk, inv_rms), gamma_chunk)
            scaled = pl.row_expand_mul(normed, scale_quant)
            quant_i32 = pl.cast(scaled, target_type=pl.INT32, mode="rint")
            quant_f16 = pl.cast(quant_i32, target_type=pl.FP16, mode="round")
            quant_i8 = pl.cast(quant_f16, target_type=pl.INT8, mode="trunc")
            pl.store(pl.set_validshape(quant_i8, valid_rows, QUANT_TILE), [tg, qa], qr_diag)

    return qr_diag


@pl.jit
def q_rope_pipeline_diagnostics_test(
    q_head_input: pl.Tensor[[T_DYN, HEAD_DIM], pl.INT32],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    cos_il_diag: pl.Out[pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32]],
    sin_signed_diag: pl.Out[pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32]],
    swap_idx_diag: pl.Out[pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32]],
    swapped_diag: pl.Out[pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32]],
    rope_rot_diag: pl.Out[pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32]],
):
    """Checkpoint the precomputed Q RoPE metadata and its first consumer."""
    q_head_input.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    cos_il_diag.bind_dynamic(0, T_DYN)
    sin_signed_diag.bind_dynamic(0, T_DYN)
    swap_idx_diag.bind_dynamic(0, T_DYN)
    swapped_diag.bind_dynamic(0, T_DYN)
    rope_rot_diag.bind_dynamic(0, T_DYN)

    t_dim = pl.tensor.dim(q_head_input, 0)
    cos_il_buf = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    sin_signed_buf = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    swap_idx_buf = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.INT32)
    q_rope_i32_buf = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.INT32)

    for qrp_idx in pl.spmd(
        (t_dim + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE,
        name_hint="q_rope_diag_prepare",
        allow_early_resolve=True,
    ):
        qrp_t0 = qrp_idx * Q_ROPE_T_TILE
        valid_rows = pl.min(Q_ROPE_T_TILE, t_dim - qrp_t0)
        ones = pl.tile.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        col = pl.col_expand_mul(
            ones,
            pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
        )
        dup_f = pl.cast(
            pl.cast(pl.mul(col, 0.5), target_type=pl.INT32, mode="trunc"),
            target_type=pl.FP32,
        )
        lane = pl.sub(col, pl.mul(dup_f, 2.0))
        swap_f = pl.sub(pl.add(col, 1.0), pl.mul(lane, 2.0))
        row_seed = pl.cast(
            pl.mul(
                pl.cast(pl.tile.arange(0, [1, Q_ROPE_T_TILE], dtype=pl.INT32), target_type=pl.FP32),
                ROPE_DIM_SCALE,
            ),
            target_type=pl.INT32,
        )
        row_grid = pl.col_expand_mul(
            pl.tile.full([ROPE_DIM, Q_ROPE_T_TILE], dtype=pl.INT32, value=1),
            row_seed,
        )
        row_offset = pl.transpose(row_grid, axis1=0, axis2=1)
        dup_idx = pl.add(pl.cast(dup_f, target_type=pl.INT32), row_offset)
        prepared_swap_idx = pl.add(pl.cast(swap_f, target_type=pl.INT32), row_offset)
        sign = pl.sub(pl.mul(lane, 2.0), 1.0)
        cos_rows = pl.cast(
            pl.load(
                rope_cos,
                [qrp_t0, 0],
                [Q_ROPE_T_TILE, ROPE_DIM],
                valid_shapes=[valid_rows, ROPE_DIM],
                target_memory=pl.MemorySpace.Vec,
            ),
            target_type=pl.FP32,
        )
        sin_rows = pl.cast(
            pl.load(
                rope_sin,
                [qrp_t0, 0],
                [Q_ROPE_T_TILE, ROPE_DIM],
                valid_shapes=[valid_rows, ROPE_DIM],
                target_memory=pl.MemorySpace.Vec,
            ),
            target_type=pl.FP32,
        )
        gather_tmp = pl.create_tile(
            [Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec
        )
        cos_il = pl.tile.gather(cos_rows, dup_idx, gather_tmp)
        sin_signed = pl.mul(pl.tile.gather(sin_rows, dup_idx, gather_tmp), sign)
        pl.store(pl.set_validshape(cos_il, valid_rows, ROPE_DIM), [qrp_t0, 0], cos_il_buf)
        pl.store(pl.set_validshape(sin_signed, valid_rows, ROPE_DIM), [qrp_t0, 0], sin_signed_buf)
        pl.store(pl.set_validshape(prepared_swap_idx, valid_rows, ROPE_DIM), [qrp_t0, 0], swap_idx_buf)
        pl.store(pl.set_validshape(cos_il, valid_rows, ROPE_DIM), [qrp_t0, 0], cos_il_diag)
        pl.store(pl.set_validshape(sin_signed, valid_rows, ROPE_DIM), [qrp_t0, 0], sin_signed_diag)
        pl.store(pl.set_validshape(prepared_swap_idx, valid_rows, ROPE_DIM), [qrp_t0, 0], swap_idx_diag)

    for tg_idx in pl.spmd(
        (t_dim + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE,
        name_hint="q_rope_diag_materialize",
        allow_early_resolve=True,
    ):
        tg = tg_idx * Q_ROPE_T_TILE
        valid_rows = pl.min(Q_ROPE_T_TILE, t_dim - tg)
        q_head_chunk = pl.load(
            q_head_input,
            [tg, 0],
            [Q_ROPE_T_TILE, HEAD_DIM],
            valid_shapes=[valid_rows, HEAD_DIM],
            target_memory=pl.MemorySpace.Vec,
        )
        q_rope_view = q_head_chunk[:, NOPE_DIM:HEAD_DIM]
        pl.store(q_rope_view, [tg, 0], q_rope_i32_buf)

    for tg_idx in pl.spmd(
        (t_dim + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE,
        name_hint="q_rope_diag_consume",
        allow_early_resolve=True,
    ):
        tg = tg_idx * Q_ROPE_T_TILE
        valid_rows = pl.min(Q_ROPE_T_TILE, t_dim - tg)
        q_rope_i32 = pl.load(
            q_rope_i32_buf,
            [tg, 0],
            [Q_ROPE_T_TILE, ROPE_DIM],
            valid_shapes=[valid_rows, ROPE_DIM],
            target_memory=pl.MemorySpace.Vec,
        )
        q_chunk = pl.cast(q_rope_i32, target_type=pl.FP32, mode="none")
        cos_chunk = pl.load(
            cos_il_buf,
            [tg, 0],
            [Q_ROPE_T_TILE, ROPE_DIM],
            valid_shapes=[valid_rows, ROPE_DIM],
            target_memory=pl.MemorySpace.Vec,
        )
        sin_chunk = pl.load(
            sin_signed_buf,
            [tg, 0],
            [Q_ROPE_T_TILE, ROPE_DIM],
            valid_shapes=[valid_rows, ROPE_DIM],
            target_memory=pl.MemorySpace.Vec,
        )
        loaded_swap_idx = pl.load(
            swap_idx_buf,
            [tg, 0],
            [Q_ROPE_T_TILE, ROPE_DIM],
            valid_shapes=[valid_rows, ROPE_DIM],
            target_memory=pl.MemorySpace.Vec,
        )
        gather_tmp = pl.create_tile(
            [Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec
        )
        swapped = pl.tile.gather(q_chunk, loaded_swap_idx, gather_tmp)
        rotated = pl.add(pl.mul(q_chunk, cos_chunk), pl.mul(swapped, sin_chunk))
        pl.store(pl.set_validshape(swapped, valid_rows, ROPE_DIM), [tg, 0], swapped_diag)
        pl.store(pl.set_validshape(rotated, valid_rows, ROPE_DIM), [tg, 0], rope_rot_diag)

    return rope_rot_diag


@pl.jit
def qkv_proj_rope_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Out[pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16]],
    kv: pl.Out[pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16]],
    qr: pl.Out[pl.Tensor[[T_DYN, Q_LORA], pl.INT8]],
    qr_scale: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    q.bind_dynamic(0, T_DYN)
    kv.bind_dynamic(0, T_DYN)
    qr.bind_dynamic(0, T_DYN)
    qr_scale.bind_dynamic(0, T_DYN)

    # Standalone: no rms_norm producer, so the barrier fences nothing (ready on submit).
    late_dep = pl.system.task_dummy(deps=[])
    qkv_proj_rope(
        x,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        rope_cos,
        rope_sin,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
        late_dep,
    )
    return q


def golden_qkv_proj_rope(tensors):
    """Torch reference: Q/KV LoRA + RoPE for an already attention-normalized input."""
    import torch

    x = tensors["x"].float()
    wq_a = tensors["wq_a"].float()
    wq_b = tensors["wq_b"]
    wq_b_scale = tensors["wq_b_scale"].float().view(-1)
    wkv = tensors["wkv"].float()
    rope_cos = tensors["rope_cos"].float()
    rope_sin = tensors["rope_sin"].float()
    gamma_cq = tensors["gamma_cq"].float()
    gamma_ckv = tensors["gamma_ckv"].float()

    def int8_quant_per_row(x):
        rows = x.reshape(-1, x.shape[-1]).float()
        amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = rows * scale_quant
        out_i32 = torch.round(scaled).to(torch.int32)
        out_half = out_i32.to(torch.float16)
        out_i8 = out_half.to(torch.int8)
        return out_i8.reshape_as(x), (1.0 / scale_quant).reshape(*x.shape[:-1], 1)

    def rms_norm(x, gamma, eps=EPS):
        inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        return x * inv * gamma

    def matmul_bf16_input_fp32(a, b):
        a_fp32 = a.to(torch.bfloat16).float()
        b_fp32 = b.to(torch.bfloat16).float()
        return torch.matmul(a_fp32, b_fp32).float()

    def apply_rope(x_rope, cos, sin):
        # x_rope: [T, ..., ROPE_DIM] with interleaved even/odd rotary pairs.
        x_pair = x_rope.unflatten(-1, (-1, 2))
        x_even, x_odd = x_pair[..., 0], x_pair[..., 1]
        cos_v = cos[..., :ROPE_HALF]
        sin_v = sin[..., :ROPE_HALF]
        while cos_v.ndim < x_even.ndim:
            cos_v = cos_v.unsqueeze(-2)
            sin_v = sin_v.unsqueeze(-2)
        y_even = (x_even * cos_v - x_odd * sin_v).to(torch.bfloat16)
        y_odd = (x_even * sin_v + x_odd * cos_v).to(torch.bfloat16)
        return torch.stack([y_even, y_odd], dim=-1).flatten(-2)

    t_dim = x.shape[0]
    token_x = x.view(t_dim, D)

    # Q path
    qr_out = rms_norm(matmul_bf16_input_fp32(token_x, wq_a), gamma_cq)   # [T, Q_LORA]
    # W8A8C16: wq_b W8 per-output-channel int8; qr_out A8 per-token int8.
    # flash: also quantizes wq_a/wkv to fp8 (default Linear dtype).
    qr_i8, qr_scale = int8_quant_per_row(qr_out.float())
    q_i32 = torch.matmul(qr_i8.to(torch.int32), wq_b.to(torch.int32))
    q_full = (q_i32.float() * qr_scale * wq_b_scale.view(1, -1)).view(t_dim, H, HEAD_DIM)
    inv = torch.rsqrt(q_full.square().mean(-1, keepdim=True) + EPS)
    q_full = q_full * inv                                            # per-head RMSNorm (no gamma)
    q_nope = q_full[..., :NOPE_DIM]
    q_rope = apply_rope(q_full[..., NOPE_DIM:], rope_cos, rope_sin)
    q_out = torch.cat([q_nope, q_rope], dim=-1)

    # KV path
    kv_full = rms_norm(matmul_bf16_input_fp32(token_x, wkv), gamma_ckv)  # [T, HEAD_DIM]
    kv_nope = kv_full[..., :NOPE_DIM]
    kv_rope_in = kv_full[..., NOPE_DIM:].unsqueeze(1)               # add a pseudo head dim
    kv_rope = apply_rope(kv_rope_in, rope_cos, rope_sin).squeeze(1)
    kv_out = torch.cat([kv_nope, kv_rope], dim=-1)

    tensors["q"][:]  = q_out.to(torch.bfloat16)
    tensors["kv"][:] = kv_out.to(torch.bfloat16)
    tensors["qr"][:] = qr_i8
    tensors["qr_scale"][:] = qr_scale


def golden_qr_pipeline_diagnostics(tensors):
    """Torch checkpoints matching each small QR projection/quant segment."""
    import torch

    x = tensors["x"].to(torch.bfloat16).float()
    wq_a = tensors["wq_a"].to(torch.bfloat16).float()
    gamma_cq = tensors["gamma_cq"].float()

    qr_proj = torch.matmul(x, wq_a).float()
    qr_sq_mean = qr_proj.square().mean(dim=-1, keepdim=True)
    qr_inv_rms = torch.rsqrt(qr_sq_mean + EPS)
    qr_amax_raw = (qr_proj * gamma_cq).abs().amax(dim=-1, keepdim=True)
    qr_amax_norm = qr_inv_rms * qr_amax_raw
    qr_tile_amax = qr_amax_norm.clamp_min(INT8_AMAX_EPS)
    qr_scale = qr_tile_amax / INT8_SCALE_MAX
    qr_normed = qr_proj * qr_inv_rms * gamma_cq
    qr_i32 = torch.round(qr_normed / qr_scale).to(torch.int32)
    qr_i8 = qr_i32.to(torch.float16).to(torch.int8)

    tensors["qr_proj_diag"][:] = qr_proj
    tensors["qr_sq_mean_diag"][:] = qr_sq_mean
    tensors["qr_inv_rms_diag"][:] = qr_inv_rms
    tensors["qr_amax_raw_diag"][:] = qr_amax_raw
    tensors["qr_amax_norm_diag"][:] = qr_amax_norm
    tensors["qr_scale_diag"][:] = qr_scale
    tensors["qr_diag"][:] = qr_i8


def golden_q_rope_pipeline_diagnostics(tensors):
    """Torch checkpoints matching Q RoPE metadata preparation and consumption."""
    import torch

    q_rope_input = tensors["q_head_input"].float()[:, NOPE_DIM:HEAD_DIM]
    rope_cos = tensors["rope_cos"].float()
    rope_sin = tensors["rope_sin"].float()
    cols = torch.arange(ROPE_DIM, dtype=torch.int64)
    dup = torch.div(cols, 2, rounding_mode="floor")
    swap = torch.bitwise_xor(cols, 1)
    sign = torch.where(cols % 2 == 0, -1.0, 1.0)
    cos_il = rope_cos[:, dup]
    sin_signed = rope_sin[:, dup] * sign
    local_rows = torch.arange(q_rope_input.shape[0], dtype=torch.int64) % Q_ROPE_T_TILE
    swap_idx = local_rows[:, None] * ROPE_DIM + swap[None, :]
    swapped = q_rope_input[:, swap]
    rotated = q_rope_input * cos_il + swapped * sin_signed

    tensors["cos_il_diag"][:] = cos_il
    tensors["sin_signed_diag"][:] = sin_signed
    tensors["swap_idx_diag"][:] = swap_idx.to(torch.int32)
    tensors["swapped_diag"][:] = swapped
    tensors["rope_rot_diag"][:] = rotated


def build_q_rope_diagnostic_specs(T):
    import torch
    from golden import TensorSpec

    return [
        TensorSpec(
            "q_head_input",
            [T, HEAD_DIM],
            torch.int32,
            init_value=lambda: torch.randint(-2048, 2048, [T, HEAD_DIM], dtype=torch.int32),
        ),
        TensorSpec(
            "rope_cos",
            [T, ROPE_DIM],
            torch.bfloat16,
            init_value=lambda: torch.empty([T, ROPE_DIM], dtype=torch.bfloat16).uniform_(-1, 1),
        ),
        TensorSpec(
            "rope_sin",
            [T, ROPE_DIM],
            torch.bfloat16,
            init_value=lambda: torch.empty([T, ROPE_DIM], dtype=torch.bfloat16).uniform_(-1, 1),
        ),
        TensorSpec("cos_il_diag", [T, ROPE_DIM], torch.float32, is_output=True),
        TensorSpec("sin_signed_diag", [T, ROPE_DIM], torch.float32, is_output=True),
        TensorSpec("swap_idx_diag", [T, ROPE_DIM], torch.int32, is_output=True),
        TensorSpec("swapped_diag", [T, ROPE_DIM], torch.float32, is_output=True),
        TensorSpec("rope_rot_diag", [T, ROPE_DIM], torch.float32, is_output=True),
    ]


def build_qr_diagnostic_specs(T):
    import torch
    from golden import TensorSpec

    return [
        TensorSpec(
            "x",
            [T, D],
            torch.bfloat16,
            init_value=lambda: torch.empty([T, D], dtype=torch.bfloat16).uniform_(-1, 1),
        ),
        TensorSpec(
            "wq_a",
            [D, Q_LORA],
            torch.bfloat16,
            init_value=lambda: torch.empty([D, Q_LORA], dtype=torch.bfloat16).uniform_(-0.1, 0.1),
        ),
        TensorSpec(
            "gamma_cq",
            [Q_LORA],
            torch.bfloat16,
            init_value=lambda: torch.empty([Q_LORA], dtype=torch.bfloat16).uniform_(-1, 1),
        ),
        TensorSpec("qr_proj_diag", [T, Q_LORA], torch.float32, is_output=True),
        TensorSpec("qr_sq_mean_diag", [T, 1], torch.float32, is_output=True),
        TensorSpec("qr_inv_rms_diag", [T, 1], torch.float32, is_output=True),
        TensorSpec("qr_amax_raw_diag", [T, 1], torch.float32, is_output=True),
        TensorSpec("qr_amax_norm_diag", [T, 1], torch.float32, is_output=True),
        TensorSpec("qr_scale_diag", [T, 1], torch.float32, is_output=True),
        TensorSpec("qr_diag", [T, Q_LORA], torch.int8, is_output=True),
    ]


def build_tensor_specs(B, S):
    import torch
    from golden import TensorSpec

    T = B * S

    def quant_w_per_output_channel(w):
        amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.view(1, H * HEAD_DIM)
        w_i32 = torch.round(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        w_i8 = w_i32.to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    # Inputs match cann test_mla_prolog_quant_pypto gen_mla_prolog_input_data (uniform).
    def init_x():
        return torch.empty([T, D], dtype=torch.bfloat16).uniform_(-1, 1)
    def init_wq_a():
        return torch.empty([D, Q_LORA], dtype=torch.bfloat16).uniform_(-0.1, 0.1)
    def init_wq_b():
        return torch.empty([Q_LORA, H * HEAD_DIM], dtype=torch.bfloat16).uniform_(-0.1, 0.1)
    def init_wkv():
        return torch.empty([D, HEAD_DIM], dtype=torch.bfloat16).uniform_(-0.1, 0.1)
    def init_cos():
        return torch.empty([T, ROPE_DIM], dtype=torch.bfloat16).uniform_(-1, 1)
    def init_sin():
        return torch.empty([T, ROPE_DIM], dtype=torch.bfloat16).uniform_(-1, 1)
    def init_gamma_cq():
        return torch.empty([Q_LORA], dtype=torch.bfloat16).uniform_(-1, 1)
    def init_gamma_ckv():
        return torch.empty([HEAD_DIM], dtype=torch.bfloat16).uniform_(-1, 1)

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = quant_w_per_output_channel(wq_b_bf16)
    wq_b_scale = wq_b_scale.view(H * HEAD_DIM)

    return [
        TensorSpec("x",         [T, D],                 torch.bfloat16, init_value=init_x),
        TensorSpec("wq_a",      [D, Q_LORA],            torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b",      [Q_LORA, H * HEAD_DIM], torch.int8,     init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [H * HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv",       [D, HEAD_DIM],          torch.bfloat16, init_value=init_wkv),
        TensorSpec("rope_cos",  [T, ROPE_DIM],          torch.bfloat16, init_value=init_cos),
        TensorSpec("rope_sin",  [T, ROPE_DIM],          torch.bfloat16, init_value=init_sin),
        TensorSpec("gamma_cq",  [Q_LORA],               torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM],             torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("q",         [T, H, HEAD_DIM],       torch.bfloat16, is_output=True),
        TensorSpec("kv",        [T, HEAD_DIM],          torch.bfloat16, is_output=True),
        TensorSpec("qr",        [T, Q_LORA],            torch.int8,     is_output=True),
        TensorSpec("qr_scale",  [T, 1],                 torch.float32,  is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    MODES = {
        "decode":  (DECODE_BATCH, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="all",
                        help="Use decode or prefill batch sizes, or 'all' to test both.")
    parser.add_argument("--enable-l2-swimlane", type=int, choices=[0, 1, 2, 4], default=0,
                        help="L2 swimlane level: 0=off, 1=per-kernel AICore timing "
                             "(prints the per-function Task Statistics table), 2=+AICPU timing.")
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=None,
        help="Override the token length used by the standalone test harness.",
    )
    parser.add_argument(
        "--qr-diagnostics-only",
        action="store_true",
        default=False,
        help="Run per-segment QR projection/RMS/amax/quant precision checkpoints.",
    )
    parser.add_argument(
        "--q-rope-diagnostics-only",
        action="store_true",
        default=False,
        help="Run Q RoPE metadata/gather/rotation precision checkpoints.",
    )
    parser.add_argument(
        "--diagnostic-tokens",
        type=int,
        default=None,
        help="Override token count for QR diagnostics, for example 73 to exercise a dynamic tail tile.",
    )
    args = parser.parse_args()

    modes_to_run = list(MODES.keys()) if args.mode == "all" else [args.mode]

    for mode_name in modes_to_run:
        B, S = MODES[mode_name]
        if args.num_tokens is not None:
            if args.num_tokens < 1 or args.num_tokens > T_MAX:
                parser.error(f"--num-tokens must be in [1, {T_MAX}], got {args.num_tokens}")
            B, S = 1, args.num_tokens
        if args.q_rope_diagnostics_only:
            diagnostic_tokens = args.diagnostic_tokens if args.diagnostic_tokens is not None else B * S
            if diagnostic_tokens < 1 or diagnostic_tokens > T_MAX:
                parser.error(f"--diagnostic-tokens must be in [1, {T_MAX}], got {diagnostic_tokens}")
            print(f"--- q_rope_pipeline_diagnostics {mode_name}: T={diagnostic_tokens} ---")
            diag_result = run_jit(
                fn=q_rope_pipeline_diagnostics_test,
                specs=build_q_rope_diagnostic_specs(diagnostic_tokens),
                golden_fn=golden_q_rope_pipeline_diagnostics,
                compare_fn={
                    "cos_il_diag": ratio_allclose(atol=0, rtol=0, max_error_ratio=0),
                    "sin_signed_diag": ratio_allclose(atol=0, rtol=0, max_error_ratio=0),
                    "swap_idx_diag": ratio_allclose(atol=0, rtol=0, max_error_ratio=0),
                    "swapped_diag": ratio_allclose(atol=0, rtol=0, max_error_ratio=0),
                    "rope_rot_diag": ratio_allclose(atol=1e-6, rtol=1e-6, max_error_ratio=0),
                },
                runtime_dir=args.runtime_dir,
                golden_data=args.golden_data,
                compile_cfg=dict(dump_passes=args.dump_passes),
                runtime_cfg=dict(
                    platform=args.platform,
                    device_id=args.device,
                    enable_l2_swimlane=args.enable_l2_swimlane,
                ),
                compile_only=args.compile_only,
            )
            if not diag_result.passed:
                if diag_result.error:
                    print(diag_result.error)
                raise SystemExit(1)
            continue
        if args.qr_diagnostics_only:
            diagnostic_tokens = args.diagnostic_tokens if args.diagnostic_tokens is not None else B * S
            if diagnostic_tokens < 1 or diagnostic_tokens > T_MAX:
                parser.error(f"--diagnostic-tokens must be in [1, {T_MAX}], got {diagnostic_tokens}")
            print(f"--- qr_pipeline_diagnostics {mode_name}: T={diagnostic_tokens} ---")
            diag_result = run_jit(
                fn=qr_pipeline_diagnostics_test,
                specs=build_qr_diagnostic_specs(diagnostic_tokens),
                golden_fn=golden_qr_pipeline_diagnostics,
                compare_fn={
                    "qr_proj_diag": ratio_allclose(atol=1e-3, rtol=5e-3),
                    "qr_sq_mean_diag": ratio_allclose(atol=1e-3, rtol=5e-3, max_error_ratio=0),
                    "qr_inv_rms_diag": ratio_allclose(atol=2.5e-5, rtol=5e-3, max_error_ratio=0),
                    "qr_amax_raw_diag": ratio_allclose(atol=1e-3, rtol=5e-3, max_error_ratio=0),
                    "qr_amax_norm_diag": ratio_allclose(atol=1e-3, rtol=5e-3, max_error_ratio=0),
                    "qr_scale_diag": ratio_allclose(atol=2.5e-5, rtol=5e-3, max_error_ratio=0),
                    "qr_diag": ratio_allclose(atol=1, rtol=0, max_error_ratio=0),
                },
                runtime_dir=args.runtime_dir,
                golden_data=args.golden_data,
                compile_cfg=dict(dump_passes=args.dump_passes),
                runtime_cfg=dict(
                    platform=args.platform,
                    device_id=args.device,
                    enable_l2_swimlane=args.enable_l2_swimlane,
                ),
                compile_only=args.compile_only,
            )
            if not diag_result.passed:
                if diag_result.error:
                    print(diag_result.error)
                raise SystemExit(1)
            continue

        print(f"--- qkv_proj_rope {mode_name}: B={B}, S={S} ---")
        result = run_jit(
            fn=qkv_proj_rope_test,
            specs=build_tensor_specs(B, S),
            golden_fn=golden_qkv_proj_rope,
            # W8A8C16 q_proj adds INT8 quant/dequant round-off before per-head RMSNorm.
            rtol=5e-3,
            atol=5e-3,
            # Precision reference: pypto mla_prolog —
            # cann-recipes-infer/ops/pypto_python/example/test_mla_prolog_pypto.py
            compare_fn={
                "q":        ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "kv":       ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "qr":       ratio_allclose(atol=1, rtol=0, max_error_ratio=0),
                "qr_scale": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            },
            runtime_dir=args.runtime_dir,
            golden_data=args.golden_data,
            compile_cfg=dict(dump_passes=args.dump_passes),
            runtime_cfg=dict(
                platform=args.platform,
                device_id=args.device,
                enable_l2_swimlane=args.enable_l2_swimlane,
            ),
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
