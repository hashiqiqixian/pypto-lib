# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Local FP32 attention entries for an existing host-owned TP worker.

Select parallelism before importing any shape-specialized model kernels. The
caller owns persistent packed payload/scale pools, page metadata and any TP
reduction; these entries contain no distributed windows or worker lifecycle.
"""

import threading

import pypto.language as pl

C = None
_INITIAL_PARALLELISM = None
_LOAD_LOCK = threading.RLock()


def load_kernel_configuration(tp_size=1, ep_size=2):
    """Configure host-owned kernels before config reads unrelated server CLI flags."""
    global C, _INITIAL_PARALLELISM
    with _LOAD_LOCK:
        _INITIAL_PARALLELISM = tp_size, ep_size
        try:
            from models.deepseek_v4_1_flash import config
            config.configure_kernel_parallelism(tp_size, ep_size)
            C = config
            return config
        finally:
            _INITIAL_PARALLELISM = None


def load_local_attention_kernels(tp_size=1, ep_size=2):
    """Return A5 L2 entries; reject incompatible configurations in one process."""
    with _LOAD_LOCK:
        load_kernel_configuration(tp_size, ep_size)
        # JIT caches retain artifact paths. Give each provider its own entries
        # so closing one worker cannot invalidate the next worker's cache.
        return _make_kernels()


def load_local_cache_kernels():
    """Bounded byte transfers for runtimes whose public copy API accepts only base pointers."""
    with _LOAD_LOCK:
        return _make_cache_kernels()


def _make_cache_kernels():
    pool_bytes = pl.dynamic("V41_POOL_BYTES")
    page_bytes = pl.dynamic("V41_PAGE_BYTES")
    tile_bytes = 16384

    @pl.jit
    def zero_cache(
        pool: pl.InOut[pl.Tensor[[1, pool_bytes], pl.UINT8]],
        size: pl.Scalar[pl.INDEX],
    ):
        for block in pl.parallel((size + tile_bytes - 1) // tile_bytes):
            start = block * tile_bytes
            with pl.at(level=pl.Level.CORE_GROUP):
                zeros = pl.tile.full([1, tile_bytes], dtype=pl.INT8, value=0)
                zeros = pl.reinterpret_view(zeros, pl.UINT8)
                zeros = pl.set_validshape(zeros, 1, pl.min(tile_bytes, size - start))
                pool = pl.store(zeros, [0, start], pool)

    @pl.jit
    def read_cache(
        pool: pl.Tensor[[1, pool_bytes], pl.UINT8],
        page: pl.Out[pl.Tensor[[1, page_bytes], pl.UINT8]],
        offset: pl.Scalar[pl.INDEX],
        size: pl.Scalar[pl.INDEX],
    ):
        for block in pl.parallel((size + tile_bytes - 1) // tile_bytes):
            start = block * tile_bytes
            with pl.at(level=pl.Level.CORE_GROUP):
                value = pl.load(pool, [0, offset + start], [1, tile_bytes],
                                valid_shape=[1, pl.min(tile_bytes, size - start)])
                page = pl.store(value, [0, start], page)

    @pl.jit
    def write_cache(
        page: pl.Tensor[[1, page_bytes], pl.UINT8],
        pool: pl.InOut[pl.Tensor[[1, pool_bytes], pl.UINT8]],
        offset: pl.Scalar[pl.INDEX],
        size: pl.Scalar[pl.INDEX],
    ):
        for block in pl.parallel((size + tile_bytes - 1) // tile_bytes):
            start = block * tile_bytes
            with pl.at(level=pl.Level.CORE_GROUP):
                value = pl.load(page, [0, start], [1, tile_bytes],
                                valid_shape=[1, pl.min(tile_bytes, size - start)])
                pool = pl.store(value, [0, offset + start], pool)

    return {"zero": zero_cache, "read": read_cache, "write": write_cache}


def _make_kernels():
    from models.deepseek_v4_1_flash.decode_swa import decode_swa_partial
    from models.deepseek_v4_1_flash.decode_c2a_full import c2a_full_partial
    from models.deepseek_v4_1_flash.decode_c2a_reuse import c2a_reuse_partial
    from models.deepseek_v4_1_flash.prefill_c1a_common import prefill_c1a_partial
    from models.deepseek_v4_1_flash.prefill_c1a_full import make_c1a_full_partial, paged_indexer as full_indexer
    from models.deepseek_v4_1_flash.prefill_c1a_reindex import make_c1a_reindex_partial, paged_indexer

    c1a_full_partial = make_c1a_full_partial(full_indexer)
    c1a_reindex_partial = make_c1a_reindex_partial(paged_indexer)

    @pl.jit
    def swa(
        x: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
        wq_a: pl.Tensor[[C.D, C.Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[C.D // 32, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[C.Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[C.D // 32, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[C.T_DYN], pl.INT64],
        window_indices: pl.Tensor[[C.T_DYN, 128], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0]],
        output: pl.Out[pl.Tensor[[C.T_DYN, C.D], pl.FP32]],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        cache_ready = pl.system.task_dummy(deps=[])
        decode_swa_partial(
            x,
            wq_a,
            wq_a_scale,
            q_norm_weight,
            wq_b,
            wq_b_scale,
            wkv,
            wkv_scale,
            kv_norm_weight,
            attn_sink,
            wo_a,
            wo_b,
            wo_b_scale,
            rope_cos,
            rope_sin,
            window_slots,
            window_indices,
            window_cache,
            window_cache_scale,
            output,
            num_tokens,
            cache_ready,
        )


    @pl.jit
    def c2a_full(
        x: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
        wq_a: pl.Tensor[[C.D, C.Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[C.D // 32, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[C.Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[C.D // 32, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[C.T_DYN], pl.INT64],
        window_indices: pl.Tensor[[C.T_DYN, 128], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 32], pl.FP8E8M0]],
        compressed_cache: pl.InOut[pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2], pl.UINT8]],
        compressed_cache_scale: pl.InOut[pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN]],
        request_ids: pl.Tensor[[C.T_DYN], pl.INT32],
        compressed_lens: pl.Tensor[[C.T_DYN], pl.INT32],
        index_cache: pl.InOut[pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // 2], pl.UINT8]],
        index_cache_scale: pl.InOut[pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP], pl.FP8E8M0]],
        index_block_table: pl.Tensor[[C.B_DYN, C.TABLE_DYN], pl.INT32],
        position_ids: pl.Tensor[[C.T_DYN], pl.INT32],
        compressed_rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        compressed_rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        compressor_wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP32],
        compressor_wgate: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP32],
        compressor_state_rows: pl.Tensor[[C.T_DYN], pl.INT64],
        compressor_state: pl.InOut[pl.Tensor[[C.MAX_BATCH_PER_DP, C.STATE_HEADS, C.HEAD_DIM], pl.FP32]],
        compressor_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
        compressed_slots: pl.Tensor[[C.T_DYN], pl.INT64],
        index_wk: pl.Tensor[[C.HEAD_DIM, C.INDEX_DIM], pl.BF16],
        index_norm_weight: pl.Tensor[[C.INDEX_DIM], pl.BF16],
        index_wq_b: pl.Tensor[[C.Q_LORA, C.INDEX_H * C.INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
        index_weights_proj: pl.Tensor[[C.D, C.INDEX_H], pl.BF16],
        topk_indices: pl.Out[pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32]],
        partial: pl.Out[pl.Tensor[[C.T_DYN, C.D], pl.FP32]],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        cache_ready = pl.system.task_dummy(deps=[])
        c2a_full_partial(
            x,
            wq_a,
            wq_a_scale,
            q_norm_weight,
            wq_b,
            wq_b_scale,
            wkv,
            wkv_scale,
            kv_norm_weight,
            attn_sink,
            wo_a,
            wo_b,
            wo_b_scale,
            rope_cos,
            rope_sin,
            window_slots,
            window_indices,
            window_cache,
            window_cache_scale,
            compressed_cache,
            compressed_cache_scale,
            request_ids,
            compressed_lens,
            index_cache,
            index_cache_scale,
            index_block_table,
            position_ids,
            compressed_rope_cos,
            compressed_rope_sin,
            compressor_wkv,
            compressor_wgate,
            compressor_state_rows,
            compressor_state,
            compressor_norm_weight,
            compressed_slots,
            index_wk,
            index_norm_weight,
            index_wq_b,
            index_wq_b_scale,
            index_weights_proj,
            topk_indices,
            partial,
            num_tokens,
            cache_ready,
        )


    @pl.jit
    def c2a_reuse(
        x: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
        wq_a: pl.Tensor[[C.D, C.Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[C.D // 32, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[C.Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[C.D // 32, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[C.T_DYN], pl.INT64],
        window_indices: pl.Tensor[[C.T_DYN, 128], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 32], pl.FP8E8M0]],
        compressed_cache: pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2], pl.UINT8],
        compressed_cache_scale: pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN],
        compressed_indices: pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32],
        partial: pl.Out[pl.Tensor[[C.T_DYN, C.D], pl.FP32]],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        cache_ready = pl.system.task_dummy(deps=[])
        c2a_reuse_partial(
            x,
            wq_a,
            wq_a_scale,
            q_norm_weight,
            wq_b,
            wq_b_scale,
            wkv,
            wkv_scale,
            kv_norm_weight,
            attn_sink,
            wo_a,
            wo_b,
            wo_b_scale,
            rope_cos,
            rope_sin,
            window_slots,
            window_indices,
            window_cache,
            window_cache_scale,
            compressed_cache,
            compressed_cache_scale,
            compressed_indices,
            partial,
            num_tokens,
            cache_ready,
        )


    @pl.jit
    def c1a_full(
        x: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
        wq_a: pl.Tensor[[C.D, C.Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[C.D // 32, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[C.Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[C.D // 32, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[C.T_DYN], pl.INT64],
        window_indices: pl.Tensor[[C.T_DYN, 128], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0]],
        compressed_cache: pl.InOut[pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2], pl.UINT8]],
        compressed_cache_scale: pl.InOut[pl.Tensor[ [C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN ]],
        request_ids: pl.Tensor[[C.T_DYN], pl.INT32],
        compressed_lens: pl.Tensor[[C.T_DYN], pl.INT32],
        index_cache: pl.InOut[pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // 2], pl.UINT8]],
        index_cache_scale: pl.InOut[pl.Tensor[ [C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP], pl.FP8E8M0 ]],
        index_block_table: pl.Tensor[[C.B_DYN, C.TABLE_DYN], pl.INT32],
        compressed_rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        compressed_rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        compressor_wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.BF16],
        compressor_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
        compressed_slots: pl.Tensor[[C.T_DYN], pl.INT64],
        index_wk: pl.Tensor[[C.HEAD_DIM, C.INDEX_DIM], pl.BF16],
        index_norm_weight: pl.Tensor[[C.INDEX_DIM], pl.BF16],
        index_wq_b: pl.Tensor[[C.Q_LORA, C.INDEX_H * C.INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
        index_weights_proj: pl.Tensor[[C.D, C.INDEX_H], pl.BF16],
        topk_indices: pl.Out[pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32]],
        candidate_mask: pl.Out[pl.Tensor[[C.T_DYN, C.CMP_POSITIONS_DYN], pl.UINT8]],
        output: pl.Out[pl.Tensor[[C.T_DYN, C.D], pl.FP32]],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        c1a_full_partial(
            x,
            wq_a,
            wq_a_scale,
            q_norm_weight,
            wq_b,
            wq_b_scale,
            wkv,
            wkv_scale,
            kv_norm_weight,
            attn_sink,
            wo_a,
            wo_b,
            wo_b_scale,
            rope_cos,
            rope_sin,
            window_slots,
            window_indices,
            window_cache,
            window_cache_scale,
            compressed_cache,
            compressed_cache_scale,
            request_ids,
            compressed_lens,
            index_cache,
            index_cache_scale,
            index_block_table,
            compressed_rope_cos,
            compressed_rope_sin,
            compressor_wkv,
            compressor_norm_weight,
            compressed_slots,
            index_wk,
            index_norm_weight,
            index_wq_b,
            index_wq_b_scale,
            index_weights_proj,
            topk_indices,
            candidate_mask,
            output,
            num_tokens,
        )


    @pl.jit
    def c1a_reindex(
        x: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
        wq_a: pl.Tensor[[C.D, C.Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[C.D // 32, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[C.Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[C.D // 32, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[C.T_DYN], pl.INT64],
        window_indices: pl.Tensor[[C.T_DYN, 128], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0]],
        compressed_cache: pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2], pl.UINT8],
        compressed_cache_scale: pl.Tensor[ [C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN ],
        request_ids: pl.Tensor[[C.T_DYN], pl.INT32],
        compressed_lens: pl.Tensor[[C.T_DYN], pl.INT32],
        index_cache: pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // 2], pl.UINT8],
        index_cache_scale: pl.Tensor[ [C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP], pl.FP8E8M0 ],
        index_block_table: pl.Tensor[[C.B_DYN, C.TABLE_DYN], pl.INT32],
        candidate_mask: pl.Tensor[[C.T_DYN, C.CMP_POSITIONS_DYN], pl.UINT8],
        index_wq_b: pl.Tensor[[C.Q_LORA, C.INDEX_H * C.INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
        index_weights_proj: pl.Tensor[[C.D, C.INDEX_H], pl.BF16],
        topk_indices: pl.Out[pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32]],
        output: pl.Out[pl.Tensor[[C.T_DYN, C.D], pl.FP32]],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        c1a_reindex_partial(
            x,
            wq_a,
            wq_a_scale,
            q_norm_weight,
            wq_b,
            wq_b_scale,
            wkv,
            wkv_scale,
            kv_norm_weight,
            attn_sink,
            wo_a,
            wo_b,
            wo_b_scale,
            rope_cos,
            rope_sin,
            window_slots,
            window_indices,
            window_cache,
            window_cache_scale,
            compressed_cache,
            compressed_cache_scale,
            request_ids,
            compressed_lens,
            index_cache,
            index_cache_scale,
            index_block_table,
            candidate_mask,
            index_wq_b,
            index_wq_b_scale,
            index_weights_proj,
            topk_indices,
            output,
            num_tokens,
        )


    @pl.jit
    def c1a_reuse(
        x: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
        wq_a: pl.Tensor[[C.D, C.Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[C.D // 32, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[C.Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[C.D // 32, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[C.T_DYN], pl.INT64],
        window_indices: pl.Tensor[[C.T_DYN, 128], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[pl.Tensor[ [C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0, ]],
        compressed_cache: pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2], pl.UINT8],
        compressed_cache_scale: pl.Tensor[ [C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN, ],
        compressed_indices: pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32],
        output: pl.Out[pl.Tensor[[C.T_DYN, C.D], pl.FP32]],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        prefill_c1a_partial(
            x,
            wq_a,
            wq_a_scale,
            q_norm_weight,
            wq_b,
            wq_b_scale,
            wkv,
            wkv_scale,
            kv_norm_weight,
            attn_sink,
            wo_a,
            wo_b,
            wo_b_scale,
            rope_cos,
            rope_sin,
            window_slots,
            window_indices,
            window_cache,
            window_cache_scale,
            compressed_cache,
            compressed_cache_scale,
            compressed_indices,
            output,
            num_tokens,
        )

    return {
        "swa": swa,
        "c2a_full": c2a_full,
        "c2a_reuse": c2a_reuse,
        "c1a_full": c1a_full,
        "c1a_reindex": c1a_reindex,
        "c1a_reuse": c1a_reuse,
    }


# Stable host ABI order, independent of PyPTO private introspection helpers.
_COMMON_ARGUMENT_NAMES = (
    "x",
    "wq_a",
    "wq_a_scale",
    "q_norm_weight",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "wkv_scale",
    "kv_norm_weight",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "rope_cos",
    "rope_sin",
    "window_slots",
    "window_indices",
    "window_cache",
    "window_cache_scale",
)
ARGUMENT_NAMES = {
    "swa": _COMMON_ARGUMENT_NAMES + (
        "output",
        "num_tokens",
    ),
    "c2a_full": _COMMON_ARGUMENT_NAMES + (
        "compressed_cache",
        "compressed_cache_scale",
        "request_ids",
        "compressed_lens",
        "index_cache",
        "index_cache_scale",
        "index_block_table",
        "position_ids",
        "compressed_rope_cos",
        "compressed_rope_sin",
        "compressor_wkv",
        "compressor_wgate",
        "compressor_state_rows",
        "compressor_state",
        "compressor_norm_weight",
        "compressed_slots",
        "index_wk",
        "index_norm_weight",
        "index_wq_b",
        "index_wq_b_scale",
        "index_weights_proj",
        "topk_indices",
        "partial",
        "num_tokens",
    ),
    "c2a_reuse": _COMMON_ARGUMENT_NAMES + (
        "compressed_cache",
        "compressed_cache_scale",
        "compressed_indices",
        "partial",
        "num_tokens",
    ),
    "c1a_full": _COMMON_ARGUMENT_NAMES + (
        "compressed_cache",
        "compressed_cache_scale",
        "request_ids",
        "compressed_lens",
        "index_cache",
        "index_cache_scale",
        "index_block_table",
        "compressed_rope_cos",
        "compressed_rope_sin",
        "compressor_wkv",
        "compressor_norm_weight",
        "compressed_slots",
        "index_wk",
        "index_norm_weight",
        "index_wq_b",
        "index_wq_b_scale",
        "index_weights_proj",
        "topk_indices",
        "candidate_mask",
        "output",
        "num_tokens",
    ),
    "c1a_reindex": _COMMON_ARGUMENT_NAMES + (
        "compressed_cache",
        "compressed_cache_scale",
        "request_ids",
        "compressed_lens",
        "index_cache",
        "index_cache_scale",
        "index_block_table",
        "candidate_mask",
        "index_wq_b",
        "index_wq_b_scale",
        "index_weights_proj",
        "topk_indices",
        "output",
        "num_tokens",
    ),
    "c1a_reuse": _COMMON_ARGUMENT_NAMES + (
        "compressed_cache",
        "compressed_cache_scale",
        "compressed_indices",
        "output",
        "num_tokens",
    ),
}
