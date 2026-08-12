# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared target LM head plus rank-256 sequential DSpark Markov sampling."""

import pypto.language as pl

from config import (
    DSPARK_MARKOV_RANK,
    DSPARK_MAX_BATCH,
    DSPARK_MOE_TOKENS,
    DSPARK_QUERY_PAD,
    DSPARK_QUERY_WIDTH,
    DSPARK_SUPPORTED_BATCHES,
    FLASH as M,
)
from markov_head import markov_head


B_DYN = pl.dynamic("DSPARK_MARKOV_B_DYN")

# model config
D = M.hidden_size
VOCAB = M.vocab_size
EPS = M.rms_norm_eps

# tiling
HIDDEN_TILE = 512
RMS_M_TILE = 8
LM_M_TILE = 16
LM_N_TILE = 128
LM_K_TILE = 256
MARKOV_M_TILE = DSPARK_MAX_BATCH
GREEDY_VOCAB_CHUNK = 256
GREEDY_NUM_CHUNKS = VOCAB // GREEDY_VOCAB_CHUNK
GREEDY_CHUNK_PAD = 512
GREEDY_TOPK = 16
NEG_INF = -3.402823e38

assert D % HIDDEN_TILE == 0
assert D % LM_K_TILE == 0
assert VOCAB % LM_N_TILE == 0
assert VOCAB % GREEDY_VOCAB_CHUNK == 0
assert GREEDY_NUM_CHUNKS <= GREEDY_CHUNK_PAD
assert DSPARK_MARKOV_RANK == LM_K_TILE


@pl.jit.inline
def compute_base_logits(
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    base_logits: pl.Tensor[[DSPARK_MOE_TOKENS, VOCAB], pl.FP32],
):
    batch = pl.tensor.dim(head_hidden, 0)
    active_tokens = batch * DSPARK_QUERY_WIDTH
    padded_tokens = batch * DSPARK_QUERY_PAD
    hidden_flat = pl.reshape(head_hidden, [active_tokens, D])
    padded_hidden = pl.create_tensor([DSPARK_MOE_TOKENS, D], dtype=pl.BF16)
    hidden_blocks = D // HIDDEN_TILE
    with pl.spmd(
        (DSPARK_MOE_TOKENS // RMS_M_TILE) * hidden_blocks,
        name_hint="dspark_hidden_zero",
    ):
        zero_task = pl.tile.get_block_idx()
        zero_row = (zero_task // hidden_blocks) * RMS_M_TILE
        zero_col = (zero_task % hidden_blocks) * HIDDEN_TILE
        padded_hidden[
            zero_row : zero_row + RMS_M_TILE,
            zero_col : zero_col + HIDDEN_TILE,
        ] = pl.full([RMS_M_TILE, HIDDEN_TILE], dtype=pl.BF16, value=0.0)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_hidden_pad"):
        for token in pl.range(active_tokens):
            padded_hidden[token : token + 1, :] = hidden_flat[token : token + 1, :]

    # Keep this normalization local. Passing a dynamically sized temporary to
    # the generic rms_norm inline kernel loses its inferred tensor metadata
    # during JIT specialization.
    normalized = pl.create_tensor([DSPARK_MOE_TOKENS, D], dtype=pl.BF16)
    with pl.spmd(
        padded_tokens // RMS_M_TILE,
        name_hint="dspark_final_norm",
        allow_early_resolve=True,
    ):
        row_block = pl.tile.get_block_idx()
        row_offset = row_block * RMS_M_TILE
        square_sum = pl.full([1, RMS_M_TILE], dtype=pl.FP32, value=0.0)
        for hidden_block in pl.pipeline(D // HIDDEN_TILE, stage=2):
            hidden_offset = hidden_block * HIDDEN_TILE
            rms_hidden_tile = pl.cast(
                padded_hidden[
                    row_offset : row_offset + RMS_M_TILE,
                    hidden_offset : hidden_offset + HIDDEN_TILE,
                ],
                target_type=pl.FP32,
            )
            square_sum = pl.add(
                square_sum,
                pl.reshape(
                    pl.row_sum(pl.mul(rms_hidden_tile, rms_hidden_tile)),
                    [1, RMS_M_TILE],
                ),
            )
        inv_rms = pl.reshape(
            pl.rsqrt(
                pl.add(pl.mul(square_sum, 1.0 / D), EPS),
                high_precision=True,
            ),
            [RMS_M_TILE, 1],
        )
        for hidden_block in pl.pipeline(D // HIDDEN_TILE, stage=2):
            hidden_offset = hidden_block * HIDDEN_TILE
            rms_hidden_tile = pl.cast(
                padded_hidden[
                    row_offset : row_offset + RMS_M_TILE,
                    hidden_offset : hidden_offset + HIDDEN_TILE,
                ],
                target_type=pl.FP32,
            )
            norm_tile = pl.cast(
                pl.reshape(
                    final_norm_weight[
                        hidden_offset : hidden_offset + HIDDEN_TILE
                    ],
                    [1, HIDDEN_TILE],
                ),
                target_type=pl.FP32,
            )
            normalized[
                row_offset : row_offset + RMS_M_TILE,
                hidden_offset : hidden_offset + HIDDEN_TILE,
            ] = pl.cast(
                pl.col_expand_mul(
                    pl.row_expand_mul(rms_hidden_tile, inv_rms),
                    norm_tile,
                ),
                target_type=pl.BF16,
                mode="rint",
            )
    row_blocks = padded_tokens // LM_M_TILE
    vocab_blocks = VOCAB // LM_N_TILE
    for task in pl.spmd(
        row_blocks * vocab_blocks,
        name_hint="dspark_base_logits",
    ):
        row_block = task // vocab_blocks
        vocab_block = task - row_block * vocab_blocks
        row_offset = row_block * LM_M_TILE
        vocab_offset = vocab_block * LM_N_TILE
        hidden_tile = normalized[
            row_offset : row_offset + LM_M_TILE,
            0:LM_K_TILE,
        ]
        weight_tile = lm_head_weight[
            vocab_offset : vocab_offset + LM_N_TILE,
            0:LM_K_TILE,
        ]
        logits_acc = pl.matmul(
            hidden_tile,
            weight_tile,
            b_trans=True,
            out_dtype=pl.FP32,
        )
        for hidden_offset in pl.pipeline(
            LM_K_TILE,
            D,
            LM_K_TILE,
            stage=2,
        ):
            hidden_tile = normalized[
                row_offset : row_offset + LM_M_TILE,
                hidden_offset : hidden_offset + LM_K_TILE,
            ]
            weight_tile = lm_head_weight[
                vocab_offset : vocab_offset + LM_N_TILE,
                hidden_offset : hidden_offset + LM_K_TILE,
            ]
            logits_acc = pl.matmul_acc(
                logits_acc,
                hidden_tile,
                weight_tile,
                b_trans=True,
            )
        base_logits[
            row_offset : row_offset + LM_M_TILE,
            vocab_offset : vocab_offset + LM_N_TILE,
        ] = logits_acc
    return base_logits


@pl.jit.inline
def greedy_markov_step(
    base_logits: pl.Tensor[[DSPARK_MOE_TOKENS, VOCAB], pl.FP32],
    anchor_token_ids: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    draft_token_ids: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.INT32],
    step: pl.Scalar[pl.INT32],
):
    batch = pl.tensor.dim(anchor_token_ids, 0)
    previous_token_ids = pl.create_tensor([batch], dtype=pl.INT64)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_markov_previous_tokens"):
        for request in pl.range(batch):
            previous_token = pl.read(anchor_token_ids, [request])
            if step > 0:
                previous_token = pl.cast(
                    pl.read(draft_token_ids, [request, step - 1]),
                    pl.INT64,
                )
            pl.write(previous_token_ids, [request], previous_token)

    step_base_logits = pl.create_tensor([MARKOV_M_TILE, VOCAB], dtype=pl.FP32)
    for gather_task in pl.spmd(
        MARKOV_M_TILE * GREEDY_NUM_CHUNKS,
        name_hint="dspark_markov_base_gather",
    ):
        request = gather_task // GREEDY_NUM_CHUNKS
        chunk = gather_task - request * GREEDY_NUM_CHUNKS
        if request < batch:
            source_row = request * DSPARK_QUERY_WIDTH + step
            vocab_offset = chunk * GREEDY_VOCAB_CHUNK
            step_base_logits[
                request : request + 1,
                vocab_offset : vocab_offset + GREEDY_VOCAB_CHUNK,
            ] = pl.slice(
                base_logits,
                [1, GREEDY_VOCAB_CHUNK],
                [pl.cast(source_row, pl.INDEX), vocab_offset],
            )

    markov_bias = pl.create_tensor([batch, VOCAB], dtype=pl.FP32)
    markov_embedding = pl.create_tensor([batch, DSPARK_MARKOV_RANK], dtype=pl.BF16)
    markov_head(previous_token_ids, markov_w1, markov_w2, markov_bias, markov_embedding)

    for request in pl.spmd(MARKOV_M_TILE, name_hint="dspark_markov_greedy"):
        if request < batch:
            chunk_indices = pl.arange(0, [1, GREEDY_VOCAB_CHUNK], dtype=pl.UINT32)
            chunk_maxima = pl.full(
                [1, GREEDY_CHUNK_PAD],
                dtype=pl.FP32,
                value=NEG_INF,
            )
            chunk_token_ids = pl.full(
                [1, GREEDY_CHUNK_PAD],
                dtype=pl.INT32,
                value=0,
            )
            for chunk in pl.range(GREEDY_NUM_CHUNKS):
                vocab_offset = chunk * GREEDY_VOCAB_CHUNK
                scores = pl.add(
                    step_base_logits[
                        request : request + 1,
                        vocab_offset : vocab_offset + GREEDY_VOCAB_CHUNK,
                    ],
                    markov_bias[
                        request : request + 1,
                        vocab_offset : vocab_offset + GREEDY_VOCAB_CHUNK,
                    ],
                )
                sorted_pairs = pl.sort32(scores, chunk_indices)
                sorted_pairs = pl.mrgsort(sorted_pairs, block_len=64)
                sorted_pairs = pl.mrgsort(
                    sorted_pairs[:, 0:GREEDY_VOCAB_CHUNK],
                    sorted_pairs[
                        :,
                        GREEDY_VOCAB_CHUNK : 2 * GREEDY_VOCAB_CHUNK,
                    ],
                )
                top_pair = sorted_pairs[:, 0 : 2 * GREEDY_TOPK]
                top_values = pl.gather(
                    top_pair,
                    mask_pattern=pl.tile.MaskPattern.P0101,
                )
                top_indices = pl.gather(
                    top_pair,
                    mask_pattern=pl.tile.MaskPattern.P1010,
                    output_dtype=pl.INT32,
                )
                pl.write(
                    chunk_maxima,
                    [0, chunk],
                    pl.read(top_values, [0, 0]),
                )
                pl.write(
                    chunk_token_ids,
                    [0, chunk],
                    pl.cast(vocab_offset, pl.INT32) + pl.read(top_indices, [0, 0]),
                )

            maxima_indices = pl.arange(
                0,
                [1, GREEDY_CHUNK_PAD],
                dtype=pl.UINT32,
            )
            sorted_maxima = pl.sort32(chunk_maxima, maxima_indices)
            sorted_maxima = pl.mrgsort(sorted_maxima, block_len=64)
            sorted_maxima = pl.mrgsort(sorted_maxima, block_len=256)
            top_maximum_pair = sorted_maxima[:, 0 : 2 * GREEDY_TOPK]
            top_maximum_values = pl.gather(
                top_maximum_pair,
                mask_pattern=pl.tile.MaskPattern.P0101,
            )
            best_value = pl.read(top_maximum_values, [0, 0])
            winning_token = pl.cast(0, pl.INT32)
            for chunk in pl.range(GREEDY_NUM_CHUNKS):
                reverse_chunk = GREEDY_NUM_CHUNKS - 1 - chunk
                if pl.read(chunk_maxima, [0, reverse_chunk]) == best_value:
                    winning_token = pl.read(chunk_token_ids, [0, reverse_chunk])
            pl.write(
                draft_token_ids,
                [request, step],
                winning_token,
            )
    return draft_token_ids


@pl.jit.inline
def markov_sample_impl(
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    anchor_token_ids: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    draft_token_ids: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.INT32],
):
    batch = pl.tensor.dim(head_hidden, 0)
    base_logits = pl.create_tensor(
        [DSPARK_MOE_TOKENS, VOCAB],
        dtype=pl.FP32,
    )
    compute_base_logits(
        head_hidden,
        final_norm_weight,
        lm_head_weight,
        base_logits,
    )
    greedy_markov_step(
        base_logits, anchor_token_ids, markov_w1, markov_w2, draft_token_ids, pl.cast(0, pl.INT32)
    )
    greedy_markov_step(
        base_logits, anchor_token_ids, markov_w1, markov_w2, draft_token_ids, pl.cast(1, pl.INT32)
    )
    greedy_markov_step(
        base_logits, anchor_token_ids, markov_w1, markov_w2, draft_token_ids, pl.cast(2, pl.INT32)
    )
    greedy_markov_step(
        base_logits, anchor_token_ids, markov_w1, markov_w2, draft_token_ids, pl.cast(3, pl.INT32)
    )
    greedy_markov_step(
        base_logits, anchor_token_ids, markov_w1, markov_w2, draft_token_ids, pl.cast(4, pl.INT32)
    )
    greedy_markov_step(
        base_logits, anchor_token_ids, markov_w1, markov_w2, draft_token_ids, pl.cast(5, pl.INT32)
    )
    greedy_markov_step(
        base_logits, anchor_token_ids, markov_w1, markov_w2, draft_token_ids, pl.cast(6, pl.INT32)
    )
    return draft_token_ids


@pl.jit
def markov_sample(
    head_hidden: pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    anchor_token_ids: pl.Tensor[[B_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB, DSPARK_MARKOV_RANK], pl.BF16],
    draft_token_ids: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH], pl.INT32]],
):
    head_hidden.bind_dynamic(0, B_DYN)
    anchor_token_ids.bind_dynamic(0, B_DYN)
    draft_token_ids.bind_dynamic(0, B_DYN)
    return markov_sample_impl(
        head_hidden,
        final_norm_weight,
        lm_head_weight,
        anchor_token_ids,
        markov_w1,
        markov_w2,
        draft_token_ids,
    )


def build_tensor_specs(batch: int):
    """Build a deterministic zero-logit validation case for one supported batch."""
    import torch
    from golden import TensorSpec

    if batch not in DSPARK_SUPPORTED_BATCHES:
        raise ValueError(f"unsupported DSpark batch {batch}; expected one of {DSPARK_SUPPORTED_BATCHES}")
    return [
        TensorSpec("head_hidden", [batch, DSPARK_QUERY_WIDTH, D], torch.bfloat16, init_value=0),
        TensorSpec("final_norm_weight", [D], torch.bfloat16, init_value=1),
        TensorSpec("lm_head_weight", [VOCAB, D], torch.bfloat16, init_value=0),
        TensorSpec("anchor_token_ids", [batch], torch.int64, init_value=0),
        TensorSpec("markov_w1", [VOCAB, DSPARK_MARKOV_RANK], torch.bfloat16, init_value=0),
        TensorSpec("markov_w2", [VOCAB, DSPARK_MARKOV_RANK], torch.bfloat16, init_value=0),
        TensorSpec(
            "draft_token_ids",
            [batch, DSPARK_QUERY_WIDTH],
            torch.int32,
            is_output=True,
        ),
    ]


def golden_zero_markov(tensors):
    """Validate the deterministic all-zero logits and Markov-bias case."""
    tensors["draft_token_ids"].zero_()


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser(description="Validate the DeepSeek V4 DSpark Markov sampler.")
    parser.add_argument("--batch", type=int, choices=DSPARK_SUPPORTED_BATCHES, default=4)
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--dump-passes", action="store_true")
    args = parser.parse_args()

    result = run_jit(
        fn=markov_sample,
        specs=build_tensor_specs(args.batch),
        golden_fn=golden_zero_markov,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=0.0,
        atol=0.0,
        compile_only=args.compile_only,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
