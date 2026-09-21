# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded device validation of query chunks, key waves, and candidate publication."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import pypto.language as pl

from golden import ScalarSpec, TensorSpec, run
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.golden import paged_indexer as reference_indexer, select_candidate_blocks
from models.deepseek_v4_1_flash.prefill_c1a_indexer import make_paged_indexer
from models.deepseek_v4_1_flash.quantization import dequantize_mxfp4_cache, pack_mx_b_scale


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("full", "reindex"), default="full")
parser.add_argument("--compile-only", action="store_true")
parser.add_argument("--cpu-only", action="store_true")
parser.add_argument("--active", type=int, default=7)
parser.add_argument("--budget", type=int, default=196608)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--dep-gen", action="store_true")
args = parser.parse_args()
torch.set_num_threads(2)
use_candidates = args.mode == "reindex"
indexer = make_paged_indexer(use_candidates=use_candidates, max_logits_bytes=args.budget)


@pl.jit
def entry(
    x: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
    query_latent: pl.Tensor[[C.T_DYN, C.Q_LORA], pl.BF16],
    request_ids: pl.Tensor[[C.T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[C.T_DYN], pl.INT32],
    index_cache: pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP], pl.FP8E8M0],
    index_block_table: pl.Tensor[[C.B_DYN, C.TABLE_DYN], pl.INT32],
    rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    index_wq_b: pl.Tensor[[C.Q_LORA, C.INDEX_H * C.INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[C.D, C.INDEX_H], pl.BF16],
    candidate_mask: pl.InOut[pl.Tensor[[C.T_DYN, C.CMP_POSITIONS_DYN], pl.UINT8]],
    topk_indices: pl.InOut[pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x.bind_dynamic(0, C.T_DYN)
    query_latent.bind_dynamic(0, C.T_DYN)
    request_ids.bind_dynamic(0, C.T_DYN)
    compressed_lens.bind_dynamic(0, C.T_DYN)
    rope_cos.bind_dynamic(0, C.T_DYN)
    rope_sin.bind_dynamic(0, C.T_DYN)
    candidate_mask.bind_dynamic(0, C.T_DYN)
    candidate_mask.bind_dynamic(1, C.CMP_POSITIONS_DYN)
    topk_indices.bind_dynamic(0, C.T_DYN)
    index_cache.bind_dynamic(0, C.INDEX_BLOCKS_DYN)
    index_cache_scale.bind_dynamic(0, C.INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, C.B_DYN)
    index_block_table.bind_dynamic(1, C.TABLE_DYN)
    completion = pl.array.create(1, pl.TASK_ID)
    completion[0] = pl.system.task_dummy(deps=[])
    for epoch in pl.range(2):
        completion[0] = indexer(
            x, query_latent, request_ids, compressed_lens, index_cache, index_cache_scale,
            index_block_table, rope_cos, rope_sin, index_wq_b, index_wq_b_scale,
            index_weights_proj, candidate_mask, topk_indices, num_tokens, completion[0],
        )
    return topk_indices


def inputs():
    torch.manual_seed(112)
    capacity, history, requests = 9, 16640, 2
    pages = history // 128
    x = torch.zeros(capacity, C.D, dtype=torch.bfloat16)
    x[:, 0] = 1
    latent = torch.zeros(capacity, C.Q_LORA, dtype=torch.bfloat16)
    latent[:, 0] = torch.tensor([1, 2, 4, 1, 2, 4, 1, 2, 4])
    weight = torch.zeros(C.Q_LORA, C.INDEX_H * C.INDEX_DIM)
    weight[0, 0] = 1
    weights_proj = torch.zeros(C.D, C.INDEX_H, dtype=torch.bfloat16)
    weights_proj[0, 0] = 1
    candidates = torch.ones(capacity, history, dtype=torch.uint8)
    if use_candidates:
        candidates[3:, 1::3] = 0
    return dict(
        x=x, query_latent=latent, request_ids=torch.tensor([1, 0, 1, 0, 1, 1, 0, 0, 1], dtype=torch.int32),
        compressed_lens=torch.tensor([0, 1, 511, 513, 8191, 16639, 16640, 1, 1], dtype=torch.int32),
        index_cache=torch.randint(0, 256, (requests * pages, 128, 1, C.INDEX_DIM // 2), dtype=torch.uint8),
        index_cache_scale=torch.randint(120, 128, (requests * pages, 128, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP),
                                       dtype=torch.uint8).view(torch.float8_e8m0fnu),
        index_block_table=torch.randperm(requests * pages).to(torch.int32).reshape(requests, pages),
        rope_cos=torch.ones(capacity, C.ROPE_DIM // 2), rope_sin=torch.zeros(capacity, C.ROPE_DIM // 2),
        index_wq_b=weight.to(torch.float8_e4m3fn),
        index_wq_b_scale=pack_mx_b_scale(torch.full((C.Q_LORA // 32, C.INDEX_H * C.INDEX_DIM), 127,
                                                  dtype=torch.uint8)).view(torch.float8_e8m0fnu),
        index_weights_proj=weights_proj, candidate_mask=candidates,
        topk_indices=torch.full((capacity, C.INDEX_TOPK), -37, dtype=torch.int32),
    )


values = inputs()
reference_scores = None


def golden(tensors):
    global reference_scores
    n = args.active
    if not n:
        return
    keys = dequantize_mxfp4_cache(tensors["index_cache"], tensors["index_cache_scale"], C.INDEX_CACHE_GROUP, "e8m0")
    scores, physical = reference_indexer(
        tensors["x"][:n], tensors["query_latent"][:n], tensors["request_ids"][:n], keys,
        tensors["index_block_table"], tensors["compressed_lens"][:n], tensors["index_wq_b"],
        tensors["index_wq_b_scale"], tensors["index_weights_proj"], tensors["rope_cos"][:n], tensors["rope_sin"][:n],
        candidates=tensors["candidate_mask"][:n] if use_candidates else None,
    )
    reference_scores = scores
    tensors["topk_indices"][:n].copy_(physical)
    if not use_candidates:
        tensors["candidate_mask"][:n].copy_(select_candidate_blocks(
            scores, tensors["compressed_lens"][:n], C.FLASH.candidate_topk_blocks, C.FLASH.candidate_block_size,
        ).to(torch.uint8))


def compare_topk(actual, expected, **kwargs):
    if not torch.equal(actual[args.active:], expected[args.active:]):
        return False
    for token in range(args.active):
        visible = int(values["compressed_lens"][token])
        request = int(values["request_ids"][token])
        reverse = {int(block) * 128 + offset: page * 128 + offset
                   for page, block in enumerate(values["index_block_table"][request]) for offset in range(128)}
        selected = [reverse.get(int(row), -1) for row in actual[token] if int(row) >= 0]
        scores = reference_scores[token]
        valid_count = int(torch.isfinite(scores).sum())
        count = min(C.INDEX_TOPK, valid_count)
        if len(selected) != count or selected != sorted(set(selected)):
            return False
        if any(row < 0 or row >= visible for row in selected):
            return False
        if count and not bool((scores[selected] >= scores.topk(count).values[-1]).all()):
            return False
        if not bool((actual[token, count:] == -1).all()):
            return False
    return True


def compare_candidates(actual, expected, **kwargs):
    if use_candidates or args.active == 0:
        return torch.equal(actual, expected)
    if not torch.equal(actual[args.active:], expected[args.active:]):
        return False
    for token in range(args.active):
        visible = int(values["compressed_lens"][token])
        count = (visible + 7) // 8
        blocks = actual[token].reshape(-1, 8)
        if not bool(((blocks == 0) | (blocks == 1)).all()) or not bool((blocks == blocks[:, :1]).all()):
            return False
        selected = torch.nonzero(blocks[:, 0]).flatten()
        if len(selected) != min(count, 2048) or bool((selected >= count).any()):
            return False
        if count:
            scores = reference_scores[token].reshape(-1, 8).amax(-1).clone()
            scores[count - 1] = torch.inf
            if not bool((scores[selected] >= scores.topk(min(count, 2048)).values[-1]).all()):
                return False
    return True


if args.cpu_only:
    expected = {k: v.clone() for k, v in values.items()}
    golden(expected)
    assert compare_topk(expected["topk_indices"], expected["topk_indices"])
    assert compare_candidates(expected["candidate_mask"], expected["candidate_mask"])
    print("CPU reference and tie-aware exact selection checks passed")
else:
    specs = [TensorSpec(k, list(v.shape), v.dtype, init_value=v) for k, v in values.items()]
    specs.append(ScalarSpec("num_tokens", torch.int32, args.active))
    result = run(
        fn=entry, specs=specs, golden_fn=golden, compile_only=args.compile_only,
        config={"platform": "a5", "device_id": args.device, "enable_dep_gen": args.dep_gen},
        compare_fn={"topk_indices": compare_topk, "candidate_mask": compare_candidates},
    )
    print(f"RESULT mode={args.mode} active={args.active} budget={args.budget}: {result}")
    if not result.passed:
        raise SystemExit(1)
