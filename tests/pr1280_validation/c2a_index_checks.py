# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validation-branch-only checks for the shared production C2A index selector."""

import argparse
import json
import os
from pathlib import Path

import torch
import pypto.language as pl

from golden import ScalarSpec, TensorSpec, run
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.decode_c2a_full import IDX_PACKED, IDX_SCALES, LEAF, index_select

QUERY_WIDTH = C.INDEX_H * C.INDEX_DIM


@pl.jit
def index_entry(
    query_input: pl.Tensor[[C.T_DYN, C.INDEX_H * C.INDEX_DIM], pl.BF16],
    weights: pl.Tensor[[C.T_DYN, C.INDEX_H], pl.BF16],
    cache: pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8],
    scales: pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, IDX_SCALES], pl.FP8E8M0],
    block_table: pl.Tensor[[C.B_DYN, C.TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[C.T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[C.T_DYN], pl.INT32],
    leaf_scores: pl.InOut[pl.Tensor[[C.T_DYN, LEAF], pl.FP32]],
    leaf_rows: pl.InOut[pl.Tensor[[C.T_DYN, LEAF], pl.INT32]],
    topk_indices: pl.InOut[pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(query_input, 0)
    query = pl.create_tensor([tokens, QUERY_WIDTH], dtype=pl.BF16)
    # A real producer supplies the existing publication dependency argument.
    with pl.spmd(num_tokens, name_hint="validation_query_publish") as publish_ready:
        token = pl.tile.get_block_idx()
        query[token : token + 1, :] = query_input[token : token + 1, :]
    index_select(
        query, weights, cache, scales, block_table, request_ids, compressed_lens,
        leaf_scores, leaf_rows, topk_indices, num_tokens, publish_ready,
    )
    return leaf_scores, leaf_rows, topk_indices


LENGTHS = (0, 1, 128, 129, 512, 513, 2048, 2049)


def address_checks():
    """Prove old invalid accesses on CPU; never execute the old device kernel."""
    for length in LENGTHS:
        width = max(1, (length + 127) // 128)
        bound = 512 if length <= 512 else ((length + 2047) // 2048) * 2048
        old_bad = [p for p in range(0, bound, 64) if p // 128 >= width]
        assert bool(old_bad) == (length not in (512, 2048))
        for p in range(0, bound, 64):
            if p < length:
                assert p // 128 < width
                assert p % 128 + 64 <= 128
        print(f"ADDRESS length={length} width={width} old_bad_first={old_bad[:1]}", flush=True)


def make_values(lengths, mixed=False):
    tokens = len(lengths)
    request_ids = torch.tensor(([2, 0, 1, 0, 2, 1, 2, 0] if mixed else [0]), dtype=torch.int32)
    requests = int(request_ids.max()) + 1
    width = max(1, (max(lengths) + 127) // 128)
    blocks = requests * width + 3
    generator = torch.Generator().manual_seed(1278 + max(lengths))
    pages = torch.randperm(blocks, generator=generator)
    table = torch.full((requests, width), -1, dtype=torch.int32)
    for request in range(requests):
        visible = max(n for i, n in enumerate(lengths) if int(request_ids[i]) == request)
        count = (visible + 127) // 128
        table[request, :count] = pages[request * width : request * width + count].int()
    # Unique row scores use binary digits encoded as exactly representable E2M1 0/1.
    # Other channels exercise signed nibbles and all four nonuniform scale groups.
    rows = blocks * 128
    codes = torch.randint(0, 16, (rows, C.INDEX_DIM), generator=generator, dtype=torch.uint8)
    codes[:, :] = torch.tensor([2, 10, 3, 11] * (C.INDEX_DIM // 4), dtype=torch.uint8)
    rank = torch.randperm(rows, generator=generator)
    for bit in range(14):
        codes[:, 2 * bit] = (((rank >> bit) & 1) * 2).to(torch.uint8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    scale_codes = torch.tensor([127, 126, 128, 127], dtype=torch.uint8).repeat(rows, 1)
    query = torch.ones((tokens, C.INDEX_H, C.INDEX_DIM), dtype=torch.bfloat16)
    for bit in range(14):
        query[:, :, bit] = 2**bit
    query[:, 1::2, :] *= -1
    weights = torch.full((tokens, C.INDEX_H), 1 / 32, dtype=torch.bfloat16)
    weights[:, 1::4] = -1 / 64
    return dict(
        query_input=query.flatten(1), weights=weights,
        cache=packed.reshape(blocks, 128, 1, IDX_PACKED),
        scales=scale_codes.reshape(blocks, 128, 1, IDX_SCALES).view(torch.float8_e8m0fnu),
        block_table=table, request_ids=request_ids,
        compressed_lens=torch.tensor(lengths, dtype=torch.int32),
        leaf_scores=torch.full((tokens, LEAF), 17.0),
        leaf_rows=torch.full((tokens, LEAF), -777, dtype=torch.int32),
        topk_indices=torch.full((tokens, C.INDEX_TOPK), -777, dtype=torch.int32),
        num_tokens=tokens,
    )


def reference(v):
    """Decode natural adjacent E2M1 pairs and rank only visible physical rows."""
    codebook = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
    packed = v["cache"].reshape(-1, IDX_PACKED).long()
    natural_codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1)
    factors = torch.exp2(v["scales"].view(torch.uint8).reshape(-1, IDX_SCALES).float() - 127)
    keys = (codebook[natural_codes] * factors.repeat_interleave(32, dim=1)).bfloat16().float()
    query = v["query_input"].reshape(-1, C.INDEX_H, C.INDEX_DIM).float()
    natural_query = torch.stack((query[..., :IDX_PACKED], query[..., IDX_PACKED:]), dim=-1).flatten(2)
    for token, length_tensor in enumerate(v["compressed_lens"]):
        length = int(length_tensor)
        request = int(v["request_ids"][token])
        positions = torch.arange(length)
        physical = v["block_table"][request, positions // 128].long() * 128 + positions % 128
        v["topk_indices"][token].fill_(-1)
        if length <= C.INDEX_TOPK:
            v["topk_indices"][token, :length] = physical.int()
            continue
        scores = ((natural_query[token] @ keys[physical].T).relu() * v["weights"][token].float()[:, None]).sum(0)
        ranked = torch.argsort(scores, descending=True)
        assert scores[ranked[511]] > scores[ranked[512]], "fixture must avoid a Top-K boundary tie"
        chosen = physical[ranked[:C.INDEX_TOPK]].sort().values
        v["topk_indices"][token] = chosen.int()
        leaf_start = (length - 1) // LEAF * LEAF
        leaf_valid = length - leaf_start
        tile_end = (leaf_valid + 63) // 64 * 64
        v["leaf_scores"][token].fill_(-3.0e38)
        v["leaf_rows"][token].fill_(-1)
        v["leaf_scores"][token, :leaf_valid] = scores[leaf_start:]
        v["leaf_scores"][token, leaf_valid:tile_end] = -1.0e30
        padded_positions = torch.arange(leaf_start, leaf_start + tile_end)
        padded_physical = v["block_table"][request, padded_positions // 128].long() * 128 + padded_positions % 128
        v["leaf_rows"][token, :tile_end] = padded_physical.int()


def exact(actual, expected, **kwargs):
    return torch.equal(actual, expected), "exact physical row / invalid sentinel comparison"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--host-only", action="store_true")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "0")))
    args = parser.parse_args()
    torch.set_num_threads(1)
    address_checks()
    cases = [(str(n), (n,), False) for n in LENGTHS] + [("mixed", LENGTHS, True)]
    results = []
    for name, lengths, mixed in cases:
        values = make_values(lengths, mixed)
        if args.host_only:
            reference(values)
            print(f"HOST {name} passed", flush=True)
            continue
        specs = [
            TensorSpec(k, list(v.shape), v.dtype, init_value=v) if isinstance(v, torch.Tensor)
            else ScalarSpec(k, torch.int32, v, compile_runtime=True)
            for k, v in values.items()
        ]
        result = run(
            fn=index_entry, specs=specs, golden_fn=reference, compile_only=args.compile_only,
            runtime_dir=args.runtime_dir,
            config=dict(platform="a2a3", device_id=args.device, dump_passes=True),
            compare_fn={"topk_indices": exact, "leaf_rows": exact}, rtol=0, atol=0,
        )
        record = dict(case=name, passed=result.passed, work_dir=str(result.work_dir), error=str(result.error))
        results.append(record)
        print("RESULT " + json.dumps(record), flush=True)
        assert result.passed, record
        if args.compile_only:
            # All extents are runtime dimensions; one artifact covers the full case matrix.
            break
    if not args.host_only:
        Path("build_output/c2a_index_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
