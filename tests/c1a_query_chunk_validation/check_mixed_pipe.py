# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Isolate local A5 vector/cube transfer directions without any model imports."""

import argparse

import pypto.language as pl
import torch
from pypto.runtime import RunConfig


def make_case(mode):
    prepare = mode != "c2v"
    epilogue = mode != "v2c"
    left = mode == "left"
    physical_transpose = mode == "transpose"

    @pl.jit
    def entry(
        keys: pl.Tensor[[64, 128], pl.FP32],
        keys_bf16: pl.Tensor[[64, 128], pl.BF16],
        query: pl.Tensor[[32, 128], pl.BF16],
        output: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
        output_left: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
    ):
        with pl.spmd(1, name_hint="mixed_pipe_probe"):
            block = pl.tile.get_block_idx()
            if prepare:
                key_values = pl.load(keys, [0, 0], [64, 128])
                key_tile = pl.cast(key_values, pl.BF16, mode="rint")
            else:
                key_tile = pl.load(keys_bf16, [0, 0], [64, 128])
            query_tile = pl.load(query, [0, 0], [32, 128])
            if left:
                scores_left = pl.matmul(key_tile, pl.tile.transpose_view(query_tile))
                pl.store(pl.maximum(scores_left, 0.0), [block, 0], output_left)
            else:
                if physical_transpose:
                    key_materialized = pl.tile.transpose(key_tile, 0, 1)
                    scores = pl.matmul(query_tile, key_materialized)
                else:
                    key_transposed = pl.tile.transpose_view(key_tile)
                    scores = pl.matmul(query_tile, key_transposed)
                if epilogue:
                    pl.store(pl.maximum(scores, 0.0), [block, 0], output)
                else:
                    pl.store(scores, [block, 0], output)
        return output, output_left

    return entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("c2v", "v2c", "both", "left", "transpose"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(712)
    # Integer inputs keep BF16 products and FP32 sums exact.
    keys = torch.randint(-3, 4, (64, 128)).float()
    query = torch.randint(-3, 4, (32, 128)).to(torch.bfloat16)
    output = torch.empty(32, 64)
    output_left = torch.empty(64, 32)
    expected = query.float() @ keys.T
    if args.mode != "v2c":
        expected = expected.relu()
    kernel = make_case(args.mode)
    if args.compile_only:
        kernel.compile(keys, keys.to(torch.bfloat16), query, output, output_left,
                       config=RunConfig(platform="a5"))
        print(f"COMPILE PASS {args.mode}", flush=True)
        raise SystemExit(0)
    kernel(keys, keys.to(torch.bfloat16), query, output, output_left,
           config=RunConfig(platform="a5", device_id=args.device))
    actual = output_left.T if args.mode == "left" else output
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(f"PASS {args.mode}: exact 32x64x128 score comparison", flush=True)
