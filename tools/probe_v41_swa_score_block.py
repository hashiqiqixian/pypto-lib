# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Inspect one frozen SWA score block at a BF16 probability boundary."""

import argparse

import pypto.language as pl
import torch

from golden import TensorSpec, run
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.decode_attn_swa import M_TILE, SOFTMAX_SCALE


@pl.jit
def score_block(
    q: pl.Tensor[[M_TILE, C.HEAD_DIM], pl.BF16],
    keys: pl.Tensor[[64, C.HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[1, 64], pl.INT32],
    scores: pl.Out[pl.Tensor[[M_TILE, 64], pl.FP32]],
    maximums: pl.Out[pl.Tensor[[M_TILE, 1], pl.FP32]],
    centered: pl.Out[pl.Tensor[[M_TILE, 64], pl.FP32]],
    probabilities: pl.Out[pl.Tensor[[M_TILE, 64], pl.FP32]],
    weights: pl.Out[pl.Tensor[[M_TILE, 64], pl.BF16]],
    fused_probabilities: pl.Out[pl.Tensor[[M_TILE, 64], pl.FP32]],
    fused_weights: pl.Out[pl.Tensor[[M_TILE, 64], pl.BF16]],
):
    for _ in pl.spmd(1):
        raw = pl.mul(pl.matmul(q, keys, b_trans=True), SOFTMAX_SCALE)
        valid = pl.minimum(pl.maximum(pl.add(pl.cast(indices, pl.FP32), 1.0), 0.0), 1.0)
        masked = pl.col_expand_add(raw, pl.mul(pl.sub(valid, 1.0), 1e30))
        maximum = pl.reshape(pl.row_max(masked), [M_TILE, 1])
        shifted = pl.row_expand_sub(masked, maximum)
        probability = pl.col_expand_mul(pl.exp(shifted), valid)
        fused_probability = pl.col_expand_mul(pl.row_expand_expdif(masked, maximum), valid)
        scores[:, :] = raw
        maximums[:, :] = maximum
        centered[:, :] = shifted
        probabilities[:, :] = probability
        weights[:, :] = pl.cast(probability, pl.BF16, mode="rint")
        fused_probabilities[:, :] = fused_probability
        fused_weights[:, :] = pl.cast(fused_probability, pl.BF16, mode="rint")
    return scores, maximums, centered, probabilities, weights, fused_probabilities, fused_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("--token", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--index", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    data = torch.load(args.capture, map_location="cpu", weights_only=True)
    head_start = args.head // M_TILE * M_TILE
    inputs = {
        "q": data["q"][args.token].reshape(C.LOCAL_H, C.HEAD_DIM)[
            head_start:head_start + M_TILE].contiguous(),
        "keys": data["selected"][args.token, :64].contiguous(),
        "indices": data["indices"][args.token, :64].unsqueeze(0).contiguous(),
    }
    specs = [TensorSpec(name, list(value.shape), value.dtype,
                        init_value=lambda value=value: value.clone()) for name, value in inputs.items()]
    specs += [TensorSpec("scores", [M_TILE, 64], torch.float32),
              TensorSpec("maximums", [M_TILE, 1], torch.float32),
              TensorSpec("centered", [M_TILE, 64], torch.float32),
              TensorSpec("probabilities", [M_TILE, 64], torch.float32),
              TensorSpec("weights", [M_TILE, 64], torch.bfloat16),
              TensorSpec("fused_probabilities", [M_TILE, 64], torch.float32),
              TensorSpec("fused_weights", [M_TILE, 64], torch.bfloat16)]

    def golden(values):
        exact = values["q"].double() @ values["keys"].double().T * SOFTMAX_SCALE
        values["scores"].copy_(exact.float())
        masked = exact.masked_fill(values["indices"][0] < 0, -torch.inf)
        maximum = masked.amax(-1, keepdim=True)
        shifted = masked - maximum
        values["maximums"].copy_(maximum.float())
        values["centered"].copy_(shifted.float())
        probability = shifted.exp()
        values["probabilities"].copy_(probability.float())
        values["weights"].copy_(probability.bfloat16())
        values["fused_probabilities"].copy_(probability.float())
        values["fused_weights"].copy_(probability.bfloat16())

    def compare(name):
        def check(actual, expected, **_):
            index = (args.head - head_start, 0 if name == "maximums" else args.index)
            print(f"[SWA BLOCK] {name} token={args.token} head={args.head} index={args.index} "
                  f"device={float(actual[index]):.12g} reference={float(expected[index]):.12g}")
            return bool(torch.isfinite(actual.float()).all()), "finite diagnostic output"
        return check

    result = run(score_block, specs, golden_fn=golden,
                 config=dict(platform="a5", device_id=args.device),
                 compare_fn={name: compare(name) for name in
                             ("scores", "maximums", "centered", "probabilities", "weights",
                              "fused_probabilities", "fused_weights")})
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
