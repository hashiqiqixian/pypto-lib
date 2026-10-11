# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read the first-block SWA probability before and after BF16 narrowing."""

import argparse

import pypto.language as pl
import torch

from golden import TensorSpec, run
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.config import HEAD_DIM


@pl.jit
def probe_probability(
    q: pl.Tensor[[1, C.LOCAL_H * C.HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[1, 128, C.HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[1, 128], pl.INT32],
    scores_out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
    probabilities_out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
    weights_out: pl.Out[pl.Tensor[[16, 64], pl.BF16]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        q_first = pl.reshape(q[:, :16 * 256], [16, 256])
        q_second = pl.reshape(q[:, 16 * 256:], [16, 256])
        kv = pl.reshape(selected[:, :64, :], [64, HEAD_DIM])
        kv_first = kv[:, :256]
        kv_second = kv[:, 256:]
        scores = pl.add(
            pl.matmul(q_first, kv_first, b_trans=True),
            pl.matmul(q_second, kv_second, b_trans=True),
        )
        scores = pl.mul(scores, HEAD_DIM ** -0.5)
        idx = pl.cast(indices[:, :64], pl.FP32)
        valid = pl.minimum(pl.maximum(pl.add(idx, 1.0), 0.0), 1.0)
        scores = pl.col_expand_add(scores, pl.mul(pl.sub(valid, 1.0), 1e30))
        maximum = pl.row_max(scores)
        probabilities = pl.col_expand_mul(
            pl.exp(pl.row_expand_sub(scores, pl.reshape(maximum, [16, 1]))), valid
        )
        scores_out[:, :] = scores
        probabilities_out[:, :] = probabilities
        weights_out[:, :] = pl.cast(probabilities, pl.BF16, mode="rint")
    return scores_out, probabilities_out, weights_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--index", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    data = torch.load(args.capture, map_location="cpu", weights_only=True)
    values = {
        "q": data["q"][args.row:args.row + 1].contiguous(),
        "selected": data["selected"][args.row:args.row + 1].contiguous(),
        "indices": data["indices"][args.row:args.row + 1].contiguous(),
    }
    outputs = ("scores_out", "probabilities_out", "weights_out")
    specs = [TensorSpec(name, list(value.shape), value.dtype,
                        init_value=lambda value=value: value.clone()) for name, value in values.items()]
    specs.extend(TensorSpec(name, [16, 64], torch.bfloat16 if name == "weights_out" else torch.float32)
                 for name in outputs)

    def golden(tensors):
        q = tensors["q"].reshape(16, 512).float()
        kv = tensors["selected"][:, :64].reshape(64, 512).float()
        scores = (q[:, :256] @ kv[:, :256].T + q[:, 256:] @ kv[:, 256:].T) * (512 ** -0.5)
        scores[:, tensors["indices"].reshape(-1) < 0] = -1e30
        probability = (scores - scores.amax(-1, keepdim=True)).exp()
        tensors["scores_out"].copy_(scores)
        tensors["probabilities_out"].copy_(probability)
        tensors["weights_out"].copy_(probability.bfloat16())

    def compare(name):
        def check(actual, expected, **_):
            index = (args.head, args.index)
            print(f"[PROBABILITY] {name} row={args.row} head={args.head} slot={args.index} "
                  f"device={float(actual[index]):.12g} reference={float(expected[index]):.12g}")
            return True, "diagnostic"
        return check

    result = run(probe_probability, specs, golden_fn=golden,
                 config=dict(platform="a5", device_id=args.device),
                 compare_fn={name: compare(name) for name in outputs})
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
