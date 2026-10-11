# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Test whether exact QK scores alone resolve frozen SWA BF16 disagreements."""

import argparse

import torch


def replay(query, keys, indices, sink, score_dtype, online_dtype):
    maximum = torch.tensor(-1e30, dtype=online_dtype)
    denominator = torch.tensor(0.0, dtype=online_dtype)
    numerator = torch.zeros_like(query, dtype=online_dtype)
    for start in (0, 64):
        selected = keys[start:start + 64]
        score = (selected.double() @ query.double()).to(score_dtype)
        score = score.to(online_dtype) * query.numel() ** -0.5
        score[indices[start:start + 64] < 0] = -torch.inf
        next_max = torch.maximum(maximum, score.max())
        correction = (maximum - next_max).exp()
        probability = (score - next_max).exp()
        denominator = denominator * correction + probability.sum()
        numerator = numerator * correction + probability.bfloat16().to(online_dtype) @ selected.to(online_dtype)
        maximum = next_max
    final_max = torch.maximum(maximum, sink.to(online_dtype))
    correction = (maximum - final_max).exp()
    denominator = denominator * correction + (sink.to(online_dtype) - final_max).exp()
    return (numerator * (correction / denominator)).bfloat16()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    args = parser.parse_args()
    data = torch.load(args.capture, map_location="cpu", weights_only=True)
    actual = data["actual"].unflatten(-1, (16, 512))
    expected = data["expected"].unflatten(-1, (16, 512))
    query = data["q"].unflatten(-1, (16, 512))
    changed = actual != expected
    pairs = list(dict.fromkeys((int(t), int(h)) for t, h, _ in changed.nonzero(as_tuple=False)))
    print(f"changed_elements={int(changed.sum())} changed_pairs={len(pairs)}", flush=True)
    for score_dtype, online_dtype in (
        (torch.float32, torch.float32),
        (torch.float32, torch.float64),
        (torch.float64, torch.float32),
        (torch.float64, torch.float64),
    ):
        fixed = introduced = 0
        for token, head in pairs:
            candidate = replay(query[token, head], data["selected"][token],
                               data["indices"][token], data["sink"][head],
                               score_dtype, online_dtype)
            mismatch = changed[token, head]
            fixed += int((candidate[mismatch] == expected[token, head][mismatch]).sum())
            introduced += int((candidate[~mismatch] != expected[token, head][~mismatch]).sum())
        print(f"score={score_dtype} online={online_dtype} fixed={fixed}/{int(changed.sum())} "
              f"introduced={introduced}", flush=True)


if __name__ == "__main__":
    main()
