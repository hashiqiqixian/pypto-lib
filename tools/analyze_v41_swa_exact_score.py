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


def replay(query, keys, indices, sink, score_dtype, online_dtype, numerator_dtype,
           accurate_probability=False, round_probability=False, round_denominator=False,
           round_correction=False, fp32_factor=False):
    maximum = torch.tensor(-1e30, dtype=online_dtype)
    denominator = torch.tensor(0.0, dtype=online_dtype)
    numerator = torch.zeros_like(query, dtype=numerator_dtype)
    for start in (0, 64):
        selected = keys[start:start + 64]
        score = (selected.double() @ query.double()).to(score_dtype)
        score = score.to(online_dtype) * query.numel() ** -0.5
        score[indices[start:start + 64] < 0] = -torch.inf
        next_max = torch.maximum(maximum, score.max())
        correction = (maximum - next_max).exp()
        if round_correction:
            correction = correction.float().to(online_dtype)
        if accurate_probability:
            probability = (score.double() - next_max.double()).exp().to(online_dtype)
        else:
            probability = (score - next_max).exp()
        probability_for_sum = probability.float().to(online_dtype) if round_probability else probability
        denominator = denominator * correction + probability_for_sum.sum()
        if round_denominator:
            denominator = denominator.float().to(online_dtype)
        numerator = (numerator * correction.to(numerator_dtype) +
                     probability.bfloat16().to(numerator_dtype) @ selected.to(numerator_dtype))
        maximum = next_max
    final_max = torch.maximum(maximum, sink.to(online_dtype))
    correction = (maximum - final_max).exp()
    if round_correction:
        correction = correction.float().to(online_dtype)
    denominator = denominator * correction + (sink.to(online_dtype) - final_max).exp()
    if round_denominator:
        denominator = denominator.float().to(online_dtype)
    factor = correction.float() / denominator.float() if fp32_factor else correction / denominator
    return (numerator * factor.to(numerator_dtype)).bfloat16()


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
    for (score_dtype, online_dtype, numerator_dtype, accurate_probability,
         round_probability, round_denominator, round_correction, fp32_factor) in (
        (torch.float32, torch.float32, torch.float32, False, False, False, False, False),
        (torch.float32, torch.float32, torch.float32, True, False, False, False, False),
        (torch.float32, torch.float64, torch.float32, False, False, False, False, False),
        (torch.float32, torch.float64, torch.float32, False, True, False, False, False),
        (torch.float32, torch.float64, torch.float32, False, False, True, False, False),
        (torch.float32, torch.float64, torch.float32, False, False, False, True, False),
        (torch.float32, torch.float64, torch.float32, False, False, False, False, True),
        (torch.float32, torch.float32, torch.float64, False, False, False, False, False),
        (torch.float32, torch.float64, torch.float64, False, False, False, False, False),
        (torch.float64, torch.float32, torch.float32, False, False, False, False, False),
        (torch.float64, torch.float64, torch.float64, False, False, False, False, False),
    ):
        fixed = introduced = 0
        for token, head in pairs:
            candidate = replay(query[token, head], data["selected"][token],
                               data["indices"][token], data["sink"][head],
                               score_dtype, online_dtype, numerator_dtype, accurate_probability,
                               round_probability, round_denominator, round_correction, fp32_factor)
            mismatch = changed[token, head]
            fixed += int((candidate[mismatch] == expected[token, head][mismatch]).sum())
            introduced += int((candidate[~mismatch] != expected[token, head][~mismatch]).sum())
        print(f"score={score_dtype} online={online_dtype} numerator={numerator_dtype} "
              f"accurate_probability={accurate_probability} "
              f"round_probability={round_probability} round_denominator={round_denominator} "
              f"round_correction={round_correction} fp32_factor={fp32_factor} "
              f"fixed={fixed}/{int(changed.sum())} "
              f"introduced={introduced}", flush=True)

if __name__ == "__main__":
    main()
