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


def compensated_dot(query, keys):
    high = torch.zeros(keys.shape[0], dtype=torch.float32)
    low = torch.zeros_like(high)
    for column in range(query.numel()):
        product = keys[:, column].float() * query[column].float()
        total = high + product
        carried = total - high
        low += (high - (total - carried)) + (product - carried)
        high = total
    return high, low


def two_sum(left, right):
    total = left + right
    carried = total - left
    return total, (left - (total - carried)) + (right - carried)


def split_product_error(left, right, product):
    split = 4097.0
    left_split = split * left
    left_high = left_split - (left_split - left)
    left_low = left - left_high
    right_split = split * right
    right_high = right_split - (right_split - right)
    right_low = right - right_high
    return (((left_high * right_high - product) + left_high * right_low) +
            left_low * right_high) + left_low * right_low


def exp_pair(high, low):
    base = high.exp()
    return base + base * low


def replay_fp32_pair(query, keys, indices, sink):
    scale_hi = torch.tensor(query.numel() ** -0.5, dtype=torch.float32)
    scale_lo = torch.tensor(query.numel() ** -0.5 - float(scale_hi), dtype=torch.float32)
    maximum_hi = torch.tensor(-1e30, dtype=torch.float32)
    maximum_lo = torch.tensor(0.0, dtype=torch.float32)
    denominator = torch.tensor(0.0, dtype=torch.float32)
    numerator = torch.zeros_like(query, dtype=torch.float32)
    for start in (0, 64):
        selected = keys[start:start + 64]
        dot_hi, dot_lo = compensated_dot(query, selected)
        product = dot_hi * scale_hi
        product_lo = (split_product_error(dot_hi, scale_hi, product) +
                      dot_hi * scale_lo + dot_lo * scale_hi + dot_lo * scale_lo)
        score_hi, score_lo = two_sum(product, product_lo)
        valid = indices[start:start + 64] >= 0
        score_hi = torch.where(valid, score_hi, -torch.inf)
        score_lo = torch.where(valid, score_lo, 0.0)
        candidate_hi = score_hi.max()
        candidate_lo = score_lo[score_hi == candidate_hi].max()
        if (candidate_hi > maximum_hi or
                (candidate_hi == maximum_hi and candidate_lo > maximum_lo)):
            next_hi, next_lo = candidate_hi, candidate_lo
        else:
            next_hi, next_lo = maximum_hi, maximum_lo
        correction_hi, correction_err = two_sum(maximum_hi, -next_hi)
        correction = exp_pair(correction_hi, correction_err + maximum_lo - next_lo)
        centered_hi, centered_err = two_sum(score_hi, -next_hi)
        centered_lo = torch.where(valid, centered_err + score_lo - next_lo, 0.0)
        probability = torch.where(valid, exp_pair(centered_hi, centered_lo), 0.0)
        denominator = denominator * correction + probability.sum()
        numerator = numerator * correction + probability.bfloat16().float() @ selected.float()
        maximum_hi, maximum_lo = next_hi, next_lo
    if sink.float() > maximum_hi or (sink.float() == maximum_hi and -maximum_lo > 0):
        final_hi, final_lo = sink.float(), torch.tensor(0.0, dtype=torch.float32)
    else:
        final_hi, final_lo = maximum_hi, maximum_lo
    correction_hi, correction_err = two_sum(maximum_hi, -final_hi)
    correction = exp_pair(correction_hi, correction_err + maximum_lo - final_lo)
    sink_hi, sink_err = two_sum(sink.float(), -final_hi)
    denominator = denominator * correction + exp_pair(sink_hi, sink_err - final_lo)
    return (numerator * (correction / denominator)).bfloat16()


def replay_compensated_logits(query, keys, indices, sink, use_compensated_dot=False):
    maximum = torch.tensor(-1e30, dtype=torch.float64)
    denominator = torch.tensor(0.0, dtype=torch.float32)
    numerator = torch.zeros_like(query, dtype=torch.float32)
    for start in (0, 64):
        selected = keys[start:start + 64]
        dot = (sum(part.double() for part in compensated_dot(query, selected))
               if use_compensated_dot else selected.double() @ query.double())
        exact = dot * query.numel() ** -0.5
        exact[indices[start:start + 64] < 0] = -torch.inf
        next_max = torch.maximum(maximum, exact.max())
        correction = (maximum - next_max).exp().float()
        centered = exact - next_max
        centered_hi = centered.float()
        centered_lo = torch.where(torch.isfinite(centered),
                                  (centered - centered_hi.double()).float(), 0.0)
        probability_hi = centered_hi.exp()
        probabilities = probability_hi + probability_hi * centered_lo
        denominator = denominator * correction + probabilities.sum()
        numerator = numerator * correction + probabilities.bfloat16().float() @ selected.float()
        maximum = next_max
    final_max = torch.maximum(maximum, sink.double())
    correction = (maximum - final_max).exp().float()
    denominator = denominator * correction + (sink.double() - final_max).exp().float()
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

    for use_compensated_dot in (False, True):
        fixed = introduced = 0
        for token, head in pairs:
            candidate = replay_compensated_logits(
                query[token, head], data["selected"][token], data["indices"][token],
                data["sink"][head], use_compensated_dot=use_compensated_dot)
            mismatch = changed[token, head]
            fixed += int((candidate[mismatch] == expected[token, head][mismatch]).sum())
            introduced += int((candidate[~mismatch] != expected[token, head][~mismatch]).sum())
        print(f"compensated_logits dot_fp32_pair={use_compensated_dot} "
              f"fixed={fixed}/{int(changed.sum())} introduced={introduced}", flush=True)

    fixed = introduced = 0
    for token, head in pairs:
        candidate = replay_fp32_pair(query[token, head], data["selected"][token],
                                     data["indices"][token], data["sink"][head])
        mismatch = changed[token, head]
        fixed += int((candidate[mismatch] == expected[token, head][mismatch]).sum())
        introduced += int((candidate[~mismatch] != expected[token, head][~mismatch]).sum())
    print(f"fp32_pair_pipeline fixed={fixed}/{int(changed.sum())} introduced={introduced}", flush=True)

if __name__ == "__main__":
    main()
