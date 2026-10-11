# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Identify a BF16 attention-weight boundary in a captured SWA head."""

import argparse

import torch

from analyze_v41_swa_compensation import scores as projected_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("--row", type=int, default=74)
    parser.add_argument("--head", type=int, default=2)
    args = parser.parse_args()
    if args.capture.endswith(".npz"):
        import numpy as np

        with np.load(args.capture) as capture:
            data = {
                name: torch.from_numpy(capture[name].copy()).to(torch.bfloat16)
                if name in {"actual", "expected", "q", "selected"}
                else torch.from_numpy(capture[name].copy())
                for name in capture.files
            }
    else:
        data = torch.load(args.capture, map_location="cpu", weights_only=True)
    row, head = args.row, args.head
    q = data["q"][row].reshape(16, 512)[head].double()
    keys = data["selected"][row].double()
    indices = data["indices"][row]
    actual = data["actual"][row].reshape(16, 512)[head]
    expected = data["expected"][row].reshape(16, 512)[head]
    maximum = torch.tensor(-1e30, dtype=torch.float64)
    denominator = torch.tensor(0.0, dtype=torch.float64)
    numerator = torch.zeros(512, dtype=torch.float64)
    weights = []
    factors = []
    for start in (0, 64):
        scores = (keys[start:start + 64] @ q) * (512 ** -0.5)
        scores[indices[start:start + 64] < 0] = -torch.inf
        next_max = torch.maximum(maximum, scores.max())
        correction = (maximum - next_max).exp()
        probability = (scores - next_max).exp().bfloat16()
        denominator = denominator * correction + (scores - next_max).exp().sum()
        numerator = numerator * correction + probability.double() @ keys[start:start + 64]
        if start:
            factors[:64] = [factor * correction for factor in factors[:64]]
        weights.append(probability)
        factors.extend([torch.tensor(1.0, dtype=torch.float64)] * 64)
        maximum = next_max
    sink = data["sink"][head].double()
    final_max = torch.maximum(maximum, sink)
    correction = (maximum - final_max).exp()
    denominator = denominator * correction + (sink - final_max).exp()
    numerator *= correction
    factors = torch.stack(factors) * (correction / denominator)
    base = numerator / denominator
    print(f"row={row} head={head} original_differences={(actual != expected).sum().item()} "
          f"replay_differences={(base.bfloat16() != expected).sum().item()}")
    candidates = []
    weights = torch.cat(weights)
    for index in torch.nonzero(indices >= 0).flatten().tolist():
        weight = weights[index]
        for direction in (-1, 1):
            bits = weight.view(torch.int16).to(torch.int32) + direction
            altered = bits.to(torch.int16).view(torch.bfloat16).double()
            trial = (base + (altered - weight.double()) * factors[index] * keys[index]).bfloat16()
            mismatch = (trial != actual).sum().item()
            fixed = ((actual != expected) & (trial == actual)).sum().item()
            introduced = ((actual == expected) & (trial != actual)).sum().item()
            candidates.append((mismatch, -fixed, introduced, index, direction,
                               float(weight), float(altered), float(factors[index])))
    best_index = min(candidates)[3]
    if best_index < 64:
        keys_block = keys[:64]
        block_indices = indices[:64]
    else:
        keys_block = keys[64:]
        block_indices = indices[64:]
    for dtype in (torch.float32, torch.float64):
        scores = (keys_block.to(dtype) @ q.to(dtype)) * (512 ** -0.5)
        scores[block_indices < 0] = -torch.inf
        probability = (scores - scores.max()).exp()
        local_index = best_index % 64
        value = float(probability[local_index])
        low = float((weights[best_index].view(torch.int16).to(torch.int32) - 1)
                    .to(torch.int16).view(torch.bfloat16))
        high = float(weights[best_index])
        print(f"best_index={best_index} dtype={dtype} score={float(scores[local_index]):.12g} "
              f"maximum={float(scores.max()):.12g} probability={value:.12g} "
              f"bf16_midpoint={(low + high) / 2:.12g}")
    for chunk in (0, 256, 128, 64, 32, -1, -3):
        scores = projected_scores(q, keys_block, chunk) * (512 ** -0.5)
        scores[block_indices < 0] = -torch.inf
        difference = scores - scores.max()
        for label, probability in (
            ("exp", difference.exp()[local_index]),
            ("half_squared", difference.mul(0.5).exp().square()[local_index]),
            ("exp2", torch.exp2(difference * 1.4426950408889634)[local_index]),
        ):
            print(f"chunk={chunk:3} method={label} score={float(scores[local_index]):.12g} "
                  f"probability={float(probability):.12g} "
                  f"bf16={float(probability.bfloat16()):.12g}")
    for item in sorted(candidates)[:12]:
        print("mismatch=%d fixed=%d introduced=%d index=%d direction=%+d "
              "weight=%.9g altered=%.9g factor=%.9g" %
              (item[0], -item[1], *item[2:]))


if __name__ == "__main__":
    main()
