# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compare FP32 SWA score and softmax strategies on a frozen device capture."""

import argparse

import torch


def scores(q, keys, chunk):
    if chunk == 0:
        return keys.float() @ q.float()
    partials = [keys[:, start:start + chunk].float() @ q[start:start + chunk].float()
                for start in range(0, q.numel(), chunk)]
    # A balanced tree keeps the error smaller than sequentially summing tiles.
    while len(partials) > 1:
        partials = [partials[i] + partials[i + 1] for i in range(0, len(partials), 2)]
    return partials[0]


def replay(q, keys, indices, sink, chunk, accurate_online):
    dtype = torch.float64 if accurate_online else torch.float32
    maximum = torch.tensor(-1e30, dtype=dtype)
    denominator = torch.tensor(0.0, dtype=dtype)
    numerator = torch.zeros_like(q, dtype=dtype)
    for start in (0, 64):
        score = scores(q, keys[start:start + 64], chunk).to(dtype) * (q.numel() ** -0.5)
        score[indices[start:start + 64] < 0] = -torch.inf
        next_max = torch.maximum(maximum, score.max())
        correction = (maximum - next_max).exp()
        probabilities = (score - next_max).exp()
        denominator = denominator * correction + probabilities.sum()
        numerator = numerator * correction + probabilities.bfloat16().to(dtype) @ keys[start:start + 64].to(dtype)
        maximum = next_max
    final_max = torch.maximum(maximum, sink.to(dtype))
    correction = (maximum - final_max).exp()
    denominator = denominator * correction + (sink.to(dtype) - final_max).exp()
    return (numerator * (correction / denominator)).bfloat16()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("--sample-pairs", type=int, default=64)
    args = parser.parse_args()
    data = torch.load(args.capture, map_location="cpu", weights_only=True)
    actual = data["actual"].unflatten(-1, (16, 512))
    expected = data["expected"].unflatten(-1, (16, 512))
    query = data["q"].unflatten(-1, (16, 512))
    changed = (actual != expected).nonzero(as_tuple=False)
    pairs = list(dict.fromkeys((int(t), int(h)) for t, h, _ in changed))
    torch.manual_seed(17)
    sampled = torch.randperm(query.shape[0] * query.shape[1])[:args.sample_pairs]
    pairs.extend((int(flat // query.shape[1]), int(flat % query.shape[1])) for flat in sampled)
    pairs = list(dict.fromkeys(pairs))
    print(f"changed_elements={len(changed)} evaluated_pairs={len(pairs)}")
    for chunk in (0, 256, 128, 64, 32):
        for accurate_online in (False, True):
            matches = 0
            introduced = 0
            score_error = 0.0
            for t, h in pairs:
                q = query[t, h]
                keys = data["selected"][t]
                candidate = replay(q, keys, data["indices"][t], data["sink"][h], chunk, accurate_online)
                mismatch = actual[t, h] != expected[t, h]
                matches += int((candidate[mismatch] == expected[t, h][mismatch]).sum())
                introduced += int((candidate[~mismatch] != expected[t, h][~mismatch]).sum())
                score_error = max(score_error, float((scores(q, keys, chunk).double() -
                    keys.double() @ q.double()).abs().max()))
            print(f"chunk={chunk:3} online={'fp64' if accurate_online else 'fp32'} "
                  f"fixed={matches}/{len(changed)} introduced={introduced} "
                  f"max_score_abs_error={score_error:.9g}", flush=True)


if __name__ == "__main__":
    main()
