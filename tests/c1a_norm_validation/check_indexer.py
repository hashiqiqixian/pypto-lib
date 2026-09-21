# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded-workspace indexer checks with independently constructed exact scores."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import pypto.language as pl
from golden import run, TensorSpec
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.prefill_c1a_indexer import make_paged_indexer

T, S = 9, 16512
P = S // 128
D, Q, H, I, R, K = C.D, C.Q_LORA, C.INDEX_H, C.INDEX_DIM, C.ROPE_DIM, C.INDEX_TOPK


def make_entry(candidates):
    indexer = make_paged_indexer(use_candidates=candidates)

    @pl.jit
    def entry(
        x: pl.Tensor[[T, D], pl.BF16],
        latent: pl.Tensor[[T, Q], pl.BF16],
        requests: pl.Tensor[[T], pl.INT32],
        lengths: pl.Tensor[[T], pl.INT32],
        cache: pl.Tensor[[2 * P, 128, 1, I // 2], pl.UINT8],
        scales: pl.Tensor[[2 * P, 128, 1, I // C.INDEX_CACHE_GROUP], pl.FP8E8M0],
        table: pl.Tensor[[2, P], pl.INT32],
        cos: pl.Tensor[[T, R // 2], pl.FP32],
        sin: pl.Tensor[[T, R // 2], pl.FP32],
        wq: pl.Tensor[[Q, H * I], pl.FP8E4M3FN],
        wq_scale: pl.Tensor[[Q // 32, H * I], pl.FP8E8M0, pl.MX_B_NN],
        weights: pl.Tensor[[D, H], pl.BF16],
        mask: pl.Tensor[[T, S], pl.UINT8],
        scores: pl.Out[pl.Tensor[[T, S], pl.FP32]],
        topk: pl.Out[pl.Tensor[[T, K], pl.INT32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="validation_cache_ready") as ready:
            pl.write(scores, [0, 0], 0.0)
        indexer(x, latent, requests, lengths, cache, scales, table, cos, sin,
                wq, wq_scale, weights, mask, scores, topk, T, ready)
        return scores, topk
    return entry


def fixture(candidates):
    torch.manual_seed(37)
    values = {
        "x": torch.zeros(T, D, dtype=torch.bfloat16),
        "latent": torch.zeros(T, Q, dtype=torch.bfloat16),
        "requests": torch.arange(T, dtype=torch.int32) % 2,
        "lengths": torch.tensor([0, 1, 128, 129, 512, 513, 2048, 2049, S - 5], dtype=torch.int32),
        "cache": torch.randint(0, 256, (2 * P, 128, 1, I // 2), dtype=torch.uint8),
        "scales": torch.full((2 * P, 128, 1, I // C.INDEX_CACHE_GROUP), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu),
        "table": torch.randperm(2 * P, dtype=torch.int32).reshape(2, P),
        "cos": torch.ones(T, R // 2), "sin": torch.zeros(T, R // 2),
        "wq": torch.zeros(Q, H * I).to(torch.float8_e4m3fn),
        "wq_scale": torch.full((Q // 32, H * I), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu),
        "weights": torch.zeros(D, H, dtype=torch.bfloat16),
        "mask": torch.ones(T, S, dtype=torch.uint8),
        "scores": torch.empty(T, S), "topk": torch.empty(T, K, dtype=torch.int32),
    }
    values["x"][:, 0] = torch.arange(1, T + 1)
    values["latent"][:, 0] = torch.arange(1, T + 1) / 2
    wq = values["wq"].float()
    wq[0, ::I] = 1
    values["wq"] = wq.to(torch.float8_e4m3fn)
    values["weights"][0, :] = 1
    if candidates:
        values["mask"][:, ::3] = 0
    return values


def reference(v, candidates):
    # One-hot projections leave one scalar key channel per head. Decode its
    # FP4 nibble via a literal codebook, independently of the production helper.
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
    key = lut[(v["cache"].reshape(-1, I // 2)[:, 0] & 15).long()]
    v["scores"].fill_(-1e30)
    v["topk"].fill_(-1)
    for t, length in enumerate(v["lengths"].tolist()):
        logical = torch.arange(length)
        physical = v["table"][v["requests"][t].long(), logical // 128].long() * 128 + logical % 128
        dot = (key[physical] * v["latent"][t, 0].float()).bfloat16().float().relu()
        weight = (v["x"][t, 0].float() * I ** -.5 * H ** -.5).bfloat16().float()
        score = (dot * weight).bfloat16().float().mul(H).bfloat16().float()
        if candidates:
            score = score.masked_fill(v["mask"][t, :length] == 0, -1e30)
        v["scores"][t, :length] = score
        if length <= K:
            v["topk"][t, :length] = physical.int()
        else:
            chosen = score.topk(K).indices.sort().values
            chosen = chosen[score[chosen] > -1e29]
            v["topk"][t, :len(chosen)] = physical[chosen].int()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--candidates", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    values = fixture(args.candidates)
    expected = {n: v.clone() for n, v in values.items()}
    reference(expected, args.candidates)

    def compare_topk(actual, expected_topk, **kwargs):
        for t, length in enumerate(values["lengths"].tolist()):
            if length <= K:
                assert torch.equal(actual[t], expected_topk[t])
                continue
            inv = {int(values["table"][values["requests"][t].long(), j // 128]) * 128 + j % 128: j
                   for j in range(length)}
            chosen = [inv[int(row)] for row in actual[t] if row >= 0]
            assert chosen == sorted(set(chosen))
            scores = expected["scores"][t, :length]
            count = min(K, int((scores > -1e29).sum()))
            assert len(chosen) == count
            assert bool((actual[t, count:] == -1).all())
            cutoff = scores.topk(count).values[-1]
            assert bool((scores[chosen] >= cutoff).all())
            assert set(torch.where(scores > cutoff)[0].tolist()).issubset(chosen)
        return True

    result = run(fn=make_entry(args.candidates),
                 specs=[TensorSpec(n, list(v.shape), v.dtype, init_value=lambda n=n: values[n].clone())
                        for n, v in values.items()],
                 golden_fn=lambda v: reference(v, args.candidates),
                 compare_fn={"scores": lambda a, b, **kw: torch.equal(a, b), "topk": compare_topk},
                 compile_only=args.compile_only,
                 config={"platform": "a5", "device_id": args.d})
    assert result.passed, result.error


if __name__ == "__main__":
    main()
