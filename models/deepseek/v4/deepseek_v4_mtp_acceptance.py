# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F401,F403,F405,F821
"""DeepSeek-V4 MTP prefix-acceptance metadata smoke.

This is the serving-facing contract after MTP candidate logits have already
been converted to candidate token ids. It does not sample from logits and does
not update KV cache. The kernel only validates the device-side metadata needed
to advance state by multiple accepted tokens:

    candidate_token_ids + verify_token_ids
        -> accepted_count
        -> accepted_token_ids
        -> next_position_ids / next_seq_lens

For the first DeepSeek-V4 MTP path ``S = 2``. Acceptance is prefix-based: the
second candidate can only be accepted when the first one was accepted.
"""

import pypto.language as pl

from config import DECODE_BATCH, DECODE_SEQ


B = DECODE_BATCH
S = DECODE_SEQ

assert S == 2, "DeepSeek-V4 MTP acceptance smoke currently models the first two-token MTP path"


@pl.jit.inline
def mtp_acceptance_metadata(
    candidate_token_ids: pl.Tensor[[B, S], pl.INT32],
    verify_token_ids: pl.Tensor[[B, S], pl.INT32],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    seq_lens: pl.Tensor[[B], pl.INT32],
    accepted_count: pl.Out[pl.Tensor[[B, 1], pl.INT32]],
    accepted_token_ids: pl.Out[pl.Tensor[[B, S], pl.INT32]],
    next_position_ids: pl.Out[pl.Tensor[[B, S], pl.INT32]],
    next_seq_lens: pl.Out[pl.Tensor[[B, 1], pl.INT32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_acceptance_metadata"):
        zero = pl.cast(0, pl.INT32)
        one = pl.cast(1, pl.INT32)
        two = pl.cast(2, pl.INT32)

        for b in pl.range(B):
            base_pos = pl.read(position_ids, [b, 0])
            base_len = pl.read(seq_lens, [b])

            cand0 = pl.read(candidate_token_ids, [b, 0])
            verify0 = pl.read(verify_token_ids, [b, 0])
            cand1 = pl.read(candidate_token_ids, [b, 1])
            verify1 = pl.read(verify_token_ids, [b, 1])

            count = zero
            token0 = zero
            token1 = zero
            if cand0 == verify0:
                count = one
                token0 = cand0
                if cand1 == verify1:
                    count = two
                    token1 = cand1

            pl.write(accepted_count, [b, 0], count)
            pl.write(accepted_token_ids, [b, 0], token0)
            pl.write(accepted_token_ids, [b, 1], token1)
            pl.write(next_seq_lens, [b, 0], base_len + count)
            pl.write(next_position_ids, [b, 0], base_pos + count)
            pl.write(next_position_ids, [b, 1], base_pos + count + one)

    return accepted_count, accepted_token_ids, next_position_ids, next_seq_lens


@pl.jit
def deepseek_v4_mtp_acceptance(
    candidate_token_ids: pl.Tensor[[B, S], pl.INT32],
    verify_token_ids: pl.Tensor[[B, S], pl.INT32],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    seq_lens: pl.Tensor[[B], pl.INT32],
    accepted_count: pl.Out[pl.Tensor[[B, 1], pl.INT32]],
    accepted_token_ids: pl.Out[pl.Tensor[[B, S], pl.INT32]],
    next_position_ids: pl.Out[pl.Tensor[[B, S], pl.INT32]],
    next_seq_lens: pl.Out[pl.Tensor[[B, 1], pl.INT32]],
):
    return mtp_acceptance_metadata(
        candidate_token_ids,
        verify_token_ids,
        position_ids,
        seq_lens,
        accepted_count,
        accepted_token_ids,
        next_position_ids,
        next_seq_lens,
    )


def golden_deepseek_v4_mtp_acceptance(tensors):
    candidate = tensors["candidate_token_ids"]
    verify = tensors["verify_token_ids"]
    position_ids = tensors["position_ids"]
    seq_lens = tensors["seq_lens"]

    accepted_count = tensors["accepted_count"]
    accepted_token_ids = tensors["accepted_token_ids"]
    next_position_ids = tensors["next_position_ids"]
    next_seq_lens = tensors["next_seq_lens"]

    accepted_count.zero_()
    accepted_token_ids.zero_()
    next_position_ids.zero_()
    next_seq_lens.zero_()

    for b in range(B):
        count = 0
        if int(candidate[b, 0].item()) == int(verify[b, 0].item()):
            count = 1
            accepted_token_ids[b, 0] = candidate[b, 0]
            if int(candidate[b, 1].item()) == int(verify[b, 1].item()):
                count = 2
                accepted_token_ids[b, 1] = candidate[b, 1]

        accepted_count[b, 0] = count
        next_seq_lens[b, 0] = seq_lens[b] + count
        next_position_ids[b, 0] = position_ids[b, 0] + count
        next_position_ids[b, 1] = position_ids[b, 0] + count + 1


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_candidate_token_ids():
        values = torch.empty(B, S, dtype=torch.int32)
        for b in range(B):
            values[b, 0] = 1000 + b * 17
            values[b, 1] = 2000 + b * 17
        return values

    candidate = init_candidate_token_ids()

    def init_verify_token_ids():
        verify = candidate.clone()
        for b in range(B):
            mode = b % 3
            if mode == 1:
                verify[b, 1] = verify[b, 1] + 13
            elif mode == 2:
                verify[b, 0] = verify[b, 0] + 11
                verify[b, 1] = verify[b, 1] + 13
        return verify

    def init_position_ids():
        base = torch.arange(4096, 4096 + B, dtype=torch.int32).reshape(B, 1)
        return base + torch.arange(S, dtype=torch.int32).reshape(1, S)

    def init_seq_lens():
        return torch.arange(4096, 4096 + B, dtype=torch.int32)

    return [
        TensorSpec("candidate_token_ids", [B, S], torch.int32, init_value=lambda: candidate.clone()),
        TensorSpec("verify_token_ids", [B, S], torch.int32, init_value=init_verify_token_ids),
        TensorSpec("position_ids", [B, S], torch.int32, init_value=init_position_ids),
        TensorSpec("seq_lens", [B], torch.int32, init_value=init_seq_lens),
        TensorSpec("accepted_count", [B, 1], torch.int32, is_output=True),
        TensorSpec("accepted_token_ids", [B, S], torch.int32, is_output=True),
        TensorSpec("next_position_ids", [B, S], torch.int32, is_output=True),
        TensorSpec("next_seq_lens", [B, 1], torch.int32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=deepseek_v4_mtp_acceptance,
        specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_mtp_acceptance,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=0.0,
        atol=0.0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
