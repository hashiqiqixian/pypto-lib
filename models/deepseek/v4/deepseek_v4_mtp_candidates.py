# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F401,F403,F405,F821
"""DeepSeek-V4 MTP candidate-token selection smoke.

This validates the device-side contract between MTP logits and prefix
acceptance metadata for a local vocab shard:

    candidate_logits -> candidate_token_ids + candidate_scores

Full TP vocabulary gathering remains covered by ``lm_head.py`` and serving
integration. This smoke only proves that the flattened `[T, vocab_shard]` MTP
logits layout can be converted into `[B, S]` candidate metadata on device.
"""

import pypto.language as pl

from config import DECODE_BATCH, DECODE_SEQ
from deepseek_v4_mtp_logits import VOCAB_SHARD


B = DECODE_BATCH
S = DECODE_SEQ
T = B * S


@pl.jit.inline
def mtp_select_top1(
    candidate_logits: pl.Tensor[[T, VOCAB_SHARD], pl.FP32],
    candidate_token_ids: pl.Out[pl.Tensor[[B, S], pl.INT32]],
    candidate_scores: pl.Out[pl.Tensor[[B, S], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_select_top1"):
        for t in pl.range(T):
            best_val = pl.read(candidate_logits, [t, 0])
            best_idx = pl.cast(0, pl.INT32)
            for v in pl.range(1, VOCAB_SHARD):
                cur = pl.read(candidate_logits, [t, v])
                if cur > best_val:
                    best_val = cur
                    best_idx = pl.cast(v, pl.INT32)

            b = t // S
            s = t - b * S
            pl.write(candidate_token_ids, [b, s], best_idx)
            pl.write(candidate_scores, [b, s], best_val)

    return candidate_token_ids, candidate_scores


@pl.jit
def deepseek_v4_mtp_candidates(
    candidate_logits: pl.Tensor[[T, VOCAB_SHARD], pl.FP32],
    candidate_token_ids: pl.Out[pl.Tensor[[B, S], pl.INT32]],
    candidate_scores: pl.Out[pl.Tensor[[B, S], pl.FP32]],
):
    return mtp_select_top1(candidate_logits, candidate_token_ids, candidate_scores)


def golden_deepseek_v4_mtp_candidates(tensors):
    values, indices = tensors["candidate_logits"].float().max(dim=-1)
    tensors["candidate_token_ids"][:] = indices.reshape(B, S).to(tensors["candidate_token_ids"].dtype)
    tensors["candidate_scores"][:] = values.reshape(B, S).to(tensors["candidate_scores"].dtype)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_candidate_logits():
        logits = torch.randn(T, VOCAB_SHARD, dtype=torch.float32)
        for t in range(T):
            idx = (t * 37 + 11) % VOCAB_SHARD
            logits[t, idx] = logits[t].max() + 10.0
        return logits

    return [
        TensorSpec("candidate_logits", [T, VOCAB_SHARD], torch.float32, init_value=init_candidate_logits),
        TensorSpec("candidate_token_ids", [B, S], torch.int32, is_output=True),
        TensorSpec("candidate_scores", [B, S], torch.float32, is_output=True),
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
        fn=deepseek_v4_mtp_candidates,
        specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_mtp_candidates,
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
