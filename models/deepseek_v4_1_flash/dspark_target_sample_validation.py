# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Isolated EP4 target-row sampling validation, outside the product PR."""

import argparse
import os

import torch
from pypto.ir import DistributedConfig

from golden import TensorSpec, run

import dspark_target_sample as sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=4)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    devices = [int(item) for item in os.environ.get("TASK_DEVICE", "0,1,2,3").split(",")]
    assert len(devices) == sample.EP_SIZE == 4
    tokens = 80
    widths = torch.tensor([6, 4, 2, 0, 6, 1, 5, 3, 6, 6, 0, 1], dtype=torch.int32).repeat(4, 1)
    if args.compile_only:
        x_hc = None
        pre_mix = None
        norm_weight = None
        head_weight = None
    else:
        x_hc = torch.zeros((4, tokens, sample.HC_MULT, sample.D), dtype=torch.float32)
        x_hc[:, :, 0, 0] = 1.0
        pre_mix = torch.zeros((4, tokens, sample.HC_MULT), dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0
        norm_weight = torch.ones((4, sample.D), dtype=torch.bfloat16)
        head_weight = torch.zeros((4, sample.VOCAB_PER_TP, sample.D), dtype=torch.bfloat16)
        head_weight[:, 7, 0] = 1.0

    def golden(values):
        expected = values["sampled_ids"]
        for rank in range(4):
            for request in range(12):
                for offset in range(int(widths[rank, request])):
                    expected[rank, request * 6 + offset, 0] = 7

    def compare_samples(actual, expected, **_kwargs):
        mismatches = (actual != expected).nonzero(as_tuple=False)
        first = mismatches[:16].tolist()
        details = [(index, int(actual[tuple(index)]), int(expected[tuple(index)])) for index in first]
        print("sample mismatch", len(mismatches), details, flush=True)
        return not bool(mismatches.numel()), f"mismatch_count={len(mismatches)} first={details}"

    specs = [
        TensorSpec("x_hc", [4, tokens, sample.HC_MULT, sample.D], torch.float32, init_value=x_hc),
        TensorSpec("pre_mix", [4, tokens, sample.HC_MULT], torch.float32, init_value=pre_mix),
        TensorSpec("final_norm_weight", [4, sample.D], torch.bfloat16, init_value=norm_weight),
        TensorSpec("lm_head_weight", [4, sample.VOCAB_PER_TP, sample.D], torch.bfloat16, init_value=head_weight),
        TensorSpec("active_widths", [4, 12], torch.int32, init_value=widths),
        TensorSpec("sampled_ids", [4, 72, 8], torch.int32),
    ]
    result = run(
        fn=sample.l3_sample_target_rows,
        specs=specs,
        golden_fn=golden,
        compare_fn={"sampled_ids": compare_samples},
        config={
            "backend_type": "a5",
            "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
            "ring_heap": (4 * 1024**3,) * 4,
        },
        compile_only=args.compile_only,
    )
    print("target_sample", result.passed, result.work_dir, flush=True)
    if not result.passed:
        raise RuntimeError("target sample mismatch")


if __name__ == "__main__":
    main()
