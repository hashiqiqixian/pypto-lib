# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""EP4 scratch validation; this file belongs only to a validation branch."""

import argparse
import os

import torch
from golden import TensorSpec, run
from pypto.ir import DistributedConfig

import dspark_state as state
import dspark_target_sample as sample
import dspark_drafter as drafter


def check(name, fn, specs, golden, devices, compile_only):
    result = run(
        fn=fn,
        specs=specs,
        golden_fn=golden,
        config={
            "backend_type": "a5",
            "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
            "ring_heap": (4 * 1024**3,) * 4,
        },
        compile_only=compile_only,
    )
    print(name, result.passed, result.work_dir, flush=True)
    if not result.passed:
        raise RuntimeError(f"{name} failed")


def validate_merge(devices, compile_only):
    chunks = [
        torch.arange(4 * 16 * 8, dtype=torch.int32).reshape(4, 16, 8) + 1000 * index
        for index in range(5)
    ]

    def golden(values):
        values["sampled_ids"].copy_(torch.cat([values[f"chunk_{i}"] for i in range(5)], dim=1)[:, :72])

    specs = [TensorSpec(f"chunk_{i}", [4, 16, 8], torch.int32, init_value=chunks[i]) for i in range(5)]
    specs.append(TensorSpec("sampled_ids", [4, 72, 8], torch.int32))
    check("merge_target_chunks", sample.l3_merge_target_chunks, specs, golden, devices, compile_only)


def geometry_inputs():
    block_tables = torch.empty((4, 3, 12, 4), dtype=torch.int32)
    for layer in range(3):
        for request in range(12):
            block_tables[:, layer, request] = 100 * layer + 4 * request + torch.arange(4)
    cos = torch.arange(256 * 32, dtype=torch.float32).reshape(1, 256, 32).repeat(4, 1, 1) / 10000
    sin = -cos
    first = torch.arange(12, dtype=torch.int32).repeat(4, 1) + 30
    first[:, 3] = 0
    return block_tables, cos, sin, first


def output_specs(tokens):
    return [
        TensorSpec("context_slots", [4, 3, tokens], torch.int64),
        TensorSpec("context_rope_cos", [4, tokens, 32], torch.float32),
        TensorSpec("context_rope_sin", [4, tokens, 32], torch.float32),
        TensorSpec("query_rope_cos", [4, 60, 32], torch.float32),
        TensorSpec("query_rope_sin", [4, 60, 32], torch.float32),
    ]


def fill_geometry(values, positions, request_ids, valid, first, blocks):
    values["context_slots"].fill_(-1)
    for name in ("context_rope_cos", "context_rope_sin", "query_rope_cos", "query_rope_sin"):
        values[name].zero_()
    for rank in range(4):
        for row in range(positions.shape[1]):
            request = int(request_ids[rank, row])
            position = int(positions[rank, row])
            if not bool(valid[rank, row]) or not 0 <= request < 12 or not 0 <= position < 256:
                continue
            values["context_rope_cos"][rank, row] = values["rope_cos_table"][rank, position]
            values["context_rope_sin"][rank, row] = values["rope_sin_table"][rank, position]
            for layer in range(3):
                physical = int(blocks[rank, layer, request, position // 128])
                if physical >= 0:
                    values["context_slots"][rank, layer, row] = physical * 128 + position % 128
        for request in range(12):
            begin = int(first[rank, request])
            if begin <= 0:
                continue
            for offset in range(5):
                if begin + offset >= 256:
                    continue
                row = request * 5 + offset
                values["query_rope_cos"][rank, row] = values["rope_cos_table"][rank, begin + offset]
                values["query_rope_sin"][rank, row] = values["rope_sin_table"][rank, begin + offset]


def validate_decode_geometry(devices, compile_only):
    blocks, cos, sin, first = geometry_inputs()
    valid = torch.ones((4, 12, 6), dtype=torch.int32)
    valid[:, 3] = 0
    valid[:, :, 5] = 0
    positions = torch.arange(12, dtype=torch.int32).reshape(1, 12, 1) * 6 + 20
    positions = (positions + torch.arange(6, dtype=torch.int32)).repeat(4, 1, 1)

    def golden(values):
        requests = torch.arange(12, dtype=torch.int32).repeat_interleave(6).repeat(4, 1)
        fill_geometry(values, positions.reshape(4, 72), requests, valid.reshape(4, 72), first, blocks)

    specs = [
        TensorSpec("context_valid", [4, 12, 6], torch.int32, init_value=valid),
        TensorSpec("context_positions", [4, 12, 6], torch.int32, init_value=positions),
        TensorSpec("first_query_positions", [4, 12], torch.int32, init_value=first),
        TensorSpec("block_tables", [4, 3, 12, 4], torch.int32, init_value=blocks),
        TensorSpec("rope_cos_table", [4, 256, 32], torch.float32, init_value=cos),
        TensorSpec("rope_sin_table", [4, 256, 32], torch.float32, init_value=sin),
        *output_specs(72),
    ]
    check("decode_geometry", state.l3_prepare_drafter_geometry, specs, golden, devices, compile_only)


def validate_prefill_geometry(devices, compile_only):
    blocks, cos, sin, first = geometry_inputs()
    positions = (torch.arange(16, dtype=torch.int32) + 20).repeat(4, 1)
    requests = torch.zeros((4, 16), dtype=torch.int32)
    requests[:, 8:] = 1
    requests[:, 15] = -1
    counts = torch.tensor([[16], [10], [8], [0]], dtype=torch.int32)

    def golden(values):
        active = (torch.arange(16).reshape(1, 16) < counts.reshape(4, 1)).to(torch.int32)
        fill_geometry(values, positions, requests, active, first, blocks)

    specs = [
        TensorSpec("positions", [4, 16], torch.int32, init_value=positions),
        TensorSpec("request_ids", [4, 16], torch.int32, init_value=requests),
        TensorSpec("num_tokens", [4, 1], torch.int32, init_value=counts),
        TensorSpec("first_query_positions", [4, 12], torch.int32, init_value=first),
        TensorSpec("block_tables", [4, 3, 12, 4], torch.int32, init_value=blocks),
        TensorSpec("rope_cos_table", [4, 256, 32], torch.float32, init_value=cos),
        TensorSpec("rope_sin_table", [4, 256, 32], torch.float32, init_value=sin),
        *output_specs(16),
    ]
    check("prefill_geometry", state.l3_prepare_prefill_drafter_geometry, specs, golden, devices, compile_only)


def validate_inactive_query(device, compile_only):
    anchors = torch.tensor([7, 0], dtype=torch.int64)
    first = torch.tensor([125, 0], dtype=torch.int32)
    tables = torch.tensor(
        [[[4, 5], [34, 35]], [[14, 15], [44, 45]], [[24, 25], [54, 55]]],
        dtype=torch.int32,
    )
    width = drafter.DSPARK_SWA_INDEX_WIDTH

    def golden(values):
        active = drafter.golden_query_metadata(anchors[:1], first[:1], tables[:, :1], drafter.DSPARK_NOISE_TOKEN_ID)
        values["query_ids"].zero_()
        values["query_positions"].zero_()
        values["query_slots"].fill_(-1)
        values["window_indices"].fill_(-1)
        values["window_lens"].zero_()
        values["query_ids"][:5] = active.token_ids.reshape(-1)
        values["query_positions"][:5] = active.positions.reshape(-1)
        values["query_slots"][:, :5] = active.query_slots.reshape(3, 5)
        values["window_indices"][:, :5] = active.window_indices.reshape(3, 5, width)
        values["window_lens"][:, :1] = active.window_lens

    specs = [
        TensorSpec("anchor_ids", [2], torch.int64, init_value=anchors),
        TensorSpec("first_query_positions", [2], torch.int32, init_value=first),
        TensorSpec("block_tables", [3, 2, 2], torch.int32, init_value=tables),
        TensorSpec("query_ids", [10], torch.int64),
        TensorSpec("query_positions", [10], torch.int32),
        TensorSpec("query_slots", [3, 10], torch.int64),
        TensorSpec("window_indices", [3, 10, width], torch.int32),
        TensorSpec("window_lens", [3, 2], torch.int32),
    ]
    result = run(
        fn=drafter.dspark_metadata_test,
        specs=specs,
        golden_fn=golden,
        config={"backend_type": "a5", "device_id": device},
        compile_only=compile_only,
    )
    print("inactive_query", result.passed, result.work_dir, flush=True)
    if not result.passed:
        raise RuntimeError("inactive query metadata mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    devices = [int(item) for item in os.environ.get("TASK_DEVICE", "0,1,2,3").split(",")]
    if args.metadata_only:
        validate_inactive_query(devices[0], args.compile_only)
        return
    assert len(devices) == 4
    validate_merge(devices, args.compile_only)
    validate_decode_geometry(devices, args.compile_only)
    validate_prefill_geometry(devices, args.compile_only)


if __name__ == "__main__":
    main()
