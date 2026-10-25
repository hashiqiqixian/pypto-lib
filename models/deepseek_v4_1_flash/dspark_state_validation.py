# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Isolated EP4 validation; this file is intentionally outside the product PR."""

import argparse
import os

import torch
from pypto.ir import DistributedConfig

from golden import TensorSpec, run

import dspark_state as state


def execute(name, fn, inputs, outputs, golden, devices, compile_only):
    names = list(inputs) + list(outputs)
    specs = []
    for item in names:
        if item in inputs:
            value = inputs[item]
            specs.append(TensorSpec(item, list(value.shape), value.dtype, init_value=value))
        else:
            shape, dtype = outputs[item]
            specs.append(TensorSpec(item, shape, dtype))
    result = run(
        fn=fn,
        specs=specs,
        golden_fn=golden,
        config={"backend_type": "a5", "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0)},
        compile_only=compile_only,
    )
    print(name, result.passed, result.work_dir, flush=True)
    if not result.passed:
        raise RuntimeError(f"{name} failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=4)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    devices = [int(item) for item in os.environ.get("TASK_DEVICE", "0,1,2,3").split(",")]
    ep = state.EP_SIZE
    batch = state.BATCH
    assert ep == len(devices) == 4
    slots = torch.full((ep, batch), -1, dtype=torch.int32)
    slots[:, :4] = torch.arange(4, dtype=torch.int32)
    generations = torch.ones((ep, batch), dtype=torch.int32)
    prompts = torch.zeros((ep, batch), dtype=torch.int32)
    prompts[:, :4] = torch.arange(20, 24, dtype=torch.int32)
    limits = torch.full((ep, batch), 100, dtype=torch.int32)
    limits[:, 3] = 29
    sampled_rows = torch.full((ep, batch), -1, dtype=torch.int32)
    sampled_rows[:, :4] = torch.arange(4, dtype=torch.int32)
    prefill_samples = torch.zeros((ep, 16, 8), dtype=torch.int32)
    prefill_samples[:, :4, 0] = torch.arange(10, 14, dtype=torch.int32)
    drafts = torch.zeros((ep, batch, 5), dtype=torch.int32)
    for request in range(4):
        drafts[:, request] = torch.arange(20 + request * 10, 25 + request * 10, dtype=torch.int32)
    initial_tokens = torch.zeros((ep, batch, 6), dtype=torch.int64)
    initial_meta = torch.zeros((ep, batch, 6), dtype=torch.int32)
    initial_meta[:, :, state.STATE_GENERATION] = 1

    def bootstrap_input_golden(values):
        for rank in range(ep):
            for request in range(4):
                values["next_prefill_tokens"][rank, request] = prefill_samples[rank, request, 0]
                values["first_query_positions"][rank, request] = prompts[rank, request]

    execute(
        "bootstrap_inputs", state.l3_bootstrap_inputs,
        dict(slot_ids=slots, generations=generations, prompt_lengths=prompts, position_limits=limits,
             sampled_rows=sampled_rows, sampled_ids=prefill_samples, state_meta=initial_meta),
        dict(next_prefill_tokens=([ep, batch], torch.int64), first_query_positions=([ep, batch], torch.int32)),
        bootstrap_input_golden, devices, args.compile_only,
    )

    def bootstrap_golden(values):
        for rank in range(ep):
            tokens, meta = state.golden_bootstrap(
                slots[rank], generations[rank], prompts[rank], limits[rank], sampled_rows[rank],
                prefill_samples[rank], drafts[rank], initial_tokens[rank], initial_meta[rank],
            )
            values["state_tokens"][rank].copy_(tokens)
            values["state_meta"][rank].copy_(meta)

    execute(
        "bootstrap_publish", state.l3_publish_bootstrap,
        dict(slot_ids=slots, generations=generations, prompt_lengths=prompts, position_limits=limits,
             sampled_rows=sampled_rows, sampled_ids=prefill_samples, draft_ids=drafts,
             state_tokens=initial_tokens, state_meta=initial_meta),
        {}, bootstrap_golden, devices, args.compile_only,
    )
    published_tokens = initial_tokens.clone()
    published_meta = initial_meta.clone()
    for rank in range(ep):
        published_tokens[rank], published_meta[rank] = state.golden_bootstrap(
            slots[rank], generations[rank], prompts[rank], limits[rank], sampled_rows[rank],
            prefill_samples[rank], drafts[rank], initial_tokens[rank], initial_meta[rank],
        )

    def prepare_golden(values):
        for rank in range(ep):
            ids, positions, widths = state.golden_prepare_target(
                slots[rank], generations[rank], published_tokens[rank], published_meta[rank],
            )
            values["input_ids"][rank].copy_(ids)
            values["position_ids"][rank].copy_(positions)
            values["active_widths"][rank].copy_(widths)

    execute(
        "target_prepare", state.l3_prepare_target,
        dict(slot_ids=slots, generations=generations, state_tokens=published_tokens, state_meta=published_meta),
        dict(input_ids=([ep, batch, 6], torch.int64), position_ids=([ep, batch, 6], torch.int32),
             active_widths=([ep, batch], torch.int32)),
        prepare_golden, devices, args.compile_only,
    )
    widths = torch.zeros((ep, batch), dtype=torch.int32)
    widths[:, :4] = 6
    target_samples = torch.zeros((ep, batch * 6, 8), dtype=torch.int32)
    target_samples[:, 0, 0] = 99
    target_samples[:, 6, 0] = drafts[:, 1, 0]
    target_samples[:, 7, 0] = drafts[:, 1, 1]
    target_samples[:, 8, 0] = 99
    target_samples[:, 12:17, 0] = drafts[:, 2, :]
    target_samples[:, 17, 0] = 99
    target_samples[:, 18:23, 0] = drafts[:, 3, :]
    target_samples[:, 23, 0] = 99

    def accept_golden(values):
        for rank in range(ep):
            tokens, meta, ids, counts = state.golden_accept_target(
                slots[rank], generations[rank], widths[rank], target_samples[rank],
                published_tokens[rank], published_meta[rank],
            )
            values["state_tokens"][rank].copy_(tokens)
            values["state_meta"][rank].copy_(meta)
            values["accepted_ids"][rank].copy_(ids)
            values["accepted_counts"][rank].copy_(counts)
            for request in range(4):
                accepted = int(counts[request])
                anchor = int(published_meta[rank, request, state.STATE_ANCHOR])
                can_draft = int(anchor + accepted + 5 < int(published_meta[rank, request, state.STATE_LIMIT]))
                values["num_sampled"][rank, request] = accepted * can_draft
                values["last_sampled"][rank, request] = tokens[request, 0]
                values["first_query_positions"][rank, request] = (anchor + accepted) * can_draft
                values["context_valid"][rank, request, :accepted] = can_draft
                values["context_positions"][rank, request, :accepted] = torch.arange(anchor, anchor + accepted)

    execute(
        "target_accept", state.l3_accept_target,
        dict(slot_ids=slots, generations=generations, active_widths=widths, sampled_ids=target_samples,
             state_tokens=published_tokens, state_meta=published_meta),
        dict(accepted_ids=([ep, batch, 6], torch.int32), accepted_counts=([ep, batch], torch.int32),
             context_valid=([ep, batch, 6], torch.int32), context_positions=([ep, batch, 6], torch.int32),
             num_sampled=([ep, batch], torch.int32), last_sampled=([ep, batch], torch.int64),
             first_query_positions=([ep, batch], torch.int32)),
        accept_golden, devices, args.compile_only,
    )
    accepted_tokens = published_tokens.clone()
    accepted_meta = published_meta.clone()
    for rank in range(ep):
        accepted_tokens[rank], accepted_meta[rank], _, _ = state.golden_accept_target(
            slots[rank], generations[rank], widths[rank], target_samples[rank],
            published_tokens[rank], published_meta[rank],
        )
    new_drafts = drafts + 100

    def commit_golden(values):
        for rank in range(ep):
            tokens, meta, committed = state.golden_commit_drafts(
                slots[rank], generations[rank], new_drafts[rank], accepted_tokens[rank], accepted_meta[rank],
            )
            values["state_tokens"][rank].copy_(tokens)
            values["state_meta"][rank].copy_(meta)
            values["committed"][rank].copy_(committed)

    execute(
        "draft_commit", state.l3_commit_drafts,
        dict(slot_ids=slots, generations=generations, draft_ids=new_drafts,
             state_tokens=accepted_tokens, state_meta=accepted_meta),
        dict(committed=([ep, batch], torch.int32)),
        commit_golden, devices, args.compile_only,
    )


if __name__ == "__main__":
    main()
