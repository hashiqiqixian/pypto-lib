# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Contract tests for the DeepSeek-V4 DSpark synthetic token loop."""

from types import SimpleNamespace

import pytest
import torch

from models.deepseek_v4_flash_dspark.synthetic_token_loop import (
    DECODE_WIDTH,
    QUERY_WIDTH,
    _compiled_alias_compatible,
    _copy_group_request_table,
    _copy_leader_request_table,
    _materialize_drafter_ios,
    _materialize_stage,
    _refresh_drafter_rope_rows,
    _replicate_group_metadata,
    _resident_names,
    _verify_target_step,
)


def test_group_context_metadata_is_rank_major_and_replicated():
    positions = torch.arange(8, dtype=torch.int32).reshape(4, 2)
    slots = torch.arange(2 * 4 * 2, dtype=torch.int64).reshape(4, 2, 2)
    group_positions = torch.empty(4, 8, dtype=torch.int32)
    group_slots = torch.empty(4, 2, 8, dtype=torch.int64)

    _replicate_group_metadata(
        positions,
        slots,
        group_positions,
        group_slots,
    )

    expected_positions = positions.reshape(-1)
    expected_slots = slots.permute(1, 0, 2).reshape(2, -1)
    assert torch.equal(group_positions, expected_positions.unsqueeze(0).expand(4, -1))
    assert torch.equal(group_slots, expected_slots.unsqueeze(0).expand(4, -1, -1))


def test_drafter_rope_rows_follow_each_position_stream():
    context_positions = torch.arange(4 * 5, dtype=torch.int32).reshape(4, 5)
    local_query_positions = (
        torch.arange(4 * 3, dtype=torch.int32).reshape(4, 3) + 32
    )
    group_query_positions = (
        torch.arange(4 * 12, dtype=torch.int32).reshape(4, 12) + 64
    )
    io = {
        "context_group_position_ids": context_positions,
        "query_group_position_ids": group_query_positions,
    }
    for prefix, positions in (
        ("context_group", context_positions),
        ("query", local_query_positions),
        ("query_group", group_query_positions),
    ):
        io[f"{prefix}_freqs_cos"] = torch.empty(
            *positions.shape,
            2,
            dtype=torch.bfloat16,
        )
        io[f"{prefix}_freqs_sin"] = torch.empty_like(
            io[f"{prefix}_freqs_cos"]
        )

    class FakeUtils:
        @staticmethod
        def token_local_rope(_model, _ratio, positions, **_kwargs):
            rows = positions.reshape(-1, 1).to(torch.bfloat16)
            return rows.expand(-1, 2).clone(), (rows + 128).expand(-1, 2).clone()

    drafter = SimpleNamespace(M=object(), MAX_SEQ_LEN=1024, ROPE_DIM=2)
    _refresh_drafter_rope_rows(
        io,
        drafter,
        FakeUtils,
        local_query_positions,
    )

    for prefix, positions in (
        ("context_group", context_positions),
        ("query", local_query_positions),
        ("query_group", group_query_positions),
    ):
        expected = positions.to(torch.bfloat16).unsqueeze(-1).expand(-1, -1, 2)
        assert torch.equal(io[f"{prefix}_freqs_cos"], expected)
        assert torch.equal(io[f"{prefix}_freqs_sin"], expected + 128)


def test_drafter_decode_materialization_resizes_dynamic_rope_rows():
    def spec(name, shape, dtype=torch.float32):
        return SimpleNamespace(
            name=name,
            shape=shape,
            dtype=dtype,
            create_tensor=lambda shape=shape, dtype=dtype: torch.zeros(
                shape,
                dtype=dtype,
            ),
        )

    prefill_specs = (
        spec("target_hidden", (4, 128, 3)),
        spec("context_group_position_ids", (4, 512), torch.int32),
        spec("context_group_slot_mapping", (4, 3, 512), torch.int64),
        spec("context_group_freqs_cos", (4, 512, 64), torch.bfloat16),
        spec("context_group_freqs_sin", (4, 512, 64), torch.bfloat16),
        spec("query_freqs_cos", (4, 112, 64), torch.bfloat16),
    )
    decode_specs = (
        spec("target_hidden", (4, 8, 3)),
        spec("context_group_position_ids", (4, 32), torch.int32),
        spec("context_group_slot_mapping", (4, 3, 32), torch.int64),
        spec("context_group_freqs_cos", (4, 32, 64), torch.bfloat16),
        spec("context_group_freqs_sin", (4, 32, 64), torch.bfloat16),
        spec("query_freqs_cos", (4, 112, 64), torch.bfloat16),
    )
    stage = SimpleNamespace(
        specs=prefill_specs,
        module=SimpleNamespace(
            build_tensor_specs=lambda _batch, *, mode: decode_specs,
        ),
    )

    prefill_io, decode_io = _materialize_drafter_ios(stage, set())

    assert decode_io["target_hidden"].shape == (4, 8, 3)
    assert decode_io["context_group_position_ids"].shape == (4, 32)
    assert decode_io["context_group_slot_mapping"].shape == (4, 3, 32)
    assert decode_io["context_group_freqs_cos"].shape == (4, 32, 64)
    assert decode_io["context_group_freqs_sin"].shape == (4, 32, 64)
    assert decode_io["query_freqs_cos"] is prefill_io["query_freqs_cos"]


def test_dynamic_cache_alias_accepts_one_shared_runtime_extent():
    prefill = SimpleNamespace(
        shape=(4, -1, 32, 1, 512),
        dtype="bf16",
        direction=SimpleNamespace(name="InOut"),
    )
    decode = SimpleNamespace(
        shape=(4, -1, 32, 1, 512),
        dtype="bf16",
        direction=prefill.direction,
    )

    assert _compiled_alias_compatible(prefill, decode, (4, 22016, 32, 1, 512))
    assert not _compiled_alias_compatible(prefill, decode, (4, 22016, 16, 1, 512))


def test_shared_head_alias_requires_input_direction():
    prefill = SimpleNamespace(
        shape=(4, 256, 64),
        dtype="bf16",
        direction=SimpleNamespace(name="In"),
    )
    markov = SimpleNamespace(
        shape=(4, 256, 64),
        dtype="bf16",
        direction=prefill.direction,
    )

    assert _compiled_alias_compatible(
        prefill,
        markov,
        (4, 256, 64),
        direction_name="In",
    )
    assert not _compiled_alias_compatible(prefill, markov, (4, 256, 64))


def test_explicit_output_state_is_resident_across_stage_calls():
    state = SimpleNamespace(
        name="kv_caches",
        is_resident=False,
        shape=(4, 3, 8),
        create_tensor=lambda: torch.zeros(4, 3, 8),
    )
    regular = SimpleNamespace(
        name="target_hidden",
        is_resident=False,
        shape=(4, 8),
        create_tensor=lambda: torch.zeros(4, 8),
    )
    param_infos = {
        "kv_caches": SimpleNamespace(direction=SimpleNamespace(name="Out")),
        "target_hidden": SimpleNamespace(direction=SimpleNamespace(name="Out")),
    }
    resident_names = _resident_names(
        (state, regular),
        param_infos,
        output_state_names=frozenset({"kv_caches"}),
    )

    assert resident_names == {"kv_caches"}
    assert set(_materialize_stage((state, regular), resident_names)) == {"target_hidden"}
    assert not _materialize_stage(
        (state, regular),
        resident_names,
        excluded_names=frozenset({"target_hidden"}),
    )


def test_prefill_allocator_table_replaces_only_tp_group_leaders():
    source = torch.arange(8 * 5, dtype=torch.int32).reshape(8, 1, 5)
    target = torch.full((8, 2, 3), 99, dtype=torch.int32)

    _copy_leader_request_table(target, source)

    assert torch.equal(target[0, 0], source[0, 0, :3])
    assert torch.equal(target[4, 0], source[4, 0, :3])
    assert torch.equal(
        target[[1, 2, 3, 5, 6, 7], 0],
        torch.full((6, 3), 99, dtype=torch.int32),
    )
    assert torch.equal(target[:, 1], torch.full((8, 3), 99, dtype=torch.int32))


def test_prefill_state_table_replaces_group_request_zero_on_every_rank():
    source = torch.arange(4 * 5, dtype=torch.int32).reshape(4, 1, 5)
    target = torch.full((4, 2, 3), 99, dtype=torch.int32)

    _copy_group_request_table(target, source)

    assert torch.equal(target[:, 0], source[:, 0, :3])
    assert torch.equal(target[:, 1], torch.full((4, 3), 99, dtype=torch.int32))


def test_verify_target_step_packs_variable_accepted_prefixes():
    tp_size = 2
    accepted_per_group = (1, 4, 8)
    group_starts = torch.tensor([10, 20, 30], dtype=torch.int32)
    ranks = len(accepted_per_group) * tp_size

    draft_token_ids = torch.empty(ranks, QUERY_WIDTH, dtype=torch.int64)
    target_sampled_ids = torch.empty(ranks, DECODE_WIDTH, dtype=torch.int64)
    next_tokens = (901, 902, 903)
    for group, accepted in enumerate(accepted_per_group):
        group_base = group * tp_size
        drafts = torch.arange(100 * (group + 1), 100 * (group + 1) + QUERY_WIDTH)
        target = torch.full((DECODE_WIDTH,), -1, dtype=torch.int64)
        matched = accepted - 1
        target[:matched] = drafts[:matched]
        target[matched] = next_tokens[group]
        for rank in range(group_base, group_base + tp_size):
            draft_token_ids[rank].copy_(drafts)
            target_sampled_ids[rank].copy_(target)

    target_hidden = torch.arange(
        ranks * DECODE_WIDTH * 3,
        dtype=torch.float32,
    ).reshape(ranks, DECODE_WIDTH, 3)

    verified = _verify_target_step(
        draft_token_ids,
        target_sampled_ids,
        target_hidden,
        group_starts,
        tp_size=tp_size,
    )

    for group, accepted in enumerate(accepted_per_group):
        group_base = group * tp_size
        start = int(group_starts[group])
        anchor = start + accepted - 1
        leader = group_base
        assert int(verified.accepted_counts[leader]) == accepted
        assert int(verified.next_tokens[leader]) == next_tokens[group]
        assert int(verified.anchors[leader]) == anchor
        assert torch.equal(
            verified.target_hidden[leader, :accepted],
            target_hidden[leader, :accepted],
        )
        assert not torch.count_nonzero(verified.target_hidden[leader, accepted:])
        assert torch.equal(
            verified.context_positions[leader, :accepted],
            torch.arange(start, start + accepted, dtype=torch.int32),
        )
        assert not torch.count_nonzero(verified.context_positions[leader, accepted:])
        assert bool(verified.valid_rows[leader, :accepted].all())
        assert not bool(verified.valid_rows[leader, accepted:].any())

        peer = leader + 1
        assert int(verified.accepted_counts[peer]) == 0
        assert int(verified.next_tokens[peer]) == 0
        assert int(verified.anchors[peer]) == -1
        assert not torch.count_nonzero(verified.target_hidden[peer])
        assert not torch.count_nonzero(verified.context_positions[peer])
        assert not bool(verified.valid_rows[peer].any())

    assert torch.equal(
        verified.anchors[::tp_size] + 1,
        torch.tensor([11, 24, 38], dtype=torch.int32),
    )


@pytest.mark.parametrize(
    ("draft_shape", "sample_shape", "hidden_shape", "starts_shape", "error"),
    (
        ((4, 6), (4, 8), (4, 8, 3), (2,), "draft ids"),
        ((4, 7), (4, 7), (4, 8, 3), (2,), "target sampled ids"),
        ((4, 7), (4, 8), (4, 7, 3), (2,), "target hidden"),
        ((4, 7), (4, 8), (4, 8, 3), (1,), "start_positions"),
    ),
)
def test_verify_target_step_rejects_invalid_shapes(
    draft_shape,
    sample_shape,
    hidden_shape,
    starts_shape,
    error,
):
    with pytest.raises(ValueError, match=error):
        _verify_target_step(
            torch.zeros(draft_shape, dtype=torch.int64),
            torch.zeros(sample_shape, dtype=torch.int64),
            torch.zeros(hidden_shape),
            torch.zeros(starts_shape, dtype=torch.int32),
            tp_size=2,
        )
