# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Serving-facing DeepSeek-V4 MTP input/state helpers.

The functions in this file intentionally stay on the CPU/PyTorch side. They
model the packed-token bookkeeping needed around the device MTP kernels without
mixing that bookkeeping into kernel math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    TensorLike = torch.Tensor
else:
    TensorLike = Any


@dataclass
class MTPState:
    spec_tokens: TensorLike
    accepted_num: TensorLike
    kv_len: TensorLike
    is_prefill: bool
    step_index: int = 0


@dataclass
class MTPDecodeInputs:
    input_ids: TensorLike
    position_ids: TensorLike
    kv_len: TensorLike


def _torch() -> Any:
    import torch

    return torch


def _as_1d_i64(value: TensorLike) -> TensorLike:
    torch = _torch()
    if not torch.is_tensor(value):
        value = torch.tensor(value, dtype=torch.int64)
    return value.reshape(-1).to(dtype=torch.int64)


def _as_i64(value: TensorLike) -> TensorLike:
    torch = _torch()
    if not torch.is_tensor(value):
        value = torch.tensor(value, dtype=torch.int64)
    return value.to(dtype=torch.int64)


def _pad_2d_tokens(tokens: TensorLike, width: int, pad_value: int = 0) -> TensorLike:
    torch = _torch()
    if tokens.dim() == 1:
        tokens = tokens.reshape(-1, 1)
    if tokens.shape[1] >= width:
        return tokens[:, :width].clone()
    pad = torch.full(
        [tokens.shape[0], width - tokens.shape[1]],
        pad_value,
        dtype=tokens.dtype,
        device=tokens.device,
    )
    return torch.cat([tokens, pad], dim=1)


def build_mtp_prefill_input_ids(input_ids: TensorLike, seq_lens: TensorLike, next_tokens: TensorLike) -> TensorLike:
    """Shift each packed prompt left and put the sampled token at its tail."""

    torch = _torch()
    input_ids = _as_1d_i64(input_ids)
    seq_lens = _as_1d_i64(seq_lens)
    next_tokens = _as_1d_i64(next_tokens).to(device=input_ids.device)

    if int(seq_lens.sum().item()) != input_ids.numel():
        raise ValueError("sum(seq_lens) must match packed input_ids length")
    if next_tokens.numel() != seq_lens.numel():
        raise ValueError("next_tokens must have one token per request")

    shifted = torch.empty_like(input_ids)
    offset = 0
    for request_idx, seq_len_t in enumerate(seq_lens.tolist()):
        seq_len = int(seq_len_t)
        if seq_len <= 0:
            raise ValueError(f"seq_lens[{request_idx}] must be positive")
        end = offset + seq_len
        if seq_len > 1:
            shifted[offset:end - 1] = input_ids[offset + 1:end]
        shifted[end - 1] = next_tokens[request_idx]
        offset = end
    return shifted


def restore_cp_prefill_next_tokens(
    local_next_tokens: TensorLike,
    output_request_indices: TensorLike,
    global_request_count: int,
) -> TensorLike:
    """Restore partition-local next tokens into global request order."""

    torch = _torch()
    local_next_tokens = _as_1d_i64(local_next_tokens)
    output_request_indices = _as_1d_i64(output_request_indices).to(device=local_next_tokens.device)
    if local_next_tokens.numel() != output_request_indices.numel():
        raise ValueError("local_next_tokens and output_request_indices must have the same length")

    restored = torch.zeros(
        [int(global_request_count)],
        dtype=local_next_tokens.dtype,
        device=local_next_tokens.device,
    )
    restored[output_request_indices] = local_next_tokens
    return restored


def build_mtp_prefill_input_ids_cp(
    input_ids: TensorLike,
    seq_lens: TensorLike,
    valid_token_indices: TensorLike,
    padded_token_count: int,
    restored_next_tokens: TensorLike,
) -> TensorLike:
    """Build partition-prefill MTP ids and place them back into padded global order."""

    torch = _torch()
    input_ids = _as_1d_i64(input_ids)
    valid_token_indices = _as_1d_i64(valid_token_indices).to(device=input_ids.device)
    restored_next_tokens = _as_1d_i64(restored_next_tokens).to(device=input_ids.device)

    shifted = build_mtp_prefill_input_ids(input_ids, seq_lens, restored_next_tokens)
    out = torch.zeros(
        [int(padded_token_count)],
        dtype=shifted.dtype,
        device=shifted.device,
    )
    if valid_token_indices.numel() != shifted.numel():
        raise ValueError("valid_token_indices must match shifted packed token length")
    out[valid_token_indices] = shifted
    return out


def build_main_decode_input_ids_for_mtp(
    current_tokens: TensorLike,
    spec_tokens: TensorLike,
    accepted_num: TensorLike,
    kv_len: TensorLike,
    next_n: int,
    pad_to: int | None = None,
) -> MTPDecodeInputs:
    """Build the main-model decode window used before MTP proposes new tokens.

    ``current_tokens`` is expected to have already been selected by the serving
    scheduler from ``accepted_num``. This helper validates the accepted counts
    and then builds the model input window from that selected token plus the
    previous speculative-token buffer.
    """

    torch = _torch()
    current_tokens = _as_1d_i64(current_tokens)
    spec_tokens = _as_1d_i64(spec_tokens).reshape(current_tokens.numel(), -1).to(device=current_tokens.device)
    kv_len = _as_1d_i64(kv_len).to(device=current_tokens.device)
    accepted_num = _as_1d_i64(accepted_num).to(device=current_tokens.device)

    batch = current_tokens.numel()
    if accepted_num.numel() != batch:
        raise ValueError("accepted_num must have one value per request")
    if bool((accepted_num < 0).any()):
        raise ValueError("accepted_num must be non-negative")
    if bool((accepted_num > spec_tokens.shape[1]).any()):
        raise ValueError("accepted_num cannot exceed speculative token width")
    q_len = int(next_n) + 1
    windows = torch.empty([batch, q_len], dtype=current_tokens.dtype, device=current_tokens.device)
    for b in range(batch):
        available = torch.cat([current_tokens[b:b + 1], spec_tokens[b]], dim=0)
        if available.numel() < q_len:
            pad_token = available[-1] if available.numel() > 0 else current_tokens[b]
            pad = torch.full(
                [q_len - available.numel()],
                int(pad_token.item()),
                dtype=current_tokens.dtype,
                device=current_tokens.device,
            )
            available = torch.cat([available, pad], dim=0)
        windows[b] = available[-q_len:]

    offsets = torch.arange(q_len - 1, -1, -1, dtype=torch.int64, device=current_tokens.device)
    position_ids = (kv_len.reshape(batch, 1) - offsets.reshape(1, q_len)).reshape(-1).to(dtype=torch.int32)
    input_ids = windows.reshape(-1)
    if pad_to is not None and input_ids.numel() < int(pad_to):
        pad = torch.zeros([int(pad_to) - input_ids.numel()], dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([input_ids, pad], dim=0)
        pos_pad = torch.zeros(
            [int(pad_to) - position_ids.numel()],
            dtype=position_ids.dtype,
            device=position_ids.device,
        )
        position_ids = torch.cat([position_ids, pos_pad], dim=0)
    return MTPDecodeInputs(input_ids=input_ids, position_ids=position_ids, kv_len=kv_len + q_len)


def build_mtp_decode_input_ids(main_next_tokens: TensorLike) -> TensorLike:
    """The first MTP decode input is the newly sampled main-model token."""

    return _as_1d_i64(main_next_tokens).reshape(-1).clone()


def verify_mtp_spec_tokens(
    spec_tokens: TensorLike,
    main_next_tokens: TensorLike,
    is_prefill: bool = False,
) -> TensorLike:
    """Return how many speculative tokens are accepted per request."""

    torch = _torch()
    spec_tokens = _as_i64(spec_tokens)
    main_next_tokens = _as_i64(main_next_tokens).to(device=spec_tokens.device)
    if spec_tokens.dim() != 2:
        raise ValueError("spec_tokens must be a 2-D tensor with shape [batch, next_n]")
    batch = spec_tokens.shape[0]
    if is_prefill:
        return torch.zeros([batch], dtype=torch.int32, device=spec_tokens.device)

    if main_next_tokens.dim() == 1:
        if main_next_tokens.numel() % batch != 0:
            raise ValueError("flattened main_next_tokens length must be divisible by batch")
        main_next_tokens = main_next_tokens.reshape(batch, -1)
    elif main_next_tokens.shape[0] != batch:
        raise ValueError("main_next_tokens batch dimension must match spec_tokens")
    else:
        main_next_tokens = main_next_tokens.reshape(batch, -1)
    compare_n = min(spec_tokens.shape[1], main_next_tokens.shape[1])
    accepted = torch.zeros([batch], dtype=torch.int32, device=spec_tokens.device)
    for b in range(batch):
        count = 0
        for s in range(compare_n):
            if int(spec_tokens[b, s].item()) != int(main_next_tokens[b, s].item()):
                break
            count += 1
        accepted[b] = count
    return accepted


def update_mtp_state_after_step(
    state: MTPState,
    logits: TensorLike | None = None,
    sampled_tokens: TensorLike | None = None,
    output_request_indices: TensorLike | None = None,
    final_token_indices: TensorLike | None = None,
    next_n: int = 1,
    q_len: int = 1,
) -> MTPState:
    """Update MTP speculative-token state after one prefill/decode step."""

    torch = _torch()
    if state.is_prefill:
        if sampled_tokens is None:
            if logits is None:
                raise ValueError("prefill update requires logits or sampled_tokens")
            sampled_tokens = torch.argmax(logits, dim=-1)
        sampled_tokens = _as_i64(sampled_tokens)
        if output_request_indices is not None:
            output_request_indices = _as_1d_i64(output_request_indices).to(device=sampled_tokens.device)
            sampled_tokens = sampled_tokens[output_request_indices]
        if final_token_indices is not None:
            final_token_indices = _as_1d_i64(final_token_indices).to(device=sampled_tokens.device)
            final_tokens = sampled_tokens.reshape(-1)[final_token_indices].reshape(-1, 1)
        elif sampled_tokens.dim() == 2:
            final_tokens = sampled_tokens[:, -1:].reshape(-1, 1)
        else:
            final_tokens = sampled_tokens.reshape(-1, 1)
        spec_tokens = _pad_2d_tokens(final_tokens, int(next_n), pad_value=0)
        accepted_num = torch.zeros([spec_tokens.shape[0]], dtype=torch.int32, device=spec_tokens.device)
        kv_len = _as_1d_i64(state.kv_len).to(device=spec_tokens.device) + int(q_len) - 1 + int(next_n) - 1
        return MTPState(
            spec_tokens=spec_tokens,
            accepted_num=accepted_num,
            kv_len=kv_len,
            is_prefill=False,
            step_index=0,
        )

    if sampled_tokens is None:
        if logits is None:
            raise ValueError("decode update requires logits or sampled_tokens")
        sampled_tokens = torch.argmax(logits, dim=-1)
    spec_tokens = _as_i64(state.spec_tokens)
    if spec_tokens.dim() != 2:
        raise ValueError("state.spec_tokens must be a 2-D tensor with shape [batch, next_n]")
    spec_tokens = spec_tokens.clone()
    batch, width = spec_tokens.shape
    sampled_tokens = _as_i64(sampled_tokens).to(device=spec_tokens.device)
    accepted_num = _as_1d_i64(state.accepted_num).to(device=spec_tokens.device)
    if accepted_num.numel() != batch:
        raise ValueError("accepted_num must have one value per request")
    if bool((accepted_num < 0).any()):
        raise ValueError("accepted_num must be non-negative")
    if bool((accepted_num > width).any()):
        raise ValueError("accepted_num cannot exceed speculative token width")

    if sampled_tokens.numel() == batch:
        selected_tokens = sampled_tokens.reshape(batch)
    elif sampled_tokens.numel() % batch == 0:
        sampled_view = sampled_tokens.reshape(batch, -1)
        rows = []
        for b in range(batch):
            row = min(int(accepted_num[b].item()), sampled_view.shape[1] - 1)
            rows.append(sampled_view[b, row])
        selected_tokens = torch.stack(rows)
    else:
        raise ValueError("decode sampled/logit token count must be batch or a batch-multiple")

    for b in range(batch):
        accepted = int(accepted_num[b].item())
        if accepted >= width:
            spec_tokens[b] = torch.zeros_like(spec_tokens[b])
            spec_tokens[b, 0] = selected_tokens[b]
        else:
            spec_tokens[b, accepted] = selected_tokens[b]
            if accepted + 1 < width:
                spec_tokens[b, accepted + 1:] = 0
    return MTPState(
        spec_tokens=spec_tokens,
        accepted_num=accepted_num,
        kv_len=_as_1d_i64(state.kv_len).to(device=spec_tokens.device) + 1,
        is_prefill=False,
        step_index=int(state.step_index) + 1,
    )


def validate_mtp_input_full_chain(candidate_logits: TensorLike | None = None) -> None:
    """Exercise the serving-side MTP input/state helpers as one CPU chain."""

    torch = _torch()

    input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    seq_lens = torch.tensor([3, 2], dtype=torch.int64)
    next_tokens = torch.tensor([9, 8], dtype=torch.int64)
    shifted = build_mtp_prefill_input_ids(input_ids, seq_lens, next_tokens)
    assert shifted.tolist() == [2, 3, 9, 5, 8]

    restored = restore_cp_prefill_next_tokens(
        torch.tensor([8, 9], dtype=torch.int64),
        torch.tensor([1, 0], dtype=torch.int64),
        2,
    )
    assert restored.tolist() == [9, 8]

    cp_shifted = build_mtp_prefill_input_ids_cp(
        input_ids,
        seq_lens,
        torch.arange(5, dtype=torch.int64),
        6,
        restored,
    )
    assert cp_shifted.tolist() == [2, 3, 9, 5, 8, 0]

    state = MTPState(
        spec_tokens=torch.zeros([2, 2], dtype=torch.int64),
        accepted_num=torch.zeros([2], dtype=torch.int32),
        kv_len=torch.tensor([5, 6], dtype=torch.int64),
        is_prefill=True,
    )
    state = update_mtp_state_after_step(
        state,
        sampled_tokens=torch.tensor([[1, 2, 33], [4, 5, 44]], dtype=torch.int64),
        next_n=2,
        q_len=3,
    )
    assert state.spec_tokens.tolist() == [[33, 0], [44, 0]]
    assert state.accepted_num.tolist() == [0, 0]
    assert state.kv_len.tolist() == [8, 9]
    assert not state.is_prefill

    decode_inputs = build_main_decode_input_ids_for_mtp(
        torch.tensor([10, 20], dtype=torch.int64),
        torch.tensor([[11, 12], [21, 22]], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([7, 9], dtype=torch.int64),
        next_n=2,
    )
    assert decode_inputs.input_ids.tolist() == [10, 11, 12, 20, 21, 22]
    assert decode_inputs.position_ids.tolist() == [5, 6, 7, 7, 8, 9]
    assert decode_inputs.kv_len.tolist() == [10, 12]

    padded_decode_inputs = build_main_decode_input_ids_for_mtp(
        torch.tensor([10], dtype=torch.int64),
        torch.tensor([[11, 12]], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([7], dtype=torch.int64),
        next_n=2,
        pad_to=5,
    )
    assert padded_decode_inputs.input_ids.tolist() == [10, 11, 12, 0, 0]
    assert padded_decode_inputs.position_ids.tolist() == [5, 6, 7, 0, 0]

    full_accept_inputs = build_main_decode_input_ids_for_mtp(
        torch.tensor([30], dtype=torch.int64),
        torch.tensor([[31, 32]], dtype=torch.int64),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([12], dtype=torch.int64),
        next_n=2,
    )
    assert full_accept_inputs.input_ids.tolist() == [30, 31, 32]
    try:
        build_main_decode_input_ids_for_mtp(
            torch.tensor([30], dtype=torch.int64),
            torch.tensor([[31, 32]], dtype=torch.int64),
            torch.tensor([-1], dtype=torch.int32),
            torch.tensor([12], dtype=torch.int64),
            next_n=2,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("negative accepted_num must be rejected")

    flat_residual = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 12)
    assert flat_residual.reshape(2, 4, 3).flatten(1).equal(flat_residual)

    mtp_decode_input = build_mtp_decode_input_ids(torch.tensor([[101], [202]], dtype=torch.int64))
    assert mtp_decode_input.tolist() == [101, 202]

    accepted = verify_mtp_spec_tokens(
        torch.tensor([[11, 12], [21, 22]], dtype=torch.int64),
        torch.tensor([[11, 99, 13], [21, 22, 23]], dtype=torch.int64),
    )
    assert accepted.tolist() == [1, 2]

    state = MTPState(
        spec_tokens=torch.tensor([[33, 0], [44, 0]], dtype=torch.int64),
        accepted_num=accepted,
        kv_len=torch.tensor([8, 9], dtype=torch.int64),
        is_prefill=False,
    )
    state = update_mtp_state_after_step(
        state,
        sampled_tokens=torch.tensor([77, 88], dtype=torch.int64),
    )
    assert state.spec_tokens.tolist() == [[33, 77], [44, 88]]
    assert state.kv_len.tolist() == [9, 10]
    assert state.step_index == 1

    logits = torch.zeros([6, 10], dtype=torch.float32)
    logits[1, 7] = 1.0
    logits[5, 8] = 1.0
    logits_state = MTPState(
        spec_tokens=torch.tensor([[33, 0], [44, 0]], dtype=torch.int64),
        accepted_num=torch.tensor([1, 2], dtype=torch.int32),
        kv_len=torch.tensor([8, 9], dtype=torch.int64),
        is_prefill=False,
    )
    logits_state = update_mtp_state_after_step(logits_state, logits=logits)
    assert logits_state.spec_tokens.tolist() == [[33, 7], [8, 0]]

    stale_tail_state = MTPState(
        spec_tokens=torch.tensor([[31, 32, 33], [41, 42, 43]], dtype=torch.int64),
        accepted_num=torch.tensor([0, 1], dtype=torch.int32),
        kv_len=torch.tensor([10, 11], dtype=torch.int64),
        is_prefill=False,
    )
    stale_tail_state = update_mtp_state_after_step(
        stale_tail_state,
        sampled_tokens=torch.tensor([91, 92], dtype=torch.int64),
    )
    assert stale_tail_state.spec_tokens.tolist() == [[91, 0, 0], [41, 92, 0]]

    if candidate_logits is not None:
        final_token_indices = torch.tensor([2, 4], dtype=torch.int64, device=candidate_logits.device)
        sampled = torch.argmax(candidate_logits, dim=-1)
        logits_state = MTPState(
            spec_tokens=torch.zeros([2, 2], dtype=torch.int64, device=candidate_logits.device),
            accepted_num=torch.zeros([2], dtype=torch.int32, device=candidate_logits.device),
            kv_len=torch.tensor([5, 6], dtype=torch.int64, device=candidate_logits.device),
            is_prefill=True,
        )
        logits_state = update_mtp_state_after_step(
            logits_state,
            logits=candidate_logits,
            final_token_indices=final_token_indices,
            next_n=2,
            q_len=3,
        )
        assert logits_state.spec_tokens[:, 0].tolist() == sampled[final_token_indices].tolist()
