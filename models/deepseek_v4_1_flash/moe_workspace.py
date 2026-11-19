# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Symmetric workspace layout for the persistent V4.1 MoE kernel."""

def _pipeline_workspace_layout(ep, local_experts, tokens, topk, hidden, intermediate, shared=1, *, ring_limit=None):
    """Describe the private fused MoE communication workspace."""
    if ep not in (2, 4, 8) or local_experts <= 0 or tokens <= 0 or tokens % 16:
        raise ValueError("Invalid EP, expert count or physical token capacity")
    if hidden % 256 or intermediate % 256 or shared not in (0, 1, 2) or topk <= 0:
        raise ValueError("Invalid persistent MegaMoE shape")
    experts = ep * local_experts
    if not topk <= experts <= 384:
        raise ValueError("TopK/expert count is outside the model contract")
    result = {}
    offset = 0

    def align(value, multiple):
        return (value + multiple - 1) // multiple * multiple

    def allocate(name, count, stride, alignment=None):
        nonlocal offset
        offset = align(offset, stride if alignment is None else alignment)
        result[name] = offset
        offset += count * stride

    def fp8(name, rows, columns):
        allocate(name + ".data", rows * columns, 1, 512)
        allocate(name + ".sf", rows * ((columns + 63) // 64), 2, 512)

    max_blocks = (ep * tokens * topk + 255) // 256 + local_experts
    ring_blocks = min(max_blocks, ring_limit if ring_limit is not None else (128 if hidden // 256 <= 16 else 32))
    shared_blocks = (tokens + 255) // 256 if shared else 0
    counters = shared_blocks + max_blocks
    full_rows, ring_rows = max_blocks * 256, ring_blocks * 256
    src_indices = experts * tokens * topk
    allocate("sync.grid_sync_count", 3, 4, 512)
    allocate("sync.rank_barriers", 2, 16, 512)
    allocate("metadata.worker_expert_count", 32 * experts, 4, 512)
    allocate("metadata.total_expert_count", experts, 4)
    allocate("metadata.combine_ready_prefix", ep, 128)
    allocate("metadata.expert_recv_count", experts, 4)
    allocate("metadata.expert_recv_count_sum", local_experts, 512)
    allocate("metadata.num_m_blocks_certificate", 1, 4, 128)
    allocate("metadata.src_token_metadata", src_indices, 8, 512)
    allocate("metadata.staged_token_topk_idx", src_indices, 4, 512)
    fp8("input.acts", tokens, hidden)
    allocate("input.topk_idx", tokens * topk, 8, 512)
    allocate("input.topk_weights", tokens * topk, 4, 512)
    allocate("progress.blocks", 4 * counters, 128)
    allocate("progress.num_issued_gemms", 32, 128)
    allocate("progress.metadata_ready", 32, 128)
    result["num_progress_bytes"] = offset - result["progress.blocks"]
    fp8("routed.l1_acts", ring_rows, hidden)
    allocate("routed.topk_weights", full_rows, 4, 512)
    allocate("routed.src_metadata", full_rows, 4, 512)
    fp8("routed.l2_acts", ring_rows, intermediate)
    allocate("routed.output", ring_rows * hidden, 2, 512)
    allocate("combine_input", (topk + bool(shared)) * tokens * hidden, 2, 512)
    fp8("shared_l2_acts", align(tokens, 256) if shared else 0, shared * intermediate)
    allocate("route_m_blocks", tokens * topk, 4, 512)
    result.update(logical_bytes=offset, bytes=align(offset, 2 * 1024 * 1024),
                  max_blocks=max_blocks, ring_blocks=ring_blocks, ring_rows=ring_rows)
    return result


def fused_workspace_bytes(ep, local_experts, tokens, topk, hidden, intermediate):
    return _pipeline_workspace_layout(ep, local_experts, tokens, topk, hidden, intermediate)["bytes"]


def model_window_shapes(local_experts, topk, packet_width, route_width, signal_pad):
    """Return the receive, metadata, route, signal and combine views for an L3 entry."""
    from config import EP_SIZE, FLASH, MOE_RECV_MAX, MOE_TOKENS, USE_MEGAMOE

    if USE_MEGAMOE:
        size = fused_workspace_bytes(
            EP_SIZE, local_experts, MOE_TOKENS, topk, FLASH.hidden_size, FLASH.moe_intermediate_size,
        )
        # The shared model signature also carries the split transport arguments.
        # Give each unused writable view a distinct, minimum-sized allocation.
        return [size], [1, 1], [1, 1], [1, 1, 1], [1, 1]
    rows = local_experts * MOE_RECV_MAX
    return (
        [rows, packet_width], [EP_SIZE, local_experts], [rows, route_width],
        [EP_SIZE, local_experts, signal_pad], [MOE_TOKENS * topk, FLASH.hidden_size],
    )
