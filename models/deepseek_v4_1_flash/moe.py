# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Expert-parallel MoE dispatch, local expert compute, and routed-output combine."""

import pypto.language as pl
import pypto.language.distributed as pld
import torch
import torch.nn.functional as F

from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.config import (
    AUX_WIDTH,
    D,
    EP_SIZE,
    FLASH,
    MOE_INTER,
    MX_GROUP,
    N_EXPERTS,
    N_LOCAL_EXPERTS,
    PREFILL_MAX_TOKENS,
    RECV_MAX,
    ROUTE_WIDTH,
    T_DYN,
    TOPK,
    TP_SIZE,
)
from models.deepseek_v4_1_flash.decode_swa import make_norm, make_projection
from models.deepseek_v4_1_flash.golden import gate, rms_norm
from models.deepseek_v4_1_flash.quantization import dequantize_mxfp4
from models.deepseek_v4_1_flash.quantization import dequantize_mxfp8
from models.deepseek_v4_1_flash.quantization import unpack_mx_b_scale
from models.deepseek_v4_1_flash.quantization import quantize_mxfp8_activation


def _golden_expert(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    route_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    quantized = quantize_mxfp8_activation(x)
    gate_value = F.linear(quantized, w1.float()).to(x.dtype).float().clamp(max=FLASH.swiglu_limit)
    up_value = F.linear(quantized, w3.float()).to(x.dtype).float()
    hidden = F.silu(gate_value) * up_value.clamp(-FLASH.swiglu_limit, FLASH.swiglu_limit)
    if route_weight is not None:
        hidden = hidden * route_weight.unsqueeze(-1)
    hidden = quantize_mxfp8_activation(hidden.to(x.dtype))
    return F.linear(hidden, w2.float()).to(x.dtype).float()


def tp_token_owners(num_tokens: int, tp_size: int, device: torch.device | None = None) -> torch.Tensor:
    """Assign each replicated TP token row to one rank before EP dispatch."""
    if tp_size <= 0:
        raise ValueError("tp_size must be positive")
    return torch.arange(num_tokens, device=device, dtype=torch.int32).remainder(tp_size)


def golden_moe(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    routed_w1: torch.Tensor,
    routed_w1_scale: torch.Tensor,
    routed_w2: torch.Tensor,
    routed_w2_scale: torch.Tensor,
    routed_w3: torch.Tensor,
    routed_w3_scale: torch.Tensor,
    shared_w1: torch.Tensor,
    shared_w1_scale: torch.Tensor,
    shared_w2: torch.Tensor,
    shared_w2_scale: torch.Tensor,
    shared_w3: torch.Tensor,
    shared_w3_scale: torch.Tensor,
    token_owners: torch.Tensor | None = None,
    tp_size: int = 1,
    num_tokens: int | None = None,
) -> torch.Tensor:
    """Evaluate the gathered MoE result and validate unique TP row ownership."""
    shape = x.shape
    normalized = rms_norm(x.reshape(-1, shape[-1]), norm_weight)
    active_tokens = (
        normalized.shape[0] if num_tokens is None else min(max(num_tokens, 0), normalized.shape[0])
    )
    if token_owners is not None:
        if token_owners.shape != (normalized.shape[0],):
            raise ValueError("token_owners must contain one TP owner per input row")
        expected = tp_token_owners(normalized.shape[0], tp_size, token_owners.device)
        if not torch.equal(token_owners.to(torch.int32), expected):
            raise ValueError("token_owners must assign every replicated row to exactly one TP rank")
    normalized = normalized[:active_tokens]
    shared_w1 = dequantize_mxfp8(shared_w1, unpack_mx_b_scale(shared_w1_scale)).transpose(-2, -1)
    shared_w2 = dequantize_mxfp8(shared_w2, unpack_mx_b_scale(shared_w2_scale)).transpose(-2, -1)
    shared_w3 = dequantize_mxfp8(shared_w3, unpack_mx_b_scale(shared_w3_scale)).transpose(-2, -1)
    route_weights, expert_indices = gate(normalized, gate_weight, correction_bias)
    output = torch.zeros_like(normalized, dtype=torch.float32)
    for expert_id in range(routed_w1.shape[0]):
        token_rows, route_columns = torch.where(expert_indices == expert_id)
        if token_rows.numel() == 0:
            continue
        routed = _golden_expert(
            normalized[token_rows],
            dequantize_mxfp4(routed_w1[expert_id], routed_w1_scale[expert_id]),
            dequantize_mxfp4(routed_w2[expert_id], routed_w2_scale[expert_id]),
            dequantize_mxfp4(routed_w3[expert_id], routed_w3_scale[expert_id]),
            route_weights[token_rows, route_columns],
        )
        output[token_rows] += routed
    output += _golden_expert(normalized, shared_w1, shared_w2, shared_w3)
    result = x.reshape(-1, shape[-1]).clone()
    result[:active_tokens] = output.to(x.dtype)
    return result.reshape(shape)


# A token can select an expert once, and only one TP rank dispatches that token.
SOURCE_CAPACITY = C.PREFILL_MAX_TOKENS // C.TP_SIZE
EXPERT_TILE = 32
PROJECTION_TILE = 32
K_TILE = 256
SOURCE_TILES = (SOURCE_CAPACITY + EXPERT_TILE - 1) // EXPERT_TILE
SWIGLU_LIMIT = FLASH.swiglu_limit
GATE_TEMPERATURE = FLASH.gate_temperature
ROUTED_SCALING_FACTOR = FLASH.routed_scaling_factor
normalize = make_norm(D)
shared_up = make_projection(D, MOE_INTER)
shared_down = make_projection(MOE_INTER, D)


def make_routed_projection(width, output_width):
    """Consume checkpoint FP4 tiles directly, without expanded expert weights in HBM."""

    @pl.jit.inline
    def project(
        x: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[output_width, width], pl.FP4],
        scale: pl.Tensor[[output_width, width // MX_GROUP], pl.FP8E8M0],
        output: pl.Tensor[[T_DYN, output_width], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        # Keep this view at the leaf: the JIT infers call-site metadata for
        # packed carriers but does not propagate reinterpret_view aliases.
        # An already-logical FP4 input has the same shape and byte extent.
        logical_weight = pl.reinterpret_view(weight, pl.FP4, shape=[output_width, width])
        for block in pl.spmd(output_width // PROJECTION_TILE, name_hint="moe_routed_projection"):
            n0 = block * PROJECTION_TILE
            accumulator = pl.create_tensor([EXPERT_TILE, PROJECTION_TILE], dtype=pl.FP32)
            for kb in pl.range(width // K_TILE):
                k0 = kb * K_TILE
                source = pl.load(x, [0, k0], [EXPERT_TILE, K_TILE], valid_shape=[num_tokens, K_TILE])
                source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), EXPERT_TILE, K_TILE)
                grouped = pl.reshape(pl.cast(source, pl.FP32), [EXPERT_TILE * (K_TILE // 32), 32])
                maximum = pl.maximum(pl.row_max(pl.abs(grouped)), 1e-4)
                bits = pl.reinterpret_view(pl.mul(maximum, 1.0 / 448.0), pl.INT32)
                exponent = pl.shrs(pl.add(bits, 8388607), 23)
                factor = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
                quantized = pl.cast(pl.row_expand_div(grouped, factor), pl.FP8E4M3FN, mode="rint")
                source_fp32 = pl.reshape(
                    pl.row_expand_mul(pl.cast(quantized, pl.FP32), factor), [EXPERT_TILE, K_TILE]
                )
                payload = pl.load(logical_weight, [n0, k0], [PROJECTION_TILE, K_TILE])
                # A5 TCVT supports FP4 -> BF16 directly; all E2M1 values are exact.
                values = pl.cast(pl.cast(payload, pl.BF16), pl.FP32)
                scale_tile = pl.load(scale, [n0, k0 // 32], [PROJECTION_TILE, K_TILE // 32])
                scale_bits = pl.ands(pl.cast(pl.reinterpret_view(scale_tile, pl.INT8), pl.INT32), 255)
                scale_value = pl.reinterpret_view(pl.maximum(pl.shls(scale_bits, 23), 4194304), pl.FP32)
                values = pl.reshape(values, [PROJECTION_TILE * (K_TILE // 32), 32])
                weights = pl.reshape(
                    pl.row_expand_mul(values, pl.reshape(scale_value, [PROJECTION_TILE * (K_TILE // 32), 1])),
                    [PROJECTION_TILE, K_TILE],
                )
                accumulator = pl.matmul_acc(
                    accumulator, source_fp32, weights, b_trans=True, init_cond=(kb == 0)
                )
            output = pl.store(
                pl.set_validshape(pl.cast(accumulator, pl.BF16, mode="rint"), num_tokens, PROJECTION_TILE),
                [0, n0], output,
            )
        return output

    return project


routed_up = make_routed_projection(D, MOE_INTER)
routed_down = make_routed_projection(MOE_INTER, D)


@pl.jit.inline
def swiglu(
    gate_value: pl.Tensor[[T_DYN, MOE_INTER], pl.BF16],
    up_value: pl.Tensor[[T_DYN, MOE_INTER], pl.BF16],
    weights: pl.Tensor[[T_DYN, AUX_WIDTH], pl.FP32],
    hidden: pl.Tensor[[T_DYN, MOE_INTER], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    for token in pl.spmd(num_tokens, name_hint="moe_swiglu"):
        gate_row = pl.minimum(pl.cast(gate_value[token:token + 1, :], pl.FP32), SWIGLU_LIMIT)
        up_row = pl.minimum(pl.maximum(pl.cast(up_value[token:token + 1, :], pl.FP32), -SWIGLU_LIMIT), SWIGLU_LIMIT)
        value = pl.mul(pl.mul(gate_row, pl.recip(pl.add(pl.exp(pl.neg(gate_row)), 1.0))), up_row)
        value = pl.mul(value, pl.read(weights, [token, 0]))
        hidden[token:token + 1, :] = pl.cast(value, pl.BF16, mode="rint")
    return hidden


@pl.jit.inline
def route(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    indices: pl.Tensor[[T_DYN, TOPK], pl.INT32],
    weights: pl.Tensor[[T_DYN * TOPK, AUX_WIDTH], pl.FP32],
    routes: pl.Tensor[[T_DYN * TOPK, ROUTE_WIDTH], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(x, 0)
    scores = pl.create_tensor([tokens, N_EXPERTS], dtype=pl.FP32)
    for block in pl.spmd((num_tokens + 15) // 16 * (N_EXPERTS // 32), name_hint="moe_gate"):
        token = block // (N_EXPERTS // 32) * 16
        column = block % (N_EXPERTS // 32) * 32
        rows = pl.min(16, num_tokens - token)
        logits = pl.create_tensor([16, 32], dtype=pl.FP32)
        for kb in pl.range(D // K_TILE):
            source = pl.load(x, [token, kb * K_TILE], [16, K_TILE], valid_shape=[rows, K_TILE])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 16, K_TILE)
            matrix = pl.load(gate_weight, [column, kb * K_TILE], [32, K_TILE])
            logits = pl.matmul_acc(logits, pl.cast(source, pl.FP32), matrix, b_trans=True, init_cond=(kb == 0))
        logits = pl.mul(logits, 1.0 / GATE_TEMPERATURE)
        softplus = pl.add(pl.maximum(logits, 0.0), pl.log(pl.add(pl.exp(pl.neg(pl.abs(logits))), 1.0)))
        softplus = pl.maximum(softplus, pl.exp(pl.minimum(logits, -20.0)))
        scores = pl.store(pl.set_validshape(pl.sqrt(softplus), rows, 32), [token, column], scores)
    # Eight six-index rows occupy three complete 64-byte scalar-store lines.
    for block in pl.spmd((num_tokens + 7) // 8, name_hint="moe_topk"):
        for token in pl.range(block * 8, pl.min((block + 1) * 8, num_tokens)):
            row = scores[token:token + 1, :]
            bias = pl.reshape(correction_bias[:], [1, N_EXPERTS])
            expert_ids = pl.arange(0, [1, N_EXPERTS], dtype=pl.UINT32)
            sorted32 = pl.sort32(pl.add(row, bias), expert_ids)
            sorted128 = pl.mrgsort(sorted32, block_len=64)
            sorted_all = pl.mrgsort(sorted128[:, 0:256], sorted128[:, 256:512], sorted128[:, 512:768])
            selected = pl.gather(sorted_all[:, :16], mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            selected_scores = pl.gather(row, index=selected)
            valid_scores = pl.fillpad(pl.set_validshape(selected_scores, 1, TOPK), pad_value=pl.PadValue.zero)
            denominator = pl.add(pl.row_sum(valid_scores), 1e-20)
            normalized = pl.mul(pl.row_expand_div(valid_scores, denominator), ROUTED_SCALING_FACTOR)
            for k in pl.range(TOPK):
                pl.write(indices, [token, k], pl.read(selected, [0, k]))
                route_id = token * TOPK + k
                weight_row = pl.full([1, AUX_WIDTH], dtype=pl.FP32, value=0.0)
                route_row = pl.full([1, ROUTE_WIDTH], dtype=pl.INT32, value=0)
                pl.write(weight_row, [0, 0], pl.read(normalized, [0, k]))
                pl.write(route_row, [0, 0], pl.cast(route_id, pl.INT32))
                weights = pl.store(weight_row, [route_id, 0], weights)
                routes = pl.store(route_row, [route_id, 0], routes)
    return indices, weights, routes


@pl.jit.inline
def quantize_dispatch(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    payload: pl.Tensor[[T_DYN, D], pl.FP8E4M3FN],
    scale: pl.Tensor[[T_DYN, D // 32], pl.UINT8],
    num_tokens: pl.Scalar[pl.INT32],
):
    for token in pl.spmd(num_tokens, name_hint="moe_dispatch_quantize"):
        grouped = pl.reshape(pl.cast(x[token:token + 1, :], pl.FP32), [D // 32, 32])
        maximum = pl.maximum(pl.row_max(pl.abs(grouped)), 1e-4)
        bits = pl.reinterpret_view(pl.mul(maximum, 1.0 / 448.0), pl.INT32)
        exponent = pl.shrs(pl.add(bits, 8388607), 23)
        factor = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
        values = pl.cast(pl.row_expand_div(grouped, factor), pl.FP8E4M3FN, mode="rint")
        payload[token:token + 1, :] = pl.reshape(values, [1, D])
        signed = pl.sub(exponent, pl.mul(pl.shrs(exponent, 7), 256))
        codes = pl.reinterpret_view(pl.cast(signed, pl.INT8), pl.UINT8)
        scale[token:token + 1, :] = pl.reshape(codes, [1, D // 32])
    return payload, scale


@pl.jit.inline(auto_scope=False)
def moe(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.FP4],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D // MX_GROUP], pl.FP8E8M0],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.FP4],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER // MX_GROUP], pl.FP8E8M0],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.FP4],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D // MX_GROUP], pl.FP8E8M0],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    token_owners: pl.Tensor[[T_DYN], pl.INT32],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.FP8E4M3FN],
    recv_scale: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], pl.UINT8],
    recv_weights: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32],
    recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    routed_output: pld.DistributedTensor[[T_DYN * TOPK, D], pl.FP32],
    combine_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    ep_rank: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Run one EP exchange for replicated TP inputs.

    ``group_base`` is the global rank of the first member of the EP group;
    ``ep_rank`` is local to that group and ``tp_rank == ep_rank % TP_SIZE``.
    TP groups are contiguous. Every rank in a TP group supplies identical rows
    and ``token_owners[t] == t % TP_SIZE``. Active rows must fit the configured
    prefill capacity. ``moe_epoch`` starts at one and increases on every reuse
    of these windows, including calls with zero active rows.
    """
    tokens = pl.tensor.dim(x, 0)
    active = pl.max(0, pl.min(num_tokens, tokens))
    route_rows = tokens * TOPK
    x_norm = pl.create_tensor([tokens, D], dtype=pl.BF16)
    x_quant = pl.create_tensor([tokens, D], dtype=pl.FP8E4M3FN)
    x_scale = pl.create_tensor([tokens, D // 32], dtype=pl.UINT8)
    indices = pl.create_tensor([tokens, TOPK], dtype=pl.INT32)
    weights = pl.create_tensor([route_rows, AUX_WIDTH], dtype=pl.FP32)
    routes = pl.create_tensor([route_rows, ROUTE_WIDTH], dtype=pl.INT32)
    x_norm = normalize(x, norm_weight, x_norm, active)
    indices, weights, routes = route(x_norm, gate_weight, correction_bias, indices, weights, routes, active)
    x_quant, x_scale = quantize_dispatch(x_norm, x_quant, x_scale, active)

    # The second arrival from each peer acknowledges consumption of its result.
    # No rank may reuse either payload window until all peers have consumed it.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_reuse_wait") as reuse_tid:
        if moe_epoch > 1:
            for peer in pl.range(EP_SIZE):
                if peer != ep_rank:
                    pld.system.defer_wait(
                        signal=combine_arrived, offsets=[peer, 0],
                        expected=pl.cast(2 * (moe_epoch - 1), pl.INT32), cmp=pld.WaitCmp.Ge,
                    )

    counts = pl.create_tensor([EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
    with pl.spmd(EP_SIZE, name_hint="moe_dispatch", deps=[reuse_tid]) as dispatch_tid:
        destination = pl.tile.get_block_idx()
        cursor = pl.array.create(N_LOCAL_EXPERTS, pl.INT32)
        for expert in pl.range(N_LOCAL_EXPERTS):
            cursor[expert] = 0
        for token in pl.range(active):
            if pl.read(token_owners, [token]) == tp_rank:
                for k in pl.range(TOPK):
                    expert_id = pl.read(indices, [token, k])
                    if expert_id // N_LOCAL_EXPERTS == destination:
                        local_expert = expert_id % N_LOCAL_EXPERTS
                        row = local_expert * RECV_MAX + ep_rank * SOURCE_CAPACITY + cursor[local_expert]
                        cursor[local_expert] = cursor[local_expert] + 1
                        peer = group_base + destination
                        route_id = token * TOPK + k
                        pld.tensor.put(dst=recv_x, peer=peer, src=x_quant,
                                       dst_offsets=[row, 0], src_offsets=[token, 0], shape=[1, D])
                        pld.tensor.put(dst=recv_scale, peer=peer, src=x_scale,
                                       dst_offsets=[row, 0], src_offsets=[token, 0], shape=[1, D // 32])
                        pld.tensor.put(dst=recv_weights, peer=peer, src=weights,
                                       dst_offsets=[row, 0], src_offsets=[route_id, 0], shape=[1, AUX_WIDTH])
                        pld.tensor.put(dst=recv_routes, peer=peer, src=routes,
                                       dst_offsets=[row, 0], src_offsets=[route_id, 0], shape=[1, ROUTE_WIDTH])
        count_row = pl.full([1, N_LOCAL_EXPERTS], dtype=pl.INT32, value=0)
        for expert in pl.range(N_LOCAL_EXPERTS):
            pl.write(count_row, [0, expert], cursor[expert])
        counts = pl.store(count_row, [destination, 0], counts)
        pld.tensor.put(dst=recv_meta, peer=group_base + destination, src=counts,
                       dst_offsets=[ep_rank, 0], src_offsets=[destination, 0], shape=[1, N_LOCAL_EXPERTS])
        if destination != ep_rank:
            pld.system.notify(target=arrived, peer=group_base + destination, offsets=[ep_rank, 0],
                              value=1, op=pld.NotifyOp.AtomicAdd)
            pld.system.notify(target=data_arrived, peer=group_base + destination, offsets=[ep_rank, 0],
                              value=1, op=pld.NotifyOp.AtomicAdd)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_dispatch_wait", deps=[dispatch_tid]) as wait_tid:
        for peer in pl.range(EP_SIZE):
            if peer != ep_rank:
                pld.system.defer_wait(signal=arrived, offsets=[peer, 0], expected=moe_epoch, cmp=pld.WaitCmp.Ge)
                pld.system.defer_wait(signal=data_arrived, offsets=[peer, 0], expected=moe_epoch, cmp=pld.WaitCmp.Ge)

    recv_counts = pl.create_tensor([EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_received_counts", deps=[wait_tid]):
        recv_counts[:, :] = recv_meta[:, :]

    # Only a 32-row expert workspace is materialized at a time. The checkpoint
    # stays FP4; the projection decodes one matrix tile on device.
    expert_tids = pl.array.create(N_LOCAL_EXPERTS, pl.TASK_ID)
    for expert in pl.range(N_LOCAL_EXPERTS):
        source_tids = pl.array.create(EP_SIZE, pl.TASK_ID)
        for source in pl.range(EP_SIZE):
            source_rows = pl.cast(pl.read(recv_counts, [source, expert]), pl.INDEX)
            tile_tids = pl.array.create(SOURCE_TILES, pl.TASK_ID)
            for chunk in pl.range((source_rows + EXPERT_TILE - 1) // EXPERT_TILE):
                rows = pl.min(EXPERT_TILE, source_rows - chunk * EXPERT_TILE)
                row_base = expert * RECV_MAX + source * SOURCE_CAPACITY + chunk * EXPERT_TILE
                with pl.scope():
                    expert_x = pl.create_tensor([EXPERT_TILE, D], dtype=pl.BF16)
                    expert_weights = pl.create_tensor([EXPERT_TILE, AUX_WIDTH], dtype=pl.FP32)
                    gate_value = pl.create_tensor([EXPERT_TILE, MOE_INTER], dtype=pl.BF16)
                    up_value = pl.create_tensor([EXPERT_TILE, MOE_INTER], dtype=pl.BF16)
                    hidden = pl.create_tensor([EXPERT_TILE, MOE_INTER], dtype=pl.BF16)
                    result = pl.create_tensor([EXPERT_TILE, D], dtype=pl.BF16)
                    send_result = pl.create_tensor([EXPERT_TILE, D], dtype=pl.FP32)
                    with pl.spmd(rows, name_hint="moe_received_input", deps=[wait_tid]) as _received_tid:
                        row = pl.tile.get_block_idx()
                        value = pl.reshape(pl.cast(recv_x[row_base + row:row_base + row + 1, :], pl.FP32), [D // 32, 32])
                        codes = pl.ands(pl.cast(pl.reinterpret_view(recv_scale[row_base + row:row_base + row + 1, :], pl.INT8), pl.INT32), 255)
                        factor = pl.reshape(pl.reinterpret_view(pl.maximum(pl.shls(codes, 23), 4194304), pl.FP32), [D // 32, 1])
                        expert_x[row:row + 1, :] = pl.reshape(pl.cast(pl.row_expand_mul(value, factor), pl.BF16, mode="rint"), [1, D])
                        expert_weights[row:row + 1, :] = recv_weights[row_base + row:row_base + row + 1, :]
                    gate_value = routed_up(expert_x, routed_w1[expert, :, :], routed_w1_scale[expert, :, :], gate_value, rows)
                    up_value = routed_up(expert_x, routed_w3[expert, :, :], routed_w3_scale[expert, :, :], up_value, rows)
                    hidden = swiglu(gate_value, up_value, expert_weights, hidden, rows)
                    result = routed_down(hidden, routed_w2[expert, :, :], routed_w2_scale[expert, :, :], result, rows)
                    for row in pl.spmd(rows, name_hint="moe_result_cast"):
                        send_result[row:row + 1, :] = pl.cast(result[row:row + 1, :], pl.FP32)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_expert_scatter") as scatter_tid:
                        source_base = source // TP_SIZE * TP_SIZE
                        for row in pl.range(rows):
                            route_id = pl.read(recv_routes, [row_base + row, 0])
                            for replica in pl.range(TP_SIZE):
                                pld.tensor.put(dst=routed_output, peer=group_base + source_base + replica,
                                               src=send_result, dst_offsets=[route_id, 0], src_offsets=[row, 0],
                                               shape=[1, D])
                    tile_tids[chunk] = scatter_tid
            source_tids[source] = pl.system.task_dummy(deps=[tile_tids])
        expert_tids[expert] = pl.system.task_dummy(deps=[source_tids])
    scatter_done = pl.system.task_dummy(deps=[expert_tids])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_combine_notify", deps=[scatter_done, dispatch_tid]) as notify_tid:
        for peer in pl.range(EP_SIZE):
            if peer != ep_rank:
                pld.system.notify(target=combine_arrived, peer=group_base + peer, offsets=[ep_rank, 0],
                                  value=1, op=pld.NotifyOp.AtomicAdd)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_combine_wait", deps=[notify_tid]) as combine_tid:
        for peer in pl.range(EP_SIZE):
            if peer != ep_rank:
                pld.system.defer_wait(signal=combine_arrived, offsets=[peer, 0],
                                      expected=pl.cast(2 * moe_epoch - 1, pl.INT32), cmp=pld.WaitCmp.Ge)

    combined_tids = pl.array.create((PREFILL_MAX_TOKENS + EXPERT_TILE - 1) // EXPERT_TILE, pl.TASK_ID)
    for chunk in pl.range((active + EXPERT_TILE - 1) // EXPERT_TILE):
        start = chunk * EXPERT_TILE
        rows = pl.min(EXPERT_TILE, active - start)
        with pl.scope():
            shared_input = pl.slice(x_norm, [EXPERT_TILE, D], [start, 0], valid_shape=[rows, D])
            gate_value = pl.create_tensor([EXPERT_TILE, MOE_INTER], dtype=pl.BF16)
            up_value = pl.create_tensor([EXPERT_TILE, MOE_INTER], dtype=pl.BF16)
            hidden = pl.create_tensor([EXPERT_TILE, MOE_INTER], dtype=pl.BF16)
            shared_result = pl.create_tensor([EXPERT_TILE, D], dtype=pl.BF16)
            shared_weights = pl.create_tensor([EXPERT_TILE, AUX_WIDTH], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_shared_weights"):
                shared_weights[:, :] = pl.full([EXPERT_TILE, AUX_WIDTH], dtype=pl.FP32, value=1.0)
            gate_value = shared_up(shared_input, shared_w1, shared_w1_scale, gate_value, rows)
            up_value = shared_up(shared_input, shared_w3, shared_w3_scale, up_value, rows)
            hidden = swiglu(gate_value, up_value, shared_weights, hidden, rows)
            shared_result = shared_down(hidden, shared_w2, shared_w2_scale, shared_result, rows)
            with pl.spmd(rows, name_hint="moe_combine", deps=[combine_tid]) as combined_tid:
                row = pl.tile.get_block_idx()
                value = pl.full([1, D], dtype=pl.FP32, value=0.0)
                for k in pl.range(TOPK):
                    route_id = (start + row) * TOPK + k
                    value = pl.add(value, routed_output[route_id:route_id + 1, :])
                value = pl.add(value, pl.cast(shared_result[row:row + 1, :], pl.FP32))
                output[start + row:start + row + 1, :] = pl.cast(value, pl.BF16, mode="rint")
            combined_tids[chunk] = combined_tid
    combined_done = pl.system.task_dummy(deps=[combined_tids])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_consumed", deps=[combined_done, combine_tid]):
        for peer in pl.range(EP_SIZE):
            if peer != ep_rank:
                pld.system.notify(target=combine_arrived, peer=group_base + peer, offsets=[ep_rank, 0],
                                  value=1, op=pld.NotifyOp.AtomicAdd)
    for row in pl.spmd(tokens - active, name_hint="moe_inactive"):
        output[active + row:active + row + 1, :] = x[active + row:active + row + 1, :]
    return output


def make_program(epochs=2):
    """Build an EP harness with packed byte carriers and retained-window feedback."""
    if epochs < 1:
        raise ValueError("epochs must be positive")

    @pl.jit
    def moe_rank(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
        norm_weight: pl.Tensor[[D], pl.BF16],
        gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
        correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
        routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D // 2], pl.UINT8],
        routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D // 32], pl.FP8E8M0],
        routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER // 2], pl.UINT8],
        routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER // 32], pl.FP8E8M0],
        routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D // 2], pl.UINT8],
        routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D // 32], pl.FP8E8M0],
        shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
        shared_w1_scale: pl.Tensor[[D // 32, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
        shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
        shared_w2_scale: pl.Tensor[[MOE_INTER // 32, D], pl.FP8E8M0, pl.MX_B_NN],
        shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
        shared_w3_scale: pl.Tensor[[D // 32, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
        token_owners: pl.Tensor[[T_DYN], pl.INT32],
        recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
        recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.FP8E4M3FN],
        recv_scale: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D // 32], pl.UINT8],
        recv_weights: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32],
        recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
        arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
        data_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
        routed_output: pld.DistributedTensor[[T_DYN * TOPK, D], pl.FP32],
        combine_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
        output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
        num_tokens: pl.Scalar[pl.INT32],
        rank: pl.Scalar[pl.INT32],
    ):
        x.bind_dynamic(0, T_DYN)
        current = x
        for epoch in pl.range(epochs):
            output = moe(
                current, norm_weight, gate_weight, correction_bias,
                routed_w1, routed_w1_scale, routed_w2, routed_w2_scale, routed_w3, routed_w3_scale,
                shared_w1, shared_w1_scale, shared_w2, shared_w2_scale, shared_w3, shared_w3_scale,
                token_owners, recv_meta, recv_x, recv_scale, recv_weights, recv_routes, arrived, data_arrived,
                routed_output, combine_arrived, output, num_tokens, rank, 0, rank % TP_SIZE, epoch + 1,
            )
            current = output
        return output

    @pl.jit.host
    def moe_group(
        x: pl.Tensor[[EP_SIZE, T_DYN, D], pl.BF16],
        norm_weight: pl.Tensor[[EP_SIZE, D], pl.BF16],
        gate_weight: pl.Tensor[[EP_SIZE, N_EXPERTS, D], pl.FP32],
        correction_bias: pl.Tensor[[EP_SIZE, N_EXPERTS], pl.FP32],
        routed_w1: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D // 2], pl.UINT8],
        routed_w1_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D // 32], pl.FP8E8M0],
        routed_w2: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D, MOE_INTER // 2], pl.UINT8],
        routed_w2_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D, MOE_INTER // 32], pl.FP8E8M0],
        routed_w3: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D // 2], pl.UINT8],
        routed_w3_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D // 32], pl.FP8E8M0],
        shared_w1: pl.Tensor[[EP_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
        shared_w1_scale: pl.Tensor[[EP_SIZE, D // 32, MOE_INTER], pl.FP8E8M0],
        shared_w2: pl.Tensor[[EP_SIZE, MOE_INTER, D], pl.FP8E4M3FN],
        shared_w2_scale: pl.Tensor[[EP_SIZE, MOE_INTER // 32, D], pl.FP8E8M0],
        shared_w3: pl.Tensor[[EP_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
        shared_w3_scale: pl.Tensor[[EP_SIZE, D // 32, MOE_INTER], pl.FP8E8M0],
        token_owners: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
        output: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, D], pl.BF16]],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        x.bind_dynamic(1, T_DYN)
        tokens = pl.tensor.dim(x, 1)
        route_rows = tokens * TOPK
        meta_buf = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
        x_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.FP8E4M3FN)
        scale_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D // 32], dtype=pl.UINT8)
        weights_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
        routes_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
        arrived_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
        data_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
        output_buf = pld.alloc_window_buffer([route_rows, D], dtype=pl.FP32)
        combine_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            recv_meta = pld.window(meta_buf, [EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
            recv_x = pld.window(x_buf, [N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.FP8E4M3FN)
            recv_scale = pld.window(scale_buf, [N_LOCAL_EXPERTS * RECV_MAX, D // 32], dtype=pl.UINT8)
            recv_weights = pld.window(weights_buf, [N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
            recv_routes = pld.window(routes_buf, [N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
            arrived = pld.window(arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)
            data_arrived = pld.window(data_buf, [EP_SIZE, 1], dtype=pl.INT32)
            routed_output = pld.window(output_buf, [route_rows, D], dtype=pl.FP32)
            combine_arrived = pld.window(combine_buf, [EP_SIZE, 1], dtype=pl.INT32)
            moe_rank(
                x[rank], norm_weight[rank], gate_weight[rank], correction_bias[rank],
                routed_w1[rank], routed_w1_scale[rank], routed_w2[rank], routed_w2_scale[rank],
                routed_w3[rank], routed_w3_scale[rank], shared_w1[rank], shared_w1_scale[rank],
                shared_w2[rank], shared_w2_scale[rank], shared_w3[rank], shared_w3_scale[rank],
                token_owners[rank], recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
                arrived, data_arrived, routed_output, combine_arrived, output[rank], num_tokens, rank, device=rank,
            )
        return output

    return moe_group


__all__ = ["golden_moe", "make_program", "moe", "tp_token_owners"]


def build_specs(tokens, active):
    """Allocate full-size resident weights only when an explicit native run is requested."""
    from golden import ScalarSpec, TensorSpec

    if not 1 <= tokens <= C.PREFILL_MAX_TOKENS:
        raise ValueError(f"tokens must be in [1, {C.PREFILL_MAX_TOKENS}]")
    if not 0 <= active <= tokens:
        raise ValueError("active must be between zero and tokens")

    def packed_weight(n, k, offset):
        result = torch.empty(C.EP_SIZE, C.N_LOCAL_EXPERTS, n, k // 2, dtype=torch.uint8)
        columns = torch.arange(k // 2, dtype=torch.int32)
        row_codes = torch.arange(n, dtype=torch.int32)[:, None]
        for rank in range(C.EP_SIZE):
            for expert in range(C.N_LOCAL_EXPERTS):
                expert_id = rank * C.N_LOCAL_EXPERTS + expert
                low = (row_codes + columns + expert_id + expert_id // 16 + offset) % 16
                high = (3 * row_codes + columns + 5 * expert_id + expert_id // 32 + offset) % 16
                result[rank, expert] = (low | (high << 4)).to(torch.uint8)
        return result

    def shared_weight(k, n, seed):
        generator = torch.Generator().manual_seed(seed)
        matrix = torch.randn(k, n, generator=generator).mul_(0.125).to(torch.float8_e4m3fn)
        return matrix.unsqueeze(0).repeat(C.EP_SIZE, 1, 1)

    generator = torch.Generator().manual_seed(240)
    x = torch.randn(C.DP_SIZE, tokens, C.D, generator=generator).to(torch.bfloat16)
    x = x.repeat_interleave(C.TP_SIZE, dim=0)
    gate_weight = torch.randn(C.N_EXPERTS, C.D, generator=generator).mul_(0.01)
    gate_weight = gate_weight.unsqueeze(0).repeat(C.EP_SIZE, 1, 1)
    correction_bias = torch.zeros(C.N_EXPERTS)
    selected = torch.arange(C.TOPK) * (C.N_EXPERTS // C.TOPK)
    correction_bias[selected] = torch.linspace(20.0, 25.0, C.TOPK)
    correction_bias = correction_bias.unsqueeze(0).repeat(C.EP_SIZE, 1)
    specs = [
        TensorSpec("x", [C.EP_SIZE, tokens, C.D], torch.bfloat16, x),
        TensorSpec("norm_weight", [C.EP_SIZE, C.D], torch.bfloat16, 1.0, resident="stacked"),
        TensorSpec("gate_weight", list(gate_weight.shape), torch.float32, gate_weight, resident="stacked"),
        TensorSpec("correction_bias", list(correction_bias.shape), torch.float32, correction_bias, resident="stacked"),
    ]
    for name, n, k, offset in (("w1", C.MOE_INTER, C.D, 1), ("w2", C.D, C.MOE_INTER, 5), ("w3", C.MOE_INTER, C.D, 9)):
        specs.append(TensorSpec(f"routed_{name}", [C.EP_SIZE, C.N_LOCAL_EXPERTS, n, k // 2], torch.uint8,
                                lambda n=n, k=k, offset=offset: packed_weight(n, k, offset), resident="stacked"))
        specs.append(TensorSpec(f"routed_{name}_scale", [C.EP_SIZE, C.N_LOCAL_EXPERTS, n, k // 32],
                                torch.float8_e8m0fnu,
                                lambda n=n, k=k: torch.full(
                                    (C.EP_SIZE, C.N_LOCAL_EXPERTS, n, k // 32), 121, dtype=torch.uint8,
                                ).view(torch.float8_e8m0fnu), resident="stacked"))
    for name, k, n, seed in (("w1", C.D, C.MOE_INTER, 11), ("w2", C.MOE_INTER, C.D, 13), ("w3", C.D, C.MOE_INTER, 17)):
        specs.append(TensorSpec(f"shared_{name}", [C.EP_SIZE, k, n], torch.float8_e4m3fn,
                                lambda k=k, n=n, seed=seed: shared_weight(k, n, seed), resident="stacked"))
        specs.append(TensorSpec(f"shared_{name}_scale", [C.EP_SIZE, k // 32, n], torch.float8_e8m0fnu,
                                lambda k=k, n=n: torch.full(
                                    (C.EP_SIZE, k // 32, n), 127, dtype=torch.uint8,
                                ).view(torch.float8_e8m0fnu), resident="stacked"))
    specs.extend([
        TensorSpec("token_owners", [C.EP_SIZE, tokens], torch.int32,
                   tp_token_owners(tokens, C.TP_SIZE).unsqueeze(0).repeat(C.EP_SIZE, 1)),
        TensorSpec("output", [C.EP_SIZE, tokens, C.D], torch.bfloat16),
        ScalarSpec("num_tokens", torch.int32, active),
    ])
    return specs


def make_distributed_golden(epochs):
    def reference(tensors):
        current = tensors["x"].clone()
        routed = {
            name: tensors[name].flatten(0, 1)
            for name in ("routed_w1", "routed_w1_scale", "routed_w2", "routed_w2_scale", "routed_w3", "routed_w3_scale")
        }
        for _ in range(epochs):
            for base in range(0, C.EP_SIZE, C.TP_SIZE):
                local = {name: tensors[name][base] for name in (
                    "norm_weight", "gate_weight", "correction_bias", "shared_w1", "shared_w1_scale",
                    "shared_w2", "shared_w2_scale", "shared_w3", "shared_w3_scale",
                )}
                output = golden_moe(current[base], **local, **routed, token_owners=tensors["token_owners"][base],
                                    tp_size=C.TP_SIZE, num_tokens=int(tensors["num_tokens"]))
                current[base:base + C.TP_SIZE] = output.unsqueeze(0)
        tensors["output"] = current

    return reference


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", help="run the full-size distributed NPU kernel")
    parser.add_argument("-p", "--platform", choices=["a5"], default="a5")
    parser.add_argument("-d", "--devices", default=",".join(str(rank) for rank in range(C.EP_SIZE)))
    parser.add_argument("--tp", type=int, default=C.TP_SIZE)
    parser.add_argument("--ep", type=int, default=C.EP_SIZE)
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--active", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.native:
        from golden import run
        from pypto.ir.distributed_compiled_program import DistributedConfig

        devices = [int(value) for value in args.devices.split(",")]
        if len(devices) != C.EP_SIZE or len(set(devices)) != len(devices):
            parser.error(f"provide {C.EP_SIZE} distinct devices")
        result = run(
            fn=make_program(args.epochs), specs=build_specs(args.tokens, args.active),
            golden_fn=make_distributed_golden(args.epochs),
            compile_only=args.compile_only,
            config=dict(platform=args.platform, distributed_config=DistributedConfig(device_ids=devices)),
            rtol=0.02, atol=0.002,
        )
        if not result.passed:
            raise SystemExit(1)
        if args.compile_only:
            print("[MoE] Compilation passed; device accuracy was not validated.")
    else:
        from models.deepseek_v4_1_flash._golden_smoke import run_moe_golden

        run_moe_golden(golden_moe)
