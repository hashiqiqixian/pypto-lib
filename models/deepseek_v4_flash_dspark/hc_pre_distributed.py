# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""Two-rank validation harness for the HC pre-pass used by the DSpark backbone."""

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from draft_backbone import N_RANKS
from hc_pre import (
    D,
    HC_DIM,
    HC_MULT,
    MIX_HC,
    golden_hc_pre,
    hc_pre_flat_with_inv_rms,
    hc_pre_static_external_scratch_test,
)
from lookup_embedding import lookup_embedding


T = 128
VOCAB = 256


@pl.jit
def lookup_then_hc_pre(
    input_ids: pl.Tensor[[T], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    hidden_states: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    x_hc: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    x_mixed: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    post: pl.Out[pl.Tensor[[T, HC_MULT], pl.FP32]],
    comb: pl.Out[pl.Tensor[[T, HC_MULT * HC_MULT], pl.FP32]],
    inv_rms: pl.Out[pl.Tensor[[T, 1], pl.FP32]],
):
    """Write HC input through embedding lookup, then consume it in the HC pre-pass."""
    x_hc_flat = pl.reshape(x_hc, [T, HC_DIM])
    _lookup_hidden, _lookup_x_hc_flat, lookup_tid = lookup_embedding(
        input_ids, embed_weight, hidden_states, x_hc_flat
    )
    hc_pre_flat_with_inv_rms(
        x_hc_flat, hc_fn, hc_scale, hc_base, x_mixed, post, comb, inv_rms, lookup_tid
    )
    return x_mixed


@pl.jit.host
def distributed_hc_pre(
    x_flat: pl.Tensor[[N_RANKS, T, HC_DIM], pl.FP32],
    hc_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    x_mixed: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.BF16]],
    post: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT], pl.FP32]],
    comb: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT * HC_MULT], pl.FP32]],
    inv_rms: pl.Out[pl.Tensor[[N_RANKS, T, 1], pl.FP32]],
):
    """Dispatch the exact static HC child used by the composed backbone once per rank."""
    for rank in pl.range(pld.world_size()):
        hc_pre_static_external_scratch_test(
            x_flat[rank],
            hc_fn[rank],
            hc_scale[rank],
            hc_base[rank],
            x_mixed[rank],
            post[rank],
            comb[rank],
            inv_rms[rank],
            device=rank,
        )


@pl.jit.host
def distributed_lookup_then_hc_pre(
    input_ids: pl.Tensor[[N_RANKS, T], pl.INT64],
    embed_weight: pl.Tensor[[N_RANKS, VOCAB, D], pl.BF16],
    hc_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    hidden_states: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.BF16]],
    x_hc: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]],
    x_mixed: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.BF16]],
    post: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT], pl.FP32]],
    comb: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT * HC_MULT], pl.FP32]],
    inv_rms: pl.Out[pl.Tensor[[N_RANKS, T, 1], pl.FP32]],
):
    """Dispatch the lookup-to-HC producer/consumer chain once per rank."""
    for rank in pl.range(pld.world_size()):
        lookup_then_hc_pre(
            input_ids[rank],
            embed_weight[rank],
            hc_fn[rank],
            hc_scale[rank],
            hc_base[rank],
            hidden_states[rank],
            x_hc[rank],
            x_mixed[rank],
            post[rank],
            comb[rank],
            inv_rms[rank],
            device=rank,
        )


def golden_distributed_hc_pre(tensors):
    """Apply the single-rank HC golden independently to every rank slice."""
    for rank in range(N_RANKS):
        golden_hc_pre({
            "x_flat": tensors["x_flat"][rank],
            "hc_fn": tensors["hc_fn"][rank],
            "hc_scale": tensors["hc_scale"][rank],
            "hc_base": tensors["hc_base"][rank],
            "x_mixed": tensors["x_mixed"][rank],
            "post": tensors["post"][rank],
            "comb": tensors["comb"][rank],
            "inv_rms": tensors["inv_rms"][rank],
        })


def golden_distributed_lookup_then_hc_pre(tensors):
    """Apply lookup and HC golden functions to each rank's output-backed handoff."""
    for rank in range(N_RANKS):
        hidden = tensors["embed_weight"][rank].index_select(0, tensors["input_ids"][rank].long())
        tensors["hidden_states"][rank] = hidden
        tensors["x_hc"][rank] = hidden.float().unsqueeze(1).repeat(1, HC_MULT, 1)
        golden_hc_pre({
            "x_flat": tensors["x_hc"][rank].reshape(T, HC_DIM),
            "hc_fn": tensors["hc_fn"][rank],
            "hc_scale": tensors["hc_scale"][rank],
            "hc_base": tensors["hc_base"][rank],
            "x_mixed": tensors["x_mixed"][rank],
            "post": tensors["post"][rank],
            "comb": tensors["comb"][rank],
            "inv_rms": tensors["inv_rms"][rank],
        })


def build_tensor_specs(with_lookup=False):
    """Build independent two-rank inputs and outputs for the static HC child."""
    import torch
    from golden import TensorSpec

    hc_specs = [
        TensorSpec(
            "x_flat",
            [N_RANKS, T, HC_DIM],
            torch.float32,
            init_value=lambda: torch.randn(N_RANKS, T, HC_DIM) * 0.05,
        ),
        TensorSpec(
            "hc_fn",
            [N_RANKS, MIX_HC, HC_DIM],
            torch.float32,
            init_value=lambda: torch.randn(N_RANKS, MIX_HC, HC_DIM) * 0.0509,
        ),
        TensorSpec(
            "hc_scale",
            [N_RANKS, 3],
            torch.float32,
            init_value=lambda: torch.tensor([0.075997, 0.032345, 0.226238]).repeat(N_RANKS, 1),
        ),
        TensorSpec(
            "hc_base",
            [N_RANKS, MIX_HC],
            torch.float32,
            init_value=lambda: torch.randn(N_RANKS, MIX_HC),
        ),
        TensorSpec("x_mixed", [N_RANKS, T, D], torch.bfloat16, is_output=True),
        TensorSpec("post", [N_RANKS, T, HC_MULT], torch.float32, is_output=True),
        TensorSpec("comb", [N_RANKS, T, HC_MULT * HC_MULT], torch.float32, is_output=True),
        TensorSpec("inv_rms", [N_RANKS, T, 1], torch.float32, is_output=True),
    ]
    if not with_lookup:
        return hc_specs

    return [
        TensorSpec(
            "input_ids",
            [N_RANKS, T],
            torch.int64,
            init_value=lambda: torch.arange(N_RANKS * T, dtype=torch.int64).reshape(N_RANKS, T) % VOCAB,
        ),
        TensorSpec(
            "embed_weight",
            [N_RANKS, VOCAB, D],
            torch.bfloat16,
            init_value=lambda: torch.randn(N_RANKS, VOCAB, D, dtype=torch.bfloat16),
        ),
        *hc_specs[1:4],
        TensorSpec("hidden_states", [N_RANKS, T, D], torch.bfloat16, is_output=True),
        TensorSpec("x_hc", [N_RANKS, T, HC_MULT, D], torch.float32, is_output=True),
        *hc_specs[4:],
    ]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser(description="Validate HC pre through the two-rank host dispatcher.")
    parser.add_argument("-p", "--platform", default="a2a3", choices=("a2a3", "a2a3sim"))
    parser.add_argument("-d", "--device", default=",".join(str(rank) for rank in range(N_RANKS)))
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--with-lookup", action="store_true")
    args = parser.parse_args()

    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) != N_RANKS:
        parser.error(f"need exactly {N_RANKS} devices, got {device_ids}")

    result = run_jit(
        fn=distributed_lookup_then_hc_pre if args.with_lookup else distributed_hc_pre,
        specs=build_tensor_specs(with_lookup=args.with_lookup),
        golden_fn=(
            golden_distributed_lookup_then_hc_pre if args.with_lookup else golden_distributed_hc_pre
        ),
        compile_only=args.compile_only,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
        ),
        runtime_cfg=dict(platform=args.platform, enable_dep_gen=True),
        rtol=1e-3,
        atol=1e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
