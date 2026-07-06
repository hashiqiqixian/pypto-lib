# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MTP decode layer.

This module is the MTP draft-model analogue of ``decode_layer.py`` for the
SWA-only MTP attention path. It wires the decode-time MTP layer as:

    current hidden / embedding state + previous pre-hc_head hidden
        -> MTP projection
        -> SWA attention tail
        -> MoE/FFN tail
        -> next pre-hc_head hidden

The input named ``prev_pre_hc_hidden`` is intentionally flat ``[T, HC_DIM]``:
it is the target model's or previous draft step's pre-``hc_head`` residual,
not the post-``hc_head`` dense hidden state. The returned ``next_pre_hc_hidden``
has the same flat contract so serving can feed it into the next MTP draft step.
"""

import pypto.language as pl

from decode_attention_swa import (
    BLOCK_SIZE,
    HEAD_DIM,
    H,
    MAX_SEQ_LEN,
    O_GROUP_IN,
    O_GROUPS,
    O_LORA,
    ORI_MAX_BLOCKS,
    Q_LORA,
    ROPE_HEAD_DIM,
    SPARSE_CMP_MAX_BLOCKS,
    build_tensor_specs as build_swa_tensor_specs,
    golden_attention_swa,
)
from moe import (
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    TOPK as MOE_TOPK,
    VOCAB as MOE_VOCAB,
    build_tensor_specs_ep1 as build_moe_ep1_tensor_specs,
    golden_moe_ep1,
)
from mtp import (
    B,
    D,
    D_CHUNK,
    HC_DIM,
    HC_MULT,
    MIX_HC,
    S,
    T,
    build_projection_tensor_specs,
    golden_mtp_projection,
    mtp_decoder_layer_tail,
    mtp_projection_impl,
)


@pl.jit
def mtp_decode_layer(
    hidden_states: pl.Tensor[[B, S, D], pl.BF16],
    prev_pre_hc_hidden: pl.Tensor[[T, HC_DIM], pl.BF16],
    position_ids: pl.Tensor[[T], pl.INT32],
    enorm_w: pl.Tensor[[D], pl.FP32],
    hnorm_w: pl.Tensor[[D], pl.FP32],
    e_proj_w: pl.Tensor[[D, D], pl.INT8],
    e_proj_w_scale: pl.Tensor[[D], pl.FP32],
    e_proj_smooth: pl.Tensor[[D], pl.FP32],
    h_proj_w: pl.Tensor[[D, D], pl.INT8],
    h_proj_w_scale: pl.Tensor[[D], pl.FP32],
    h_proj_smooth: pl.Tensor[[D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.Tensor[[B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    cmp_kv: pl.Tensor[[B * SPARSE_CMP_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[MOE_VOCAB, MOE_TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    next_pre_hc_hidden: pl.Out[pl.Tensor[[T, HC_DIM], pl.BF16]],
) -> pl.Tensor[[T, HC_DIM], pl.BF16]:
    projected_hidden = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    projected_hidden = mtp_projection_impl(
        hidden_states,
        prev_pre_hc_hidden,
        position_ids,
        enorm_w,
        hnorm_w,
        e_proj_w,
        e_proj_w_scale,
        e_proj_smooth,
        h_proj_w,
        h_proj_w_scale,
        h_proj_smooth,
        projected_hidden,
    )
    next_pre_hc_stack = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    next_pre_hc_stack = mtp_decoder_layer_tail(
        projected_hidden,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        gamma_cq,
        gamma_ckv,
        freqs_cos,
        freqs_sin,
        kv_cache,
        block_table,
        ori_slot_mapping,
        position_ids,
        cmp_kv,
        cmp_block_table,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        hc_ffn_fn,
        hc_ffn_scale,
        hc_ffn_base,
        norm_w,
        gate_w,
        gate_bias,
        tid2eid,
        input_ids,
        routed_w1,
        routed_w1_scale,
        routed_w3,
        routed_w3_scale,
        routed_w2,
        routed_w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w3,
        shared_w3_scale,
        shared_w2,
        shared_w2_scale,
        next_pre_hc_stack,
    )
    next_pre_hc_flat = pl.reshape(next_pre_hc_hidden, [T, HC_DIM])
    next_pre_hc_stack_flat = pl.reshape(next_pre_hc_stack, [T, HC_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_decode_layer_flatten"):
        for k0 in pl.pipeline(0, HC_DIM, D_CHUNK, stage=2):
            next_pre_hc_flat[:, k0:k0 + D_CHUNK] = next_pre_hc_stack_flat[:, k0:k0 + D_CHUNK]
    return next_pre_hc_hidden


def golden_mtp_decode_layer(tensors):
    import torch

    tensors["prev_hidden_states"] = tensors["prev_pre_hc_hidden"]
    tensors["projected_hidden"] = torch.empty(T, HC_MULT, D, dtype=torch.bfloat16)
    golden_mtp_projection(tensors)

    tensors["x_hc"] = tensors["projected_hidden"]
    tensors["x_out"] = torch.empty(T, HC_MULT, D, dtype=torch.bfloat16)
    golden_attention_swa(tensors)

    tensors["x_hc"] = tensors["x_out"]
    tensors["x_next"] = torch.empty(T, HC_MULT, D, dtype=torch.bfloat16)
    golden_moe_ep1(tensors)

    next_pre_hc = tensors["x_next"].flatten(1)
    tensors["next_pre_hc_hidden"][:] = next_pre_hc

    step1_tensors = dict(tensors)
    step1_tensors["prev_hidden_states"] = next_pre_hc
    step1_tensors["projected_hidden"] = torch.empty(T, HC_MULT, D, dtype=torch.bfloat16)
    golden_mtp_projection(step1_tensors)
    assert tuple(step1_tensors["projected_hidden"].shape) == (T, HC_MULT, D)


def _extend_specs(specs, candidates, skip_names, rename=None):
    rename = rename or {}
    names = {getattr(spec, "name", None) for spec in specs}
    for spec in candidates:
        name = getattr(spec, "name", None)
        if name in skip_names:
            continue
        out_name = rename.get(name, name)
        if out_name in names:
            continue
        if out_name == name:
            specs.append(spec)
        else:
            from golden import TensorSpec

            specs.append(TensorSpec(
                out_name,
                spec.shape,
                spec.dtype,
                init_value=spec.init_value,
                is_output=spec.is_output,
            ))
        names.add(out_name)
    return specs


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    swa_specs = build_swa_tensor_specs()
    swa_by_name = {spec.name: spec for spec in swa_specs if isinstance(spec, TensorSpec)}
    specs = []
    _extend_specs(
        specs,
        build_projection_tensor_specs(),
        {"projected_hidden"},
        rename={"prev_hidden_states": "prev_pre_hc_hidden"},
    )
    for idx, spec in enumerate(specs):
        if spec.name == "position_ids":
            src = swa_by_name["position_ids"]
            specs[idx] = TensorSpec(
                "position_ids",
                src.shape,
                src.dtype,
                init_value=src.init_value,
                is_output=src.is_output,
            )
            break
    _extend_specs(
        specs,
        swa_specs,
        {
            "x_hc",
            "x_out",
        },
    )
    _extend_specs(
        specs,
        build_moe_ep1_tensor_specs(layer_id=0),
        {
            "x_hc",
            "x_next",
            "layer_id",
        },
    )
    specs.append(TensorSpec("next_pre_hc_hidden", [T, HC_DIM], torch.bfloat16, is_output=True))
    return specs


if __name__ == "__main__":
    import argparse
    import torch
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    result = run_jit(
        fn=mtp_decode_layer,
        specs=build_tensor_specs(),
        golden_fn=golden_mtp_decode_layer,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "next_pre_hc_hidden": ratio_allclose(atol=1e-2, rtol=1e-2, max_error_ratio=0.02),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
