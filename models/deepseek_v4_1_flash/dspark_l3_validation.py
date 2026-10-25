# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Private four-rank zero-input smoke for the complete V4.1 DSpark drafter."""

import inspect

import torch

from config import DP_SIZE, DSPARK_DRAFT_LAYERS, DSPARK_NOISE_TOKEN_ID, DSPARK_QUERY_WIDTH, EP_SIZE, TP_SIZE
from dspark_drafter import l3_dspark_drafter
from golden import ScalarSpec, TensorSpec, run
from hc import golden_hc_post, golden_hc_pre, golden_mhc_mixes
from pypto.ir import DistributedConfig
from pypto.runtime import ExecutionMode


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "fp8e4m3fn": torch.float8_e4m3fn,
    "fp8e8m0": torch.float8_e8m0fnu,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
}


def _dimension(value):
    if isinstance(value, int):
        return value
    name = str(value)
    if "V41_DSPARK_QUERY_T_DYN" in name:
        return 5
    if any(marker in name for marker in ("V41_T_DYN", "V41_B_DYN", "V41_TABLE_DYN", "V41_ORI_BLOCKS_DYN")):
        return 1
    raise ValueError(f"unknown dynamic dimension: {value}")


def _zero_cache(actual, _expected, **_kwargs):
    raw = actual.contiguous().view(torch.uint8)
    ok = not bool(raw.any())
    return ok, "zero-input cache payload"


def _finite_cache_scale(actual, _expected, **_kwargs):
    active = actual[:, :, 0, : DSPARK_QUERY_WIDTH + 1, 0, :]
    untouched = actual[:, :, 0, DSPARK_QUERY_WIDTH + 1 :, 0, :]
    ok = bool(torch.isfinite(actual.float()).all() and (active > 0).all() and not untouched.any())
    return ok, "context and query cache rows published; unused rows unchanged"


def _embedding(shape):
    weight = torch.zeros(shape, dtype=torch.bfloat16)
    pattern = (torch.arange(shape[-1], dtype=torch.float32) % 29 - 14) / 32
    for rank in range(EP_SIZE):
        weight[rank, 0] = ((rank + 1) * pattern).bfloat16()
        weight[rank, DSPARK_NOISE_TOKEN_ID] = ((rank + 1) * pattern.flip(0)).bfloat16()
    return weight


def build_specs():
    specs = []
    for name, parameter in inspect.signature(l3_dspark_drafter._func).parameters.items():
        annotation = parameter.annotation
        dtype = _DTYPES[str(annotation.dtype)]
        if not hasattr(annotation, "shape") or annotation.shape is None:
            specs.append(ScalarSpec(name, dtype, 1, compile_runtime=True))
            continue
        shape = [_dimension(value) for value in annotation.shape]
        initial = None
        if name in {"main_norm_weight", "attn_norm_weight", "q_norm_weight", "kv_norm_weight", "ffn_norm_weight"}:
            initial = 1
        elif name in {"context_rope_cos", "query_rope_cos"}:
            initial = 1
        elif name == "first_query_positions":
            initial = 1
        elif name == "mxfp4_pair_lut":
            from quant import build_mxfp4_pair_lut

            initial = build_mxfp4_pair_lut().unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()
        elif name == "embedding_weight":
            initial = lambda shape=shape: _embedding(shape)
        specs.append(TensorSpec(name, shape, dtype, init_value=initial, resident="stacked"))
    return specs


def golden_residual(values):
    ids = torch.tensor([0] + [DSPARK_NOISE_TOKEN_ID] * (DSPARK_QUERY_WIDTH - 1))
    for rank in range(EP_SIZE):
        hidden = values["embedding_weight"][rank, ids].float()
        x_hc = hidden.unsqueeze(1).expand(-1, 4, -1).clone()
        pre_mix = torch.zeros(DSPARK_QUERY_WIDTH, 4)
        pre_mix[:, 0] = 1
        sublayer = torch.zeros_like(hidden, dtype=torch.bfloat16)
        for layer in range(DSPARK_DRAFT_LAYERS):
            attn = golden_mhc_mixes(
                x_hc,
                values["hc_attn_fn"][rank, layer],
                values["hc_attn_scale"][rank, layer],
                values["hc_attn_base"][rank, layer],
            )
            x_hc = golden_hc_post(sublayer, x_hc, attn[1], attn[2]).float()
            pre_mix, post_mix, residual_mix = golden_mhc_mixes(
                x_hc,
                values["hc_ffn_fn"][rank, layer],
                values["hc_ffn_scale"][rank, layer],
                values["hc_ffn_base"][rank, layer],
            )
            x_hc = golden_hc_post(sublayer, x_hc, post_mix, residual_mix).float()
        values["head_hidden"][rank, 0] = golden_hc_pre(x_hc, pre_mix).bfloat16()
    if not bool(values["head_hidden"].abs().sum() > 0):
        raise AssertionError("the residual reference must be nonzero")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=TP_SIZE)
    parser.add_argument("--dp", type=int, default=DP_SIZE)
    parser.add_argument("--ep", type=int, default=EP_SIZE)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    if (args.tp, args.dp, args.ep) != (1, 4, 4):
        parser.error("this validation case requires TP1/DP4/EP4")
    devices = [int(value) for value in args.device.split(",")]
    if len(devices) != 4:
        parser.error("exactly four device ids are required")
    result = run(
        fn=l3_dspark_drafter,
        specs=build_specs(),
        golden_fn=golden_residual,
        compare_fn={"window_cache": _zero_cache, "window_cache_scale": _finite_cache_scale},
        config={
            "backend_type": "a5",
            "execution_mode": ExecutionMode.ONBOARD,
            "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
            "ring_heap": 2 * 1024**3,
            "log_level": "warn",
        },
        atol=0.03,
        rtol=0.03,
    )
    print(result)
    if not result.passed:
        raise SystemExit(1)
