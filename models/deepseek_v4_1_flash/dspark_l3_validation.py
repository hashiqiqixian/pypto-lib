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

from config import DP_SIZE, EP_SIZE, TP_SIZE
from dspark_drafter import l3_dspark_drafter
from golden import ScalarSpec, TensorSpec, run
from pypto.ir import DistributedConfig


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
    ok = bool(torch.isfinite(actual.float()).all())
    return ok, "finite cache scales"


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
        specs.append(TensorSpec(name, shape, dtype, init_value=initial, resident="stacked"))
    return specs


def golden_zero(values):
    values["head_hidden"].zero_()


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
        golden_fn=golden_zero,
        compare_fn={"window_cache": _zero_cache, "window_cache_scale": _finite_cache_scale},
        config={
            "backend_type": "a5",
            "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
            "ring_heap": 2 * 1024**3,
            "log_level": "warn",
        },
        atol=0.0,
        rtol=0.0,
    )
    print(result)
    if not result.passed:
        raise SystemExit(1)
