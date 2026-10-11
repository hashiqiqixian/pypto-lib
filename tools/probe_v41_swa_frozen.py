# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Replay a frozen one-card SWA attention input without loading model weights."""

import argparse

import pypto.language as pl
import torch

from golden import ScalarSpec, TensorSpec, run
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.decode_attn_swa import attend_window


@pl.jit
def frozen_attention(
    q: pl.Tensor[[C.T_DYN, C.LOCAL_H * C.HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[C.T_DYN, 128, C.HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[C.T_DYN, 128], pl.INT32],
    sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
    attended: pl.Out[pl.Tensor[[C.T_DYN, C.LOCAL_H * C.HEAD_DIM], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    attend_window(q, selected, indices, sink, attended, num_tokens)
    return attended


REFERENCE = None


def golden(values):
    values["attended"].copy_(REFERENCE)


def compare(actual, expected, **_):
    changed = actual != expected
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
    print(f"[FROZEN SWA] changed={int(changed.sum())}/{actual.numel()} "
          f"relative_l2={relative_l2.item():.9g}", flush=True)
    return bool(torch.isfinite(actual.float()).all()), "finite attention output"


def main():
    global REFERENCE
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    data = torch.load(args.capture, map_location="cpu", weights_only=True)
    REFERENCE = data["expected"].contiguous()
    inputs = {"q": data["q"], "selected": data["selected"],
              "indices": data["indices"], "sink": data["sink"]}
    specs = [TensorSpec(name, list(value.shape), value.dtype,
                        init_value=lambda value=value: value.clone()) for name, value in inputs.items()]
    specs.extend((TensorSpec("attended", list(REFERENCE.shape), REFERENCE.dtype),
                  ScalarSpec("num_tokens", torch.int32, REFERENCE.shape[0])))
    result = run(frozen_attention, specs, golden_fn=golden,
                 config=dict(platform="a5", device_id=args.device),
                 compare_fn={"attended": compare})
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
