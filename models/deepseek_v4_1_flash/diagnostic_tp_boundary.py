# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Replay captured FP32 partials through the unmodified TP output collective.

This diagnostic checks a two-addend sum and one BF16 conversion, not model
accuracy. It loads no model weights and does not change an acceptance budget.
"""
# ci: no-sim
# ci: a5
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig
from golden import TensorSpec, run
from models.deepseek_v4_1_flash.config import D, TP_SIZE, PREFILL_MAX_TOKENS, T_DYN
from models.deepseek_v4_1_flash.attention_sp import (
    SP_T_DYN, prefill_sp_input_allgather, prefill_sp_output_reduce_scatter,
)


def make_program(world, capacity, with_gather=False):
    @pl.jit
    def replay_rank(
        partial: pl.Tensor[[T_DYN, D], pl.FP32],
        local_input: pl.Tensor[[SP_T_DYN, D], pl.BF16],
        gathered: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
        output: pl.Out[pl.Tensor[[SP_T_DYN, D], pl.BF16]],
        input_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
        input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
        arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
    ):
        partial.bind_dynamic(0, T_DYN)
        output.bind_dynamic(0, SP_T_DYN)
        tokens = pl.tensor.dim(partial, 0)
        if with_gather:
            prefill_sp_input_allgather(local_input, input_window, input_arrived, gathered,
                                      rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, tokens, 1)
        prefill_sp_output_reduce_scatter(
            partial, window, arrived, output, rank // TP_SIZE * TP_SIZE,
            rank % TP_SIZE, tokens, 1,
        )
        return output

    @pl.jit.host
    def replay_group(
        partial: pl.Tensor[[world, T_DYN, D], pl.FP32],
        local_input: pl.Tensor[[world, SP_T_DYN, D], pl.BF16],
        gathered: pl.Out[pl.Tensor[[world, T_DYN, D], pl.BF16]],
        output: pl.Out[pl.Tensor[[world, SP_T_DYN, D], pl.BF16]],
    ):
        partial.bind_dynamic(1, T_DYN)
        output.bind_dynamic(1, SP_T_DYN)
        input_buf = pld.alloc_window_buffer([capacity, D], dtype=pl.BF16)
        input_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        data_buf = pld.alloc_window_buffer([capacity, D], dtype=pl.FP32)
        signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            input_data = pld.window(input_buf, [capacity, D], dtype=pl.BF16)
            input_signal = pld.window(input_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
            data = pld.window(data_buf, [capacity, D], dtype=pl.FP32)
            signal = pld.window(signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
            replay_rank(partial[rank], local_input[rank], gathered[rank], output[rank],
                        input_data, input_signal, data, signal, rank, device=rank)
    return replay_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, help="Directory containing rankN-tp-boundaries.pt")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--with-gather", action="store_true", help="Precede reduction with the SWA input collective")
    args = parser.parse_args()
    devices = [int(d) for d in args.devices.split(",")]
    if TP_SIZE != 2 or args.tp != 2 or len(devices) != 4 or args.ep != 4:
        parser.error("this captured diagnostic requires TP2/DP2/EP4")
    if len(set(devices)) != len(devices):
        parser.error("devices must be distinct")
    torch.set_num_threads(4)
    values = [torch.load(Path(args.trace) / f"rank{rank}-tp-boundaries.pt", weights_only=True)
              for rank in range(4)]
    partial = torch.stack([v["published"] for v in values])
    if partial.dtype != torch.float32 or partial.shape != (4, 32, D) or not torch.isfinite(partial).all():
        raise ValueError("expected finite FP32 captured partials [4,32,D]")
    def golden(tensors):
        for base in (0, 2):
            if args.with_gather:
                full = tensors["local_input"][base:base + 2].flatten(0, 1)
                tensors["gathered"][base:base + 2].copy_(full.unsqueeze(0).expand(2, -1, -1))
            total = (tensors["partial"][base] + tensors["partial"][base + 1]).bfloat16()
            tensors["output"][base:base + 2].copy_(total.reshape(2, 16, D))
    def compare(actual, expected, **kwargs):
        artifact = Path(args.artifact)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if artifact.exists():
            raise FileExistsError("refusing to overwrite diagnostic result")
        torch.save(dict(partial=partial, actual=actual, expected=expected), artifact)
        coords = (actual != expected).nonzero()
        for rank, row, col in coords[:20].tolist():
            print("MISMATCH",rank,row,col,float(actual[rank,row,col]),float(expected[rank,row,col]),flush=True)
        print("TP REPLAY mismatches",len(coords),flush=True)
        return not len(coords), "exact two-addend sum followed by BF16 rounding"
    torch.manual_seed(20260929)
    result = run(fn=make_program(4, 32, args.with_gather), specs=[
        TensorSpec("partial", [4, 32, D], torch.float32, init_value=partial, resident="stacked"),
        TensorSpec("local_input", [4, 16, D], torch.bfloat16,
                   init_value=torch.randn(4, 16, D).bfloat16(), resident="stacked"),
        TensorSpec("gathered", [4, 32, D], torch.bfloat16, resident="stacked"),
        TensorSpec("output", [4, 16, D], torch.bfloat16, resident="stacked"),
    ], golden_fn=golden, compare_fn={"output": compare,
        "gathered": lambda a, e, **kw: (not args.with_gather or torch.equal(a, e), "input gather")}, config=dict(
        platform="a5", distributed_config=DistributedConfig(device_ids=devices),
        ring_heap=512 << 20, save_kernels=True, save_kernels_dir=args.build_dir,
    ))
    print("TP REPLAY",result.passed,result.work_dir,flush=True)
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
