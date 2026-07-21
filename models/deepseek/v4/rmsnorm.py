# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 attention RMSNorm (dynamic shape): normalizes token-major
activations for both decode and prefill attention paths."""

import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ
T_DYN = pl.dynamic("RMS_NORM_T_DYN")


# model config
D = M.hidden_size
EPS = M.rms_norm_eps

# tiling
D_TILE = 128
T_TILE = 8
assert D % D_TILE == 0, "D must be divisible by D_TILE"
assert (DECODE_BATCH * DECODE_SEQ) % T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % T_TILE == 0


# This kernel is shared by dynamic prefill and fixed-shape decode/MoE paths;
# infer token-bearing tensor metadata from each caller.
@pl.jit.inline
def rms_norm(
    x: pl.Tensor,
    norm_w: pl.Tensor[[D], pl.BF16],
    x_normed: pl.Tensor,
):
    t_dim = pl.tensor.dim(x, 0)
    # Capture form (not `for ... in pl.spmd`): callers need the producer TaskId to
    # hang a `pl.system.task_dummy` barrier off it and defer non-critical consumers.
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    with pl.spmd(token_tiles, name_hint="rms_norm", allow_early_resolve=True) as rms_tid:
        tg_idx = pl.tile.get_block_idx()
        tg = tg_idx * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - tg)
        if valid_rows == T_TILE:
            # Preserve the established Tensor-level reduction and rounding path
            # for complete tiles. Decode and aligned prefill inputs use this
            # branch, so dynamic-shape support does not perturb their numerics.
            full_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            for full_rms_db in pl.pipeline(D // D_TILE, stage=2):
                full_rms_d0 = full_rms_db * D_TILE
                full_rms_x = pl.cast(
                    x[tg : tg + T_TILE, full_rms_d0 : full_rms_d0 + D_TILE],
                    target_type=pl.FP32,
                )
                full_sq_sum = pl.add(
                    full_sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(full_rms_x, full_rms_x)), [1, T_TILE]),
                )
            full_inv_rms = pl.rsqrt(pl.add(pl.mul(full_sq_sum, 1.0 / D), EPS), high_precision=True)
            full_inv_rms_t = pl.reshape(full_inv_rms, [T_TILE, 1])
            for full_apply_db in pl.pipeline(D // D_TILE, stage=2):
                full_apply_d0 = full_apply_db * D_TILE
                full_apply_x = pl.cast(
                    x[tg : tg + T_TILE, full_apply_d0 : full_apply_d0 + D_TILE],
                    target_type=pl.FP32,
                )
                full_norm_w = pl.cast(
                    pl.reshape(norm_w[full_apply_d0 : full_apply_d0 + D_TILE], [1, D_TILE]),
                    pl.FP32,
                )
                full_normed = pl.col_expand_mul(
                    pl.row_expand_mul(full_apply_x, full_inv_rms_t),
                    full_norm_w,
                )
                x_normed[tg : tg + T_TILE, full_apply_d0 : full_apply_d0 + D_TILE] = pl.cast(
                    full_normed,
                    target_type=pl.BF16,
                    mode="rint",
                )
        else:
            row_reduce_tmp = pl.create_tile(
                [T_TILE, D_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            tail_sq_sum = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            for tail_rms_db in pl.range(D // D_TILE):
                tail_rms_d0 = tail_rms_db * D_TILE
                tail_rms_input = pl.load(
                    x,
                    [tg, tail_rms_d0],
                    [T_TILE, D_TILE],
                    valid_shapes=[valid_rows, D_TILE],
                    target_memory=pl.MemorySpace.Vec,
                )
                tail_rms_x = pl.cast(tail_rms_input, target_type=pl.FP32)
                tail_sq_sum = pl.add(
                    tail_sq_sum,
                    pl.reshape(
                        pl.row_sum(pl.mul(tail_rms_x, tail_rms_x), row_reduce_tmp),
                        [1, T_TILE],
                    ),
                )
            tail_inv_rms = pl.rsqrt(pl.add(pl.mul(tail_sq_sum, 1.0 / D), EPS), high_precision=True)
            tail_inv_rms_t = pl.reshape(tail_inv_rms, [T_TILE, 1])
            for tail_apply_db in pl.pipeline(D // D_TILE, stage=2):
                tail_apply_d0 = tail_apply_db * D_TILE
                tail_apply_input = pl.load(
                    x,
                    [tg, tail_apply_d0],
                    [T_TILE, D_TILE],
                    valid_shapes=[valid_rows, D_TILE],
                    target_memory=pl.MemorySpace.Vec,
                )
                tail_apply_x = pl.cast(tail_apply_input, target_type=pl.FP32)
                tail_norm_w_input = pl.load(
                    norm_w,
                    [tail_apply_d0],
                    [D_TILE],
                    target_memory=pl.MemorySpace.Vec,
                )
                tail_norm_w = pl.cast(pl.reshape(tail_norm_w_input, [1, D_TILE]), pl.FP32)
                tail_normed = pl.col_expand_mul(
                    pl.row_expand_mul(tail_apply_x, tail_inv_rms_t),
                    tail_norm_w,
                )
                tail_normed_bf16 = pl.cast(tail_normed, target_type=pl.BF16, mode="rint")
                pl.store(
                    pl.set_validshape(tail_normed_bf16, valid_rows, D_TILE),
                    [tg, tail_apply_d0],
                    x_normed,
                )

    return rms_tid


@pl.jit
def rms_norm_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_normed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    x_normed.bind_dynamic(0, T_DYN)

    rms_norm(x, norm_w, x_normed)
    return x_normed


def golden_rms_norm(x, norm_w):
    import torch

    x = x.float()
    norm_w = norm_w.float()
    inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return (x * inv * norm_w).to(torch.bfloat16)


def golden_rms_norm_test(tensors):
    tensors["x_normed"][:] = golden_rms_norm(tensors["x"], tensors["norm_w"])


def build_tensor_specs(B, S):
    import torch
    from golden import TensorSpec

    T = B * S

    def init_x():
        return torch.randn(T, D) - 0.5

    def init_norm_w():
        return torch.randn(D) * 0.1 + 1.0

    return [
        TensorSpec("x", [T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("norm_w", [D], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("x_normed", [T, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    MODES = {
        "decode":  (DECODE_BATCH, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 attention RMSNorm validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="all",
                        help="Use decode or prefill batch sizes, or 'all' to test both.")
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2, 4))
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    modes_to_run = list(MODES.keys()) if args.mode == "all" else [args.mode]

    for mode_name in modes_to_run:
        B, S = MODES[mode_name]
        print(f"--- rms_norm_test {mode_name}: B={B}, S={S} ---")
        result = run_jit(
            fn=rms_norm_test,
            specs=build_tensor_specs(B, S),
            golden_fn=golden_rms_norm_test,
            runtime_dir=args.runtime_dir,
            golden_data=args.golden_data,
            compile_cfg=dict(dump_passes=args.dump_passes),
            runtime_cfg=dict(
                platform=args.platform,
                device_id=args.device,
                enable_l2_swimlane=args.enable_l2_swimlane,
            ),
            rtol=5e-3,
            atol=5e-3,
            compare_fn={
                "x_normed": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            },
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
