# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
# ci: no-sim
"""DeepSeek-V4.1 fused shared and routed experts with distributed transport."""

import argparse
import hashlib
import os
from pathlib import Path

import torch
import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig
from pypto.runtime import pto_isa_include_dir
from simpler_setup.environment import PROJECT_ROOT as SIMPLER_ROOT


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=str, default=None)
    parser.add_argument("--ep", type=int, choices=[2, 4, 8], default=2)
    parser.add_argument("--experts-per-rank", type=int, default=4)
    parser.add_argument("--case", choices=["all", "bounds", "tails", "zero", "full", "balanced"], default="all")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    args = parser.parse_args()
    from dataclasses import replace

    import config

    test_experts = args.ep * args.experts_per_rank
    if args.experts_per_rank <= 0 or not config.FLASH.num_experts_per_tok <= test_experts <= 384:
        parser.error("--experts-per-rank must give between TopK and 384 total routed experts")
    config.FLASH = replace(config.FLASH, n_routed_experts=test_experts)
    config.N_LOCAL_EXPERTS = args.experts_per_rank
    device_ids = list(range(args.ep)) if args.device is None else [int(value) for value in args.device.split(",")]
    if len(device_ids) != args.ep or len(set(device_ids)) != args.ep:
        parser.error("--device must name one distinct device per EP rank")


from config import FLASH as M
from config import EP_SIZE as EP_WORLD_SIZE, MOE_RECV_MAX as RECV_MAX
from config import N_LOCAL_EXPERTS, MOE_TOKENS
from config import dspark_local_experts, DSPARK_NUM_EXPERTS_PER_TOK as DRAFT_TOPK

from expert_routed import (
    H_SCALE, K_SCALE, MX_GROUP, MX_PACKED_LANE_COLS, MX_W1_PACKED_ROWS,
    MX_W2_PACKED_ROWS, MX_W3_PACKED_ROWS, OUTPUT_SENTINEL, build_tensor_specs,
    compare_local_compute, golden_local_compute,
)
from moe_workspace import fused_workspace_bytes


D = M.hidden_size
TOPK = M.num_experts_per_tok
MOE_INTER = M.moe_intermediate_size


def _fused_external_kernel(local_experts, topk):
    """Prepare a content-addressed, shape-specialized native entry."""
    kernels = Path(__file__).parent / "kernels"
    entry = kernels / "moe_fused.cpp"
    values = dict(MOE_EP=EP_WORLD_SIZE, MOE_EXPERTS=local_experts, MOE_TOKENS=MOE_TOKENS,
                  MOE_TOPK=topk, MOE_HIDDEN=D, MOE_INTER=MOE_INTER)
    text = "".join(f"#define {name} {value}\n" for name, value in values.items())
    text += f'#include "{entry.as_posix()}"\n'
    digest = hashlib.sha256(text.encode())
    for name in ("moe_fused.cpp", "moe_pipeline.hpp", "moe_pipeline_compute.hpp"):
        digest.update((kernels / name).read_bytes())
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    directory = cache / "pypto-lib" / "moe-fused" / digest.hexdigest()
    wrapper = directory / "entry.cpp"
    if not wrapper.exists():
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / f"entry.{os.getpid()}.tmp"
        temporary.write_text(text)
        temporary.replace(wrapper)
    elif wrapper.read_text() != text:
        raise RuntimeError("Fused MoE source cache mismatch")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/latest"))
    include_dirs = [pto_isa_include_dir(), kernels, SIMPLER_ROOT / "src/common/platform_comm"]
    for base in (cann / "compiler", cann, cann / (os.uname().machine + "-linux")):
        for relative in ("asc", "asc/include", "include"):
            if (base / relative).is_dir():
                include_dirs.append(base / relative)
    return pl.jit.extern(core_type="mixed", aic_source=wrapper, aiv_source=wrapper,
                         dual_aiv_dispatch=True, include_dirs=tuple(include_dirs))


@_fused_external_kernel(N_LOCAL_EXPERTS, TOPK)
def _fused_experts(
    indices: pl.Tensor, x: pl.Tensor, x_scale: pl.Tensor, weights: pl.Tensor,
    w1: pl.Tensor, w1_scale: pl.Tensor, w3: pl.Tensor, w3_scale: pl.Tensor,
    w2: pl.Tensor, w2_scale: pl.Tensor,
    shared_w1: pl.Tensor, shared_w1_scale: pl.Tensor,
    shared_w3: pl.Tensor, shared_w3_scale: pl.Tensor,
    shared_w2: pl.Tensor, shared_w2_scale: pl.Tensor,
    output: pl.Out[pl.Tensor], workspace: pl.InOut[pld.DistributedTensor],
    active_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    return output


DRAFT_LOCAL_EXPERTS = dspark_local_experts(EP_WORLD_SIZE)


@_fused_external_kernel(DRAFT_LOCAL_EXPERTS, DRAFT_TOPK)
def _fused_experts_draft(
    indices: pl.Tensor, x: pl.Tensor, x_scale: pl.Tensor, weights: pl.Tensor,
    w1: pl.Tensor, w1_scale: pl.Tensor, w3: pl.Tensor, w3_scale: pl.Tensor,
    w2: pl.Tensor, w2_scale: pl.Tensor,
    shared_w1: pl.Tensor, shared_w1_scale: pl.Tensor,
    shared_w3: pl.Tensor, shared_w3_scale: pl.Tensor,
    shared_w2: pl.Tensor, shared_w2_scale: pl.Tensor,
    output: pl.Out[pl.Tensor], workspace: pl.InOut[pld.DistributedTensor],
    active_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    return output


@pl.jit.inline(auto_scope=False)
def routed_moe(
    indices: pl.Tensor,
    x_norm_mx: pl.Tensor,
    x_norm_scale: pl.Tensor,
    weights: pl.Tensor,
    output: pl.Tensor,
    w1: pl.Tensor,
    w1_scale: pl.Tensor[None, pl.FP8E8M0, pl.MX_B_NN],
    w3: pl.Tensor,
    w3_scale: pl.Tensor[None, pl.FP8E8M0, pl.MX_B_NN],
    w2: pl.Tensor,
    w2_scale: pl.Tensor[None, pl.FP8E8M0, pl.MX_B_NN],
    shared_w1: pl.Tensor,
    shared_w1_scale: pl.Tensor,
    shared_w3: pl.Tensor,
    shared_w3_scale: pl.Tensor,
    shared_w2: pl.Tensor,
    shared_w2_scale: pl.Tensor,
    recv_x: pld.DistributedTensor,
    num_tokens: pl.Scalar[pl.INT32],
    local_experts: pl.constexpr = N_LOCAL_EXPERTS,
    topk: pl.constexpr = TOPK,
):
    """Execute shared/routed experts with persistent native tile pipelines."""
    if local_experts == DRAFT_LOCAL_EXPERTS and topk == DRAFT_TOPK:
        with pl.spmd(pl.system.available_cluster_count(), sync_start=True, name_hint="moe_fused_draft") as draft_ready:
            output = _fused_experts_draft(
                indices, x_norm_mx, x_norm_scale, weights,
                w1, w1_scale, w3, w3_scale, w2, w2_scale,
                shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale,
                output, recv_x, num_tokens,
            )
        ready = draft_ready
    else:
        with pl.spmd(pl.system.available_cluster_count(), sync_start=True, name_hint="moe_fused") as main_ready:
            output = _fused_experts(
                indices, x_norm_mx, x_norm_scale, weights,
                w1, w1_scale, w3, w3_scale, w2, w2_scale,
                shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale,
                output, recv_x, num_tokens,
            )
        ready = main_ready
    return ready


@pl.jit.inline(auto_scope=False)
def run_megamoe(
    indices: pl.Tensor,
    x_norm_mx: pl.Tensor,
    x_norm_scale: pl.Tensor,
    weights: pl.Tensor,
    output: pl.Tensor,
    w1: pl.Tensor,
    w1_scale: pl.Tensor[None, pl.FP8E8M0, pl.MX_B_NN],
    w3: pl.Tensor,
    w3_scale: pl.Tensor[None, pl.FP8E8M0, pl.MX_B_NN],
    w2: pl.Tensor,
    w2_scale: pl.Tensor[None, pl.FP8E8M0, pl.MX_B_NN],
    shared_w1: pl.Tensor,
    shared_w1_scale: pl.Tensor,
    shared_w3: pl.Tensor,
    shared_w3_scale: pl.Tensor,
    shared_w2: pl.Tensor,
    shared_w2_scale: pl.Tensor,
    recv_x: pld.DistributedTensor,
    recv_meta: pld.DistributedTensor,
    recv_routes: pld.DistributedTensor,
    data_arrived: pld.DistributedTensor,
    routed_output: pld.DistributedTensor,
    combine_arrived: pld.DistributedTensor,
    recv_scale: pld.DistributedTensor,
    recv_weights: pld.DistributedTensor,
    arrived: pld.DistributedTensor,
    num_tokens: pl.Scalar[pl.INT32],
    ep_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    local_experts: pl.constexpr = N_LOCAL_EXPERTS,
    topk: pl.constexpr = TOPK,
    dispatch_worker_limit: pl.constexpr = 0,
):
    return routed_moe(
        indices, x_norm_mx, x_norm_scale, weights, output,
        w1, w1_scale, w3, w3_scale, w2, w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale,
        recv_x, num_tokens, local_experts=local_experts, topk=topk,
    )


@pl.jit(auto_scope=False)
def local_compute_test(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.FP8E4M3FN],
    recv_mx_scale: pl.Tensor[[1, N_LOCAL_EXPERTS * RECV_MAX * K_SCALE], pl.FP8E8M0],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * H_SCALE, D], pl.FP8E8M0, pl.MX_B_NN],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[H_SCALE, D], pl.FP8E8M0, pl.MX_B_NN],
    routed_y: pl.InOut[pl.Tensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.BF16]],
    workspace: pld.DistributedTensor,
    rank: pl.Scalar[pl.INT32],
):
    """Validate expert rows through the production EP pipeline."""
    recv_x_flat = pl.reshape(recv_x, [N_LOCAL_EXPERTS * RECV_MAX, D])
    indices = pl.create_tensor([MOE_TOKENS, TOPK], dtype=pl.INT32)
    route_weights = pl.create_tensor([MOE_TOKENS, TOPK], dtype=pl.FP32)
    output = pl.create_tensor([MOE_TOKENS, D], dtype=pl.BF16)
    completed = pl.array.create(1, pl.TASK_ID)
    completed[0] = pl.system.task_dummy(deps=[])
    peer = (rank + 1) % EP_WORLD_SIZE
    for expert in pl.range(N_LOCAL_EXPERTS):
        count = pl.read(recv_expert_count, [expert, 0])
        for begin in pl.range(0, RECV_MAX, MOE_TOKENS):
            active = pl.max(0, pl.min(MOE_TOKENS, count - begin))
            source_row = expert * RECV_MAX + begin
            source = pl.slice(recv_x_flat, [MOE_TOKENS, D], [source_row, 0])
            source_scale = pl.slice(recv_mx_scale, [1, MOE_TOKENS * K_SCALE], [0, source_row * K_SCALE])
            with pl.spmd(1, name_hint="expert_test_routes", deps=[completed[0]]):
                writer = pl.tile.get_block_idx()
                for row in pl.range(writer, MOE_TOKENS):
                    for slot in pl.range(TOPK):
                        index = pl.cast(-1, pl.INT32)
                        weight = pl.cast(0.0, pl.FP32)
                        if row < active and slot == 0:
                            index = pl.cast(peer * N_LOCAL_EXPERTS + expert, pl.INT32)
                            weight = pl.read(recv_weights, [expert, begin + row])
                        pl.write(indices, [row, slot], index)
                        pl.write(route_weights, [row, slot], weight)
            down_ready = routed_moe(
                indices, source, source_scale, route_weights, output,
                routed_w1_packed, routed_w1_scale, routed_w3_packed, routed_w3_scale,
                routed_w2_packed, routed_w2_scale,
                shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale,
                workspace, active, local_experts=N_LOCAL_EXPERTS, topk=TOPK,
            )
            with pl.spmd(64, name_hint="ffn_copy", deps=[down_ready]) as copied:
                core = pl.tile.get_block_idx()
                for work in pl.range(core, (active + 1) // 2, 64):
                    local_row = work * 2
                    valid = pl.min(2, active - local_row)
                    result = pl.load(output, [local_row, 0], [2, D], valid_shape=[valid, D])
                    masked = pl.set_validshape(result, valid, D)
                    routed_y = pl.store(masked, [source_row + local_row, 0], routed_y)
            completed[0] = copied
    return routed_y


_LOCAL_COMPUTE_WINDOW_BYTES = fused_workspace_bytes(EP_WORLD_SIZE, N_LOCAL_EXPERTS, MOE_TOKENS, TOPK, D, MOE_INTER)


@pl.jit.host
def distributed_compute_test(
    recv_x: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS, RECV_MAX, D], pl.FP8E4M3FN],
    recv_mx_scale: pl.Tensor[[EP_WORLD_SIZE, 1, N_LOCAL_EXPERTS * RECV_MAX * K_SCALE], pl.FP8E8M0],
    recv_weights: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1_packed: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0],
    routed_w3_packed: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0],
    routed_w2_packed: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS * H_SCALE, D], pl.FP8E8M0],
    shared_w1: pl.Tensor[[EP_WORLD_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[EP_WORLD_SIZE, K_SCALE, MOE_INTER], pl.FP8E8M0],
    shared_w3: pl.Tensor[[EP_WORLD_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[EP_WORLD_SIZE, K_SCALE, MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[EP_WORLD_SIZE, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[EP_WORLD_SIZE, H_SCALE, D], pl.FP8E8M0],
    routed_y: pl.InOut[pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS * RECV_MAX, D], pl.BF16]],
):
    """Run guarded expert-row validation across EP peers."""
    storage = pld.alloc_window_buffer([_LOCAL_COMPUTE_WINDOW_BYTES], dtype=pl.UINT8)
    for rank in pl.range(pld.world_size()):
        workspace = pld.window(storage, [_LOCAL_COMPUTE_WINDOW_BYTES], dtype=pl.UINT8)
        w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = routed_w1_scale[rank]
        w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = routed_w3_scale[rank]
        w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * H_SCALE, D], pl.FP8E8M0, pl.MX_B_NN] = routed_w2_scale[rank]
        sw1_scale: pl.Tensor[[K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w1_scale[rank]
        sw3_scale: pl.Tensor[[K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w3_scale[rank]
        sw2_scale: pl.Tensor[[H_SCALE, D], pl.FP8E8M0, pl.MX_B_NN] = shared_w2_scale[rank]
        local_compute_test(
            recv_x[rank], recv_mx_scale[rank], recv_weights[rank], recv_expert_count[rank],
            routed_w1_packed[rank], w1_scale, routed_w3_packed[rank], w3_scale, routed_w2_packed[rank], w2_scale,
            shared_w1[rank], sw1_scale, shared_w3[rank], sw3_scale, shared_w2[rank], sw2_scale,
            routed_y[rank], workspace, rank, device=rank,
        )


def golden_distributed_compute(tensors):
    """Apply the unchanged guarded expert golden to every source rank."""
    for rank in range(EP_WORLD_SIZE):
        golden_local_compute({name: value[rank] for name, value in tensors.items()})


def compare_distributed_compute(actual, expected, *, inputs, **kwargs):
    """Check each source rank with the original expert-row comparator."""
    for rank in range(EP_WORLD_SIZE):
        local_inputs = {name: value[rank] for name, value in inputs.items()}
        passed, message = compare_local_compute(actual[rank], expected[rank], inputs=local_inputs, **kwargs)
        if not passed:
            return False, f"rank {rank}: {message}"
    return True, ""


def build_distributed_tensor_specs(case="bounds", *, tail_offset=0):
    """Build independent peer inputs, replicated expert weights, and neutral shared weights."""
    from golden import TensorSpec

    local_specs = [
        build_tensor_specs(case, tail_offset=tail_offset + rank, seed=20261028 + rank)
        for rank in range(EP_WORLD_SIZE)
    ]
    specs = []
    for index, local in enumerate(local_specs[0][:-1]):
        shape = [EP_WORLD_SIZE, *local.shape]
        resident = "stacked" if local.name.startswith("routed_") else None
        initializers = tuple(local_specs[rank][index].init_value for rank in range(EP_WORLD_SIZE))
        if resident:
            initializers = (local.init_value,) * EP_WORLD_SIZE

        def initialize(values=initializers):
            return torch.stack([value() if callable(value) else value for value in values])

        specs.append(TensorSpec(local.name, shape, local.dtype, init_value=initialize, resident=resident))
    for layer, rows, columns in (("w1", D, MOE_INTER), ("w3", D, MOE_INTER), ("w2", MOE_INTER, D)):
        shape = [EP_WORLD_SIZE, rows, columns]
        scale_shape = [EP_WORLD_SIZE, rows // MX_GROUP, columns]

        def unit_scale(shape=scale_shape):
            return torch.full(shape, 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)

        specs.append(TensorSpec("shared_" + layer, shape, torch.float8_e4m3fn, init_value=0, resident="stacked"))
        specs.append(TensorSpec("shared_" + layer + "_scale", scale_shape, torch.float8_e8m0fnu, init_value=unit_scale, resident="stacked"))
    specs.append(TensorSpec("routed_y", [EP_WORLD_SIZE, N_LOCAL_EXPERTS * RECV_MAX, D], torch.bfloat16, init_value=OUTPUT_SENTINEL))
    return specs


if __name__ == "__main__":
    from golden import run

    cases = ("bounds", "tails", "zero") if args.case == "all" else (args.case,)
    for case in cases:
        tail_rounds = (6 + N_LOCAL_EXPERTS - 1) // N_LOCAL_EXPERTS if case == "tails" else 1
        for tail_round in range(tail_rounds):
            print(f"Fused routed compute: case={case}, tail_round={tail_round}, ep={EP_WORLD_SIZE}, experts={N_LOCAL_EXPERTS}, stride={RECV_MAX}")
            result = run(
                fn=distributed_compute_test,
                specs=build_distributed_tensor_specs(case, tail_offset=tail_round * N_LOCAL_EXPERTS),
                golden_fn=golden_distributed_compute,
                golden_data=args.golden_data,
                save_data=args.save_data,
                compile_only=args.compile_only,
                runtime_dir=args.runtime_dir,
                config=dict(
                    dump_passes=args.dump_passes,
                    backend_type=args.backend_type,
                    execution_mode=args.execution_mode,
                    distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
                    enable_chip_swimlane=args.enable_chip_swimlane,
                ),
                rtol=1e-3,
                atol=1e-3,
                compare_fn={"routed_y": compare_distributed_compute},
            )
            if not result.passed:
                if result.error:
                    print(result.error)
                raise SystemExit(1)
            if args.compile_only or args.golden_data:
                break
        if args.compile_only or args.golden_data:
            break
