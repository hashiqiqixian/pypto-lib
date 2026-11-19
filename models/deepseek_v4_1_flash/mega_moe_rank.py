# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Rank-local Ascend 950 MegaMoE using the model's packed experts and BF16 inputs.

Build an EP4 specialization on CPU with this module's CLI. Each HCCL rank
constructs one MegaMoERank on its current NPU stream with model-format
weights (including the physical MX_B_NN scale packing). Keep the weights and
their storage unchanged for the lifetime of the instance. Keep the instance
alive across calls. All four ranks enter construction and close collectively.
"""

import argparse
import ctypes as ct
import hashlib
import json
import os
import secrets
from pathlib import Path
import struct
import subprocess
import tempfile

_KEYS = tuple(
    f"{kind}_w{projection}{suffix}"
    for kind in ("routed", "shared")
    for projection in (1, 3, 2)
    for suffix in ("", "_scale")
)
_PTR = ct.c_void_p


class _LaunchArgs(ct.Structure):
    _fields_ = [
        ("tensors", ct.c_uint64 * 16),
        ("windows", ct.c_uint64 * 4),
        ("rank", ct.c_uint32),
        ("active", ct.c_uint32),
    ]


def _check(result, action):
    if result:
        raise RuntimeError(f"{action} failed with ACL status {result}")


def build(binary, capacity, hidden, intermediate, local_experts):
    """Compile externally to the device allocation using the active PyPTO ISA pin."""
    from moe_workspace import _pipeline_workspace_layout
    from pypto.runtime import pto_isa_include_dir

    if hidden <= 0 or intermediate <= 0 or hidden % 1024 or intermediate % 256:
        raise ValueError("Expected hidden divisible by 1024 and intermediate divisible by 256")
    compact_waves = (capacity, hidden, intermediate, local_experts) in (
        (4096, 4096, 3072, 24),
        (2048, 7168, 3072, 16),
    )
    shifted_scale = (capacity, hidden, intermediate, local_experts) == (8192, 4096, 2048, 16)
    window_bytes = _pipeline_workspace_layout(
        4, local_experts, capacity, 6, hidden, intermediate, ring_limit=32 if compact_waves else None
    )["bytes"]
    cfg = dict(
        ep=4,
        topk=6,
        capacity=capacity,
        hidden=hidden,
        intermediate=intermediate,
        local_experts=local_experts,
        window_bytes=window_bytes,
        interleave_w13=compact_waves,
    )
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    kern = Path(__file__).resolve().parent / "kernels"
    isa = Path(pto_isa_include_dir()).resolve() / "pto"
    binary = Path(binary).resolve()
    binary.parent.mkdir(parents=True, exist_ok=True)

    def contents(directory):
        return [
            dict(type="directory", name=p.name, contents=contents(p))
            if p.is_dir()
            else {"type": "file", "name": p.name, "external-contents": str(p)}
            for p in sorted(directory.iterdir())
        ]

    with tempfile.TemporaryDirectory(dir=binary.parent) as work:
        tmp = Path(work)
        tree = contents(isa)
        # Bisheng adds its bundled PTO headers ahead of -I for mixed kernels.
        roots = {cann / "include/pto", cann.resolve() / "include/pto", (cann / "include/pto").resolve()}
        toolkit_root = cann.resolve()
        if toolkit_root.name.endswith("-linux"):
            roots.add(toolkit_root.parent / "include/pto")
        overlay = dict(version=0, roots=[dict(type="directory", name=str(p), contents=tree) for p in roots])
        (tmp / "isa.json").write_text(json.dumps(overlay))
        macros = dict(
            MOE_EP=4,
            MOE_TOPK=6,
            MOE_EXPERTS=local_experts,
            MOE_TOKENS=capacity,
            MOE_HIDDEN=hidden,
            MOE_INTER=intermediate,
            MOE_BF16_INPUT=1,
        )
        for shift in (False, True) if shifted_scale else (False,):
            macros.update(MOE_SHIFTED_SCALE=int(shift), MOE_COMPACT_WAVES=int(compact_waves))
            subprocess.run(
                [
                    str(cann / "bin/bisheng"),
                    "-ivfsoverlay",
                    str(tmp / "isa.json"),
                    "--npu-arch=dav-3510",
                    "-O2",
                    "--cce-aicore-only",
                    "--cce-disable-vf-stack-reserved-ubuf",
                    "-std=c++17",
                    "-mllvm",
                    "-enable-hiipu-vf-loop-unroll",
                    "-mllvm",
                    "-cce-aicore-dcpreload-args=false",
                    "-DPTO_NPU_ARCH_A5",
                    *[f"-D{k}={v}" for k, v in macros.items()],
                    f"-I{isa.parent}",
                    f"-I{kern}",
                    f"-I{cann}/asc/include",
                    f"-I{cann}/asc",
                    str(kern / "moe_rank_entry.asc"),
                    "-o",
                    str(tmp / "entry.rel.o"),
                ],
                check=True,
            )
            subprocess.run(
                [
                    str(cann / "bin/ld.lld"),
                    "-m",
                    "aicorelinux",
                    "-Ttext",
                    "0",
                    "--no-mmap-output-file",
                    str(tmp / "entry.rel.o"),
                    "-o",
                    str(tmp / "entry.o"),
                ],
                check=True,
            )
            output = binary.with_suffix(".shifted.o") if shift else binary
            cfg["shifted_sha256" if shift else "sha256"] = hashlib.sha256(
                (tmp / "entry.o").read_bytes()
            ).hexdigest()
            (tmp / "entry.o").replace(output)
        (tmp / "shape.json").write_text(json.dumps(cfg))
        (tmp / "shape.json").replace(binary.with_suffix(".json"))
    return cfg


class MegaMoERank:
    def __init__(self, binary, group, weights, *, context_type=None):
        import torch
        import torch_npu
        import torch.distributed as dist

        if context_type is None:
            from cann_ops_transformer.ops.mc2.common.comm_context import CommContextManager as context_type
        binary = Path(binary)
        self.cfg = cfg = json.loads(binary.with_suffix(".json").read_text())
        if cfg["ep"] != 4 or dist.get_world_size(group) != 4 or cfg["topk"] != 6:
            raise ValueError("This specialization requires EP4 and TopK 6")
        if hashlib.sha256(binary.read_bytes()).hexdigest() != cfg["sha256"]:
            raise ValueError("MegaMoE binary and shape manifest differ")
        self.device = torch.npu.current_device()
        self.stream = torch.npu.current_stream()
        self.group = group
        self._weights = tuple(weights[k] for k in _KEYS)
        self.args = _LaunchArgs()
        self.args.rank = rank = dist.get_rank(group)
        for slot, (key, tensor) in enumerate(zip(_KEYS, self._weights), 4):
            routed = key.startswith("routed")
            down = "_w2" in key
            k, n = (cfg["intermediate"], cfg["hidden"]) if down else (cfg["hidden"], cfg["intermediate"])
            if key.endswith("_scale"):
                shape = ((cfg["local_experts"] if routed else 1) * k // 32, n)
                types = (torch.uint8, torch.float8_e8m0fnu)
            elif routed:
                shape = (cfg["local_experts"], k * n // 256, 128)
                types = (torch.uint8,)
            else:
                shape = (k, n)
                types = (torch.uint8, torch.float8_e4m3fn)
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype not in types
                or not tensor.is_contiguous()
                or tensor.device.type != "npu"
                or tensor.device.index != self.device
                or torch_npu.get_npu_format(tensor) not in (0, 2)
            ):
                raise ValueError(f"Invalid model-format weight {key}")
            self.args.tensors[slot] = tensor.data_ptr()
        if cfg.get("interleave_w13"):
            packed = list(self._weights)
            gate_slot, up_slot = _KEYS.index("routed_w1"), _KEYS.index("routed_w3")
            gate = packed[gate_slot].view(cfg["local_experts"], -1, 128)
            up = packed[up_slot].view_as(gate)
            for slot, half in ((gate_slot, slice(0, 64)), (up_slot, slice(64, 128))):
                packed[slot] = torch.cat((gate[..., half], up[..., half]), dim=-1).view_as(packed[slot])
                self.args.tensors[4 + slot] = packed[slot].data_ptr()
            self._weights = tuple(packed)
        # Advance only finite E8M0 codes whose +6 exponent remains finite.
        # Other checkpoints keep the original conversion and original bytes.
        if "shifted_sha256" in cfg:
            scale_slots = [
                i for i, key in enumerate(_KEYS) if key.startswith("routed") and key.endswith("_scale")
            ]
            safe = (
                torch.stack([self._weights[i].view(torch.uint8).max() for i in scale_slots]).max().item()
                <= 248
            )
            if safe:
                binary = binary.with_suffix(".shifted.o")
                if hashlib.sha256(binary.read_bytes()).hexdigest() != cfg["shifted_sha256"]:
                    raise ValueError("Shifted-scale binary and shape manifest differ")
                packed = list(self._weights)
                for i in scale_slots:
                    packed[i] = packed[i].view(torch.uint8) + 6
                    self.args.tensors[4 + i] = packed[i].data_ptr()
                self._weights = tuple(packed)
        self.acl = ct.CDLL("libascendcl.so")
        self.bin_handle = _PTR()
        self.function = _PTR()
        self.manager = None
        check = _check
        check(
            self.acl.aclrtBinaryLoadFromFile(os.fsencode(binary), None, ct.byref(self.bin_handle)),
            "binary load",
        )
        try:
            check(
                self.acl.aclrtBinaryGetFunction(self.bin_handle, b"lib_mega_moe", ct.byref(self.function)),
                "kernel lookup",
            )
            comm = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
            # HCCL identifies a window by group and operation tag. Give each
            # collective construction its own tag, including repeated opens.
            nonce = torch.tensor(
                [secrets.randbits(48) if rank == 0 else 0], device=self.stream.device, dtype=torch.int64
            )
            dist.broadcast(nonce, src=dist.get_global_rank(group, 0), group=group)
            self.manager = context_type(
                comm,
                4,
                backend="channel",
                opName=f"ptmoe{nonce.item():012x}",
                customCclBufferSize=cfg["window_bytes"],
            )
            context = self.manager.create_context()
            header = context.cpu().contiguous().numpy().tobytes()
            # Ascend 950 channel CommContext starts with rank, local rank count,
            # KFC context address and epHcclBuffer[HCCL_MAX_RANK_SIZE].
            actual_rank, local_count = struct.unpack_from("II", header)
            if actual_rank != rank or local_count < 4 or self.manager.ccl_buffer_size < cfg["window_bytes"]:
                raise RuntimeError("Incompatible HCCL communication context")
            addresses = struct.unpack_from("4Q", header, 16)
            local = self.manager.get_local_buffer_tensor(torch.uint8)
            if not all(addresses) or addresses[rank] != local.data_ptr():
                raise RuntimeError("Invalid HCCL window address table")
            self.args.windows[:] = addresses
            self.launch = self.acl.aclrtLaunchKernelWithHostArgs
            self.launch.argtypes = [_PTR, ct.c_uint32, _PTR, _PTR, _PTR, ct.c_size_t, _PTR, ct.c_size_t]
            self.stream_ptr = _PTR(self.stream.npu_stream)
        except BaseException:
            if self.manager is not None:
                self.manager.destroy()
            self.acl.aclrtBinaryUnLoad(self.bin_handle)
            raise

    def __call__(self, normalized, indices, weights, *, active=None, out=None):
        """Enqueue on the construction stream. Inputs have the specialization's padded row capacity."""
        import torch
        import torch_npu

        if self.manager is None:
            raise RuntimeError("MegaMoE context is closed")
        if torch.npu.current_stream(self.device).npu_stream != self.stream_ptr.value:
            raise ValueError("Use the stream owning this MegaMoE instance")
        rows = self.cfg["capacity"]
        if active is None:
            active = rows
        if not isinstance(active, int) or not 0 <= active <= rows:
            raise ValueError("Invalid active row count")
        supplied_output = out is not None
        if out is None:
            out = torch.empty_like(normalized)
        for i, (x, dtype, width) in enumerate(
            (
                (normalized, torch.bfloat16, self.cfg["hidden"]),
                (indices, torch.int32, 6),
                (weights, torch.float32, 6),
                (out, torch.bfloat16, self.cfg["hidden"]),
            )
        ):
            if (
                tuple(x.shape) != (rows, width)
                or x.dtype != dtype
                or x.device.type != "npu"
                or x.device.index != self.device
                or not x.is_contiguous()
                or torch_npu.get_npu_format(x) not in (0, 2)
            ):
                raise ValueError(f"Invalid MegaMoE tensor at slot {i}")
            x.record_stream(self.stream)
            self.args.tensors[i] = x.data_ptr()
        if supplied_output and any(
            torch._C._overlaps(out, t) for t in (normalized, indices, weights, *self._weights)
        ):
            raise ValueError("Output must not alias inputs")
        if self.cfg.get("interleave_w13") and active < rows:
            # The compact wave schedule uses one physical token capacity on
            # every rank. Inactive rows contribute neither shared nor routed output.
            padded_ids = indices.remainder(4 * self.cfg["local_experts"])
            padded = [torch.zeros_like(normalized), padded_ids, torch.zeros_like(weights)]
            for slot, (source, target) in enumerate(zip((normalized, indices, weights), padded)):
                target[:active].copy_(source[:active])
                self.args.tensors[slot] = target.data_ptr()
            active = rows
        # Flush torch_npu's host enqueue queue before a direct ACL launch.
        # This preserves stream order with preceding Torch producers, including
        # the padded copies above, without waiting for device completion.
        stream_ptr = _PTR(torch_npu._C._npu_getCurrentRawStream(self.device))
        self.args.active = active
        _check(
            self.launch(
                self.function, 32, stream_ptr, None, ct.byref(self.args), ct.sizeof(self.args), None, 0
            ),
            "MegaMoE launch",
        )
        return out

    def close(self):
        if self.manager is None:
            return
        import torch.distributed as dist

        self.stream.synchronize()
        dist.barrier(group=self.group)
        self.manager.destroy()
        self.manager = None
        _check(self.acl.aclrtBinaryUnLoad(self.bin_handle), "binary unload")
        self._weights = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build an EP4 Ascend 950 MegaMoE specialization")
    for field in ("capacity", "hidden", "intermediate", "local-experts"):
        p.add_argument("--" + field, type=int, required=True)
    p.add_argument("--binary", type=Path, required=True)
    build(**vars(p.parse_args()))
