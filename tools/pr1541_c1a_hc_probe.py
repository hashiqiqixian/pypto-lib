# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Temporary fresh-process HC boundary diagnostic for the decode C1A CI fixture.

The exact BF16 diagnostic does not replace the original C1A acceptance test.
Only the existing HC kernel runs; subsequent FP8 quantization is host analysis.
"""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import sys


LIB_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = LIB_ROOT / "models" / "deepseek_v4_1_flash"
sys.path[:0] = [str(MODEL_ROOT), str(LIB_ROOT)]

import torch
import pypto.language as pl
from golden import TensorSpec, ratio_allclose, run
from hc import golden_hc_collapse, hc_head_norm
from quant import _quantize_mxfp8_activation
from rmsnorm import golden_rms_norm


class _BoundedNativeCompilePool(ThreadPoolExecutor):
    def __init__(self, max_workers=None, *args, **kwargs):
        super().__init__(min(4, max_workers if max_workers is not None else 4), *args, **kwargs)


def _bound_native_compile_pool():
    from pypto.runtime import device_runner

    tree = ast.parse(Path(device_runner.__file__).read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "_compile_and_assemble_locked"]
    if len(functions) != 1 or device_runner.ThreadPoolExecutor is not ThreadPoolExecutor:
        raise RuntimeError("Expected the unpatched 950a30e native compiler")
    calls = [node for node in ast.walk(functions[0]) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "ThreadPoolExecutor"]
    expected = ast.parse("ThreadPoolExecutor(max_workers=max_workers)", mode="eval").body
    if len(calls) != 1 or ast.dump(calls[0]) != ast.dump(expected):
        raise RuntimeError("Native compiler ThreadPoolExecutor call changed")
    device_runner.ThreadPoolExecutor = _BoundedNativeCompilePool


@pl.jit
def l2_c1a_hc_probe(
    x_hc: pl.Tensor[[72, 4, 5120], pl.FP32],
    pre_mix: pl.Tensor[[72, 4], pl.FP32],
    attn_norm_weight: pl.Tensor[[5120], pl.BF16],
    x_norm: pl.Out[pl.Tensor[[72, 5120], pl.BF16]],
):
    hc_head_norm(x_hc, pre_mix, attn_norm_weight, x_norm)
    return x_norm


def _sha256(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _golden(values):
    collapsed = golden_hc_collapse(values["x_hc"], values["pre_mix"]).bfloat16()
    values["x_norm"].copy_(golden_rms_norm(collapsed, values["attn_norm_weight"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a5",), default="a5")
    parser.add_argument("-d", "--device", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not 1 <= torch.get_num_threads() <= 16:
        parser.error("PYPTO_GOLDEN_NUM_THREADS must remain between 1 and 16")
    if args.input.stat().st_size > 16 * 1024**2:
        parser.error("The HC-only input artifact must be at most 16 MiB")
    saved = torch.load(args.input, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("metadata", {}).get("entry") != "decode_c1a_full.py":
        parser.error("Expected the decode C1A diagnostic input artifact")
    tensors = saved["tensors"]
    expected = {
        "x_hc": ((72, 4, 5120), torch.float32),
        "pre_mix": ((72, 4), torch.float32),
        "attn_norm_weight": ((5120,), torch.bfloat16),
        "hc_attn_fn": ((24, 20480), torch.float32),
        "hc_attn_scale": ((3,), torch.float32),
        "hc_attn_base": ((24,), torch.float32),
    }
    if set(tensors) != set(expected):
        parser.error("The input artifact must contain exactly the six HC tensors")
    input_hashes = {}
    for name, (shape, dtype) in expected.items():
        value = tensors[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
            parser.error(f"Unexpected tensor contract: {name}")
        if not bool(torch.isfinite(value).all()):
            parser.error(f"Nonfinite input: {name}")
        digest = _sha256(value)
        if saved["tensor_metadata"][name]["sha256"] != digest:
            parser.error(f"Input artifact hash mismatch: {name}")
        input_hashes[name] = digest
    output_dir = (args.output_dir or args.input.parent / "decode_hc_probe").resolve()
    if LIB_ROOT / "build_output" not in output_dir.parents:
        parser.error("--output-dir must be a new directory beneath this checkout's build_output")
    output_dir.mkdir(parents=True, exist_ok=False)
    _bound_native_compile_pool()
    context = {
        "diagnostic": "decode_c1a_full_HC_boundary",
        "input_metadata": saved["metadata"], "input_sha256": input_hashes,
        "cpu_arch": platform.machine(), "torch_version": str(torch.__version__),
        "torch_num_threads": torch.get_num_threads(), "native_compile_workers": 4,
        "source_sha256": {name: hashlib.sha256((MODEL_ROOT / name).read_bytes()).hexdigest()
                          for name in ("hc.py", "rmsnorm.py", "quant.py")},
        "original_C1A_acceptance_unchanged": True,
    }
    print(json.dumps(context, allow_nan=False, sort_keys=True), flush=True)
    exact = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)

    def compare(actual, expected, **kwargs):
        actual = actual.detach().cpu().contiguous()
        expected = expected.detach().cpu().contiguous()
        with (output_dir / "actual_x_norm.pt").open("xb") as stream:
            torch.save(actual.clone(), stream)
        bit_diff = actual.view(torch.int16) != expected.view(torch.int16)
        counts = bit_diff.sum(dim=1)
        columns = bit_diff[3].nonzero().flatten().tolist()
        finite = bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all())
        report = {
            **context, "output_shape": list(actual.shape), "output_dtype": str(actual.dtype),
            "actual_sha256": _sha256(actual), "golden_sha256": _sha256(expected),
            "finite": finite, "bf16_bitwise_different_elements": int(bit_diff.sum()),
            "different_rows": int((counts > 0).sum()), "row3_differences": len(columns),
            "row3_first_differences": [
                {"column": column, "actual": float(actual[3, column]), "golden": float(expected[3, column]),
                 "actual_bits": f"0x{int(actual.view(torch.int16)[3, column]) & 65535:04x}",
                 "golden_bits": f"0x{int(expected.view(torch.int16)[3, column]) & 65535:04x}"}
                for column in columns[:24] if finite
            ],
            "top_rows": [{"row": row, "different_elements": int(counts[row])}
                         for row in torch.argsort(counts, descending=True)[:10].tolist() if counts[row]],
            "device_projection_quantizer_executed": False,
        }
        if finite:
            report["max_abs"] = float((actual.float() - expected.float()).abs().max())
            a_payload, a_scale = _quantize_mxfp8_activation(actual, scale_alg=0)
            e_payload, e_scale = _quantize_mxfp8_activation(expected, scale_alg=0)
            payload_diff = a_payload.view(torch.uint8).reshape(72, -1) != e_payload.view(torch.uint8).reshape(72, -1)
            scale_diff = a_scale != e_scale
            report["host_OCP_quantization"] = {
                "payload_differences": int(payload_diff.sum()), "scale_differences": int(scale_diff.sum()),
                "row3_payload_differences": int(payload_diff[3].sum()),
                "row3_scale_differences": int(scale_diff[3].sum()),
                "row3_first_payload_columns": payload_diff[3].nonzero().flatten()[:32].tolist(),
                "row3_first_scale_groups": scale_diff[3].nonzero().flatten()[:32].tolist(),
                "actual_payload_sha256": _sha256(a_payload), "golden_payload_sha256": _sha256(e_payload),
                "actual_scale_sha256": _sha256(a_scale), "golden_scale_sha256": _sha256(e_scale),
            }
        with (output_dir / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, allow_nan=False, sort_keys=True)
        print(json.dumps(report, allow_nan=False, sort_keys=True), flush=True)
        return exact(actual, expected, **kwargs)

    specs_by_name = {
        name: TensorSpec(name, list(tensors[name].shape), tensors[name].dtype,
                         init_value=lambda name=name: tensors[name].clone())
        for name in ("x_hc", "pre_mix", "attn_norm_weight")
    }
    specs_by_name["x_norm"] = TensorSpec("x_norm", [72, 5120], torch.bfloat16)
    specs = [specs_by_name[name] for name in l2_c1a_hc_probe.param_names]
    result = run(fn=l2_c1a_hc_probe, specs=specs, golden_fn=_golden,
                 compare_fn={"x_norm": compare}, rtol=0.0, atol=0.0,
                 config={"backend_type": args.platform, "execution_mode": False, "device_id": args.device})
    print(result, flush=True)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
