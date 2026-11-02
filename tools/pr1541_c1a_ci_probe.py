# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Temporary C1A CI diagnostics; preserve the original entry and validation result."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import traceback


HC_NAMES = ("x_hc", "pre_mix", "attn_norm_weight", "hc_attn_fn", "hc_attn_scale", "hc_attn_base")
MAX_REPORT_BYTES = 100_000
MAX_HC_BYTES = 16 * 1024 * 1024


def _sha256(tensor):
    import torch

    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    for offset in range(0, raw.numel(), 1024 * 1024):
        digest.update(raw[offset:offset + 1024 * 1024].numpy().tobytes())
    return digest.hexdigest()


def _metadata(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": _sha256(tensor)}


def _difference(actual, expected):
    import torch

    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("diagnostic comparison requires identical shape and dtype")
    left = actual.detach().cpu().contiguous().reshape(-1)
    right = expected.detach().cpu().contiguous().reshape(-1)
    left_bytes, right_bytes = left.view(torch.uint8), right.view(torch.uint8)
    xor = torch.bitwise_xor(left_bytes, right_bytes)
    bit_counts = torch.tensor([value.bit_count() for value in range(256)], dtype=torch.int32)
    different = left != right
    indices = torch.nonzero(different).flatten()
    return {
        "elements": left.numel(), "different_elements": int(different.sum()),
        "different_bytes": int(torch.count_nonzero(xor)),
        "different_bits": int(bit_counts[xor.long()].sum()),
        "max_abs": float((left.float() - right.float()).abs().max()) if left.numel() else 0.0,
        "first": [{"flat_index": int(index), "actual": float(left[index]),
                   "reference": float(right[index])} for index in indices[:8].tolist()],
    }


def _write_json(path, report):
    encoded = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(encoded) >= MAX_REPORT_BYTES:
        raise ValueError(f"diagnostic report exceeds {MAX_REPORT_BYTES} bytes")
    with path.open("xb") as stream:
        stream.write(encoded)


def _prefill_report(actual, expected, inputs, report):
    import torch
    from compressor import golden_c1a_compressor
    from qkv_proj_rope import golden_rope_interleave
    from quant import quantize_mxfp4_cache

    x_actual, x_expected = actual["attn_input"], expected["attn_input"]
    if x_actual.ndim != 3 or x_actual.shape[-1] != 5120 or x_actual.shape[1] > 128:
        raise ValueError(f"unsupported prefill diagnostic shape {tuple(x_actual.shape)}")
    active = int(inputs["num_tokens"])
    if not 0 <= active <= x_actual.shape[1]:
        raise ValueError("invalid active-token count")
    report["attn_input"] = {"actual": _metadata(x_actual), "golden": _metadata(x_expected),
                            "difference": _difference(x_actual, x_expected)}
    report["compressed_boundary"] = ranks = []
    for rank in range(x_actual.shape[0]):
        slots = inputs["compressed_slots"][rank, :active].long()
        valid = slots >= 0
        selected = slots[valid]
        if selected.unique().numel() != selected.numel():
            raise ValueError("diagnostic requires unique compressed slots")
        cosine = inputs["compressed_rope_cos"][rank, :active]
        sine = inputs["compressed_rope_sin"][rank, :active]

        def recompute(source):
            latent = golden_c1a_compressor(
                source, inputs["compressor_wkv"][rank], inputs["compressor_norm_weight"][rank]
            )
            tail_width = cosine.shape[-1] * 2
            tail = golden_rope_interleave(latent[..., -tail_width:], cosine, sine)
            rotated = torch.cat((latent[..., :-tail_width], tail), dim=-1)
            payload, scale = quantize_mxfp4_cache(
                rotated[valid], group_size=16, scale_format="bf16", round_to_even=True
            )
            return latent, payload, scale

        original = recompute(x_expected[rank, :active])
        forced = recompute(x_actual[rank, :active])
        rank_report = {"rank": rank, "token_rows": torch.arange(active)[valid].tolist(),
                       "physical_slots": selected.tolist(),
                       "latent_actual_input_vs_original_input": _difference(forced[0], original[0])}
        for name, index in (("compressed_cache", 1), ("compressed_cache_scale", 2)):
            width = actual[name].shape[-1]
            pool = actual[name][rank].reshape(-1, width)
            if selected.numel() and int(selected.max()) >= pool.shape[0]:
                raise ValueError("compressed slot exceeds the actual pool")
            device_rows = pool[selected]
            golden_rows = expected[name][rank].reshape(-1, width)[selected]
            rank_report[name] = {
                "actual_vs_original_golden": _difference(device_rows, golden_rows),
                "recomputed_original_vs_original_golden": _difference(original[index], golden_rows),
                "actual_vs_actual_input_reference": _difference(device_rows, forced[index]),
                "actual_input_reference_vs_original": _difference(forced[index], original[index]),
            }
        ranks.append(rank_report)


def _decode_report(actual, expected, inputs, report, directory):
    import torch

    left, right = actual["topk_indices"], expected["topk_indices"]
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] > 128:
        raise ValueError("unsupported decode TopK diagnostic shape")
    rows = []
    for row, (a, b) in enumerate(zip(left.tolist(), right.tolist())):
        a_set, b_set = {value for value in a if value >= 0}, {value for value in b if value >= 0}
        rows.append({"row": row, "position_differences": sum(x != y for x, y in zip(a, b)),
                     "actual_only_count": len(a_set - b_set), "golden_only_count": len(b_set - a_set),
                     "actual_only_first": sorted(a_set - b_set)[:8],
                     "golden_only_first": sorted(b_set - a_set)[:8],
                     "actual_duplicates": sum(value >= 0 for value in a) - len(a_set),
                     "golden_duplicates": sum(value >= 0 for value in b) - len(b_set)})
    report["topk_rows"] = rows
    size = sum(inputs[name].numel() * inputs[name].element_size() for name in HC_NAMES)
    if size > MAX_HC_BYTES:
        raise ValueError(f"HC inputs exceed {MAX_HC_BYTES} bytes: {size}")
    tensors = {name: inputs[name].detach().cpu().contiguous().clone() for name in HC_NAMES}
    artifact = {"metadata": report["environment"], "tensors": tensors,
                "tensor_metadata": {name: _metadata(value) for name, value in tensors.items()}}
    path = directory / "decode_hc_inputs.pt"
    with path.open("xb") as stream:
        torch.save(artifact, stream)
    report["hc_inputs"] = {"file": path.name, "tensor_bytes": size,
                           "tensor_metadata": artifact["tensor_metadata"]}


def _run_hc_probe(root, directory, device):
    artifact = directory / "decode_hc_inputs.pt"
    if not artifact.is_file():
        return
    helper = root / "tools" / "pr1541_c1a_hc_probe.py"
    status = {"helper": helper.name, "timeout_seconds": 180}
    try:
        with (directory / "hc_probe.log").open("xb") as log:
            result = subprocess.run(
                [sys.executable, str(helper), "-p", "a5", "-d", str(device), "--input", str(artifact)],
                cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=False,
            )
        status["exit_code"] = result.returncode
    except Exception as error:
        status["error"] = {"type": type(error).__name__, "detail": str(error)[:2000]}
    print("[C1A-CI-PROBE] HC subprocess " + json.dumps(status, sort_keys=True), flush=True)
    try:
        _write_json(directory / "hc_probe_status.json", status)
    except Exception as error:
        print(f"[C1A-CI-PROBE] HC status write error: {error!r}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entry", type=Path)
    parser.add_argument("-p", "--platform", choices=["a5"], default="a5")
    parser.add_argument("-d", "--device", type=int, required=True)
    args, remainder = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    model = root / "models" / "deepseek_v4_1_flash"
    entry = args.entry.resolve(strict=True)
    if entry.parent != model or entry.name not in ("prefill_c1a_full.py", "decode_c1a_full.py"):
        parser.error("entry must be this checkout's prefill_c1a_full.py or decode_c1a_full.py")
    if args.device < 0:
        parser.error("device must be non-negative")
    directory = root / "build_output" / "pr1541_c1a_ci_probe"
    directory.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(model), str(root)]
    sys.argv = [str(entry), "-p", args.platform, "-d", str(args.device), *remainder]
    import torch

    runner = importlib.import_module("golden.runner")
    original_validate = runner.validate_golden
    saved_hc = False

    def checked_validate(actual_outputs, expected_outputs, **kwargs):
        nonlocal saved_hc
        report = {}
        try:
            sources = (entry, model / "compressor.py", model / "kernels" / "compressor_publish.cpp")
            report["environment"] = {
                "entry": entry.name, "argv": sys.argv[1:], "cpu_arch": platform.machine(),
                "torch_version": str(torch.__version__), "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
                "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sources},
            }
            inputs = kwargs["inputs"]
            names = (*HC_NAMES, "compressor_wkv", "compressor_norm_weight", "compressed_rope_cos",
                     "compressed_rope_sin", "compressed_slots")
            report["input_hashes"] = {name: _metadata(inputs[name]) for name in names if name in inputs}
            report["original_golden_hashes"] = {name: _metadata(value) for name, value in expected_outputs.items()}
            report["actual_output_hashes"] = {name: _metadata(value) for name, value in actual_outputs.items()}
            if entry.name == "prefill_c1a_full.py":
                _prefill_report(actual_outputs, expected_outputs, inputs, report)
            else:
                _decode_report(actual_outputs, expected_outputs, inputs, report, directory)
                saved_hc = True
        except Exception as error:
            report["diagnostic_error"] = {"type": type(error).__name__, "detail": str(error)[:2000],
                                          "traceback": traceback.format_exc()[-6000:]}
        try:
            path = directory / f"{entry.stem}.json"
            _write_json(path, report)
            print(f"[C1A-CI-PROBE] report={path} diagnostic_error={report.get('diagnostic_error')}", flush=True)
        except Exception as error:
            print(f"[C1A-CI-PROBE] report write error: {error!r}", flush=True)
        # No diagnostic result changes the original data, comparator mapping, or exception.
        return original_validate(actual_outputs, expected_outputs, **kwargs)

    runner.validate_golden = checked_validate
    entry_exit = {"exit_code": 0}
    try:
        runpy.run_path(str(entry), run_name="__main__")
    except SystemExit as error:
        entry_exit = {"exit_code": 0 if error.code is None else error.code if isinstance(error.code, int) else 1,
                      "system_exit_detail": str(error.code)[:2000]}
        raise
    except BaseException as error:
        entry_exit = {"exception_type": type(error).__name__, "detail": str(error)[:2000]}
        raise
    finally:
        runner.validate_golden = original_validate
        try:
            _write_json(directory / f"{entry.stem}_exit.json", entry_exit)
        except Exception as error:
            print(f"[C1A-CI-PROBE] entry status write error: {error!r}", flush=True)
        if saved_hc:
            try:
                _run_hc_probe(root, directory, args.device)
            except Exception as error:
                print(f"[C1A-CI-PROBE] HC subprocess diagnostic error: {error!r}", flush=True)


if __name__ == "__main__":
    main()
