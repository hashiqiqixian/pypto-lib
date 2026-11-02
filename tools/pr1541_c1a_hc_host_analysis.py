"""Bounded host-only analysis of saved C1A HC inputs and observed BF16 output.

No device program runs and no formal reference or tolerance is replaced.
Modeled reduction trees are hypotheses until checked against device captures.
"""

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import platform
import struct
import sys


def fp32(code):
    return struct.unpack("!f", struct.pack("!I", code))[0]


def fp32_bits(value):
    return struct.unpack("!I", struct.pack("!f", value))[0]


def rounded_fp32(exact):
    seed = fp32_bits(float(exact))
    code = min(range(seed - 2, seed + 3), key=lambda item: (
        abs(Decimal.from_float(fp32(item)) - exact), item & 1,
    ))
    return fp32(code)


def nearest_bf16(exact):
    seed = fp32_bits(float(exact)) >> 16
    codes = range(max(0, seed - 2), min(65536, seed + 3))
    code = min(codes, key=lambda item: (
        abs(Decimal.from_float(fp32(item << 16)) - exact), item & 1,
    ))
    return {"value": fp32(code << 16), "bits": f"0x{code:04x}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--capture-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row", type=int, default=3)
    parser.add_argument("--threads", type=int, default=16, choices=range(1, 17))
    args = parser.parse_args()
    if not 0 <= args.row < 72:
        parser.error("--row must be in [0, 71]")
    for path, maximum in ((args.input, 16 * 1024**2), (args.actual, 1024**2),
                          (args.capture_report, 128 * 1024)):
        if path.stat().st_size > maximum:
            parser.error(f"Oversized diagnostic artifact: {path.name}")
    report = json.loads(args.capture_report.read_text(encoding="utf-8"))
    model = args.lib_root.resolve(strict=True) / "models" / "deepseek_v4_1_flash"
    for name in ("hc.py", "rmsnorm.py", "quant.py"):
        if hashlib.sha256((model / name).read_bytes()).hexdigest() != report["source_sha256"][name]:
            parser.error(f"Reference source differs from the captured revision: {name}")
    sys.path[:0] = [str(model), str(args.lib_root.resolve())]
    import torch

    from hc import golden_hc_collapse
    from quant import _quantize_mxfp8_activation
    from rmsnorm import EPS, golden_rms_norm

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    def sha256(value):
        raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    saved = torch.load(args.input, weights_only=True, map_location="cpu")
    inputs = saved["tensors"]
    expected = {"x_hc": ((72, 4, 5120), torch.float32), "pre_mix": ((72, 4), torch.float32),
                "attn_norm_weight": ((5120,), torch.bfloat16)}
    for name, (shape, dtype) in expected.items():
        value = inputs[name]
        if tuple(value.shape) != shape or value.dtype != dtype or not bool(torch.isfinite(value).all()):
            parser.error(f"Invalid input tensor: {name}")
        digest = sha256(value)
        if digest != saved["tensor_metadata"][name]["sha256"] or digest != report["input_sha256"][name]:
            parser.error(f"Input hash mismatch: {name}")
    actual = torch.load(args.actual, weights_only=True, map_location="cpu")
    if tuple(actual.shape) != (72, 5120) or actual.dtype != torch.bfloat16:
        parser.error("Expected a compact BF16 [72, 5120] actual HC output")
    if sha256(actual) != report["actual_sha256"] or not bool(torch.isfinite(actual).all()):
        parser.error("Actual output hash or finiteness mismatch")

    # Preserve the original 72-row reference shape: vector and scalar CPU
    # rsqrt kernels need not return identical FP32 results.
    collapse_pre = golden_hc_collapse(inputs["x_hc"], inputs["pre_mix"])
    collapsed = collapse_pre.bfloat16()
    golden = golden_rms_norm(collapsed, inputs["attn_norm_weight"])
    value = collapsed.float()
    full_mean = value.square().mean(dim=-1, keepdim=True)
    full_variance = full_mean + EPS
    full_inverse = torch.rsqrt(full_variance)
    row = args.row
    source, gamma = value[row], inputs["attn_norm_weight"].float()
    row_actual, row_golden = actual[row], golden[row]
    observed = (row_actual.view(torch.int16) != row_golden.view(torch.int16)).nonzero().flatten().tolist()
    recorded = [item["column"] for item in report.get("row3_first_differences", [])] if row == 3 else []
    columns = sorted(set(observed + recorded))
    if len(columns) > 64:
        parser.error("More than 64 focus differences; do not emit an unbounded report")

    def adjacent(values):
        while values.numel() > 1:
            values = values[0::2] + values[1::2]
        return values[0]

    def halves(values):
        while values.numel() > 1:
            half = values.numel() // 2
            values = values[:half] + values[half:]
        return values[0]

    def serial(values):
        total = torch.zeros((), dtype=torch.float32)
        for item in values:
            total = total + item
        return total

    def compensated(values):
        # Match _rms_norm_tiled(compensated=True) across the same ten partials.
        total = torch.zeros((), dtype=torch.float32)
        correction = torch.zeros((), dtype=torch.float32)
        for item in values:
            delta = item - correction
            next_sum = total + delta
            correction = (next_sum - total) - delta
            total = next_sum
        return total

    def reduction_models(row_value, original_mean):
        square = row_value.square()
        partial_sets = {"torch_512_chunks": [part.sum() for part in square.reshape(10, 512)]}
        for name, reduce64 in (("adjacent", adjacent), ("halves", halves)):
            partial_sets[f"modeled_TROWSUM64_{name}_then512_chunks"] = [
                serial([reduce64(chunk) for chunk in part.reshape(8, 64)])
                for part in square.reshape(10, 512)
            ]
        models = {"original_whole_mean": original_mean}
        for name, partials in partial_sets.items():
            models[name] = serial(partials) * (1.0 / 5120)
            models[name + "_compensated"] = compensated(partials) * (1.0 / 5120)
        return models

    all_models = [reduction_models(value[index], full_mean[index, 0]) for index in range(72)]
    means = all_models[row]

    variants = []
    truth = []
    with localcontext() as context:
        context.prec = 100
        exact_means = [sum((Decimal.from_float(item) ** 2 for item in values), Decimal(0)) / Decimal(5120)
                       for values in value.tolist()]
        for index, exact in enumerate(exact_means):
            all_models[index]["independent_mean_rounded_fp32"] = torch.tensor(rounded_fp32(exact), dtype=torch.float32)
        exact_mean = exact_means[row]
        exact_variance = exact_mean + Decimal.from_float(EPS)
        exact_inverse = Decimal(1) / exact_variance.sqrt()

        for name, mean in means.items():
            variance = mean + EPS
            # Keep the complete original shape for both CPU inverse paths.
            vector_variance = full_variance.clone()
            vector_variance[row, 0] = variance
            exact_var32 = Decimal.from_float(float(variance))
            sqrt32 = rounded_fp32(exact_var32.sqrt())
            two_step = rounded_fp32(Decimal(1) / Decimal.from_float(sqrt32))
            direct = rounded_fp32(Decimal(1) / exact_var32.sqrt())
            inverses = {
                "torch_rsqrt_vector72": torch.rsqrt(vector_variance)[row, 0],
                "torch_rsqrt_scalar": torch.rsqrt(variance),
                "torch_sqrt_div_vector72": (1 / torch.sqrt(vector_variance))[row, 0],
                "RNE_sqrt32_then_RNE_reciprocal32": torch.tensor(two_step, dtype=torch.float32),
                "RNE_direct_rsqrt32": torch.tensor(direct, dtype=torch.float32),
            }
            for inverse_name, inverse in inverses.items():
                pre = (source * inverse) * gamma
                normalized = pre.bfloat16()
                device_diff = normalized.view(torch.int16) != row_actual.view(torch.int16)
                golden_diff = normalized.view(torch.int16) != row_golden.view(torch.int16)
                variants.append({
                    "mean_method": name, "inverse_method": inverse_name,
                    "mean": float(mean), "mean_bits": f"0x{fp32_bits(float(mean)):08x}",
                    "mean_error_vs_independent": str(Decimal.from_float(float(mean)) - exact_mean),
                    "inverse": float(inverse), "inverse_bits": f"0x{fp32_bits(float(inverse)):08x}",
                    "correct_sqrt32_bits": f"0x{fp32_bits(sqrt32):08x}",
                    "row_differences_vs_actual": int(device_diff.sum()),
                    "row_differences_vs_original_golden": int(golden_diff.sum()),
                    "focus": [{"column": col, "pre_bf16": float(pre[col]),
                               "pre_bf16_bits": f"0x{fp32_bits(float(pre[col])):08x}",
                               "bf16": float(normalized[col]),
                               "matches_actual": bool(normalized[col] == row_actual[col])}
                              for col in columns],
                })

        # The exact real-valued collapse is an independent comparison, not
        # the kernel's specified four FP32 products and pairwise FP32 sums.
        x_row = inputs["x_hc"][row].double()
        p_row = inputs["pre_mix"][row].double()
        collapse64 = (x_row * p_row[:, None]).sum(dim=0)
        collapse64_rounded = collapse64.bfloat16()
        collapse_differences = (collapse64_rounded.view(torch.int16) != collapsed[row].view(torch.int16))
        for col in columns:
            exact_collapse = sum((Decimal.from_float(float(inputs["x_hc"][row, stream, col]))
                                  * Decimal.from_float(float(inputs["pre_mix"][row, stream]))
                                  for stream in range(4)), Decimal(0))
            exact_norm = Decimal.from_float(float(source[col])) * exact_inverse * Decimal.from_float(float(gamma[col]))
            actual_dec = Decimal.from_float(float(row_actual[col]))
            golden_dec = Decimal.from_float(float(row_golden[col]))
            midpoint = (actual_dec + golden_dec) / 2
            truth.append({
                "column": col, "source_collapsed_bf16": float(source[col]), "gamma_bf16": float(gamma[col]),
                "collapse_pairwise_fp32": float(collapse_pre[row, col]),
                "collapse_independent_decimal": str(exact_collapse),
                "collapse_independent_nearest_bf16": nearest_bf16(exact_collapse),
                "norm_independent_decimal_from_contracted_collapse": str(exact_norm),
                "norm_independent_nearest_bf16": nearest_bf16(exact_norm),
                "actual_bf16": float(row_actual[col]), "golden_bf16": float(row_golden[col]),
                "actual_golden_midpoint": str(midpoint),
                "independent_minus_midpoint": str(exact_norm - midpoint),
                "actual_error_vs_independent": str(actual_dec - exact_norm),
                "golden_error_vs_independent": str(golden_dec - exact_norm),
            })
        independent = {"mean_decimal": str(exact_mean), "mean_fp64": float(exact_mean),
                       "inverse_decimal": str(exact_inverse), "inverse_fp64": float(exact_inverse)}

    bit_diff = actual.view(torch.int16) != golden.view(torch.int16)
    golden_payload, golden_scales = _quantize_mxfp8_activation(golden, scale_alg=0)
    actual_payload, actual_scales = _quantize_mxfp8_activation(actual, scale_alg=0)
    all_row_statistics = {}
    with localcontext() as context:
        context.prec = 100
        for name in means:
            model_means = torch.stack([item[name] for item in all_models]).reshape(72, 1)
            errors = [float(Decimal.from_float(float(model_means[index, 0])) - exact_means[index])
                      for index in range(72)]
            ulps = [fp32_bits(float(model_means[index, 0])) - fp32_bits(rounded_fp32(exact_means[index]))
                    for index in range(72)]
            worst_row = max(range(72), key=lambda index: abs(errors[index]))
            inverse_values = []
            for variance in (model_means + EPS).flatten().tolist():
                sqrt32 = rounded_fp32(Decimal.from_float(variance).sqrt())
                inverse_values.append(rounded_fp32(Decimal(1) / Decimal.from_float(sqrt32)))
            model_inverse = torch.tensor(inverse_values, dtype=torch.float32).reshape(72, 1)
            prediction = ((value * model_inverse) * gamma).bfloat16()
            against_golden = prediction.view(torch.int16) != golden.view(torch.int16)
            against_actual = prediction.view(torch.int16) != actual.view(torch.int16)
            predicted_payload, predicted_scales = _quantize_mxfp8_activation(prediction, scale_alg=0)
            payload_vs_golden = (predicted_payload.view(torch.uint8) != golden_payload.view(torch.uint8)).reshape(72, -1)
            payload_vs_actual = (predicted_payload.view(torch.uint8) != actual_payload.view(torch.uint8)).reshape(72, -1)
            scale_vs_golden = predicted_scales != golden_scales
            scale_vs_actual = predicted_scales != actual_scales
            all_row_statistics[name] = {
                "rows": 72, "mean_abs_error_vs_independent": sum(abs(error) for error in errors) / 72,
                "max_abs_error_vs_independent": abs(errors[worst_row]), "worst_mean_row": worst_row,
                "correctly_rounded_mean_rows": sum(ulp == 0 for ulp in ulps),
                "max_mean_ulp_distance": max(abs(ulp) for ulp in ulps),
                "mean_ulp_deltas_by_row": ulps,
                "row3_mean_bits": f"0x{fp32_bits(float(model_means[3, 0])):08x}",
                "row3_inverse_bits": f"0x{fp32_bits(float(model_inverse[3, 0])):08x}",
                "inverse_contract": "RNE FP32 sqrt followed by RNE FP32 reciprocal; two FP32 output multiplies",
                "bf16_differences_vs_golden": int(against_golden.sum()),
                "bf16_differences_vs_actual": int(against_actual.sum()),
                "bf16_differences_vs_golden_by_row": against_golden.sum(dim=1).tolist(),
                "bf16_existing_differences_resolved": int((bit_diff & ~against_golden).sum()),
                "bf16_new_differences_vs_golden": int((~bit_diff & against_golden).sum()),
                "host_OCP_payload_differences_vs_golden": int(payload_vs_golden.sum()),
                "host_OCP_payload_differences_vs_actual": int(payload_vs_actual.sum()),
                "host_OCP_scale_differences_vs_golden": int(scale_vs_golden.sum()),
                "host_OCP_scale_differences_vs_actual": int(scale_vs_actual.sum()),
                "host_OCP_payload_differences_vs_golden_by_row": payload_vs_golden.sum(dim=1).tolist(),
                "row3_host_OCP_bad_columns_vs_golden": payload_vs_golden[3].nonzero().flatten()[:32].tolist(),
                "device_projection_quantizer_executed": False,
            }
    matching = [{"mean_method": item["mean_method"], "inverse_method": item["inverse_method"]}
                for item in variants if item["row_differences_vs_actual"] == 0]
    result = {
        "analysis": "saved_C1A_HC_host_rounding", "row": row, "focus_columns": columns,
        "cpu_arch": platform.machine(), "torch_version": str(torch.__version__),
        "torch_threads": torch.get_num_threads(), "recorded_arch": report["cpu_arch"],
        "input_metadata": saved["metadata"], "original_full_shape": [72, 5120],
        "actual_sha256_verified": sha256(actual), "host_golden_sha256": sha256(golden),
        "recorded_golden_sha256": report["golden_sha256"],
        "host_golden_matches_recorded_hash": sha256(golden) == report["golden_sha256"],
        "host_golden_vs_actual_elements": int(bit_diff.sum()),
        "host_golden_vs_actual_rows": bit_diff.any(dim=1).nonzero().flatten().tolist(),
        "original_mean_bits": f"0x{fp32_bits(float(full_mean[row, 0])):08x}",
        "original_inverse_bits": f"0x{fp32_bits(float(full_inverse[row, 0])):08x}",
        "independent": independent, "focus_independent_reference": truth, "variants": variants,
        "all_72_row_compensation_statistics": all_row_statistics,
        "compensation_scope": "Existing Kahan recurrence across the same ten 512-column partials only; no changes within each partial.",
        "collapse_fp64_torch_cast_vs_pairwise_bf16_differences": int(collapse_differences.sum()),
        "collapse_fp64_torch_cast_vs_pairwise_first_columns": collapse_differences.nonzero().flatten()[:32].tolist(),
        "RMS_models_reconstructing_entire_actual_row": matching,
        "attribution": ("At least one RMS model reconstructs the whole observed row from the original host collapse. "
                        "This supports an RMS boundary explanation but does not measure native collapse."
                        if matching else "No tested RMS model reconstructs the whole observed row; collapse versus RMS remains unresolved."),
        "limitations": ["Native collapse, mean, and inverse were not captured by the HC probe.",
                        "Modeled vcadd lane order is not assumed to be a hardware identity.",
                        "Whole-row FP64 collapse uses Torch's BF16 cast, which may double-round; focus nearest BF16 uses Decimal distances.",
                        "An independent real-valued reference does not replace the specified FP32 operation contract.",
                        "No device, production math, original Golden, or acceptance threshold is changed."],
    }
    encoded = json.dumps(result, allow_nan=False, indent=2)
    if len(encoded.encode()) > 256 * 1024:
        raise RuntimeError("Refusing an oversized diagnostic report")
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(encoded + "\n")
    print(json.dumps(result, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
