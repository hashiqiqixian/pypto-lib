"""Validate one V4.1 SWA Block with serving's real checkpoint weight ABI."""

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from golden import TensorSpec
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash import decode_attn_swa, decode_common, decode_layer


def real_weight_specs(checkpoint, layer_id, seed, *, prepared=None, save_prepared=None):
    from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology
    from pypto_serving.model.deepseek_v41.swa_weights import load_swa_layer_weights

    checkpoint = str(Path(checkpoint).resolve())
    if prepared is None:
        attention, moe = load_swa_layer_weights(
            checkpoint, layer_id, SegmentTopology(tp=C.TP_SIZE, dp=C.EP_SIZE // C.TP_SIZE)
        )
        weights = dict(attention)
        weights.update({"ffn_norm_weight" if key == "norm_weight" else key: value
                        for key, value in moe.items()})
    else:
        bundle = torch.load(prepared, map_location="cpu", weights_only=True)
        if bundle["checkpoint"] != checkpoint or bundle["layer_id"] != layer_id:
            raise ValueError("prepared weight bundle has a different checkpoint or layer")
        weights = bundle["weights"]
    specs = decode_layer.build_tensor_specs(layer_id, seed)
    names = {spec.name for spec in specs if isinstance(spec, TensorSpec)}
    missing = sorted(set(weights) - names)
    if missing:
        raise ValueError(f"checkpoint loader returned unknown lib ABI names: {missing}")
    result = []
    for spec in specs:
        if not isinstance(spec, TensorSpec) or spec.name not in weights:
            result.append(spec)
            continue
        value = weights[spec.name]
        if tuple(value.shape) != tuple(spec.shape) or value.dtype != spec.dtype:
            raise ValueError(
                f"{spec.name}: checkpoint {tuple(value.shape)} {value.dtype} "
                f"!= lib {tuple(spec.shape)} {spec.dtype}"
            )
        result.append(replace(spec, init_value=lambda value=value: value.clone()))
    print(f"[REAL WEIGHTS] layer={layer_id} bound={len(weights)} ABI tensors", flush=True)
    if save_prepared is not None:
        target = Path(save_prepared)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        torch.save({"checkpoint": checkpoint, "layer_id": layer_id, "weights": weights}, temporary)
        temporary.replace(target)
        print(f"[REAL WEIGHTS] saved prepared bundle: {target}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--layer-id", type=int, default=0, choices=(0, 1))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--save-prepared", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--device", default=",".join(map(str, range(C.EP_SIZE))))
    parser.add_argument("--ring-heap-mib", type=int, default=1024)
    parser.add_argument("--enable-scope-stats", action="store_true")
    args = parser.parse_args()
    if args.save_prepared is not None and (args.prepared is not None or not args.prepare_only):
        parser.error("--save-prepared requires --prepare-only and cannot combine with --prepared")
    specs = real_weight_specs(
        args.checkpoint, args.layer_id, args.seed,
        prepared=args.prepared, save_prepared=args.save_prepared,
    )
    if args.prepare_only:
        return

    from golden import ratio_allclose, ratio_reldiff, run
    from pypto.ir import DistributedConfig

    devices = [int(value) for value in args.device.split(",")]
    if len(devices) != C.EP_SIZE or len(set(devices)) != C.EP_SIZE:
        parser.error(f"need {C.EP_SIZE} distinct A5 devices")
    if args.ring_heap_mib < 1 or args.ring_heap_mib & (args.ring_heap_mib - 1):
        parser.error("--ring-heap-mib must be a positive power of two")
    counts = torch.full((C.EP_SIZE,), C.MOE_TOKENS, dtype=torch.int32)
    result = run(
        fn=decode_layer.l3_decode_layer,
        specs=specs,
        golden_fn=decode_layer.golden_l3_decode_layer,
        config=dict(
            platform="a5",
            ring_heap=(0, 0, 0, args.ring_heap_mib << 20),
            enable_scope_stats=args.enable_scope_stats,
            distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
        ),
        compare_fn={
            "window_cache": decode_attn_swa.compare_distributed_cache,
            "window_cache_scale": decode_attn_swa.compare_scales,
            "compressed_cache": decode_common.compare_unchanged("compressed_cache"),
            "compressed_cache_scale": decode_common.compare_unchanged("compressed_cache_scale"),
            "index_cache": decode_common.compare_unchanged("index_cache"),
            "index_cache_scale": decode_common.compare_unchanged("index_cache_scale"),
            "state_cache": decode_common.compare_unchanged("state_cache"),
            "topk_indices": decode_common.compare_unchanged("topk_indices"),
            "candidate_mask": decode_common.compare_unchanged("candidate_mask"),
            "attention_output": decode_layer._compare_active_rows_per_rank(
                decode_attn_swa.compare_output, counts,
            ),
            "attention_hidden": decode_layer._compare_active_rows_per_rank(
                decode_attn_swa.compare_output, counts,
            ),
            "attention_pre_mix": ratio_allclose(atol=1e-4, rtol=1e-4),
            "gathered": decode_common.compare_attention_gather,
            "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3, max_error_ratio=0.03),
            "x_mixed": decode_layer._compare_active_rows_per_rank(
                ratio_reldiff(diff_thd=0.01, pct_thd=0.05), counts,
            ),
            "x_next": decode_layer._compare_active_rows_per_rank(
                ratio_reldiff(diff_thd=0.01, pct_thd=0.05), counts,
            ),
        },
    )
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
