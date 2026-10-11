"""Replay one captured SWA output projection before TP reduction."""

import argparse

import pypto.language as pl
import torch

from golden import ScalarSpec, TensorSpec, run
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.config import HEAD_DIM, LOCAL_H, LOCAL_O_WIDTH
from models.deepseek_v4_1_flash.decode_attn_swa import (
    official_grouped_o_a, official_linear, official_rope,
)
from models.deepseek_v4_1_flash.o_proj import _project_ob, _rotate_output, grouped_output
from probe_v41_window import golden_window, inputs_from_capture


@pl.jit
def probe_o_proj(
    attended: pl.Tensor[[C.T_DYN, C.LOCAL_H * C.HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    unrotated: pl.Out[pl.Tensor[[C.T_DYN, C.LOCAL_H * C.HEAD_DIM], pl.BF16]],
    latent: pl.Out[pl.Tensor[[C.T_DYN, C.LOCAL_O_WIDTH], pl.BF16]],
    partial: pl.Out[pl.Tensor[[C.T_DYN, C.D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(attended, 0)
    rotated_local = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    _rotate_output(attended, rope_cos, rope_sin, rotated_local, num_tokens)
    latent_local = pl.create_tensor([tokens, LOCAL_O_WIDTH], dtype=pl.BF16)
    grouped_output(rotated_local, wo_a, latent_local, num_tokens)
    _project_ob(latent_local, wo_b, wo_b_scale, partial, num_tokens)
    unrotated[:, :] = rotated_local
    latent[:, :] = latent_local
    return unrotated, latent, partial


def golden_o_proj(values):
    attended = values["attended"].unflatten(-1, (C.LOCAL_H, C.HEAD_DIM))
    unrotated = official_rope(attended, values["rope_cos"], values["rope_sin"], inverse=True)
    values.setdefault("unrotated", torch.empty_like(values["attended"]))
    values.setdefault("latent", torch.empty(
        values["attended"].shape[0], C.LOCAL_O_WIDTH, dtype=torch.bfloat16
    ))
    values["unrotated"].copy_(unrotated.flatten(-2))
    grouped = unrotated.reshape(-1, C.LOCAL_O_GROUPS, C.O_GROUP_IN)
    latent = official_grouped_o_a(grouped, values["wo_a"])
    values["latent"].copy_(latent.flatten(1))
    values["partial"].copy_(official_linear(
        latent.flatten(1), values["wo_b"], values["wo_b_scale"], fp32=True))


def compare(actual, expected, *, actual_outputs=None, inputs=None, **_):
    a, e = actual.float(), expected.float()
    rel_l2 = (a - e).norm() / e.norm().clamp_min(1e-12)
    row_l2 = (a - e).norm(dim=-1) / e.norm(dim=-1).clamp_min(1e-12)
    print(f"[O_PROJ] partial rel_l2={rel_l2.item():.8g} "
          f"max_row_rel_l2={row_l2.max().item():.8g} max_abs={(a-e).abs().max().item():.8g}")
    if actual_outputs is not None and actual.shape[-1] == C.LOCAL_O_WIDTH:
        grouped = actual_outputs["unrotated"].reshape(-1, C.LOCAL_O_GROUPS, C.O_GROUP_IN)
        for width in (256, 128, 64, 32):
            accumulated = torch.zeros(grouped.shape[0], C.LOCAL_O_GROUPS, C.O_LORA)
            for start in range(0, C.O_GROUP_IN, width):
                part = torch.einsum("tgd,grd->tgr",
                                    grouped[..., start:start + width].float(),
                                    inputs["wo_a"][..., start:start + width].float())
                accumulated += part
            chunked = accumulated.bfloat16().flatten(1)
            print(f"[O_PROJ] latent chunk_k={width} "
                  f"device_mismatch={(actual != chunked).sum().item()}")
        fp64 = torch.einsum("tgd,grd->tgr", grouped.double(),
                            inputs["wo_a"].double()).bfloat16().flatten(1)
        print(f"[O_PROJ] latent chunked_vs_full_fp64={(expected != fp64).sum().item()}")
        changed = expected != fp64
        mismatch = actual != expected
        print(f"[O_PROJ] latent FP64 changed={changed.sum().item()} "
              f"device_mismatch={mismatch.sum().item()} "
              f"resolved={(mismatch & (actual == fp64)).sum().item()} "
              f"new={(~mismatch & (actual != fp64)).sum().item()}")
    if actual_outputs is not None and "latent" in actual_outputs and actual.shape[-1] == C.D:
        conditioned = official_linear(actual_outputs["latent"], inputs["wo_b"],
                                      inputs["wo_b_scale"], fp32=True)
        conditioned_l2 = (a - conditioned).norm() / conditioned.norm().clamp_min(1e-12)
        print(f"[O_PROJ] partial conditioned_on_actual_latent rel_l2={conditioned_l2.item():.8g}")
    return bool(torch.isfinite(a).all() and rel_l2 <= 0.01), "relative L2 <= 1%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("--rank", type=int, default=0, choices=range(4))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    data = torch.load(args.capture, map_location="cpu", weights_only=True)
    captured = data["attention_inputs"]
    if args.cpu_only:
        partials = []
        for rank in range(C.TP_SIZE):
            window_values = inputs_from_capture(captured, rank)
            window_values["attended"] = torch.empty_like(window_values["q"])
            golden_window(window_values)
            source = {name: value[rank].contiguous() for name, value in captured.items()}
            values = {"attended": window_values["attended"], "wo_a": source["wo_a"],
                      "wo_b": source["wo_b"], "wo_b_scale": source["wo_b_scale"],
                      "rope_cos": source["rope_cos"], "rope_sin": source["rope_sin"],
                      "unrotated": torch.empty_like(window_values["attended"]),
                      "latent": torch.empty(window_values["q"].shape[0], C.LOCAL_O_WIDTH,
                                            dtype=torch.bfloat16),
                      "partial": torch.empty(window_values["q"].shape[0], C.D)}
            golden_o_proj(values)
            partials.append(values["partial"])
        reduced = sum(partials).bfloat16()
        for rank in range(C.TP_SIZE):
            start = rank * data["expected_attention_output"].shape[1]
            expected = data["expected_attention_output"][rank]
            actual = data["actual_attention_output"][rank]
            replay = reduced[start:start + expected.shape[0]]
            rel_ref = (replay.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
            rel_device = (replay.float() - actual.float()).norm() / expected.float().norm().clamp_min(1e-12)
            print(f"[CPU_REPLAY] rank={rank} reference_rel_l2={rel_ref.item():.8g} "
                  f"device_rel_l2={rel_device.item():.8g}")
        return
    window_values = inputs_from_capture(captured, args.rank)
    window_values["attended"] = torch.empty_like(window_values["q"])
    golden_window(window_values)
    source = {name: value[args.rank].contiguous() for name, value in captured.items()}
    values = {"attended": window_values["attended"], "wo_a": source["wo_a"],
              "wo_b": source["wo_b"], "wo_b_scale": source["wo_b_scale"],
              "rope_cos": source["rope_cos"], "rope_sin": source["rope_sin"]}
    specs = [TensorSpec(name, list(value.shape), value.dtype,
                        init_value=lambda value=value: value.clone()) for name, value in values.items()]
    tokens = values["attended"].shape[0]
    specs.extend((TensorSpec("unrotated", [tokens, C.LOCAL_H * C.HEAD_DIM], torch.bfloat16),
                  TensorSpec("latent", [tokens, C.LOCAL_O_WIDTH], torch.bfloat16),
                  TensorSpec("partial", [tokens, C.D], torch.float32),
                  ScalarSpec("num_tokens", torch.int32, tokens)))
    result = run(probe_o_proj, specs, golden_fn=golden_o_proj,
                 config=dict(platform="a5", device_id=args.device),
                 compare_fn={"unrotated": compare, "latent": compare, "partial": compare})
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
