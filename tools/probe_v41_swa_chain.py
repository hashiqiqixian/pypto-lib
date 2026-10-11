"""Tap every stage of one captured SWA leaf without the TP collective."""

import argparse
import os

import pypto.language as pl
import torch

from golden import ScalarSpec, TensorSpec, run
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.config import ORI_BLOCKS_DYN, T_DYN
from models.deepseek_v4_1_flash.decode_attn_swa import (
    attend_window, gather_window, official_quantize, publish_window,
)
from models.deepseek_v4_1_flash.o_proj import o_proj
from models.deepseek_v4_1_flash.qkv_proj_rope import qkv_proj_rope
from models.deepseek_v4_1_flash.quantization import decode_e8m0
from probe_v41_o_proj import golden_o_proj
from probe_v41_qkv import golden_qkv
from probe_v41_window import golden_window


@pl.jit
def probe_chain(
    x: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
    wq_a: pl.Tensor[[C.D, C.Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[C.D // 32, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[C.Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[C.D // 32, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[C.T_DYN], pl.INT64],
    window_indices: pl.Tensor[[C.T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 32], pl.FP8E8M0]],
    qr: pl.Out[pl.Tensor[[C.T_DYN, C.Q_LORA], pl.BF16]],
    q: pl.Out[pl.Tensor[[C.T_DYN, C.LOCAL_H * C.HEAD_DIM], pl.BF16]],
    kv: pl.Out[pl.Tensor[[C.T_DYN, C.HEAD_DIM], pl.BF16]],
    selected: pl.Out[pl.Tensor[[C.T_DYN, 128, C.HEAD_DIM], pl.BF16]],
    attended: pl.Out[pl.Tensor[[C.T_DYN, C.LOCAL_H * C.HEAD_DIM], pl.BF16]],
    partial: pl.Out[pl.Tensor[[C.T_DYN, C.D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    qkv_proj_rope(x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
                  wkv, wkv_scale, kv_norm_weight, rope_cos, rope_sin,
                  qr, q, kv, num_tokens)
    cache_ready = pl.system.task_dummy(deps=[])
    publish_window(kv, window_slots, window_cache, window_cache_scale, num_tokens, cache_ready)
    gather_window(window_cache, window_cache_scale, window_indices, selected, num_tokens)
    attend_window(q, selected, window_indices, attn_sink, attended, num_tokens)
    o_proj(attended, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, partial, num_tokens)
    return qr, q, kv, selected, attended, partial, window_cache, window_cache_scale


def golden_chain(values):
    golden_qkv(values)
    slots = values["window_slots"].long()
    valid = slots >= 0
    payload, codes = official_quantize(values["kv"])
    cache = values["window_cache"].view(torch.uint8).reshape(-1, C.HEAD_DIM)
    scales = values["window_cache_scale"].view(torch.uint8).reshape(-1, C.HEAD_DIM // 32)
    cache[slots[valid]] = payload.view(torch.uint8)[valid]
    scales[slots[valid]] = codes[valid]
    decoded = values["window_cache"].float() * decode_e8m0(
        values["window_cache_scale"]).repeat_interleave(32, -1)
    indices = values["window_indices"]
    selected = decoded.reshape(-1, C.HEAD_DIM)[indices.long().clamp_min(0)].bfloat16()
    values["selected"].copy_(selected.masked_fill((indices < 0)[..., None], 0))
    values["indices"] = indices
    values["sink"] = values["attn_sink"]
    golden_window(values)
    golden_o_proj(values)


def compare(name):
    def check(actual, expected, *, actual_outputs, inputs, **_):
        if name == "partial" and os.environ.get("V41_PARTIAL_DUMP"):
            torch.save({"actual": actual.cpu(), "expected": expected.cpu()},
                       os.environ["V41_PARTIAL_DUMP"])
        if name == "attended" and os.environ.get("V41_ATTENDED_DUMP"):
            torch.save({
                "actual": actual.cpu(), "expected": expected.cpu(),
                "q": actual_outputs["q"].cpu(), "selected": actual_outputs["selected"].cpu(),
                "indices": inputs["window_indices"].cpu(), "sink": inputs["attn_sink"].cpu(),
            }, os.environ["V41_ATTENDED_DUMP"])
        a, e = actual.float(), expected.float()
        rel_l2 = (a-e).norm() / e.norm().clamp_min(1e-12)
        mismatches = (actual != expected).sum().item()
        print(f"[CHAIN] {name} rel_l2={rel_l2.item():.8g} "
              f"mismatches={mismatches}/{actual.numel()} max_abs={(a-e).abs().max().item():.8g}")
        if name == "partial":
            conditioned = dict(inputs, attended=actual_outputs["attended"],
                               partial=torch.empty_like(actual))
            golden_o_proj(conditioned)
            rel = (a-conditioned["partial"]).norm() / conditioned["partial"].norm().clamp_min(1e-12)
            print(f"[CHAIN] partial conditioned_on_actual_attended rel_l2={rel.item():.8g}")
            conditioned = dict(inputs, q=actual_outputs["q"],
                               selected=actual_outputs["selected"],
                               indices=inputs["window_indices"], sink=inputs["attn_sink"],
                               attended=torch.empty_like(actual_outputs["attended"]),
                               partial=torch.empty_like(actual))
            golden_window(conditioned)
            golden_o_proj(conditioned)
            rel = (a-conditioned["partial"]).norm() / conditioned["partial"].norm().clamp_min(1e-12)
            print(f"[CHAIN] partial conditioned_on_actual_selected rel_l2={rel.item():.8g}")
        return bool(torch.isfinite(a).all() and rel_l2 <= 0.01), "relative L2 <= 1%"
    return check


def compare_cache(name):
    def check(actual, expected, *, actual_outputs, expected_outputs, inputs, **_):
        a = actual.contiguous().view(torch.uint8)
        e = expected.contiguous().view(torch.uint8)
        mismatches = (a != e).sum().item()
        print(f"[CHAIN] {name} byte_mismatch={mismatches}/{a.numel()} (diagnostic)")
        if mismatches <= 16:
            for index in torch.nonzero(a != e, as_tuple=False).tolist():
                key = tuple(index)
                print(f"[CHAIN] {name} mismatch_at={key} actual=0x{int(a[key]):02x} "
                      f"expected=0x{int(e[key]):02x}")
                if name == "window_cache":
                    row = index[0] * 128 + index[1]
                    token_rows = torch.nonzero(inputs["window_slots"] == row).flatten().tolist()
                    for token in token_rows:
                        feature = index[-1]
                        actual_kv = actual_outputs["kv"][token, feature]
                        expected_kv = expected_outputs["kv"][token, feature]
                        print(f"[CHAIN] cache_source token={token} feature={feature} "
                              f"actual_kv={actual_kv.item():.9g} expected_kv={expected_kv.item():.9g}")
        return True, "cache is diagnostic here"
    return check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("--rank", type=int, default=0, choices=range(4))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    captured = torch.load(args.capture, map_location="cpu", weights_only=True)["attention_inputs"]
    from models.deepseek_v4_1_flash.decode_attn_swa import INPUT_NAMES
    values = {name: captured[name][args.rank].contiguous() for name in INPUT_NAMES}
    tokens = values["x"].shape[0]
    specs = [TensorSpec(name, list(value.shape), value.dtype,
                        init_value=lambda value=value: value.clone()) for name, value in values.items()]
    specs.extend((
        TensorSpec("qr", [tokens, C.Q_LORA], torch.bfloat16),
        TensorSpec("q", [tokens, C.LOCAL_H * C.HEAD_DIM], torch.bfloat16),
        TensorSpec("kv", [tokens, C.HEAD_DIM], torch.bfloat16),
        TensorSpec("selected", [tokens, 128, C.HEAD_DIM], torch.bfloat16),
        TensorSpec("attended", [tokens, C.LOCAL_H * C.HEAD_DIM], torch.bfloat16),
        TensorSpec("partial", [tokens, C.D], torch.float32),
        ScalarSpec("num_tokens", torch.int32, tokens),
    ))
    result = run(probe_chain, specs, golden_fn=golden_chain,
                 config=dict(platform="a5", device_id=args.device),
                 compare_fn={**{name: compare(name) for name in
                              ("qr", "q", "kv", "selected", "attended", "partial")},
                             **{name: compare_cache(name) for name in
                                ("window_cache", "window_cache_scale")}})
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
