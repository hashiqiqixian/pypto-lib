"""Replay the captured SWA online-attention stage on one A5 card."""

import argparse

import pypto.language as pl
import torch

from golden import ScalarSpec, TensorSpec, run
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.decode_attn_swa import attend_window, official_linear, official_rope
from models.deepseek_v4_1_flash.quantization import decode_e8m0


@pl.jit
def probe_window(
    q: pl.Tensor[[C.T_DYN, C.LOCAL_H * C.HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[C.T_DYN, 128, C.HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[C.T_DYN, 128], pl.INT32],
    sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
    attended: pl.Out[pl.Tensor[[C.T_DYN, C.LOCAL_H * C.HEAD_DIM], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    attend_window(q, selected, indices, sink, attended, num_tokens)
    return attended


def inputs_from_capture(captured, rank):
    values = {name: value[rank].contiguous() for name, value in captured.items()}
    def norm(x, weight):
        x = x.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-20)
                * weight.float()).bfloat16()
    qr = norm(official_linear(values["x"], values["wq_a"], values["wq_a_scale"]),
              values["q_norm_weight"])
    q = official_linear(qr, values["wq_b"], values["wq_b_scale"])
    q = official_rope(q.unflatten(-1, (C.LOCAL_H, C.HEAD_DIM)),
                      values["rope_cos"], values["rope_sin"]).flatten(-2)
    cache = values["window_cache"].float()
    scales = decode_e8m0(values["window_cache_scale"]).repeat_interleave(32, -1)
    cache = (cache * scales).reshape(-1, C.HEAD_DIM)
    indices = values["window_indices"]
    selected = cache[indices.long().clamp_min(0)].bfloat16()
    selected = selected.masked_fill((indices < 0)[..., None], 0)
    return {"q": q, "selected": selected, "indices": indices, "sink": values["attn_sink"]}


def golden_window(values):
    query = values["q"].double().unflatten(-1, (C.LOCAL_H, C.HEAD_DIM))
    selected = values["selected"].double()
    indices = values["indices"]
    maximum = torch.full(query.shape[:2], -1e30, dtype=torch.float64)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros_like(query)
    for start in range(0, 128, 64):
        keys = selected[:, start:start + 64]
        logits = torch.einsum("thd,tkd->thk", query, keys) * C.HEAD_DIM ** -0.5
        valid = indices[:, start:start + 64] >= 0
        logits = logits.masked_fill(~valid[:, None], -torch.inf)
        next_max = torch.maximum(maximum, logits.amax(-1))
        correction = (maximum - next_max).exp()
        probabilities = (logits - next_max[..., None]).exp()
        denominator = denominator * correction + probabilities.sum(-1)
        numerator = numerator * correction[..., None] + torch.einsum(
            "thk,tkd->thd", probabilities.bfloat16().double(), keys)
        maximum = next_max
    sink = values["sink"].double()[None]
    final_max = torch.maximum(maximum, sink)
    correction = (maximum - final_max).exp()
    denominator = denominator * correction + (sink - final_max).exp()
    result = (numerator * (correction / denominator)[..., None]).bfloat16()
    values["attended"].copy_(result.flatten(-2))


def compare(actual, expected, **_):
    a, e = actual.float(), expected.float()
    rel_l2 = (a - e).norm() / e.norm().clamp_min(1e-12)
    row_l2 = (a - e).norm(dim=-1) / e.norm(dim=-1).clamp_min(1e-12)
    print(f"[WINDOW] attended rel_l2={rel_l2.item():.8g} "
          f"max_row_rel_l2={row_l2.max().item():.8g} max_abs={(a-e).abs().max().item():.8g}")
    return bool(torch.isfinite(a).all() and rel_l2 <= 0.01), "relative L2 <= 1%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("--rank", type=int, default=0, choices=range(4))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    captured = torch.load(args.capture, map_location="cpu", weights_only=True)["attention_inputs"]
    values = inputs_from_capture(captured, args.rank)
    specs = [TensorSpec(name, list(value.shape), value.dtype,
                        init_value=lambda value=value: value.clone()) for name, value in values.items()]
    tokens = values["q"].shape[0]
    specs += [TensorSpec("attended", [tokens, C.LOCAL_H * C.HEAD_DIM], torch.bfloat16),
              ScalarSpec("num_tokens", torch.int32, tokens)]
    result = run(probe_window, specs, golden_fn=golden_window,
                 config=dict(platform="a5", device_id=args.device), compare_fn={"attended": compare})
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
