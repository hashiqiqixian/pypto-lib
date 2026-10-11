"""Replay one captured SWA Q/KV projection on one A5 card."""

import argparse

import pypto.language as pl
import torch

from golden import ScalarSpec, TensorSpec, run
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.decode_attn_swa import official_linear, official_rope
from models.deepseek_v4_1_flash.qkv_proj_rope import qkv_proj_rope


@pl.jit
def probe_qkv(
    x: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
    wq_a: pl.Tensor[[C.D, C.Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[C.D // 32, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[C.Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[C.D // 32, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    qr: pl.Out[pl.Tensor[[C.T_DYN, C.Q_LORA], pl.BF16]],
    q: pl.Out[pl.Tensor[[C.T_DYN, C.LOCAL_H * C.HEAD_DIM], pl.BF16]],
    kv: pl.Out[pl.Tensor[[C.T_DYN, C.HEAD_DIM], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    qkv_proj_rope(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
        kv_norm_weight, rope_cos, rope_sin, qr, q, kv, num_tokens,
    )
    return qr, q, kv


def golden_qkv(values):
    def norm(x, weight):
        x = x.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-20)
                * weight.float()).bfloat16()

    qr = norm(official_linear(values["x"], values["wq_a"], values["wq_a_scale"]),
              values["q_norm_weight"])
    q = official_linear(qr, values["wq_b"], values["wq_b_scale"])
    q = official_rope(q.unflatten(-1, (C.LOCAL_H, C.HEAD_DIM)),
                      values["rope_cos"], values["rope_sin"]).flatten(-2)
    kv = norm(official_linear(values["x"], values["wkv"], values["wkv_scale"]),
              values["kv_norm_weight"])
    kv = official_rope(kv, values["rope_cos"], values["rope_sin"])
    values["qr"].copy_(qr)
    values["q"].copy_(q)
    values["kv"].copy_(kv)


def compare(name):
    def check(actual, expected, **_):
        a = actual.float()
        e = expected.float()
        rel_l2 = (a - e).norm() / e.norm().clamp_min(1e-12)
        row_l2 = (a - e).norm(dim=-1) / e.norm(dim=-1).clamp_min(1e-12)
        print(f"[QKV] {name} rel_l2={rel_l2.item():.8g} "
              f"max_row_rel_l2={row_l2.max().item():.8g} max_abs={(a - e).abs().max().item():.8g}")
        return bool(torch.isfinite(a).all() and rel_l2 <= 0.01), "relative L2 <= 1%"
    return check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("--rank", type=int, default=0, choices=range(4))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    captured = torch.load(args.capture, map_location="cpu", weights_only=True)["attention_inputs"]
    values = {name: value[args.rank].contiguous() for name, value in captured.items()}
    inputs = ("x", "wq_a", "wq_a_scale", "q_norm_weight", "wq_b", "wq_b_scale",
              "wkv", "wkv_scale", "kv_norm_weight", "rope_cos", "rope_sin")
    specs = [TensorSpec(name, list(values[name].shape), values[name].dtype,
                        init_value=lambda name=name: values[name].clone()) for name in inputs]
    tokens = values["x"].shape[0]
    specs.extend((
        TensorSpec("qr", [tokens, C.Q_LORA], torch.bfloat16),
        TensorSpec("q", [tokens, C.LOCAL_H * C.HEAD_DIM], torch.bfloat16),
        TensorSpec("kv", [tokens, C.HEAD_DIM], torch.bfloat16),
        ScalarSpec("num_tokens", torch.int32, tokens),
    ))
    result = run(probe_qkv, specs, golden_fn=golden_qkv,
                 config=dict(platform="a5", device_id=args.device),
                 compare_fn={name: compare(name) for name in ("qr", "q", "kv")})
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
