"""Compare one sensitive SWA QK tile with independent CPU dot products."""

import argparse

import pypto.language as pl
import torch

from golden import TensorSpec, run
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.decode_attn_swa import M_TILE, SOFTMAX_SCALE


@pl.jit
def score_tile(
    q: pl.Tensor[[M_TILE, C.HEAD_DIM], pl.BF16],
    keys: pl.Tensor[[64, C.HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[1, 64], pl.INT32],
    sink: pl.Tensor[[M_TILE], pl.FP32],
    scores: pl.Out[pl.Tensor[[M_TILE, 64], pl.FP32]],
    weights: pl.Out[pl.Tensor[[M_TILE, 64], pl.BF16]],
    weighted: pl.Out[pl.Tensor[[M_TILE, C.HEAD_DIM], pl.FP32]],
    factor: pl.Out[pl.Tensor[[M_TILE, 1], pl.FP32]],
    precast: pl.Out[pl.Tensor[[M_TILE, C.HEAD_DIM], pl.FP32]],
    attended: pl.Out[pl.Tensor[[M_TILE, C.HEAD_DIM], pl.BF16]],
):
    for _ in pl.spmd(1):
        raw = pl.mul(pl.matmul(q, keys, b_trans=True), SOFTMAX_SCALE)
        valid = pl.minimum(pl.maximum(pl.add(pl.cast(indices, pl.FP32), 1.0), 0.0), 1.0)
        bias = pl.mul(pl.sub(valid, 1.0), 1e30)
        masked = pl.col_expand_add(raw, bias)
        maximum = pl.reshape(pl.row_max(masked), [1, M_TILE])
        probability = pl.col_expand_mul(
            pl.exp(pl.row_expand_sub(masked, pl.reshape(maximum, [M_TILE, 1]))), valid
        )
        denominator = pl.reshape(pl.row_sum(probability), [1, M_TILE])
        weights_local = pl.cast(probability, pl.BF16, mode="rint")
        numerator = pl.matmul(weights_local, keys)
        weighted[:, :] = numerator
        sinks = pl.reshape(sink, [1, M_TILE])
        final_max = pl.maximum(maximum, sinks)
        correction = pl.exp(pl.sub(maximum, final_max))
        final_denominator = pl.add(
            pl.mul(denominator, correction), pl.exp(pl.sub(sinks, final_max))
        )
        multiplier = pl.reshape(pl.div(correction, final_denominator), [M_TILE, 1])
        factor[:, :] = multiplier
        result = pl.row_expand_mul(numerator, multiplier)
        precast[:, :] = result
        scores[:, :] = raw
        weights[:, :] = weights_local
        attended[:, :] = pl.cast(result, pl.BF16, mode="rint")
    return scores, weights, weighted, factor, precast, attended


def golden_score(values):
    exact = values["q"].double() @ values["keys"].double().T * SOFTMAX_SCALE
    values["scores"].copy_(exact.float())
    valid = values["indices"][0] >= 0
    masked = exact.masked_fill(~valid[None], -torch.inf)
    maximum = masked.amax(-1)
    probability = (masked - maximum[:, None]).exp()
    weights = probability.bfloat16()
    values["weights"].copy_(weights)
    weighted = weights.double() @ values["keys"].double()
    values["weighted"].copy_(weighted.float())
    sink = values["sink"].double()
    final_max = torch.maximum(maximum, sink)
    correction = (maximum - final_max).exp()
    final_denominator = probability.sum(-1) * correction + (sink - final_max).exp()
    multiplier = (correction / final_denominator)[:, None]
    values["factor"].copy_(multiplier.float())
    precast = weighted * multiplier
    values["precast"].copy_(precast.float())
    values["attended"].copy_(precast.bfloat16())


def compare(actual, expected, *, inputs, **_):
    fp32 = inputs["q"].float() @ inputs["keys"].float().T * SOFTMAX_SCALE
    exact_error = (actual - expected).abs()
    fp32_error = (fp32 - expected).abs()
    print(f"[SCORE] device_vs_fp64 max_abs={exact_error.max().item():.9g} "
          f"mean_abs={exact_error.mean().item():.9g} "
          f"fp32_vs_fp64 max_abs={fp32_error.max().item():.9g} "
          f"mean_abs={fp32_error.mean().item():.9g}", flush=True)
    print(f"[SCORE] device_closer={(exact_error < fp32_error).sum().item()} "
          f"fp32_closer={(fp32_error < exact_error).sum().item()} "
          f"equal={(fp32_error == exact_error).sum().item()}", flush=True)
    return bool(torch.isfinite(actual).all()), "finite scores"


def compare_weights(actual, expected, **_):
    changed = actual != expected
    print(f"[WEIGHTS] changed={changed.sum().item()}/{changed.numel()} "
          f"per_head={changed.sum(-1).tolist()} "
          f"head15_indices={torch.nonzero(changed[15]).flatten().tolist()}", flush=True)
    return bool(torch.isfinite(actual.float()).all()), "finite probabilities"


def compare_weighted(actual, expected, **_):
    delta = (actual - expected).abs()
    finite = torch.isfinite(actual).all() and torch.isfinite(expected).all()
    print(f"[WEIGHTED] max_abs={delta.max().item():.9g} "
          f"mean_abs={delta.mean().item():.9g} "
          f"head15_max_abs={delta[15].max().item():.9g} "
          f"actual_head1_first={actual[1, :4].tolist()} "
          f"expected_head1_first={expected[1, :4].tolist()}", flush=True)
    return bool(finite and torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)), "weighted values"


def compare_attended(actual, expected, **_):
    changed = actual != expected
    print(f"[ATTENDED] changed={changed.sum().item()}/{changed.numel()} "
          f"per_head={changed.sum(-1).tolist()} "
          f"head15_indices={torch.nonzero(changed[15]).flatten().tolist()[:30]}",
          flush=True)
    return bool(torch.isfinite(actual.float()).all() and torch.isfinite(expected.float()).all()), "finite attended values"


def compare_factor(actual, expected, **_):
    delta = (actual - expected).abs()
    print(f"[FACTOR] max_abs={delta.max().item():.9g} "
          f"head1_device={actual[1, 0].item():.12g} "
          f"head1_fp64={expected[1, 0].item():.12g}", flush=True)
    return bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()), "finite factor"


def compare_precast(actual, expected, **_):
    delta = (actual - expected).abs()
    print(f"[PRECAST] max_abs={delta.max().item():.9g} "
          f"head1_dim445_device={actual[1, 445].item():.12g} "
          f"head1_dim445_fp64={expected[1, 445].item():.12g}", flush=True)
    return bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()), "finite precast"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("--token", type=int, default=102)
    parser.add_argument("--head", type=int, default=15)
    parser.add_argument("--part", type=int, default=0, choices=(0, 1))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    data = torch.load(args.capture, map_location="cpu", weights_only=True)
    head_start = args.head // M_TILE * M_TILE
    q = data["q"][args.token].reshape(C.LOCAL_H, C.HEAD_DIM)[
        head_start:head_start + M_TILE
    ].contiguous()
    keys = data["selected"][args.token, args.part * 64:(args.part + 1) * 64].contiguous()
    indices = data["indices"][args.token, args.part * 64:(args.part + 1) * 64].unsqueeze(0).contiguous()
    sink = data["sink"][head_start:head_start + M_TILE].contiguous()
    captured_attended = data["actual"][args.token].reshape(C.LOCAL_H, C.HEAD_DIM)[
        head_start:head_start + M_TILE
    ]
    valid_keys = int((indices >= 0).sum().item())
    if not valid_keys:
        raise ValueError("selected tile has no valid keys; replay it with the preceding online state")
    print(f"[INPUT] valid_keys={(indices >= 0).sum().item()} "
          f"sink_head15={sink[15].item():.9g} "
          f"captured_head15_nonzero={(captured_attended[15] != 0).sum().item()}",
          flush=True)
    specs = [
        TensorSpec("q", list(q.shape), q.dtype, init_value=lambda: q.clone()),
        TensorSpec("keys", list(keys.shape), keys.dtype, init_value=lambda: keys.clone()),
        TensorSpec("indices", list(indices.shape), indices.dtype,
                   init_value=lambda: indices.clone()),
        TensorSpec("sink", list(sink.shape), sink.dtype, init_value=lambda: sink.clone()),
        TensorSpec("scores", [M_TILE, 64], torch.float32),
        TensorSpec("weights", [M_TILE, 64], torch.bfloat16),
        TensorSpec("weighted", [M_TILE, C.HEAD_DIM], torch.float32),
        TensorSpec("factor", [M_TILE, 1], torch.float32),
        TensorSpec("precast", [M_TILE, C.HEAD_DIM], torch.float32),
        TensorSpec("attended", [M_TILE, C.HEAD_DIM], torch.bfloat16),
    ]
    print(f"[SCORE] token={args.token} heads={head_start}:{head_start + M_TILE} "
          f"part={args.part}", flush=True)
    result = run(score_tile, specs, golden_fn=golden_score,
                 config=dict(platform="a5", device_id=args.device),
                 compare_fn={"scores": compare, "weights": compare_weights,
                             "weighted": compare_weighted,
                             "factor": compare_factor, "precast": compare_precast,
                             "attended": compare_attended})
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
