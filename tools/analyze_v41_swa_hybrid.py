"""Compare captured SWA outputs with mixed-precision online references."""

import argparse

import torch


def replay(data, *, score_dtype, reduce_dtype, batch_size=8):
    q = data["q"].unflatten(-1, (16, 512))
    selected = data["selected"]
    indices = data["indices"]
    sink = data["sink"]
    outputs = []
    for offset in range(0, len(q), batch_size):
        query = q[offset:offset + batch_size].to(score_dtype)
        keys = selected[offset:offset + batch_size].to(score_dtype)
        valid = indices[offset:offset + batch_size]
        maximum = torch.full(query.shape[:2], -1e30, dtype=score_dtype)
        denominator = torch.zeros(query.shape[:2], dtype=reduce_dtype)
        numerator = torch.zeros_like(query, dtype=reduce_dtype)
        for part in (0, 64):
            tile = keys[:, part:part + 64]
            scores = torch.einsum("thd,tkd->thk", query, tile) * (512 ** -0.5)
            scores = scores.masked_fill(valid[:, None, part:part + 64] < 0, -torch.inf)
            next_maximum = torch.maximum(maximum, scores.amax(-1))
            correction = (maximum - next_maximum).exp().to(reduce_dtype)
            probabilities = (scores - next_maximum[..., None]).exp()
            denominator = denominator * correction + probabilities.to(reduce_dtype).sum(-1)
            weighted = torch.einsum(
                "thk,tkd->thd", probabilities.bfloat16().to(reduce_dtype),
                tile.to(reduce_dtype),
            )
            numerator = numerator * correction[..., None] + weighted
            maximum = next_maximum
        sink_value = sink[None].to(score_dtype)
        final_maximum = torch.maximum(maximum, sink_value)
        correction = (maximum - final_maximum).exp().to(reduce_dtype)
        denominator = denominator * correction + (sink_value - final_maximum).exp().to(reduce_dtype)
        factor = correction / denominator
        outputs.append((numerator * factor[..., None]).bfloat16())
    return torch.cat(outputs).flatten(-2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    args = parser.parse_args()
    torch.set_num_threads(8)
    data = torch.load(args.capture, map_location="cpu", weights_only=True)
    actual = data["actual"]
    for name, score_dtype, reduce_dtype in (
        ("fp64_all", torch.float64, torch.float64),
        ("fp64_scores_fp32_reduce", torch.float64, torch.float32),
        ("fp32_all", torch.float32, torch.float32),
    ):
        result = replay(data, score_dtype=score_dtype, reduce_dtype=reduce_dtype)
        mismatch = result != actual
        delta = result.float() - actual.float()
        print(f"{name}: mismatch={mismatch.sum().item()}/{mismatch.numel()} "
              f"rel_l2={(delta.norm() / actual.float().norm()).item():.9g} "
              f"token1_head1_dim445={result[1, 1 * 512 + 445].item():.9g}", flush=True)
    expected = data["expected"]
    print(f"saved_expected: mismatch={(expected != actual).sum().item()}/{actual.numel()}")


if __name__ == "__main__":
    main()
