"""Inspect saved one-card SWA stages without allocating an NPU."""

import argparse

import torch


def replay_attention(data, score_dtype, online_dtype, value_dtype):
    q = data["q"].reshape(-1, 16, 512).double()
    selected = data["selected"].double()
    idx = data["indices"]
    maximum = torch.full(q.shape[:2], -1e30, dtype=online_dtype)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros_like(q, dtype=value_dtype)
    for start in range(0, 128, 64):
        keys = selected[:, start:start + 64]
        logits = torch.einsum("thd,tkd->thk", q.to(score_dtype), keys.to(score_dtype))
        logits = (logits * 512 ** -0.5).to(online_dtype)
        logits = logits.masked_fill(idx[:, None, start:start + 64] < 0, -torch.inf)
        next_maximum = torch.maximum(maximum, logits.amax(-1))
        correction = (maximum - next_maximum).exp()
        probabilities = (logits - next_maximum[..., None]).exp()
        denominator = denominator * correction + probabilities.sum(-1)
        numerator = numerator * correction[..., None].to(value_dtype) + torch.einsum(
            "thk,tkd->thd", probabilities.bfloat16().to(value_dtype), keys.to(value_dtype)
        )
        maximum = next_maximum
    sink = data["sink"].to(online_dtype)[None]
    final_maximum = torch.maximum(maximum, sink)
    correction = (maximum - final_maximum).exp()
    denominator = denominator * correction + (sink - final_maximum).exp()
    return (numerator * (correction / denominator).to(value_dtype)[..., None]).bfloat16().reshape_as(data["actual"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    args = parser.parse_args()
    data = torch.load(args.capture, map_location="cpu", weights_only=True)
    for name, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"{name}: shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"{name}: {type(value).__name__}")
    for actual_name, expected_name in (
        ("actual", "expected"),
        ("actual_attended", "expected_attended"),
        ("actual_q", "expected_q"),
        ("actual_selected", "expected_selected"),
    ):
        if actual_name not in data or expected_name not in data:
            continue
        actual, expected = data[actual_name], data[expected_name]
        changed = actual != expected
        print(f"{actual_name}: changed={changed.sum().item()}/{changed.numel()}")
        if changed.any():
            coords = changed.nonzero()[:32].tolist()
            print(f"first_changed={coords}")
            for coord in coords[:8]:
                index = tuple(coord)
                print(f"  {coord}: actual={actual[index].item()} expected={expected[index].item()}")
            rows = changed.reshape(changed.shape[0], -1).sum(-1)
            print(f"changed_per_row={[(i, int(v)) for i, v in enumerate(rows) if v]}")
    if {"actual", "q", "selected", "indices", "sink"} <= data.keys():
        variants = (
            ("all-fp64", torch.float64, torch.float64, torch.float64),
            ("exact-scores-fp32-online-and-value", torch.float64, torch.float32, torch.float32),
            ("exact-scores-online-fp32-value", torch.float64, torch.float64, torch.float32),
            ("fp32-scores-exact-online-value", torch.float32, torch.float64, torch.float64),
        )
        for label, score_dtype, online_dtype, value_dtype in variants:
            replayed = replay_attention(data, score_dtype, online_dtype, value_dtype)
            changed = replayed != data["actual"]
            print(f"{label}: changed={changed.sum().item()}/{changed.numel()}")
            coords = changed.nonzero().tolist()
            print(f"  changed coordinates={coords[:32]}")
            rows = changed.sum(-1)
            print(f"  changed per row={[(i, int(v)) for i, v in enumerate(rows) if v]}")


if __name__ == "__main__":
    main()
