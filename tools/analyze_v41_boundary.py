"""Attribute a saved decode Block boundary difference to hidden or pre-mix inputs."""

import argparse

import torch

from models.deepseek_v4_1_flash.hc_mixes import golden_mhc_mixes
from models.deepseek_v4_1_flash.hc_post import golden_mhc_post
from models.deepseek_v4_1_flash.hc_pre import golden_mhc_pre
from models.deepseek_v4_1_flash.decode_attn_swa import official_quantize


def report(label, actual, expected):
    actual = actual.float()
    expected = expected.float()
    delta = (actual - expected).abs()
    scale = torch.maximum(actual.abs(), expected.abs()).clamp_min(1 / (2**14 * 0.01)) + 1e-9
    relative = torch.where(delta < 0.01, delta, delta / scale)
    rel_l2 = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
    bad = (relative > 0.01).reshape(relative.shape[0], -1).float().mean(dim=1)
    print(f"{label}: rel_l2={rel_l2.item():.8g}, max_abs={delta.max().item():.8g}, "
          f"bad_ratio_per_rank={[round(value, 6) for value in bad.tolist()]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    args = parser.parse_args()
    data = torch.load(args.capture, map_location="cpu", weights_only=True)
    actual_h = data["actual_attention_hidden"]
    expected_h = data["expected_attention_hidden"]
    actual_p = data["actual_attention_pre_mix"]
    expected_p = data["expected_attention_pre_mix"]
    report("attention_hidden", actual_h, expected_h)
    report("attention_pre_mix", actual_p, expected_p)
    report("next_pre_mix independent", data["actual_next_pre_mix"], data["expected_next_pre_mix"])
    report("next_pre_mix conditioned", data["actual_next_pre_mix"], data["conditioned_next_pre_mix"])
    report("x_mixed independent", data["actual_x_mixed"], data["expected_x_mixed"])
    report("x_mixed conditioned", data["actual_x_mixed"], data["conditioned_x_mixed"])
    actual_mixed = data["actual_x_mixed"]
    expected_mixed = data["expected_x_mixed"]
    for rank in range(actual_mixed.shape[0]):
        actual_payload, actual_scale = official_quantize(actual_mixed[rank])
        expected_payload, expected_scale = official_quantize(expected_mixed[rank])
        scale_changed = actual_scale != expected_scale
        payload_changed = actual_payload.view(torch.uint8) != expected_payload.view(torch.uint8)
        groups = actual_scale.numel()
        print(f"moe_input rank={rank}: scale_changed={scale_changed.sum().item()}/{groups} "
              f"payload_changed={payload_changed.sum().item()}/{payload_changed.numel()} "
              f"payload_changed_in_scale_groups="
              f"{(payload_changed.reshape(*actual_scale.shape, 32) & scale_changed[..., None]).sum().item()} "
              f"payload_changed_in_same_scale_groups="
              f"{(payload_changed.reshape(*actual_scale.shape, 32) & ~scale_changed[..., None]).sum().item()}")
    if "actual_x_next" in data:
        report("x_next independent", data["actual_x_next"], data["expected_x_next"])
        report("x_next conditioned", data["actual_x_next"], data["conditioned_x_next"])
        report("x_next input propagation", data["conditioned_x_next"], data["expected_x_next"])
        for rank in range(data["actual_x_next"].shape[0]):
            actual = data["actual_x_next"][rank].float()
            expected = data["expected_x_next"][rank].float()
            conditioned = data["conditioned_x_next"][rank].float()
            total = (actual - expected).norm().item()
            device = (actual - conditioned).norm().item()
            propagated = (conditioned - expected).norm().item()
            print(f"x_next rank={rank}: total_l2={total:.8g}, "
                  f"device_l2={device:.8g}, propagation_l2={propagated:.8g}")
            delta = (actual - expected).abs()
            scale = torch.maximum(actual.abs(), expected.abs()).clamp_min(1 / (2**14 * 0.01)) + 1e-9
            bad = torch.where(delta < 0.01, delta, delta / scale) > 0.01
            row_bad = bad.flatten(1).float().mean(-1)
            largest = torch.topk(row_bad, min(8, row_bad.numel()))
            print(f"x_next rank={rank} worst_rows="
                  f"{list(zip(largest.indices.tolist(), [round(v, 6) for v in largest.values.tolist()]))}")
    report("x_mixed using actual hidden, expected pre-mix", data["actual_x_mixed"],
           golden_mhc_pre(actual_h, expected_p))
    report("x_mixed using expected hidden, actual pre-mix", data["actual_x_mixed"],
           golden_mhc_pre(expected_h, actual_p))
    if "actual_attention_output" in data:
        report("attention_output", data["actual_attention_output"],
               data["expected_attention_output"])
        actual_o = data["actual_attention_output"].float()
        expected_o = data["expected_attention_output"].float()
        row_error = (actual_o - expected_o).norm(dim=-1) / expected_o.norm(dim=-1).clamp_min(1e-12)
        for rank, rows in enumerate(row_error):
            largest = torch.topk(rows, min(8, rows.numel()))
            print(f"attention_output rank={rank} mean_row_rel_l2={rows.mean().item():.8g} "
                  f"worst_rows={list(zip(largest.indices.tolist(), [round(v, 8) for v in largest.values.tolist()]))}")
        conditioned_h = []
        reference_h = []
        for rank in range(actual_h.shape[0]):
            _, post, residual = golden_mhc_mixes(
                data["x_hc"][rank], data["hc_attn_fn"][rank],
                data["hc_attn_scale"][rank], data["hc_attn_base"][rank],
            )
            conditioned_h.append(golden_mhc_post(
                data["actual_attention_output"][rank], data["x_hc"][rank], post, residual,
            ))
            reference_h.append(golden_mhc_post(
                data["expected_attention_output"][rank], data["x_hc"][rank], post, residual,
            ))
        conditioned_h = torch.stack(conditioned_h)
        reference_h = torch.stack(reference_h)
        report("attention_hidden conditioned on device output", actual_h, conditioned_h)
        report("attention_hidden golden replay", expected_h, reference_h)
        report("attention_hidden output propagation", conditioned_h, expected_h)


if __name__ == "__main__":
    main()
