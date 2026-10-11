# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
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
            attention_delta = (data["actual_attention_output"][rank].float() -
                               data["expected_attention_output"][rank].float())
            attention_row_error = attention_delta.norm(dim=-1) / data[
                "expected_attention_output"][rank].float().norm(dim=-1).clamp_min(1e-12)
            actual_payload, _ = official_quantize(actual_mixed[rank])
            expected_payload, _ = official_quantize(expected_mixed[rank])
            payload_row_changes = (actual_payload.view(torch.uint8) !=
                                   expected_payload.view(torch.uint8)).reshape(actual.shape[0], -1).sum(-1)
            for label, rows in (("unchanged", payload_row_changes == 0),
                                ("changed", payload_row_changes > 0)):
                if rows.any():
                    print(f"x_next rank={rank} fp8_payload_{label}_rows={int(rows.sum())} "
                          f"bad_ratio={bad[rows].float().mean().item():.8g} "
                          f"mean_row_rel_l2={attention_row_error[rows].mean().item():.8g}")
            top = torch.topk(row_bad, min(8, row_bad.numel())).indices
            causes = [(int(row), round(float(attention_row_error[row]), 7),
                       int(payload_row_changes[row]), round(float(row_bad[row]), 6))
                      for row in top]
            print(f"x_next rank={rank} worst_row_causes={causes} "
                  "(row, attention_rel_l2, fp8_payload_changes, x_next_bad_ratio)")
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
