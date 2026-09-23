# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Execute real forward tails with torch tensor semantics, without model imports.

These tests check HC mathematics and local/group row selection. They do not
exercise PyPTO compilation, scheduling, distributed execution, or NPU rounding.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_flash_dspark"
D, HC_MULT, TP_SIZE = 8, 4, 4


def _function(filename, name):
    tree = ast.parse((MODEL_DIR / filename).read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _writes_target(node):
    return any(
        isinstance(child, ast.Subscript)
        and isinstance(child.ctx, ast.Store)
        and isinstance(child.value, ast.Name)
        and child.value.id == "dspark_target_hidden"
        for child in ast.walk(node)
    )


def _run_tail(filename, name, values):
    # Select the active output branch, not a handwritten copy of its equations.
    branch = next(
        node for node in ast.walk(_function(filename, name))
        if isinstance(node, ast.If)
        and any(isinstance(stmt, ast.For) and _writes_target(stmt) for stmt in node.body)
    )
    body = []
    for stmt in branch.body:
        if any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "lm_head"
            for node in ast.walk(stmt)
        ):
            break
        body.append(stmt)
    code = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(code, str(MODEL_DIR / filename), "exec"), values)


def _hc_rows(rows, offset=0):
    values = torch.arange(rows * HC_MULT * D, dtype=torch.float32).reshape(rows, HC_MULT, D)
    return torch.sin(values * 0.17 + offset) + values * 0.007 + 0.2


def _learned_head(source, weight, scale, base):
    flat = source.flatten(1).float()
    inv_rms = torch.rsqrt(flat.square().mean(dim=1, keepdim=True) + 1e-6)
    gates = torch.sigmoid((flat @ weight.T) * inv_rms * scale + base) + 1e-6
    return (source * gates.unsqueeze(-1)).sum(dim=1).to(torch.bfloat16)


def _norm(source, weight):
    source = source.float()
    return (source * torch.rsqrt(source.square().mean(dim=1, keepdim=True) + 1e-6) * weight).to(torch.bfloat16)


def _environment(output_rows):
    def hc_head(source, weight, scale, base, output):
        output.copy_(_learned_head(source, weight, scale, base))
        return output

    def rms_norm(source, weight, output):
        output.copy_(_norm(source, weight))
        return 0

    # Only tensor primitives used by the selected production tail are adapted.
    language = SimpleNamespace(
        BF16=torch.bfloat16,
        FP32=torch.float32,
        spmd=lambda count, **_kwargs: range(count),
        reshape=lambda value, shape: value.reshape(shape),
        col_sum=lambda value: value.sum(dim=0, keepdim=True),
        mul=lambda left, right: left * right,
        cast=lambda value, target_type, **_kwargs: value.to(target_type),
        create_tensor=lambda shape, dtype: torch.empty(shape, dtype=dtype),
        slice=lambda value, shape, offset: value[
            tuple(slice(start, start + size) for start, size in zip(offset, shape))
        ],
    )
    return dict(
        pl=language, D=D, HC_MULT=HC_MULT, hc_head=hc_head, rms_norm=rms_norm,
        hc_head_fn=torch.cos(torch.arange(HC_MULT * HC_MULT * D).reshape(HC_MULT, HC_MULT * D) * 0.11) * 0.1,
        hc_head_scale=torch.tensor([0.7]), hc_head_base=torch.tensor([-2.0, -0.5, 0.7, 2.0]),
        final_norm_w=1 + torch.arange(D) * 0.02,
        x_out=torch.zeros(output_rows, D, dtype=torch.bfloat16),
    )


def _assert_main_head(values, source):
    expected = _norm(
        _learned_head(source, values["hc_head_fn"], values["hc_head_scale"], values["hc_head_base"]),
        values["final_norm_w"],
    )
    torch.testing.assert_close(values["x_out"], expected, rtol=0, atol=0)
    mean_output = _norm(source.mean(dim=1).to(torch.bfloat16), values["final_norm_w"])
    assert not torch.equal(expected, mean_output), "fixture must distinguish the main head from an HC mean"


@pytest.mark.parametrize("local_t", [1, 3, 5])
def test_decode_auxiliary_means_preserve_main_head(local_t):
    taps = [_hc_rows(local_t, offset) for offset in (0.0, 1.1, 2.2)]
    values = _environment(local_t)
    values.update(
        MOE_TOKENS=8, local_t=local_t,
        x_pong=taps[0], x_ping=taps[1], pre_hc_hidden_out=taps[2],
        hidden_workspace=torch.empty(local_t, D, dtype=torch.bfloat16),
        dspark_target_hidden=torch.full((local_t, 3 * D), float("nan"), dtype=torch.bfloat16),
    )
    _run_tail("decode_fwd.py", "_decode_fwd", values)
    expected = torch.cat([tap.mean(dim=1) for tap in taps], dim=1).to(torch.bfloat16)
    torch.testing.assert_close(values["dspark_target_hidden"], expected, rtol=0, atol=0)
    _assert_main_head(values, taps[2])


@pytest.mark.parametrize("local_tokens", [1, 3])
@pytest.mark.parametrize("tp_rank", [0, 1, 3])
def test_prefill_auxiliary_means_use_local_snapshots_and_group_layer42(local_tokens, tp_rank):
    group_rows = TP_SIZE * local_tokens
    local_start = tp_rank * local_tokens
    snapshots = [_hc_rows(local_tokens, offset) for offset in (0.0, 1.1)]
    layer42 = _hc_rows(group_rows, 2.2)
    values = _environment(group_rows)
    values.update(
        local_tokens=local_tokens, group_rows=group_rows, local_start=local_start,
        target_l41_start=local_tokens, target_hc_stack=torch.cat(snapshots), x_hc=layer42,
        dspark_target_hidden=torch.full((local_tokens, 3 * D), float("nan"), dtype=torch.bfloat16),
    )
    _run_tail("prefill_fwd.py", "prefill_fwd", values)
    taps = snapshots + [layer42[local_start : local_start + local_tokens]]
    expected = torch.cat([tap.mean(dim=1) for tap in taps], dim=1).to(torch.bfloat16)
    torch.testing.assert_close(values["dspark_target_hidden"], expected, rtol=0, atol=0)
    _assert_main_head(values, layer42)


def test_prefill_oracle_accepts_means_and_rejects_learned_head_or_inactive_data():
    namespace = dict(D=D, TP_SIZE=TP_SIZE, TARGET_LAYER_IDS=(40, 41, 42))
    function = _function("prefill_fwd.py", "dspark_target_hidden_compare")
    exec(compile(ast.Module(body=[function], type_ignores=[]), "prefill_fwd.py", "exec"), namespace)
    compare = namespace["dspark_target_hidden_compare"]
    local_tokens, ranks = 3, 2 * TP_SIZE
    group_rows = TP_SIZE * local_tokens
    hidden = _hc_rows(ranks * group_rows).reshape(ranks, group_rows, HC_MULT, D)
    actual = torch.ones(ranks, local_tokens, 3 * D, dtype=torch.bfloat16)
    for rank in range(TP_SIZE):
        start = rank * local_tokens
        actual[rank, :, 2 * D :] = hidden[rank, start : start + local_tokens].mean(dim=1)
    actual[TP_SIZE:] = 0
    query_start = torch.zeros(ranks, 2, dtype=torch.int32)
    query_start[:TP_SIZE, 1] = group_rows
    kwargs = dict(
        inputs=dict(query_start_loc=query_start, ori_slot_mapping_full=torch.ones(ranks, group_rows)),
        actual_outputs=dict(x_hc=hidden),
    )
    assert compare(actual, None, **kwargs)[0]

    old_output = actual.clone()
    weights = _environment(local_tokens)
    old_output[1, :, 2 * D :] = _learned_head(
        hidden[1, local_tokens : 2 * local_tokens],
        weights["hc_head_fn"], weights["hc_head_scale"], weights["hc_head_base"],
    )
    assert not compare(old_output, None, **kwargs)[0]
    inactive_modified = actual.clone()
    inactive_modified[TP_SIZE, 0, 0] = 1
    assert not compare(inactive_modified, None, **kwargs)[0]
