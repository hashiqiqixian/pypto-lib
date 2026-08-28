# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static contracts for the DeepSeek-V4 DSpark target-hidden bridge."""

import ast
from pathlib import Path


MODEL_DIR = Path(__file__).parents[2] / "models" / "deepseek_v4_flash_dspark"


def _tree(name: str) -> ast.Module:
    return ast.parse((MODEL_DIR / name).read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _out_names(function: ast.FunctionDef) -> set[str]:
    return {
        arg.arg
        for arg in function.args.args
        if arg.annotation is not None and ast.unparse(arg.annotation).startswith("pl.Out[")
    }


def _capture_slots(function: ast.FunctionDef) -> list[int]:
    slots = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "capture_dspark_target_hidden":
            continue
        slot = node.args[2]
        assert isinstance(slot, ast.Call) and isinstance(slot.func, ast.Attribute) and slot.func.attr == "const"
        slots.append(ast.literal_eval(slot.args[0]))
    return slots


def test_target_hidden_averages_four_hc_streams_into_three_layer_slots():
    helper = _function(_tree("dspark_proj.py"), "capture_dspark_target_hidden")
    source = ast.unparse(helper)

    assert "HC_MULT * D" in source
    assert all(f"{stream} * D + d0" in source for stream in (2, 3))
    assert "D + d0" in source
    assert "1.0 / HC_MULT" in source
    assert "target_layer_slot * D" in source
    assert "target_type=pl.BF16" in source


def test_prefill_and_decode_export_post_layer_40_41_42_target_hidden():
    for file_name, child_name, host_name in (
        ("prefill_fwd.py", "prefill_fwd", "l3_prefill_fwd"),
        ("decode_fwd.py", "decode_fwd", "l3_decode_fwd"),
    ):
        tree = _tree(file_name)
        child = _function(tree, child_name)
        host = _function(tree, host_name)

        assert "target_hidden" in _out_names(child)
        assert "target_hidden" in _out_names(host)
        assert sorted(_capture_slots(child)) == [0, 1, 2]


def test_drafter_consumes_the_same_three_layer_hidden_width():
    tree = _tree("dspark_drafter.py")
    drafter = _function(tree, "dspark_drafter")
    prepare = _function(tree, "prepare_dspark_inputs")
    target_hidden = next(arg for arg in drafter.args.args if arg.arg == "target_hidden")

    assert "MAIN_IN" in ast.unparse(target_hidden.annotation)
    assert "MAIN_IN = DSPARK_DRAFT_LAYERS * D" in (MODEL_DIR / "dspark_drafter.py").read_text(encoding="utf-8")
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dspark_proj"
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "target_hidden"
        for node in ast.walk(prepare)
    )
