# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static contracts for the DeepSeek-V4 Flash prefill MoE epoch protocol."""

import ast
from pathlib import Path


MODEL_DIR = Path(__file__).parents[2] / "models" / "deepseek_v4_flash_mtp"


def _tree(name: str) -> ast.Module:
    return ast.parse((MODEL_DIR / name).read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _call_name(node.value)
        return f"{owner}.{node.attr}" if owner else node.attr
    return ""


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and _call_name(child.func) == name
    ]


def _keyword(call: ast.Call, name: str) -> ast.AST:
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


def _context(function: ast.FunctionDef, name_hint: str) -> ast.With:
    matches = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.With)
        and isinstance(node.items[0].context_expr, ast.Call)
        and any(
            keyword.arg == "name_hint"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == name_hint
            for keyword in node.items[0].context_expr.keywords
        )
    ]
    assert len(matches) == 1
    return matches[0]


def _assigned_call(function: ast.FunctionDef, target_name: str, call_name: str) -> ast.Call:
    matches = []
    for node in ast.walk(function):
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        value = node.value if isinstance(node, (ast.AnnAssign, ast.Assign)) else None
        if (
            isinstance(target, ast.Name)
            and target.id == target_name
            and isinstance(value, ast.Call)
            and _call_name(value.func) == call_name
        ):
            matches.append(value)
    assert len(matches) == 1
    return matches[0]


def test_prefill_protocol_keeps_reduce_taskid_local() -> None:
    tree = _tree("moe.py")
    dispatch = _function(tree, "prefill_dispatch")
    combine = _function(tree, "prefill_combine")
    prefill_moe = _function(tree, "prefill_moe")

    call_names = {_call_name(call.func) for call in ast.walk(prefill_moe) if isinstance(call, ast.Call)}
    assert {"prefill_dispatch", "prefill_combine"} <= call_names
    assert {"moe", "dispatch", "combine"}.isdisjoint(call_names)
    assert "pl.TASK_ID" not in ast.unparse(dispatch)
    assert "_reduce_tid" not in ast.unparse(dispatch)
    assert "pl.TASK_ID" not in ast.unparse(prefill_moe)
    assert "_reduce_tid" not in ast.unparse(prefill_moe)
    assert [ast.unparse(node.value) for node in ast.walk(prefill_moe) if isinstance(node, ast.Return)] == [
        "x_next"
    ]

    submissions = _calls(combine, "pl.spmd_submit")
    assert len(submissions) == 1
    assignment = next(
        node for node in combine.body if isinstance(node, ast.Assign) and node.value is submissions[0]
    )
    assert ast.unparse(assignment.targets[0]) == "(ffn_out, _reduce_tid)"
    publication = combine.body[combine.body.index(assignment) + 1]
    assert publication is _context(combine, "moe_consumed")
    publication_call = publication.items[0].context_expr
    assert ast.unparse(_keyword(publication_call, "deps")) == "[_reduce_tid]"
    consumed_notify = _calls(publication, "pld.system.notify")
    assert len(consumed_notify) == 1
    assert ast.unparse(_keyword(consumed_notify[0], "target")) == "consumed"
    assert all("_reduce_tid" not in ast.unparse(node.value) for node in ast.walk(combine) if isinstance(node, ast.Return))


def test_prefill_dispatch_uses_padded_set_epoch_grid() -> None:
    tree = _tree("moe.py")
    dispatch = _function(tree, "prefill_dispatch")
    combine = _function(tree, "prefill_combine")
    signal_pad = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SIGNAL_PAD" for target in node.targets)
    )
    assert ast.literal_eval(signal_pad.value) == 128
    assert "NotifyOp.AtomicAdd" not in ast.unparse(dispatch)
    assert "NotifyOp.AtomicAdd" not in ast.unparse(combine)

    notifications = _calls(dispatch, "pld.system.notify") + _calls(combine, "pld.system.notify")
    assert len(notifications) == 4
    notify_by_target = {ast.unparse(_keyword(call, "target")): call for call in notifications}
    expected_notifies = {
        "arrived": ("dst", "[my_rank, 0]"),
        "data_arrived": ("dst", "[my_rank, loc_e, 0]"),
        "combine_arrived": ("peer", "[my_rank, e, 0]"),
        "consumed": ("peer", "[my_rank, 0]"),
    }
    assert set(notify_by_target) == set(expected_notifies)
    for target, (peer, offsets) in expected_notifies.items():
        call = notify_by_target[target]
        assert ast.unparse(_keyword(call, "peer")) == peer
        assert ast.unparse(_keyword(call, "offsets")) == offsets
        assert ast.unparse(_keyword(call, "value")) == "moe_epoch"
        assert ast.unparse(_keyword(call, "op")) == "pld.NotifyOp.Set"

    waits = _calls(dispatch, "pld.system.wait") + _calls(combine, "pld.system.wait")
    assert len(waits) == 4
    wait_by_signal = {ast.unparse(_keyword(call, "signal")): call for call in waits}
    expected_waits = {
        "consumed": ("[src, 0]", "pl.cast(moe_epoch - 1, pl.INT32)"),
        "arrived": ("[src, 0]", "moe_epoch"),
        "data_arrived": ("[src, loc_e, 0]", "moe_epoch"),
        "combine_arrived": ("[src, e, 0]", "moe_epoch"),
    }
    assert set(wait_by_signal) == set(expected_waits)
    for signal, (offsets, expected) in expected_waits.items():
        call = wait_by_signal[signal]
        assert ast.unparse(_keyword(call, "offsets")) == offsets
        assert ast.unparse(_keyword(call, "expected")) == expected
        assert ast.unparse(_keyword(call, "cmp")) == "pld.WaitCmp.Ge"

    reuse_source = ast.unparse(_context(dispatch, "moe_reuse_wait"))
    assert "_indices_anchor = pl.read(indices, [0, 0])" in reuse_source
    assert "if moe_epoch > 1" in reuse_source
    dependencies = {
        "dispatch_stage": (dispatch, "pl.at", None, "[_reuse_tid]"),
        "dispatch_meta": (dispatch, "pl.at", None, "[_reuse_tid]"),
        "dispatch_push": (dispatch, "pl.spmd", "N_LOCAL", "[_reuse_tid, _stage_tid]"),
        "dispatch_wait": (dispatch, "pl.spmd", "N_LOCAL", "[_meta_tid, _push_tid]"),
        "dispatch_gather": (dispatch, "pl.spmd", "N_LOCAL", "[_wait_tid, _push_tid]"),
        "combine_wait": (combine, "pl.spmd", "N_LOCAL", "[_cscatter_tid]"),
    }
    for name_hint, (function, call_name, block_count, deps) in dependencies.items():
        call = _context(function, name_hint).items[0].context_expr
        assert _call_name(call.func) == call_name
        if block_count is not None:
            assert ast.unparse(call.args[0]) == block_count
        assert ast.unparse(_keyword(call, "deps")) == deps


def test_prefill_window_shapes_match_all_jit_and_host_boundaries() -> None:
    expected = {
        "arrived": "pld.DistributedTensor[[N_RANKS, SIGNAL_PAD], pl.INT32]",
        "data_arrived": "pld.DistributedTensor[[N_RANKS, N_LOCAL, SIGNAL_PAD], pl.INT32]",
        "combine_arrived": "pl.InOut[pld.DistributedTensor[[N_RANKS, N_LOCAL, SIGNAL_PAD], pl.INT32]]",
        "consumed": "pl.InOut[pld.DistributedTensor[[N_RANKS, SIGNAL_PAD], pl.INT32]]",
    }
    boundaries = {
        "moe.py": {
            "clear_prefill_moe_signals": tuple(expected),
            "prefill_dispatch": tuple(expected),
            "prefill_combine": ("combine_arrived", "consumed"),
            "prefill_moe": tuple(expected),
        },
        "prefill_fwd.py": {"_prefill_request": tuple(expected), "prefill_fwd": tuple(expected)},
        "prefill_layer.py": {"_prefill_layer_tile": tuple(expected), "prefill_layer_core": tuple(expected)},
    }
    for file_name, functions in boundaries.items():
        tree = _tree(file_name)
        for function_name, signal_names in functions.items():
            annotations = {
                arg.arg: ast.unparse(arg.annotation)
                for arg in _function(tree, function_name).args.args
                if arg.arg in expected
            }
            assert annotations == {name: expected[name] for name in signal_names}

    shapes = {
        "arrived": "[N_RANKS, SIGNAL_PAD]",
        "data_arrived": "[N_RANKS, N_LOCAL, SIGNAL_PAD]",
        "combine_arrived": "[N_RANKS, N_LOCAL, SIGNAL_PAD]",
        "consumed": "[N_RANKS, SIGNAL_PAD]",
    }
    for file_name, host_name in (("prefill_fwd.py", "l3_prefill_fwd"), ("prefill_layer.py", "l3_prefill_layer")):
        host = _function(_tree(file_name), host_name)
        for window_name, shape in shapes.items():
            allocation = _assigned_call(host, f"{window_name}_buf", "pld.alloc_window_buffer")
            assert ast.unparse(allocation.args[0]) == shape
            assert ast.unparse(_keyword(allocation, "dtype")) == "pl.INT32"
            view = _assigned_call(host, window_name, "pld.window")
            assert ast.unparse(view.args[0]) == f"{window_name}_buf"
            assert ast.unparse(view.args[1]) == shape
            assert ast.unparse(_keyword(view, "dtype")) == "pl.INT32"

    moe = _function(_tree("moe.py"), "prefill_moe")
    parameter_names = [arg.arg for arg in moe.args.args]
    caller_trees = (_tree("prefill_fwd.py"), _tree("prefill_layer.py"))
    calls = [call for tree in caller_trees for call in _calls(tree, "prefill_moe")]
    assert len(calls) == 6
    for call in calls:
        assert not call.keywords and len(call.args) == len(parameter_names)
        for window_name in expected:
            assert ast.unparse(call.args[parameter_names.index(window_name)]) == window_name


def test_clear_prefill_moe_signals_retires_then_resets_local_slots() -> None:
    clear = _function(_tree("moe.py"), "clear_prefill_moe_signals")
    contexts = [node for node in ast.walk(clear) if isinstance(node, ast.With)]
    assert contexts == [_context(clear, "moe_signal_retire")]
    read = _calls(clear, "pl.read")
    waits = _calls(clear, "pld.system.wait")
    notifies = _calls(clear, "pld.system.notify")
    assert len(read) == 1 and ast.unparse(read[0].args[0]) == "completion_anchor"
    assert len(waits) == 1 and ast.unparse(_keyword(waits[0], "signal")) == "consumed"
    assert ast.unparse(_keyword(waits[0], "expected")) == "final_epoch"
    assert ast.unparse(_keyword(waits[0], "cmp")) == "pld.WaitCmp.Ge"
    assert read[0].lineno < waits[0].lineno < min(call.lineno for call in notifies)
    assert not _calls(clear, "pl.write")
    assert len(notifies) == 4

    reset_by_target = {ast.unparse(_keyword(call, "target")): call for call in notifies}
    expected_offsets = {
        "arrived": "[src, 0]",
        "consumed": "[src, 0]",
        "data_arrived": "[src, e, 0]",
        "combine_arrived": "[src, e, 0]",
    }
    assert set(reset_by_target) == set(expected_offsets)
    for target, offsets in expected_offsets.items():
        call = reset_by_target[target]
        assert ast.unparse(_keyword(call, "peer")) == "my_rank"
        assert ast.unparse(_keyword(call, "offsets")) == offsets
        assert ast.unparse(_keyword(call, "value")) == "0"
        assert ast.unparse(_keyword(call, "op")) == "pld.NotifyOp.Set"

    caller_trees = (_tree("prefill_fwd.py"), _tree("prefill_layer.py"))
    calls = [call for tree in caller_trees for call in _calls(tree, "clear_prefill_moe_signals")]
    assert len(calls) == 2
    for call in calls:
        assert not call.keywords and len(call.args) == 7
        assert ast.unparse(call.args[5]) == "my_rank"
    assert {ast.unparse(call.args[6]) for call in calls} == {"request_last_moe_epoch", "moe_epoch"}
