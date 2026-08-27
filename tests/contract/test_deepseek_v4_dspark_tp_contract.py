# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static contracts for the DeepSeek-V4 DSpark TP data flow."""

import ast
from pathlib import Path


MODEL_DIR = Path(__file__).parents[2] / "models" / "deepseek_v4_flash_dspark"


def _source(name: str) -> str:
    return (MODEL_DIR / name).read_text(encoding="utf-8")


def _tree(name: str) -> ast.Module:
    return ast.parse(_source(name))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _call_names(function: ast.FunctionDef) -> set[str]:
    return {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_decode_uses_main_tp_output_collectives_without_hidden_allgather():
    source = _source("dspark_drafter.py")
    drafter = _function(ast.parse(source), "draft_layer")
    calls = _call_names(drafter)

    assert "o_group_a2a" in calls
    assert "decode_sharded_o_projection_reduce_scatter" in calls
    assert "dspark_cp_barrier" not in source
    assert "reset_dspark_cp_signal" not in source
    assert "dspark_cp_allgather_hidden" not in source


def test_decode_host_abi_keeps_context_and_cache_rank_local():
    host = _function(_tree("dspark_drafter.py"), "l3_dspark_drafter")
    annotations = {
        argument.arg: ast.unparse(argument.annotation)
        for argument in host.args.args
        if argument.annotation is not None
    }

    assert "[N_RANKS, T_MAIN_DYN]" in annotations["context_position_ids"]
    assert (
        "[N_RANKS, DSPARK_DRAFT_LAYERS, T_MAIN_DYN]"
        in annotations["context_slot_mapping"]
    )
    assert "query_group_position_ids" not in annotations
    assert "query_group_slot_mapping" not in annotations
    assert "hidden_gather_window" not in annotations
    assert "hidden_gather_signal" not in annotations


def test_decode_o_projection_weights_are_tp_sharded():
    drafter = _function(_tree("dspark_drafter.py"), "l3_dspark_drafter")
    annotations = {
        argument.arg: ast.unparse(argument.annotation)
        for argument in drafter.args.args
        if argument.annotation is not None
    }

    assert "DSPARK_DRAFT_LAYERS * LOCAL_O_GROUPS" in annotations["wo_a"]
    assert "LOCAL_O_WIDTH" in annotations["wo_b"]
    assert "O_GROUPS * O_LORA" not in annotations["wo_b"]


def test_drafter_calls_match_the_tp_window_abi():
    tree = _tree("dspark_drafter.py")
    draft_layer = _function(tree, "draft_layer")
    drafter = _function(tree, "dspark_drafter")
    host = _function(tree, "l3_dspark_drafter")

    layer_calls = [
        node
        for node in ast.walk(drafter)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "draft_layer"
    ]
    host_calls = [
        node
        for node in ast.walk(host)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dspark_drafter"
    ]

    assert len(layer_calls) == 3
    assert all(len(call.args) == len(draft_layer.args.args) for call in layer_calls)
    assert len(host_calls) == 1
    assert len(host_calls[0].args) == len(drafter.args.args)


def test_dspark_attention_only_produces_grouped_heads():
    source = _source("dspark_attention.py")
    tree = ast.parse(source)
    attention = _function(tree, "dspark_attention")
    argument_names = {argument.arg for argument in attention.args.args}

    assert "o_packed_heads" in argument_names
    assert "wo_a" not in argument_names
    assert "wo_b" not in argument_names
    assert "wo_b_scale" not in argument_names
    assert "kv_x" not in argument_names
    assert "kv_position_ids" not in argument_names
    assert 'name_hint="dspark_grouped_head_compact"' in source


def test_prefill_and_decode_keep_separate_rank_local_host_contracts():
    drafter_source = _source("dspark_drafter.py")
    prefill_source = _source("dspark_prefill.py")

    assert "local_context_tokens = batch * DECODE_SEQ" in drafter_source
    assert "local_context_tokens = PREFILL_TOKENS" in drafter_source
    assert "PREFILL_TOKENS // TP" not in drafter_source
    assert 'build_tensor_specs(args.batch, mode="prefill")' in prefill_source


def test_dynamic_batch_contract_is_unchanged():
    tree = _tree("dspark_drafter.py")
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "DSPARK_SUPPORTED_BATCHES"
            for target in node.targets
        )
    )

    assert ast.literal_eval(assignment.value) == (4, 8, 12, 16)


def test_imported_parallel_defaults_do_not_change_standalone_harnesses():
    for name, assignment_name in (
        ("decode_o_proj.py", "_TP_DEFAULT"),
        ("moe.py", "_EP_DEFAULT"),
    ):
        tree = _tree(name)
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == assignment_name
                for target in node.targets
            )
        )
        expected = (
            "2 if __name__ == '__main__' else config.TP"
            if assignment_name == "_TP_DEFAULT"
            else "2 if __name__ == '__main__' else config.EP"
        )
        assert ast.unparse(assignment.value) == expected
