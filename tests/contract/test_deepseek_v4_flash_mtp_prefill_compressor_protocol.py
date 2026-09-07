# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static contracts for DeepSeek-V4 Flash prefill compressor partial tails."""

import ast
from pathlib import Path


MODEL_DIR = Path(__file__).parents[2] / "models" / "deepseek_v4_flash_mtp"


def _tensor(shape: str, dtype: str) -> str:
    return f"pl.Tensor[[{shape}], pl.{dtype}]"


MAIN_ABI = {
    "x": _tensor("T, D", "BF16"),
    "compress_state": _tensor("STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM", "FP32"),
    "compress_state_block_table": _tensor("CSA_STATE_MAX_BLOCKS", "INT32"),
    "wkv": _tensor("OUT_DIM, D", "BF16"),
    "wgate": _tensor("OUT_DIM, D", "BF16"),
    "ape": _tensor("COMPRESS_RATIO, OUT_DIM", "FP32"),
    "norm_w": _tensor("HEAD_DIM", "BF16"),
    "freqs_cos": _tensor("MAX_SEQ_LEN, ROPE_HEAD_DIM", "BF16"),
    "freqs_sin": _tensor("MAX_SEQ_LEN, ROPE_HEAD_DIM", "BF16"),
    "cmp_kv": _tensor("CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM", "BF16"),
    "position_ids": _tensor("T", "INT32"),
    "num_tokens": "pl.Scalar[pl.INT32]",
    "cmp_slot_mapping": _tensor("T", "INT64"),
    "state_slot_mapping": _tensor("T", "INT64"),
    "completion": "pl.Array[1, pl.TASK_ID]",
}

INDEXER_ABI = {
    "x": _tensor("T, D", "BF16"),
    "compress_state": (
        "pl.InOut["
        + _tensor("STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM", "FP32")
        + "]"
    ),
    "inner_compress_state_block_table": _tensor("INNER_STATE_MAX_BLOCKS", "INT32"),
    "wkv": _tensor("OUT_DIM, D", "BF16"),
    "wgate": _tensor("OUT_DIM, D", "BF16"),
    "ape": _tensor("COMPRESS_RATIO, OUT_DIM", "FP32"),
    "norm_w": _tensor("HEAD_DIM", "BF16"),
    "freqs_cos": _tensor("MAX_SEQ_LEN, ROPE_HEAD_DIM", "BF16"),
    "freqs_sin": _tensor("MAX_SEQ_LEN, ROPE_HEAD_DIM", "BF16"),
    "hadamard": _tensor("HEAD_DIM, HEAD_DIM", "BF16"),
    "idx_kv_cache": _tensor("IDX_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, HEAD_DIM", "INT8"),
    "idx_kv_scale": _tensor("IDX_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, 1", "FP32"),
    "idx_block_table": _tensor("IDX_CACHE_MAX_BLOCKS", "INT32"),
    "position_ids": _tensor("T", "INT32"),
    "num_tokens": "pl.Scalar[pl.INT32]",
    "idx_slot_mapping": _tensor("T", "INT64"),
    "inner_state_slot_mapping": _tensor("T", "INT64"),
}


def _tree(file_name: str) -> ast.Module:
    return ast.parse((MODEL_DIR / file_name).read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _call_name(node.value)
        return f"{owner}.{node.attr}" if owner else node.attr
    return ""


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


def _mapping_guard(context: ast.With) -> ast.If:
    matches = [
        node
        for node in ast.walk(context)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "dst_row_raw >= 0"
    ]
    assert len(matches) == 1
    return matches[0]


def _assert_abi(function: ast.FunctionDef, expected: dict[str, str], returned: str) -> None:
    arguments = function.args
    assert not arguments.posonlyargs
    assert not arguments.kwonlyargs
    assert arguments.vararg is None
    assert arguments.kwarg is None
    assert not arguments.defaults
    assert not arguments.kw_defaults
    assert {
        argument.arg: ast.unparse(argument.annotation) for argument in arguments.args
    } == expected
    returns = [node.value for node in ast.walk(function) if isinstance(node, ast.Return)]
    assert [ast.unparse(value) for value in returns] == [returned]


def test_invalid_partial_tail_slots_do_not_issue_keepalive_writes() -> None:
    cases = (
        ("prefill_compressor_ratio4.py", "compressor_ratio4", "prefill_c4_cache_write"),
        (
            "prefill_indexer_compressor.py",
            "_prefill_indexer_compressor_with_completion",
            "prefill_idx_c4_cache_write",
        ),
        (
            "prefill_indexer_compressor.py",
            "_prefill_indexer_compressor_with_completion",
            "prefill_idx_c4_scale_scatter",
        ),
    )
    for file_name, function_name, name_hint in cases:
        function = _function(_tree(file_name), function_name)
        assert not _mapping_guard(_context(function, name_hint)).orelse


def test_active_pool_taskid_orders_state_update() -> None:
    cases = (
        (
            "prefill_compressor_ratio4.py",
            "compressor_ratio4",
            "prefill_c4_softmax_pool",
            "prefill_c4_state_update",
        ),
        (
            "prefill_indexer_compressor.py",
            "_prefill_indexer_compressor_with_completion",
            "prefill_idx_c4_softmax_pool",
            "prefill_idx_c4_state_update",
        ),
    )
    for file_name, function_name, pool_hint, update_hint in cases:
        function = _function(_tree(file_name), function_name)
        pool = _context(function, pool_hint)
        pool_call = pool.items[0].context_expr
        assert _call_name(pool_call.func) == "pl.spmd"
        assert ast.unparse(pool_call.args[0]) == "active_pool_blocks"
        pool_tid = pool.items[0].optional_vars
        assert isinstance(pool_tid, ast.Name)
        block_indices = [
            node.value
            for node in pool.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _call_name(node.value.func) == "pl.tile.get_block_idx"
        ]
        assert len(block_indices) == 1

        update_call = _context(function, update_hint).items[0].context_expr
        assert _call_name(update_call.func) == "pl.spmd"
        deps = _keyword(update_call, "deps")
        assert isinstance(deps, ast.List)
        assert pool_tid.id in [ast.unparse(dependency) for dependency in deps.elts]


def test_prefill_compressor_abi_is_stable() -> None:
    main = _function(_tree("prefill_compressor_ratio4.py"), "compressor_ratio4")
    indexer_tree = _tree("prefill_indexer_compressor.py")
    private_indexer = _function(indexer_tree, "_prefill_indexer_compressor_with_completion")
    public_indexer = _function(indexer_tree, "prefill_indexer_compressor")

    _assert_abi(main, MAIN_ABI, "(cmp_kv, compress_state)")
    _assert_abi(
        private_indexer,
        {**INDEXER_ABI, "completion": "pl.Array[1, pl.TASK_ID]"},
        "(idx_kv_cache, idx_kv_scale, compress_state)",
    )
    _assert_abi(
        public_indexer,
        INDEXER_ABI,
        "(idx_kv_cache_out, idx_kv_scale_out, compress_state_out)",
    )

    completion_arrays = [
        node.value
        for node in ast.walk(public_indexer)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "completion" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and _call_name(node.value.func) == "pl.array.create"
    ]
    assert len(completion_arrays) == 1
    assert [ast.unparse(argument) for argument in completion_arrays[0].args] == ["1", "pl.TASK_ID"]
    helper_calls = [
        node
        for node in ast.walk(public_indexer)
        if isinstance(node, ast.Call)
        and _call_name(node.func) == "_prefill_indexer_compressor_with_completion"
    ]
    assert len(helper_calls) == 1
    assert ast.unparse(helper_calls[0].args[-1]) == "completion"
