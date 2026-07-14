# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Source contracts for the Attention-first dynamic-token migration."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "deepseek" / "v4"


def _function(filename: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((MODEL / filename).read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_attention_contract_uses_physical_token_shapes_below_compatibility_boundary() -> None:
    for filename, name in (
        ("prefill_attention_swa.py", "prefill_attention_swa"),
        ("prefill_attention_hca.py", "prefill_attention_hca"),
        ("prefill_attention_csa.py", "prefill_attention_csa"),
    ):
        function = _function(filename, name)
        params = tuple(arg.arg for arg in function.args.args)
        source = ast.unparse(function)
        annotations = [ast.unparse(arg.annotation) for arg in function.args.args if arg.annotation is not None]

        assert params[-1] == "num_tokens"
        assert any("T_DYN" in annotation for annotation in annotations)
        assert "token_count = pl.cast(num_tokens, pl.INDEX)" in source
        assert "num_tokens = pl.tensor.dim(x_hc, 0)" in source


def test_attention_children_do_not_accept_num_tokens() -> None:
    functions = {
        "prefill_compressor_ratio4.py": "prefill_compressor_ratio4",
        "prefill_compressor_ratio128.py": "prefill_compressor_ratio128",
        "prefill_indexer.py": "prefill_indexer",
        "prefill_indexer_compressor.py": "prefill_indexer_compressor",
        "prefill_sparse_attn.py": "prefill_sparse_attn",
    }
    for filename, name in functions.items():
        params = tuple(arg.arg for arg in _function(filename, name).args.args)
        assert "num_tokens" not in params, f"{filename}:{name}"


def test_dynamic_output_stores_do_not_receive_tensor_slices() -> None:
    for filename in ("hc_pre.py", "prefill_indexer.py", "prefill_sparse_attn.py"):
        tree = ast.parse((MODEL / filename).read_text(encoding="utf-8"))
        for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
            tensor_slice_names = {
                node.targets[0].id
                for node in ast.walk(function)
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "pl.slice"
            }
            for node in ast.walk(function):
                if not isinstance(node, ast.Call) or ast.unparse(node.func) != "pl.store":
                    continue
                source = node.args[0]
                assert not (isinstance(source, ast.Call) and ast.unparse(source.func) == "pl.slice")
                assert not (isinstance(source, ast.Name) and source.id in tensor_slice_names)


def test_hc_pre_materializes_post_as_a_vec_tile_before_store() -> None:
    source = ast.unparse(_function("hc_pre.py", "_hc_pre_separate"))
    assert "post_pad_store = pl.create_tensor" in source
    assert "post_pad_store = pl.assemble(post_pad_store, post_pad" in source
    assert "post_tile = pl.load(post_pad_store" in source
    assert "pl.store(post_tile" in source


def test_hc_pre_materializes_mixed_output_as_a_vec_tile_before_store() -> None:
    for name in ("_hc_pre_syncall", "_hc_pre_separate"):
        source = ast.unparse(_function("hc_pre.py", name))
        assert "x_mixed_pad_store = pl.create_tensor" in source
        assert "x_mixed_pad_store = pl.assemble(x_mixed_pad_store, y_bf16" in source
        assert "y_out = pl.load(x_mixed_pad_store" in source
        assert "pl.store(y_out" in source


def test_dynamic_attention_tile_kernels_load_partial_rows_as_tiles() -> None:
    for filename, name in (
        ("rmsnorm.py", "rms_norm"),
        ("qkv_proj_rope.py", "qkv_proj_rope"),
    ):
        function = _function(filename, name)
        calls = {ast.unparse(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)}
        assert "pl.load" in calls
        assert "pl.slice" not in calls


def test_dynamic_attention_tile_reductions_provide_scratch_tiles() -> None:
    for filename, name in (
        ("rmsnorm.py", "rms_norm"),
        ("qkv_proj_rope.py", "qkv_proj_rope"),
    ):
        function = _function(filename, name)
        reductions = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and ast.unparse(node.func) in {"pl.row_sum", "pl.row_max"}
        ]
        assert reductions
        assert all(len(node.args) == 2 for node in reductions)

    rms_source = ast.unparse(_function("rmsnorm.py", "rms_norm"))
    assert "x_sq_sum = pl.tile.full" in rms_source
    assert "norm_w_input = pl.load(norm_w" in rms_source

    qkv_source = ast.unparse(_function("qkv_proj_rope.py", "qkv_proj_rope"))
    assert "pl.gather" not in qkv_source
    assert qkv_source.count("pl.tile.gather") == 6
    for accumulator in ("q_acc", "col_acc", "kv_acc"):
        assert f"{accumulator} = pl.create_tile" in qkv_source
    assert "pl.assemble(qr_fp32, q_acc" not in qkv_source
    assert "pl.assemble(kv_fp32, kv_acc" not in qkv_source
    assert "pl.store(q_acc" in qkv_source
    assert "pl.store(col_acc" in qkv_source
    assert "pl.store(kv_acc" in qkv_source
    assert "pl.pipeline(0, Q_LORA // Q_PROJ_TILE, stage=1)" in qkv_source
    assert "pl.spmd(H * HEAD_DIM // QPROJ_MM_N_TILE, name_hint='qproj_matmul'" in qkv_source
    qproj_n_tile = next(
        node.value
        for node in _tree("qkv_proj_rope.py").body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "QPROJ_MM_N_TILE"
    )
    assert ast.literal_eval(qproj_n_tile) == 512
    qkv_function = _function("qkv_proj_rope.py", "qkv_proj_rope")
    loads = {
        node.targets[0].id: node.value
        for node in ast.walk(qkv_function)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "pl.load"
    }
    for name in ("q_x_chunk_bf16", "qr_i8_chunk", "kv_x_chunk_bf16"):
        assert all(keyword.arg != "valid_shapes" for keyword in loads[name].keywords)
    assert ast.unparse(loads["q_x_chunk_bf16"].args[0]) == "x_matmul"
    assert ast.unparse(loads["kv_x_chunk_bf16"].args[0]) == "x_matmul"
    assert "qkv_dynamic_pad_x" in qkv_source
    assert "with pl.at(level=pl.Level.CORE_GROUP, name_hint='qkv_dynamic_pad_x')" in qkv_source
    assert "pl.spmd(T_MAX, name_hint='qkv_dynamic_pad_x')" not in qkv_source
    assert "qr_i8_matmul[ts0:ts0 + QR_M_TILE" in qkv_source
    assert "dtype=pl.INT8, value=0" not in qkv_source
    assert "pl.full([QR_M_TILE, QR_N_TILE], dtype=pl.FP16, value=0.0)" in qkv_source


def test_sparse_dynamic_padding_is_row_tiled() -> None:
    source = ast.unparse(_function("prefill_sparse_attn.py", "prefill_sparse_attn"))
    assert "prefill_sparse_dynamic_pad_q" in source
    assert "prefill_sparse_dynamic_pad_meta" in source
    assert "pl.slice(q_in, [T, H, HEAD_DIM]" not in source
    assert "q_row = pl.tile.full([1, Q_PAD_COLS]" in source
    assert "q_row = pl.load(q_in_flat" in source
