# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Source contracts for DeepSeek V4 dynamic-token packed prefill."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "deepseek" / "v4"


def _tree(filename: str) -> ast.Module:
    return ast.parse((MODEL / filename).read_text(encoding="utf-8"))


def _function(filename: str, name: str) -> ast.FunctionDef:
    return next(
        node for node in _tree(filename).body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _params(filename: str, name: str) -> tuple[str, ...]:
    return tuple(arg.arg for arg in _function(filename, name).args.args)


def test_prefill_compiled_apis_do_not_accept_num_tokens() -> None:
    functions = {
        "moe.py": ("dispatch", "combine", "moe", "moe_test", "l3_moe"),
        "prefill_attention_swa.py": ("prefill_attention_swa", "prefill_attention_swa_test"),
        "prefill_attention_hca.py": ("prefill_attention_hca", "prefill_attention_hca_test"),
        "prefill_attention_csa.py": ("prefill_attention_csa", "prefill_attention_csa_test"),
        "prefill_compressor_ratio4.py": ("prefill_compressor_ratio4", "prefill_compressor_ratio4_test"),
        "prefill_compressor_ratio128.py": ("prefill_compressor_ratio128", "prefill_compressor_ratio128_test"),
        "prefill_indexer.py": ("prefill_indexer", "prefill_indexer_test"),
        "prefill_indexer_compressor.py": ("prefill_indexer_compressor", "prefill_indexer_compressor_test"),
        "prefill_sparse_attn.py": ("prefill_sparse_attn", "prefill_sparse_attn_test"),
        "prefill_fwd.py": ("prefill_fwd", "l3_prefill_fwd"),
    }
    for filename, names in functions.items():
        for name in names:
            assert "num_tokens" not in _params(filename, name), f"{filename}:{name}"


def test_leaf_apis_expose_the_shared_dynamic_token_symbol() -> None:
    functions = {
        "expert_shared.py": "expert_shared",
        "moe.py": "moe",
        "prefill_attention_swa.py": "prefill_attention_swa",
        "prefill_attention_hca.py": "prefill_attention_hca",
        "prefill_attention_csa.py": "prefill_attention_csa",
        "prefill_compressor_ratio4.py": "prefill_compressor_ratio4",
        "prefill_compressor_ratio128.py": "prefill_compressor_ratio128",
        "prefill_indexer.py": "prefill_indexer",
        "prefill_indexer_compressor.py": "prefill_indexer_compressor",
        "prefill_sparse_attn.py": "prefill_sparse_attn",
    }
    for filename, name in functions.items():
        annotations = [
            ast.unparse(arg.annotation)
            for arg in _function(filename, name).args.args
            if arg.annotation is not None
        ]
        assert any("T_DYN" in annotation for annotation in annotations), f"{filename}:{name}"


def test_packed_layer_passes_request_length_tensors_to_children() -> None:
    source = ast.unparse(_function("prefill_layer.py", "prefill_layer_core"))
    assert "[tile_tokens, HC_MULT, D]" in source
    assert "[TOK_TILE, HC_MULT, D]" not in source
    assert "valid_n" not in source
    assert "num_tokens" not in source


def test_packed_fixture_uses_contiguous_logical_token_offsets() -> None:
    source = ast.unparse(_function("prefill_layer.py", "_resolve_batch"))
    assert "for c in chunk_lens_v" in source
    assert "padded_lens" not in source


def test_bounded_distributed_entry_points_validate_capacity() -> None:
    for filename in ("moe.py", "prefill_fwd.py"):
        source = ast.unparse(_function(filename, "build_tensor_specs"))
        assert "validate_token_count(num_tokens)" in source


def test_dynamic_output_stores_do_not_receive_tensor_slices() -> None:
    for filename in ("hc_pre.py", "prefill_indexer.py", "prefill_sparse_attn.py"):
        tree = _tree(filename)
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
    assert "pl.tile.arange(0, [1, Q_ROPE_T_TILE]" in qkv_source
    assert "pl.mul(pl.cast(pl.tile.arange(0, [1, Q_ROPE_T_TILE]" in qkv_source
    assert "ROPE_DIM_SCALE" in qkv_source
    assert "pl.tile.full([ROPE_DIM, Q_ROPE_T_TILE]" in qkv_source
    assert "qrp_dup_idx = pl.add(pl.cast(qrp_dup_f, target_type=pl.INT32), qrp_row_offset)" in qkv_source
    assert "qrp_swap_idx = pl.add(pl.cast(qrp_swap_f, target_type=pl.INT32), qrp_row_offset)" in qkv_source
    assert "pl.tile.arange(0, [1, KV_RMS_T_TILE]" in qkv_source
    assert "pl.mul(pl.cast(pl.tile.arange(0, [1, KV_RMS_T_TILE]" in qkv_source
    assert "pl.tile.full([ROPE_DIM, KV_RMS_T_TILE]" in qkv_source
    assert "kv_dup_idx = pl.add(pl.cast(kv_dup_f, target_type=pl.INT32), kv_row_offset)" in qkv_source
    assert "kv_swap_idx = pl.add(pl.cast(kv_swap_f, target_type=pl.INT32), kv_row_offset)" in qkv_source
    assert "qr_sum_tmp = pl.create_tile" in qkv_source
    assert "qr_max_tmp = pl.create_tile" in qkv_source
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
    assert "q_rope_dq = pl.create_tensor([t_dim, H * ROPE_DIM], dtype=pl.FP32)" in qkv_source
    assert "q_head_inv_rms_buf = pl.create_tensor([t_dim, H * ROPE_DIM], dtype=pl.FP32)" in qkv_source
    assert "q_head_inv_rms_expanded = pl.row_expand_mul" in qkv_source
    assert "q_head_inv_rms_loaded = pl.load(q_head_inv_rms_buf" in qkv_source
    assert "q_rope_dense_gather" in qkv_source


def test_qr_diagnostics_cover_each_precision_boundary() -> None:
    qkv_source = ast.unparse(_tree("qkv_proj_rope.py"))

    assert "def qr_pipeline_diagnostics_test(" in qkv_source
    assert "qr_proj_diag: pl.Out" in qkv_source
    for checkpoint in (
        "qr_sq_mean_diag",
        "qr_inv_rms_diag",
        "qr_amax_raw_diag",
        "qr_amax_norm_diag",
        "qr_scale_diag",
        "qr_diag",
    ):
        assert checkpoint in qkv_source
    assert "--qr-diagnostics-only" in qkv_source
    assert "--diagnostic-tokens" in qkv_source
    assert "--num-tokens" in qkv_source
    diagnostic_function = ast.unparse(_function("qkv_proj_rope.py", "qr_pipeline_diagnostics_test"))
    assert "qr_diag_proj_matmul" in diagnostic_function
    assert "qr_diag_proj_snapshot" in diagnostic_function
    assert "qr_diag_reduce_quant" in diagnostic_function


def test_q_rope_diagnostics_cover_metadata_and_rotation_boundaries() -> None:
    qkv_source = ast.unparse(_tree("qkv_proj_rope.py"))

    assert "def q_rope_pipeline_diagnostics_test(" in qkv_source
    for checkpoint in (
        "cos_il_diag",
        "sin_signed_diag",
        "swap_idx_diag",
        "swapped_diag",
        "rope_rot_diag",
    ):
        assert checkpoint in qkv_source
    assert "--q-rope-diagnostics-only" in qkv_source
    diagnostic_function = ast.unparse(_function("qkv_proj_rope.py", "q_rope_pipeline_diagnostics_test"))
    assert "q_rope_diag_prepare" in diagnostic_function
    assert "q_rope_diag_materialize" in diagnostic_function
    assert "q_rope_diag_consume" in diagnostic_function


def test_sparse_dynamic_padding_is_row_tiled() -> None:
    source = ast.unparse(_function("prefill_sparse_attn.py", "prefill_sparse_attn"))
    assert "prefill_sparse_dynamic_pad_q" in source
    assert "prefill_sparse_dynamic_pad_meta" in source
    assert "pl.slice(q_in, [T, H, HEAD_DIM]" not in source
    assert "q_row = pl.tile.full([1, Q_PAD_COLS]" in source
    assert "q_row = pl.load(q_in_flat" in source


def test_indexer_dynamic_x_padding_is_row_tiled() -> None:
    for filename, name, stage in (
        ("prefill_indexer.py", "prefill_indexer", "prefill_indexer_dynamic_pad_x"),
        (
            "prefill_indexer_compressor.py",
            "prefill_indexer_compressor",
            "prefill_idx_compressor_dynamic_pad_x",
        ),
    ):
        source = ast.unparse(_function(filename, name))
        assert stage in source
        assert "pl.slice(x_in, [T, D]" not in source
        assert "x_row = pl.tile.full([1, D]" in source
        assert "x_row = pl.load(x_in" in source
        assert "pl.slice(position_ids_in, [T]" not in source


def test_indexer_golden_derives_all_token_reshapes_from_inputs() -> None:
    source = ast.unparse(_function("prefill_indexer.py", "golden_prefill_indexer_core"))
    assert "num_tokens = tensors['x'].shape[0]" in source
    assert ".view(T," not in source
    assert "q.reshape(T * IDX_N_HEADS" not in source
    assert source.count("view(num_tokens,") >= 5


def test_indexer_and_sparse_attention_cli_accept_dynamic_token_counts() -> None:
    for filename in (
        "prefill_indexer.py",
        "prefill_indexer_compressor.py",
        "prefill_sparse_attn.py",
    ):
        source = ast.unparse(_tree(filename))
        assert "parser.add_argument('--num-tokens'" in source
        assert "args.num_tokens" in source


def test_shared_expert_cli_accepts_dynamic_token_counts() -> None:
    source = ast.unparse(_tree("expert_shared.py"))
    assert "parser.add_argument('--num-tokens'" in source
    assert "build_tensor_specs(num_tokens=args.num_tokens)" in source


def test_moe_frontend_diagnostics_cover_each_local_precision_boundary() -> None:
    source = ast.unparse(_tree("moe.py"))
    diagnostic_function = ast.unparse(_function("moe.py", "moe_frontend_diagnostics_test"))

    assert "--frontend-diagnostics-only" in source
    assert "hc_pre(" in diagnostic_function
    assert "gate_done = gate(" in diagnostic_function
    assert "expert_shared(" in diagnostic_function
    for checkpoint in (
        "x_mixed_diag",
        "post_diag",
        "comb_diag",
        "x_norm_diag",
        "x_norm_i8_diag",
        "x_norm_scale_diag",
        "indices_diag",
        "weights_diag",
        "sh_diag",
    ):
        assert checkpoint in diagnostic_function


def test_indexer_preserves_dynamic_shape_at_inner_compressor_boundary() -> None:
    source = ast.unparse(_function("prefill_indexer.py", "prefill_indexer"))
    assert "prefill_indexer_compressor(x_in," in source
    assert "prefill_indexer_compressor(x," not in source


def test_indexer_paths_read_dynamic_metadata_directly() -> None:
    indexer_source = ast.unparse(_function("prefill_indexer.py", "prefill_indexer"))
    compressor_source = ast.unparse(_function("prefill_indexer_compressor.py", "prefill_indexer_compressor"))
    for metadata in ("position_ids", "idx_slot_mapping", "inner_state_slot_mapping"):
        assert f"{metadata} = pl.create_tensor([T]" not in indexer_source
        assert f"{metadata} = pl.create_tensor([T]" not in compressor_source
        assert f"{metadata}_in" in indexer_source
        assert f"pl.read({metadata}_in" in compressor_source
    assert "pl.read(position_ids_in" in indexer_source


def test_dynamic_prefill_compressors_use_constant_physical_matmul_tiles() -> None:
    for filename, name in (
        ("prefill_compressor_ratio4.py", "prefill_compressor_ratio4"),
        ("prefill_compressor_ratio128.py", "prefill_compressor_ratio128"),
    ):
        source = ast.unparse(_function(filename, name))
        assert "pl.create_tensor([matmul_tokens, OUT_TILE]" not in source
        assert "pl.create_tensor([TOKEN_M_TILE, OUT_TILE]" in source
        assert "pl.slice(x" in source
        assert "[TOKEN_M_TILE, K_TILE]" in source
        assert "valid_shape=[valid_rows, K_TILE]" in source
        assert "[matmul_tokens, K_TILE]" not in source


def test_dynamic_full_prefill_tile_kernels_load_partial_rows_as_tiles() -> None:
    for filename, name in (
        ("gate.py", "_gate_core"),
        ("expert_shared.py", "expert_shared"),
        ("hc_head.py", "hc_head"),
    ):
        function = _function(filename, name)
        calls = {ast.unparse(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)}
        assert "pl.load" in calls
        assert "pl.slice" not in calls


def test_dynamic_full_prefill_tile_reductions_provide_scratch_tiles() -> None:
    for filename, name in (
        ("gate.py", "_gate_core"),
        ("hc_head.py", "hc_head"),
    ):
        function = _function(filename, name)
        reductions = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and ast.unparse(node.func) in {"pl.row_sum", "pl.row_max"}
        ]
        assert reductions
        assert all(len(node.args) == 2 for node in reductions)

    hc_head_source = ast.unparse(_function("hc_head.py", "hc_head"))
    assert "sq_sum = pl.tile.full" in hc_head_source
    assert "acc = pl.create_tile" in hc_head_source
    assert "pl.assemble(inv_rms, inv" not in hc_head_source
    assert "pl.assemble(mixes_raw, acc" not in hc_head_source
    assert "pl.store(pl.set_validshape(inv" in hc_head_source
    assert "pl.store(acc" in hc_head_source
    assert "y = pl.reshape(y_flat" not in hc_head_source

    gate_source = ast.unparse(_function("gate.py", "_gate_core"))
    assert "sq_sum = pl.row_sum" in gate_source
    assert "pl.reshape(pl.row_sum(pl.mul(rms_x, rms_x)" not in gate_source
    assert "pl.recip(pl.sqrt(" in gate_source
    assert "an_w_input = pl.load(norm_w" in gate_source
    assert "xn_amax = pl.row_max" in gate_source
    assert "pl.reshape(pl.row_max(pl.abs(xn_a_f32)" not in gate_source
    assert "xn_a_f32 = pl.load(x_norm_gate_buf" in gate_source
    assert "xn_q_f32 = pl.load(x_norm_gate_buf" in gate_source
    assert "topk_vals_pad = pl.load(topk_vals_pad_buf" in gate_source
    assert "pl.store(an_gate_out" in gate_source
    assert "x_norm_gate_buf[t0:t0 + T_TILE" not in gate_source
    assert "deps=[norm_tid]" in gate_source
    assert "deps=[gate_tid]" in gate_source
    assert "return _quant_tid" in gate_source

    shared_source = ast.unparse(_function("expert_shared.py", "expert_shared"))
    assert "sh_dynamic_pad" in shared_source
    assert "deps=[late_dep]" in shared_source
    assert "x_local_i8_pad" in shared_source
    assert "x_local_scale_dq_pad" in shared_source
    assert shared_source.count("deps=[pad_tid]") == 2

    dispatch_source = ast.unparse(_function("moe.py", "dispatch"))
    assert "name_hint='dispatch_meta', deps=[late_dep]" in dispatch_source
    assert "name_hint='dispatch_push', deps=[late_dep]" in dispatch_source
    assert "deps=[pad_tid, gate_tid, up_tid]" in shared_source
    assert "sh_tile_buf = pl.create_tensor" in shared_source
    assert "sh_tile_buf = pl.assemble(sh_tile_buf, y_bf16" in shared_source
    assert "y_bf16_tile = pl.load(sh_tile_buf" in shared_source
    assert "pl.store(y_bf16_tile" in shared_source
