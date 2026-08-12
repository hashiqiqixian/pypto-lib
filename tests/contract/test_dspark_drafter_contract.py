# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import ast
import sys
from pathlib import Path

import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek_v4_flash_dspark"
sys.path.insert(0, str(_MODEL_DIR))

from config import (  # noqa: E402
    DECODE_SEQ,
    DSPARK_DRAFT_LAYERS,
    DSPARK_MOE_TOKENS,
    DSPARK_MARKOV_RANK,
    DSPARK_MOE_EPOCHS,
    DSPARK_NOISE_TOKEN_ID,
    DSPARK_QUERY_WIDTH,
    DSPARK_QUERY_TOKENS,
    DSPARK_SUPPORTED_BATCHES,
    DSPARK_SWA_INDEX_WIDTH,
    DSPARK_TARGET_HIDDEN_LAYERS,
)
from dspark_reference import (  # noqa: E402
    backbone_reference,
    markov_sample_reference,
    noncausal_attention_reference,
    noncausal_metadata_reference,
    prepare_reference,
)
from dspark_backbone import (  # noqa: E402
    N_RANKS,
    _anchor_position_set,
    _block_tables,
    build_tensor_specs as build_backbone_tensor_specs,
)
from markov_sample import build_tensor_specs as build_markov_tensor_specs  # noqa: E402


def _metadata_inputs(
    context_position: int,
    *,
    layers: int = 3,
    batch: int = 1,
    logical_blocks: int = 16,
    physical_stride: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    context_positions = torch.full((batch,), context_position, dtype=torch.int32)
    block_tables = torch.empty(layers, batch, logical_blocks, dtype=torch.int32)
    for layer in range(layers):
        logical = torch.arange(logical_blocks, dtype=torch.int32)
        block_tables[layer] = logical + layer * physical_stride
    return context_positions, block_tables


def _build_metadata(
    context_position: int,
    *,
    block_size: int = 4,
    window_size: int = 5,
    index_width: int = 16,
    block_tables: torch.Tensor | None = None,
):
    if block_tables is None:
        context_positions, block_tables = _metadata_inputs(context_position)
    else:
        context_positions = torch.tensor([context_position], dtype=torch.int32)
    return noncausal_metadata_reference(
        context_positions,
        block_tables,
        block_size=block_size,
        window_size=window_size,
        query_width=7,
        index_width=index_width,
    )


def test_config_matches_issue_model_contract() -> None:
    assert DSPARK_DRAFT_LAYERS == 3
    assert DSPARK_TARGET_HIDDEN_LAYERS == (40, 41, 42)
    assert DSPARK_QUERY_WIDTH == 7
    assert DSPARK_NOISE_TOKEN_ID == 128799
    assert DSPARK_MARKOV_RANK == 256
    assert DSPARK_MOE_EPOCHS == (1, 2, 3)
    assert DSPARK_QUERY_TOKENS == 16 * 7
    assert DSPARK_MOE_TOKENS == 16 * 8
    assert DSPARK_SWA_INDEX_WIDTH == 192


def test_prepare_projects_three_states_and_builds_anchor_noise_layout() -> None:
    torch.manual_seed(3)
    batch, hidden_size, vocab = 2, 4, 12
    target_hidden = torch.randn(batch * 2, 3 * hidden_size)
    projection = torch.randn(hidden_size, 3 * hidden_size)
    norm_weight = torch.randn(hidden_size)
    anchors = torch.tensor([2, 5])
    embedding = torch.randn(vocab, hidden_size)
    main_hidden, query_ids, query_hc = prepare_reference(
        target_hidden,
        projection,
        norm_weight,
        anchors,
        embedding,
        noise_token_id=7,
        query_width=7,
        query_pad=8,
        max_batch=16,
        hc_mult=4,
        eps=1e-6,
    )
    manual = target_hidden.matmul(projection.t())
    manual = manual * torch.rsqrt(manual.square().mean(-1, keepdim=True) + 1e-6)
    manual = manual * norm_weight
    torch.testing.assert_close(main_hidden, manual)
    assert query_ids[: batch * 7].view(batch, 7)[:, 0].tolist() == anchors.tolist()
    assert torch.equal(query_ids[: batch * 7].view(batch, 7)[:, 1:], torch.full((batch, 6), 7))
    assert query_ids[batch * 7 :].eq(0).all()
    assert query_ids.shape == (16 * 8,)
    assert query_hc.shape == (16 * 8, 4, hidden_size)
    assert query_hc[batch * 7 :].eq(embedding[0].view(1, 1, hidden_size)).all()


def test_noncausal_rows_include_every_query_kv() -> None:
    _, _, indices, lengths = _build_metadata(2)
    active_row = indices[0, 0]
    assert int(lengths[0, 0]) == 10
    future_slot = int(active_row[int(lengths[0, 0]) - 1])
    assert future_slot in active_row.tolist()

    cache = torch.zeros(32, 2)
    cache[future_slot] = torch.tensor([20.0, 5.0])
    query = torch.tensor([[1.0, 0.0]]).expand(7, -1).contiguous()
    before = noncausal_attention_reference(
        query,
        cache,
        active_row.expand(7, -1),
        lengths[0, 0].expand(7),
        1.0,
    )
    cache[future_slot] = 0
    after = noncausal_attention_reference(
        query,
        cache,
        active_row.expand(7, -1),
        lengths[0, 0].expand(7),
        1.0,
    )
    assert not torch.equal(before[0], after[0])


def test_short_context_keeps_all_available_history() -> None:
    _, _, indices, lengths = _build_metadata(1)
    assert int(lengths[0, 0]) == 9
    assert indices[0, 0, :9].tolist() == list(range(9))


def test_full_window_keeps_window_plus_full_query_block() -> None:
    _, _, indices, lengths = _build_metadata(9)
    assert int(lengths[0, 0]) == 12
    assert indices[0, 0, :12].tolist() == list(range(5, 17))


def test_page_boundary_maps_each_visible_position() -> None:
    context_slots, query_slots, indices, _ = _build_metadata(3)
    assert int(context_slots[0, 0]) == 3
    assert query_slots[0, 0, :7].tolist() == list(range(4, 11))
    assert indices[0, 0, :11].tolist() == list(range(11))


def test_ring_wrap_uses_block_table_physical_mapping() -> None:
    block_table = (
        torch.tensor(
            [[[7, 2, 7, 2, 7, 2, 7, 2, 7, 2, 7, 2, 7, 2, 7, 2]]],
            dtype=torch.int32,
        )
        .expand(3, -1, -1)
        .clone()
    )
    context_slots, query_slots, indices, lengths = _build_metadata(
        9,
        block_tables=block_table,
    )
    assert int(context_slots[0, 0]) == 7 * 4 + 1
    assert int(query_slots[0, 0, 0]) == 7 * 4 + 2
    assert int(lengths[0, 0]) == 12
    assert indices[0, 0, :12].tolist() == [9, 10, 11, 28, 29, 30, 31, 8, 9, 10, 11, 28]


def test_three_layers_use_isolated_cache_tables() -> None:
    context_slots, query_slots, indices, _ = _build_metadata(5)
    assert len(set(context_slots[:, 0].tolist())) == 3
    assert len(set(query_slots[:, 0, 0].tolist())) == 3
    assert not torch.equal(indices[0], indices[1])
    assert not torch.equal(indices[1], indices[2])


def test_backbone_reference_exposes_three_intermediates_and_head_hidden() -> None:
    query_hc = torch.arange(2 * 8 * 2 * 3, dtype=torch.float32).reshape(2, 8, 2, 3)
    layers = [
        lambda hidden, layer_id: hidden + layer_id + 1,
        lambda hidden, layer_id: hidden * (layer_id + 1),
        lambda hidden, layer_id: hidden - layer_id,
    ]
    intermediates, head_hidden = backbone_reference(
        query_hc,
        layers,
        lambda hidden: hidden.sum(dim=-2),
        7,
    )
    assert len(intermediates) == 3
    torch.testing.assert_close(intermediates[0], query_hc + 1)
    torch.testing.assert_close(intermediates[1], (query_hc + 1) * 2)
    torch.testing.assert_close(intermediates[2], (query_hc + 1) * 2 - 2)
    assert head_hidden.shape == (2, 7, 3)


def test_backbone_source_orders_attention_hc_post_moe_and_clears_each_layer() -> None:
    source = (_MODEL_DIR / "draft_backbone.py").read_text()
    tree = ast.parse(source)
    helper_source = (_MODEL_DIR / "dspark_draft_layer.py").read_text()
    assert helper_source is not None
    assert helper_source.index("hc_pre_flat_with_inv_rms(") < helper_source.index("dspark_attention(")
    assert helper_source.index("dspark_attention(") < helper_source.index("hc_post_prefill(")
    assert helper_source.index("hc_post_prefill(") < helper_source.index("moe(")
    helper_tree = ast.parse(helper_source)
    helper = next(
        node for node in helper_tree.body if isinstance(node, ast.FunctionDef) and node.name == "dspark_draft_layer"
    )
    helper_body = ast.get_source_segment(helper_source, helper)
    assert helper_body is not None
    assert "padded_attention = pl.create_tensor([T, D]" in helper_body
    assert "hc_post_prefill(" in helper_body
    assert "active_tokens," in helper_body
    assert "query_hc[0:T_QUERY" not in helper_body
    assert "def dspark_query_rms_norm(" in helper_source
    assert "from rmsnorm import rms_norm" not in helper_source
    backbone = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "draft_backbone"
    )
    backbone_args = [arg.arg for arg in backbone.args.args]
    assert backbone_args[:1] == ["target_hidden"]
    assert backbone_args[-15:-10] == [
        "initial_hidden",
        "hc_attn_inv_rms_0",
        "hc_attn_inv_rms_1",
        "hc_attn_inv_rms_2",
        "head_hidden",
    ]
    backbone_source = ast.get_source_segment(source, backbone)
    assert backbone_source is not None
    assert backbone_source.count("dspark_draft_layer(") == 3
    assert backbone_source.count("dspark_context_kv_query(") == 3
    assert backbone_source.count("rebase_moe_signals(") == 2
    assert backbone_source.count("= dspark_moe_barrier(") == 2
    assert backbone_source.count("clear_moe_signals(") == 1
    assert backbone_source.count("clear_dspark_moe_barrier(") == 1
    for layer in range(3):
        assert backbone_source.count(f"kv_caches_{layer}") == 3
    assert "hc_head(" in backbone_source
    assert "context_main_x = pl.create_tensor([T_QUERY, D]" in backbone_source
    assert "context_positions = pl.create_tensor([T_QUERY]" in backbone_source
    for layer in range(3):
        assert f"context_slots_{layer} = pl.create_tensor([T_QUERY]" in backbone_source
        assert f"slot_{layer} = pl.cast(-1, pl.INT64)" in backbone_source
    assert "context_row = (token + 1) * DECODE_SEQ - 1" in backbone_source
    for epoch_index in range(3):
        assert f"DSPARK_MOE_EPOCHS[{epoch_index}]" in backbone_source


def test_backbone_reuses_dspark_leaf_operators() -> None:
    prepare_source = (_MODEL_DIR / "dspark_prepare.py").read_text()
    backbone_source = (_MODEL_DIR / "draft_backbone.py").read_text()
    layer_source = (_MODEL_DIR / "dspark_draft_layer.py").read_text()
    sampler_source = (_MODEL_DIR / "markov_sample.py").read_text()

    assert "from dspark_proj import dspark_proj" in prepare_source
    assert prepare_source.count("dspark_proj(") == 1
    assert "from lookup_embedding import lookup_embedding" in prepare_source
    assert prepare_source.count("lookup_embedding(") == 1
    assert "from dspark_context_kv import dspark_context_kv_query" in backbone_source
    assert "from dspark_attention import dspark_attention" in layer_source
    attention_source = (_MODEL_DIR / "dspark_attention.py").read_text()
    assert "swa_lens_2d = pl.reshape(swa_lens, [B, 1])" in attention_source
    assert "swa_lens[v_b0 : v_b0 + BIAS_B_TILE]" not in attention_source
    assert "from markov_head import markov_head" in sampler_source
    assert sampler_source.count("markov_head(") == 1
    assert not (_MODEL_DIR / "dspark_sparse_attention.py").exists()


def test_composed_backbone_uses_unambiguous_dynamic_symbols() -> None:
    qkv_source = (_MODEL_DIR / "qkv_proj_rope.py").read_text()
    rms_source = (_MODEL_DIR / "rmsnorm.py").read_text()

    assert 'pl.dynamic("QKV_Q_T_DYN")' in qkv_source
    assert 'pl.dynamic("QKV_KV_T_DYN")' in qkv_source
    assert 'pl.dynamic("QKV_ROPE_T_DYN")' in qkv_source
    assert 'pl.dynamic("RMS_NORM_T_DYN")' in rms_source
    qkv_tree = ast.parse(qkv_source)
    q_proj = next(
        node for node in qkv_tree.body if isinstance(node, ast.FunctionDef) and node.name == "q_proj_rope"
    )
    assert isinstance(q_proj.decorator_list[0], ast.Call)
    assert isinstance(q_proj.decorator_list[0].func, ast.Attribute)
    assert q_proj.decorator_list[0].func.attr == "inline"
    assert any(keyword.arg == "auto_scope" for keyword in q_proj.decorator_list[0].keywords)


def test_markov_reference_matches_base_bias_and_final_ids() -> None:
    torch.manual_seed(4)
    batch, width, hidden_size, vocab, rank = 3, 7, 4, 9, 2
    head_hidden = torch.randn(batch, width, hidden_size)
    norm_weight = torch.randn(hidden_size)
    lm_head = torch.randn(vocab, hidden_size)
    anchors = torch.tensor([0, 1, 2])
    markov_w1 = torch.randn(vocab, rank)
    markov_w2 = torch.randn(vocab, rank)
    base_logits, biases, draft_ids = markov_sample_reference(
        head_hidden,
        norm_weight,
        lm_head,
        anchors,
        markov_w1,
        markov_w2,
        1e-6,
    )
    previous = anchors
    for step in range(width):
        expected_bias = markov_w1.index_select(0, previous).matmul(markov_w2.t())
        torch.testing.assert_close(biases[:, step], expected_bias)
        previous = torch.argmax(base_logits[:, step] + expected_bias, dim=-1)
        assert torch.equal(draft_ids[:, step], previous)


def test_markov_causality_changes_later_ids_without_changing_base_logits() -> None:
    head_hidden = torch.zeros(1, 7, 2)
    norm_weight = torch.ones(2)
    lm_head = torch.zeros(3, 2)
    markov_w1 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    markov_w2 = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    base_a, _, ids_a = markov_sample_reference(
        head_hidden,
        norm_weight,
        lm_head,
        torch.tensor([0]),
        markov_w1,
        markov_w2,
        1e-6,
    )
    base_b, _, ids_b = markov_sample_reference(
        head_hidden,
        norm_weight,
        lm_head,
        torch.tensor([1]),
        markov_w1,
        markov_w2,
        1e-6,
    )
    torch.testing.assert_close(base_a, base_b)
    assert int(ids_a[0, 0]) != int(ids_b[0, 0])
    assert not torch.equal(ids_a[0, 1:], ids_b[0, 1:])


def test_supported_dynamic_batches_preserve_public_shapes() -> None:
    assert DSPARK_SUPPORTED_BATCHES == (4, 8, 12, 16)
    for batch in DSPARK_SUPPORTED_BATCHES:
        head_hidden = torch.zeros(batch, 7, 2)
        _, _, draft_ids = markov_sample_reference(
            head_hidden,
            torch.ones(2),
            torch.zeros(5, 2),
            torch.zeros(batch, dtype=torch.int64),
            torch.zeros(5, 2),
            torch.zeros(5, 2),
            1e-6,
        )
        assert draft_ids.shape == (batch, 7)


def test_markov_runtime_specs_cover_minimum_hardware_batches() -> None:
    for batch in (4, 16):
        specs = {spec.name: spec for spec in build_markov_tensor_specs(batch)}
        assert specs["head_hidden"].shape == [batch, 7, 4096]
        assert specs["draft_token_ids"].shape == [batch, 7]
        assert specs["draft_token_ids"].is_output


def test_backbone_runtime_specs_cover_multirank_public_outputs() -> None:
    for batch in (4, 16):
        specs = {spec.name: spec for spec in build_backbone_tensor_specs(batch)}
        assert specs["target_hidden"].shape == [N_RANKS, batch * DECODE_SEQ, 3 * 4096]
        assert specs["context_position_ids"].shape == [N_RANKS, DSPARK_QUERY_TOKENS]
        assert specs["context_slot_mapping"].shape == [
            N_RANKS,
            DSPARK_DRAFT_LAYERS,
            DSPARK_QUERY_TOKENS,
        ]
        assert specs["head_hidden"].shape == [N_RANKS, batch, 7, 4096]
        assert specs["head_hidden"].is_output
        assert specs["initial_hidden"].shape == [N_RANKS, 128, 4, 4096]
        assert specs["initial_hidden"].is_output
        assert specs["hc_attn_inv_rms"].shape == [N_RANKS, DSPARK_DRAFT_LAYERS, 128, 1]
        assert specs["hc_attn_inv_rms"].is_output
        for layer in range(3):
            cache = specs[f"kv_caches_{layer}"]
            assert cache.is_output
            assert cache.init_value is not None


def test_backbone_harness_covers_context_and_cache_boundaries() -> None:
    positions = _anchor_position_set(16)
    assert int(positions.min()) < 128
    assert int(positions.max()) >= 128
    assert any(int(position) % 64 == 63 for position in positions)

    tables = _block_tables(16)
    assert tables.shape[:3] == (N_RANKS, 3, 16)
    assert not torch.equal(tables[:, 0], tables[:, 1])
    assert not torch.equal(tables[:, 1], tables[:, 2])
    assert int(tables.max()) < 512


def test_backbone_harness_provisions_the_full_three_layer_ring_heap() -> None:
    source = (_MODEL_DIR / "dspark_backbone.py").read_text()
    assert "_DSPARK_RING_HEAP = (4 * 1024 * 1024 * 1024,) * 4" in source
    assert "ring_heap=_DSPARK_RING_HEAP" in source
    assert "enable_dep_gen=True" in source
    assert 'parser.add_argument("--enable-scope-stats", action="store_true")' in source
    assert 'parser.add_argument("--dump-args", type=int, choices=(0, 1, 2, 3), default=0)' in source
    assert "enable_dump_args=args.dump_args" in source
    assert "enable_scope_stats=args.enable_scope_stats" in source


def test_dspark_projection_keeps_cube_and_vector_work_separate() -> None:
    source = (_MODEL_DIR / "dspark_proj.py").read_text()
    assert 'name_hint="dspark_main_proj"' in source
    assert 'name_hint="dspark_main_proj_cast"' in source
    assert "projected_fp32 = pl.create_tensor([t_dim, D], dtype=pl.FP32)" in source
    assert "CAST_T_TILE = 8" in source
    assert "token : token + CAST_T_TILE" in source


def test_backbone_gates_embedding_and_attention_metadata() -> None:
    lookup_source = (_MODEL_DIR / "lookup_embedding.py").read_text()
    backbone_source = (_MODEL_DIR / "draft_backbone.py").read_text()
    metadata_source = (_MODEL_DIR / "dspark_metadata.py").read_text()
    assert 'with pl.spmd(SPMD_BLOCKS, name_hint="lookup_embedding") as lookup_tid:' in lookup_source
    assert 'with pl.spmd(1, name_hint="dspark_query_metadata") as query_metadata_tid:' in metadata_source
    assert "for token in pl.range(metadata_core, DSPARK_QUERY_TOKENS):" in metadata_source
    assert 'pl.spmd(DSPARK_QUERY_TOKENS, name_hint="dspark_query_metadata")' not in metadata_source
    assert "hidden_0_flat = pl.reshape(initial_hidden, [T, HC_MULT * D])" in backbone_source
    assert "context_kv_0_tid = dspark_context_kv_query(" in backbone_source
    assert "context_kv_1_tid = dspark_context_kv_query(" in backbone_source
    assert "context_kv_2_tid = dspark_context_kv_query(" in backbone_source
    assert "context_slots_0, kv_caches_0, lookup_tid" in backbone_source
    assert "context_slots_1, kv_caches_1, context_kv_0_tid" in backbone_source
    assert "context_slots_2, kv_caches_2, context_kv_1_tid" in backbone_source
    metadata_view_start = backbone_source.index('name_hint="dspark_layer_metadata_views"')
    metadata_view_end = backbone_source.index("hidden_1 =", metadata_view_start)
    metadata_view_source = backbone_source[metadata_view_start:metadata_view_end]
    assert ") as metadata_views_tid:" in metadata_view_source
    assert "deps=[query_metadata_tid, visible_metadata_tid]" in metadata_view_source
    assert "first_layer_tid = lookup_tid" in metadata_view_source
    assert 'name_hint="dspark_first_layer_gate"' not in backbone_source
    assert "hidden_0 = initial_hidden" in backbone_source
    layer_source = (_MODEL_DIR / "dspark_draft_layer.py").read_text()
    assert "hc_pre_flat_with_inv_rms(" in layer_source
    assert "hc_attn_inv_rms," in layer_source
    assert "hc_attn_inv_rms_0," in backbone_source
    assert "hc_attn_inv_rms_1," in backbone_source
    assert "hc_attn_inv_rms_2," in backbone_source
    assert "start_tid," in layer_source
    assert "context_cache_ready_tid: pl.Scalar[pl.TASK_ID]" in layer_source
    assert "metadata_ready_tid: pl.Scalar[pl.TASK_ID]" in layer_source
    assert "kv_cache, context_cache_ready_tid, metadata_ready_tid, query_slot_mapping" in layer_source
    for layer in range(3):
        assert (
            f"kv_caches_{layer},\n"
            f"        context_kv_{layer}_tid,\n"
            "        metadata_views_tid,\n"
            f"        query_slot_mapping_{layer},"
        ) in backbone_source
    attention_source = (_MODEL_DIR / "dspark_attention.py").read_text()
    attention_tree = ast.parse(attention_source)
    attention = next(
        node for node in attention_tree.body if isinstance(node, ast.FunctionDef) and node.name == "dspark_attention"
    )
    commit_scope = next(
        node
        for node in ast.walk(attention)
        if isinstance(node, ast.With)
        and isinstance(node.items[0].context_expr, ast.Call)
        and any(
            keyword.arg == "name_hint"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "dspark_kv_commit_valid_bias"
            for keyword in node.items[0].context_expr.keywords
        )
    )
    commit_deps = next(
        keyword.value for keyword in commit_scope.items[0].context_expr.keywords if keyword.arg == "deps"
    )
    assert isinstance(commit_deps, ast.List)
    assert [dep.id for dep in commit_deps.elts if isinstance(dep, ast.Name)] == [
        "cache_ready_tid",
        "metadata_ready_tid",
    ]
    hc_pre_source = (_MODEL_DIR / "hc_pre.py").read_text()
    assert "init_value=0.0" in hc_pre_source
    assert 'name_hint="hc_pre_seed"' not in hc_pre_source
    assert "deps=[start_dep]" in hc_pre_source
    assert "deps=[rms_tid]" in hc_pre_source
    assert "deps=[linear_tid]" in hc_pre_source
    assert "deps=[gate_tid]" in hc_pre_source
    assert "deps=[sinkhorn_tid]" in hc_pre_source
    qkv_source = (_MODEL_DIR / "qkv_proj_rope.py").read_text()
    assert 'name_hint="kv_proj_seed", deps=[late_dep]' in qkv_source
    assert 'name_hint="kv_proj_matmul", deps=[kv_seed_tid]' in qkv_source


def test_ci_selects_both_public_dspark_programs() -> None:
    import subprocess

    changed = "\n".join(
        [
            "models/deepseek_v4_flash_dspark/dspark_backbone.py",
            "models/deepseek_v4_flash_dspark/markov_sample.py",
        ]
    )
    selected = subprocess.run(
        [sys.executable, str(_REPO_ROOT / ".github" / "scripts" / "detect_changes.py")],
        input=changed,
        text=True,
        check=True,
        capture_output=True,
        cwd=_REPO_ROOT,
    ).stdout.split()
    assert "models/deepseek_v4_flash_dspark/dspark_backbone.py" in selected
    assert "models/deepseek_v4_flash_dspark/markov_sample.py" in selected
