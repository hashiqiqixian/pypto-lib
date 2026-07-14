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
