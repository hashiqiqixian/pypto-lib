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
