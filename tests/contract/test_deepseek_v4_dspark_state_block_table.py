# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host checks for deployment-sized DSpark compressor-state rings."""

import sys
from pathlib import Path

import pytest


pytest.importorskip("torch")

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_flash_dspark"
sys.path.insert(0, str(MODEL_DIR))

from utils import block_table  # noqa: E402


def test_block_table_can_model_hca_deployment_request_slots() -> None:
    physical_blocks = 1088
    blocks_per_request = 17
    table = block_table(
        batch=2,
        table_blocks=blocks_per_request + 1,
        physical_blocks=physical_blocks,
        request_slots=64,
    )

    assert table.shape == (2, blocks_per_request + 1)
    assert table[0, :blocks_per_request].tolist() == [64 * block for block in range(blocks_per_request)]
    assert table[1, :blocks_per_request].tolist() == [64 * block + 1 for block in range(blocks_per_request)]
    assert table[0, blocks_per_request].item() == table[0, 0].item()
    assert set(table[0].tolist()).isdisjoint(table[1].tolist())


def test_block_table_request_slots_default_is_backward_compatible() -> None:
    default = block_table(batch=2, table_blocks=7, physical_blocks=32)
    explicit = block_table(batch=2, table_blocks=7, physical_blocks=32, request_slots=2)
    assert default.equal(explicit)


def test_block_table_rejects_fewer_request_slots_than_rows() -> None:
    with pytest.raises(ValueError, match="request_slots must be >= batch"):
        block_table(batch=2, table_blocks=7, physical_blocks=32, request_slots=1)
