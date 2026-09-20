# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validation-only coverage for the three retained PR1280 commits."""
import ast
import subprocess
import pytest
import torch
from models.deepseek_v4_1_flash.config import FLASH
from models.deepseek_v4_1_flash.metadata import paged_slots, window_metadata, build_forward_metadata

def test_baseline_regressions():
    source = subprocess.check_output(['git', 'show', '8c0c165:models/deepseek_v4_1_flash/metadata.py'], text=True)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'paged_slots')
    scope = {'torch': torch, 'BLOCK_SIZE': 128}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'baseline', 'exec'), scope)
    old = scope['paged_slots']
    p, r, table = torch.tensor([256]), torch.tensor([0]), torch.tensor([[0]])
    with pytest.raises(IndexError):
        old(p, r, table, 128, 2, True)
    assert paged_slots(p, r, table, 128, 2, True).tolist() == [-1]
    p, table = torch.tensor([128]), torch.tensor([[0, -1]])
    assert old(p, r, table).tolist() == [-128]
    with pytest.raises(ValueError, match='allocation'):
        paged_slots(p, r, table)


@pytest.mark.parametrize('ratio', [1, 2])
@pytest.mark.parametrize('positions', [[], [0], [1], [127, 128], [255, 256, 257], [511, 512]])
def test_pages(ratio, positions):
    p = torch.tensor(positions, dtype=torch.int32)
    r = torch.arange(len(p), dtype=torch.int32) % 2
    table = torch.tensor([[9, 3, 8, 1, 7], [6, 4, 0, 2, 5]], dtype=torch.int32)
    got = paged_slots(p, r, table, 128, ratio, True)
    expected = [-1 if (v+1) % ratio else int(table[int(req), (v//ratio)//128])*128 + (v//ratio)%128 for v, req in zip(positions, r)]
    assert got.tolist() == expected


@pytest.mark.parametrize('position', [0, 1, 127, 128, 255, 256])
def test_window(position):
    table = torch.arange(position//128+1, dtype=torch.int32).flip(0).reshape(1, -1)
    slot, indices, lens = window_metadata(torch.tensor([position]), torch.tensor([0]), table)
    visible = range(max(0, position-FLASH.sliding_window+1), position+1)
    expected = [int(table[0, v//128])*128+v%128 for v in visible]
    assert int(lens[0]) == len(expected)
    assert indices[0, :len(expected)].tolist() == expected
    assert bool((indices[0, len(expected):] == -1).all())
    assert int(slot[0]) == expected[-1]


@pytest.mark.parametrize('empty', [False, True])
def test_visible_history(empty):
    q = torch.tensor([0, 0 if empty else 1, 0 if empty else 1], dtype=torch.int32)
    kv = torch.tensor([0 if empty else 256, 0], dtype=torch.int32)
    window = torch.tensor([[2, 0, 1], [3, 4, 5]], dtype=torch.int32)
    tables = {}
    for source in FLASH.kv_source_layer_ids:
        pages = ((int(kv[0])+int(q[-1]))//FLASH.compress_ratios[source]+127)//128
        tables[source] = torch.arange(2*pages, dtype=torch.int32).reshape(2, pages)
    result = build_forward_metadata(q, kv, window, tables)
    assert result.position_ids.numel() == int(q[-1])
    if not empty:
        for source in FLASH.kv_source_layer_ids:
            if FLASH.compress_ratios[source] == 2:
                assert result.compressed_slots[source].tolist() == [-1]
        for table in tables.values():
            table[0, 0] = -1
        with pytest.raises(ValueError, match='visible compressed'):
            build_forward_metadata(q, kv, window, tables)

@pytest.mark.parametrize('table', [torch.tensor([[0]]), torch.tensor([[0, -1]])])
def test_missing_required_pages(table):
    with pytest.raises(ValueError):
        paged_slots(torch.tensor([128]), torch.tensor([0]), table)


def test_paused_history_does_not_require_pages():
    q = torch.tensor([0, 1, 1], dtype=torch.int32)
    kv = torch.tensor([0, 4096], dtype=torch.int32)
    window = torch.tensor([[0], [-1]], dtype=torch.int32)
    tables = {s: torch.tensor([[0], [-1]], dtype=torch.int32) for s in FLASH.kv_source_layer_ids}
    result = build_forward_metadata(q, kv, window, tables)
    assert result.position_ids.tolist() == [0]
