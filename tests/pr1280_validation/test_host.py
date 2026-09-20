# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Private-branch host coverage for PR1280."""
import ast
import importlib
import subprocess
import pytest
import torch
from models.deepseek_v4_1_flash.config import FLASH, HEAD_DIM
from models.deepseek_v4_1_flash.metadata import paged_slots, window_metadata, build_forward_metadata
from models.deepseek_v4_1_flash.compressor_state import CompressorStateCache, CompressorStateSnapshot
from models.deepseek_v4_1_flash.golden import compressor_ratio2_paged


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
def test_builder(empty):
    q = torch.tensor([0, 0 if empty else 1, 0 if empty else 1], dtype=torch.int32)
    kv = torch.tensor([0 if empty else 256, 0], dtype=torch.int32)
    window = torch.tensor([[2, 0, 1], [3, 4, 5]], dtype=torch.int32)
    tables = {}
    for s in FLASH.kv_source_layer_ids:
        pages = ((int(kv[0])+int(q[-1]))//FLASH.compress_ratios[s]+127)//128
        tables[s] = torch.arange(2*pages, dtype=torch.int32).reshape(2, pages)
    state = {s: torch.tensor([[2], [0]], dtype=torch.int32) for s in FLASH.kv_source_layer_ids if FLASH.compress_ratios[s] == 2}
    m = build_forward_metadata(q, kv, window, tables, state_block_tables=state)
    assert m.position_ids.numel() == int(q[-1])
    if not empty:
        for s in state:
            assert m.compressed_slots[s].tolist() == [-1]
        tables[next(iter(tables))][0, 0] = -1
        with pytest.raises(ValueError, match='visible compressed'):
            build_forward_metadata(q, kv, window, tables, state_block_tables=state)


def test_lifecycle():
    pool = CompressorStateCache(num_blocks=3, capacity=3)
    assert pool.allocate('a') == 0
    assert pool.allocate('b') == 1
    for buffer in pool.buffers.values():
        buffer[1].copy_(torch.arange(3*2*HEAD_DIM).reshape(3, -1))
    snapshot = pool.snapshot('b', prefix_length=5)
    assert all(t.tolist() == [[1], [0]] for t in pool.block_tables(['b', 'a']).values())
    pool.release('b')
    assert pool.allocate('c') == 1
    assert all(torch.count_nonzero(b[1]) == 0 for b in pool.buffers.values())
    assert pool.allocate('b2', prefix_length=5, snapshot=snapshot) == 2
    for s, buffer in pool.buffers.items():
        assert torch.equal(buffer[2], snapshot.rings[s])
        assert buffer[2].data_ptr() != snapshot.rings[s].data_ptr()
    with pytest.raises(RuntimeError):
        pool.allocate('d')
    pool.release('b2')
    with pytest.raises(ValueError):
        pool.allocate('bad', prefix_length=3)
    with pytest.raises(ValueError):
        pool.allocate('bad', prefix_length=6, snapshot=snapshot)
    bad = CompressorStateSnapshot(5, {s: v[:1] for s, v in snapshot.rings.items()})
    with pytest.raises(ValueError):
        pool.allocate('bad', prefix_length=5, snapshot=bad)
    with pytest.raises(ValueError):
        pool.block_tables(['a', 'a'])
    with pytest.raises(KeyError):
        pool.block_tables(['unknown'])


@pytest.mark.parametrize('capacity', [1, 2, 3, 8])
def test_chunked(capacity):
    torch.manual_seed(42)
    x = torch.randn(19, 16, dtype=torch.bfloat16)
    wk, wg, norm = torch.randn(16, 8), torch.randn(16, 8), torch.ones(8)
    state = torch.zeros(1, capacity, 16)
    expected, _ = compressor_ratio2_paged(x, torch.arange(19), torch.tensor([0, 19]), torch.zeros(19, dtype=torch.int32), torch.tensor([[0]]), state, wk, wg, norm)
    split = torch.zeros(3, capacity, 16)
    parts = []
    for begin, end in [(0, 3), (3, 14), (14, 18), (18, 19)]:
        out, _ = compressor_ratio2_paged(x[begin:end], torch.arange(begin, end), torch.tensor([0, end-begin]), torch.zeros(end-begin, dtype=torch.int32), torch.tensor([[2]]), split, wk, wg, norm)
        parts.append(out)
    torch.testing.assert_close(torch.cat(parts), expected, rtol=0, atol=0)
    assert torch.equal(split[2], state[0])
    assert torch.count_nonzero(split[:2]) == 0


@pytest.mark.parametrize('phase', ['prefill', 'decode'])
@pytest.mark.parametrize('variant,ratio,mode', [('c2a_full', 2, 'full'), ('c2a_reuse', 2, 'reuse'), ('c1a_full', 1, 'full'), ('c1a_reindex', 1, 'reindex'), ('c1a_reuse', 1, 'reuse')])
def test_goldens(phase, variant, ratio, mode):
    from models.deepseek_v4_1_flash._golden_smoke import run_attention_golden
    m = importlib.import_module(f'models.deepseek_v4_1_flash.{phase}_{variant}')
    run_attention_golden(getattr(m, f'golden_{phase}_{variant}'), ratio, mode)

