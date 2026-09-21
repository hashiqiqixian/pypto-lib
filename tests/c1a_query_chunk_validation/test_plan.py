# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Execute the production orchestration AST with metadata-only tensor views."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]


class View:
    def __init__(self, shape, name="", start=0):
        self.shape, self.name, self.start = shape, name, start


def exercise(capacity, active, history, budget, candidates):
    source = ROOT / "models/deepseek_v4_1_flash/prefill_c1a_indexer.py"
    factory = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef)
                   and n.name == "make_paged_indexer")
    entry = next(n for n in factory.body if isinstance(n, ast.FunctionDef) and n.name == "paged_indexer")
    entry.decorator_list = []
    for arg in entry.args.args:
        arg.annotation = None
    entry.returns = None
    trace, allocations = [], []
    current = [11]

    def allocate(shape, dtype):
        allocations.append(shape)
        return View(shape, "scores")

    def view(tensor, shape, offset):
        assert all(start >= 0 and start + size <= extent
                   for start, size, extent in zip(offset, shape, tensor.shape))
        return View(shape, tensor.name, tensor.start + offset[0])

    def kernel(*args):
        x, latent, requests, lens = args[:4]
        cos, sin = args[7:9]
        mask, scores, topk, count, dependency = args[12:]
        assert all(t.start == x.start for t in (latent, requests, lens, cos, sin, mask, topk))
        assert all(t.shape[0] == count for t in (x, latent, requests, lens, cos, sin, mask, scores, topk))
        assert scores.start == 0 and dependency == current[0]
        trace.append((x.start, count, "topk"))
        current[0] += 1
        return current[0]

    def select(scores, lens, mask):
        assert scores.shape[0] == lens.shape[0] == mask.shape[0]
        trace.append((mask.start, mask.shape[0], "candidates"))
        current[0] += 1
        return current[0]

    def join(deps):
        assert deps == [current[0] - 1, current[0]]
        current[0] += 1
        return current[0]

    fake = SimpleNamespace(
        tensor=SimpleNamespace(dim=lambda t, axis: t.shape[axis]),
        array=SimpleNamespace(create=lambda n, dtype: [None] * n),
        system=SimpleNamespace(task_dummy=join),
        cast=lambda v, dtype: v, min=min, max=max, range=range, scope=nullcontext,
        create_tensor=allocate, slice=view, FP32=None, INDEX=None, INT32=None, TASK_ID=None,
    )
    namespace = dict(pl=fake, TOPK_LEAF=8192, D=5120, Q_LORA=1024, ROPE_DIM=64, INDEX_TOPK=512,
                     use_candidates=candidates, max_logits_bytes=budget, paged_indexer_chunk=kernel,
                     _hierarchical_sparse_indexer=select)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[entry], type_ignores=[])), str(source), "exec"), namespace)
    params = []
    row_widths = dict(x=5120, query_latent=1024, request_ids=None, compressed_lens=None,
                      rope_cos=32, rope_sin=32, candidate_mask=history, topk_indices=512)
    for arg in entry.args.args:
        if arg.arg == "num_tokens":
            params.append(active)
        elif arg.arg == "cache_ready":
            params.append(11)
        elif arg.arg in row_widths:
            width = row_widths[arg.arg]
            params.append(View([capacity] if width is None else [capacity, width], arg.arg))
        else:
            params.append(View([1], arg.arg))
    namespace["paged_indexer"](*params)
    return trace, allocations


class ChunkPlanTests(unittest.TestCase):
    def test_long_context_budget(self):
        trace, allocations = exercise(4096, 4096, 65536, 512 << 20, False)
        self.assertEqual(allocations, [[2048, 65536]])
        self.assertEqual(trace, [(0, 2048, "topk"), (0, 2048, "candidates"),
                                 (2048, 2048, "topk"), (2048, 2048, "candidates")])

    def test_tail_and_inactive_rows(self):
        for candidates in (False, True):
            trace, allocations = exercise(11, 7, 16640, 196608, candidates)
            self.assertEqual(allocations, [[2, 24576]])
            self.assertEqual([(s, n) for s, n, k in trace if k == "topk"], [(0, 2), (2, 2), (4, 2), (6, 1)])
            self.assertEqual(sum(k == "candidates" for _, _, k in trace), 0 if candidates else 4)

    def test_empty(self):
        self.assertEqual(exercise(7, 0, 16640, 196608, False), ([], []))

    def test_one_row_floor(self):
        trace, allocations = exercise(3, 3, 65536, 32768, True)
        self.assertEqual(allocations, [[1, 65536]])
        self.assertEqual(trace, [(0, 1, "topk"), (1, 1, "topk"), (2, 1, "topk")])


if __name__ == "__main__":
    unittest.main()
