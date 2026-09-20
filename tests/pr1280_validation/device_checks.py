# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validation-only hardware wrappers around unchanged PR1280 functions."""
import argparse
import itertools
import os
import torch
import pypto.language as pl
from golden import TensorSpec, ScalarSpec, run
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.decode_c2a_full import compressor_pair, compressor_state_write
from models.deepseek_v4_1_flash.rope_tables import materialize_rope_rows, precompute_rope_tables, ROPE_ROWS_DYN


@pl.jit
def ring_sequence(
    kv: pl.Tensor[[3, C.T_DYN, C.HEAD_DIM], pl.FP32],
    scores: pl.Tensor[[3, C.T_DYN, C.HEAD_DIM], pl.FP32],
    positions: pl.Tensor[[3, C.T_DYN], pl.INT32],
    requests: pl.Tensor[[3, C.T_DYN], pl.INT32],
    starts: pl.Tensor[[3, C.Q_START_DYN], pl.INT32],
    tables: pl.Tensor[[3, C.B_DYN, 1], pl.INT32],
    counts: pl.Tensor[[3], pl.INT32],
    state: pl.InOut[pl.Tensor[[C.STATE_BLOCKS_DYN, C.STATE_CAPACITY_DYN, 2*C.HEAD_DIM], pl.FP32]],
    previous_kv: pl.InOut[pl.Tensor[[3, C.T_DYN, C.HEAD_DIM], pl.FP32]],
    previous_scores: pl.InOut[pl.Tensor[[3, C.T_DYN, C.HEAD_DIM], pl.FP32]],
):
    count0 = pl.read(counts, [0])
    pk0 = previous_kv[0]
    ps0 = previous_scores[0]
    pair0 = compressor_pair(kv[0], scores[0], positions[0], requests[0], starts[0], tables[0], state, pk0, ps0, count0, None)
    ready0 = compressor_state_write(kv[0], scores[0], positions[0], requests[0], starts[0], tables[0], state, count0, pair0)
    count1 = pl.read(counts, [1])
    pk1 = previous_kv[1]
    ps1 = previous_scores[1]
    pair1 = compressor_pair(kv[1], scores[1], positions[1], requests[1], starts[1], tables[1], state, pk1, ps1, count1, ready0)
    ready1 = compressor_state_write(kv[1], scores[1], positions[1], requests[1], starts[1], tables[1], state, count1, pair1)
    count2 = pl.read(counts, [2])
    pk2 = previous_kv[2]
    ps2 = previous_scores[2]
    pair2 = compressor_pair(kv[2], scores[2], positions[2], requests[2], starts[2], tables[2], state, pk2, ps2, count2, ready1)
    ready2 = compressor_state_write(kv[2], scores[2], positions[2], requests[2], starts[2], tables[2], state, count2, pair2)
    return state, previous_kv, previous_scores


@pl.jit
def rope_entry(
    cos: pl.Tensor[[ROPE_ROWS_DYN, C.ROPE_DIM//2], pl.FP32],
    sin: pl.Tensor[[ROPE_ROWS_DYN, C.ROPE_DIM//2], pl.FP32],
    positions: pl.Tensor[[C.T_DYN], pl.INT32],
    count: pl.Scalar[pl.INT32],
    out_cos: pl.InOut[pl.Tensor[[C.T_DYN, C.ROPE_DIM//2], pl.FP32]],
    out_sin: pl.InOut[pl.Tensor[[C.T_DYN, C.ROPE_DIM//2], pl.FP32]],
):
    materialize_rope_rows(cos, sin, positions, count, out_cos, out_sin)
    return out_cos, out_sin


def ring_values(capacity, case):
    torch.manual_seed(13)
    extent = 32
    v = dict(kv=torch.randn(3, extent, C.HEAD_DIM), scores=torch.randn(3, extent, C.HEAD_DIM), positions=torch.full((3, extent), 999999, dtype=torch.int32), requests=torch.full((3, extent), 999, dtype=torch.int32), starts=torch.zeros(3, 4, dtype=torch.int32), tables=torch.zeros(3, 3, 1, dtype=torch.int32), counts=torch.zeros(3, dtype=torch.int32), state=torch.randn(5, capacity, 2*C.HEAD_DIM), previous_kv=torch.full((3, extent, C.HEAD_DIM), 17.), previous_scores=torch.full((3, extent, C.HEAD_DIM), 19.))
    offsets = [0, 5, 126]
    for step, (order, lengths) in enumerate([([0,1,2], [17,0,2]), ([2,0,1], [1,3,1]), ([1,2,0], [1,1,1])]):
        if case == 'empty':
            lengths = [0,0,0]
        v['starts'][step] = torch.tensor([0]+list(itertools.accumulate(lengths)))
        v['counts'][step] = sum(lengths)
        cursor = 0
        for req, (owner, length) in enumerate(zip(order, lengths)):
            v['tables'][step,req,0] = [3,0,4][owner]
            v['positions'][step,cursor:cursor+length] = torch.arange(offsets[owner], offsets[owner]+length)
            v['requests'][step,cursor:cursor+length] = req
            offsets[owner] += length
            cursor += length
        if case == 'invalid':
            v['tables'][step,0,0] = -1
            v['tables'][step,2,0] = 5
    return v


def golden_ring(v):
    head = C.HEAD_DIM
    capacity = v['state'].shape[1]
    for step in range(3):
        for token in range(int(v['counts'][step])):
            v['previous_kv'][step,token].zero_()
            v['previous_scores'][step,token].zero_()
            request = int(v['requests'][step,token])
            position = int(v['positions'][step,token])
            block = int(v['tables'][step,request,0])
            if not 0 <= block < v['state'].shape[0]:
                continue
            if position % 2:
                previous = v['state'][block,(position-1)%capacity]
                v['previous_kv'][step,token] = previous[:head]
                v['previous_scores'][step,token] = previous[head:]
            v['state'][block,position%capacity,:head] = v['kv'][step,token]
            v['state'][block,position%capacity,head:] = v['scores'][step,token]


def golden_rope(v):
    for t in range(v['count']):
        p = int(v['positions'][t])
        v['out_cos'][t] = v['cos'][p] if p >= 0 else 1.
        v['out_sin'][t] = v['sin'][p] if p >= 0 else 0.


def specs(values):
    return [TensorSpec(k, list(v.shape), v.dtype, init_value=v) if isinstance(v, torch.Tensor) else ScalarSpec(k, torch.int32, v) for k,v in values.items()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--device', type=int, default=int(os.environ.get('TASK_DEVICE','0')))
    parser.add_argument('--case', default='all')
    args = parser.parse_args()
    cases = []
    for cap in (1,3,8):
        cases.append((f'ring-{cap}', ring_sequence, ring_values(cap,'mixed'), golden_ring))
    for mode in ('empty','invalid'):
        cases.append((f'ring-{mode}', ring_sequence, ring_values(3,mode), golden_ring))
    for compressed in (False,True):
        cos,sin = precompute_rope_tables(257, compressed)
        for count in (0,1,7):
            values = dict(cos=cos,sin=sin,positions=torch.tensor([256,-1,0,3,3,129,-9,999999],dtype=torch.int32),count=count,out_cos=torch.full((8,C.ROPE_DIM//2),17.),out_sin=torch.full((8,C.ROPE_DIM//2),19.))
            cases.append((f'rope-{int(compressed)}-{count}',rope_entry,values,golden_rope))
    failures=[]
    for name,fn,values,reference in cases:
        if args.case not in ('all',name):
            continue
        print(f'CASE {name}',flush=True)
        try:
            result=run(fn=fn,specs=specs(values),golden_fn=reference,compile_only=args.compile_only,config=dict(platform='a2a3',device_id=args.device),rtol=0,atol=0)
            print(f'RESULT {name} passed={result.passed} work_dir={result.work_dir} error={result.error}',flush=True)
            if not result.passed:
                failures.append(name)
        except Exception as exc:
            print(f'RESULT {name} exception={type(exc).__name__}: {exc}',flush=True)
            failures.append(name)
    assert not failures, failures


if __name__ == '__main__':
    main()
