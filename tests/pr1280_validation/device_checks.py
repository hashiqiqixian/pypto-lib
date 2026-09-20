# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Private-branch NPU coverage of the unchanged full-table RoPE stage."""
import argparse
import os
import torch
import pypto.language as pl
from golden import TensorSpec, ScalarSpec, run
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.rope_tables import materialize_rope_rows, precompute_rope_tables, ROPE_ROWS_DYN

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


def golden_rope(v):
    for t in range(v['count']):
        p = int(v['positions'][t])
        v['out_cos'][t] = v['cos'][p] if p >= 0 else 1.
        v['out_sin'][t] = v['sin'][p] if p >= 0 else 0.


def specs(values):
    return [TensorSpec(k, list(v.shape), v.dtype, init_value=v) if isinstance(v, torch.Tensor) else ScalarSpec(k, torch.int32, v, compile_runtime=True) for k,v in values.items()]


def exact_compare(name):
    def compare(actual, expected, **kwargs):
        same = torch.equal(actual, expected)
        if not same:
            mismatch = actual != expected
            rows = mismatch.reshape(actual.shape[0], -1).any(dim=1).nonzero().flatten().tolist()
            print(f'DIFF {name}: rows={rows} max_abs={(actual.float()-expected.float()).abs().max().item()}', flush=True)
        return same, 'exact copy/state comparison'
    return compare


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--compile-only',action='store_true')
    parser.add_argument('--device',type=int,default=int(os.environ.get('TASK_DEVICE','0')))
    args=parser.parse_args()
    failures=[]
    for compressed in (False,True):
        cos,sin=precompute_rope_tables(257,compressed)
        for count in (0,1,7,33,65):
            name=f'rope-{int(compressed)}-{count}'
            positions=torch.arange(66,dtype=torch.int32)
            positions[:7]=torch.tensor([256,-1,0,3,3,129,-9])
            positions[-1]=999999
            values=dict(cos=cos,sin=sin,positions=positions,count=count,out_cos=torch.full((66,C.ROPE_DIM//2),17.),out_sin=torch.full((66,C.ROPE_DIM//2),19.))
            result=run(fn=rope_entry,specs=specs(values),golden_fn=golden_rope,compile_only=args.compile_only,config=dict(platform='a2a3',device_id=args.device),compare_fn={k:exact_compare(k) for k in ('out_cos','out_sin')},rtol=0,atol=0)
            print(f'RESULT {name} passed={result.passed} work_dir={result.work_dir} error={result.error}',flush=True)
            if not result.passed:
                failures.append(name)
    assert not failures,failures

if __name__=='__main__':
    main()
