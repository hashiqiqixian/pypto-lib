# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reuse actual Engram windows across changed inputs and alternating rank skew."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig
from golden import run, TensorSpec, ratio_allclose
from models.deepseek_v4_1_flash import engram as E
from models.deepseek_v4_1_flash.engram import engram_tp

TP, T, EPOCHS = E.TP_SIZE, 3, 3
WINDOW_TOKENS = E.TP_MAX_TOKENS
N, ROWS, HD, K, V, H, D = E.N_HASH_COLS, E.ROWS_PER_RANK, E.HEAD_DIM, E.ENGRAM_K, E.KV_OUT, E.HC_MULT, E.D


@pl.jit
def rank_entry(
    ids: pl.Tensor[[EPOCHS, T, N], pl.INT32],
    table: pl.Tensor[[ROWS, HD], pl.BF16],
    wkv: pl.Tensor[[K, V], pl.BF16],
    weight: pl.Tensor[[H, D], pl.FP32],
    x: pl.Tensor[[EPOCHS, T, H, D], pl.BF16],
    delay_map: pl.Tensor[[2], pl.INT32],
    out: pl.Out[pl.Tensor[[EPOCHS, T, H, D], pl.BF16]],
    window: pld.DistributedTensor[[WINDOW_TOKENS, K], pl.BF16],
    signal: pld.DistributedTensor[[TP, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    for epoch in pl.range(EPOCHS):
        prepared = pl.create_tensor([T, N], dtype=pl.INT32)
        # The map fixes zero, but the compiler cannot eliminate these dependent
        # loads. Alternate the slower producer without changing its hash IDs.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="validation_rank_skew"):
            offset = pl.cast(0, pl.INT32)
            for _ in pl.range(((rank + epoch) % 2) * 20000):
                offset = pl.read(delay_map, [offset])
            for token in pl.range(T):
                for col in pl.range(N):
                    pl.write(prepared, [token, col], pl.read(ids, [epoch, token, col]) + offset)
        engram_tp(prepared, table, wkv, weight, x[epoch], out[epoch],
                    window, signal, rank, epoch + 1)
    return out


@pl.jit.host
def host(
    ids: pl.Tensor[[TP, EPOCHS, T, N], pl.INT32],
    table: pl.Tensor[[TP, ROWS, HD], pl.BF16],
    wkv: pl.Tensor[[TP, K, V], pl.BF16],
    weight: pl.Tensor[[TP, H, D], pl.FP32],
    x: pl.Tensor[[TP, EPOCHS, T, H, D], pl.BF16],
    delay_map: pl.Tensor[[TP, 2], pl.INT32],
    out: pl.Out[pl.Tensor[[TP, EPOCHS, T, H, D], pl.BF16]],
):
    window_buf = pld.alloc_window_buffer([WINDOW_TOKENS, K], dtype=pl.BF16)
    signal_buf = pld.alloc_window_buffer([TP, 1], dtype=pl.INT32)
    for rank in pl.range(TP):
        window = pld.window(window_buf, [WINDOW_TOKENS, K], dtype=pl.BF16)
        signal = pld.window(signal_buf, [TP, 1], dtype=pl.INT32)
        rank_entry(ids[rank], table[rank], wkv[rank], weight[rank], x[rank], delay_map[rank],
                   out[rank], window, signal, rank, device=rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("-d", default="0,1")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(73)
    table = (torch.randn(E.NUM_EMBEDDINGS, HD) * .02).bfloat16().reshape(TP, ROWS, HD)
    wkv = (torch.randn(K, V) / K ** .5).bfloat16().unsqueeze(0).repeat(TP, 1, 1)
    ids = torch.randint(0, E.NUM_EMBEDDINGS, (EPOCHS, T, N), dtype=torch.int32)
    # Every epoch reads both shards; change the selected rows each time.
    for rank in range(TP):
        ids[:, :, rank::TP] = ids[:, :, rank::TP] % ROWS + rank * ROWS
    values = {
        "ids": ids.unsqueeze(0).repeat(TP, 1, 1, 1), "table": table, "wkv": wkv,
        "weight": (.5 + torch.rand(H, D)).unsqueeze(0).repeat(TP, 1, 1),
        "x": torch.randn(EPOCHS, T, H, D).bfloat16().unsqueeze(0).repeat(TP, 1, 1, 1, 1),
        "delay_map": torch.tensor([0, 1], dtype=torch.int32).repeat(TP, 1),
        "out": torch.empty(TP, EPOCHS, T, H, D, dtype=torch.bfloat16),
    }

    def reference(v):
        for epoch in range(EPOCHS):
            expected = E.golden_engram(v["ids"][0, epoch], v["table"].reshape(-1, HD),
                                       v["wkv"][0], v["weight"][0], v["x"][0, epoch])
            v["out"][:, epoch] = expected.unsqueeze(0)

    result = run(fn=host,
                 specs=[TensorSpec(n, list(v.shape), v.dtype, init_value=lambda n=n: values[n].clone())
                        for n, v in values.items()],
                 golden_fn=reference, compile_only=args.compile_only,
                 compare_fn={"out": E._precision_compare("out", ratio_allclose(atol=1e-3, rtol=1e-2))},
                 config={"platform": "a5", "distributed_config": DistributedConfig(
                     device_ids=[int(d) for d in args.d.split(",")], num_sub_workers=0)})
    assert result.passed, result.error


if __name__ == "__main__":
    main()
