# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validate full-table RoPE through the exported C2A attention entries."""
import argparse
from types import SimpleNamespace

import torch
from golden import TensorSpec, ScalarSpec, run
from pypto.ir import DistributedConfig
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash import decode_c2a_full as M
from models.deepseek_v4_1_flash.prefill_c2a_full import prefill_c2a_full


def stack(value):
    if value.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
        return value.view(torch.uint8).unsqueeze(0).clone().view(value.dtype)
    return value.unsqueeze(0).clone()


def values_for(mode):
    values = M.make_c2a_inputs(tokens=8, requests=3, seed=29, mode=mode, full_rope_tables=True)
    # Caller-owned, nonstandard profiles prove that dispatch consumes supplied
    # tables instead of regenerating frequencies or treating them as token rows.
    for name in M.FULL_ROPE_NAMES.values():
        table = values[name]
        extra = torch.full((37, table.shape[1]), 0.25, dtype=torch.float32)
        values[name] = torch.cat((table.roll(5, 0), extra), 0)
    return values


def host_checks(mode):
    full = M.make_c2a_inputs(tokens=8, requests=3, seed=29, mode=mode, full_rope_tables=True)
    rows = M.make_c2a_inputs(tokens=8, requests=3, seed=29, mode=mode)
    assert tuple(full) == M.PROGRAM_INPUT_NAMES
    for old, new in M.FULL_ROPE_NAMES.items():
        positions = full['compressed_rope_positions' if old.startswith('compressed_') else 'position_ids'].long()
        expected = full[new][positions.clamp_min(0)].masked_fill(
            positions[:, None] < 0, 1.0 if old.endswith('cos') else 0.0)
        assert torch.equal(rows[old], expected), old
    args = SimpleNamespace(dp=1, tokens=8, requests=3, seed=29, case='mixed', epochs=1, bench=False)
    specs = M.build_specs(args, mode)
    assert [spec.name for spec in specs[:len(M.PROGRAM_INPUT_NAMES)]] == list(M.PROGRAM_INPUT_NAMES)
    print(f'HOST {mode}: fixture and program input contract PASS', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp', type=int, default=1, choices=[1])
    parser.add_argument('--ep', type=int, default=2, choices=[2])
    parser.add_argument('--mode', choices=['decode', 'prefill'], default='decode')
    parser.add_argument('--platform', choices=['a5', 'a5sim'], default='a5sim')
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--host-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert C.TP_SIZE == 1
    if args.host_only:
        host_checks(args.mode)
        return
    values = values_for(args.mode)
    specs = [TensorSpec(name, [1, *value.shape], value.dtype, init_value=stack(value), resident='stacked')
             for name, value in values.items()]
    specs += [TensorSpec('topk_indices', [1, 8, C.INDEX_TOPK], torch.int32, resident='stacked'),
              TensorSpec('output', [1, 8, C.D], torch.bfloat16, resident='stacked'),
              ScalarSpec('num_tokens', torch.int32, 8),
              ScalarSpec('attention_epoch', torch.int32, 1, compile_runtime=True)]
    operator = M.decode_c2a_full if args.mode == 'decode' else prefill_c2a_full
    capacity = C.DECODE_MAX_TOKENS if args.mode == 'decode' else C.PREFILL_MAX_TOKENS
    compare = {'output': M.compare_replicated(M.compare_output),
               'topk_indices': M.compare_per_rank(M.compare_topk),
               'compressor_state': M.compare_per_rank(M.compare_state)}
    for name, slots in M.CACHE_SLOTS.items():
        compare[name] = M.compare_per_rank(M.compare_cache(name), slots)
    result = run(fn=M.make_program(operator, capacity, 1, args.epochs), specs=specs,
                 golden_fn=M.make_golden(args.epochs), compile_only=args.compile_only,
                 config=dict(platform=args.platform,
                             distributed_config=DistributedConfig(device_ids=[0], num_sub_workers=0)),
                 compare_fn=compare)
    print(f'RESULT {args.mode} {args.platform} epochs={args.epochs}: passed={result.passed} '
          f'work_dir={result.work_dir} error={result.error}', flush=True)
    assert result.passed, result.error


if __name__ == '__main__':
    main()
