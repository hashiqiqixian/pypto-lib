# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the production full-table forward with repeated row consumers."""
import argparse
from types import SimpleNamespace

import torch
from golden import run
from pypto.ir import DistributedConfig
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash import decode_c2a_full as D
from models.deepseek_v4_1_flash import prefill_c2a_full as P


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp', type=int, default=1, choices=[1])
    parser.add_argument('--ep', type=int, default=2, choices=[2])
    parser.add_argument('--mode', choices=['decode', 'prefill'], default='decode')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--platform', choices=['a5', 'a5sim'], default='a5')
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--host-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert C.TP_SIZE == 1
    cfg = SimpleNamespace(dp=1, tokens=8, requests=3, seed=29, case='mixed',
                          epochs=args.epochs, bench=False)
    if args.mode == 'decode':
        specs = D.build_specs(cfg, 'decode')
        program = D.make_program(D.decode_c2a_full, C.DECODE_MAX_TOKENS, 1, args.epochs)
        golden = D.make_golden(args.epochs)
        compare = {'output': D.compare_replicated(D.compare_output),
                   'topk_indices': D.compare_per_rank(D.compare_topk),
                   'state_cache': D.compare_per_rank(D.compare_state, D.STATE_METADATA)}
        for name, slots in D.CACHE_SLOTS.items():
            compare[name] = D.compare_per_rank(D.compare_cache(name), slots)
    else:
        initial_state = {}
        specs = P.build_specs(cfg, 'full', initial_state)
        program = P.make_hc_program(C.PREFILL_MAX_TOKENS, 1, args.epochs)
        golden = P.make_golden('full', args.epochs)
        compare = P.make_compare('full', args.epochs, initial_state)
    # A different but finite profile exercises caller-owned data and prevents
    # accidental regeneration of default frequencies from passing the golden.
    for spec in specs:
        if spec.name in D.FULL_ROPE_NAMES.values():
            original = spec.init_value
            spec.init_value = lambda original=original: original().roll(5, dims=1)
    if args.host_only:
        tensors = {}
        for spec in specs:
            if hasattr(spec, 'shape'):
                init = spec.init_value
                tensors[spec.name] = init() if callable(init) else (
                    init if isinstance(init, torch.Tensor) else torch.zeros(spec.shape, dtype=spec.dtype))
        golden(tensors)
        key = 'output' if args.mode == 'decode' else 'x_hc_out'
        assert torch.isfinite(tensors[key]).all()
        assert tensors['freqs_cos'].shape[1] != cfg.tokens
        print(f'HOST {args.mode} epochs={args.epochs}: PASS', flush=True)
        return
    result = run(fn=program, specs=specs, golden_fn=golden, compare_fn=compare,
                 compile_only=args.compile_only,
                 config=dict(platform=args.platform,
                             distributed_config=DistributedConfig(device_ids=[0], num_sub_workers=0)))
    print(f'RESULT {args.mode} epochs={args.epochs} {args.platform}: '
          f'passed={result.passed} work_dir={result.work_dir} error={result.error}', flush=True)
    assert result.passed, result.error


if __name__ == '__main__':
    main()
