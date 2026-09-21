# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the production inline entries with active-prefix and poisoned padding."""
import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from golden import run, TensorSpec
from pypto.ir.distributed_compiled_program import DistributedConfig
from models.deepseek_v4_1_flash import decode_c1a_full as F

TOKEN_NAMES = {
    "x", "x_hc", "pre_mix", "rope_cos", "rope_sin", "window_slots", "window_indices",
    "request_ids", "compressed_lens", "compressed_rope_cos", "compressed_rope_sin",
    "compressed_slots", "compressed_indices", "candidate_mask", "topk_indices",
    "output", "next_pre_mix",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full", "reindex", "reuse"), required=True)
    parser.add_argument("--active", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=9)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("-d", default="0")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    active = args.active
    values = F.build_hc_validation_values(args.mode, args.tokens, 2, seed=37)
    values["output"].fill_(17)
    values["next_pre_mix"].fill_(19)
    values["x_hc"][:, active:] = float("nan")
    values["pre_mix"][:, active:] = float("nan")
    for name in ("window_slots", "compressed_slots", "request_ids"):
        if name in values:
            values[name][:, active:] = -1
    wrapper = importlib.import_module(f"padding_{args.mode}")
    module = importlib.import_module(f"models.deepseek_v4_1_flash.decode_c1a_{args.mode}")
    host = wrapper.make_program(args.tokens, 2, epochs=2, active=active)

    def reference(tensors):
        if active:
            prefix = {n: (v[:, :active] if n in TOKEN_NAMES else v) for n, v in tensors.items()}
            F.golden_c1a_hc_case(prefix, getattr(module, f"golden_decode_attn_c1a_{args.mode}"), 2)

    def compare_prefix(compare):
        def check(actual, expected, **kwargs):
            assert torch.equal(actual[:, active:], expected[:, active:]), "padding changed"
            if not active:
                return True, "empty prefix preserves padding"
            return compare(actual[:, :active], expected[:, :active], **kwargs)
        return check

    comparisons = {
        "output": compare_prefix(F.output_hc_compare),
        "next_pre_mix": compare_prefix(F.next_pre_mix_compare),
        "topk_indices": F.exact_bytes,
        "candidate_mask": F.exact_bytes,
    }
    for name, scale, slots, group, fmt in (
        ("window_cache", "window_cache_scale", "window_slots", F.WINDOW_CACHE_GROUP, "e8m0"),
        ("compressed_cache", "compressed_cache_scale", "compressed_slots", F.COMPRESSED_CACHE_GROUP, "e4m3"),
        ("index_cache", "index_cache_scale", "compressed_slots", F.INDEX_CACHE_GROUP, "e8m0"),
    ):
        if name not in host.param_names:
            continue
        base = F.quantized_cache_compare(name, scale, slots, F.MXFP4_CACHE_MAX_RELATIVE_L2,
                                        group_size=group, scale_format=fmt)
        def checked(actual, expected, _base=base, **kwargs):
            inputs = dict(kwargs.get("inputs", {}))
            for n in ("window_slots", "compressed_slots"):
                if n in inputs:
                    inputs[n] = inputs[n][:, :active]
            kwargs["inputs"] = inputs
            return _base(actual, expected, **kwargs)
        comparisons[name] = comparisons[scale] = checked
    specs = [TensorSpec(n, list(values[n].shape), values[n].dtype,
                        init_value=lambda n=n: values[n].clone()) for n in host.param_names]
    result = run(fn=host, specs=specs, golden_fn=reference, compile_only=args.compile_only,
                 compare_fn=comparisons,
                 config={"platform": "a5", "distributed_config": DistributedConfig(
                     device_ids=[int(d) for d in args.d.split(",")], num_sub_workers=0)})
    assert result.passed, result


if __name__ == "__main__":
    main()
