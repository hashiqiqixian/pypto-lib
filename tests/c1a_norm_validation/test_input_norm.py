# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent checks for C1A hidden-size normalization and its reference wiring."""
import functools
import importlib

import pytest
import torch

from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash import decode_c1a_full as F


def expected_input(x_hc, pre_mix, weight):
    # Preserve the model's BF16 collapse boundary, then independently evaluate
    # RMSNorm through Torch's functional implementation rather than lib golden.
    collapsed = torch.einsum("thd,th->td", x_hc.float(), pre_mix.float()).bfloat16()
    normalized = torch.nn.functional.rms_norm(
        collapsed.float(), (C.D,), weight.float(), eps=C.FLASH.rms_norm_eps
    ).bfloat16()
    return collapsed, normalized


@pytest.mark.parametrize("mode", ["full", "reindex", "reuse"])
@pytest.mark.parametrize("tokens", [1, 9])
def test_real_golden_normalizes_before_attention(mode, tokens):
    torch.set_num_threads(4)
    values = F.build_hc_validation_values(mode, tokens, 1, seed=37, case="random")
    values["x_hc"] *= torch.linspace(0.25, 3.0, tokens).reshape(1, tokens, 1, 1)
    module = importlib.import_module(f"models.deepseek_v4_1_flash.decode_c1a_{mode}")
    original = getattr(module, f"golden_decode_attn_c1a_{mode}")
    seen = []

    @functools.wraps(original)
    def checked(*args, **kwargs):
        rank = len(seen) % C.TP_SIZE
        collapsed, expected = expected_input(
            values["x_hc"][rank], values["pre_mix"][rank], values["attn_norm_weight"][rank]
        )
        torch.testing.assert_close(kwargs["x"], expected, rtol=0.008, atol=0.001)
        assert not torch.allclose(collapsed.float(), expected.float(), rtol=0.01, atol=0.01)
        seen.append(rank)
        return original(*args, **kwargs)

    before_x = values["x_hc"].clone()
    before_pre = values["pre_mix"].clone()
    F.golden_c1a_hc_case(values, checked, epochs=2)
    assert len(seen) == 2 * C.TP_SIZE
    assert torch.isfinite(values["output"]).all()
    assert values["output"].abs().max() > 0
    assert torch.equal(values["x_hc"], before_x)
    assert torch.equal(values["pre_mix"], before_pre)
    host = module.make_program(tokens, 1, 2)
    assert set(host.param_names).issubset(values)
    assert values["attn_norm_weight"].shape == (C.TP_SIZE, C.D)
    assert not torch.all(values["attn_norm_weight"] == 1)
    for rank in range(1, C.TP_SIZE):
        assert torch.equal(values["attn_norm_weight"][0], values["attn_norm_weight"][rank])


@pytest.mark.parametrize("mode", ["full", "reindex"])
def test_topk_comparator_reconstructs_normalized_input(monkeypatch, mode):
    gen = torch.Generator().manual_seed(61)
    inputs = {
        "x_hc": 3 * torch.randn(C.TP_SIZE, 3, C.HC_MULT, C.D, generator=gen),
        "pre_mix": torch.rand(C.TP_SIZE, 3, C.HC_MULT, generator=gen),
        "attn_norm_weight": torch.linspace(0.5, 1.5, C.D).bfloat16().repeat(C.TP_SIZE, 1),
    }
    calls = []

    def check(actual, expected, **kwargs):
        for rank in range(C.TP_SIZE):
            _, normalized = expected_input(
                inputs["x_hc"][rank], inputs["pre_mix"][rank], inputs["attn_norm_weight"][rank]
            )
            torch.testing.assert_close(kwargs["inputs"]["x"][rank], normalized, rtol=0.008, atol=0.001)
        calls.append(True)
        return True, ""

    monkeypatch.setattr(F, "topk_indices_compare", lambda selected_mode: check)
    assert F.hc_topk_indices_compare(mode)(None, None, inputs=inputs)[0]
    assert calls == [True]


if __name__ == "__main__":
    # config.py consumes --tp/--ep on import; pytest gets only its own options.
    raise SystemExit(pytest.main([__file__, "-q", "-x", "--disable-warnings"]))
