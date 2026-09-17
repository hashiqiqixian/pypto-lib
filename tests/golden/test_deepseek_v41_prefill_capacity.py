# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-request prefill metadata through the complete 8K query capacity."""

import pytest

torch = pytest.importorskip("torch")

from models.deepseek_v4_1_flash.prefill_c1a_test_utils import _case_metadata


@pytest.mark.parametrize("tokens", [1, 128, 4096, 8192])
def test_causal_prefill_preserves_one_request_across_all_pages(tokens: int) -> None:
    request_ids, lengths, table, slots, window, compressed, candidates = _case_metadata(tokens, "causal")
    assert request_ids.shape == (tokens,)
    assert torch.count_nonzero(request_ids) == 0
    assert torch.equal(lengths, torch.arange(1, tokens + 1, dtype=torch.int32))
    assert torch.equal(slots, torch.arange(tokens, dtype=torch.int64))
    assert table.shape == (1, (tokens + 127) // 128)
    assert torch.equal(window[-1, :min(tokens, 128)], torch.arange(max(0, tokens - 128), tokens))
    assert (window[0, 1:] == -1).all()
    assert (compressed[0, 1:] == -1).all()
    assert candidates[0, 0] == 1
    assert torch.count_nonzero(candidates[0, 1:]) == 0
    assert candidates[-1, :tokens].all()
    assert torch.count_nonzero(candidates[-1, tokens:]) == 0


@pytest.mark.parametrize("tokens", [0, 8193])
def test_causal_prefill_rejects_rows_outside_the_window(tokens: int) -> None:
    with pytest.raises(ValueError, match="token_count must be in"):
        _case_metadata(tokens, "causal")
