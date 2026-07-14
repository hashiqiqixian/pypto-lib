# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dynamic-token contract for DeepSeek V4 packed prefill.

Token-aligned tensors in one compiled leaf invocation share ``TOKENS_DYN``.
Their physical dimension 0 is the number of logical tokens for that call, so
leaf operators derive the loop bound with ``pl.tensor.dim(tensor, 0)`` instead
of accepting a separate valid-prefix scalar.  One invocation is bounded by
``MAX_TOKENS_PER_CALL``; larger requests are split only by the orchestration
layer, while request-boundary and cache metadata remain explicit tensors.

During the Attention-first migration, the public Attention entry points retain
the legacy ``num_tokens`` argument only as a compatibility boundary for callers
that still provide padded tensors.  They immediately slice token-aligned inputs
and outputs to the logical physical shape.  The compatibility argument is
removed when packed prefill and MoE adopt this contract in the follow-up commit.
"""

import pypto.language as pl

from config import PREFILL_TOKENS


TOKENS_DYN = pl.dynamic("DEEPSEEK_V4_TOKENS_DYN")
MAX_TOKENS_PER_CALL = PREFILL_TOKENS


def validate_token_count(num_tokens: int) -> int:
    """Validate one bounded dynamic prefill child invocation."""
    if not 0 < num_tokens <= MAX_TOKENS_PER_CALL:
        raise ValueError(
            f"num_tokens must be in [1, {MAX_TOKENS_PER_CALL}] for one "
            f"DeepSeek V4 prefill child call, got {num_tokens}"
        )
    return num_tokens
