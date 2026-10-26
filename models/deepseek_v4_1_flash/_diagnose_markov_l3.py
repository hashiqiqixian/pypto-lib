"""Temporary four-rank Markov numerical check; not part of PR #1541."""

import torch

from golden import ScalarSpec, TensorSpec, run
from pypto.ir import DistributedConfig

import config as C
from dspark_markov import D, MARKOV_RANK, VOCAB, l3_markov_sample_from_hidden


assert (C.TP_SIZE, C.DP_SIZE, C.EP_SIZE) == (1, 4, 4)
RANKS = C.EP_SIZE
BATCH = 1
STEPS = C.DSPARK_QUERY_WIDTH

w1 = torch.zeros((RANKS, VOCAB, MARKOV_RANK), dtype=torch.bfloat16)
w2 = torch.zeros_like(w1)
for rank in range(RANKS):
    token = 11 + rank
    w1[rank, 7, 0] = 2
    w1[rank, token, 0] = 2
    w2[rank, token, 0] = 3


def golden(values):
    for rank in range(RANKS):
        values["draft_ids"][rank].fill_(11 + rank)
    values["confidences"].fill_(0.5)


specs = [
    TensorSpec("head_hidden", [RANKS, BATCH, STEPS, D], torch.bfloat16, init_value=0),
    TensorSpec("final_norm_weight", [RANKS, D], torch.bfloat16, init_value=1),
    TensorSpec("lm_head_weight", [RANKS, VOCAB, D], torch.bfloat16, init_value=0),
    TensorSpec("num_sampled", [RANKS, BATCH], torch.int32, init_value=0),
    TensorSpec("last_sampled", [RANKS, BATCH], torch.int64, init_value=0),
    TensorSpec("next_prefill_tokens", [RANKS, BATCH], torch.int64, init_value=7),
    TensorSpec("markov_w1", [RANKS, VOCAB, MARKOV_RANK], torch.bfloat16, init_value=w1),
    TensorSpec("markov_w2", [RANKS, VOCAB, MARKOV_RANK], torch.bfloat16, init_value=w2),
    TensorSpec("confidence_weight", [RANKS, 1, D + MARKOV_RANK], torch.float32, init_value=0),
    TensorSpec("draft_ids", [RANKS, BATCH, STEPS], torch.int32),
    TensorSpec("confidences", [RANKS, BATCH, STEPS], torch.float32),
    ScalarSpec("logits_epoch", torch.int32, 1, compile_runtime=True),
]

result = run(
    fn=l3_markov_sample_from_hidden,
    specs=specs,
    golden_fn=golden,
    config={"platform": "a5", "distributed_config": DistributedConfig(device_ids=[0, 1, 2, 3], num_sub_workers=0)},
    rtol=1e-3,
    atol=1e-3,
)
print(result)
if not result.passed:
    raise SystemExit(1)
