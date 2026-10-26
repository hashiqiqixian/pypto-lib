"""Temporary EP4 target-tail sampling check; not part of PR #1541."""

import torch

from golden import ScalarSpec, TensorSpec, run
from pypto.ir import DistributedConfig

import config as C
from dspark_target_sample import D, HC_MULT, ROWS, l3_sample_target_chunk
from lm_head import MAX_LOGIT_ROWS, SAMPLED_IDS_PAD, VOCAB_PER_TP


assert (C.TP_SIZE, C.DP_SIZE, C.EP_SIZE) == (1, 4, 4)
RANKS = C.EP_SIZE
TOKENS = (ROWS + 15) // 16 * 16
CHUNK = 4

hidden = torch.zeros((RANKS, TOKENS, HC_MULT, D), dtype=torch.float32)
hidden[:, :, 0, 0] = 1
weights = torch.zeros((RANKS, VOCAB_PER_TP, D), dtype=torch.bfloat16)
for rank in range(RANKS):
    weights[rank, 13 + rank, 0] = 2


def golden(values):
    values["sampled_ids"].zero_()
    for rank in range(RANKS):
        values["sampled_ids"][rank, : ROWS - CHUNK * MAX_LOGIT_ROWS, 0] = 13 + rank


specs = [
    TensorSpec("x_hc", list(hidden.shape), torch.float32, init_value=hidden),
    TensorSpec("pre_mix", [RANKS, TOKENS, HC_MULT], torch.float32, init_value=1),
    TensorSpec("final_norm_weight", [RANKS, D], torch.bfloat16, init_value=1),
    TensorSpec("lm_head_weight", list(weights.shape), torch.bfloat16, init_value=weights),
    TensorSpec("active_widths", [RANKS, C.MAX_BATCH_PER_DP], torch.int32, init_value=6),
    TensorSpec("sampled_ids", [RANKS, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], torch.int32),
    ScalarSpec("chunk", torch.int32, CHUNK, compile_runtime=True),
]

result = run(
    fn=l3_sample_target_chunk,
    specs=specs,
    golden_fn=golden,
    config={"platform": "a5", "distributed_config": DistributedConfig(device_ids=[0, 1, 2, 3], num_sub_workers=0)},
)
print(result)
if not result.passed:
    raise SystemExit(1)
