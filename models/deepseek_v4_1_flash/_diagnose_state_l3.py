"""Temporary EP4 DSpark state-transition check; not part of PR #1541."""

import torch

from golden import TensorSpec, run
from pypto.ir import DistributedConfig

import config as C
import dspark_state as S


assert (C.TP_SIZE, C.DP_SIZE, C.EP_SIZE) == (1, 4, 4)
RANKS = C.EP_SIZE
BATCH = S.BATCH
ROWS = S.ROWS
config = {"platform": "a5", "distributed_config": DistributedConfig(device_ids=[0, 1, 2, 3], num_sub_workers=0)}

slots = torch.full((RANKS, BATCH), -1, dtype=torch.int32)
slots[:, 0] = 0
generations = torch.zeros((RANKS, BATCH), dtype=torch.int32)
generations[:, 0] = torch.arange(1, RANKS + 1, dtype=torch.int32)
tokens = torch.zeros((RANKS, BATCH, S.STATE_TOKEN_WIDTH), dtype=torch.int64)
tokens[:, 0] = torch.arange(20, 20 + ROWS, dtype=torch.int64)
meta = torch.zeros((RANKS, BATCH, S.STATE_META_WIDTH), dtype=torch.int32)
meta[:, 0, S.STATE_VALID] = 1
meta[:, 0, S.STATE_GENERATION] = generations[:, 0]
meta[:, 0, S.STATE_ANCHOR] = 10
meta[:, 0, S.STATE_LIMIT] = 100
meta[:, 0, S.STATE_DRAFT_COUNT] = C.DSPARK_QUERY_WIDTH


def check(name, fn, specs, golden_fn):
    result = run(fn=fn, specs=specs, golden_fn=golden_fn, config=config)
    print(name, result)
    if not result.passed:
        raise SystemExit(1)


common = [
    TensorSpec("slot_ids", list(slots.shape), torch.int32, init_value=slots),
    TensorSpec("generations", list(generations.shape), torch.int32, init_value=generations),
    TensorSpec("state_tokens", list(tokens.shape), torch.int64, init_value=tokens),
    TensorSpec("state_meta", list(meta.shape), torch.int32, init_value=meta),
]


def golden_prepare(values):
    for rank in range(RANKS):
        ids, positions, widths = S.golden_prepare_target(
            values["slot_ids"][rank], values["generations"][rank],
            values["state_tokens"][rank], values["state_meta"][rank],
        )
        values["input_ids"][rank].copy_(ids)
        values["position_ids"][rank].copy_(positions)
        values["active_widths"][rank].copy_(widths)


check("prepare_target", S.l3_prepare_target, common + [
    TensorSpec("input_ids", [RANKS, BATCH, ROWS], torch.int64),
    TensorSpec("position_ids", [RANKS, BATCH, ROWS], torch.int32),
    TensorSpec("active_widths", [RANKS, BATCH], torch.int32),
], golden_prepare)

widths = torch.zeros((RANKS, BATCH), dtype=torch.int32)
widths[:, 0] = ROWS
sampled = torch.zeros((RANKS, BATCH * ROWS, S.SAMPLED_STRIDE), dtype=torch.int32)
for rank in range(RANKS):
    sampled[rank, :rank, 0] = torch.arange(21, 21 + rank, dtype=torch.int32)
    sampled[rank, rank, 0] = 90 + rank


def golden_accept(values):
    for rank in range(RANKS):
        updated_tokens, updated_meta, accepted_ids, counts = S.golden_accept_target(
            values["slot_ids"][rank], values["generations"][rank],
            values["active_widths"][rank], values["sampled_ids"][rank],
            values["state_tokens"][rank], values["state_meta"][rank],
        )
        values["state_tokens"][rank].copy_(updated_tokens)
        values["state_meta"][rank].copy_(updated_meta)
        values["accepted_ids"][rank].copy_(accepted_ids)
        values["accepted_counts"][rank].copy_(counts)
        accepted = int(counts[0])
        values["context_valid"][rank, 0, :accepted] = 1
        values["context_positions"][rank, 0, :accepted] = torch.arange(10, 10 + accepted)
        values["num_sampled"][rank, 0] = accepted
        values["last_sampled"][rank, 0] = 90 + rank
        values["first_query_positions"][rank, 0] = 10 + accepted


check("accept_target", S.l3_accept_target, common + [
    TensorSpec("active_widths", [RANKS, BATCH], torch.int32, init_value=widths),
    TensorSpec("sampled_ids", list(sampled.shape), torch.int32, init_value=sampled),
    TensorSpec("accepted_ids", [RANKS, BATCH, ROWS], torch.int32),
    TensorSpec("accepted_counts", [RANKS, BATCH], torch.int32),
    TensorSpec("context_valid", [RANKS, BATCH, ROWS], torch.int32),
    TensorSpec("context_positions", [RANKS, BATCH, ROWS], torch.int32),
    TensorSpec("num_sampled", [RANKS, BATCH], torch.int32),
    TensorSpec("last_sampled", [RANKS, BATCH], torch.int64),
    TensorSpec("first_query_positions", [RANKS, BATCH], torch.int32),
], golden_accept)

commit_meta = meta.clone()
commit_meta[:, 0, S.STATE_DRAFT_COUNT] = 0
drafts = torch.zeros((RANKS, BATCH, C.DSPARK_QUERY_WIDTH), dtype=torch.int32)
for rank in range(RANKS):
    drafts[rank, 0] = torch.arange(30 + rank, 30 + rank + C.DSPARK_QUERY_WIDTH)


def golden_commit(values):
    for rank in range(RANKS):
        updated_tokens, updated_meta, committed = S.golden_commit_drafts(
            values["slot_ids"][rank], values["generations"][rank],
            values["draft_ids"][rank], values["state_tokens"][rank],
            values["state_meta"][rank],
        )
        values["state_tokens"][rank].copy_(updated_tokens)
        values["state_meta"][rank].copy_(updated_meta)
        values["committed"][rank].copy_(committed)


check("commit_drafts", S.l3_commit_drafts, [
    common[0], common[1],
    TensorSpec("draft_ids", list(drafts.shape), torch.int32, init_value=drafts),
    common[2],
    TensorSpec("state_meta", list(commit_meta.shape), torch.int32, init_value=commit_meta),
    TensorSpec("committed", [RANKS, BATCH], torch.int32),
], golden_commit)
