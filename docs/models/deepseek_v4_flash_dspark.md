# DeepSeek V4-Flash DSpark drafter

`models/deepseek_v4_flash_dspark/` contains the V4-Flash target operators and
the DSpark kernel-only drafter. The DSpark slice exposes two public programs:

- `dspark_drafter.py` projects target hidden rows, precomputes the three isolated
  context-KV streams, runs three draft layers, and returns `head_hidden[B, 7, D]`;
- `dspark_markov.py` applies the shared final norm and LM head, then performs seven
  sequential rank-256 Markov transitions and returns `draft_token_ids[B, 7]`.

Serving scheduling, target verification, rejection sampling, bonus-token
commit, confidence-head execution, and checkpoint conversion are outside this
kernel-only contract.

## Token layouts

The backbone keeps the target and draft streams separate.

| Layout | Shape | Purpose |
| --- | --- | --- |
| Target stream | `[T_main, 3 * D]` | Dynamic target/verify rows from hidden layers 40, 41, and 42 |
| Draft attention | `[16 * 7, D]` | Fixed anchor-first query blocks consumed by `dspark_attention` |
| Draft HC/MoE | `[16 * 8, HC_MULT, D]` | Packed active query prefix plus neutral rows for the distributed MoE shape |

`T_main` is independent of the request batch and can carry all
`B * DECODE_SEQ` target rows. This kernel-only contract selects the last target
row of each request as its current anchor. The first `B` entries of
`context_position_ids[DSPARK_QUERY_TOKENS]` and
`context_slot_mapping[3, DSPARK_QUERY_TOKENS]` describe those anchors; the
remaining entries are padding, with cache slots set to `-1`. Precomputing KV for
every target row belongs to the serving handoff and is outside this contract.

The active draft rows are packed as `B * 7` at the start of the fixed buffers.
Each request contributes `[anchor_token, noise_token * 6]`. Rows after that
prefix are zeroed before every MoE stage.

## Backbone composition

The backbone reuses the DSpark leaf operators directly:

```text
dspark_proj(target_hidden) -> main_x
  -> dspark_context_kv(layer 0 cache)
  -> dspark_context_kv(layer 1 cache)
  -> dspark_context_kv(layer 2 cache)

lookup_embedding(anchor/noise ids) -> FP32 HC rows

repeat for mtp.0, mtp.1, mtp.2:
  hc_pre -> rms_norm -> dspark_attention -> hc_post -> moe

hc_head -> head_hidden[B, 7, D]
```

`dspark_attention.py` owns the anchor-block MLA math and cache update.
The local `draft_layer` helper in `dspark_drafter.py` is only the HC/MoE
composition around that operator.
Context KV is produced separately from the same projected target stream for
each layer, matching the vLLM DSpark split between context precompute and the
draft forward.

The visible SWA list is per request, not per query token. Every query in the
seven-token block sees the valid historical window and the entire current
block without an intra-block causal mask. The list width is 192: the 135
possible visible rows are padded to the 64-row attention tile boundary.

Each draft layer owns a distinct slice of one stacked paged KV-cache argument.
The three MoE calls reuse the same signal windows with monotonically increasing
epochs 1, 2, and 3. The composed layer data flow supplies the required ordering,
so no DSpark-specific signal rebasing or cross-rank barrier is needed.

## Markov sampling

The `markov_sample` program in `dspark_markov.py` computes all seven base-logit
rows once. Each unrolled step
then calls `markov_head` with the anchor token or the preceding sampled token,
adds the resulting bias to that step's base logits, and performs greedy
selection. The seven calls remain sequential because step `k + 1` depends on
the token selected at step `k`.

## Validation

The two public programs have independent synthetic validation entrypoints. The
backbone entrypoint uses two expert-parallel ranks, checks the final head hidden
states, all three composed-layer intermediate hidden states, and all three
in/out KV caches. It mixes short-context, full-window, page-boundary, and
wrapped block-table cases in one batch. The Markov entrypoint uses nonzero
synthetic LM-head and rank-256 transition weights. The deterministic final IDs
exercise nonzero base logits and Markov biases at every step, while the Golden
recomputes the same preceding-token causal chain without exposing diagnostic
score tensors through the production interface.

```bash
python models/deepseek_v4_flash_dspark/dspark_drafter.py \
  --batch 4 --ep 2 -p a2a3 -d 0,1
python models/deepseek_v4_flash_dspark/dspark_drafter.py \
  --batch 16 --ep 2 -p a2a3 -d 0,1
python models/deepseek_v4_flash_dspark/dspark_markov.py --batch 4 -p a2a3 -d 0
python models/deepseek_v4_flash_dspark/dspark_markov.py --batch 16 -p a2a3 -d 0
```

The supported dynamic request batches are `{4, 8, 12, 16}`. Contract coverage
includes packed anchor/noise construction, per-request noncausal visibility,
page and ring boundaries, layer cache isolation, three layer intermediates,
and Markov causality. Real-NPU validation should exercise at least B4 and B16
for both public programs with deterministic synthetic inputs and nonzero
projection, embedding, and KV paths.
