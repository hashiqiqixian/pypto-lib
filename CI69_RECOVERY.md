# ci-69 test3 working-tree recovery (2026-09-24)

Base: 251322362374c422d3253e23ce18ba3660c94b82.
Recovered all 56 tracked changes and six untracked documents together.
This is a mixed working-tree snapshot, not an integration-ready patch.

Changes include the golden run API/output-spec migration and corresponding
model/example updates; DSpark context/metadata changes (including the 1M
context ceiling and distributed CSA start-position handling); documentation
relocations; CI edits; and removal of older helper implementations.
These coupled changes are kept together to preserve the original state.

Evidence: dspark-latest-ab69-20260923/baseline/run.json identifies clean lib
3ff92ebb72271d9227c7792e12af709132d7ec0c. Its hc_mean/run.json identifies
97d99bf42b08649b74154449e06a824c00d10972. The screenshot reproduction
run/run.json identifies 7b2c3f94020b963c1afdf3033917676adb074378.
All three point to other worktrees, so their results do NOT validate this
uncommitted snapshot. The daily-CI memory-limit explanation is recovered
source commentary, not a newly reproduced measurement.
No new device run was performed during recovery.

Local recovery checks: all 28 changed Python files parsed successfully.
Pre-commit did not pass: the recovered decode_sparse_attn_csa.py is empty;
two recovered documentation links point to absent targets (DSpark
prefill_metadata.py and Pro utils.py). A Windows console encoding error
also affected the initial English-only hook. Ruff passed. These preexisting
snapshot problems are retained rather than repaired in this archive.
