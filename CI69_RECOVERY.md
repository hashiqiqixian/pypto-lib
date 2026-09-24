# ci-69 issue 1275 diagnostic scripts (2026-09-24)

Six standalone scripts recovered from test0, outside its Git repositories.
They are archived under recovery/issue1275 without functional changes.
Base 5a96d70d6fe549c590421adcd6cea734027c6ac3 is the lib version found in
serving's submodule at recovery time, NOT proven to be the exact historical
version used by every script/log. Adapt imports/toolchain before reuse.

Historical log annotations:
- issue1275-concrete-codegen.log: CONCRETE L3 CODEGEN PASS. Codegen only.
- issue1275-tail-a5-build.log: A5 TAIL PTOAS PASS and A5 TAIL BINARY PASS.
  This does not establish A5 device execution or full MoE correctness.
- issue1275-boundary-device.log, boundary-decode-device.log and
  boundary-c1a-device.log: output comparison PASS, shape (3,8,68,4,5120).
  These boundary tests do not load real expert weights.
- issue1275-transport-a3-device-v2.log: comparison PASS including output
  and scale_out; final RUN PASS (7.99s).
- issue1275-transport-loop-a3-device.log: comparison PASS including output
  and scale_out; final RUN PASS (8.33s).
- The A3 transport scripts explicitly substitute byte dtype aliases and
  omit the A5-only final tmov_x2zz scale repack. They test transport, not
  expert arithmetic. These 69 timings are not production benchmarks.
- Earlier issue1275-transport-a3-device.log failed with 'No chip-level
  tasks found'; dispatch-a3-compile.log failed on tile.tmov_x2zz;
  new-codegen.log reported a dynamic extent/codegen error.
- build1275_runtime_bounded.py caps compiler concurrency; the associated
  runtime build/install logs do not constitute numerical validation.

Logs were associated by filename/content. No exact script hash was recorded
in these log excerpts, so historical passes are not a fresh validation of
the recovered bytes. No new device run was performed during recovery.

Local recovery checks: all six scripts parsed successfully. Pre-commit
reported missing standard headers on the standalone scripts and an unused
import; the archive preserves their original contents, including that
import, rather than applying lint rewrites. This is not merge-ready code.
