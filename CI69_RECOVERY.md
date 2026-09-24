# ci-69 MTP prefill diagnostic recovery (2026-09-24)

Recovered from test0/pypto-lib at base 77e094adc3e12093c019d122cabb6a39277d6ae9.
The only source delta exposes x_attn_debug through the prefill layer,
records its golden value, and compares the intermediate attention residual.
This is diagnostic instrumentation, not a proven numerical fix.

Historical build directories named _jit_l3_prefill_layer_20260904_* and
_jit_l3_prefill_layer_20260907_* contain distributed metadata. Their presence
only establishes historical build activity: no matching numerical PASS log
or source fingerprint was recovered for this exact dirty delta.
No new device run was performed during recovery.

The source is archived separately from active development. Build outputs,
tensor snapshots and machine environments are not included.

Local recovery checks: Python syntax and all pre-commit checks passed.
This is not a kernel/device validation.
