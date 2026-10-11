#!/usr/bin/env bash
set -euo pipefail

base=/data/disk2/pyptouser/chenshenai/test0
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export PTOAS_ROOT="$base/.venv-v41-m0"
export PYTHONPATH="$root:$base/pypto-lib-main:$base/pypto:$base/pypto/runtime:$base/pypto/runtime/python:$base/pypto/runtime/build/cp311-cp311-linux_aarch64/python/bindings"

exec "$base/.venv-v41-m0/bin/python" tools/probe_v41_swa_score_block.py \
  "$base/v41-boundary-probe-20261010/attended-seed17-rank${V41_RANK:-2}-current.pt" \
  --token "${V41_TOKEN:-74}" --head "${V41_HEAD:-2}" --device "${TASK_DEVICE:-0}"
