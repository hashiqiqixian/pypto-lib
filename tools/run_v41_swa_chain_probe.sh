#!/usr/bin/env bash
set -euo pipefail

base=/data/disk2/pyptouser/chenshenai/test0
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export PTOAS_ROOT="$base/.venv-v41-m0"
export PYTHONPATH="$root:$root/tools:$base/pypto-lib-main:$base/pypto:$base/pypto/runtime:$base/pypto/runtime/python:$base/pypto/runtime/build/cp311-cp311-linux_aarch64/python/bindings"

exec "$base/.venv-v41-m0/bin/python" tools/probe_v41_swa_chain.py \
  "${V41_CAPTURE:-$base/v41-boundary-probe-20261010/boundary-layer0-seed17-v4.pt}" \
  --rank "${V41_RANK:-0}" --device "${TASK_DEVICE:-0}"
