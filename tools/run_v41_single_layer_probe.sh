#!/usr/bin/env bash
set -euo pipefail

base=/data/disk2/pyptouser/chenshenai/test0
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export PTOAS_ROOT="$base/.venv-v41-m0"
export PYTHONPATH="$root:$base/pypto-lib-main:$base/pypto:$base/pypto/runtime:$base/pypto/runtime/python/bindings:$base/pypto/runtime/build/cp311-cp311-linux_aarch64/python/bindings"

exec "$base/.venv-v41-m0/bin/python" -m models.deepseek_v4_1_flash.decode_layer \
  --platform a5 --tp 4 --ep 8 --device 0,1,2,3,4,5,6,7 \
  --layer-id 0 --seed "${V41_SEED:-17}"
