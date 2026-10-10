#!/usr/bin/env bash
set -euo pipefail

base=/data/disk2/pyptouser/chenshenai/test0
root="$base/v41-single-layer-real-20261011"
source /usr/local/Ascend/cann-9.3.0/set_env.sh
export PTOAS_ROOT="$base/.venv-v41-m0"
export PYTHONPATH="$root:$base/pypto-serving-pr258:$base/pypto:$base/pypto/runtime:$base/pypto/runtime/python:$base/pypto/runtime/build/cp311-cp311-linux_aarch64/python/bindings"
cd "$root"

prepared="$root/build_output/v41-layer0-real-weights.pt"
if [[ "${V41_PREPARE_ONLY:-0}" == 1 ]]; then
    options=(--prepare-only --save-prepared "$prepared")
else
    test -s "$prepared"
    options=(--prepared "$prepared")
fi

exec "$base/.venv-v41-m0/bin/python" tools/validate_v41_real_swa_layer.py \
    /mnt/old-root/srv/models/DeepSeek-V4.1-Flash --layer-id 0 --seed 17 \
    --device 0,1,2,3,4,5,6,7 "${options[@]}"
