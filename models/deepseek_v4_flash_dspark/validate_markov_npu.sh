#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
batch="${DSPARK_BATCH:?DSPARK_BATCH must be set}"

cd "${repo_root}"
exec .venv/bin/python models/deepseek_v4_flash_dspark/markov_sample.py \
    --batch "${batch}" \
    --platform a2a3 \
    --device 0
