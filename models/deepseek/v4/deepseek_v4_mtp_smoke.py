# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MTP smoke runner.

Runs the lib-local MTP validation chain:

* projection: ``hidden_states + prev_hidden_states -> mtp_hidden``
* logits: ``mtp_hidden -> candidate_logits``
* candidates: ``candidate_logits -> candidate_token_ids``
* tail: ``projection -> logits``
* acceptance: ``candidate ids + verify ids -> accepted metadata``

This runner intentionally stays inside pypto-lib model smokes. Full serving
integration, token sampling, TP vocabulary routing, and KV-cache commit remain
owned by the serving/decode integration layer.
"""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


SMOKE_STEPS = (
    (
        "projection",
        "mtp_projection.py",
        (),
    ),
    (
        "local-logits",
        "deepseek_v4_mtp_logits.py",
        ("--case", "local-logits"),
    ),
    (
        "shared-head-logits",
        "deepseek_v4_mtp_logits.py",
        ("--case", "shared-head-logits"),
    ),
    (
        "candidates",
        "deepseek_v4_mtp_candidates.py",
        (),
    ),
    (
        "projection-to-logits-tail",
        "deepseek_v4_mtp_tail.py",
        (),
    ),
    (
        "acceptance-metadata",
        "deepseek_v4_mtp_acceptance.py",
        (),
    ),
)


def _step_command(args, script, extra_args):
    cmd = [
        sys.executable,
        str(ROOT / script),
        "-p",
        args.platform,
        "-d",
        str(args.device),
    ]
    if args.compile_only:
        cmd.append("--compile-only")
    if args.runtime_dir:
        cmd.extend(["--runtime-dir", str(Path(args.runtime_dir) / Path(script).stem)])
    if args.dump_passes:
        cmd.append("--dump-passes")
    if args.enable_l2_swimlane:
        cmd.append("--enable-l2-swimlane")
    cmd.extend(extra_args)
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument(
        "--step",
        action="append",
        choices=[name for name, _, _ in SMOKE_STEPS],
        help="Run only the selected step. May be passed multiple times.",
    )
    args = parser.parse_args()

    selected = set(args.step or [])
    for name, script, extra_args in SMOKE_STEPS:
        if selected and name not in selected:
            continue
        cmd = _step_command(args, script, extra_args)
        print(f"[mtp-smoke] running {name}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
