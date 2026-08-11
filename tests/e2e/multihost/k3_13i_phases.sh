#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Attempt-13i phase driver. Phase order is failure-robust (the scoring
# probe is minutes and runs before anything that has historically killed
# the server), and phase failures are ACCUMULATED, not short-circuited:
# a dead instrument must fail the job, but must not rob us of the later
# phases' data. Green-by-state with a dead instrument has now happened
# three times (775 bench exit-0, 779's 0.062 "pass", the 812 validator);
# this driver is the fix.
set -u
DIR="$(dirname "$0")"
overall_rc=0

echo "=== [phase 1/3] serving probes ==="
python3 "${DIR}/serving_probe.py" --concurrency 4 --long-tokens 256 \
  --needle-tokens 8000
rc=$?
echo "[13i-phases] serving_probe rc=${rc}"
if [ "${rc}" -ne 0 ]; then overall_rc=1; fi

echo "=== [phase 2/3] teacher-forced scoring stability probe ==="
python3 "${DIR}/k3_server_scoring_probe.py"
rc=$?
echo "[13i-phases] scoring_probe rc=${rc} (2 = INCOMPLETE instrument)"
if [ "${rc}" -ne 0 ]; then overall_rc=1; fi

echo "=== [phase 3/3] gsm8k 250 + wrong-answer dump ==="
bash "${DIR}/k3_acc_bench.sh" --skip-bench
rc=$?
echo "[13i-phases] acc_bench rc=${rc}"
if [ "${rc}" -ne 0 ]; then overall_rc=1; fi

echo "[13i-phases] overall_rc=${overall_rc}"
exit "${overall_rc}"
