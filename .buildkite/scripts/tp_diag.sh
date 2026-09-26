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

# Prints what the container sees, then times test_tp_performance twice in the
# same container. Run inside the test image on kube and on bare metal to find
# why TP=8 generation is slower on kube while TP=1 is not.

set -u

section() { echo "--- $*"; }

section "host"
uname -a
cat /proc/cmdline 2>/dev/null
nproc
lscpu 2>/dev/null | grep -E 'Model name|Thread|Core\(s\)|Socket|NUMA|MHz'
for f in enabled defrag; do
  echo "thp $f: $(cat /sys/kernel/mm/transparent_hugepage/$f 2>/dev/null)"
done
grep -H . /sys/devices/system/cpu/vulnerabilities/* 2>/dev/null
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null
echo "cpuidle driver: $(cat /sys/devices/system/cpu/cpuidle/current_driver 2>/dev/null)" \
  "governor: $(cat /sys/devices/system/cpu/cpuidle/current_governor_ro 2>/dev/null)"
for s in /sys/devices/system/cpu/cpu0/cpuidle/state*; do
  [ -d "$s" ] && echo "$(basename "$s"): $(cat "$s/name") latency=$(cat "$s/latency")us" \
    "usage=$(cat "$s/usage") disable=$(cat "$s/disable")"
done

section "wake-up latency"
python3 "$(dirname "$0")/wakeup_bench.py"

section "process"
ulimit -a
grep -E 'Seccomp|NoNewPrivs|Speculation|Cpus_allowed_list|Mems_allowed_list' /proc/self/status
cat /proc/self/cgroup
cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpu.weight 2>/dev/null
df -h /dev/shm
env | grep -E '^(TPU_|LIBTPU|XLA_|JAX_|VLLM_|MEGASCALE|OMP_|MALLOC)' | sort

section "load"
uptime
ps -eo pcpu,pid,comm --sort=-pcpu 2>/dev/null | head -8

[ "${1:-}" = "--no-tp" ] && exit 0

for i in 1 2; do
  section "test_tp_performance run $i"
  python3 -m pytest -s -v -x \
    /workspace/tpu_inference/tests/e2e/test_tensor_parallel.py::test_tp_performance
  echo "run $i exit $?"
done
exit 0
