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

# Prints what the container sees, then times test_tp_performance in the
# same container. Run inside the test image on kube and on bare metal to find
# why TP=8 generation is slower on kube while TP=1 is not.

set -u

# --no-tp          facts and wake-up latency only
# --runs N         test_tp_performance runs in this container (default 2)
# --log-compiles   log every JAX compile, to see any that land in the timed run
# --prewarm-image  read the container's root filesystem first, so files the
#                  image streamer would fetch on first touch are already local
# --sleep N        idle N seconds before the first run, past pod-start work
#                  such as the gcsfuse sidecar's metadata prefetch
# --sample         log node-wide CPU busy % and load every 10s in the background
# --spin           keep every CPU awake with one SCHED_IDLE busy loop per CPU.
#                  Any normal thread preempts them, so a woken engine thread
#                  lands on a running CPU instead of waking a halted one - the
#                  userspace analogue of booting with idle=poll.
# --warmup-ici     30s of 8-chip psum (ICI all-reduce) before the first run
# --warmup-chip    30s of per-chip matmuls, no collectives, before the first run
# --warmup-hbm     write ~85% of every chip's HBM once, in a separate process,
#                  before the first run
# --local-jax-cache  compile into an empty local directory instead of the
#                  shared cache, so nothing is read through gcsfuse lazily
# --local-hf       copy the test model to local disk and load it from there
# --diff-writes    list every file run 1 creates or changes, by directory
# --compileall     byte-compile the installed Python before the first run
RUNS=2
NO_TP=0
WARMUP=
LOCAL_HF=0
DIFF_WRITES=0
COMPILEALL=0
PREWARM=0
SLEEP=0
SAMPLE=0
SPIN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --no-tp) NO_TP=1 ;;
    --runs) RUNS="$2"; shift ;;
    --log-compiles) export JAX_LOG_COMPILES=1 ;;
    --prewarm-image) PREWARM=1 ;;
    --sleep) SLEEP="$2"; shift ;;
    --sample) SAMPLE=1 ;;
    --spin) SPIN=1 ;;
    --warmup-ici) WARMUP=ici ;;
    --warmup-chip) WARMUP=chip ;;
    --warmup-hbm) WARMUP=hbm ;;
    --local-jax-cache)
      export JAX_COMPILATION_CACHE_DIR=/tmp/jax-cache VLLM_XLA_CACHE_PATH=/tmp/jax-cache ;;
    --local-hf) LOCAL_HF=1 ;;
    --diff-writes) DIFF_WRITES=1 ;;
    --compileall) COMPILEALL=1 ;;
  esac
  shift
done

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

SPINNERS=()
if [ "$SPIN" = 1 ]; then
  for _ in $(seq 1 "$(nproc)"); do
    python3 -c 'import os; os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
while True: pass' &
    SPINNERS+=($!)
  done
  trap 'kill ${SPINNERS[*]} ${SAMPLER:-} 2>/dev/null' EXIT
  echo "started ${#SPINNERS[@]} SCHED_IDLE spinners"
fi

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

[ "$NO_TP" = 1 ] && exit 0

# /proc/stat is not namespaced, so this sees the whole node, sidecars included.
node_sampler() {
  local prev_busy=0 prev_total=0
  while true; do
    read -r _ user nice system idle iowait irq softirq steal _ < /proc/stat
    local busy=$((user + nice + system + irq + softirq + steal))
    local total=$((busy + idle + iowait))
    if [ "$prev_total" -gt 0 ]; then
      echo "[sample $(date +%H:%M:%S)] node cpu busy" \
        "$(( 100 * (busy - prev_busy) / (total - prev_total) ))%" \
        "load $(cut -d' ' -f1-3 /proc/loadavg)" \
        "| node psi $(head -1 /proc/pressure/cpu 2>/dev/null | cut -d' ' -f2)" \
        "| pod psi $(head -1 /sys/fs/cgroup/cpu.pressure 2>/dev/null | cut -d' ' -f2)"
    fi
    prev_busy=$busy prev_total=$total
    sleep 10
  done
}
[ "$SAMPLE" = 1 ] && { node_sampler & SAMPLER=$!; trap 'kill ${SPINNERS[*]:-} $SAMPLER 2>/dev/null' EXIT; }

if [ "$SLEEP" -gt 0 ]; then
  section "idle ${SLEEP}s"
  sleep "$SLEEP"
fi

if [ "$PREWARM" = 1 ]; then
  section "prewarm image"
  start=$(date +%s)
  find / -xdev -type f -size -2G -print0 2>/dev/null |
    xargs -0 -P 32 -n 200 cat > /dev/null 2>&1
  echo "read the root filesystem in $(( $(date +%s) - start ))s"
fi

if [ "$LOCAL_HF" = 1 ]; then
  section "copy the test model to local disk"
  src="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--meta-llama--Llama-3.1-8B-Instruct"
  start=$(date +%s)
  mkdir -p /tmp/hf/hub && cp -a "$src" /tmp/hf/hub/
  du -sh /tmp/hf/hub/* 2>/dev/null
  echo "copied in $(( $(date +%s) - start ))s"
  export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1
fi

if [ -n "$WARMUP" ]; then
  section "warm-up: $WARMUP"
  WARMUP="$WARMUP" python3 - <<'PY'
import os, time
import jax, jax.numpy as jnp
devs = jax.devices()
mode = os.environ["WARMUP"]
end = time.time() + 30
n = 0
if mode == "ici":
    from jax.sharding import NamedSharding, PartitionSpec as P
    shard = NamedSharding(jax.make_mesh((len(devs),), ("i",)), P("i"))
    x = jax.device_put(jnp.ones((len(devs), 1 << 22), jnp.float32), shard)
    # Summing over the sharded axis is an all-reduce across every chip.
    f = jax.jit(lambda a: a + a.sum(axis=0, keepdims=True) * 1e-9,
                out_shardings=shard)
    while time.time() < end:
        x = f(x)
        x.block_until_ready()
        n += 1
elif mode == "hbm":
    chunk = 1 << 30  # bytes per array
    for d in devs:
        limit = d.memory_stats().get("bytes_limit", 32 << 30)
        held = []
        while (len(held) + 1) * chunk < 0.85 * limit:
            held.append(jnp.ones((chunk // 4,), jnp.float32, device=d))
        for a in held:
            a.block_until_ready()
        n += len(held)
        del held
else:
    xs = [jax.device_put(jnp.ones((4096, 4096), jnp.bfloat16), d) for d in devs]
    mm = jax.jit(lambda a: a @ a)
    while time.time() < end:
        xs = [mm(a) * (1.0 / 4096) for a in xs]
        for a in xs:
            a.block_until_ready()
        n += 1
print(f"{mode} warm-up: {n} {'GiB written' if mode == 'hbm' else 'iterations'} on {len(devs)} devices")
PY
fi

section "bytecode already in the image"
for d in /workspace/vllm /workspace/tpu_inference /usr/local/lib/python3*/site-packages; do
  [ -d "$d" ] && echo "$d: $(find "$d" -name '*.py' | wc -l) .py, $(find "$d" -name '*.pyc' | wc -l) .pyc"
done
echo "HOME=$HOME XDG_CACHE_HOME=${XDG_CACHE_HOME:-} VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-}"
ls -la "$HOME/.cache" 2>/dev/null

if [ "$COMPILEALL" = 1 ]; then
  section "compileall"
  start=$(date +%s)
  python3 -m compileall -q -j 0 /workspace /usr/local/lib/python3*/site-packages > /dev/null 2>&1
  echo "byte-compiled in $(( $(date +%s) - start ))s"
fi

writes_since() {
  find / -xdev \( -path /proc -o -path /sys -o -path /dev \) -prune -o \
    -type f -newer "$1" -print 2>/dev/null
}

for i in $(seq 1 "$RUNS"); do
  [ "$DIFF_WRITES" = 1 ] && [ "$i" = 1 ] && touch /tmp/.before_run1
  section "test_tp_performance run $i"
  python3 -m pytest -s -v -x \
    /workspace/tpu_inference/tests/e2e/test_tensor_parallel.py::test_tp_performance
  echo "run $i exit $?"
  if [ "$DIFF_WRITES" = 1 ] && [ "$i" = 1 ]; then
    section "files run 1 wrote, by top directory"
    writes_since /tmp/.before_run1 > /tmp/run1_writes.txt
    echo "$(wc -l < /tmp/run1_writes.txt) files"
    awk -F/ '{print "/"$2"/"$3"/"$4}' /tmp/run1_writes.txt | sort | uniq -c | sort -rn | head -40
    grep -v '\.pyc$' /tmp/run1_writes.txt | head -60
  fi
done
exit 0
