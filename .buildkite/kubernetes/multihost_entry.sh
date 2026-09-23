#!/usr/bin/env bash
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

# Starts Ray across a slice and runs a command on its head.
#
#   multihost_entry.sh <command> [args...]
#
# Every host of the slice runs this with the same arguments and decides its own
# role; nothing outside the pods orchestrates them.
#
# Environment, from manifests/workloads/ray-multihost-slice.yaml:
#   HEAD_HOST      the DNS name of index 0, which every host addresses.
#   NUM_HOSTS      hosts in the slice; the head waits for this many Ray nodes.
#   JOB_COMPLETION_INDEX  set by Kubernetes. 0 is the head.
set -uo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

: "${HEAD_HOST:?HEAD_HOST must name index 0 of the slice}"
: "${NUM_HOSTS:?NUM_HOSTS must be set}"
INDEX="${JOB_COMPLETION_INDEX:-0}"
REPO_DIR="${REPO_DIR:-/workspace/tpu_inference}"
cd "$REPO_DIR" || { echo "$0: no such directory: $REPO_DIR" >&2; exit 2; }

# Where the benchmark scripts write. The pod is deleted when the run ends, so
# the head uploads this before exiting.
# Everything under ARTIFACTS_DIR is uploaded by the head pod before it exits.
# The step names it; unset means this run keeps nothing.
[ -n "${ARTIFACTS_DIR:-}" ] && mkdir -p "$ARTIFACTS_DIR"

# Core dumps are gigabytes each and fill the node's ephemeral storage.
ulimit -c 0

export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
export TPU_MULTIHOST_BACKEND=ray
export JAX_PLATFORMS=""
# Ray addresses the chips; in a pod the metadata server would describe the node
# rather than the workload.
export TPU_SKIP_MDS_QUERY=1

RAY_PORT=6379

# Is something listening on the head's Ray port? A worker asks this both before
# it has joined anything and after the head has gone, so `ray status` cannot
# answer it.
#
# Bounded, because the probe has to be able to fail and /dev/tcp has no deadline
# of its own. A completed pod stays a ready endpoint of the headless service, so
# this keeps resolving after the head exits - to an address that no longer
# answers and drops rather than refuses. An unbounded connect hangs there, the
# miss is never counted, and the worker waits out the deadline holding chips.
head_listening() {
  timeout 5 bash -c "exec 3<>/dev/tcp/${HEAD_HOST}/${RAY_PORT}" 2>/dev/null
}

# How many Ray nodes the head can see, or 0 if it could not ask.
#
# The count comes back through a file: Ray's C++ logging floods stdout the
# moment a driver attaches, and the digit is lost in it.
#
# A node counts only once it advertises chips. A raylet registers as alive
# before libtpu has handed it its chips, so counting registrations lets the head
# start vLLM against a cluster with 0.0/16.0 TPU - which fails an hour later in
# a placement timeout that looks nothing like its cause.
alive_nodes() {
  local report="${TMPDIR:-/tmp}/ray_alive.$$"
  rm -f "$report"
  python3 -c "
import ray
ray.init(address='auto', logging_level='error')
n = sum(1 for n in ray.nodes()
        if n['Alive'] and n.get('Resources', {}).get('TPU', 0) > 0)
with open('$report', 'w') as f:
    f.write(str(n))
" >/dev/null 2>"$report.err"
  local alive
  alive=$(cat "$report" 2>/dev/null)
  rm -f "$report"
  echo "${alive:-0}"
}

# Why alive_nodes could not answer. The head is always its own TPU-bearing node,
# so a zero means the query failed, not that nobody has joined.
report_last_error() {
  local err="${TMPDIR:-/tmp}/ray_alive.$$.err"
  if [ -s "$err" ]; then
    echo "last ray query error:"
    tail -20 "$err"
  fi
  rm -f "$err"
}

publish_results() {
  [ -n "${ARTIFACTS_DIR:-}" ] || return 0
  [ -n "$(ls -A "$ARTIFACTS_DIR" 2>/dev/null)" ] || return 0
  # Not `|| true`: these artifacts are this lane's only output, so a lost upload
  # has to fail the step.
  if ! buildkite-agent artifact upload "$ARTIFACTS_DIR/**/*"; then
    echo "ERROR: artifacts were produced but could not be uploaded"
    return 1
  fi
}

run_worker() {
  # `ray start --address` fails outright against a head that has not come up.
  local deadline=$((SECONDS + 1800))
  until head_listening; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "ERROR: no Ray head at ${HEAD_HOST}:${RAY_PORT} after 30 minutes"
      return 1
    fi
    sleep 10
  done

  # Not --block: nothing kills these pods, so a blocking worker would hold its
  # chips until the JobSet deadline and leave the step waiting on a JobSet that
  # can never complete.
  ray start --address="${HEAD_HOST}:${RAY_PORT}" || return 1
  echo "worker ${INDEX} joined ${HEAD_HOST}; waiting for the head to finish"
  # Three consecutive refusals, not one: a DNS blip on the headless service
  # looks identical to a closed port from here, and leaving is irreversible.
  local misses=0
  while [ "$misses" -lt 3 ]; do
    if head_listening; then
      misses=0
    else
      misses=$((misses + 1))
    fi
    sleep 15
  done
  echo "worker ${INDEX}: head is gone; exiting"
  ray stop --force >/dev/null 2>&1 || true
}

run_head() {
  ray start --head --port="${RAY_PORT}" || return 1

  echo "--- Waiting for ${NUM_HOSTS} Ray nodes to be alive"
  local deadline=$((SECONDS + 1800)) alive reported=-1
  while :; do
    alive=$(alive_nodes)
    if [ "$alive" -ge "$NUM_HOSTS" ]; then
      echo "Ray cluster complete: ${alive} nodes."
      break
    fi
    if [ "$alive" != "$reported" ]; then
      echo "  ${alive}/${NUM_HOSTS} nodes alive"
      reported="$alive"
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "ERROR: Ray cluster incomplete after 30 minutes (alive=${alive}/${NUM_HOSTS})"
      report_last_error
      ray status || true
      return 1
    fi
    sleep 15
  done
  ray status || true

  echo "--- Running on the head: $*"
  "$@"
  local rc=$?

  return "$rc"
}

if [ "$INDEX" = "0" ]; then
  run_head "$@"
  rc=$?
  # A failed upload decides the step only when the work itself passed; a run
  # that already failed keeps its own exit code.
  if ! publish_results && [ "$rc" -eq 0 ]; then
    rc=1
  fi
else
  run_worker
  rc=$?
fi
exit "$rc"
