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
# DEV ONLY, READ-ONLY: is anybody using this multi-host TPU slice? Reports
# logged-in users, TPU-device holders, serving/ray processes and containers
# on the head node and every worker. Buildkite queue state only shows
# Buildkite work; a human on the box directly is invisible to it.
set -u

report_node() {
  echo "--- logged-in users ---"
  who 2>/dev/null || true
  echo "--- processes holding a TPU device ---"
  for d in /dev/accel0 /dev/vfio/0; do
    if [ -e "$d" ]; then sudo fuser -v "$d" 2>&1 | head -8 || true; fi
  done
  echo "--- serving / ray / engine processes ---"
  pgrep -af 'vllm|EngineCore|ray::|jax' 2>/dev/null | grep -v pgrep | head -10 || echo "(none)"
  echo "--- containers ---"
  docker ps --format '{{.Names}} | {{.Status}} | {{.Image}}' 2>/dev/null | head -8 || echo "(none)"
  echo "--- load ---"
  uptime 2>/dev/null || true
}

echo "################ HEAD $(hostname) ################"
report_node

SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o IPQoS=none -i "$HOME/.ssh/id_rsa")
TPU_NAME="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/description 2>/dev/null || echo '')"
ZONE="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null | awk -F/ '{print $NF}')"
ALL_IPS="$(gcloud compute tpus tpu-vm describe "${TPU_NAME}" --zone "${ZONE}" --format='value(networkEndpoints[].ipAddress)' 2>/dev/null | tr ';' '\n' | tr ',' '\n')"
SELF="$(hostname -I | awk '{print $1}')"
for ip in ${ALL_IPS}; do
  if [ "${ip}" = "${SELF}" ]; then continue; fi
  echo "################ WORKER ${ip} ################"
  # shellcheck disable=SC2029
  ssh "${SSH_OPTS[@]}" "buildkite-agent@${ip}" "$(declare -f report_node); report_node" \
    || echo "[occupancy] ssh to ${ip} FAILED"
done
echo "[occupancy] done"
