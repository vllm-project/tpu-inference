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
# DEV ONLY, READ-ONLY: report disk usage on the multi-host pod's head node
# and every worker. Written after a pod run died with "no space left on
# device" during the head docker build while `docker system prune -a
# --volumes -f` reclaimed 0 B -- i.e. the consumer is not docker state.
# Measures only; deletions are a separate, targeted step.
set -u

report_node() {
  echo "===== df -h / ====="
  df -h / 2>/dev/null || true
  echo "===== top-level usage (depth 1) ====="
  sudo du -xh --max-depth=1 / 2>/dev/null | sort -rh | head -15 || true
  echo "===== usual suspects ====="
  for p in /var/lib/docker /root/.cache /root/.cache/vllm /tmp /tmp/hf_home \
           /var/log /var/lib/buildkite-agent /home/buildkite-agent; do
    if [ -e "$p" ]; then sudo du -sh "$p" 2>/dev/null || true; fi
  done
  echo "===== vllm streamer assets (per entry) ====="
  sudo du -sh /root/.cache/vllm/assets/model_streamer/* 2>/dev/null | sort -rh | head -10 || true
  echo "===== buildkite build dirs ====="
  sudo du -sh /var/lib/buildkite-agent/builds/* 2>/dev/null | sort -rh | head -10 || true
  echo "===== stray big files (>2G, outside docker) ====="
  sudo find / -xdev -type f -size +2G -not -path '/var/lib/docker/*' \
    -printf '%s\t%p\n' 2>/dev/null | sort -rn | head -15 || true
}

echo "################ HEAD NODE $(hostname) ################"
report_node

SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o IPQoS=none -i "$HOME/.ssh/id_rsa")
TPU_NAME="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/description 2>/dev/null || echo '')"
ZONE="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null | awk -F/ '{print $NF}')"
echo "[triage] TPU_NAME=${TPU_NAME} ZONE=${ZONE}"
ALL_IPS="$(gcloud compute tpus tpu-vm describe "${TPU_NAME}" --zone "${ZONE}" --format='value(networkEndpoints[].ipAddress)' 2>/dev/null | tr ';' '\n' | tr ',' '\n')"
SELF="$(hostname -I | awk '{print $1}')"
for ip in ${ALL_IPS}; do
  if [ "${ip}" = "${SELF}" ]; then continue; fi
  echo "################ WORKER ${ip} ################"
  # shellcheck disable=SC2029
  ssh "${SSH_OPTS[@]}" "buildkite-agent@${ip}" "$(declare -f report_node); report_node" \
    || echo "[triage] ssh to ${ip} FAILED"
done
echo "[triage] done"
