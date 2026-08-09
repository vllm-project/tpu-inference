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

# Read-only disk probe for a multi-host TPU slice.
#
# Reports, for this host and every peer: block devices, filesystems, mounts,
# fstab, and whether a locally staged model checkpoint directory exists.
# Nothing is mounted, written, or signalled.

set -uo pipefail

probe_one() {
  echo "########## $(hostname) ##########"
  echo "--- lsblk ---"
  lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL,SERIAL 2>/dev/null || lsblk
  echo "--- disk by-id ---"
  for d in /dev/disk/by-id/*; do
    case "$d" in *-part*) continue;; esac
    printf '  %s -> %s\n' "$d" "$(readlink -f "$d")"
  done
  echo "--- fstab ---"
  grep -v '^#' /etc/fstab 2>/dev/null | sed 's/^/  /'
  echo "--- real mounts ---"
  mount | grep -vE 'cgroup|proc |sysfs|tmpfs|overlay|devpts|mqueue|debugfs|tracefs|fusectl|configfs|bpf|pstore|securityfs|hugetlbfs|nsfs|binfmt' | sed 's/^/  /'
  echo "--- df -h (all) ---"
  df -h | sed 's/^/  /'
  echo "--- candidate checkpoint paths ---"
  for p in /root/.cache/huggingface /mnt/disks /mnt; do
    echo "  [$p]"
    sudo -n ls -la "$p" 2>/dev/null | head -15 | sed 's/^/    /' || echo "    (unreadable or absent)"
  done
  echo "  [find Kimi-K3 dirs, depth 4]"
  sudo -n find /root/.cache /mnt -maxdepth 4 -iname '*kimi*' 2>/dev/null | head -10 | sed 's/^/    /'
}

probe_one

SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10)
[[ -f "$HOME/.ssh/id_rsa" ]] && SSH_OPTS+=(-i "$HOME/.ssh/id_rsa")

ZONE=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/zone" | awk -F/ '{print $NF}')
TPU_NAME=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/description")
IPS=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" --format='value(networkEndpoints[].ipAddress)' | tr ';,' '  ')

SELF=$(hostname -I | awk '{print $1}')
for ip in $IPS; do
  [[ "$ip" == "$SELF" ]] && continue
  echo ""
  echo "================= peer $ip ================="
  # Ship this function to the peer and run it there.
  # shellcheck disable=SC2029  # client-side expansion of the function body is intended
  ssh "${SSH_OPTS[@]}" "$ip" "$(declare -f probe_one); probe_one" 2>&1 | grep -v "Warning: Permanently added"
done
echo ""
echo "disk probe complete (read-only)."
