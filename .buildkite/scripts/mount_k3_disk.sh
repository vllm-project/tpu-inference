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

# Idempotently mount the read-only checkpoint disk on every host of the slice
# and report its contents.
#
# The slice has a hyperdisk-ml attached READ_ONLY as google-persistent-disk-1
# on all hosts, carrying a staged model checkpoint. The mount does not survive
# a host maintenance reboot (no fstab entry), so any job that serves from the
# local path must ensure the mount first. Mounting read-only cannot modify the
# disk; -o noload is the fallback for a journal the read-only attachment
# cannot replay.

set -uo pipefail

DEVICE="/dev/disk/by-id/google-persistent-disk-1"
MOUNT_POINT="/mnt/disks/jiries-disk_data"

ensure_mount() {
  echo "########## $(hostname) ##########"
  if ! sudo -n true 2>/dev/null; then
    echo "ERROR: passwordless sudo unavailable for $(whoami); cannot mount"
    return 1
  fi
  if [[ ! -e "$DEVICE" ]]; then
    echo "ERROR: $DEVICE not present on this host"
    return 1
  fi
  sudo -n mkdir -p "$MOUNT_POINT"
  if mountpoint -q "$MOUNT_POINT"; then
    echo "already mounted: $MOUNT_POINT"
  elif sudo -n mount -o ro "$DEVICE" "$MOUNT_POINT" 2>/dev/null; then
    echo "mounted (ro): $DEVICE -> $MOUNT_POINT"
  elif sudo -n mount -o ro,noload "$DEVICE" "$MOUNT_POINT"; then
    echo "mounted (ro,noload): $DEVICE -> $MOUNT_POINT"
  else
    echo "ERROR: mount failed on $(hostname)"
    return 1
  fi
  echo "--- top level ---"
  # shellcheck disable=SC2012  # human-readable listing; exotic names not expected
  ls -la "$MOUNT_POINT" | head -20 | sed 's/^/  /'
  echo "--- model dirs (depth 3) ---"
  find "$MOUNT_POINT" -maxdepth 3 -type d 2>/dev/null | head -20 | sed 's/^/  /'
  echo "--- safetensors inventory ---"
  find "$MOUNT_POINT" -maxdepth 4 -name '*.safetensors' 2>/dev/null | wc -l | sed 's/^/  files: /'
  find "$MOUNT_POINT" -maxdepth 4 -name '*.safetensors' -printf '%s\n' 2>/dev/null \
    | awk '{s+=$1} END {printf "  bytes: %.1f GB\n", s/1e9}'
  echo "--- config files ---"
  find "$MOUNT_POINT" -maxdepth 4 \( -name 'config.json' -o -name '*.index.json' -o -name 'tokenizer*' \) 2>/dev/null | head -10 | sed 's/^/  /'
  echo "--- df ---"
  df -h "$MOUNT_POINT" | sed 's/^/  /'
}

ensure_mount

SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10)
[[ -f "$HOME/.ssh/id_rsa" ]] && SSH_OPTS+=(-i "$HOME/.ssh/id_rsa")

ZONE=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/zone" | awk -F/ '{print $NF}')
TPU_NAME=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/description")
IPS=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" --format='value(networkEndpoints[].ipAddress)' | tr ';,' '  ')

SELF=$(hostname -I | awk '{print $1}')
FAIL=0
for ip in $IPS; do
  [[ "$ip" == "$SELF" ]] && continue
  echo ""
  echo "================= peer $ip ================="
  # shellcheck disable=SC2029  # client-side expansion of the function body is intended
  ssh "${SSH_OPTS[@]}" "$ip" \
    "DEVICE='$DEVICE'; MOUNT_POINT='$MOUNT_POINT'; $(declare -f ensure_mount); ensure_mount" \
    2>&1 | grep -v "Warning: Permanently added" || FAIL=1
done
echo ""
if [[ $FAIL -ne 0 ]]; then
  echo "mount FAILED on at least one peer"
  exit 1
fi
echo "checkpoint disk mounted (read-only) on all hosts."
