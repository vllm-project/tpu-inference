#!/bin/bash
# Download and verify the MLPerf OpenOrca dataset into a caller-owned cache.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <destination.pkl> <expected-sha256>" >&2
  exit 2
fi

destination=$1
expected_sha=$2
compressed="${destination}.gz"

mkdir -p "$(dirname "$destination")"

if [[ -f "$destination" ]] && echo "$expected_sha  $destination" | sha256sum -c - >/dev/null; then
  echo "Using verified MLPerf dataset at $destination"
  exit 0
fi

rm -f "$destination" "$compressed"
if ! command -v rclone >/dev/null; then
  echo "rclone is required to prepare the MLPerf dataset" >&2
  exit 1
fi

rclone config create mlc-inference s3 provider=Cloudflare \
  access_key_id=f65ba5eef400db161ea49967de89f47b \
  secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b \
  endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com \
  >/dev/null
rclone copy \
  mlc-inference:mlcommons-inference-wg-public/open_orca \
  "$(dirname "$destination")" \
  --no-traverse

if [[ -f "$compressed" ]]; then
  gzip -d "$compressed"
fi
if [[ ! -f "$destination" ]] || ! echo "$expected_sha  $destination" | sha256sum -c -; then
  echo "CRITICAL SECURITY ERROR: MLPerf dataset checksum mismatch" >&2
  rm -f "$destination" "$compressed"
  exit 1
fi
echo "Prepared and verified MLPerf dataset at $destination"
