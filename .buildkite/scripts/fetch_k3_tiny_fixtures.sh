#!/bin/bash
# Copyright 2025 Google LLC
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

# Fetches the Kimi-K3 test checkpoints into the persistent HF cache and exports
# the paths the Kimi test suites read.
#
# SOURCE this script, do not execute it -- it only has an effect through the
# variables it exports:
#
#   source .buildkite/scripts/fetch_k3_tiny_fixtures.sh
#   .buildkite/scripts/run_in_docker.sh bash -c "python3 -m pytest ..."
#
# The checkpoints hold random weights on the Kimi-K3 text-stack architecture
# plus goldens captured from the HF reference implementation; they are ~1.4 GiB
# in total, which is why they live in GCS rather than in the repo. Every suite
# that reads them skips when its variable is empty or points nowhere, so a
# failed or skipped fetch degrades to the same coverage as before this script
# existed rather than to a red build.
#
# Kimi-K3 needs a TPU generation with the MLA and KDA kernels, so the fetch is
# a no-op anywhere except tpu7x and the suites skip there.

K3_FIXTURES_GCS="${K3_FIXTURES_GCS:-gs://tpu-commons-ci/moonshootai/kimi/k3-tiny-fixtures}"
# run_in_docker.sh mounts /mnt/disks/persist/models at /tmp/hf_home, so the
# same directory has two names; the exported paths must be the container's.
K3_FIXTURES_HOST_DIR="${K3_FIXTURES_HOST_DIR:-/mnt/disks/persist/models/k3-tiny-fixtures}"
K3_FIXTURES_DOCKER_DIR="${K3_FIXTURES_DOCKER_DIR:-/tmp/hf_home/k3-tiny-fixtures}"

if [[ "${TPU_VERSION:-tpu6e}" != "tpu7x" ]]; then
  echo "[k3-fixtures] TPU_VERSION=${TPU_VERSION:-tpu6e}, skipping fetch"
elif ! mkdir -p "${K3_FIXTURES_HOST_DIR}" ||
  ! gcloud storage rsync -r "${K3_FIXTURES_GCS}" "${K3_FIXTURES_HOST_DIR}"; then
  echo "[k3-fixtures] fetch FAILED from ${K3_FIXTURES_GCS}; the Kimi suites" \
    "will skip. Check bucket permissions for this agent."
else
  export K3_TINY_CKPT="${K3_FIXTURES_DOCKER_DIR}/k3-tiny"
  export K3_TINY_MXFP4_CKPT="${K3_FIXTURES_DOCKER_DIR}/k3-tiny-mxfp4"
  export K3_TINY_SOFTPLUS_CKPT="${K3_FIXTURES_DOCKER_DIR}/k3-tiny-softplus"
  export K3_KDA_GOLDENS="${K3_FIXTURES_DOCKER_DIR}/k3-tiny/kda_op_goldens.npz"
  echo "[k3-fixtures] ready: $(du -sh "${K3_FIXTURES_HOST_DIR}" | cut -f1) at" \
    "${K3_FIXTURES_DOCKER_DIR} (in container)"
fi
