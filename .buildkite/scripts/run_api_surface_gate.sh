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

# Runs the CPU-only vLLM API-surface checks inside the image build_docker just
# pushed.
#
# Deliberately not run_in_docker.sh: that wrapper wants a TPU host's persistent
# disk, HF model cache and GCS-backed JAX cache. This gate runs on the CPU
# builder queue precisely so it costs no accelerator capacity, so it does a
# plain `docker run` against the prebuilt image instead.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The image tag is keyed on TPU generation. Either generation carries the same
# Python API surface, so the cheaper tpu6e image is enough.
export TPU_VERSION="${TPU_VERSION:-tpu6e}"
export USE_PREBUILT_IMAGE=1

# setup_environment pulls the image, verifies it really contains the vLLM
# commit this build is testing, and exports EXPORTED_CI_CACHE_IMAGE.
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/setup_docker_env.sh"
setup_environment "vllm-tpu"

echo "--- :microscope: Checking the vLLM API surface tpu_inference depends on"
docker run --rm \
  -e JAX_PLATFORMS=cpu \
  "${EXPORTED_CI_CACHE_IMAGE}" \
  python3 -m pytest -q -p no:cacheprovider \
    /workspace/tpu_inference/tests/compat/test_vllm_api_surface.py
