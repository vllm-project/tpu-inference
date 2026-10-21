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

# Check that the two wheels this build made install together from PyPI alone,
# as a user's `pip install vllm-tpu==<version>` would after the publish:
# resolve the combined wheel against the local core wheel plus pypi.org, for
# the platform the wheels support (CPython 3.12, x86_64 Linux, glibc >= 2.31).
set -euo pipefail
# shellcheck source=/dev/null
source "${RELEASE_TOOLS}/release/lib.sh"

VERSION="$(release_version)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

echo "--- :inbox_tray: Wheels built in this build"
(cd "${WORK}" && buildkite-agent artifact download "dist/*" .)
ls -l "${WORK}/dist"

echo "--- Installing uv"
curl -LsSf --retry 3 https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR="${WORK}/uv" UV_NO_MODIFY_PATH=1 sh >/dev/null
UV="${WORK}/uv/uv"

echo "--- :mag: Resolving vllm-tpu==${VERSION} from these wheels and pypi.org"
echo "vllm-tpu==${VERSION}" > "${WORK}/req.in"
"${UV}" pip compile "${WORK}/req.in" --no-config --quiet \
    --python-version 3.12 --python-platform x86_64-manylinux_2_31 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --find-links "${WORK}/dist" --index-url https://pypi.org/simple \
    -o "${WORK}/resolved.txt"

grep -E "^(vllm-tpu|tpu-inference|torch|torchvision|jax|jaxlib|libtpu|tokamax|qwix|flax|transformers)==" \
    "${WORK}/resolved.txt" | tee "${WORK}/pins.txt"
grep -qx "tpu-inference==${VERSION}" "${WORK}/pins.txt" \
    || die "the combined wheel does not resolve tpu-inference==${VERSION}."

NAMES=""
for f in "${WORK}"/dist/*; do NAMES+="\`$(basename "${f}")\` "; done
buildkite-agent annotate --style info --context release-wheels <<EOF
Wheels: ${NAMES}\`vllm-tpu==${VERSION}\` resolves from them plus pypi.org ($(grep -c '==' "${WORK}/resolved.txt") packages):

\`\`\`
$(cat "${WORK}/pins.txt")
\`\`\`
EOF
