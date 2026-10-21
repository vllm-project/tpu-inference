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
# ------------------------------------------------------
# Build the combined `vllm-tpu` wheel and sdist from upstream vLLM.
#
# The result is vLLM renamed to `vllm-tpu`, with a pip dependency on
# `tpu-inference` (this repo). The core is not bundled; pip installs it.
#
# Steps (all on a temp clone of vLLM, deleted on exit; nothing is committed):
#   1. Shallow-clone vLLM at the ref pinned in this repo's pyproject.toml
#      (`vllm @ git+...@<ref>`), the same ref CI tests against.
#   2. Rename the package "vllm" -> "vllm-tpu" in pyproject.toml and in
#      vLLM's own importlib.metadata lookups (else they fail at runtime).
#   3. Add tpu-inference==<version> to requirements/common.txt, which is
#      where setup.py reads dependencies from when VLLM_TARGET_DEVICE=empty.
#   4. Build the wheel and sdist into OUTPUT_DIR.
#   5. Retag the wheel linux_<arch> -> manylinux_2_28_x86_64, since PyPI
#      rejects linux_<arch>. Refused if the wheel contains native binaries.
#   6. `twine check --strict` the wheel and sdist (metadata PyPI would reject).
#
# Usage:
#   ./build_vllm_tpu.sh <version>
#
#   <version>    tpu-inference version to pin; also used as the vllm-tpu version.
# ------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Provides get_vllm_commit_hash (the vLLM pin in pyproject.toml).
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/configs/pipeline_config.sh"

VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/dist}"
KEEP_VLLM_SRC="${KEEP_VLLM_SRC:-0}"
PYTHON="${PYTHON:-python3}"

# Linux x86_64 with glibc >= 2.28, the same baseline as upstream vLLM.
WHEEL_PLATFORM_TAG="manylinux_2_28_x86_64"

# vLLM files that look up their own package name via importlib.metadata.
METADATA_FILES=(
    "vllm/entrypoints/cli/main.py"
    "vllm/entrypoints/cli/run_batch.py"
    "vllm/utils/__init__.py"
    "vllm/platforms/__init__.py"
    "vllm/triton_utils/importing.py"
    "vllm/utils/import_utils.py"
    "vllm/snapshot/runtime.py"
)

usage() {
    echo "Usage: $0 <version>"
    echo ""
    echo "  <version>   Release version for both packages. The vllm-tpu wheel"
    echo "              gets this version and pins tpu-inference==<version>."
    echo ""
    echo "vLLM is built at the ref pinned in pyproject.toml (tag or full commit SHA)"
    echo "(VLLM_REF_OVERRIDE=<ref> or VLLM_COMMIT_HASH=<ref> for dev builds only)."
    echo ""
    echo "Examples: $0 X.Y.Z        # official release"
    echo "          $0 X.Y.ZrcN     # pre-release"
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "$2"
}

# --- Arguments ---------------------------------------------------------------
CORE_VERSION="${1:-${RELEASE_VERSION:-}}"
CORE_VERSION="${CORE_VERSION#v}"
if [[ -z "${CORE_VERSION}" ]]; then
    usage
    exit 1
fi
WHEEL_VERSION="${CORE_VERSION}"

# The vLLM ref CI tested: the pin in pyproject.toml, unless overridden.
PINNED_VLLM_REF="$(get_vllm_commit_hash)" \
    || die "could not read the vLLM pin from ${REPO_ROOT}/pyproject.toml."
VLLM_REF="${VLLM_REF_OVERRIDE:-${VLLM_COMMIT_HASH:-${PINNED_VLLM_REF}}}"
if [[ "${VLLM_REF}" != "${PINNED_VLLM_REF}" ]]; then
    echo "WARNING: building vLLM ${VLLM_REF} (override), not the" \
         "pyproject.toml pin ${PINNED_VLLM_REF}. Do not publish this wheel." >&2
fi

# Make it absolute; the build runs from inside the temp clone.
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

# --- Preflight checks (fail fast, before the long build) ---------------------
[[ "$(uname -s)" == "Linux" ]] \
    || die "must run on Linux (found $(uname -s)); vLLM forces a CPU build" \
           "on macOS. Use a Linux machine, TPU VM or container."
require_cmd git "git not found (apt-get install git)."
require_cmd curl "curl not found (apt-get install curl)."
require_cmd "${PYTHON}" "${PYTHON} not found; set PYTHON=<interpreter>."
"${PYTHON}" -c 'import sys; sys.exit(not ((3, 11) <= sys.version_info < (3, 14)))' \
    || die "Python >=3.11,<3.14 required," \
           "found $("${PYTHON}" -V 2>&1)."
for tool in cargo rustc; do
    if command -v "$tool" >/dev/null 2>&1; then
        die "${tool} found at $(command -v "$tool"). With Rust on PATH vLLM" \
            "builds native extensions and the wheel cannot be retagged" \
            "${WHEEL_PLATFORM_TAG}. Run without Rust on PATH."
    fi
done

# --- Temp workspace and cleanup ----------------------------------------------
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vllm-tpu-build.XXXXXX")"
VLLM_SRC_DIR="${WORK_DIR}/vllm"
cleanup() {
    if [[ "${KEEP_VLLM_SRC}" == "1" ]]; then
        echo "KEEP_VLLM_SRC=1: patched vLLM source kept at ${VLLM_SRC_DIR}"
    else
        rm -rf "${WORK_DIR}"
    fi
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 131' QUIT
trap 'exit 141' PIPE
trap 'exit 143' TERM

# Private build venv. Created without pip (CI agents lack python3-venv), then
# pip is installed from get-pip.py.
echo "Creating build venv with $("${PYTHON}" -V 2>&1)..."
"${PYTHON}" -m venv --without-pip "${WORK_DIR}/venv" \
    || die "could not create a venv with ${PYTHON}."
PY="${WORK_DIR}/venv/bin/python"
curl -fsSL --retry 3 -o "${WORK_DIR}/get-pip.py" \
    https://bootstrap.pypa.io/get-pip.py \
    || die "could not download get-pip.py from bootstrap.pypa.io."
"$PY" "${WORK_DIR}/get-pip.py" --quiet

echo "--- Starting vllm-tpu wheel build ---"
echo "tpu-inference version  : ${CORE_VERSION}"
echo "vllm-tpu version       : ${WHEEL_VERSION}"
echo "vLLM repo              : ${VLLM_REPO}"
echo "vLLM ref               : ${VLLM_REF}"
echo "Output dir             : ${OUTPUT_DIR}"

# --- Step 1: Shallow-clone vLLM ----------------------------------------------
echo "Fetching ${VLLM_REF} from ${VLLM_REPO} (shallow)..."
git init --quiet "${VLLM_SRC_DIR}"
cd "${VLLM_SRC_DIR}"
git remote add origin "${VLLM_REPO}"
git fetch --quiet --depth 1 origin "${VLLM_REF}" \
    || die "failed to fetch '${VLLM_REF}' from ${VLLM_REPO}" \
           "(short SHAs are not supported; use a branch, tag or full SHA)."
git checkout --quiet --detach FETCH_HEAD
echo "vLLM commit            : $(git rev-parse HEAD)"

PYPROJECT_FILE="pyproject.toml"
REQUIREMENTS_FILE="requirements/common.txt"
for f in "$PYPROJECT_FILE" "$REQUIREMENTS_FILE"; do
    [[ -f "$f" ]] || die "${f} not found in vLLM ref '${VLLM_REF}'."
done

# --- Step 2: Rename the package to vllm-tpu ----------------------------------
echo "Renaming distribution in ${PYPROJECT_FILE} to vllm-tpu..."
awk '!done && /^name = "vllm"[[:space:]]*$/ { print "name = \"vllm-tpu\""; done = 1; next } { print }' \
    "$PYPROJECT_FILE" > "${PYPROJECT_FILE}.tmp"
mv "${PYPROJECT_FILE}.tmp" "$PYPROJECT_FILE"
grep -q '^name = "vllm-tpu"' "$PYPROJECT_FILE" \
    || die "could not find 'name = \"vllm\"' in ${PYPROJECT_FILE}."

echo "Repointing importlib.metadata lookups of 'vllm' to 'vllm-tpu'..."
for f in "${METADATA_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "WARNING: ${f} not found in this vLLM ref; skipping." >&2
        continue
    fi
    sed -i.bak \
        -e "s/importlib.metadata.version(\(['\"]\)vllm\1)/importlib.metadata.version(\1vllm-tpu\1)/g" \
        -e "s/importlib.metadata.metadata(\(['\"]\)vllm\1)/importlib.metadata.metadata(\1vllm-tpu\1)/g" \
        -e "s/version(\(['\"]\)vllm\1)/version(\1vllm-tpu\1)/g" \
        "$f"
    rm -f "${f}.bak"
done

REMAINING="$(grep -rnE "(metadata|version)\((['\"])vllm\2\)" vllm --include='*.py' || true)"
if [[ -n "$REMAINING" ]]; then
    echo "ERROR: unpatched importlib.metadata lookups of 'vllm' remain:" >&2
    echo "$REMAINING" >&2
    echo "         Add these files to METADATA_FILES in $0." >&2
    exit 1
fi

# --- Step 3: Pin tpu-inference -----------------------------------------------
REQUIRED_LINE="tpu-inference==${CORE_VERSION}"
echo "Adding '${REQUIRED_LINE}' to ${REQUIREMENTS_FILE}..."
if [[ -n "$(tail -c 1 "$REQUIREMENTS_FILE")" ]]; then
    echo "" >> "$REQUIREMENTS_FILE"
fi
echo "$REQUIRED_LINE" >> "$REQUIREMENTS_FILE"

# --- Step 4: Build -----------------------------------------------------------
echo "Installing build dependencies..."
"$PY" -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "cmake>=3.26.1" \
    ninja \
    "packaging>=24.2" \
    "jinja2>=3.1.6" \
    "setuptools>=77.0.3,<81.0.0" \
    "setuptools-scm>=8" \
    "setuptools-rust>=1.9.0" \
    "torch==2.10.0+cpu" \
    build wheel twine

# Remove old wrapper files only; `vllm_tpu-` never matches core files
# (`tpu_inference-`).
rm -f "${OUTPUT_DIR}"/vllm_tpu-*.whl "${OUTPUT_DIR}"/vllm_tpu-*.tar.gz

echo "Building vllm-tpu sdist and wheel into ${OUTPUT_DIR}..."
VLLM_TARGET_DEVICE=empty \
VLLM_VERSION_OVERRIDE="${WHEEL_VERSION}" \
SETUPTOOLS_SCM_PRETEND_VERSION="${WHEEL_VERSION}" \
    "$PY" -m build --sdist --wheel --no-isolation --skip-dependency-check --outdir "${OUTPUT_DIR}"

shopt -s nullglob
BUILT_SDISTS=("${OUTPUT_DIR}"/vllm_tpu-*.tar.gz)
BUILT_WHEELS=("${OUTPUT_DIR}"/vllm_tpu-*.whl)
shopt -u nullglob
[[ "${#BUILT_SDISTS[@]}" -eq 1 ]] \
    || die "expected one vllm_tpu-*.tar.gz in ${OUTPUT_DIR}," \
           "found ${#BUILT_SDISTS[@]}."
[[ "${#BUILT_WHEELS[@]}" -eq 1 ]] \
    || die "expected one vllm_tpu-*.whl in ${OUTPUT_DIR}," \
           "found ${#BUILT_WHEELS[@]}."
BUILT_SDIST="${BUILT_SDISTS[0]}"
BUILT_WHEEL="${BUILT_WHEELS[0]}"

# --- Step 5: Retag the wheel as manylinux ------------------------------------
BINARIES="$("$PY" - "$BUILT_WHEEL" <<'EOF'
import sys, zipfile
names = zipfile.ZipFile(sys.argv[1]).namelist()
print("\n".join(n for n in names
                if n.endswith((".so", ".pyd", ".dylib")) or n.endswith("/vllm-rs")))
EOF
)"
if [[ -n "$BINARIES" ]]; then
    rm -f "$BUILT_WHEEL"
    echo "ERROR: the built wheel contains native binaries:" >&2
    echo "$BINARIES" >&2
    echo "       Refusing to retag it as ${WHEEL_PLATFORM_TAG} (wheel removed)." >&2
    echo "       Build without a Rust toolchain on PATH." >&2
    exit 1
fi

case "$(basename "$BUILT_WHEEL")" in
    *-"${WHEEL_PLATFORM_TAG}".whl)
        echo "Wheel is already ${WHEEL_PLATFORM_TAG}."
        FINAL_WHEEL="$BUILT_WHEEL" ;;
    *)
        echo "Retagging $(basename "$BUILT_WHEEL") to ${WHEEL_PLATFORM_TAG}..."
        NEW_NAME="$("$PY" -m wheel tags --remove \
            --platform-tag "${WHEEL_PLATFORM_TAG}" "$BUILT_WHEEL")"
        FINAL_WHEEL="${OUTPUT_DIR}/${NEW_NAME}" ;;
esac
[[ -f "$FINAL_WHEEL" ]] || die "retagged wheel not found at ${FINAL_WHEEL}."

# --- Step 6: Check metadata --------------------------------------------------
echo "--- Checking metadata ---"
"$PY" -m twine check --strict "$FINAL_WHEEL" "$BUILT_SDIST"

echo "--- Build complete! ---"
ls -lh "$FINAL_WHEEL" "$BUILT_SDIST"
echo "Wheel: ${FINAL_WHEEL}"
echo "Sdist: ${BUILT_SDIST}"
