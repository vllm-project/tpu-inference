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
# Build the combined `vllm-tpu` release wheel and sdist from upstream vLLM.
#
# The result is vLLM renamed to `vllm-tpu`, built for its in-tree TPU target,
# with a pip dependency on `tpu-inference` (this repo). tpu-inference is not
# bundled; pip installs it from PyPI.
#
# Only vLLM is cloned: this script runs from the tpu-inference checkout, and
# the combined wheel needs just the tpu-inference version, not its source.
#
# Steps (all on a temp clone of vLLM, deleted on exit; nothing is committed):
#   1. Shallow-clone vLLM at the ref in .buildkite/vllm_lkg.version, the
#      same ref CI tests tpu-inference against.
#   2. Rename the package "vllm" -> "vllm-tpu" in pyproject.toml and in
#      vLLM's own importlib.metadata lookups (else they fail at runtime).
#   3. Pin tpu-inference==<version> in requirements/tpu.txt, which is where
#      setup.py reads dependencies from when VLLM_TARGET_DEVICE=tpu. vLLM
#      already lists a tpu-inference version there; it is replaced.
#   4. Build the wheel and sdist into OUTPUT_DIR, with the build deps in this
#      repo's requirements/build/tpu.txt (not the vLLM clone's).
#   5. Retag the wheel linux_<arch> -> manylinux_<glibc>_<arch>, since PyPI
#      rejects linux_<arch>. Done by vLLM's own
#      .buildkite/scripts/detect-manylinux-tag.py from the clone, which asks
#      auditwheel which glibc level the wheel's binaries need.
#   6. `twine check --strict` the wheel and sdist (metadata PyPI would reject).
#
# This does the same rename as vLLM's tools/vllm-tpu/build.sh but does not
# call it: that script patches a fixed list of files, so lookups added
# upstream go unpatched, and it installs build deps into the caller's Python.
#
# Usage:
#   ./build_vllm_tpu_wheel.sh <version>
#
#   <version>    tpu-inference version to pin, built in the same release (the
#                release pipeline checks the two resolve together before
#                publishing either); also used as the vllm-tpu version.
#
# Examples:
#   ./build_vllm_tpu_wheel.sh 0.31.0
#   ./build_vllm_tpu_wheel.sh 0.31.0rc1
#
# Requirements (checked up front):
#   - Linux (vLLM forces a CPU build on macOS). No TPU needed.
#   - git, curl, Python 3.12, and network access to github.com, PyPI,
#     bootstrap.pypa.io and download.pytorch.org.
#   - The built wheel must contain native binaries (ELF files): auditwheel
#     derives the manylinux tag from them and cannot tag a pure-Python
#     wheel. The script stops with a clear error if there are none.
#   Build dependencies go into a private temp venv, so the caller's packages
#   (e.g. TPU jax on a TPU VM) are never touched.
#
# Building a wheel from the sdist later needs the same VLLM_TARGET_DEVICE=tpu
# and VLLM_VERSION_OVERRIDE as below; otherwise vLLM builds its CUDA target.
#
# Environment:
#   VLLM_REPO              vLLM git remote (default: upstream vllm-project/vllm).
#   VLLM_REF_OVERRIDE      Dev only: build this vLLM branch, tag or full SHA
#                          instead of vllm_lkg.version. Never set it for a
#                          release; the result was not tested with this
#                          tpu-inference.
#   OUTPUT_DIR             Output directory (default: <repo-root>/dist). Old
#                          vllm_tpu-* files are removed; tpu_inference-* files
#                          are kept.
#   KEEP_VLLM_SRC          Set to 1 to keep the patched clone for debugging.
#   PYTHON                 Interpreter for the build venv (default: python3).
#
# Output:
#   ${OUTPUT_DIR}/vllm_tpu-<version>-cpXY-cpXY-manylinux_<glibc>_x86_64.whl
#   ${OUTPUT_DIR}/vllm_tpu-<version>.tar.gz
#   (cpXY = the Python that ran the build, e.g. cp312; <glibc> as detected,
#   e.g. 2_17)
# ------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WHEEL_UTILS="${SCRIPT_DIR}/wheel_utils.py"
# Provides get_vllm_commit_hash (the vLLM commit in .buildkite/vllm_lkg.version).
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/configs/pipeline_config.sh"
# Provides die and is_valid_release_version.
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../release/lib.sh"

VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/dist}"
KEEP_VLLM_SRC="${KEEP_VLLM_SRC:-0}"
PYTHON="${PYTHON:-python3}"

# auditwheel for detect-manylinux-tag.py. That script uses auditwheel
# internals, so pin the version vLLM's .buildkite/scripts/lib/manylinux.sh
# uses (also on Python 3.12).
AUDITWHEEL_VERSION="6.6.0"

# Use own build deps under requirements/build/tpu.txt, not the vLLM clone's.
BUILD_REQUIREMENTS_FILE="${REPO_ROOT}/requirements/build/tpu.txt"

# vLLM files that look up their own package name via importlib.metadata.
# Files missing at a given ref are skipped; lookups in files not listed here
# fail the build (see Step 2).
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
    echo "  <version>   The tpu-inference version of this release. Used to"
    echo "              pin tpu-inference==<version> and as vllm-tpu's own"
    echo "              version, e.g. 0.31.0rc1 -> both are 0.31.0rc1."
    echo ""
    echo "vLLM is built at the commit in .buildkite/vllm_lkg.version"
    echo "(VLLM_REF_OVERRIDE=<ref> for dev builds only)."
    echo ""
    echo "Examples: $0 0.31.0"
    echo "          $0 0.31.0rc1"
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "$2"
}

# --- Arguments ---------------------------------------------------------------
if [[ "$#" -ne 1 ]]; then
    usage
    exit 1
fi

CORE_VERSION=$1
[[ -n "${CORE_VERSION}" ]] || die "<version> must not be empty."
# The release pipeline passes the version preflight settled; this catches a
# mistyped one in a manual run before the long build.
is_valid_release_version "${CORE_VERSION}" \
    || die "'${CORE_VERSION}' is not a release version PyPI accepts" \
           "(X.Y.Z, optionally followed by aN, bN, rcN, .postN or .devN)."
WHEEL_VERSION="${CORE_VERSION}"

# The vLLM ref CI tested (vllm_lkg.version), unless overridden.
# get_vllm_commit_hash reads a path relative to the repo root, so run it there.
[[ -f "${REPO_ROOT}/.buildkite/vllm_lkg.version" ]] \
    || echo "WARNING: .buildkite/vllm_lkg.version not found;" \
            "get_vllm_commit_hash falls back to vLLM's current HEAD." >&2
PINNED_VLLM_REF="$(cd "${REPO_ROOT}" && get_vllm_commit_hash)" \
    || die "could not resolve the vLLM commit (get_vllm_commit_hash)."
[[ -n "${PINNED_VLLM_REF}" ]] || die "get_vllm_commit_hash returned nothing."
VLLM_REF="${VLLM_REF_OVERRIDE:-${PINNED_VLLM_REF}}"
if [[ "${VLLM_REF}" != "${PINNED_VLLM_REF}" ]]; then
    echo "WARNING: building vLLM ${VLLM_REF} (VLLM_REF_OVERRIDE), not the" \
         "vllm_lkg.version pin ${PINNED_VLLM_REF}. Do not publish this wheel." >&2
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
[[ -f "${BUILD_REQUIREMENTS_FILE}" ]] \
    || die "${BUILD_REQUIREMENTS_FILE} not found; it lists the build deps."
# 3.12 matches the CI/Docker images (python:3.12-slim-bookworm); the wheel is
# tagged cpXY for the Python that builds it.
"${PYTHON}" -c 'import sys; sys.exit(not ((3, 12) <= sys.version_info < (3, 13)))' \
    || die "Python 3.12 required (the wheel is tagged for the building" \
           "Python), found $("${PYTHON}" -V 2>&1)."

# --- Temp workspace and cleanup ----------------------------------------------
# Set up before cloning so the clone is always removed, even on failure.
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
# Stop on signals instead of continuing; the EXIT trap then cleans up once.
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
# wheel_utils.py needs packaging.
"$PY" -m pip install --quiet packaging

echo "--- Starting vllm-tpu wheel build ---"
echo "tpu-inference version  : ${CORE_VERSION}"
echo "vllm-tpu version       : ${WHEEL_VERSION}"
echo "vLLM repo              : ${VLLM_REPO}"
echo "vLLM ref               : ${VLLM_REF}"
echo "Output dir             : ${OUTPUT_DIR}"

# --- Step 1: Shallow-clone vLLM ----------------------------------------------
# init + fetch (not `clone --branch`) so full commit SHAs work too.
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
REQUIREMENTS_FILE="requirements/tpu.txt"
# vLLM's retag script, used in Step 5; checked now so a ref without it fails
# before the long build.
DETECT_MANYLINUX_SCRIPT=".buildkite/scripts/detect-manylinux-tag.py"
for f in "$PYPROJECT_FILE" "$REQUIREMENTS_FILE" "$DETECT_MANYLINUX_SCRIPT"; do
    [[ -f "$f" ]] || die "${f} not found in vLLM ref '${VLLM_REF}'."
done

# --- Step 2: Rename the package to vllm-tpu ----------------------------------
# Replace only the first `name = "vllm"` line (the [project] name).
# awk instead of sed -i so it works on both GNU and BSD.
echo "Renaming distribution in ${PYPROJECT_FILE} to vllm-tpu..."
awk '!done && /^name = "vllm"[[:space:]]*$/ { print "name = \"vllm-tpu\""; done = 1; next } { print }' \
    "$PYPROJECT_FILE" > "${PYPROJECT_FILE}.tmp"
mv "${PYPROJECT_FILE}.tmp" "$PYPROJECT_FILE"
grep -q '^name = "vllm-tpu"' "$PYPROJECT_FILE" \
    || die "could not find 'name = \"vllm\"' in ${PYPROJECT_FILE}."

# Rewrite version('vllm') / metadata('vllm') to use 'vllm-tpu'
# (both quote styles). `-i.bak` + rm works with GNU and BSD sed.
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

# Fail on lookups in files missing from METADATA_FILES (new upstream code):
# unpatched, they raise PackageNotFoundError at runtime.
REMAINING="$(grep -rnE "(metadata|version)\((['\"])vllm\2\)" vllm --include='*.py' || true)"
if [[ -n "$REMAINING" ]]; then
    echo "ERROR: unpatched importlib.metadata lookups of 'vllm' remain:" >&2
    echo "$REMAINING" >&2
    echo "         Add these files to METADATA_FILES in $0." >&2
    exit 1
fi

# --- Step 3: Pin tpu-inference -----------------------------------------------
# vLLM's tpu.txt already pins some tpu-inference version; replace every
# tpu-inference line with ours, or append it if there is none.
REQUIRED_LINE="tpu-inference==${CORE_VERSION}"
echo "Pinning '${REQUIRED_LINE}' in ${REQUIREMENTS_FILE}..."
awk -v line="$REQUIRED_LINE" '
    /^[[:space:]]*tpu[-_]inference([[:space:]]|[=<>!~@\[;]|$)/ {
        if (!done) { print line; done = 1 }
        next
    }
    { print }
    END { if (!done) print line }
' "$REQUIREMENTS_FILE" > "${REQUIREMENTS_FILE}.tmp"
mv "${REQUIREMENTS_FILE}.tmp" "$REQUIREMENTS_FILE"
PIN_LINES="$(grep -cE '^[[:space:]]*tpu[-_]inference' "$REQUIREMENTS_FILE" || true)"
if [[ "${PIN_LINES}" -ne 1 ]] || ! grep -qxF "$REQUIRED_LINE" "$REQUIREMENTS_FILE"; then
    die "expected exactly one '${REQUIRED_LINE}' line in ${REQUIREMENTS_FILE}."
fi

# --- Step 4: Build -----------------------------------------------------------
# setup.py imports torch, setuptools-rust and setuptools-scm, so they must be
# installed for a --no-isolation build. Our requirements/build/tpu.txt is just
# build tools + CPU torch; `build` fails if it no longer fits vLLM's [build-system].
echo "Installing build dependencies from ${BUILD_REQUIREMENTS_FILE}..."
"$PY" -m pip install -r "$BUILD_REQUIREMENTS_FILE" build wheel twine \
    "auditwheel==${AUDITWHEEL_VERSION}"

# Remove old vllm-tpu files only; `vllm_tpu-` never matches tpu_inference-*.
rm -f "${OUTPUT_DIR}"/vllm_tpu-*.whl "${OUTPUT_DIR}"/vllm_tpu-*.tar.gz

echo "Building vllm-tpu sdist and wheel into ${OUTPUT_DIR}..."
VLLM_TARGET_DEVICE=tpu \
VLLM_VERSION_OVERRIDE="${WHEEL_VERSION}" \
SETUPTOOLS_SCM_PRETEND_VERSION="${WHEEL_VERSION}" \
    "$PY" -m build --sdist --wheel --no-isolation --outdir "${OUTPUT_DIR}"

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
# detect-manylinux-tag.py renames the wheel in place (only the platform tag
# changes; cp312-cp312 stays), prints the new path on stdout and logs to
# stderr.
# auditwheel needs at least one ELF binary to derive a tag; check first so a
# pure-Python wheel fails with a clear message instead of an auditwheel error.
ELF_FILES="$("$PY" - "$BUILT_WHEEL" <<'EOF'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as zf:
    for name in zf.namelist():
        with zf.open(name) as f:
            if f.read(4) == b"\x7fELF":
                print(name)
EOF
)"
if [[ -z "$ELF_FILES" ]]; then
    die "$(basename "$BUILT_WHEEL") has no native binaries, so auditwheel" \
        "cannot derive a manylinux tag."
fi
echo "Native binaries in the wheel:"
printf '  %s\n' "${ELF_FILES//$'\n'/$'\n'  }"

echo "Detecting the manylinux tag with ${DETECT_MANYLINUX_SCRIPT}..."
FINAL_WHEEL="$("$PY" "${VLLM_SRC_DIR}/${DETECT_MANYLINUX_SCRIPT}" "$BUILT_WHEEL")" \
    || die "detect-manylinux-tag.py failed for $(basename "$BUILT_WHEEL")."
[[ -f "$FINAL_WHEEL" ]] || die "retagged wheel not found at ${FINAL_WHEEL}."
case "$(basename "$FINAL_WHEEL")" in
    *-manylinux_*.whl) ;;
    *) die "$(basename "$FINAL_WHEEL") is not manylinux-tagged; PyPI rejects it." ;;
esac

# The file name must carry exactly the release version; anything else means
# VLLM_VERSION_OVERRIDE did not take effect.
"$PY" "$WHEEL_UTILS" verify_built_wheel "$FINAL_WHEEL" "$WHEEL_VERSION" 2>/dev/null \
    || die "$(basename "$FINAL_WHEEL") does not carry version ${WHEEL_VERSION}."

# --- Step 6: Check metadata --------------------------------------------------
# Same check as the tpu-inference wheel: catch metadata PyPI would reject.
echo "--- Checking metadata ---"
"$PY" -m twine check --strict "$FINAL_WHEEL" "$BUILT_SDIST"

echo "--- Build complete! ---"
ls -lh "$FINAL_WHEEL" "$BUILT_SDIST"
echo "Wheel: ${FINAL_WHEEL}"
echo "Sdist: ${BUILT_SDIST}"
