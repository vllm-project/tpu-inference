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

# Update the release support matrices of a release branch from its
# tpu-inference-ci runs, without committing. The first step of the release prep
# pipeline (prep_pipeline.yml); it also runs by hand on a clean checkout of the
# branch.
#
# The checkout's commit needs one finished NIGHTLY=1 tpu-inference-ci run per
# MODEL_IMPL_TYPE (auto, flax_nnx, vllm), and the newest run of each type must
# have passed "Notify test results", i.e. every model and feature passed. Their
# support matrix tarballs replace support_matrices/release/, and
# scripts/update_readme.py --only release regenerates the release tables of
# README.md and docs/includes. Nothing changes unless all three runs passed.
#
# Output, in RELEASE_PREP_DIR (default: a new temporary directory):
#   manifest.txt             the runs and tarballs used, and the result
#   support_matrices.patch   the change (git diff --binary), empty if none
#   tarballs/                the tarballs as downloaded
# In Buildkite the directory is uploaded as artifacts under release-prep/, and
# the meta-data release-prep-commit, release-prep-changed,
# release-prep-patch-sha256 and release-prep-runs is set for the next steps.
#
# Env:
#   RELEASE_BUILDKITE_READ_TOKEN   Buildkite API token (read_builds, read_artifacts)
#   BUILDKITE_BRANCH               releases/vX.Y.Z (Buildkite sets it)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib.sh"

ORG="vllm"
CI_PIPELINE="${CI_PIPELINE_SLUG:-tpu-inference-ci}"
IMPL_TYPES=(auto flax_nnx vllm)
TPU_VERSIONS=(v6e v7x)
# Steps of nightly_verify.yml that upload the support matrix tarballs.
MATRIX_STEPS='["v6_generate_matrices", "v7_generate_matrices", "notify_test_results"]'
# The only paths the update may change.
ALLOWED_PATHS_REGEX='^(support_matrices/release/|README\.md$|docs/includes/)'

BRANCH="${BUILDKITE_BRANCH:-}"
[[ "${BRANCH}" =~ ${RELEASE_BRANCH_REGEX} ]] \
  || die "not a release branch: '${BRANCH}'."
COMMIT="$(git rev-parse HEAD)"
[[ -z "$(git status --porcelain)" ]] \
  || die "the checkout has local changes; start from a clean checkout of ${BRANCH}."

{ set +x; } 2>/dev/null # Never trace the token.
[[ -n "${RELEASE_BUILDKITE_READ_TOKEN:-}" ]] \
  || die "RELEASE_BUILDKITE_READ_TOKEN is not set."

# Buildkite REST API GET. The token reaches curl through its config on stdin, so
# it never appears on a command line.
bk_get() {
  printf 'header = "Authorization: Bearer %s"\n' "${RELEASE_BUILDKITE_READ_TOKEN}" \
    | curl -fsSL --retry 3 --config - "$@"
}
BK_PIPELINE_API="https://api.buildkite.com/v2/organizations/${ORG}/pipelines/${CI_PIPELINE}"

# <algo> <file>: the file's SHA-1 (1) or SHA-256 (256), on Linux or macOS.
sha_of() {
  if command -v "sha$1sum" >/dev/null; then "sha$1sum" "$2"; else shasum -a "$1" "$2"; fi \
    | cut -d' ' -f1
}

OUT="${RELEASE_PREP_DIR:-$(mktemp -d)/release-prep}"
[[ ! -e "${OUT}" || -z "$(ls -A "${OUT}")" ]] || die "${OUT} is not empty."
mkdir -p "${OUT}/tarballs"
OUT="$(cd "${OUT}" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
MANIFEST="${OUT}/manifest.txt"
# Prints to the log and records in the manifest.
log() { printf '%s\n' "$*" | tee -a "${MANIFEST}"; }

echo "--- :mag: tpu-inference-ci runs of ${COMMIT:0:12}"
log "Release support matrices for ${BRANCH}"
log "Commit:      ${COMMIT} (all three runs must be on it)"
log "CI pipeline: https://buildkite.com/${ORG}/${CI_PIPELINE}"
BRANCH_QUERY="$(jq -rn --arg b "${BRANCH}" '$b | @uri')"
bk_get "${BK_PIPELINE_API}/builds?branch=${BRANCH_QUERY}&commit=${COMMIT}&per_page=100" \
  > "${WORK}/builds.json"

# Check every run and fetch its tarballs before changing anything, so a failed
# or missing run leaves the checkout untouched.
RUNS=""
for impl in "${IMPL_TYPES[@]}"; do
  impl_dir="${impl}"
  [[ "${impl}" == "auto" ]] && impl_dir="default"

  # NIGHTLY=1 runs of this impl type on the commit, newest first.
  jq -c --arg impl "${impl}" --arg sha "${COMMIT}" '
    [.[] | select(.commit == $sha and .env.NIGHTLY == "1"
                  and ((.env.MODEL_IMPL_TYPE // "auto") == $impl))]
    | sort_by(-.number)' "${WORK}/builds.json" > "${WORK}/runs.json"
  [[ "$(jq 'length' "${WORK}/runs.json")" != "0" ]] \
    || die "no NIGHTLY=1 MODEL_IMPL_TYPE=${impl} run of ${COMMIT} in ${CI_PIPELINE}."
  # A run still in progress may supersede the finished ones.
  unfinished="$(jq -r '[.[] | select(.finished_at == null)] | .[0] // empty
    | "#\(.number) (\(.state)) \(.web_url)"' "${WORK}/runs.json")"
  [[ -z "${unfinished}" ]] \
    || die "${impl} run ${unfinished} is not finished; start this again once it is."
  # The newest run on the commit decides; an older pass doesn't count.
  build="$(jq -c '.[0]' "${WORK}/runs.json")"
  number="$(jq -r '.number' <<< "${build}")"
  url="$(jq -r '.web_url' <<< "${build}")"
  log "[${impl_dir}] MODEL_IMPL_TYPE=${impl}: #${number} ${url}"

  # The latest attempt of "Notify test results" decides.
  notify_state="$(jq -r '
    [.jobs[] | select(.step_key == "notify_test_results" and (.retried | not))]
    | (.[0].state // "missing")' <<< "${build}")"
  log "  Notify test results: ${notify_state}"
  [[ "${notify_state}" == "passed" ]] \
    || die "${impl} run #${number} did not pass: not every model and feature passed."
  RUNS+="${impl} #${number} ${url}"$'\n'

  # Every artifact of the steps that upload the tarballs.
  : > "${WORK}/artifacts.json"
  for job_id in $(jq -r --argjson steps "${MATRIX_STEPS}" \
      '.jobs[] | select(.step_key as $k | $steps | index($k)) | .id' <<< "${build}"); do
    bk_get "${BK_PIPELINE_API}/builds/${number}/jobs/${job_id}/artifacts?per_page=100" \
      | jq -c '.[]' >> "${WORK}/artifacts.json"
  done

  for tpu in "${TPU_VERSIONS[@]}"; do
    tarball="${tpu}_${impl_dir}.tar.gz"
    # Retries upload a newer copy of the same tarball; the newest one counts.
    artifact="$(jq -cs --arg p "${tarball}" \
      '[.[] | select(.path == $p)] | max_by(.created_at) // empty' "${WORK}/artifacts.json")"
    [[ -n "${artifact}" ]] || die "run #${number} has no ${tarball}."
    bk_get -o "${OUT}/tarballs/${tarball}" "$(jq -r '.download_url' <<< "${artifact}")"
    # Buildkite records each artifact's SHA-1; a mismatch means a bad download.
    expected_sha1="$(jq -r '.sha1sum // empty' <<< "${artifact}")"
    actual_sha1="$(sha_of 1 "${OUT}/tarballs/${tarball}")"
    [[ -z "${expected_sha1}" || "${expected_sha1}" == "${actual_sha1}" ]] \
      || die "${tarball} doesn't match the SHA-1 Buildkite recorded."
    log "  ${tarball}: sha1 ${actual_sha1}"
    mkdir -p "${WORK}/extracted"
    tar -xzf "${OUT}/tarballs/${tarball}" -C "${WORK}/extracted"
    compgen -G "${WORK}/extracted/${tpu}/${impl_dir}/*.csv" >/dev/null \
      || die "${tarball} holds no ${tpu}/${impl_dir}/*.csv."
  done
done

echo "--- :memo: Update support_matrices/release, README.md and docs/includes"
for impl in "${IMPL_TYPES[@]}"; do
  impl_dir="${impl}"
  [[ "${impl}" == "auto" ]] && impl_dir="default"
  for tpu in "${TPU_VERSIONS[@]}"; do
    dest="support_matrices/release/${tpu}/${impl_dir}"
    mkdir -p "${dest}"
    rm -f "${dest}"/*.csv
    cp "${WORK}/extracted/${tpu}/${impl_dir}"/*.csv "${dest}/"
  done
done
python3 scripts/update_readme.py --only release >/dev/null
git diff --quiet -- 'docs/includes/nightly_*.md' \
  || die "a nightly snippet in docs/includes changed."
git add -A -- support_matrices/release README.md docs/includes
unexpected="$(git status --porcelain | grep -v '^[MAD] ' || true)"
[[ -z "${unexpected}" ]] || die "files outside the release matrices changed:"$'\n'"${unexpected}"
outside="$(git diff --cached --name-only | grep -Ev "${ALLOWED_PATHS_REGEX}" || true)"
[[ -z "${outside}" ]] || die "the update touches files it must not:"$'\n'"${outside}"
git diff --cached --binary > "${OUT}/support_matrices.patch"

CHANGED="no"
if [[ -s "${OUT}/support_matrices.patch" ]]; then
  CHANGED="yes"
  STAT="$(git diff --cached --stat)"
  log "Change:"
  log "${STAT}"
else
  log "Change: none, the branch already matches these runs."
fi
PATCH_SHA="$(sha_of 256 "${OUT}/support_matrices.patch")"
log "Patch:       support_matrices.patch, sha256 ${PATCH_SHA}"

if [[ -z "${BUILDKITE:-}" ]]; then
  echo
  echo "Review: git diff --cached    Record: ${OUT}"
  exit 0
fi

buildkite-agent meta-data set release-prep-commit "${COMMIT}"
buildkite-agent meta-data set release-prep-changed "${CHANGED}"
buildkite-agent meta-data set release-prep-patch-sha256 "${PATCH_SHA}"
buildkite-agent meta-data set release-prep-runs "${RUNS}"
(cd "$(dirname "${OUT}")" && buildkite-agent artifact upload "release-prep/**/*")

RUN_LINES="$(sed -E 's|^([a-z_]+) (#[0-9]+) (.*)$|- \1: [\2](\3)|' <<< "${RUNS%$'\n'}")"
if [[ "${CHANGED}" == "yes" ]]; then
  CHANGE_LINE="Change (artifact \`release-prep/support_matrices.patch\`):"$'\n\n```\n'"${STAT}"$'\n```'
else
  CHANGE_LINE="No change: the branch already matches these runs."
fi
buildkite-agent annotate --style info --context release-prep <<EOF
**Release support matrices for \`${BRANCH}\`** at \`${COMMIT:0:12}\`

NIGHTLY=1 tpu-inference-ci runs, all passed:
${RUN_LINES}

${CHANGE_LINE}
EOF
