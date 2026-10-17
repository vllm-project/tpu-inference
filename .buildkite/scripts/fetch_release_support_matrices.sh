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

# Clones tpu-inference, checks out a release branch, fetches the support matrix
# tarballs of the branch's tpu-inference CI runs (one per MODEL_IMPL_TYPE:
# auto, flax_nnx, vllm) into a folder next to the clone, updates
# support_matrices/release/ in the clone from them, and regenerates only the
# release tables of README.md and docs/includes with
# scripts/update_readme.py --only release. It never commits or pushes.
#
#   <clone dir>/              the clone; review it with git diff
#   <clone dir>-artifacts/    what was fetched, kept as a record:
#     manifest.txt            which runs and artifacts were used, and the result
#     changes.diff            the full change made in the clone
#     <tpu>_<impl>.tar.gz     the tarballs as downloaded
#     extracted/              their contents
#
# All three runs must be on the commit the clone is on (the branch's latest
# commit), finished, and the newest of their type there. A run counts only if
# its "Notify test results" step passed, i.e. every model and feature passed
# after any retries; otherwise no matrix is changed. Retries upload a newer copy
# of the same tarball, so the newest one is used.
#
# Usage:
#   BUILDKITE_API_TOKEN=<token with read_builds, read_artifacts> \
#     fetch_release_support_matrices.sh releases/vX.Y.Z [clone dir]
# The clone dir defaults to ./tpu-inference-vX.Y.Z; neither it nor its
# -artifacts folder may exist yet.
set -euo pipefail

# The checkout this script lives in, for tools a release branch may predate.
TOOL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

GITHUB_REPO="vllm-project/tpu-inference"
ORG="vllm"
CI_PIPELINE="${CI_PIPELINE_SLUG:-tpu-inference-ci}"
IMPL_TYPES=(auto flax_nnx vllm)
TPU_VERSIONS=(v6e v7x)
# Steps of nightly_verify.yml that upload the support matrix tarballs.
MATRIX_STEPS='["v6_generate_matrices", "v7_generate_matrices", "notify_test_results"]'

BRANCH="${1:?usage: $0 releases/vX.Y.Z [clone dir]}"
if [[ ! "${BRANCH}" =~ ^releases/v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "ERROR: expected a release branch releases/vX.Y.Z, not '${BRANCH}'." >&2
  exit 1
fi
VERSION="${BRANCH#releases/}"
CLONE_DIR="${2:-tpu-inference-${VERSION}}"
ARTIFACT_DIR="${CLONE_DIR%/}-artifacts"
for dir in "${CLONE_DIR}" "${ARTIFACT_DIR}"; do
  if [[ -e "${dir}" ]]; then
    echo "ERROR: '${dir}' already exists; pass another clone dir or remove it." >&2
    exit 1
  fi
done

{ set +x; } 2>/dev/null # Never trace the token.
: "${BUILDKITE_API_TOKEN:?BUILDKITE_API_TOKEN must be set}"

# Buildkite REST API GET. The token reaches curl through its config on stdin
# (printf is a builtin), so it never appears on a command line.
bk_get() {
  printf 'header = "Authorization: Bearer %s"\n' "${BUILDKITE_API_TOKEN}" \
    | curl -fsSL --retry 3 --config - "$@"
}
BK_PIPELINE_API="https://api.buildkite.com/v2/organizations/${ORG}/pipelines/${CI_PIPELINE}"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

mkdir -p "${ARTIFACT_DIR}/extracted"
ARTIFACT_DIR="$(cd "${ARTIFACT_DIR}" && pwd)"
MANIFEST="${ARTIFACT_DIR}/manifest.txt"
# Prints to the terminal and records in the manifest.
log() { printf '%s\n' "$*" | tee -a "${MANIFEST}"; }

log "Release support matrices for ${BRANCH}"
log "Fetched:     $(date -u +%Y-%m-%dT%H:%M:%SZ) by ${USER:-unknown}"
log "CI pipeline: https://buildkite.com/${ORG}/${CI_PIPELINE}"

log ""
log "--- Clone"
git clone -q --branch "${BRANCH}" --single-branch \
  "https://github.com/${GITHUB_REPO}.git" "${CLONE_DIR}"
cd "${CLONE_DIR}"
log "Repository:  https://github.com/${GITHUB_REPO}"
log "Clone dir:   $(pwd)"
log "Checked out: ${BRANCH} at $(git rev-parse HEAD)"

log ""
log "--- CI runs"
# Every run must be on the commit the clone is on, i.e. the latest commit of
# the branch, so all three matrices come from the code being released.
HEAD_SHA="$(git rev-parse HEAD)"
log "Commit:      ${HEAD_SHA} (all three runs must be on it)"
BRANCH_QUERY="$(jq -rn --arg b "${BRANCH}" '$b | @uri')"
bk_get "${BK_PIPELINE_API}/builds?branch=${BRANCH_QUERY}&commit=${HEAD_SHA}&per_page=100" \
  > "${WORK_DIR}/builds.json"

# Check every run and fetch its tarballs before touching the clone, so a failed
# or missing run leaves every matrix unchanged.
for impl in "${IMPL_TYPES[@]}"; do
  impl_dir="${impl}"
  [[ "${impl}" == "auto" ]] && impl_dir="default"

  # Support matrix runs of this impl type on the commit, newest first.
  jq -c --arg impl "${impl}" --arg sha "${HEAD_SHA}" '
    [.[] | select(.commit == $sha and .env.NIGHTLY == "1"
                  and ((.env.MODEL_IMPL_TYPE // "auto") == $impl))]
    | sort_by(-.number)' "${WORK_DIR}/builds.json" > "${WORK_DIR}/runs.json"
  if [[ "$(jq 'length' "${WORK_DIR}/runs.json")" == "0" ]]; then
    log "ERROR: CI has not run MODEL_IMPL_TYPE=${impl} on ${HEAD_SHA} yet." >&2
    exit 1
  fi
  # A run still in progress may supersede the finished ones; wait for it.
  unfinished="$(jq -r '[.[] | select(.finished_at == null)] | .[0] // empty
    | "#\(.number) (\(.state)) \(.web_url)"' "${WORK_DIR}/runs.json")"
  if [[ -n "${unfinished}" ]]; then
    log "ERROR: ${impl} run ${unfinished} is not finished; wait for it and run this again." >&2
    exit 1
  fi
  # The newest run on the commit decides; an older pass does not count.
  build="$(jq -c '.[0]' "${WORK_DIR}/runs.json")"
  number="$(jq -r '.number' <<<"${build}")"

  log "[${impl_dir}] MODEL_IMPL_TYPE=${impl}"
  log "  run:      #${number} $(jq -r '.web_url' <<<"${build}")"
  log "  commit:   $(jq -r '.commit' <<<"${build}")"
  log "  created:  $(jq -r '.created_at' <<<"${build}")   finished: $(jq -r '.finished_at' <<<"${build}")"
  older="$(jq -r '[.[1:][] | "#\(.number)"] | join(", ")' "${WORK_DIR}/runs.json")"
  if [[ -n "${older}" ]]; then
    log "  older runs on this commit, not used: ${older}"
  fi

  # The latest attempt of "Notify test results" decides; a retried attempt
  # is superseded by the one that replaced it.
  notify_state="$(jq -r '
    [.jobs[] | select(.step_key == "notify_test_results" and (.retried | not))]
    | (.[0].state // "missing")' <<<"${build}")"
  log "  notify:   ${notify_state}"
  if [[ "${notify_state}" != "passed" ]]; then
    log "ERROR: run #${number} did not pass; not every model and feature passed, so no matrix was changed." >&2
    exit 1
  fi

  # Every artifact uploaded by the steps that write the tarballs.
  : > "${WORK_DIR}/artifacts.json"
  for job_id in $(jq -r --argjson steps "${MATRIX_STEPS}" \
      '.jobs[] | select(.step_key as $k | $steps | index($k)) | .id' <<<"${build}"); do
    bk_get "${BK_PIPELINE_API}/builds/${number}/jobs/${job_id}/artifacts?per_page=100" \
      | jq -c '.[]' >> "${WORK_DIR}/artifacts.json"
  done

  for tpu in "${TPU_VERSIONS[@]}"; do
    tarball="${tpu}_${impl_dir}.tar.gz"
    artifact="$(jq -cs --arg p "${tarball}" \
      '[.[] | select(.path == $p)] | max_by(.created_at) // empty' "${WORK_DIR}/artifacts.json")"
    if [[ -z "${artifact}" ]]; then
      log "ERROR: run #${number} has no ${tarball}." >&2
      exit 1
    fi
    step="$(jq -r --arg id "$(jq -r '.job_id' <<<"${artifact}")" \
      '.jobs[] | select(.id == $id) | .step_key' <<<"${build}")"
    bk_get -o "${ARTIFACT_DIR}/${tarball}" "$(jq -r '.download_url' <<<"${artifact}")"

    # Buildkite records each artifact's SHA-1; a mismatch means a bad download.
    expected_sha1="$(jq -r '.sha1sum // empty' <<<"${artifact}")"
    actual_sha1="$(shasum -a 1 "${ARTIFACT_DIR}/${tarball}" | cut -d' ' -f1)"
    if [[ -n "${expected_sha1}" && "${expected_sha1}" != "${actual_sha1}" ]]; then
      log "ERROR: ${tarball} does not match the SHA-1 Buildkite recorded." >&2
      exit 1
    fi
    log "  ${tarball}: from ${step}, uploaded $(jq -r '.created_at' <<<"${artifact}"), sha1 ${actual_sha1}"

    tar -xzf "${ARTIFACT_DIR}/${tarball}" -C "${ARTIFACT_DIR}/extracted"
    if ! compgen -G "${ARTIFACT_DIR}/extracted/${tpu}/${impl_dir}/*.csv" >/dev/null; then
      log "ERROR: ${tarball} holds no ${tpu}/${impl_dir}/*.csv." >&2
      exit 1
    fi
  done
done

log ""
log "--- Update support_matrices/release"
for impl in "${IMPL_TYPES[@]}"; do
  impl_dir="${impl}"
  [[ "${impl}" == "auto" ]] && impl_dir="default"
  for tpu in "${TPU_VERSIONS[@]}"; do
    dest="support_matrices/release/${tpu}/${impl_dir}"
    mkdir -p "${dest}"
    rm -f "${dest}"/*.csv
    cp "${ARTIFACT_DIR}/extracted/${tpu}/${impl_dir}"/*.csv "${dest}/"
  done
done
CHANGES="$(git status --short -- support_matrices/release)"
if [[ -z "${CHANGES}" ]]; then
  log "No change: the branch already matches these runs."
else
  log "${CHANGES}"
fi

log ""
log "--- Update README.md and docs/includes (release tables only)"
# This script's own update_readme.py, run on the clone: a release branch may
# have been cut before --only existed.
python3 "${TOOL_ROOT}/scripts/update_readme.py" --only release >/dev/null
if ! git diff --quiet -- 'docs/includes/nightly_*.md'; then
  log "ERROR: a nightly snippet in docs/includes changed." >&2
  exit 1
fi
log "$(git status --short -- README.md docs/includes)"

# Keep the full change as part of the record.
git diff > "${ARTIFACT_DIR}/changes.diff"
log ""
log "--- Result"
log "Changed files: $(git status --short | wc -l | tr -d ' ') (full diff: ${ARTIFACT_DIR}/changes.diff)"

echo
echo "Done. Review: cd $(pwd) && git diff"
echo "Record:  ${MANIFEST}"
