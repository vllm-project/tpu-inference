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

# After the approval: commit the reviewed support_matrices.patch to the release
# branch as the CI bot and push it. Applies the patch artifact itself, not a
# fresh update, so what is pushed is what was approved; refuses if the patch
# isn't the reviewed one or the branch moved since the update step. Records the
# pushed commit as release-prep-release-commit.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib.sh"

GITHUB_REPO="vllm-project/tpu-inference"
ALLOWED_PATHS_REGEX='^(support_matrices/release/|README\.md$|docs/includes/)'

BRANCH="${BUILDKITE_BRANCH:-}"
[[ "${BRANCH}" =~ ${RELEASE_BRANCH_REGEX} ]] \
  || die "not a release branch: '${BRANCH}'."
VERSION="v${BASH_REMATCH[1]}"
TESTED="$(buildkite-agent meta-data get release-prep-commit)"
WANT_SHA="$(buildkite-agent meta-data get release-prep-patch-sha256)"

sha256_of() {
  if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi \
    | cut -d' ' -f1
}

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

echo "--- :lock: Check the reviewed patch and the branch"
buildkite-agent artifact download "release-prep/support_matrices.patch" "${WORK}"
PATCH="${WORK}/release-prep/support_matrices.patch"
GOT_SHA="$(sha256_of "${PATCH}")"
[[ "${GOT_SHA}" == "${WANT_SHA}" ]] \
  || die "the patch is ${GOT_SHA}, not the reviewed ${WANT_SHA}."
[[ "$(git rev-parse HEAD)" == "${TESTED}" ]] \
  || die "the checkout is $(git rev-parse HEAD), not the tested ${TESTED}."

git fetch -q origin "${BRANCH}"
REMOTE="$(git rev-parse FETCH_HEAD)"
if [[ "${REMOTE}" != "${TESTED}" ]]; then
  # A retry after a successful push finds its own commit on top of the tested one.
  if [[ "$(git log -1 --format=%P "${REMOTE}")" == "${TESTED}" ]] \
      && git log -1 --format=%B "${REMOTE}" | grep -qx "Tested-on: ${TESTED}"; then
    echo "Already pushed as ${REMOTE}."
    buildkite-agent meta-data set release-prep-release-commit "${REMOTE}"
    exit 0
  fi
  die "${BRANCH} moved to ${REMOTE} since ${TESTED} was tested; start a new prep build."
fi

echo "--- :memo: Commit"
git apply --index "${PATCH}"
outside="$(git diff --cached --name-only | grep -Ev "${ALLOWED_PATHS_REGEX}" || true)"
[[ -z "${outside}" ]] || die "the patch touches files it must not:"$'\n'"${outside}"
git -c user.name="vllm-ci-bot[bot]" \
  -c user.email="vllm-ci-bot[bot]@users.noreply.github.com" \
  commit -q -s -m "[skip ci] Update release support matrices for ${VERSION}" \
  -m "Tested-on: ${TESTED}" -m "Reviewed-in: ${BUILDKITE_BUILD_URL:-unknown}"
RELEASE_COMMIT="$(git rev-parse HEAD)"
git show --stat --format='%H%n%s' HEAD

echo "--- :github: Push to ${BRANCH}"
{ set +x; } 2>/dev/null # Never trace the token.
# shellcheck source=/dev/null
source .buildkite/scripts/export_ci_bot_token.sh
[[ -n "${GITHUB_CI_BOT_TOKEN:-}" ]] || die "no GitHub token for the CI bot."
# The token reaches git through GIT_ASKPASS, never on a command line.
# shellcheck disable=SC2016 # Expanded by the askpass script, not here.
printf '#!/bin/sh\necho "${GITHUB_CI_BOT_TOKEN}"\n' > "${WORK}/askpass"
chmod 700 "${WORK}/askpass"
# A plain push: fast-forward only, so a branch that moved meanwhile rejects it.
GIT_ASKPASS="${WORK}/askpass" GIT_TERMINAL_PROMPT=0 \
  git push -q "https://x-access-token@github.com/${GITHUB_REPO}.git" \
  "HEAD:refs/heads/${BRANCH}"
unset GITHUB_CI_BOT_TOKEN

buildkite-agent meta-data set release-prep-release-commit "${RELEASE_COMMIT}"
buildkite-agent annotate --style success --context release-prep-push \
  "Pushed [\`${RELEASE_COMMIT:0:12}\`](https://github.com/${GITHUB_REPO}/commit/${RELEASE_COMMIT}) to \`${BRANCH}\`: the reviewed support matrix update on top of \`${TESTED:0:12}\`."
