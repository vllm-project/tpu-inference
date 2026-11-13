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

# Decides whether a build runs on the bare-metal TPU agents or the kube fleet.
# Sourced by bootstrap.sh, which uploads one set of pipeline files or the other.
# In order, the first of these that applies decides:
#
#   CI_FLEET=bare|kube  on the pipeline or the build. bare on the pipeline
#                        takes everything off kube at once.
#   [ci kube] / [ci bare] in the message of the commit being built. On a PR
#                        that is the head commit; a squash merge carries the
#                        PR's title into the merged commit's message.
#   the ramp             in .buildkite/kube_rollout.conf; see there.

ci_fleet_bucket() {
  local digest
  digest=$(printf '%s' "$1" | sha256sum | cut -c1-8)
  echo $((16#${digest} % 100))
}

resolve_ci_fleet() {
  local requested="${CI_FLEET:-auto}" key bucket
  local KUBE_PERCENT SALT
  CI_FLEET_REASON=""
  case "$requested" in
    bare|kube)
      CI_FLEET="$requested"
      CI_FLEET_REASON="CI_FLEET=${requested}"
      ;;
    auto)
      if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" && -n "${BUILDKITE_PULL_REQUEST:-}" ]]; then
        key="${BUILDKITE_PIPELINE_SLUG:-}/pr/${BUILDKITE_PULL_REQUEST}"
      else
        key="${BUILDKITE_PIPELINE_SLUG:-}/commit/${BUILDKITE_COMMIT:-}"
      fi
      if [[ "${BUILDKITE_MESSAGE:-}" =~ \[ci[[:space:]]bare\] ]]; then
        CI_FLEET=bare
        CI_FLEET_REASON="[ci bare] tag"
      elif [[ "${BUILDKITE_MESSAGE:-}" =~ \[ci[[:space:]]kube\] ]]; then
        CI_FLEET=kube
        CI_FLEET_REASON="[ci kube] tag"
      else
        # shellcheck source=/dev/null
        source "$(dirname "${BASH_SOURCE[0]}")/../kube_rollout.conf"
        if ! [[ "${KUBE_PERCENT:-}" =~ ^[0-9]+$ ]] || (( KUBE_PERCENT > 100 )); then
          echo "kube_rollout.conf: KUBE_PERCENT must be 0-100, got '${KUBE_PERCENT:-}'" >&2
          return 1
        fi
        bucket=$(ci_fleet_bucket "${SALT:-v1}:${key}")
        if (( bucket < KUBE_PERCENT )); then
          CI_FLEET=kube
        else
          CI_FLEET=bare
        fi
        CI_FLEET_REASON="bucket ${bucket} vs KUBE_PERCENT=${KUBE_PERCENT} (${key})"
      fi
      ;;
    *)
      echo "CI_FLEET must be bare, kube or auto, got '${requested}'" >&2
      return 1
      ;;
  esac
  export CI_FLEET CI_FLEET_REASON
  echo "CI fleet: ${CI_FLEET} (${CI_FLEET_REASON})"
  buildkite-agent meta-data set "ci-fleet" "$CI_FLEET"
  buildkite-agent meta-data set "ci-fleet-reason" "$CI_FLEET_REASON"
}
