#!/usr/bin/env bash
set -euo pipefail

BUILDKITE_BUILD_NUMBER=${BUILDKITE_BUILD_NUMBER:-63}
BUILDKITE_ORGANIZATION_SLUG=${BUILDKITE_ORGANIZATION_SLUG:-tpu-commons}
BUILDKITE_PIPELINE_SLUG=${BUILDKITE_PIPELINE_SLUG:-wip-tpu-inference-kernel-autotune}
API_URL="https://api.buildkite.com/v2/organizations/${BUILDKITE_ORGANIZATION_SLUG}/pipelines/${BUILDKITE_PIPELINE_SLUG}/builds/${BUILDKITE_BUILD_NUMBER}/jobs"

AUTH_TOKEN="${BUILDKITE_API_TOKEN:-${BUILDKITE_AGENT_ACCESS_TOKEN:-}}"

if [[ -z "${AUTH_TOKEN}" ]]; then
	echo "Error: no token found. Set BUILDKITE_API_TOKEN (preferred) or BUILDKITE_AGENT_ACCESS_TOKEN."
	echo "Example: export BUILDKITE_API_TOKEN=<token-with-read_builds-scope>"
	exit 1
fi

STEP_KEYS=$(
	curl -fsSL \
		-H "Authorization: Bearer ${AUTH_TOKEN}" \
		"$API_URL" \
	| jq -r '.[].step_key // empty'
)

echo "STEP_KEYS: $STEP_KEYS"

printf '%s\n' "$STEP_KEYS" | grep -E '^(PRE_KERNEL_AUTOTUNE|POST_KERNEL_AUTOTUNE).*'

echo 'completed...'
