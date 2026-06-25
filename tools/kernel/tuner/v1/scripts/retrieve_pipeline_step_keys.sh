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

HTTP_CODE="$(
	curl -sS \
		-o /tmp/buildkite_jobs_response.json \
		-w '%{http_code}' \
		-H "Authorization: Bearer ${AUTH_TOKEN}" \
		"$API_URL"
)"

if [[ "${HTTP_CODE}" != "200" ]]; then
	echo "Error: Buildkite API request failed with HTTP ${HTTP_CODE}."
	if [[ "${HTTP_CODE}" == "401" ]]; then
		echo "Hint: token is invalid for Buildkite REST API."
		echo "Use BUILDKITE_API_TOKEN with read_builds scope."
		echo "Note: BUILDKITE_AGENT_ACCESS_TOKEN may not have REST API access in your org configuration."
	fi
	echo "Request URL: ${API_URL}"
	exit 22
fi

STEP_KEYS="$(jq -r '.[].step_key // empty' /tmp/buildkite_jobs_response.json)"

echo "STEP_KEYS: $STEP_KEYS"

printf '%s\n' "$STEP_KEYS" | grep -E '^(PRE_KERNEL_AUTOTUNE|POST_KERNEL_AUTOTUNE).*'

echo 'completed...'
