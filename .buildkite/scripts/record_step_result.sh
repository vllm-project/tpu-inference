#!/bin/sh
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <step_key>"
    exit 1
fi

STEP_KEY="$1"

echo "--- Checking ${STEP_KEY} Outcome"

# Try to get the custom string you saved
CUSTOM_STATUS=$(buildkite-agent meta-data get "${STEP_KEY}" --default "")

if [ -n "$CUSTOM_STATUS" ]; then
    OUTCOME="$CUSTOM_STATUS"
else
    OUTCOME=$(buildkite-agent step get "outcome" --step "${STEP_KEY}" || echo "skipped")
fi

echo "Step ${STEP_KEY} outcome: ${OUTCOME}"
message=""

case $OUTCOME in
  "passed")
    message="✅"
    ;;
  "skipped")
    message="N/A"
    ;;
  "unverified")
    message="unverified"
    ;;
  *)
    message="❌"
    ;;
esac

buildkite-agent meta-data set "${CI_TARGET}_category" "${CI_CATEGORY}"
buildkite-agent meta-data set "${CI_TARGET}:${CI_STAGE}" "${message}"

if [ "${OUTCOME}" != "passed" ] && [ "${OUTCOME}" != "skipped" ] && [ "${OUTCOME}" != "unverified" ]; then
    exit 1
fi
