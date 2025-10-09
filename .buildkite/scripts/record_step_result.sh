#!/bin/sh
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <step_key>"
    exit 1
fi

STEP_KEY="$1"

echo "--- Checking ${STEP_KEY} Outcome"

OUTCOME=$(buildkite-agent step get "outcome" --step "${STEP_KEY}" || echo "skipped")
echo "Step ${STEP_KEY} outcome: ${OUTCOME}"
message=""

case $OUTCOME in
  "passed")
    message="✅"
    ;;
  "skipped")
    message="N/A"
    ;;
  *)
    message="❌"
    ;;
esac

buildkite-agent meta-data set "${CI_TARGET}:${CI_STAGE}" "${message}"

if [ "${OUTCOME}" != "passed" ] && [ "${OUTCOME}" != "skipped" ] ; then
    exit 1
fi
