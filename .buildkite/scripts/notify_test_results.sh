#!/bin/sh
set -e

ANY_FAILED=$(buildkite-agent meta-data get "CI_TESTS_FAILED")
FAILURE_LABEL="Not all models and/or features passed"

echo "--- Checking test outcomes"

if [ "${ANY_FAILED}" = "true" ] ; then
  echo "${FAILURE_LABEL}"
  exit 1
else
  echo "All models & features passed."
fi
