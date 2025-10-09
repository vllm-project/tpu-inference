#!/bin/bash

set -euo pipefail

#PACKAGE_VERSION="${BUILDKITE_TAG:-0.1.0.build.${BUILDKITE_BUILD_NUMBER}}"
export TPU_COMMONS_VERSION="0.1.0"
pip install build twine

# If OIDC setup done, can remove this part
export TWINE_USERNAME="__token__"
if [ -z "${PYPI_API_TOKEN:-}" ]; then
  echo "Error: PYPI_API_TOKEN environment variable is not set. Cannot proceed with publishing."
  exit 1
fi
export TWINE_PASSWORD="${PYPI_API_TOKEN}"

rm -rf dist/
# python -m build outputs to the standard 'dist/' directory
python3 -m build

python -m twine upload dist/* \
    --verbose \
    --repository-url https://test.pypi.org/legacy/
