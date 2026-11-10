#!/usr/bin/env bash
# Cache write-back for Kubernetes TPU jobs. Usage: kube_jax_cache.sh snapshot|publish
#
# The POC reads a golden PVC and writes nothing back, which is fine while the
# golden covers the shapes under test. Production needs the return path: build
# 35 showed the golden holds usable keys but not the full shape matrix, with
# adjacent num_reqs=16 and num_reqs=20 shapes each costing ~115s to recompile
# on every build.
#
# OFF BY DEFAULT. Both modes exit immediately unless KUBE_JAX_CACHE_WRITE=1, so
# the POC environment is unchanged and production opts in through Agent Stack
# environment configuration rather than a pipeline fork.
#
#   snapshot   before the test commands, records what the golden already had
#   publish    after them; Buildkite aborts the script on the first failure, so
#              a failed test never reaches this and cannot poison the cache --
#              the same guard run_in_docker.sh gets from the container exit code
set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  snapshot|publish) ;;
  *) echo "usage: $0 snapshot|publish" >&2; exit 2 ;;
esac

if [ "${KUBE_JAX_CACHE_WRITE:-0}" != "1" ]; then
  echo "[cache] KUBE_JAX_CACHE_WRITE!=1; skipping ${MODE}"
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="${SCRIPT_DIR}/kube_jax_cache.py"

CACHE_PATH="${KUBE_JAX_CACHE_PATH:-/cache/tpu_jax_cache}"
CACHE_BASE="${KUBE_JAX_CACHE_BASE:-gs://ullm-ci-cache/jax_cache}"
MANIFEST="${KUBE_JAX_CACHE_MANIFEST:-/cache/.buildkite/jax-cache-manifest.json}"
WORKERS="${KUBE_JAX_CACHE_WORKERS:-32}"

# Resolved from the running image, and deliberately identical to the namespace
# run_in_docker.sh uses on bare metal, so the two fleets warm each other. It is
# printed on every run: the golden-refresh CronJob carries this same namespace
# as static configuration, and a JAX bump that misses it shows up here as a
# prefix nobody is refreshing.
JAX_VERSION="$(python3 -c 'import jax; print(jax.__version__)')"
NAMESPACE="jax${JAX_VERSION}_tpu${TPU_VERSION:-tpu6e}"
PREFIX="${CACHE_BASE}/${NAMESPACE}"

echo "[cache] mode=${MODE} namespace=${NAMESPACE} path=${CACHE_PATH}"
started="$(date +%s)"

if [ "$MODE" = "snapshot" ]; then
  # A missing baseline must not fail the build, but it must not silently become
  # an empty one either: publish would then re-upload the entire golden.
  if ! python3 "$HELPER" snapshot --cache-path "$CACHE_PATH" --manifest "$MANIFEST"; then
    echo "[cache] WARN snapshot failed; publish will be skipped for this job" >&2
    rm -f "$MANIFEST"
  fi
else
  if [ ! -f "$MANIFEST" ]; then
    echo "[cache] WARN no baseline manifest; nothing published" >&2
    exit 0
  fi
  # A GCS failure here must not fail a green TPU test. The cost of a missed
  # publish is one slow future build; the cost of failing is a wasted TPU hour.
  if ! python3 "$HELPER" publish \
      --cache-path "$CACHE_PATH" \
      --manifest "$MANIFEST" \
      --write-prefix "$PREFIX" \
      --workers "$WORKERS"; then
    echo "[cache] WARN publish failed; the test result stands" >&2
  fi
fi

echo "[cache] ${MODE} finished in $(( $(date +%s) - started ))s"
