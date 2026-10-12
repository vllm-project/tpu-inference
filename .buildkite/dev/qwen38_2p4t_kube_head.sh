#!/bin/bash
# Head-pod driver for the Qwen3.8-2.4T-A95B-FP8 runs on the kube v7x-32 slice.
#
# .buildkite/kubernetes/run.sh hands this to the launcher as the command of
# ray-multihost-slice.yaml, and multihost_entry.sh runs it on the head once all
# four hosts have joined the Ray cluster. It does what run_multihost.sh did for
# these runs on bare metal: start the server, wait for /health, run the legs,
# stop the server. multihost_entry.sh uploads ARTIFACTS_DIR when this returns.
#
# Legs, in the order of the bare-metal builds: the smoke test, then lm_eval
# (STD_EVAL_SUITE), then the perf sweep (BENCH_SWEEP), then the fork-based
# harness (EVAL_SUITE; "none" makes it exit 0 at once). `&&`: each eval leg
# records its own failures, so reaching the sweep means the evals ran, not that
# they passed.
#
# Logs: the server's own output is tee'd to ${ART_DIR}/vllm_serve.log. The Ray
# workers, including the one on this host, log to their pod's stderr
# (RAY_LOG_TO_STDERR=1 in the manifest), so their lines -- the MoE dispatch and
# collect plans, the profiler -- are only in the Buildkite job log.
set -uo pipefail

S="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ART_DIR="${ART_DIR:?ART_DIR is not set}"
mkdir -p "${ART_DIR}"
PORT="${VLLM_PORT:-8000}"
export VLLM_PORT="${PORT}"

echo "--- code under test"
{
  echo "buildkite: build ${BUILDKITE_BUILD_NUMBER:-?} commit ${BUILDKITE_COMMIT:-?} branch ${BUILDKITE_BRANCH:-?}"
  echo "tpu_inference: $(git -C "${S}/../.." log --oneline -1 2>/dev/null || echo '(no git metadata in the image)')"
  python3 -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null | tail -1
  python3 -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)" 2>/dev/null | tail -1
  echo "hostname=$(hostname) date=$(date -u +%FT%TZ)"
} | tee "${ART_DIR}/code_under_test.txt"

# kv_fit.py reads the server log from where run_multihost.sh used to leave it.
SERVE_LOG="${ART_DIR}/vllm_serve.log"
ln -sfn "${SERVE_LOG}" /root/vllm_serve.log 2>/dev/null || true

# Whole-run server sampler: KV usage, batch and preemptions every 10 s from
# /health onwards, covering the eval legs as well as the sweep. The client's
# own sampler only covers the sweep.
FULL_SERIES="${ART_DIR}/metrics_full_series.prom"
SAMPLER_PID=""
start_full_sampler() {
  (
    while :; do
      printf '# SNAPSHOT t=%s\n' "$(date +%s.%N)"
      curl -sS --max-time 5 "http://127.0.0.1:${PORT}/metrics" 2>/dev/null \
        | grep -E '^vllm:(kv_cache_usage_perc|num_requests_(running|waiting)|num_preemptions|prompt_tokens_total|generation_tokens_total|request_success_total)' || true
      sleep 10
    done
  ) >> "${FULL_SERIES}" &
  SAMPLER_PID=$!
}

SERVE_PID=""
finish() {
  local rc="$1"
  [ -n "${SAMPLER_PID}" ] && kill "${SAMPLER_PID}" 2>/dev/null
  curl -sS --max-time 30 "http://127.0.0.1:${PORT}/metrics" > "${ART_DIR}/metrics_final.prom" 2>/dev/null || true
  if [ -n "${SERVE_PID}" ] && kill -0 "${SERVE_PID}" 2>/dev/null; then
    echo "--- stopping vllm serve (pid ${SERVE_PID})"
    kill -TERM "${SERVE_PID}" 2>/dev/null
    for _ in $(seq 1 24); do kill -0 "${SERVE_PID}" 2>/dev/null || break; sleep 5; done
    kill -KILL "${SERVE_PID}" 2>/dev/null || true
  fi
  # Each host's Ray worker wrote its own trace, but only this pod uploads, so
  # pull the other hosts' profile files over the still-running Ray cluster.
  if [ -n "${PHASED_PROFILING_DIR:-}" ] && [ "${COLLECT_WORKER_PROFILES:-1}" = "1" ]; then
    echo "--- collecting profile files from the other hosts"
    timeout 900 python3 "${S}/qwen38_2p4t_collect_profiles.py" "${PHASED_PROFILING_DIR}" \
      || echo "[head] WARNING: profile collection did not finish"
  fi
  # What the head can see of the flags under test. The worker-side lines
  # (MOE_HIERARCHICAL_*, "Starting profiling for") are in the job log only.
  {
    echo "runner_slot_mamba manager lines: $(grep -c 'GDN groups take no scheduler blocks' "${SERVE_LOG}" 2>/dev/null)"
    echo "runner_slot_mamba ignored warnings: $(grep -c 'Ignoring SKIP_MAMBA_SCHEDULER_BLOCKS' "${SERVE_LOG}" 2>/dev/null)"
    grep -m3 -E 'GPU KV cache size|Maximum concurrency' "${SERVE_LOG}" 2>/dev/null
    echo "final num_preemptions: $(grep -E '^vllm:num_preemptions_total' "${ART_DIR}/metrics_final.prom" 2>/dev/null | awk '{s+=$NF} END{print s}')"
    if [ -n "${PHASED_PROFILING_DIR:-}" ]; then
      echo "profile files (this host plus those collected from the others):"
      find "${PHASED_PROFILING_DIR}" \( -name '*.xplane.pb' -o -name '*.trace.json.gz' \) -printf '%s\t%p\n' 2>/dev/null
    fi
    echo "legs rc=${rc}"
  } | tee "${ART_DIR}/flag_check.txt"
  echo "${rc}" > "${ART_DIR}/.run_rc"
  exit "${rc}"
}

echo "--- starting vllm serve"
bash "${S}/qwen38_2p4t_serve.sh" > >(tee "${SERVE_LOG}") 2>&1 &
SERVE_PID=$!

# Bounds the load and precompile, not the legs; the launcher's
# TPU_MAX_RUNTIME_SECONDS bounds the whole run.
deadline=$((SECONDS + ${SERVER_TIMEOUT_S:-7200}))
until curl -sf --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
    echo "[head] vllm serve exited before /health came up"
    finish 1
  fi
  if [ "${SECONDS}" -ge "${deadline}" ]; then
    echo "[head] /health not up after ${SERVER_TIMEOUT_S:-7200}s"
    finish 1
  fi
  sleep 30
done
echo "--- /health is up after ${SECONDS}s"
start_full_sampler

CLIENT_SMOKE_ONLY=1 bash "${S}/qwen38_2p4t_client.sh" \
  && bash "${S}/qwen38_2p4t_eval_std.sh" \
  && SKIP_SMOKE=1 bash "${S}/qwen38_2p4t_client.sh" \
  && bash "${S}/qwen38_2p4t_eval.sh"
finish $?
