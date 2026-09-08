#!/bin/bash
# Client leg for the Qwen3.8-2.4T-A95B-FP8 v7x-32 bringup.
#
# Runs inside the head node's Ray container after run_multihost.sh has seen
# /health come up. The smoke test still goes first and still gets a long
# deadline: precompilation is on now, so the first prefill should no longer pay
# for the whole 92-layer graph, but it is the cheapest place to find out if a
# shape slipped through the buckets.
set -euo pipefail

MODEL="${SERVED_NAME:-Qwen/Qwen3.8-2.4T-A95B-FP8}"
PORT="${VLLM_PORT:-8000}"
ART="${ART_DIR:-/workspace/artifacts}"
DEV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${ART}"

# RandomDataset defaults range_ratio to 0.0, so every request is exactly
# INPUT_LEN in and, with --ignore-eos, exactly OUTPUT_LEN out. The shape is
# therefore identical across runs and the only variance left is the server's.
INPUT_LEN="${INPUT_LEN:-8192}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"

# --fail-with-body, not plain -sS: curl exits 0 on an HTTP 500, which on build
# tc#940 let a dead engine report the job as passed.
echo "--- smoke test (triggers first-request compilation) ---"
time curl -sS --fail-with-body --max-time 5400 "http://localhost:${PORT}/v1/completions" \
  -X POST -H 'Content-Type: application/json' \
  -d "{\"model\": \"${MODEL}\", \"prompt\": \"San Francisco is a\", \"max_tokens\": 32, \"temperature\": 0}" \
  | tee "${ART}/smoke.json"
echo

# --num-warmups: build tc#947 ran with 0 and the means were unusable -- mean TTFT
# 44.5s against a 1.35s median, because the first few requests absorbed
# just-in-time compilation and the percentile machinery counts them like any
# other. serve.py fires the warmups concurrently under the same
# --max-concurrency semaphore before timing starts, so a count at or above
# MAX_CONCURRENCY also drives the scheduler to steady-state batch shapes rather
# than only the single-request path.
NUM_WARMUPS="${NUM_WARMUPS:-32}"

# One server, N benchmark legs. Latency and throughput are the same measurement
# read at two ends of the concurrency curve and there is no single point that
# reports both honestly: at conc 8 the TTFT/TPOT numbers are per-request latency
# with the machine nearly idle, at saturation they are queueing delay and the
# throughput number is the one that means something. Running both against the
# one server costs a few minutes on top of a ~55 min bring-up and makes the
# throughput gain attributable -- a lone saturated run cannot tell "decode got
# slower" from "the batch got bigger".
#
# Format: space-separated "concurrency[:num_prompts]", run in the order given.
# Defaults to the single legacy leg, so an unset BENCH_SWEEP is a no-op.
BENCH_SWEEP="${BENCH_SWEEP:-${MAX_CONCURRENCY}:${NUM_PROMPTS}}"

# --- server-side metrics ----------------------------------------------------
# The client can only see TTFT, which is `queue_time + prefill_time` glued
# together (v1/metrics/stats.py:540,544), so it cannot say whether a slow first
# token means slow prefill or a long admission queue. vLLM exports the two
# phases separately on /metrics along with vllm:prompt_tokens (an exact
# prefill-token counter, no 10s bucketing) and vllm:iteration_tokens_total (the
# per-step token histogram, which is the direct test of whether
# max_num_batched_tokens is the binding constraint). Scrape it.
#
# Counters are cumulative from server start, so a leg is a delta between two
# scrapes. METRICS_SAMPLE_S additionally leaves a coarse timeseries behind so a
# sub-window can be sliced offline; set it to 0 to take only the endpoints.
METRICS_ENABLE="${METRICS_ENABLE:-1}"
METRICS_SAMPLE_S="${METRICS_SAMPLE_S:-5}"
METRICS_URL="http://127.0.0.1:${PORT}/metrics"
# The two endpoint scrapes are stored in full. The timeseries exists only to
# slice rates over a sub-window, so it drops the _bucket lines -- those are 80%
# of the bytes and the distributions are already in the endpoints.
METRICS_FILTER='^vllm:(prompt_tokens|generation_tokens|request_success|num_preemptions|num_requests_(running|waiting)|prefix_cache_|request_(queue|prefill|decode|inference)_time_seconds|time_to_first_token_seconds|iteration_tokens_total|kv_cache_usage_perc)'
METRICS_FILTER_DROP='_bucket\{'

# Never let instrumentation fail the run: a scrape that 404s or times out
# leaves an empty file and the report degrades to "n/a".
snap_metrics() {
  [ "${METRICS_ENABLE}" = "1" ] || return 0
  curl -sS --max-time 30 "${METRICS_URL}" > "$1" 2>/dev/null \
    || { echo "[metrics] scrape failed for ${1##*/}; continuing" >&2; : > "$1"; }
}

SAMPLER_PID=""
start_sampler() {
  [ "${METRICS_ENABLE}" = "1" ] || return 0
  [ "${METRICS_SAMPLE_S}" -gt 0 ] 2>/dev/null || return 0
  : > "$1"
  (
    while :; do
      printf '# SNAPSHOT t=%s\n' "$(date +%s.%N)" >> "$1"
      curl -sS --max-time 5 "${METRICS_URL}" 2>/dev/null \
        | grep -E "${METRICS_FILTER}" | grep -vE "${METRICS_FILTER_DROP}" >> "$1" || true
      sleep "${METRICS_SAMPLE_S}"
    done
  ) &
  SAMPLER_PID=$!
}
stop_sampler() {
  [ -n "${SAMPLER_PID}" ] || return 0
  kill "${SAMPLER_PID}" 2>/dev/null || true
  wait "${SAMPLER_PID}" 2>/dev/null || true
  SAMPLER_PID=""
}
# A benchmark that dies mid-leg must not leave the sampler running.
trap stop_sampler EXIT

declare -a LEG_FILES=()
for leg in ${BENCH_SWEEP}; do
  conc="${leg%%:*}"
  if [ "${leg}" = "${conc}" ]; then prompts="${NUM_PROMPTS}"; else prompts="${leg##*:}"; fi

  # Warmups have to reach the concurrency being measured or the timed run pays
  # for the batch shapes the warmups never compiled -- see the note above.
  warmups="${NUM_WARMUPS}"
  [ "${warmups}" -lt "${conc}" ] && warmups="${conc}"

  # Will the requested shape fit the KV cache, or will the scheduler preempt to
  # make it fit? At 8192+1024 a single request holds 72 blocks, so this is a
  # live constraint rather than a formality, and an oversubscribed run still
  # reports numbers -- they just measure re-prefill. Fail before the benchmark
  # rather than publish those.
  python3 "${DEV_DIR}/qwen38_2p4t_kv_fit.py" \
    --tokens-per-request $((INPUT_LEN + OUTPUT_LEN)) \
    --concurrency "${conc}" \
    --block-size "${BLOCK_SIZE:-128}"

  echo "--- benchmark (in=${INPUT_LEN} out=${OUTPUT_LEN} n=${prompts} conc=${conc} num_warmups=${warmups}) ---"
  snap_metrics "${ART}/metrics_c${conc}_pre.prom"
  start_sampler "${ART}/metrics_c${conc}_series.prom"
  metrics_t0="$(date +%s.%N)"
  vllm bench serve \
    --backend vllm \
    --model "${MODEL}" \
    --host 127.0.0.1 --port "${PORT}" \
    --dataset-name random \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --num-prompts "${prompts}" \
    --max-concurrency "${conc}" \
    --num-warmups "${warmups}" \
    --request-rate inf --seed 42 --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-dir "${ART}" --result-filename "bench_c${conc}.json" \
    2>&1 | tee "${ART}/bench_c${conc}.log"

  # Snapshot and report before the pass/fail check below, so a leg that dies
  # still leaves its server-side metrics behind -- that is exactly the leg
  # whose metrics are worth having.
  stop_sampler
  if [ "${METRICS_ENABLE}" = "1" ]; then
    metrics_t1="$(date +%s.%N)"
    metrics_window="$(awk -v a="${metrics_t0}" -v b="${metrics_t1}" \
      'BEGIN{printf "%.3f", b-a}' || echo 0)"
    snap_metrics "${ART}/metrics_c${conc}_post.prom"
    python3 "${DEV_DIR}/qwen38_2p4t_metrics.py" \
      --pre "${ART}/metrics_c${conc}_pre.prom" \
      --post "${ART}/metrics_c${conc}_post.prom" \
      --series "${ART}/metrics_c${conc}_series.prom" \
      --bench "${ART}/bench_c${conc}.json" \
      --label "c${conc}" --window-s "${metrics_window}" \
      --json "${ART}/metrics_c${conc}.json" || true
  fi

  # `vllm bench serve` exits 0 even when every request failed, so check.
  python3 - "${ART}/bench_c${conc}.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
done, failed = d.get("completed", 0), d.get("failed", 0)
print(f"[bench] completed={done} failed={failed} "
      f"out_tok/s={d.get('output_throughput', 0):.1f} "
      f"median_ttft_ms={d.get('median_ttft_ms', 0):.0f} "
      f"median_tpot_ms={d.get('median_tpot_ms', 0):.1f}")
# With warmups the means should have converged on the medians. A mean TTFT
# still an order of magnitude above the median means compilation leaked into
# the timed run, i.e. the warmups missed a shape the benchmark hits.
mt, md = d.get("mean_ttft_ms", 0), d.get("median_ttft_ms", 0)
print(f"[bench] mean_ttft_ms={mt:.0f} median_ttft_ms={md:.0f} "
      f"ratio={mt / md if md else float('nan'):.1f}x "
      f"mean_tpot_ms={d.get('mean_tpot_ms', 0):.1f}")
if done == 0:
    sys.exit("[bench] FAILED: no request completed")
PY
  LEG_FILES+=("${ART}/bench_c${conc}.json")
done

# Keep the legacy artifact name pointing at the leg run at MAX_CONCURRENCY, so
# every prior MANIFEST's `bench.json` still means the same thing.
if [ -f "${ART}/bench_c${MAX_CONCURRENCY}.json" ]; then
  cp "${ART}/bench_c${MAX_CONCURRENCY}.json" "${ART}/bench.json"
  cp "${ART}/bench_c${MAX_CONCURRENCY}.log" "${ART}/bench.log"
fi

# One table across the sweep. Throughput alone is not a result -- the point of
# the low-concurrency leg is that median TPOT next to it says whether a
# throughput gain came from batching or from something getting faster.
if [ "${#LEG_FILES[@]}" -gt 1 ]; then
  echo "--- benchmark sweep summary"
  python3 - "${LEG_FILES[@]}" <<'PY'
import json, sys
# max_concurrent_requests is the batch the server actually reached, as opposed
# to the ceiling the client asked for -- the one column that says whether
# max_num_seqs x dp_size was really achieved.
# med_ttft is kept for continuity with earlier builds, but q_ms/pf_ms are the
# columns to read: they are the server's own split of that same TTFT into
# admission queue and prefill, so they say which half moved. Note the windows
# differ -- out_tok/s and tot_tok/s are the timed run, while pf_tok/s, q_ms and
# pf_ms come from counters spanning the whole invocation, warmups included.
hdr = ("conc", "peak_bs", "n", "dur_s", "out_tok/s", "tot_tok/s", "pf_tok/s",
       "med_ttft", "p99_ttft", "q_ms", "pf_ms", "med_tpot", "p99_tpot", "med_e2el")
print("  ".join(f"{h:>10}" for h in hdr))
for path in sys.argv[1:]:
    d = json.load(open(path))
    try:
        m = json.load(open(path.replace("bench_c", "metrics_c")))
    except (OSError, ValueError):
        m = {}
    def mv(k, prec=1):
        v = m.get(k)
        return f"{v:.{prec}f}" if isinstance(v, (int, float)) else "n/a"
    row = (d.get("max_concurrency", "?"), d.get("max_concurrent_requests", "?"),
           d.get("completed", 0), f"{d.get('duration', 0):.1f}",
           f"{d.get('output_throughput', 0):.1f}",
           f"{d.get('total_token_throughput', 0):.1f}",
           mv("prompt_tokens_per_s"),
           f"{d.get('median_ttft_ms', 0):.0f}", f"{d.get('p99_ttft_ms', 0):.0f}",
           mv("queue_time_ms", 0), mv("prefill_time_ms", 0),
           f"{d.get('median_tpot_ms', 0):.2f}", f"{d.get('p99_tpot_ms', 0):.2f}",
           f"{d.get('median_e2el_ms', 0):.0f}")
    print("  ".join(f"{str(c):>10}" for c in row))
PY
fi

# Did the local-disk compilation cache actually get written? This is the head
# host's copy only; each of the other three hosts has its own under the same
# bind mount. A non-zero count here means the next run on this slice starts
# warm -- that is the whole point of the /tmp/jax_cache_tpu7x mount.
JAX_CACHE_DIR="${VLLM_XLA_CACHE_PATH:-/root/jax_cache}"
echo "--- jax compilation cache (head host) ---"
echo "dir=${JAX_CACHE_DIR} entries=$(find "${JAX_CACHE_DIR}" -type f 2>/dev/null | wc -l)"
du -sh "${JAX_CACHE_DIR}" 2>/dev/null || true
# Post-run disk state. With precompilation on, this is the number that says
# whether the 20GB cap and the 25GB floor are set anywhere near right.
echo "--- disk after run (head host) ---"
df -h "${JAX_CACHE_DIR}" / 2>/dev/null || true
