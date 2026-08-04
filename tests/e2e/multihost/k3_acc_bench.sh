#!/bin/bash
# Copyright 2025 Google LLC
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

# Post-probe measurement phases for a multi-host K3 serve, run inside the head
# container against the already-healthy server. Accuracy runs BEFORE the
# throughput bench so a bench-phase crash cannot cost the accuracy score.
# Each phase is under its own `timeout` so a hang fails the phase (rc=124)
# instead of eating the step's whole time budget.
set -u

MODEL="${K3_MODEL:-gs://tpu-commons-ci/moonshootai/kimi/k3}"
BASE="${K3_BASE_URL:-http://localhost:8000}"
TOK_DIR=/tmp/k3_tok
# Flags (any order):
#   --skip-bench   run accuracy only (bench numbers already recorded for this
#                  exact serve config, e.g. build 761).
#   --skip-acc     skip the gsm8k phase entirely (accuracy for this code is
#                  covered by a separate run); bench phases still run.
#   --full-gsm8k   run the full 1319-question set instead of --limit 250.
#                  ~105 min at the measured 250-in-20-min rate; cap 160 min.
#   --bs1-bench    single-request decode benchmark: 8k/1k at max-concurrency 1
#                  (the batch-size-1 shape public K3 numbers are quoted at).
#   --matrix-bench replace the default concurrency-8 suite with the 2x2
#                  matrix {1k/1k, 8k/1k} x {concurrency 8 (32 prompts),
#                  concurrency 64 (128 prompts)}, each phase capped 60 min.
SKIP_BENCH=""
SKIP_ACC=""
BS1_BENCH=""
MATRIX_BENCH=""
GSM8K_LIMIT_ARGS=(--limit 250)
GSM8K_DESC="--limit 250"
ACC_CAP=5400
for arg in "$@"; do
  case "$arg" in
    --skip-bench) SKIP_BENCH="--skip-bench" ;;
    --skip-acc) SKIP_ACC="1" ;;
    --full-gsm8k) GSM8K_LIMIT_ARGS=(); GSM8K_DESC="FULL 1319"; ACC_CAP=12600 ;;
    --bs1-bench) BS1_BENCH="1" ;;
    --matrix-bench) MATRIX_BENCH="1" ;;
    *) echo "[k3-measure] unknown flag: $arg" >&2; exit 2 ;;
  esac
done
if [ -n "${MATRIX_BENCH}" ] && [ "${SKIP_BENCH}" = "--skip-bench" ]; then
  echo "[k3-measure] --matrix-bench and --skip-bench are contradictory" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Tokenizer: the server reads it from GCS via the streamer, but lm_eval and
# `vllm bench serve` are clients and need local files. No gsutil in the image;
# use the python storage client (node service account provides credentials).
# Skip weights; everything else in the checkpoint dir is small.
# ---------------------------------------------------------------------------
echo "[k3-tok] fetching tokenizer/config files from the checkpoint dir"
pip install -q google-cloud-storage 2>/dev/null || true
python3 - << 'PY'
import os
from google.cloud import storage

os.makedirs('/tmp/k3_tok', exist_ok=True)
client = storage.Client()
n = 0
for blob in client.list_blobs('tpu-commons-ci', prefix='moonshootai/kimi/k3/'):
    name = blob.name.rsplit('/', 1)[-1]
    if not name or name.endswith('.safetensors'):
        continue
    if (blob.size or 0) > 100 * 1024 * 1024:
        continue
    blob.download_to_filename(f'/tmp/k3_tok/{name}')
    n += 1
print(f"[k3-tok] downloaded {n} files: {sorted(os.listdir('/tmp/k3_tok'))}")
PY
tok_rc=$?
if [ "${tok_rc}" -ne 0 ]; then
  echo "[k3-tok] FAILED rc=${tok_rc} — bench needs the tokenizer and will" \
       "fail; accuracy uses tokenized_requests=False and can still run"
fi

# ---------------------------------------------------------------------------
# Phase 1: gsm8k accuracy against the live endpoint. limit 250 sets/checks the
# publish gate (strict-match > 0.85). Concurrency 8 = --max-num-seqs.
# ---------------------------------------------------------------------------
if [ -n "${SKIP_ACC}" ]; then
  echo "[k3-acc] gsm8k SKIPPED (--skip-acc; accuracy covered by a separate run)"
  acc_rc=0
else
  echo "[k3-acc] gsm8k ${GSM8K_DESC}, num_concurrent=8, cap $((ACC_CAP / 60)) min"
  # tokenizer_backend=None: build 761 showed the default huggingface backend
  # calls AutoTokenizer.from_pretrained on the model name, which is a gs://
  # path transformers cannot load. gsm8k is generation-only, so no client-side
  # tokenizer is needed (tokenized_requests=False sends plain text).
  timeout "${ACC_CAP}" python3 -m lm_eval \
    --model local-completions \
    --model_args "model=${MODEL},base_url=${BASE}/v1/completions,num_concurrent=8,max_retries=2,timeout=600,tokenized_requests=False,tokenizer_backend=None" \
    --tasks gsm8k ${GSM8K_LIMIT_ARGS[@]+"${GSM8K_LIMIT_ARGS[@]}"} \
    --log_samples --output_path /root/k3_gsm8k_samples \
    2>&1 | tee /root/k3_gsm8k.log
  acc_rc=${PIPESTATUS[0]}
  echo "[k3-acc] rc=${acc_rc} (124 means the cap killed a hang)"
  grep -E "strict-match|flexible-extract|exact_match" /root/k3_gsm8k.log \
    | tail -10 || true
  # Surface a handful of raw generations so a collapse is diagnosable
  # from the CI log (truncation vs garbage vs wrong-context answers).
  python3 - << 'PYEOF' || true
import glob, json
for f in glob.glob('/root/k3_gsm8k_samples/**/*.jsonl', recursive=True)[:1]:
    print(f'[k3-acc-samples] from {f}')
    with open(f) as fh:
        for i, line in enumerate(fh):
            if i >= 5: break
            d = json.loads(line)
            resp = str(d.get('resps') or d.get('filtered_resps') or '')[:400]
            tgt = str(d.get('target'))[:80]
            print(f'[k3-acc-samples] target={tgt} resp={resp}')
PYEOF
fi

run_bench() {
  local in_len="$1" out_len="$2" cap="$3" log="$4" nprompts="$5" conc="$6"
  echo "[k3-bench] random ${in_len}/${out_len}, ${nprompts} prompts, max-concurrency ${conc}, cap $((cap / 60)) min"
  timeout "${cap}" vllm bench serve \
    --base-url "${BASE}" \
    --model "${MODEL}" \
    --tokenizer "${TOK_DIR}" \
    --trust-remote-code \
    --dataset-name random \
    --random-input-len "${in_len}" \
    --random-output-len "${out_len}" \
    --num-prompts "${nprompts}" \
    --max-concurrency "${conc}" \
    --ignore-eos \
    2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  echo "[k3-bench] ${in_len}/${out_len} conc=${conc} rc=${rc} (124 means the timeout killed a hang)"
  return "${rc}"
}

# ---------------------------------------------------------------------------
# Phase 2a (--bs1-bench): single-request decode rate, 8k/1k at concurrency 1
# — the batch-size-1 shape public K3 serving numbers are quoted at. At BS=1
# the reported output-token throughput IS the per-request decode rate
# (1000 / TPOT-ms tok/s). 3 requests to average; runs even with --skip-bench.
# ---------------------------------------------------------------------------
bs1_rc=0
if [ -n "${BS1_BENCH}" ]; then
  run_bench 8192 1024 3600 /root/k3_bench_bs1.log 3 1
  bs1_rc=$?
fi

# ---------------------------------------------------------------------------
# Phase 2b: throughput at two shapes, 32 requests each, max-concurrency 8
# (= --max-num-seqs). ignore-eos forces the full output length per request.
# ---------------------------------------------------------------------------
if [ "${SKIP_BENCH}" = "--skip-bench" ]; then
  echo "[k3-bench] concurrency-8 suite SKIPPED (--skip-bench; numbers already recorded)"
  if [ "${acc_rc}" -ne 0 ] || [ "${bs1_rc}" -ne 0 ]; then
    echo "[k3-measure] FAILED acc_rc=${acc_rc} bs1_rc=${bs1_rc}"
    exit 1
  fi
  echo "[k3-measure] phases completed"
  exit 0
fi

if [ -n "${MATRIX_BENCH}" ]; then
  # 2x2 matrix: both shapes at moderate and saturating concurrency. 4 prompts
  # per concurrency slot so every combination measures a full steady state.
  run_bench 1024 1024 3600 /root/k3_bench_1k1k_c8.log 32 8
  bench_1k_c8_rc=$?
  run_bench 8192 1024 3600 /root/k3_bench_8k1k_c8.log 32 8
  bench_8k_c8_rc=$?
  run_bench 1024 1024 3600 /root/k3_bench_1k1k_c64.log 128 64
  bench_1k_c64_rc=$?
  run_bench 8192 1024 3600 /root/k3_bench_8k1k_c64.log 128 64
  bench_8k_c64_rc=$?
  if [ "${acc_rc}" -ne 0 ] || [ "${bs1_rc}" -ne 0 ] ||
     [ "${bench_1k_c8_rc}" -ne 0 ] || [ "${bench_8k_c8_rc}" -ne 0 ] ||
     [ "${bench_1k_c64_rc}" -ne 0 ] || [ "${bench_8k_c64_rc}" -ne 0 ]; then
    echo "[k3-measure] FAILED acc_rc=${acc_rc} bs1_rc=${bs1_rc}" \
         "1k1k_c8=${bench_1k_c8_rc} 8k1k_c8=${bench_8k_c8_rc}" \
         "1k1k_c64=${bench_1k_c64_rc} 8k1k_c64=${bench_8k_c64_rc}"
    exit 1
  fi
  echo "[k3-measure] all phases completed"
  exit 0
fi

run_bench 1024 1024 3600 /root/k3_bench_1k1k.log 32 8
bench_1k_rc=$?
run_bench 8192 1024 3600 /root/k3_bench_8k1k.log 32 8
bench_8k_rc=$?

if [ "${acc_rc}" -ne 0 ] || [ "${bs1_rc}" -ne 0 ] || [ "${bench_1k_rc}" -ne 0 ] || [ "${bench_8k_rc}" -ne 0 ]; then
  echo "[k3-measure] FAILED acc_rc=${acc_rc} bs1_rc=${bs1_rc} bench_1k1k_rc=${bench_1k_rc} bench_8k1k_rc=${bench_8k_rc}"
  exit 1
fi
echo "[k3-measure] all phases completed"
