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
# --skip-bench: run accuracy only (used when a prior run already produced the
# bench numbers for this exact serve config, e.g. build 761).
SKIP_BENCH="${1:-}"

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
echo "[k3-acc] gsm8k --limit 250, num_concurrent=8, cap 90 min"
# tokenizer_backend=None: build 761 showed the default huggingface backend
# calls AutoTokenizer.from_pretrained on the model name, which is a gs://
# path transformers cannot load. gsm8k is generation-only, so no client-side
# tokenizer is needed (tokenized_requests=False sends plain text).
timeout 5400 python3 -m lm_eval \
  --model local-completions \
  --model_args "model=${MODEL},base_url=${BASE}/v1/completions,num_concurrent=8,max_retries=2,timeout=600,tokenized_requests=False,tokenizer_backend=None" \
  --tasks gsm8k --limit 250 \
  2>&1 | tee /root/k3_gsm8k.log
acc_rc=${PIPESTATUS[0]}
echo "[k3-acc] rc=${acc_rc} (124 means the 90-min timeout killed a hang)"
grep -E "strict-match|flexible-extract|exact_match" /root/k3_gsm8k.log \
  | tail -10 || true

# ---------------------------------------------------------------------------
# Phase 2: 8k-in/1k-out throughput, 32 requests. max-concurrency 8 matches
# --max-num-seqs; the KV pool (256 blocks) holds 8 full 9k-token requests.
# ignore-eos forces the full 1024 output tokens per request.
# ---------------------------------------------------------------------------
if [ "${SKIP_BENCH}" = "--skip-bench" ]; then
  echo "[k3-bench] SKIPPED (--skip-bench; numbers already recorded for this config)"
  if [ "${acc_rc}" -ne 0 ]; then
    echo "[k3-measure] FAILED acc_rc=${acc_rc}"
    exit 1
  fi
  echo "[k3-measure] accuracy phase completed"
  exit 0
fi

echo "[k3-bench] random 8192/1024, 32 prompts, max-concurrency 8, cap 60 min"
timeout 3600 vllm bench serve \
  --base-url "${BASE}" \
  --model "${MODEL}" \
  --tokenizer "${TOK_DIR}" \
  --trust-remote-code \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --num-prompts 32 \
  --max-concurrency 8 \
  --ignore-eos \
  2>&1 | tee /root/k3_bench.log
bench_rc=${PIPESTATUS[0]}
echo "[k3-bench] rc=${bench_rc} (124 means the 60-min timeout killed a hang)"

if [ "${acc_rc}" -ne 0 ] || [ "${bench_rc}" -ne 0 ]; then
  echo "[k3-measure] FAILED acc_rc=${acc_rc} bench_rc=${bench_rc}"
  exit 1
fi
echo "[k3-measure] both phases completed"
