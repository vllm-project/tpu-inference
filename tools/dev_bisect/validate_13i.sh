#!/bin/bash
# Copyright 2026 Google LLC
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
#
# Validates the attempt-13i instrumentation on the CI stack: the wrong-answer
# dump script (unit fixture), a served slice at pod topology, the HTTP
# teacher-forced scoring probe, and the CHURN_TRACE step logging.
set -e
export MODEL_IMPL_TYPE=flax_nnx NEW_MODEL_DESIGN=1
export VLLM_XLA_CHECK_RECOMPILATION=0
export TPU_INFERENCE_UNSAFE_MLA_WITHOUT_DP_ATTENTION=1
export KV_CACHE_WARMUP_RESERVE_FRACTION=0
export MXFP4_SHARD_THEN_DECODE=1
export CHURN_TRACE=1 CHURN_TRACE_EVERY=20
cd /workspace/tpu_inference

echo "=== [1/3] dump-script unit check ==="
mkdir -p /tmp/fake_lmeval
python3 - << 'PYEOF'
import json
rows = []
for i in range(6):
    ok = i in (0, 3)
    rows.append({
        "doc_id": i,
        "doc": {"question": f"Q{i}?", "answer": f"#### {2*i}"},
        "target": f"#### {2*i}",
        "resps": [[("right " if ok else "wrong ") * 30
                   + f"#### {2*i if ok else 999}"]],
        "filtered_resps": [str(2*i if ok else 999)],
        "exact_match": 1.0 if ok else 0.0,
    })
with open("/tmp/fake_lmeval/samples_gsm8k_x.jsonl", "w") as f:
    f.write("\n".join(json.dumps(r) for r in rows))
PYEOF
python3 tests/e2e/multihost/k3_dump_wrong_samples.py /tmp/fake_lmeval | tail -3

echo "=== [2/3] serve the slice (pod topology) ==="
vllm serve gs://tpu-commons-ci/moonshootai/kimi/k3-sliced \
  --port 8256 --tensor-parallel-size 8 --trust-remote-code \
  --max-model-len 2048 --max-num-seqs 64 --max-num-batched-tokens 512 \
  --no-async-scheduling --load-format=runai_streamer \
  --no-enable-prefix-caching --gpu-memory-utilization 0.8 \
  --additional-config '{"sharding":{"sharding_strategy":{"enable_dp_attention":false,"tensor_parallelism":2,"expert_parallelism":4}}}' \
  > /tmp/serve.log 2>&1 &
SERVE_PID=$!
for _ in $(seq 1 240); do
  if curl -sf http://localhost:8256/health > /dev/null 2>&1; then break; fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "SERVER DIED:"; tail -40 /tmp/serve.log; exit 1
  fi
  sleep 10
done
curl -sf http://localhost:8256/health > /dev/null \
  || { echo "health timeout"; tail -40 /tmp/serve.log; exit 1; }

echo "=== [3/3] scoring probe over HTTP + trace check ==="
SCORING_BASE=http://localhost:8256 \
  SCORING_MODEL=gs://tpu-commons-ci/moonshootai/kimi/k3-sliced \
  SCORING_WAVES=5 python3 tests/e2e/multihost/k3_server_scoring_probe.py

echo "=== churn-trace lines in the serve log: ==="
grep -c "churn-trace" /tmp/serve.log || { echo "NO TRACE LINES"; exit 1; }
grep -m 3 "churn-trace" /tmp/serve.log
kill "$SERVE_PID" 2>/dev/null || true
echo "[validate-13i] ALL CHECKS PASSED"
