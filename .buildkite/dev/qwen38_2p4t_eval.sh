#!/bin/bash
# Accuracy leg for the Qwen3.8-2.4T-A95B-FP8 v7x-32 bringup.
#
# Runs after qwen38_2p4t_client.sh against the same live server: a 2.4 TB model
# takes ~35 min to load and ~20 min to precompile, so re-serving it per eval
# would cost more than the evals themselves.
#
# Three datasets, cheapest first, so a broken harness shows up in minutes rather
# than at the end:
#
#   GPQA-Diamond  198 q   benchmark_serving.py, chat template
#   GSM8K       1319 q    lm_eval local-completions, 8-shot CoT completion
#   MMLU        1000 of 14042   benchmark_serving.py, chat template
#
# Each eval is time-boxed and its failure is recorded rather than fatal -- a
# harness fault on the third eval should not throw away the first two, and the
# slice is too expensive to re-book for one bad regex. The exit status is
# non-zero only if every eval failed.
#
# Two things about this model drive the configuration below.
#
# Mandatory thinking. Its chat template raises on enable_thinking=false; the
# only knob is reasoning_effort, and only xhigh/medium/low are accepted. So
# every chat-templated response arrives wrapped in a reasoning block, output
# budgets have to clear that block before the answer, and reasoning_effort=low
# is what makes the token counts above affordable. GSM8K sidesteps it entirely
# by not applying a chat template: an 8-shot completion prompt injects no
# special tokens, so there is no template to raise and no block to clear, and
# it is also the configuration the published numbers use.
#
# Answer extraction. Both extractors in benchmark_utils.py take the first regex
# match in the completion, which for a reasoning model is its chain of thought
# -- a trace that weighs "(B)" before answering "(D)" scored as B. strip_reasoning
# there now drops the block first, and the eval reports unparsed_rate so a
# harness failure is distinguishable from a wrong model.
set -uo pipefail

MODEL="${SERVED_NAME:-Qwen/Qwen3.8-2.4T-A95B-FP8}"
PORT="${VLLM_PORT:-8000}"
ART="${ART_DIR:-/workspace/artifacts}"
BENCH_DIR="${BENCH_DIR:-/workspace/tpu_inference/scripts/vllm/benchmarking}"
DEV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA="${EVAL_DATA_DIR:-/root/eval-data}"
EVAL_DATA_URI="${EVAL_DATA_URI:-gs://tpu-commons-qwen38-2p4t-a95b-fp8/eval-data}"
mkdir -p "${ART}" "${DATA}"

# Sized in ../dev notes at ~500 aggregate output tok/s with reasoning_effort=low.
# The caps are ~3x the estimate: they exist to stop a pathological run eating
# the 480 min build, not to trim a slow one.
# Output budgets are sized against measured behaviour, not guessed. On the first
# run GPQA generated 1392 tokens per question on average against a 2048 cap, and
# 31 of 198 completions (15.7%) never closed their reasoning block -- the only
# way that happens with EOS honoured is hitting the cap mid-<think>, and
# strip_reasoning scores an unterminated block as unparsed, i.e. wrong. Grading
# a model on answers it was cut off before giving measures the cap, not the
# model. 6144 is ~4.4x the observed mean; unparsed_rate stays in the summary so
# the next run can say whether that was enough.
#
# Concurrency is per-eval because the KV pool, not the request count, is the
# binding constraint: a longer output budget has to buy itself a smaller batch.
# The defaults below are sized against the 660-block pool that hybrid sizing
# used to produce (660 x 128 = 84480 tokens); with mamba-aware block sizing the
# pool is ~23x that, and the binding constraint becomes max_num_reqs instead.
# Either way the kv-fit call in run_serving_eval checks the arithmetic against
# the live pool rather than against this comment.
GPQA_N="${GPQA_NUM_PROMPTS:-198}"        # the whole of GPQA-Diamond
GPQA_OUT="${GPQA_OUTPUT_LEN:-6144}"
GPQA_CONC="${GPQA_CONCURRENCY:-10}"      # 10 x 56 blocks = 560 of 660 (85%)
GPQA_TIMEOUT="${GPQA_TIMEOUT_S:-3600}"
MMLU_N="${MMLU_NUM_PROMPTS:-1000}"       # of 14042; the full set is ~8h of decode
MMLU_OUT="${MMLU_OUTPUT_LEN:-4096}"
MMLU_CONC="${MMLU_CONCURRENCY:-14}"      # 14 x 40 blocks = 560 of 660 (85%)
MMLU_TIMEOUT="${MMLU_TIMEOUT_S:-4800}"
GSM8K_LIMIT="${GSM8K_LIMIT:-0}"          # 0 = all 1319
GSM8K_TIMEOUT="${GSM8K_TIMEOUT_S:-3600}"
# GSM8K only: short completions over the raw endpoint, so it fits a wider batch.
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-24}"
REASONING_EFFORT="${EVAL_REASONING_EFFORT:-low}"

# Which evals to run, comma-separated. Isolating one variable -- reasoning
# effort, sampling, a KV sizing change -- wants a single leg, not all three:
# the other two cost slice time without informing the comparison.
EVAL_SUITE="${EVAL_SUITE:-gpqa,gsm8k,mmlu}"
wants() { [[ ",${EVAL_SUITE}," == *",$1,"* ]]; }
SYS_PROMPT="${EVAL_SYSTEM_PROMPT:-Answer with only the letter in parentheses, e.g. (A).}"

STATUS_JSON="${ART}/eval_summary.json"
declare -a RESULTS=()

record() {  # name status seconds note
  RESULTS+=("{\"eval\": \"$1\", \"status\": \"$2\", \"seconds\": $3, \"note\": \"$4\"}")
  printf '%s\n' "[eval] $1: $2 (${3}s) $4"
  # Rewritten after every eval, so a build killed mid-MMLU still uploads the
  # GPQA and GSM8K results.
  printf '{"model": "%s", "reasoning_effort": "%s", "results": [%s]}\n' \
    "${MODEL}" "${REASONING_EFFORT}" "$(IFS=,; echo "${RESULTS[*]}")" > "${STATUS_JSON}"
}

# ---------------------------------------------------------------------------
# Datasets.
#
# Staged in the same bucket as the weights rather than pulled from Hugging Face
# at eval time: MMLU's loader wants a mirror that has moved once already, and a
# 90-minute-old server is the wrong place to discover a dataset URL has rotted.
# SHA256SUMS is checked because a truncated download would otherwise surface as
# a mysteriously low score.
# ---------------------------------------------------------------------------
fetch() {
  if command -v gcloud >/dev/null 2>&1; then
    gcloud storage cp -r "${EVAL_DATA_URI}/*" "${DATA}/"
  elif command -v gsutil >/dev/null 2>&1; then
    gsutil -m cp -r "${EVAL_DATA_URI}/*" "${DATA}/"
  else
    python3 - "$EVAL_DATA_URI" "$DATA" <<'PY'
import sys
from google.cloud import storage
uri, dest = sys.argv[1], sys.argv[2]
bucket_name, _, prefix = uri[len("gs://"):].partition("/")
client = storage.Client()
for blob in client.list_blobs(bucket_name, prefix=prefix + "/"):
    name = blob.name.rsplit("/", 1)[-1]
    if name:
        blob.download_to_filename(f"{dest}/{name}")
        print(f"downloaded {name}")
PY
  fi
}

if ( cd "${DATA}" && sha256sum --status -c SHA256SUMS ) 2>/dev/null; then
  echo "--- eval datasets already in ${DATA}, checksums match"
else
  echo "--- eval datasets from ${EVAL_DATA_URI}"
  if ! fetch; then
    echo "[eval] FATAL: could not fetch eval datasets. Nothing to evaluate." >&2
    exit 1
  fi
  ( cd "${DATA}" && sha256sum -c SHA256SUMS ) || {
    echo "[eval] FATAL: dataset checksum mismatch, refusing to score against it." >&2
    exit 1
  }
fi
tar -xf "${DATA}/mmlu.tar" -C "${DATA}"
MMLU_PATH="${DATA}/data/test"
GPQA_PATH="${DATA}/gpqa_diamond.csv"
GSM8K_PATH="${DATA}/gsm8k_test.jsonl"
ls -la "${DATA}"; ls "${MMLU_PATH}" | head -3

# eval_accuracy_mmlu() calls nltk.download() and evaluate.load() *after* the
# expensive part, so a network hiccup there would discard a finished 20-minute
# run. Pay for both now, while nothing is at stake.
echo "--- pre-warm scoring dependencies"
python3 - <<'PY' || echo "[eval] WARNING: pre-warm failed; scoring may hit the network later"
import evaluate, nltk
nltk.download("punkt"); nltk.download("punkt_tab")
evaluate.load("accuracy")
print("[eval] evaluate+nltk ready")
PY

# Build #952 died here: lm_eval put 576 requests on a server sized for 64, the
# KV pool hit 99.8%, and the resulting preemption storm tripped a mamba
# slot-pool underflow in the runner that killed the engine outright -- taking
# the eval that was running and the one after it with it. So the shape check is
# no longer advisory-and-forget: every eval states its own shape before it runs,
# and a dead server is reported as such instead of being hammered.
server_up() { curl -sf -m 10 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# GPQA-Diamond and MMLU, over the OpenAI completions endpoint.
#
# No --ignore-eos, unlike the perf leg: the answer ends when the model stops,
# and forcing generation to the cap would bury it in continuation text.
# ---------------------------------------------------------------------------
run_serving_eval() {  # name path num-prompts timeout concurrency out-len extra...
  local name="$1" path="$2" n="$3" timeout_s="$4" conc="$5" out="$6"; shift 6
  local t0 rc secs
  if ! server_up; then
    record "${name}" "skipped" 0 "server not healthy"
    return
  fi
  echo "--- ${name} (n=${n}, concurrency=${conc}, out=${out}, reasoning_effort=${REASONING_EFFORT})"
  # 1024, not the ~291-token GPQA mean: the check is worth nothing if it is
  # optimistic about the prompt. Advisory -- an accuracy number is still correct
  # if the scheduler preempted to produce it, just slower.
  python3 "${DEV_DIR}/qwen38_2p4t_kv_fit.py" \
    --tokens-per-request $((1024 + out)) --concurrency "${conc}" \
    --block-size "${BLOCK_SIZE:-128}" --warn-only || true
  t0=$SECONDS
  # This fork of benchmark_serving.py has no --save-result; the accuracy dict
  # only ever reaches stdout, so the tee is the result file.
  timeout "${timeout_s}" python3 benchmark_serving.py \
    --backend vllm \
    --model "${MODEL}" \
    --host 127.0.0.1 --port "${PORT}" \
    --dataset-name "${name}" --dataset-path "${path}" \
    --num-prompts "${n}" \
    --max-concurrency "${conc}" \
    --request-rate inf --seed 42 \
    --chat-template-system-prompt "${SYS_PROMPT}" \
    --chat-template-kwargs "{\"reasoning_effort\": \"${REASONING_EFFORT}\"}" \
    --run-eval \
    "$@" 2>&1 | tee "${ART}/eval_${name}.log"
  rc=${PIPESTATUS[0]}
  secs=$((SECONDS - t0))
  if [ "${rc}" -eq 124 ]; then
    record "${name}" "timeout" "${secs}" "exceeded ${timeout_s}s"
  elif [ "${rc}" -ne 0 ]; then
    record "${name}" "failed" "${secs}" "exit ${rc}"
  else
    record "${name}" "ok" "${secs}" "$(scrape_accuracy "${name}")"
  fi
}

# The accuracy dict is python repr on stdout. Lifted out into its own json file
# so the numbers survive as data, not just as a line in a log, and folded into
# the summary note as `k=v` pairs.
scrape_accuracy() {
  python3 - "${ART}/eval_$1.log" "${ART}/eval_$1_accuracy.json" <<'PY'
import ast, json, re, sys
log, out = sys.argv[1], sys.argv[2]
text = open(log, errors="replace").read()
found = [m.group(0) for m in re.finditer(r"\{[^{}]*'accuracy'[^{}]*\}", text)]
if not found:
    print("no accuracy dict in output")
    sys.exit(0)
d = ast.literal_eval(found[-1])
json.dump(d, open(out, "w"), indent=2)
print(" ".join(f"{k}={v}" for k, v in d.items()))
PY
}

cd "${BENCH_DIR}" || exit 1

if wants gpqa; then
run_serving_eval gpqa "${GPQA_PATH}" "${GPQA_N}" "${GPQA_TIMEOUT}" \
  "${GPQA_CONC}" "${GPQA_OUT}" \
  --gpqa-use-chat-template --gpqa-output-len "${GPQA_OUT}"
fi

# ---------------------------------------------------------------------------
# GSM8K, via lm_eval against the same server.
#
# benchmark_serving.py has no GSM8K path at all -- it is not in --dataset-name
# and eval_benchmark_dataset_result raises on it -- so this goes through
# lm_eval's `local-completions` model, which speaks the same endpoint.
#
# gsm8k_cot, not gsm8k: its eight CoT exemplars are embedded in the task yaml,
# so it needs no train split and matches what tests/e2e/check_lm_eval.sh
# already runs for the other MoE models here. No --apply_chat_template, so
# nothing injects a reasoning block, and flexible-extract (last number in the
# completion) is robust to one anyway. strict-match wants a literal "#### N"
# and the exemplars teach it, so both numbers are meaningful.
#
# The task is repointed at the staged copy of the test split so the eval needs
# no Hugging Face round trip; if that rewrite does not apply cleanly to
# whatever lm_eval version the image has, fall back to the stock task rather
# than lose the eval.
# ---------------------------------------------------------------------------
if wants gsm8k; then
echo "--- gsm8k (lm_eval local-completions, 8-shot CoT)"
GSM8K_TASK="gsm8k_cot"
INCLUDE_DIR="${DATA}/lm_eval_tasks"
if python3 - "${GSM8K_PATH}" "${INCLUDE_DIR}" <<'PY'
import os, sys, yaml, lm_eval.tasks
data_path, out_dir = sys.argv[1], sys.argv[2]
src = os.path.join(os.path.dirname(lm_eval.tasks.__file__), "gsm8k", "gsm8k-cot.yaml")
cfg = yaml.safe_load(open(src))
cfg["task"] = "gsm8k_cot_local"
cfg["dataset_path"] = "json"
cfg["dataset_name"] = None
cfg["dataset_kwargs"] = {"data_files": {"test": data_path}}
cfg["test_split"] = "test"
for k in ("training_split", "validation_split", "fewshot_split"):
    cfg.pop(k, None)
os.makedirs(out_dir, exist_ok=True)
yaml.safe_dump(cfg, open(os.path.join(out_dir, "gsm8k_cot_local.yaml"), "w"))
print("[eval] gsm8k task repointed at", data_path)
PY
then
  GSM8K_TASK="gsm8k_cot_local"
  GSM8K_INCLUDE=(--include_path "${INCLUDE_DIR}")
else
  echo "[eval] WARNING: could not repoint gsm8k at the staged split; using the stock task (needs Hugging Face)"
  GSM8K_INCLUDE=()
fi

# tokenizer_backend defaults to "auto", which probes the server for /tokenize
# and uses it -- so nothing here needs a Hugging Face round trip. Naming the
# local checkpoint as the tokenizer anyway covers the case where that probe
# fails and lm_eval falls back to loading one itself.
#
# max_gen_toks 256 -> 512 and max_length 2048 -> 4096: the 8-shot CoT prompt is
# ~1000 tokens on its own, and lm_eval's defaults leave no room for a model that
# reasons before answering. Truncation there reads as a wrong answer.
GSM8K_MODEL_ARGS="model=${MODEL}"
GSM8K_MODEL_ARGS+=",base_url=http://127.0.0.1:${PORT}/v1/completions"
GSM8K_MODEL_ARGS+=",num_concurrent=${EVAL_CONCURRENCY},max_retries=3"
GSM8K_MODEL_ARGS+=",tokenized_requests=False,max_gen_toks=512,max_length=4096"
if [ -n "${MODEL_PATH:-}" ] && [ -d "${MODEL_PATH:-}" ]; then
  GSM8K_MODEL_ARGS+=",tokenizer=${MODEL_PATH}"
fi

GSM8K_ARGS=(
  --model local-completions
  --model_args "${GSM8K_MODEL_ARGS}"
  --tasks "${GSM8K_TASK}"
  --num_fewshot 8
  # batch_size 1, NOT EVAL_CONCURRENCY. lm_eval's local-completions puts
  # batch_size prompts in one request body and runs num_concurrent of those
  # bodies at once, so the two multiply: build #952 asked for 24 and 24 and got
  # 24 x 24 = 576 requests in flight (observed: 62 running, 514 waiting, KV at
  # 99.8%), which oversubscribed the pool badly enough to kill the engine. With
  # batch_size 1, num_concurrent is the real in-flight cap.
  --batch_size 1
  --output_path "${ART}/eval_gsm8k"
  "${GSM8K_INCLUDE[@]}"
)
[ "${GSM8K_LIMIT}" != "0" ] && GSM8K_ARGS+=(--limit "${GSM8K_LIMIT}")

if ! server_up; then
  record gsm8k "skipped" 0 "server not healthy"
  rc=0 secs=0
else
# The 8-shot CoT prompt is ~1000 tokens and max_gen_toks is 512, so this is the
# cheapest shape of the three -- but it is the one that broke #952, so it states
# its budget like the others.
python3 "${DEV_DIR}/qwen38_2p4t_kv_fit.py" \
  --tokens-per-request $((1536 + 512)) --concurrency "${EVAL_CONCURRENCY}" \
  --block-size "${BLOCK_SIZE:-128}" --warn-only || true
t0=$SECONDS
timeout "${GSM8K_TIMEOUT}" lm_eval "${GSM8K_ARGS[@]}" 2>&1 | tee "${ART}/eval_gsm8k.log"
rc=${PIPESTATUS[0]}
secs=$((SECONDS - t0))
if [ "${rc}" -eq 124 ]; then
  record gsm8k "timeout" "${secs}" "exceeded ${GSM8K_TIMEOUT}s"
elif [ "${rc}" -ne 0 ]; then
  record gsm8k "failed" "${secs}" "exit ${rc}"
else
  # awk over the results table, the same shape check_lm_eval.sh does.
  flex=$(grep "flexible-extract" "${ART}/eval_gsm8k.log" | awk -F'|' '{print $8}' | xargs)
  strict=$(grep "strict-match" "${ART}/eval_gsm8k.log" | awk -F'|' '{print $8}' | xargs)
  record gsm8k "ok" "${secs}" "flexible-extract=${flex:-?} strict-match=${strict:-?}"
fi
fi
fi

# MMLU last: the longest of the three, and the one whose partial loss costs
# least because the other two have already landed.
if wants mmlu; then
run_serving_eval mmlu "${MMLU_PATH}" "${MMLU_N}" "${MMLU_TIMEOUT}" \
  "${MMLU_CONC}" "${MMLU_OUT}" \
  --mmlu-use-chat-template --mmlu-output-len "${MMLU_OUT}" --mmlu-num-shots 0
fi

echo "--- eval summary"
cat "${STATUS_JSON}"; echo
if ! grep -q '"status": "ok"' "${STATUS_JSON}"; then
  echo "[eval] FAILED: no eval completed" >&2
  exit 1
fi
