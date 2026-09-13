#!/bin/bash
# Standard-harness accuracy leg for the Qwen3.8-2.4T-A95B-FP8 v7x-32 bringup.
#
# Companion to qwen38_2p4t_eval.sh, deliberately not a replacement. That script
# scores GPQA and MMLU through this repo's fork of benchmark_serving.py, whose
# prompt template, answer extractor and subsetting are ours alone; its numbers
# are comparable across our builds and to nothing else. This one runs the same
# three datasets through lm-evaluation-harness on the stock task definitions, so
# the results can be set beside a GPU run of the same checkpoint. Keep both:
# when a score moves the first question is whether the model moved or the
# harness did, and that is only answerable while both harnesses still run.
#
# Target configuration, i.e. what the GPU side ran:
#
#   python3 -m lm_eval --model local-chat-completions --apply_chat_template \
#     --tasks gsm8k --model_args ...,num_concurrent=64,tokenized_requests=False,\
#     max_length=16384 --gen_kwargs max_tokens=1024,temperature=0,top_p=1.0
#   python3 -m lm_eval --model local-completions \
#     --tasks mmlu --model_args ...,num_concurrent=64
#
# Note what is *absent* from both: --num_fewshot. gsm8k carries num_fewshot: 5
# in its own yaml so it is 5-shot; mmlu's _default_template_yaml carries none,
# so the GPU MMLU run was 0-shot, not the 5-shot number usually published under
# that name. This script reproduces that, because matching the GPU run is the
# point. Set MMLU_NUM_FEWSHOT=5 to get the canonical number instead, and then
# do not compare it to the GPU one.
#
# Deviations from stock lm_eval, all of them forced and all of them here:
#
#   gpqa   dataset_path Idavidrein/gpqa -> the staged gpqa_diamond.csv. The HF
#          dataset is gated; the staged file is the verbatim diamond CSV from
#          that repo, same columns, so process_docs is untouched.
#   gpqa   generation_kwargs gains chat_template_kwargs {reasoning_effort},
#          max_gen_toks, and the sampling set selected by GPQA_SAMPLING (default
#          the model card's temperature=1.0/top_p=0.95/top_k=20, replacing
#          stock's temperature=0). Nothing else in the task changes.
#   gsm8k  generation_kwargs gains chat_template_kwargs {reasoning_effort}.
#          Skipped entirely, leaving the stock task, when the effort is "".
#   mmlu   unchanged. It is loglikelihood, so there is no generation to think
#          during and no chat template to carry a thinking level.
#
# Both overrides are expressed as `include:` of the stock yaml plus the changed
# keys, so everything not listed above tracks whatever lm_eval the image ships
# rather than a copy that silently rots.
#
# Reproducibility: lm_eval calls random.seed(0) before building tasks
# (evaluator.py:200), so GPQA's process_docs shuffle -- which reads the global
# random module -- is deterministic run to run. No seeding patch is needed, and
# unlike the MMLU leg in the other script this one is reproducible across hosts.
#
# Three legs, cheapest first, each time-boxed and each failing on its own so a
# harness fault in the third does not throw away the first two. Exit status is
# non-zero only if every leg failed.
set -uo pipefail

MODEL="${SERVED_NAME:-Qwen/Qwen3.8-2.4T-A95B-FP8}"
PORT="${VLLM_PORT:-8000}"
ART="${ART_DIR:-/workspace/artifacts}"
DEV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA="${EVAL_DATA_DIR:-/root/eval-data}"
EVAL_DATA_URI="${EVAL_DATA_URI:-gs://tpu-commons-qwen38-2p4t-a95b-fp8/eval-data}"
TASK_DIR="${DATA}/lm_eval_tasks_std"
mkdir -p "${ART}" "${DATA}" "${TASK_DIR}"

# Which legs to run, comma-separated. Isolating one variable rarely wants all
# three: the other two cost slice time without informing the comparison.
STD_SUITE="${STD_EVAL_SUITE:-gpqa,gsm8k,mmlu}"
wants() { [[ ",${STD_SUITE}," == *",$1,"* ]]; }

# Reasoning effort. This model's chat template raises on enable_thinking=false,
# so the only knob is reasoning_effort and only xhigh/medium/low are accepted.
# The split below is the one the fork-based script established: xhigh for GPQA
# (what the 0.899 baseline was measured at), low for the cheap legs. Set either
# to "" to send no chat_template_kwargs at all and take the model's default,
# which for GSM8K is exact parity with the GPU command above.
GPQA_EFFORT="${GPQA_REASONING_EFFORT:-xhigh}"
GSM8K_EFFORT="${GSM8K_REASONING_EFFORT:-low}"

# GPQA sampling. "spec" is the Qwen3.8 model card's Best Practices set
# (temperature=1.0, top_p=0.95, top_k=20, min_p=0.0); "greedy" is temperature=0,
# which is what #20 and every fork run before it used.
#
# Greedy is off-spec for this model and the card names "endless repetition" as
# the failure it causes, pointing at presence_penalty as the remedy. #20 is
# consistent with that: 21/198 generations ran to the 32768-token wall and
# scored 0.143, dragging 0.881 down to 0.803, while the 177 that finished
# averaged only 8120 tokens. So the runaway tail, not the model, is what the
# stock harness is currently measuring, and sampling is the one thing we are
# doing that Qwen explicitly advises against.
#
# lm_eval's local-chat-completions pins seed=1234 in every payload
# (openai_completions.py:206), so temperature>0 is still reproducible against a
# fixed server build. top_k/min_p are not OpenAI fields but vLLM accepts them on
# ChatCompletionRequest, and _create_payload splats unrecognised gen_kwargs
# straight into the body.
GPQA_SAMPLING="${GPQA_SAMPLING:-spec}"
case "${GPQA_SAMPLING}" in
  spec|greedy) ;;
  *) echo "[std-eval] GPQA_SAMPLING must be spec or greedy, got '${GPQA_SAMPLING}'" >&2
     exit 2 ;;
esac

# Concurrency is the number of requests actually in flight, and it has to stay
# under MAX_NUM_SEQS (80 on this deployment). Build tc#952 died from getting
# this wrong: lm_eval's local-completions puts batch_size prompts in one request
# body and runs num_concurrent bodies at once, so the two MULTIPLY. 24 and 24
# became 576 in flight, the KV pool hit 99.8%, and the preemption storm tripped
# a mamba slot-pool underflow that killed the engine outright. Every leg below
# states batch_size x num_concurrent explicitly and keeps the product <= 64.
GPQA_CONC="${GPQA_CONCURRENCY:-32}"
GPQA_MAX_GEN="${GPQA_MAX_GEN_TOKS:-32768}"
GPQA_TIMEOUT="${GPQA_TIMEOUT_S:-10800}"
GPQA_LIMIT="${GPQA_LIMIT:-0}"           # 0 = all 198

GSM8K_CONC="${GSM8K_CONCURRENCY:-64}"   # batch_size forced to 1 by chat completions
GSM8K_MAX_TOKENS="${GSM8K_MAX_TOKENS:-1024}"  # the GPU run's value; see below
GSM8K_TIMEOUT="${GSM8K_TIMEOUT_S:-7200}"
GSM8K_LIMIT="${GSM8K_LIMIT:-0}"         # 0 = all 1319

# MMLU multiple_choice emits one request PER CHOICE, so the full set is
# 4 x 14042 = 56168 requests -- but each is a prompt plus max_tokens=1, so the
# cost is prefill, not decode, and batching them 8 to a body is what keeps the
# request count from dominating. 8 x 8 = 64 prompts in flight.
MMLU_BATCH="${MMLU_BATCH_SIZE:-8}"
MMLU_CONC="${MMLU_CONCURRENCY:-8}"
MMLU_TIMEOUT="${MMLU_TIMEOUT_S:-14400}"
MMLU_LIMIT="${MMLU_LIMIT:-0}"           # 0 = all 14042 per subject
MMLU_NUM_FEWSHOT="${MMLU_NUM_FEWSHOT:-}"  # empty = stock (0-shot), as the GPU run

MAX_LEN="${EVAL_MAX_LENGTH:-40960}"     # matches MAX_MODEL_LEN

STATUS_JSON="${ART}/eval_std_summary.json"
declare -a RESULTS=()

if [ -z "${STD_SUITE}" ] || [ "${STD_SUITE}" = "none" ]; then
  echo "--- standard eval leg skipped (STD_EVAL_SUITE='${STD_SUITE}')"
  printf '{"model": "%s", "harness": "lm_eval", "skipped": true, "results": []}\n' \
    "${MODEL}" > "${STATUS_JSON}"
  exit 0
fi

record() {  # name status seconds note
  RESULTS+=("{\"eval\": \"$1\", \"status\": \"$2\", \"seconds\": $3, \"note\": \"$4\"}")
  printf '%s\n' "[std-eval] $1: $2 (${3}s) $4"
  # Rewritten after every leg, so a build killed mid-MMLU still uploads GPQA.
  printf '{"model": "%s", "harness": "lm_eval", "results": [%s]}\n' \
    "${MODEL}" "$(IFS=,; echo "${RESULTS[*]}")" > "${STATUS_JSON}"
}

server_up() { curl -sf -m 10 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; }

echo "--- lm_eval version"
python3 -c "import lm_eval; print('lm_eval', lm_eval.__version__)" || {
  echo "[std-eval] FATAL: lm_eval not importable" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Datasets.
#
# GPQA comes from the weights bucket because Idavidrein/gpqa is gated. GSM8K and
# MMLU come from Hugging Face because they are not, and because pulling them
# from anywhere else is a deviation from the GPU run that buys nothing. They are
# fetched here, before any eval, rather than lazily: discovering a dataset
# problem 40 minutes into a leg wastes slice time on a 2.4 TB server that took
# 35 minutes to load.
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

GPQA_PATH="${DATA}/gpqa_diamond.csv"
if wants gpqa; then
  if ( cd "${DATA}" && sha256sum --status -c SHA256SUMS ) 2>/dev/null; then
    echo "--- eval datasets already in ${DATA}, checksums match"
  else
    echo "--- eval datasets from ${EVAL_DATA_URI}"
    fetch || echo "[std-eval] WARNING: dataset fetch reported an error"
    ( cd "${DATA}" && sha256sum -c SHA256SUMS ) || {
      echo "[std-eval] WARNING: dataset checksum mismatch"; }
  fi
  # The staged file must be the real diamond CSV, not a reshaped export: stock
  # process_docs indexes these four columns by name and a KeyError here would
  # otherwise surface 40 minutes later.
  python3 - "${GPQA_PATH}" <<'PY' || { echo "[std-eval] GPQA csv unusable"; }
import csv, sys
path = sys.argv[1]
need = {"Question", "Correct Answer", "Incorrect Answer 1",
        "Incorrect Answer 2", "Incorrect Answer 3"}
with open(path, newline="", encoding="utf-8") as fh:
    r = csv.DictReader(fh)
    missing = need - set(r.fieldnames or [])
    n = sum(1 for _ in r)
assert not missing, f"gpqa csv missing columns: {sorted(missing)}"
print(f"[std-eval] gpqa csv ok: {n} rows, columns verified")
PY
fi

# ---------------------------------------------------------------------------
# Task staging.
#
# Both overrides are `include:` of the stock yaml plus the keys that change.
# lm_eval resolves an absolute include path directly and lets local keys win
# (tasks/_yaml_loader.py:193-207), and it resolves !function tags against the
# directory of the file that wrote them -- so GPQA's process_docs still comes
# from the stock gpqa/cot_zeroshot/utils.py, not from anything staged here.
#
# generation_kwargs is replaced wholesale rather than merged, so each override
# restates the stock until/do_sample/temperature verbatim alongside its addition.
# ---------------------------------------------------------------------------
stage_tasks() {
  python3 - "${TASK_DIR}" "${GPQA_PATH}" "${GPQA_EFFORT}" "${GPQA_MAX_GEN}" \
           "${GSM8K_EFFORT}" "${GPQA_SAMPLING}" <<'PY'
import os, sys, yaml, lm_eval.tasks

(task_dir, gpqa_csv, gpqa_effort, gpqa_max_gen, gsm8k_effort,
 gpqa_sampling) = sys.argv[1:7]
root = os.path.dirname(lm_eval.tasks.__file__)
os.makedirs(task_dir, exist_ok=True)

gpqa_stock = os.path.join(root, "gpqa", "cot_zeroshot", "_gpqa_cot_zeroshot_yaml")
gsm8k_stock = os.path.join(root, "gsm8k", "gsm8k.yaml")
for p in (gpqa_stock, gsm8k_stock):
    if not os.path.isfile(p):
        raise SystemExit(f"stock task yaml not found: {p}")

gpqa_gen = {"until": ["</s>"], "max_gen_toks": int(gpqa_max_gen)}
if gpqa_sampling == "greedy":
    gpqa_gen.update({"do_sample": False, "temperature": 0.0})
else:
    # Qwen3.8 card, Best Practices. do_sample is dropped by _create_payload
    # before the request is built, so it is documentation only either way.
    gpqa_gen.update({"do_sample": True, "temperature": 1.0, "top_p": 0.95,
                     "top_k": 20, "min_p": 0.0})
if gpqa_effort:
    gpqa_gen["chat_template_kwargs"] = {"reasoning_effort": gpqa_effort}
gpqa = {
    "include": gpqa_stock,
    "task": "gpqa_diamond_cot_zeroshot_std",
    # The include carries tag: gpqa, which would re-register the stock tag
    # group to point at this task ("Task 'gpqa' overrides existing task").
    # Nothing here resolves tasks by tag, but the warning invites a misreading
    # of the log, so give the override its own tag.
    "tag": "gpqa_std",
    # Stock resolves docs from validation_split (= "train"), test_split being
    # null, so the CSV is loaded as the train split to keep that resolution.
    "dataset_path": "csv",
    "dataset_name": None,
    "dataset_kwargs": {"data_files": {"train": gpqa_csv}},
    "generation_kwargs": gpqa_gen,
}
with open(os.path.join(task_dir, "gpqa_diamond_cot_zeroshot_std.yaml"), "w") as fh:
    yaml.safe_dump(gpqa, fh, sort_keys=False)
print("[std-eval] staged gpqa_diamond_cot_zeroshot_std ->", gpqa_csv,
      f"(reasoning_effort={gpqa_effort or 'model default'}, max_gen_toks={gpqa_max_gen},"
      f" sampling={gpqa_sampling}: "
      + ", ".join(f"{k}={gpqa_gen[k]}" for k in
                  ("temperature", "top_p", "top_k", "min_p") if k in gpqa_gen)
      + ")")

if gsm8k_effort:
    gsm8k = {
        "include": gsm8k_stock,
        "task": "gsm8k_std",
        "tag": "math_word_problems_std",   # same reason as gpqa above
        "generation_kwargs": {
            "until": ["Question:", "</s>", "<|im_end|>"],
            "do_sample": False,
            "temperature": 0.0,
            "chat_template_kwargs": {"reasoning_effort": gsm8k_effort},
        },
    }
    with open(os.path.join(task_dir, "gsm8k_std.yaml"), "w") as fh:
        yaml.safe_dump(gsm8k, fh, sort_keys=False)
    print(f"[std-eval] staged gsm8k_std (reasoning_effort={gsm8k_effort})")
else:
    print("[std-eval] gsm8k: stock task, no chat_template_kwargs (GPU parity)")
PY
}

GPQA_TASK="gpqa_diamond_cot_zeroshot_std"
GSM8K_TASK="gsm8k_std"
if stage_tasks; then
  [ -n "${GSM8K_EFFORT}" ] || GSM8K_TASK="gsm8k"
else
  echo "[std-eval] WARNING: task staging failed; gpqa will be skipped and gsm8k falls back to stock"
  GPQA_TASK=""
  GSM8K_TASK="gsm8k"
fi

# Warm the HF cache for the two ungated datasets. Non-fatal: the leg that needs
# a missing dataset will fail on its own and say so.
echo "--- prefetch Hugging Face datasets"
python3 - <<'PY' || echo "[std-eval] WARNING: HF prefetch incomplete; a leg may fail on download"
import datasets, lm_eval.tasks, os, yaml
datasets.load_dataset("openai/gsm8k", "main")
print("[std-eval] openai/gsm8k ready")
root = os.path.join(os.path.dirname(lm_eval.tasks.__file__), "mmlu", "default")
subjects = []
for name in sorted(os.listdir(root)):
    if not name.endswith(".yaml") or name.startswith("_"):
        continue
    cfg = yaml.safe_load(open(os.path.join(root, name)))
    if cfg.get("dataset_name"):
        subjects.append(cfg["dataset_name"])
for i, subj in enumerate(subjects, 1):
    datasets.load_dataset("cais/mmlu", subj)
print(f"[std-eval] cais/mmlu ready ({len(subjects)} subjects)")
PY

# ---------------------------------------------------------------------------
# Scoring.
#
# lm_eval writes results_<ts>.json under output_path; read that rather than
# awk-ing the pretty-printed table, so the numbers survive as data and a
# renamed column does not silently produce "?".
#
# The samples file gives the diagnostic the table does not: what fraction of
# generations the filters could not extract an answer from. A reasoning model
# cut off mid-<think> scores as wrong, and a low score caused by a too-small
# token budget is indistinguishable from a bad model unless this is reported.
# ---------------------------------------------------------------------------
scrape() {  # output_dir
  python3 - "$1" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
res = sorted(glob.glob(os.path.join(out, "**", "results_*.json"), recursive=True),
             key=os.path.getmtime)
if not res:
    print("no results json"); raise SystemExit(0)
data = json.load(open(res[-1]))
parts = []
SKIP = ("alias", "sample_len")  # sample_len is a doc count, not a score
for task, metrics in sorted(data.get("results", {}).items()):
    for k, v in sorted(metrics.items()):
        # "_stderr," not endswith("_stderr,none"): a task with named filters
        # reports exact_match_stderr,flexible-extract, which that misses.
        if k in SKIP or "_stderr," in k or not isinstance(v, (int, float)):
            continue
        parts.append(f"{task}:{k}={v:.4f}")
# Only the aggregate for a 57-task group, else the note is unreadable.
if len(parts) > 8:
    groups = data.get("groups", {})
    agg = [f"{g}:{k}={v:.4f}" for g, m in sorted(groups.items())
           for k, v in sorted(m.items())
           if isinstance(v, (int, float)) and k not in SKIP and "_stderr," not in k]
    parts = agg or parts[:8] + [f"(+{len(parts) - 8} more, see results json)"]

# Per filter, not pooled. lm_eval writes one samples row per (doc, filter), so
# a two-filter task like gpqa contributes 396 rows for 198 questions. Pooling
# them mixes a filter that extracts nothing (strict-match, 197/198 invalid) with
# one that extracts everything, and #20 duly reported 0.4975 -- a number that
# describes neither filter. Truncation is counted separately because on a
# reasoning model it is the usual reason a score is low, and it is invisible in
# unparsed_rate: a generation cut off mid-<think> still contains some "(X)" for
# flexible-extract to find, so it scores wrong rather than unparsed.
from collections import defaultdict
inv = defaultdict(int); tot = defaultdict(int); trunc = defaultdict(int)
for sf in glob.glob(os.path.join(out, "**", "samples_*.jsonl"), recursive=True):
    for line in open(sf, errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        fr = d.get("filtered_resps")
        if not fr:
            continue
        f = d.get("filter", "none")
        tot[f] += 1
        if any(isinstance(x, str) and "invalid" in x for x in fr):
            inv[f] += 1
        # Qwen3.8's template pre-fills the opening <think>, so the completion
        # carries the closing tag and never the opening one. Missing </think>
        # means the generation ran out of budget mid-thought.
        r = (d.get("resps") or [[""]])[0]
        r = r[0] if isinstance(r, list) else r
        if isinstance(r, str) and "<think" not in r and "</think>" not in r:
            trunc[f] += 1
for f in sorted(tot):
    parts.append(f"unparsed_rate[{f}]={inv[f] / tot[f]:.4f}")
    parts.append(f"truncated[{f}]={trunc[f] / tot[f]:.4f}({trunc[f]}/{tot[f]})")
print(" ".join(parts))
PY
}

# ---------------------------------------------------------------------------
# GPQA-Diamond: stock gpqa_diamond_cot_zeroshot, chat completions, thinking on.
#
# 198 questions, 0-shot, generate_until. Two filters are reported. strict-match
# wants the literal "The answer is (X)", which the stock prompt never asks for,
# so it reads low by construction; flexible-extract takes the LAST "(X)" in the
# completion (group_select: -1) and is the number to compare. Taking the last
# match is also what makes this safe against the reasoning block: no reasoning
# parser is configured on the server, so the whole chain of thought arrives in
# message.content, and an extractor that took the first match would score the
# model on whatever option it considered first.
# ---------------------------------------------------------------------------
run_leg() {  # name task model_kind timeout_s lm_eval_args...
  local name="$1" task="$2" kind="$3" timeout_s="$4"; shift 4
  local t0 rc secs out
  if [ -z "${task}" ]; then
    record "${name}" "skipped" 0 "task unavailable"
    return
  fi
  if ! server_up; then
    record "${name}" "skipped" 0 "server not healthy"
    return
  fi
  out="${ART}/eval_std_${name}"
  echo "--- ${name} (lm_eval ${kind}, task=${task}, timeout=${timeout_s}s)"
  t0=$SECONDS
  timeout "${timeout_s}" lm_eval "$@" 2>&1 | tee "${ART}/eval_std_${name}.log"
  rc=${PIPESTATUS[0]}
  secs=$((SECONDS - t0))
  if [ "${rc}" -eq 124 ]; then
    record "${name}" "timeout" "${secs}" "exceeded ${timeout_s}s"
  elif [ "${rc}" -ne 0 ]; then
    record "${name}" "failed" "${secs}" "exit ${rc}"
  else
    record "${name}" "ok" "${secs}" "$(scrape "${out}")"
  fi
}

# tokenizer_backend is left at its default "auto", which probes the server for
# /tokenize and uses it if present -- the same thing the GPU run did. Naming the
# local checkpoint as the tokenizer covers the fallback, where lm_eval would
# otherwise pull it from the Hub under the served name.
#
# tokenized_requests is NOT set here, because the right value differs by leg and
# getting it wrong fails in opposite directions. The two generative legs pass
# False (below): it matches the GPU command and it disables lm_eval's
# client-side length check, so nothing is left-truncated behind our back. MMLU
# must not, because loglikelihood scoring needs a token count for the context to
# know where the continuation starts -- _loglikelihood_tokens asserts
# `self.tokenizer is not None` (api_models.py:634) and parse_logprobs slices
# token_logprobs[ctxlen:-1]. Passing False there aborts the leg on that assert.
common_model_args() {  # endpoint
  local a="model=${MODEL},base_url=http://127.0.0.1:${PORT}/v1/$1"
  a+=",api_key=EMPTY,max_retries=3,timeout=1800"
  if [ -n "${MODEL_PATH:-}" ] && [ -d "${MODEL_PATH:-}" ]; then
    a+=",tokenizer=${MODEL_PATH}"
  fi
  printf '%s' "$a"
}

if wants gpqa; then
  python3 "${DEV_DIR}/qwen38_2p4t_kv_fit.py" \
    --tokens-per-request $((1024 + GPQA_MAX_GEN)) --concurrency "${GPQA_CONC}" \
    --block-size "${BLOCK_SIZE:-128}" --warn-only || true
  GPQA_ARGS=(
    --model local-chat-completions
    --model_args "$(common_model_args chat/completions),tokenized_requests=False,num_concurrent=${GPQA_CONC},max_gen_toks=${GPQA_MAX_GEN},max_length=${MAX_LEN}"
    --tasks "${GPQA_TASK}"
    --apply_chat_template
    --include_path "${TASK_DIR}"
    --batch_size 1
    --log_samples
    --output_path "${ART}/eval_std_gpqa"
  )
  [ "${GPQA_LIMIT}" != "0" ] && GPQA_ARGS+=(--limit "${GPQA_LIMIT}")
  run_leg gpqa "${GPQA_TASK}" local-chat-completions "${GPQA_TIMEOUT}" "${GPQA_ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# GSM8K: stock gsm8k, 5-shot, chat completions.
#
# Standard gsm8k, not gsm8k_cot: its five exemplars are sampled from the train
# split (which is why that split is downloaded at all) and its strict-match
# wants the literal "#### N" the train answers teach. This is the task the GPU
# run scored, so it is the one that compares.
#
# max_tokens is the GPU run's 1024 by default. On a model that must think, a
# 1024-token budget is tight and a truncated answer scores as wrong -- so the
# note reports unparsed_rate. If that comes back high, raise GSM8K_MAX_TOKENS
# and say so, rather than reporting the number as if it measured the model.
# ---------------------------------------------------------------------------
if wants gsm8k; then
  python3 "${DEV_DIR}/qwen38_2p4t_kv_fit.py" \
    --tokens-per-request $((1536 + GSM8K_MAX_TOKENS)) --concurrency "${GSM8K_CONC}" \
    --block-size "${BLOCK_SIZE:-128}" --warn-only || true
  GSM8K_ARGS=(
    --model local-chat-completions
    --model_args "$(common_model_args chat/completions),tokenized_requests=False,num_concurrent=${GSM8K_CONC},max_length=16384"
    --tasks "${GSM8K_TASK}"
    --apply_chat_template
    --gen_kwargs "max_tokens=${GSM8K_MAX_TOKENS},temperature=0,top_p=1.0"
    --batch_size 1
    --log_samples
    --output_path "${ART}/eval_std_gsm8k"
  )
  [ "${GSM8K_TASK}" = "gsm8k_std" ] && GSM8K_ARGS+=(--include_path "${TASK_DIR}")
  [ "${GSM8K_LIMIT}" != "0" ] && GSM8K_ARGS+=(--limit "${GSM8K_LIMIT}")
  run_leg gsm8k "${GSM8K_TASK}" local-chat-completions "${GSM8K_TIMEOUT}" "${GSM8K_ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# MMLU: stock mmlu, all 14042 questions across 57 subjects, loglikelihood.
#
# multiple_choice, so lm_eval scores it by asking the server for the logprob of
# each of "A".."D" continuing the prompt -- max_tokens=1, logprobs=1, echo=True
# over /v1/completions. vLLM turns echo into prompt_logprobs
# (completion/protocol.py:356-358) and the TPU runner implements those
# (tpu_runner.py:2083), so this should work; it is the canonical MMLU number and
# the one the GPU run produced, so it is worth the risk. But "should" is doing
# work there -- prompt_logprobs on the torchax path with attn_dp and a hybrid
# mamba model is not something we have exercised -- so a single probe request
# decides, before 56168 of them are queued behind a wrong assumption.
#
# The fallback is mmlu_generative over the raw completions endpoint. It is a
# different number and not comparable to the GPU run; it is there so a run that
# cannot do loglikelihood still comes back with something. No chat template on
# that path deliberately: mmlu_generative stops at the first newline, which a
# mandatory reasoning block would trip over immediately.
# ---------------------------------------------------------------------------
if wants mmlu; then
  MMLU_TASK=""
  if ! server_up; then
    record mmlu "skipped" 0 "server not healthy"
  else
    echo "--- mmlu: probing /v1/completions for echo+logprobs"
    if python3 - "${PORT}" "${MODEL}" <<'PY'
import json, sys, urllib.request
port, model = sys.argv[1], sys.argv[2]
body = json.dumps({"model": model, "prompt": "The capital of France is",
                   "max_tokens": 1, "logprobs": 1, "echo": True,
                   "temperature": 0}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body,
                             {"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as r:
    d = json.load(r)
lp = d["choices"][0].get("logprobs") or {}
tl = lp.get("token_logprobs") or []
# Echoed prompt logprobs: several entries, first is null (no preceding token).
if len(tl) < 3 or sum(x is not None for x in tl) < 2:
    print(f"[std-eval] echo+logprobs unusable: token_logprobs={tl!r}")
    raise SystemExit(1)
print(f"[std-eval] echo+logprobs ok: {len(tl)} entries, "
      f"{sum(x is not None for x in tl)} non-null")
PY
    then
      MMLU_TASK="mmlu"
      MMLU_ARGS=(
        --model local-completions
        --model_args "$(common_model_args completions),num_concurrent=${MMLU_CONC},max_length=${MAX_LEN}"
        --tasks mmlu
        --batch_size "${MMLU_BATCH}"
        --log_samples
        --output_path "${ART}/eval_std_mmlu"
      )
      [ -n "${MMLU_NUM_FEWSHOT}" ] && MMLU_ARGS+=(--num_fewshot "${MMLU_NUM_FEWSHOT}")
    else
      echo "[std-eval] WARNING: loglikelihood unavailable; falling back to mmlu_generative."
      echo "[std-eval] WARNING: that number is NOT comparable to the GPU mmlu run."
      MMLU_TASK="mmlu_generative"
      MMLU_ARGS=(
        --model local-completions
        --model_args "$(common_model_args completions),tokenized_requests=False,num_concurrent=64,max_gen_toks=256,max_length=${MAX_LEN}"
        --tasks mmlu_generative
        --batch_size 1
        --log_samples
        --output_path "${ART}/eval_std_mmlu"
      )
      [ -n "${MMLU_NUM_FEWSHOT}" ] && MMLU_ARGS+=(--num_fewshot "${MMLU_NUM_FEWSHOT}")
    fi
    [ "${MMLU_LIMIT}" != "0" ] && MMLU_ARGS+=(--limit "${MMLU_LIMIT}")
    run_leg mmlu "${MMLU_TASK}" local-completions "${MMLU_TIMEOUT}" "${MMLU_ARGS[@]}"
  fi
fi

echo "--- standard eval summary"
cat "${STATUS_JSON}"; echo
if ! grep -q '"status": "ok"' "${STATUS_JSON}"; then
  echo "[std-eval] FAILED: no eval completed" >&2
  exit 1
fi
