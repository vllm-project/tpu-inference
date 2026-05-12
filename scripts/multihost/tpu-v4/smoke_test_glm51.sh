#!/bin/bash
# Fire 5 canonical prompts against a running GLM-5.1 vllm server and
# compare against PP=8 golden reference outputs.
#
# Usage:
#   bash smoke_test_glm51.sh [PORT]        # default PORT=8000
#
# Server expectations (switches the PR makes opt-in; flip them on for
# the EP=8 TP=4 v4-64 target — otherwise the CPU-RAM budget is exceeded
# during weight load or the MoE path stays on the slower requantize
# branch):
#   export NEW_MODEL_DESIGN=1
#   export VLLM_MOE_SKIP_REQUANTIZATION=1   # direct FP8 MoE path
#   vllm serve <model> \
#       --load-format tpu_streaming_loader \  # per-host streaming load
#       --kv-cache-dtype auto \
#       --tensor-parallel-size 4 --enable-expert-parallel \
#       --additional-config '{"sharding":{"sharding_strategy":{"expert_parallelism":8}}}'
#
# Golden reference (PP=8 TP=4, serve_20260419_102620.log):
#   CLEAN   -> '5\n5+4=9\n9+'
#   BROKEN  -> ' - ProProfs Discuss\n\n# If there are'
#   CAPITAL -> ' Paris. Distance from London to Paris is 34'
#   MATH    -> ' 56\nQ: What is 7 times'
#   CODE    -> '  # function to add two numbers\n    return'

set -e

PORT="${1:-8000}"
BASE="http://127.0.0.1:${PORT}"
# vLLM's /v1/completions requires the `model` field to match the served
# name (`vllm serve <path>` uses the path as its id).  Override via
# MODEL when the server registered a different name.
: "${MODEL:?set MODEL to the id the server registered for GLM-5.1-FP8 (usually the value you passed to \"vllm serve\")}"
MAX_TOKENS="${MAX_TOKENS:-10}"
TIMEOUT="${TIMEOUT:-300}"

LOG_DIR="${LOG_DIR:-/tmp/vllm_multih}"
OUTDIR="$LOG_DIR/smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

echo "=== GLM-5.1 smoke test ==="
echo "  Server  : $BASE"
echo "  Outdir  : $OUTDIR"
echo "  Tokens  : $MAX_TOKENS"
echo ""

# ── Golden reference ────────────────────────────────────────────────────────
declare -A GOLDEN
GOLDEN[CLEAN]='5\n5+4=9\n9+'
GOLDEN[BROKEN]=' - ProProfs Discuss\n\n# If there are'
GOLDEN[CAPITAL]=' Paris. Distance from London to Paris is 34'
GOLDEN[MATH]=' 56\nQ: What is 7 times'
GOLDEN[CODE]='  # function to add two numbers\n    return'

# ── Prompts ─────────────────────────────────────────────────────────────────
declare -A PROMPTS
PROMPTS[CLEAN]="2+3="
PROMPTS[BROKEN]="If there are 5 apples and I eat 2, how many are left?"
PROMPTS[CAPITAL]="The capital of France is"
PROMPTS[MATH]="Q: What is 7 times 8? A:"
PROMPTS[CODE]="def add(a, b):"

# ── Fire each prompt ─────────────────────────────────────────────────────────
fire_prompt() {
  local name="$1"
  local prompt="$2"
  local outfile="$OUTDIR/${name}.json"

  echo -n "  $name ... "
  local start=$SECONDS
  curl -sf --max-time "$TIMEOUT" "$BASE/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"prompt\": $(echo -n "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'), \"max_tokens\": $MAX_TOKENS, \"temperature\": 0}" \
    -o "$outfile"
  local elapsed=$((SECONDS - start))
  local text
  text=$(python3 -c "import json,sys; d=json.load(open('$outfile')); print(repr(d['choices'][0]['text']))" 2>/dev/null || echo "ERROR")
  printf "%s [%ds]\n" "$text" "$elapsed"
}

for name in CLEAN BROKEN CAPITAL MATH CODE; do
  fire_prompt "$name" "${PROMPTS[$name]}"
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Results vs PP=8 golden ==="
PASS=0
FAIL=0
for name in CLEAN BROKEN CAPITAL MATH CODE; do
  text=$(python3 -c "import json; d=json.load(open('$OUTDIR/${name}.json')); print(d['choices'][0]['text'][:40])" 2>/dev/null || echo "ERROR")
  golden_prefix="${GOLDEN[$name]}"
  # Compare first 20 chars of text vs golden prefix (escape sequences not expanded)
  if python3 -c "
import json, sys
d = json.load(open('$OUTDIR/${name}.json'))
text = d['choices'][0]['text']
golden = '${golden_prefix}'.encode().decode('unicode_escape')
n = min(len(golden), 20)
ok = text[:n] == golden[:n]
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
" 2>/dev/null; then
    echo "  PASS  $name"
    PASS=$((PASS+1))
  else
    echo "  FAIL  $name  got=$(python3 -c "import json; d=json.load(open('$OUTDIR/${name}.json')); print(repr(d['choices'][0]['text'][:50]))")"
    FAIL=$((FAIL+1))
  fi
done

echo ""
echo "  $PASS/5 passed"
echo "  Results: $OUTDIR/"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
