#!/bin/bash
# DEV ONLY: run the hash-table unit tests on the follow-up, then check that
# the vllm_moe_apply tests catch two regressions.
set -u
cd /workspace/tpu_inference
MOE=tpu_inference/layers/vllm/interface/moe.py
TEST=tests/layers/vllm/test_moe.py

echo "--- unit tests on the follow-up"
python3 -m pytest -v -rs "$TEST" \
  tests/models/vllm/experimental/test_deepseek_v4_hash_routing.py || exit 1
cp "$MOE" /tmp/moe.py.orig

mutate() {  # $1 = description, $2 = python snippet editing source `s`
  cp /tmp/moe.py.orig "$MOE"
  python3 - "$MOE" "$2" <<'PY' || exit 1
import sys
path, snippet = sys.argv[1], sys.argv[2]
s = open(path).read()
before = s
exec(snippet)
assert s != before, "mutation not applied"
open(path, "w").write(s)
PY
  echo "--- mutation: $1"
  if python3 -m pytest -q -rN "$TEST" -k vllm_moe_apply; then
    echo "MUTATION SURVIVED: $1"
    exit 1
  fi
  echo "MUTATION KILLED: $1"
}

mutate "read the plain CPU hash_indices_table instead of tpu_hash_indices_table" \
  's = s.replace("hash_table = getattr(layer, \"tpu_hash_indices_table\", None)", "hash_table = getattr(layer, \"hash_indices_table\", None)")'
mutate "silently fall back to the plain hash_indices_table (drop the raise)" \
  'import re; s = re.sub(r"raise RuntimeError\((?:.|\n)*?before the forward pass\.\"\)", "hash_table = layer.hash_indices_table", s)'

cp /tmp/moe.py.orig "$MOE"
echo "All mutations killed."
