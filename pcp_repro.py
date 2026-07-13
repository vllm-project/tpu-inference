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
"""Isolate the PCP serve failure: offline 8B tp4 pcp2 with a LONG prompt
(matching the serve config: max-model-len/bnt = 4096). If this collapses into
gibberish, the bug is long-context-at-8B (kernel/wrapper). If it's coherent,
the bug is the serve path (scheduler / chat template / API)."""
import os
import sys

os.environ.setdefault("SKIP_JAX_PRECOMPILE", "1")
os.environ.setdefault("NEW_MODEL_DESIGN", "1")
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-0.6B" if "--06b" in sys.argv else "Qwen/Qwen3-8B-Base"
TP = 2 if "--tp2" in sys.argv else 4
PCP = 1 if "--nopcp" in sys.argv else 2

# ~2000-token prompt (like gsm8k 5-shot). Coherent few-shot-style filler.
shot = (
    "Question: A store had 120 apples and sold 45 in the morning and 30 in "
    "the afternoon. How many are left? Answer: The store sold 45 + 30 = 75 "
    "apples, so 120 - 75 = 45 apples are left. #### 45\n")
prompt = (
    shot * 20 +
    "Question: Every day, Wendi feeds each of her chickens three cups of "
    "mixed chicken feed. She gives 15 cups in the morning and 25 cups in "
    "the afternoon. How many cups does she need in the final meal if her "
    "flock is 20 chickens? Answer:")

llm = LLM(model=MODEL,
          max_model_len=4096,
          max_num_batched_tokens=4096,
          max_num_seqs=1,
          gpu_memory_utilization=0.6 if "--06b" in sys.argv else 0.9,
          tensor_parallel_size=TP,
          prefill_context_parallel_size=PCP,
          enable_prefix_caching=False)
mt = 400 if "--long" in sys.argv else 120
out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=mt))
txt = out[0].outputs[0].text
print("n_prompt_tokens =", len(out[0].prompt_token_ids))
print("n_gen_tokens =", len(out[0].outputs[0].token_ids))
# flag byte-level / non-ascii garbage bursts
bad = sum(1 for c in txt if ord(c) > 0x2000)
print("non_ascii_garbage_chars =", bad)
print("GEN:", repr(txt))
