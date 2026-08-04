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
"""Weight-integrity check for the runai streamer concurrency knob.

Loads the sliced K3 once and fingerprints every parameter (per-leaf float64
sum + abs-sum via jax, reduced to one line per top-level module group).
Run twice — PROBE_STREAMER_CONC unset vs =64 — and diff the outputs; any
difference means the concurrent shard read corrupted weight bytes.
"""
import hashlib
import os

conc = os.environ.get("PROBE_STREAMER_CONC")
tag = "conc" + (conc or "default")

from vllm import LLM  # noqa: E402
from vllm.plugins import load_general_plugins  # noqa: E402

load_general_plugins()
from tpu_inference.models.vllm.experimental import \
    register_models  # noqa: E402

register_models()

kwargs = {}
if conc:
    kwargs["model_loader_extra_config"] = {"concurrency": int(conc)}

llm = LLM(
    model="gs://tpu-commons-ci/moonshootai/kimi/k3-sliced",
    trust_remote_code=True,
    load_format="runai_streamer",
    max_model_len=2048,
    max_num_batched_tokens=512,
    max_num_seqs=8,
    tensor_parallel_size=8,
    gpu_memory_utilization=0.8,
    enable_prefix_caching=False,
    async_scheduling=False,
    num_gpu_blocks_override=256,
    additional_config={
        "sharding": {
            "sharding_strategy": {
                "tensor_parallelism": 2,
                "expert_parallelism": 4,
                "enable_dp_attention": False,
            }
        },
    },
    **kwargs,
)

import numpy as np  # noqa: E402
# Functional fingerprint: greedy tokens + chosen-token logprobs on a fixed
# prompt are a whole-stack weight fingerprint that needs no engine
# internals. Plus, when the engine is in-process (set
# VLLM_ENABLE_V1_MULTIPROCESSING=0), digest the state leaves directly.
from vllm import SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

rng = np.random.default_rng(7)
fixed = [int(x) for x in rng.integers(1000, 150000, size=256)]
sp = SamplingParams(temperature=0.0,
                    max_tokens=32,
                    ignore_eos=True,
                    logprobs=1)
out = llm.generate([TokensPrompt(prompt_token_ids=fixed)], sp,
                   use_tqdm=False)[0]
toks = [int(x) for x in out.outputs[0].token_ids]
lps = []
for tok, lpd in zip(toks, out.outputs[0].logprobs or []):
    ent = lpd.get(tok)
    lps.append(round(float(ent.logprob), 4) if ent is not None else None)
print("[streamer-check]", tag, "TOKENS:", toks)
print("[streamer-check]", tag, "LPS:", lps)

try:
    import jax
    import jax.numpy as jnp
    core = llm.llm_engine.engine_core
    runner = core.engine_core.model_executor.driver_worker.model_runner
    leaves = jax.tree_util.tree_leaves(runner.state_leaves)
    digest = hashlib.sha256()
    for i, leaf in enumerate(leaves):
        if not hasattr(leaf, "dtype"):
            continue
        s = jnp.sum(jnp.abs(leaf.astype(jnp.float32))).item()
        digest.update(f"{i}:{s:.6e};".encode())
    print("[streamer-check]", tag, "leaves:", len(leaves), "DIGEST:",
          digest.hexdigest())
except Exception as exc:  # noqa: BLE001
    print("[streamer-check]", tag, "leaf digest unavailable:", exc)
