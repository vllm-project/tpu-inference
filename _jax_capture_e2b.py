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
"""JAX-side hidden-state capture for E2B oracle-diff debugging.

Runs E2B end-to-end via vllm + JAX path. Uses jax.debug.callback to push
intermediate tensors out of the jit trace into a Python `captures` dict.
Writes /tmp/hf_home/jax_e2b_hidden.npz for buildkite-agent artifact upload.

Invoke via .buildkite/pipeline_dev.yml (see "JAX hidden-state capture" step).

THROWAWAY. Reverted before opening any production PR.
"""

import os

# Set env vars BEFORE any vllm / tpu_inference imports — the monkey-patched
# Gemma4Model.__call__ has a different identity than the cached compile, so
# the runner's recompilation watchdog would reject it. run_in_docker.sh hard-
# codes VLLM_XLA_CHECK_RECOMPILATION=1; override here at module load time.
os.environ["VLLM_XLA_CHECK_RECOMPILATION"] = "0"
os.environ["JITTED_MM_MODULE_KEYS"] = "model.vision_tower.encoder"
os.environ["REGISTER_MM_MODULE_CUSTOM_PYTREE_CLASSES"] = (
    "transformers.modeling_outputs.BaseModelOutputWithPast")
os.environ["SKIP_JAX_PRECOMPILE"] = "1"

import time  # noqa: E402
from itertools import islice  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

OUTPUT_PATH = "/tmp/hf_home/jax_e2b_hidden.npz"
captures: dict = {}


def _store(name: str, value: np.ndarray) -> None:
    """Python-side callback. Records the first occurrence per name."""
    if name not in captures:
        captures[name] = np.asarray(value).astype(np.float32)


def _push(name: str, x):
    """Schedule a host callback inside the jit trace to record `x` as `name`.

    `ordered=True` ensures the callback fires in program order; without it the
    runtime can reorder side effects.
    """
    jax.debug.callback(lambda v, n=name: _store(n, v), x, ordered=True)


def install_hooks():
    from tpu_inference.models.jax import gemma4

    orig_compute = gemma4.Gemma4Model.compute_per_layer_inputs

    def patched_compute(self, input_ids, inputs_embeds, is_multimodal=None):
        _push("inputs_embeds", inputs_embeds)
        out = orig_compute(self, input_ids, inputs_embeds, is_multimodal)
        if out is not None:
            _push("per_layer_inputs", out)
        return out

    gemma4.Gemma4Model.compute_per_layer_inputs = patched_compute

    def patched_call(self,
                     kv_caches,
                     input_ids,
                     attention_metadata,
                     inputs_embeds=None,
                     layer_name_to_kv_cache=None,
                     is_multimodal=None):
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)
            x = x * self.embedding_scale
        per_layer_inputs = self.compute_per_layer_inputs(
            input_ids, x, is_multimodal=is_multimodal)
        all_expert_ids = []
        for i, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer)):
            layer_idx = i + self.start_layer
            layer_name = f"layer.{layer_idx}"
            if isinstance(attention_metadata, dict):
                lam = attention_metadata[layer_name]
            else:
                lam = attention_metadata
            if layer_name_to_kv_cache and layer_name in layer_name_to_kv_cache:
                cache_idx = layer_name_to_kv_cache[layer_name]
            else:
                cache_idx = layer_idx
            kv_cache = kv_caches[cache_idx]
            ple = (per_layer_inputs[:, layer_idx, :]
                   if per_layer_inputs is not None else None)
            kv_cache, x, expert_ids = layer(kv_cache,
                                            x,
                                            lam,
                                            per_layer_input=ple)
            if expert_ids is not None:
                all_expert_ids.append(expert_ids)
            kv_caches[cache_idx] = kv_cache
            _push(f"layer_{layer_idx}", x)
        x = self.norm(x)
        _push("final_norm", x)
        stacked = jnp.stack(all_expert_ids, axis=0) if all_expert_ids else None
        return kv_caches, x, stacked

    gemma4.Gemma4Model.__call__ = patched_call


def main():
    install_hooks()
    print("[hooks] installed", flush=True)

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model="google/gemma-4-E2B-it",
        enforce_eager=True,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_batched_tokens=4096,  # E2B vision tower needs 2496 tokens/image
        tensor_parallel_size=1,
    )
    print(f"[load] {time.time() - t0:.1f}s", flush=True)

    prompt = "The capital of France is"
    out = llm.generate([prompt], SamplingParams(max_tokens=4, temperature=0.0))
    print(f"[output] {out[0].outputs[0].text!r}", flush=True)
    print(f"[captures] count: {len(captures)}", flush=True)
    if captures:
        print(f"[captures] keys: {sorted(captures.keys())[:6]} ...",
              flush=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez_compressed(OUTPUT_PATH, **captures)
    print(f"[save] {OUTPUT_PATH}", flush=True)
    print(f"[save] size: {os.path.getsize(OUTPUT_PATH)} bytes", flush=True)


if __name__ == "__main__":
    main()
