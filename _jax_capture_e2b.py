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

Runs E2B end-to-end via vllm + JAX path, intercepts intermediate tensors via
monkey-patching `Gemma4Model.compute_per_layer_inputs` and `__call__`, and
writes them to /tmp/hf_home/jax_e2b_hidden.npz so a Buildkite step can
upload it as an artifact.

Invoke via .buildkite/pipeline_dev.yml — see the "JAX hidden-state capture"
step on cuiq-bringup-gemma4-e2b-jax.

THROWAWAY. Reverted before opening any production PR.
"""

import os
import time
from itertools import islice

import jax.numpy as jnp
import numpy as np

OUTPUT_PATH = "/tmp/hf_home/jax_e2b_hidden.npz"
captures: dict = {}


def install_hooks():
    from tpu_inference.models.jax import gemma4

    orig_compute = gemma4.Gemma4Model.compute_per_layer_inputs

    def patched_compute(self, input_ids, inputs_embeds, is_multimodal=None):
        out = orig_compute(self, input_ids, inputs_embeds, is_multimodal)
        if "inputs_embeds" not in captures:
            captures["inputs_embeds"] = np.asarray(
                jnp.asarray(inputs_embeds, dtype=jnp.float32))
        if out is not None and "per_layer_inputs" not in captures:
            captures["per_layer_inputs"] = np.asarray(
                jnp.asarray(out, dtype=jnp.float32))
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
            key = f"layer_{layer_idx}"
            if key not in captures:
                captures[key] = np.asarray(jnp.asarray(x, dtype=jnp.float32))
        x = self.norm(x)
        if "final_norm" not in captures:
            captures["final_norm"] = np.asarray(
                jnp.asarray(x, dtype=jnp.float32))
        stacked = jnp.stack(all_expert_ids, axis=0) if all_expert_ids else None
        return kv_caches, x, stacked

    gemma4.Gemma4Model.__call__ = patched_call


def main():
    os.environ.setdefault("JITTED_MM_MODULE_KEYS",
                          "model.vision_tower.encoder")
    os.environ.setdefault(
        "REGISTER_MM_MODULE_CUSTOM_PYTREE_CLASSES",
        "transformers.modeling_outputs.BaseModelOutputWithPast",
    )
    os.environ.setdefault("SKIP_JAX_PRECOMPILE", "1")
    os.environ.setdefault("VLLM_XLA_CHECK_RECOMPILATION", "0")

    install_hooks()
    print("[hooks] installed", flush=True)

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model="google/gemma-4-E2B-it",
        enforce_eager=True,
        dtype="bfloat16",
        max_model_len=512,
        max_num_batched_tokens=512,
        tensor_parallel_size=1,
    )
    print(f"[load] {time.time() - t0:.1f}s", flush=True)

    prompt = "The capital of France is"
    out = llm.generate([prompt], SamplingParams(max_tokens=4, temperature=0.0))
    print(f"[output] {out[0].outputs[0].text!r}", flush=True)
    print(f"[captures] count: {len(captures)}", flush=True)
    print(f"[captures] keys: {sorted(captures.keys())[:6]} ...", flush=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez_compressed(OUTPUT_PATH, **captures)
    print(f"[save] {OUTPUT_PATH}", flush=True)
    print(f"[save] size: {os.path.getsize(OUTPUT_PATH)} bytes", flush=True)


if __name__ == "__main__":
    main()
