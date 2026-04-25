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
"""Offline KV-cache offload example.

Non-hybrid model (Qwen3, Llama, ...):
    python examples/offload/offline_inference_kv_cache.py \\
        --model Qwen/Qwen3-0.6B \\
        --tensor-parallel-size 1 \\
        --max-model-len 1024 --max-num-batched-tokens 1024 --block-size 128 \\
        --kv-transfer-config '{"kv_connector":"TPUOffloadConnector",
            "kv_connector_module_path":"tpu_inference.offload.tpu_offload_connector",
            "kv_role":"kv_both"}'

Hybrid model (Qwen3.5 attn+mamba): two extra flags are required for the
connector to load (vLLM otherwise auto-disables HMA), and one env var to
avoid OOM during the per-bucket swap precompile.

    SKIP_JAX_PRECOMPILE=1 TPU_OFFLOAD_SKIP_JAX_PRECOMPILE=1 \\
    python examples/offload/offline_inference_kv_cache.py \\
        --model Qwen/Qwen3.5-35B-A3B-FP8 \\
        --tensor-parallel-size 1 \\
        --max-model-len 5120 --max-num-batched-tokens 16384 --block-size 256 \\
        --no-disable-hybrid-kv-cache-manager \\
        --kv-transfer-config '{"kv_connector":"TPUOffloadConnector",
            "kv_connector_module_path":"tpu_inference.offload.tpu_offload_connector",
            "kv_role":"kv_both"}'

Notes on Qwen3.5 hybrid offload scope (TPU as of vLLM main):
    - The connector registers ALL kv-cache groups (attn + mamba) via
      SupportsHMA. On `--enable-prefix-caching`, each saved CPU chunk
      carries both the attn block payload AND the request's mamba
      state (1 block per mamba group); on load, both are scattered back.
      Bit-exact verified against the no-offload baseline (see
      examples/offload/verify_offload.sh).
    - vLLM's hybrid prefix-cache hit length is gated by the
      worst-covered group across attn + mamba (see
      vllm/v1/core/kv_cache_coordinator.py find_longest_cache_hit). With
      `mamba_cache_mode="align"` (the default Qwen3Next enforces — it
      raises NotImplementedError on "all"), vLLM only checkpoints mamba
      at LCM(group block sizes) boundaries. Cross-request cache hits
      therefore fire only when the new prompt matches a saved prompt at
      one of those alignment points. To unlock per-attn-block mamba
      reuse, the upstream fix is to make `Qwen3NextForCausalLM`
      implement `SupportsMambaPrefixCaching` (it already returns the
      correct gated_delta_net state copy func, same as Mamba2 which
      does support "all" mode). After that, this connector's mamba
      transfer code path delivers full speedup with no further changes.
    - For non-hybrid Qwen3 (e.g. Qwen3-235B-A22B-Instruct-2507-FP8) the
      offload path is fully active today and is what
      examples/offload/gke/benchmarks/deploy-cpu-offload.yaml exercises.
    - Comparison with vLLM main GPU connectors:
        * `OffloadingConnector` (the production GPU one) asserts
          single-group in scheduler.py:212/277/319/336 and CRASHES at
          startup on hybrid models. The HMA series #34805..#39403 +
          #38261 will eventually fix it; #39403 (store) and #38261
          (mamba alignment) are still open as of this writing.
        * `SimpleCPUOffloadConnector` works on hybrid via a CPU mirror
          of vLLM's KVCacheConfig. We use a different (chunk-keyed)
          architecture cherry-picked from cpu-offloading/dev-merge-0401
          but reach the same per-group transfer outcome.
"""
import os
import time

from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct")
    parser.set_defaults(max_model_len=1024)

    return parser


def parse_outputs(outputs):
    output_token_ids = []
    generated_texts = []
    for output in outputs:
        prompt = output.prompt
        completion = output.outputs[0]
        generated_text = completion.text
        token_ids = completion.token_ids
        print(
            f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}\nToken IDs: {token_ids!r}"
        )
        generated_texts.append(generated_text)
        output_token_ids.append(token_ids)
    return generated_texts, output_token_ids


def main(args: dict):
    # Pop arguments not used by LLM
    # Create an LLM
    llm = LLM(**args)

    # Create a sampling params object
    sampling_params = llm.get_default_sampling_params()

    sampling_params.temperature = 0.0
    sampling_params.seed = 42
    sampling_params.max_tokens = 20
    sampling_params.skip_special_tokens = True

    if os.environ.get("VLLM_TORCH_PROFILER_DIR") is not None:
        llm.start_profile()

    # 1st generate
    prompt = "Every Bill which shall have passed the House of Representatives and the Senate, shall, before it become a Law, be presented to the President of the United States; If he approve he shall sign it, but if not he shall return it, with his Objections to that House in which it shall have originated, who shall enter the Objections at large on their Journal, and proceed to reconsider it. If after such Reconsideration two thirds of that House shall agree to pass the Bill, it shall be sent, together with the Objections, to the other House, by which it shall likewise be reconsidered, and if approved by two thirds of that House, it shall become a Law. But in all such Cases the Votes of both Houses shall be determined by yeas and Nays, and the Names of the Persons voting for and against the Bill shall be entered on the Journal of each House respectively. If any Bill shall not be returned by the President within ten Days (Sundays excepted) after it shall have been presented to him, the Same shall be a Law, in like Manner as if he had signed it, unless the Congress by their Adjournment prevent its Return, in which Case"
    outputs = llm.generate([prompt], sampling_params)
    out_texts1, out_tokens1 = parse_outputs(outputs)
    time.sleep(1)

    # manually let llm scheduler's kv_cache_manager forget all prefixes' hash
    print("Resetting prefix cache...")
    llm.llm_engine.engine_core.reset_prefix_cache()
    time.sleep(1)

    # 2nd generate
    outputs = llm.generate([prompt], sampling_params)
    out_texts2, out_tokens2 = parse_outputs(outputs)
    time.sleep(1)

    if os.environ.get("VLLM_TORCH_PROFILER_DIR") is not None:
        llm.stop_profile()

    # output1 and output2 should be idential
    assert len(out_texts1) == len(out_texts2)
    assert len(out_tokens1) == len(out_tokens2)
    for text1, text2 in zip(out_texts1, out_texts2):
        assert text1 == text2
    for tokens1, tokens2 in zip(out_tokens1, out_tokens2):
        assert tokens1 == tokens2


if __name__ == "__main__":
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)
