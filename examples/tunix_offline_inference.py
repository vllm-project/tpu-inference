# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import vllm.envs as envs
from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser

from tpu_commons.core import disagg_utils
from tunix.models.llama3 import model as llama_lib
from flax import nnx
from qwix import lora
import jax
from jax.interpreters import pxla
import numpy as np
from jax.sharding import Mesh
from qwix import lora

os.environ["TPU_BACKEND_TYPE"] = "jax"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["JAX_RANDOM_WEIGHTS"] = "True"

def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="/workspace/tunix/rl/grpo/models/meta-llama/Meta-Llama-3-8B-Instruct/")
    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)

    return parser

def get_lora_model(base_model):
  lora_provider = lora.LoraProvider(
      module_path=(
          ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj"
      ),
      rank=64,
      alpha=64.0,
  )

  model_input = base_model.get_model_input()
  lora_model = lora.apply_lora_to_model(
      base_model, lora_provider, **model_input
  )

  state = nnx.state(lora_model)
  pspecs = nnx.get_partition_spec(state)
  sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
  nnx.update(lora_model, sharded_state)

  return lora_model

def load_llama3_model(model_version: str = "llama3-1b", enable_lora: bool = False):
  model_config = {
      "llama3-1b": llama_lib.ModelConfig.llama3_1b,
      "llama3-8b": llama_lib.ModelConfig.llama3_8b,
  }
  assert (
      model_version in model_config
  ), f"Invalid model version: {model_version}"
  model_config = model_config[model_version]()

  @nnx.jit(static_argnames=["enable_lora"])
  def create_sharded_model(enable_lora: bool = False):
      model = llama_lib.Llama3(model_config, rngs=nnx.Rngs(params=0))
      state = nnx.state(model)
      pspecs = nnx.get_partition_spec(state)
      sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
      nnx.update(model, sharded_state)
      return model

  devices = np.array(jax.devices()).reshape(1, -1)  # e.g., (1, 8) for v2-8
  axis_names = ("fsdp", "tp")  #
  mesh = Mesh(devices, axis_names)
  print(f"Current mesh is {mesh=}")
  with mesh:
      model = create_sharded_model(enable_lora=enable_lora)
      if enable_lora:
        model = get_lora_model(model)
  return model


def print_mem_stats(label: str):
  print(f"\nMemstats: {label}:")
  try:
    for d in jax.local_devices():
      stats = d.memory_stats()
      used = round(stats["bytes_in_use"] / 2**30, 2)
      limit = round(stats["bytes_limit"] / 2**30, 2)
      print(f"\tUsing (GB) {used} / {limit} ({used/limit:%}) on {d}")
  except (RuntimeError, KeyError, TypeError) as ex:
      print(f"\tMemstats unavailable, error: {ex}")


def main(args: dict):
    # Pop arguments not used by LLM
    args["max_model_len"] = 1024
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")

    model = args["model"]
    if "8B" in model:
        tunix_model_type = "llama3-8b"
    elif "1B" in model:
        tunix_model_type = "llama3-1b"
    else:
        raise ValueError(f"Unsupported model type: {model}")

    # enable_lora = args["enable_lora"] == 1
    enable_lora = False

    tunix_model = load_llama3_model(tunix_model_type, enable_lora=enable_lora)
    # _, params = nnx.split(tunix_model)

    # args["additional_config"]["custom_nnx_weights"] = params
    if enable_lora:
      args["additional_config"]["lora_config"] = {
            "rank": 64,
            "alpha": 64.0,
            "module_path":
                ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj",
            # "dropout": 0.0,
            # "bias": "none",
          }

    print_mem_stats("After loading tunix model")
    # Create an LLM
    llm = LLM(**args)
    print_mem_stats("After loading LLM model")
    # Create a sampling params object
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    else:
        sampling_params.max_tokens = 1024  # Default value if not provided
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The colors of the rainbow are"
        "The future of AI is",
        "The president of the United States is",
    ]
    if envs.VLLM_TORCH_PROFILER_DIR is not None:
        llm.start_profile()
    outputs = llm.generate(prompts, sampling_params)
    if envs.VLLM_TORCH_PROFILER_DIR is not None:
        llm.stop_profile()

    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    updated_weights = nnx.state(tunix_model)
    mappings = {
      "lm_head.w": ("lm_head", (None, "model")),
      "embedder.input_embedding": ("embed.embedding", ("model", None)),
      "layers.*.input_layernorm.w":
      ("model.layers.*.input_layernorm.scale", (None, )),
      "layers.*.mlp.down_proj.kernel":
      ("model.layers.*.mlp.down_proj.kernel", ("model", None)),
      "layers.*.mlp.gate_proj.kernel":
      ("model.layers.*.mlp.gate_proj.kernel", (None, "model")),
      "layers.*.mlp.up_proj.kernel": ("model.layers.*.mlp.up_proj.kernel",
                                    (None, "model")),
      "layers.*.post_attention_layernorm.w":
      ("model.layers.*.post_attention_layernorm.scale", (None, )),
      "layers.*.attn.k_proj.w":
      ("model.layers.*.self_attn.k_proj.kernel", ("model", None, None)),
      "layers.*.attn.o_proj.w":
      ("model.layers.*.self_attn.o_proj.kernel", ("model", None, None)),
      "layers.*.attn.q_proj.w":
      ("model.layers.*.self_attn.q_proj.kernel", ("model", None, None)),
      "layers.*.attn.v_proj.w":
      ("model.layers.*.self_attn.v_proj.kernel", ("model", None, None)),
      "final_norm.w": ("model.norm.scale", (None, )),
    }
    if enable_lora:
       mappings |= {
        "layers.*.mlp.gate_proj.kernel_lora_a": ("model.layers.*.mlp.gate_proj.kernel_lora_a",(None, None)),
        "layers.*.mlp.gate_proj.kernel_lora_b": ("model.layers.*.mlp.gate_proj.kernel_lora_b",(None, "model")),
        "layers.*.mlp.up_proj.kernel_lora_a": ("model.layers.*.mlp.up_proj.kernel_lora_a",(None, None)),
        "layers.*.mlp.up_proj.kernel_lora_b": ("model.layers.*.mlp.up_proj.kernel_lora_b",(None, "model")),
        "layers.*.mlp.down_proj.kernel_lora_a": ("model.layers.*.mlp.down_proj.kernel_lora_a",("model", None)),
        "layers.*.mlp.down_proj.kernel_lora_b": ("model.layers.*.mlp.down_proj.kernel_lora_b",(None, None)),
        "layers.*.attn.q_proj.w_lora_a": ("model.layers.*.self_attn.q_proj.kernel_lora_a",("model", None)),
        "layers.*.attn.q_proj.w_lora_b": ("model.layers.*.self_attn.q_proj.kernel_lora_b",(None, None)),
        "layers.*.attn.k_proj.w_lora_a": ("model.layers.*.self_attn.k_proj.kernel_lora_a",("model", None)),
        "layers.*.attn.k_proj.w_lora_b": ("model.layers.*.self_attn.k_proj.kernel_lora_b",(None, None)),
        "layers.*.attn.v_proj.w_lora_a": ("model.layers.*.self_attn.v_proj.kernel_lora_a",("model", None)),
        "layers.*.attn.v_proj.w_lora_b": ("model.layers.*.self_attn.v_proj.kernel_lora_b",(None, None)),
        "layers.*.attn.o_proj.w_lora_a": ("model.layers.*.self_attn.o_proj.kernel_lora_a",("model", None)),
        "layers.*.attn.o_proj.w_lora_b": ("model.layers.*.self_attn.o_proj.kernel_lora_b",(None, None))
      }
    llm.llm_engine.model_executor.collective_rpc("sync_weights", args=(updated_weights, mappings))
    outputs = llm.generate(prompts, sampling_params)
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text after loading weights: {generated_text!r}")
        print("-" * 50)

if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())

    if not disagg_utils.is_disagg_enabled():
        main(args)
    else:
        from unittest.mock import patch

        from tpu_commons.core.core_tpu import EngineCore as TPUEngineCore
        from tpu_commons.core.core_tpu import \
            EngineCoreProc as TPUEngineCoreProc
        with patch('vllm.v1.engine.core.EngineCore', TPUEngineCore):
            with patch('vllm.v1.engine.core.EngineCoreProc',
                       TPUEngineCoreProc):
                main(args)
