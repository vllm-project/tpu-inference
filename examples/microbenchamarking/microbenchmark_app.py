# pytype: disable=import-error
# pytype: disable=module-attr
# pytype: disable=attribute-error
# pytype: disable=wrong-arg-types
import json
import os
import time
from typing import Any, Sequence

import jax
from absl import app, flags
from flax import nnx
from microbenchmark_input_utils import InputArgs, InputCreator
from microbenchmark_utils import Sampler, init_mesh

from tpu_commons.logger import init_logger
from tpu_commons.mock.vllm_config_utils import (CacheConfig, ModelConfig,
                                                VllmConfig)
from tpu_commons.models.jax.model_loader import get_model

logger = init_logger(__name__)

_MAX_SEQ_LEN = flags.DEFINE_integer("max_seq_len", 4096,
                                    "Maximum sequence length.")

_PHASE = flags.DEFINE_string("phase", "decode", "Phase to benchmark.")

_DECODE_OFFSET_FROM_PREFILL = flags.DEFINE_integer(
    "decode_offset_from_prefill",
    0,
    "Offset from the prefill length to start decoding.",
)

_MODEL_NAME = flags.DEFINE_string(
    "model_name", "qwen3-32b",
    "Model name to benchmark. Supported models: qwen3-32b, deepseek_v3")

_MODEL_HF_CONFIG = flags.DEFINE_string(
    "model_hf_config",
    "",
    "Model HF config in json format.",
)

# this has to be overriden as the calculation is not very correct yet on microbenchmark side.
#TODO: @(vijaya) Fix the calculation and remove this flag as an override.
_KV_NUM_BLOCK_OVERRIDE = flags.DEFINE_integer(
    "num_block_override",
    2048,
    "Override number of blocks.",
)

_MAX_PREFILL_LEN = flags.DEFINE_integer("max_prefill_len", 512,
                                        "Maximum prefill length.")

# for prefill, max_num_seq = max_seq_len // max_prefill_len
# for decode max_num_seq = max_seq_len // 1
_MAX_NUM_SEQ = flags.DEFINE_integer(
    "max_num_seq",
    2048,
    "maximum number of sequences to be benchmarked.",
)

_MODEL_CALL_STEPS = flags.DEFINE_integer("model_call_steps", 5,
                                         "Number of model call steps.")

_BLOCK_SIZE = flags.DEFINE_integer("block_size", 128, "Block size.")
_SAMPLER_TYPE = flags.DEFINE_string("sampler_type", "fixed", "Sampler type.")
_SAMPLER_STD = flags.DEFINE_float("sampler_std", 1.0,
                                  "Sampler standard deviation.")
_ADDITIONAL_CONFIG = flags.DEFINE_string(
    "additional_config",
    "",
    "Additional configuration for the model.",
)
_MODEL_CONFIG = flags.DEFINE_string(
    "model_config",
    "",
    "Model configuration for the model.",
)

NEW_MODEL_DESIGN = flags.DEFINE_string(
    "NEW_MODEL_DESIGN",
    "True",
    "Model design to use. If True, uses the new model design which is used for newer models like DeepseekV3 and Llama4",
)

_TRACE_DIR = flags.DEFINE_string(
    "trace_dir",
    "/tmp/tpu_commons_traces",
    "Directory to save the trace files.",
)


def get_hf_config_attribute_map(model_hf_config: str):
    with open(model_hf_config, 'r') as file:
        # Load the JSON data from the file
        data = json.load(file)
    return data


def validate_command_line_args():
    if _PHASE.value not in ["prefill", "decode"]:
        raise ValueError(
            f"Phase {_PHASE.value} not supported. Choose either 'prefill' or 'decode'."
        )
    if _MAX_SEQ_LEN.value % _BLOCK_SIZE.value != 0:
        raise ValueError(
            f"Max sequence length {_MAX_SEQ_LEN.value} must be divisible by block size {_BLOCK_SIZE.value}."
        )

    if _PHASE.value == "prefill":
        if _MAX_SEQ_LEN.value % _MAX_PREFILL_LEN.value != 0:
            raise ValueError(
                f"Max sequence length {_MAX_SEQ_LEN.value} must be divisible by max prefill length {_MAX_PREFILL_LEN.value}."
            )
        if _MAX_SEQ_LEN.value // _MAX_PREFILL_LEN.value != _MAX_NUM_SEQ.value:
            raise ValueError(
                f"Max number of sequences {_MAX_NUM_SEQ.value} must be equal to max sequence length {_MAX_SEQ_LEN.value} divided by max prefill length {_MAX_PREFILL_LEN.value}."
            )


class Benchmarker:

    def __init__(self, vllm_config: VllmConfig, model: Any, mesh: Any,
                 sampler: Sampler, rng: nnx.Rngs, model_hf_config, state,
                 trace_directory):
        """
        Class takes in VllmConfig, model function, mesh, sampler, rng, model_hf_config, state and trace_directory.
        and benchmarks the model for the given phase after creating the input using InputCreator class.
        """
        self.vllm_config = vllm_config
        self.model = model
        self.mesh = mesh
        self.sampler = sampler
        self.rng = rng
        self.model_hf_config = model_hf_config
        self.state = state
        self.trace_directory = trace_directory

    def benchmark(self, phase: str):
        input_args = InputArgs(
            block_size=_BLOCK_SIZE.value,
            max_num_seq=_MAX_NUM_SEQ.value,
            min_prefill_len=1,
            max_prefill_len=_MAX_PREFILL_LEN.value,
            max_model_len=_MAX_SEQ_LEN.value,
            decode_offset_from_prefill=_DECODE_OFFSET_FROM_PREFILL.value,
            sampler=self.sampler,
            model_hf_config=self.model_hf_config,
            phase=phase,
            num_blocks_override=_KV_NUM_BLOCK_OVERRIDE.value)

        input_creator = InputCreator(input_args=input_args,
                                     sharding=None,
                                     mesh=self.mesh,
                                     rng=self.rng)
        model_input = input_creator.create_input(phase=phase)

        jax.profiler.start_trace(self.trace_directory)
        start_time = time.time()
        kv_caches, act = self.model(
            self.state,
            model_input.kv_caches,
            model_input.input_ids,
            model_input.attention_metadata,
        )

        act.block_until_ready()
        end_time = time.time()
        jax.profiler.stop_trace()
        logger.info(
            f"Time taken for model call in phase {phase}: {end_time - start_time} seconds. and profile trace is saved in {self.trace_directory}"
        )


def main(argv: Sequence[str]):
    sampler = Sampler(type=_SAMPLER_TYPE.value, std=_SAMPLER_STD.value)
    rng = nnx.Rngs(params=0)
    vllm_config = VllmConfig(
        additional_config=json.loads(_ADDITIONAL_CONFIG.value),
        model_config=ModelConfig(**json.loads(_MODEL_CONFIG.value)),
        cache_config=CacheConfig(block_size=_BLOCK_SIZE.value),
    )

    validate_command_line_args()

    vllm_config.model_config.hf_config.attribute_map = get_hf_config_attribute_map(
        _MODEL_HF_CONFIG.value)

    mesh = init_mesh(vllm_config, jax.devices())
    model_fn, compute_logits_fn, get_multimodal_embeddings_fn, get_input_embeddings_fn, state = get_model(
        vllm_config,
        rng.params(),
        mesh,
    )

    benchmarker = Benchmarker(vllm_config, model_fn, mesh, sampler, rng,
                              vllm_config.model_config.hf_config, state,
                              _TRACE_DIR.value)
    for _ in range(_MODEL_CALL_STEPS.value):
        benchmarker.benchmark(_PHASE.value)


if __name__ == "__main__":
    # uncomment below line to enable new model design
    # os.environ['NEW_MODEL_DESIGN'] = 'True'
    os.environ['JAX_RANDOM_WEIGHTS'] = 'True'
    os.environ['TPU_BACKEND_TYPE'] = 'JAX'
    app.run(main)
