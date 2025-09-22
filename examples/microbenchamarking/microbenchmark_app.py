# pytype: disable=import-error
# pytype: disable=module-attr
# pytype: disable=attribute-error
# pytype: disable=wrong-arg-types
import json
import os
from typing import Any, Sequence

import jax
import numpy as np
from absl import app, flags
from flax import nnx
from microbenchmark_hf_configs import qwen3_32b_hf_config, deepseek_v3_hf_config
from microbenchmark_input_utils import InputArgs, InputCreator
from tpu_commons.models.jax.model_loader import get_model
from microbenchmark_utils import Sampler, init_mesh

from tpu_commons.mock.vllm_config_utils import VllmConfig
from tpu_commons.logger import init_logger
from tpu_commons.mock.vllm_config_utils import ModelConfig, VllmConfig

logger = init_logger(__name__)
power_of_two = np.pow(2, np.arange(18))  # up to 128k seq lens
_MAX_SEQ_LEN = flags.DEFINE_integer("max_seq_len", 1024,
                                    "Maximum sequence length.")

_PHASE = flags.DEFINE_string("phase", "prefill", "Phase to benchmark.")

_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    "qwen3-32b",
    "Model name to benchmark. Supported models: qwen3-32b, deepseek_v3"
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
    2,
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
    "Model design to use. If True, uses the new model design.",
)

_TRACE_DIR = flags.DEFINE_string(
    "trace_dir",
    "/tmp/tpu_commons_traces",
    "Directory to save the trace files.",
)

def get_hf_config_attribute_map(model_name: str):
    if model_name == "qwen3-32b":
        return qwen3_32b_hf_config
    elif model_name == "deepseek_v3":
        return deepseek_v3_hf_config
    else:
        raise ValueError(f"Model {model_name} not supported.")

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
            sampler=self.sampler,
            num_kv_heads=self.model_hf_config.num_key_value_heads,
            head_dim=self.model_hf_config.head_dim,
            num_layers=self.model_hf_config.num_hidden_layers,
            vocab_size=self.model_hf_config.vocab_size,
            phase=phase,
            num_blocks_override=_KV_NUM_BLOCK_OVERRIDE.value)

        input_creator = InputCreator(input_args=input_args,
                                     sharding=None,
                                     mesh=self.mesh,
                                     rng=self.rng)
        model_input = input_creator.create_input(phase=phase)

        jax.profiler.start_trace(self.trace_directory)
        kv_caches, act = self.model(
            self.state,
            model_input.kv_caches,
            model_input.input_ids,
            model_input.attention_metadata,
        )
        # kv_caches, act = self.model(self.state, *inputs[:3])
        act.block_until_ready()
        jax.profiler.stop_trace()


def main(argv: Sequence[str]):
    sampler = Sampler(type=_SAMPLER_TYPE.value, std=_SAMPLER_STD.value)
    rng = nnx.Rngs(params=0)
    vllm_config = VllmConfig(
        additional_config=json.loads(_ADDITIONAL_CONFIG.value),
        model_config=ModelConfig(**json.loads(_MODEL_CONFIG.value)),
    )
    
    validate_command_line_args()

    vllm_config.model_config.hf_config.attribute_map = get_hf_config_attribute_map(_MODEL_NAME.value)

    mesh = init_mesh(vllm_config, jax.devices())
    model_fn, compute_logits_fn, get_multimodal_embeddings_fn, get_input_embeddings_fn, state = get_model(
        vllm_config,
        rng.params(),
        mesh,
    )

    benchmarker = Benchmarker(vllm_config, model_fn, mesh, sampler, rng,
                              vllm_config.model_config.hf_config, state, _TRACE_DIR.value)
    for _ in range(_MODEL_CALL_STEPS.value):
        benchmarker.benchmark(_PHASE.value)


if __name__ == "__main__":
    # os.environ['NEW_MODEL_DESIGN'] = 'True'
    os.environ['JAX_RANDOM_WEIGHTS'] = 'True'
    os.environ['TPU_BACKEND_TYPE'] = 'JAX'
    app.run(main)
