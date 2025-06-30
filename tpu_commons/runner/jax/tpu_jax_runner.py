# Here we try to bring as much code as possible from Hex-LLM, instead of `tpu_torch_xla_runner.py` -> jax conversion.
# This runner is a port of https://source.corp.google.com/h/vertex-model-garden/hex-llm/+/main:hex_llm/worker/runner_jax.py
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import vllm.envs as envs
from jax.sharding import NamedSharding, PartitionSpec
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.utils import cdiv
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import ModelRunnerOutput

from tpu_commons.core.jetstream_commons.engine import PATHWAYS_ENABLED
from tpu_commons.logger import init_logger
from tpu_commons.models.jax.common.model import Model
from tpu_commons.models.jax.common.sharding import Sharding
from tpu_commons.models.jax.model_loader import (_get_model_architecture,
                                                 get_model)
from tpu_commons.runner.jax.input_batch_jax import (CachedRequestState,
                                                    InputBatch)
from tpu_commons.runner.jax.input_prep import InputPrep
from tpu_commons.runner.tpu_torch_xla_runner import _get_token_paddings
from tpu_commons.runner.update_states_mixin import UpdateStatesMixin

logger = init_logger(__name__)

MIN_NUM_SEQS = 8


class TPUModelRunner(UpdateStatesMixin):

    def __init__(
        self,
        vllm_config: VllmConfig,
        devices: List[Any],
    ):
        self.vllm_config = vllm_config
        self.eviction_algorithm = None
        self.devices = devices

        self.cache_config = self.vllm_config.cache_config
        self.scheduler_config = self.vllm_config.scheduler_config
        self.model_config = self.vllm_config.model_config
        # TODO (jacobplatin)
        setattr(self.model_config, "load_format", "auto")
        self.max_model_len = self.model_config.max_model_len
        self.block_size = self.cache_config.block_size
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        self.max_num_reqs = max(self.scheduler_config.max_num_seqs,
                                MIN_NUM_SEQS)
        self.num_tokens_paddings = _get_token_paddings(
            min_token_size=16,
            max_token_size=self.scheduler_config.max_num_batched_tokens,
            padding_gap=envs.VLLM_TPU_BUCKET_PADDING_GAP)
        # In case `max_num_tokens < max(num_tokens_paddings)` use the actual
        # padded max value to pre-allocate data structures and pre-compile.
        self.max_num_tokens = self.num_tokens_paddings[-1]
        self.vocab_size = self.model_config.get_vocab_size()

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}

        self.encoder_cache: dict[str, dict[int, jax.Array]] = {}

        self.params = None

        # TODO(pooyam): These should be set from vllm side with some defaults.
        self.cache_config.sink_size = None
        self.scheduler_config.decode_blocks_padding = 128
        self.scheduler_config.prefill_len_padding = 128
        self.scheduler_config.chunked_prefill_tokens_padding = 256
        self.perplexity_reference_text = None
        self.cache_config.output_logits = False  # To make model run without error

        self.scheduler_config.page_aligned_scheduling = getattr(
            self.scheduler_config, 'page_aligned_scheduling', None)

        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=None,
            pin_memory=False,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
        )
        self.kv_caches: List[Tuple[jax.Array, jax.Array]] = []

        self._init_random()
        self._init_mesh()
        self._init_model()
        self._init_jit()

        self._verify_chunked_prefill_config()
        logger.info("TPUModelRunner created!")

        self.input_prep = InputPrep(vllm_config=self.vllm_config,
                                    mesh=self.mesh,
                                    input_batch=self.input_batch,
                                    max_num_reqs=self.max_num_reqs,
                                    jitted_read_outputs=self.read_outputs)

    def _init_random(self):
        if self.model_config.seed is None:
            self.model_config.seed = 0
        np.random.seed(self.model_config.seed)
        self.rng_key = jax.random.key(self.model_config.seed)

    def _verify_chunked_prefill_config(self):
        if self.scheduler_config.page_aligned_scheduling:
            if (self.scheduler_config.max_num_batched_tokens %
                    self.cache_config.block_size != 0):
                raise ValueError(
                    "max_num_batched_tokens needs to be multiple of block_size."
                )

            # TODO(pooyam): Explain why.
            if (self.scheduler_config.max_num_batched_tokens //
                    self.cache_config.block_size == 1):
                raise ValueError(
                    "max_num_batched_tokens must be at least 2x of block_size."
                )

        if (self.scheduler_config.max_num_batched_tokens
                < self.scheduler_config.max_num_seqs):
            raise ValueError(
                "max_num_batched_tokens needs to be larger than or equal to max_num_seqs."
            )

        if (self.scheduler_config.chunked_prefill_tokens_padding
                < self.scheduler_config.max_num_seqs):
            raise ValueError(
                "chunked_prefill_tokens_padding needs to be larger than or equal to max_num_seqs."
            )

        # TODO(b/396129273): as the feature-cross compatibiltiy work is done, remove
        # the following check.
        if self.model_config.get_sliding_window():
            raise ValueError(
                "Chunked prefill is not yet supported for model with sliding window."
            )

        # TODO(pooyam): Detect and enable this.
        # if self.model_config.is_mixed_attentions():
        #     raise ValueError(
        #         f"Chunked prefill is not yet supported for model with mixed attention."
        #     )

    def load_model(self):
        pass

    def get_kv_cache_spec(self):
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        # TODO(xiang): this hack tricks engine core to init successfully
        import torch
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        model_config = self.vllm_config.model_config
        parallel_config = self.vllm_config.parallel_config
        for i in range(model_config.get_num_layers(parallel_config)):
            kv_cache_spec[f"layers.{i}"] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=model_config.get_total_num_kv_heads(),
                head_size=model_config.get_head_size(),
                dtype=torch.bfloat16,
                use_mla=False,
            )

        return kv_cache_spec

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        self.kv_cache_config = kv_cache_config

        kv_cache_groups = kv_cache_config.kv_cache_groups
        if len(kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")

        kv_cache_spec = kv_cache_groups[0].kv_cache_spec
        layer_names = kv_cache_groups[0].layer_names
        cache_dtype = jnp.bfloat16
        # TODO(xiang): fix this together with get_kv_cache_spec
        # cache_dtype = kv_cache_spec.dtype

        # TODO(xiang): vllm's pallas backend uses a different shape:
        # (num_blocks, block_size, num_kv_heads * 2, head_size)
        # need to figure out the performance diff.
        cache_shape = (
            kv_cache_spec.num_kv_heads,
            kv_cache_config.num_blocks,
            kv_cache_spec.block_size,
            kv_cache_spec.head_size,
        )
        if PATHWAYS_ENABLED:
            cache_shape = (
                kv_cache_spec.num_kv_heads,  # 8
                #kv_cache_config.num_blocks,  # 7199
                1024,
                kv_cache_spec.block_size,  # 32
                kv_cache_spec.head_size,  # 128
            )
        logger.info(
            f"Init kv-cache | shape={len(layer_names)} * {cache_shape}")
        logger.info(jax.lib.xla_bridge.get_backend().platform_version)

        # Shard the num_kv_heads dim along the 'model' axis.
        self.kv_cache_sharding = NamedSharding(self.mesh,
                                               PartitionSpec("model"))

        def _allocate() -> Any:
            return jnp.empty(
                shape=cache_shape,
                dtype=cache_dtype,
            )

        logger.info("Trace allocate fn.")
        sharded_allocate = jax.jit(_allocate,
                                   out_shardings=self.kv_cache_sharding)
        logger.info(f"allocate kv cache: {self.kv_cache_sharding}")
        for name in layer_names:
            logger.info(f"allocate for {name}")
            k_cache = sharded_allocate()
            v_cache = sharded_allocate()
            self.kv_caches.append((k_cache, v_cache))

        logger.info(f"allocate kv cache: {self.kv_cache_sharding}, done")
        self.output_cache = self.init_output_cache()
        logger.info(
            f"Init kv-cache | shape={len(layer_names)} * {cache_shape}, Done!")

    def init_output_cache(self):
        output_size = (
            self.max_num_reqs + 1,
            self.vllm_config.model_config.max_model_len + 8,
        )
        output_dtype = jnp.int32

        def _allocate() -> Any:
            return jnp.empty(
                shape=output_size,
                dtype=output_dtype,
            )

        # Replicate the output_cache across all devices.
        sharded_allocate = jax.jit(_allocate,
                                   out_shardings=self.outputs_sharding)
        output_cache = sharded_allocate()
        return output_cache

    def capture_model(self) -> None:
        pass

    def _is_generating_new_token(self, scheduler_output: VllmSchedulerOutput,
                                 seq: list[NewRequestData
                                           | CachedRequestData]):
        index = self.input_batch.req_id_to_index[seq.req_id]
        num_tokens = self.input_batch.num_tokens[index]
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
            seq.req_id]
        num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]

        assert num_computed_tokens + num_scheduled_tokens <= num_tokens

        return not (num_computed_tokens + num_scheduled_tokens < num_tokens)

    def execute_model(
        self,
        scheduler_output: "VllmSchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> ModelRunnerOutput:
        self._update_states(scheduler_output)
        inputs = self.input_prep.prepare_inputs(scheduler_output,
                                                self.kv_caches,
                                                self.output_cache)
        if inputs is not None:
            model_inputs, (running_indices, output_token_indices) = inputs
            # TODO (jacobplatin): use logits and single_step_attn_scores_decode?
            self.kv_caches, next_tokens, logits = self.model_fn(*model_inputs)
            self.output_cache = self.write_outputs(self.output_cache,
                                                   next_tokens,
                                                   running_indices,
                                                   output_token_indices)

        prompt_logprobs_dict = {}

        all_reqs = scheduler_output.scheduled_new_reqs + scheduler_output.scheduled_cached_reqs

        running_indices = []
        output_token_indices = []
        req_ids = []

        for i, seq in enumerate(all_reqs):
            # NOTE(pooyam): Unfinished prefills should not return anything to vLLM scheduler.
            if not self._is_generating_new_token(scheduler_output, seq):
                continue

            req_ids.append(seq.req_id)
            index = self.input_batch.req_id_to_index[seq.req_id]
            output_token_index = max(
                self.input_batch.num_computed_tokens_cpu[index] -
                self.input_batch.num_prompt_tokens[index] + 1, 0)
            running_indices.append(index)
            output_token_indices.append(output_token_index)

            prompt_logprobs_dict[seq.req_id] = None

        # NOTE(pooyam): device-to-host transfer step by step is inefficient.
        sampled_token_ids = [[] for _ in range(self.input_batch.num_reqs)]

        if running_indices:
            outputs = self.output_cache.at[running_indices,
                                           output_token_indices].get()
            outputs = jax.device_get(outputs).tolist()
            # NOTE(pooyam): vLLM scheduler reads via `sampled_token_ids[req_index]` where sampled_token_ids is `list[list[int]]`.
            # Not sure why they didn't make it dictionary because not all running sequences will be scheduled at each iter and
            # we are sending pointless [] as the output of such requests. I think it's possible to optimize this if we just send a dict.
            for req_id, running_index, output, output_token_index in zip(
                    req_ids,
                    running_indices,
                    outputs,
                    output_token_indices,
                    strict=True):
                req_state = self.requests[req_id]
                seq_len = (req_state.num_computed_tokens +
                           scheduler_output.num_scheduled_tokens[req_id])
                assert seq_len <= req_state.num_tokens
                assert output_token_index >= 0
                sampled_token_ids[running_index] = [output]

                self.input_batch.token_ids_cpu[running_index, seq_len] = output
                req_state.output_token_ids.append(output)
                self.input_batch.num_tokens[running_index] += 1

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            prompt_logprobs_dict=prompt_logprobs_dict,
            logprobs=None,
            spec_token_ids=None,
            sampled_token_ids=sampled_token_ids,
            pooler_output=[],
        )

    def _init_mesh(self) -> None:
        hf_config = self.model_config.hf_config
        model_class = _get_model_architecture(hf_config)
        #TODO merge the if-else branches when new design is ready
        # Llama4Scout is only used for new model design,
        # so that we use it as a flag for testing the new model design
        if issubclass(model_class, Model):
            try:
                # TODO: Update override steps.
                sharding_strategy = \
                    self.vllm_config.additional_config["overrides"]["sharding"]["sharding_strategy"]
            except KeyError:
                logger.warning(
                    f"No sharding strategy passed! Using default of full model parallelism={len(self.devices)}"
                )
                sharding_strategy = {"tensor_parallelism": len(self.devices)}
            sharding = Sharding(strategy_dict=sharding_strategy)
            self.mesh = sharding.mesh
            logger.info(f"Init mesh | mesh={self.mesh}")
        else:
            # Support for legacy tpu_commons.
            axis_names = ("data", "model")
            mesh_shape = (1, len(self.devices))
            self.mesh = jax.make_mesh(mesh_shape,
                                      axis_names,
                                      devices=self.devices)

            logger.info(f"Init mesh | mesh={self.mesh} done")
            # In case we are in disagg mode, the number of devices can exceed 8.
            # TODO(fhzhang): fix this properly as we implement disagg serving.
            if len(self.devices) > 8:
                self.devices = self.devices[:8]
            mesh_shape = (1, len(self.devices))
            self.mesh = jax.make_mesh(mesh_shape,
                                      axis_names,
                                      devices=self.devices)
            logger.info(f"Init mesh | mesh={self.mesh}")

    def _init_model(self) -> None:
        logger.info("Init model start...")
        self.model_fn = get_model(
            self.vllm_config,
            self.rng_key,
            self.mesh,
        )

    def _init_jit(self) -> None:
        self.outputs_sharding = NamedSharding(self.mesh, PartitionSpec(None))
        self.write_outputs = jax.jit(write_outputs,
                                     donate_argnums=0,
                                     out_shardings=self.outputs_sharding)
        self.read_outputs = jax.jit(read_outputs,
                                    out_shardings=self.outputs_sharding)


def write_outputs(
    output_cache: jax.Array,
    outputs: jax.Array,
    running_indices: jax.Array,
    token_indices: jax.Array,
) -> jax.Array:
    output_cache = output_cache.at[running_indices, token_indices].set(outputs)
    return output_cache


def read_outputs(
    output_cache: jax.Array,
    running_indices: jax.Array,
    token_indices: jax.Array,
) -> jax.Array:
    outputs = output_cache.at[running_indices, token_indices].get()
    outputs = jnp.expand_dims(outputs, 1)
    return outputs
