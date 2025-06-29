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
from tpu_commons.models.jax.common.sharding import Sharding
from tpu_commons.models.jax.model_loader import get_model
from tpu_commons.runner.jax.input_batch_jax import (CachedRequestState,
                                                    InputBatch)
from tpu_commons.runner.jax.input_prep import InputPrep
from tpu_commons.runner.tpu_torch_xla_runner import _get_token_paddings

logger = init_logger(__name__)

MIN_NUM_SEQS = 8


class TPUModelRunner():

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
        architectures = getattr(hf_config, "architectures", [])
        #TODO merge the if-else branches when new design is ready
        # Llama4Scout is only used for new model design,
        # so that we use it as a flag for testing the new model design
        if architectures == ["Llama4Scout"]:
            try:
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
        self.model_fn, self.total_model_params_num = get_model(
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

    # Copy-pasted from `tpu_torch_xla_runner.py`. This is for state management.
    # TODO(pooyam): Consider extraction and reuse instead of copy-paste as this is common logic.
    def _update_states(self, scheduler_output: "VllmSchedulerOutput") -> bool:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        Returns:
            True if there is a new/resumed/paused/finished request.
            If False, we can skip copying SamplingMetadata to the GPU.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)

        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                generator=None,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            # Update the cached states.
            req_state.num_computed_tokens = req_data.num_computed_tokens
            if not req_data.resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                for block_ids, new_block_ids in zip(req_state.block_ids,
                                                    req_data.new_block_ids,
                                                    strict=True):
                    block_ids.extend(new_block_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = req_data.new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                req_data.num_computed_tokens)
            self.input_batch.block_table.append_row(req_data.new_block_ids,
                                                    req_index)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        return len(unscheduled_req_ids) > 0 or len(req_ids_to_add) > 0


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
