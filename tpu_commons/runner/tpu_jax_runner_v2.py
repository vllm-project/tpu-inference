# Here we try to bring as much code as possible from Hex-LLM, instead of `tpu_torch_xla_runner.py` -> jax conversion.
# This runner is a port of https://source.corp.google.com/h/vertex-model-garden/hex-llm/+/main:hex_llm/worker/runner_jax.py
import math
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

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.model_loader import get_model
from tpu_commons.runner.input_batch_jax import CachedRequestState, InputBatch
from tpu_commons.runner.tpu_torch_xla_runner import _get_token_paddings
from tpu_commons.runner.utils import determine_do_sampling

logger = init_logger(__name__)

MIN_NUM_SEQS = 8
# When chunked prefill is enabled, this is the max number of prefill segments that
# could scheduled in one token batch.
MAX_PREFILL_SEQS_PER_TOKEN_BATCH = None
MAX_ALLOWED_PAGE_INDICES_N = (
    128 * 1024
)  # Based on experiments on v5e, 256x1024 results in smem oom but 128x1024 not. TODO: Adjust this based on TPU version.


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

        global MAX_PREFILL_SEQS_PER_TOKEN_BATCH
        # NOTE(pooyam): Currently we don't have a logic in vLLM scheduler to enfore certain upper-bound for number of prefilling seqs.
        # We can remove this and make it configurable once we have such thing in vLLM scheduler.
        # Also gxd@ mentioned to me he had not benchmarked the previous number which was `5` so it's not clear even if it's needed or not.
        MAX_PREFILL_SEQS_PER_TOKEN_BATCH = self.scheduler_config.max_num_seqs

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}

        self.encoder_cache: dict[str, dict[int, jax.Array]] = {}

        self.params = None

        # TODO(pooyam): These should be set from vllm side with some defaults.
        self.cache_config.sink_size = None
        self.scheduler_config.decode_blocks_padding = 128
        self.scheduler_config.prefill_len_padding = 128
        self.perplexity_reference_text = None
        self.decode_seqs_padding = 8
        self.cache_config.output_logits = False  # To make model run without error
        self.scheduler_config.chunked_prefill_tokens_padding = 256
        self.prefill_seqs_padding = 8

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

        self.phase = ""

    def _init_random(self):
        if self.model_config.seed is None:
            self.model_config.seed = 0
        np.random.seed(self.model_config.seed)
        self.rng_key = jax.random.key(self.model_config.seed)

    def _verify_chunked_prefill_config(self):
        if (self.scheduler_config.max_num_batched_tokens %
                self.cache_config.block_size != 0):
            raise ValueError(
                "max_num_batched_tokens needs to be multiple of block_size.")

        # TODO(pooyam): Explain why.
        if (self.scheduler_config.max_num_batched_tokens //
                self.cache_config.block_size == 1):
            raise ValueError(
                "max_num_batched_tokens must be at least 2x of block_size.")

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
        logger.info(
            f"Init kv-cache | shape={len(layer_names)} * {cache_shape}")

        # Shard the num_kv_heads dim along the 'model' axis.
        sharding = NamedSharding(self.mesh, PartitionSpec("model"))

        def _allocate() -> Any:
            return jnp.empty(
                shape=cache_shape,
                dtype=cache_dtype,
            )

        sharded_allocate = jax.jit(_allocate, out_shardings=sharding)
        for _ in layer_names:
            k_cache = sharded_allocate()
            v_cache = sharded_allocate()
            self.kv_caches.append((k_cache, v_cache))

        # From @pooyam to @xiangxu: Feel free to edit the output_cache however you want. I added to unblock testing.
        self.output_cache = self.init_output_cache()

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

    def _prepare_inputs(self, scheduler_output: "VllmSchedulerOutput"):
        # We don't want to use ragged attention kernel all the time as paged attention is faster for decoding only.
        # NOTE(pooyam): For full prefill, is not clear which one is faster. We have to benchmark it.
        # NOTE(pooyam): If newer OSS ragged attention is also faster for decode, we should always used chunked prefill kernel and simplify the code.
        new_full_prefill_seqs, new_partial_prefill_seqs, subsequent_partial_prefill_seqs, decoding_seqs = self._get_prefill_and_decode_seqs(
            scheduler_output)

        has_new_full = bool(new_full_prefill_seqs)
        has_new_partial = bool(new_partial_prefill_seqs)
        has_subsequent_partial = bool(subsequent_partial_prefill_seqs)
        has_decoding = bool(decoding_seqs)

        if not (has_new_full or has_new_partial or has_subsequent_partial
                or has_decoding):
            return None

        # Check for the "no prefill" case
        if not new_full_prefill_seqs and not new_partial_prefill_seqs and not subsequent_partial_prefill_seqs:
            self.phase = "decode"
            return self._prepare_decode(scheduler_output)

        # Check for the "only full prefill" case
        # TODO(pooyam): We probably can use ragged attention for new_partial_prefills as well.
        if new_full_prefill_seqs and not new_partial_prefill_seqs and not subsequent_partial_prefill_seqs and not decoding_seqs:
            self.phase = "prefill"
            return self._prepare_prefill(scheduler_output)

        # All other cases fall into the "chunked prefill" category
        # TODO(pooyam): Change `prepare_prefill` to respect num scheduled tokens, so we can use prefill_kernel even for new partial prefills.
        # This is useful only if we conclude paged attention for prefill is still faster than ragged paged attention for prefill.
        self.phase = "chunked_prefill"
        return self._prepare_chunked_prefill(scheduler_output)

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

        inputs = self._prepare_inputs(scheduler_output)
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

            # TODO(pooyam): Figure out why all three of `num_tokens`, `num_prompt_tokens`, and 'num_computed_tokens_cpu` exist.
            prompt_logprobs_dict[seq.req_id] = None

        # TODO(pooyam): device-to-host transfer step by step is inefficient. Should we execute for longer decoding steps?
        # Not sure yet how that would work with vLLM engine that calls `execute_model`
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
        )

    def _init_mesh(self) -> None:
        mesh_shape = (1, len(self.devices))
        axis_names = ("data", "model")
        self.mesh = jax.make_mesh(mesh_shape, axis_names)
        logger.info(f"Init mesh | mesh={self.mesh}")

    def _init_model(self) -> None:
        self.model_fn, self.total_model_params_num = get_model(  # Unpack into local variables firstAdd commentMore actions
            self.vllm_config,
            self.rng_key,
            self.mesh,
        )

        # Prepare buffers used by chunk prefill
        max_num_running_seq = self.scheduler_config.max_num_seqs
        num_blocks_per_seq = (pad_to_multiple(
            self.model_config.max_model_len,
            self.cache_config.block_size,
        ) // self.cache_config.block_size)
        self.decode_seq_lens = np.zeros((max_num_running_seq, ),
                                        dtype=np.int32)
        self.decode_block_indices = np.zeros(
            (max_num_running_seq, num_blocks_per_seq), dtype=np.int32)
        self.prefill_seq_lens = np.zeros((MAX_PREFILL_SEQS_PER_TOKEN_BATCH, ),
                                         dtype=np.int32)
        self.prefill_block_indices = np.zeros(
            (MAX_PREFILL_SEQS_PER_TOKEN_BATCH, num_blocks_per_seq),
            dtype=np.int32)
        self.prefill_query_start_offsets = np.zeros(
            (MAX_PREFILL_SEQS_PER_TOKEN_BATCH + 1, ), dtype=np.int32)

    def _init_jit(self) -> None:
        self.outputs_sharding = NamedSharding(self.mesh, PartitionSpec(None))
        self.write_outputs = jax.jit(write_outputs,
                                     donate_argnums=0,
                                     out_shardings=self.outputs_sharding)
        self.read_outputs = jax.jit(read_outputs,
                                    out_shardings=self.outputs_sharding)

    def _device_array(self, *args, sharding=None, **kwargs) -> jax.Array:
        if sharding is None:
            sharding = NamedSharding(self.mesh, PartitionSpec(None))
        return jax.device_put(*args, device=sharding, **kwargs)

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

    # Modified from https://source.corp.google.com/h/vertex-model-garden/hex-llm/+/main:hex_llm/worker/runner_jax.py;drc=3ed287d21d5f95a053cb5fe3b249373064ac2f23;l=803.
    def _prepare_decode(self, scheduler_output: VllmSchedulerOutput) -> Any:
        if not len(scheduler_output.scheduled_cached_reqs):
            return None

        num_seqs = len(scheduler_output.scheduled_cached_reqs)
        block_size = self.cache_config.block_size
        sliding_window = self.model_config.get_sliding_window()
        sink_size = self.cache_config.sink_size
        cache_kv_before_rope = self.eviction_algorithm is not None
        max_model_len = self.model_config.max_model_len

        if sliding_window is not None:
            max_num_blocks = ((sliding_window + sink_size) //
                              block_size if sink_size else sliding_window //
                              block_size)
        else:
            max_possible_num_blocks = math.ceil(max_model_len / block_size)
            max_num_blocks = 0
            for seq in scheduler_output.scheduled_cached_reqs:
                seq_index = self.input_batch.req_id_to_index[seq.req_id]
                max_num_blocks = max(
                    max_num_blocks, self.input_batch.block_table.
                    block_tables[0].num_blocks_per_row[seq_index])

            max_num_blocks = pad_to_multiple(
                max_num_blocks,
                self.scheduler_config.decode_blocks_padding,
                max_possible_num_blocks,
                keep_one=True,
            )

        is_moe = hasattr(self.vllm_config.model_config.hf_config,
                         "num_local_experts")
        is_mistral = self.vllm_config.model_config.hf_config.model_type == "mistral"
        keep_one = not (
            (is_mistral and max_num_blocks > MAX_ALLOWED_PAGE_INDICES_N)
            or is_moe)
        batch_size = pad_to_multiple(
            num_seqs,
            self.decode_seqs_padding,
            self.max_num_reqs,
            keep_one=keep_one,
        )

        do_sampling = False
        running_indices = np.full((batch_size, ), -1, dtype=np.int32)
        input_token_indices = np.full((batch_size, ), -1, dtype=np.int32)
        input_positions = np.zeros([batch_size, 1], dtype=np.int32)
        seq_lens = np.zeros((batch_size, ), dtype=np.int32)
        input_ids = np.zeros((batch_size, ), dtype=np.int32)
        block_indices = np.zeros((batch_size, max_num_blocks), dtype=np.int32)
        kv_cache_write_indices = np.full((batch_size, ), -1, dtype=np.int32)
        temperatures = np.full((batch_size, ), 1.0, dtype=np.float32)
        top_ps = np.full((batch_size, ), 1.0, dtype=np.float32)
        top_ks = np.full((batch_size, ), 1, dtype=np.int32)
        output_token_indices = np.full((batch_size, ), -1, dtype=np.int32)
        kv_cache_position_indices = (np.zeros(
            (batch_size, max_num_blocks *
             block_size), dtype=np.int32) if cache_kv_before_rope else None)
        # Physical indices of the evicted tokens and the replacement tokens.
        evict_write_indices = (np.full((batch_size, ), -1, dtype=np.int32)
                               if self.eviction_algorithm is not None
                               and self.eviction_algorithm != "streamingllm"
                               else None)
        replacement_write_indices = (
            np.full((batch_size, ), -1, dtype=np.int32)
            if self.eviction_algorithm is not None
            and self.eviction_algorithm != "streamingllm" else None)

        for i, seq in enumerate(scheduler_output.scheduled_cached_reqs):
            seq_index = self.input_batch.req_id_to_index[seq.req_id]
            # num cached tokens + token of this decode
            seq_len = self.input_batch.num_computed_tokens_cpu[seq_index] + 1

            num_blocks = self.input_batch.block_table.block_tables[
                0].num_blocks_per_row[seq_index]
            block_table = self.input_batch.block_table.block_tables[
                0].block_table_cpu[seq_index, :num_blocks]

            position = seq_len - 1

            running_indices[i] = seq_index
            #input_token_indices[i] = seq.get_decoded_len() - 1
            # TODO(pooyam): Make sure this is correct
            input_token_indices[i] = self.input_batch.num_computed_tokens_cpu[
                seq_index] - self.input_batch.num_prompt_tokens[seq_index]
            assert input_token_indices[i] >= 0

            if self.eviction_algorithm in ["streamingllm", "h2o"]:
                input_positions[i][:] = [
                    min(position, sliding_window + sink_size - 1)
                ]
            else:
                input_positions[i][:] = [position]

            block_indices[i][:len(block_table)] = block_table

            if sliding_window is None:
                seq_lens[i] = seq_len
                assert position // block_size == len(block_table) - 1
                block_id = -1
                block_offset = position % block_size
            else:
                # Let's remove for now for simplicity.
                raise NotImplementedError("Refer to original impl in hex-llm.")

            kv_cache_write_indices[i] = (block_table[block_id] * block_size +
                                         block_offset)

            temperatures[i] = self.input_batch.temperature_cpu[seq_index]
            top_ps[i] = self.input_batch.top_p_cpu[seq_index]
            top_ks[i] = self.input_batch.top_k_cpu[seq_index]
            output_token_indices[i] = input_token_indices[i] + 1
            do_sampling = determine_do_sampling(top_ks[i], temperatures[i])

        running_indices = self._device_array(running_indices)
        input_token_indices = self._device_array(input_token_indices)

        input_ids = self.read_outputs(self.output_cache, running_indices,
                                      input_token_indices)

        # For perplexity experiments
        if self.perplexity_reference_text is not None:
            raise NotImplementedError("Not implemented.")
            assert len(scheduler_output.scheduled_seqs) == 1
            seq = scheduler_output.scheduled_seqs[0]
            seq_len = seq.get_prompt_len() + seq.get_decoded_len()
            input_ids = input_ids.at[0, 0].set(
                self.perplexity_reference_text[seq_len - 1])

        for seq in scheduler_output.scheduled_cached_reqs:
            req_id = seq.req_id
            seq_index = self.input_batch.req_id_to_index[req_id]

        (
            input_positions,
            seq_lens,
            block_indices,
            kv_cache_write_indices,
            temperatures,
            top_ps,
            top_ks,
            output_token_indices,
        ) = self._device_array((
            input_positions,
            seq_lens,
            block_indices,
            kv_cache_write_indices,
            temperatures,
            top_ps,
            top_ks,
            output_token_indices,
        ))

        if kv_cache_position_indices is not None:
            kv_cache_position_indices = self._device_array(
                kv_cache_position_indices)
        if evict_write_indices is not None:
            evict_write_indices = self._device_array(evict_write_indices)
        eviction_score_mask = None
        return (
            False,  # is prefill
            do_sampling,
            self.kv_caches,
            input_ids,
            AttentionMetadata(input_positions, seq_lens, block_indices,
                              kv_cache_write_indices),
            temperatures,
            top_ps,
            top_ks,
            kv_cache_position_indices,
            evict_write_indices,
            replacement_write_indices,
            eviction_score_mask,
        ), (
            running_indices,
            output_token_indices,
        )

    def _prepare_prefill(self, scheduler_output: VllmSchedulerOutput) -> Any:
        if not len(scheduler_output.scheduled_new_reqs):
            return None

        block_size = self.vllm_config.cache_config.block_size
        sliding_window = self.vllm_config.model_config.get_sliding_window()

        batch_size = len(scheduler_output.scheduled_new_reqs)
        batch_size = pad_to_multiple(
            batch_size,
            self.prefill_seqs_padding,
            self.scheduler_config.max_num_seqs,
            True,
        )

        # Full prompt length.
        max_prompt_len = max([
            len(seq.prompt_token_ids)
            for seq in scheduler_output.scheduled_new_reqs
        ])

        # Unfilled prompt length.
        # TODO: Fix this for prefix caching.
        max_unfilled_prompt_len = max_prompt_len

        # Padded full prompt length.
        padded_prompt_len = pad_to_multiple(
            max_prompt_len,
            self.scheduler_config.prefill_len_padding,
            self.model_config.max_model_len,
        )

        # Padded unfilled prompt length.
        padded_unfilled_prompt_len = pad_to_multiple(
            max_unfilled_prompt_len,
            self.scheduler_config.prefill_len_padding,
            self.model_config.max_model_len,
        )

        images_flattened = None

        if sliding_window:
            raise NotImplementedError("Sliding window not implemented.")
        else:
            padded_num_blocks = math.ceil(padded_prompt_len / block_size)
            # same as `padded_num_blocks` when no cache hit.
            padded_num_unfilled_blocks = math.ceil(padded_unfilled_prompt_len /
                                                   block_size)

        do_sampling = False
        input_ids = np.zeros((batch_size, padded_unfilled_prompt_len),
                             dtype=np.int32)
        input_positions = np.zeros((batch_size, padded_unfilled_prompt_len),
                                   dtype=np.int32)
        seq_lens = np.zeros((batch_size, ), dtype=np.int32)
        image_lens = np.zeros((batch_size, ), dtype=np.int32)
        block_indices = np.zeros((batch_size, padded_num_blocks),
                                 dtype=np.int32)
        kv_cache_write_indices = np.zeros(
            (batch_size, padded_num_unfilled_blocks), dtype=np.int32)
        temperatures = np.full((batch_size, ), 1.0, dtype=np.float32)
        top_ps = np.full((batch_size, ), 1.0, dtype=np.float32)
        top_ks = np.full((batch_size, ), 1, dtype=np.int32)
        running_indices = np.full((batch_size, ), -1, dtype=np.int32)
        output_token_indices = np.full((batch_size, ), -1, dtype=np.int32)

        eviction_score_mask = None

        for i, seq in enumerate(scheduler_output.scheduled_new_reqs):
            seq_index = self.input_batch.req_id_to_index[seq.req_id]

            effective_cached_prompt_len = 0
            num_effective_cached_blocks = 0
            prompt_token_ids = seq.prompt_token_ids[
                effective_cached_prompt_len:]
            prompt_len = len(prompt_token_ids)
            input_ids[i][:prompt_len] = prompt_token_ids
            input_positions[i][:prompt_len] = list(
                range(
                    effective_cached_prompt_len,
                    effective_cached_prompt_len + prompt_len,
                ))
            seq_lens[i] = prompt_len

            # Full prompt associated block indices.
            block_table = seq.block_ids[0]
            assert len(block_table) <= padded_num_blocks
            block_indices[i][:len(block_table)] = block_table
            # Unfilled prompt associated block indices.
            num_unfilled_blocks = len(
                block_table) - num_effective_cached_blocks
            assert num_unfilled_blocks <= padded_num_unfilled_blocks
            kv_cache_write_indices[i][:num_unfilled_blocks] = block_table[
                num_effective_cached_blocks:]
            temperatures[i] = self.input_batch.temperature_cpu[seq_index]
            top_ps[i] = self.input_batch.top_p_cpu[seq_index]
            top_ks[i] = self.input_batch.top_k_cpu[seq_index]

            running_indices[i] = seq_index

            # TODO(pooyam): double check this.
            output_token_indices[i] = seq.num_computed_tokens

            do_sampling = determine_do_sampling(
                seq.sampling_params.top_k, seq.sampling_params.temperature)
            if eviction_score_mask is not None:
                raise NotImplementedError("Evication not implemented.")

        input_ids = self._device_array(input_ids)
        input_positions = self._device_array(input_positions)
        seq_lens = self._device_array(seq_lens)
        temperatures = self._device_array(temperatures).astype(jnp.bfloat16)
        top_ps = self._device_array(top_ps).astype(jnp.bfloat16)
        top_ks = self._device_array(top_ks)
        block_indices = self._device_array(block_indices)
        kv_cache_write_indices = self._device_array(kv_cache_write_indices)
        running_indices = self._device_array(running_indices)
        output_token_indices = self._device_array(output_token_indices)
        kv_cache_position_indices = None
        evict_write_indices = None
        replacement_write_indices = None
        eviction_score_mask = (self._device_array(eviction_score_mask)
                               if eviction_score_mask is not None else None)

        return (
            True,
            do_sampling,
            self.kv_caches,
            input_ids,
            AttentionMetadata(input_positions, seq_lens, block_indices,
                              kv_cache_write_indices),
            temperatures,
            top_ps,
            top_ks,
            kv_cache_position_indices,
            evict_write_indices,
            replacement_write_indices,
            eviction_score_mask,
            images_flattened,
            image_lens,
        ), (
            running_indices,
            output_token_indices,
        )

    def _get_prefill_and_decode_seqs(
        self, scheduler_output: VllmSchedulerOutput
    ) -> Tuple[List[NewRequestData], List[CachedRequestData],
               List[CachedRequestData]]:
        # NOTE(pooyam): We categorize sequences into 4 different categories to have freedom to choose which kernel to use for each of them:
        # 1. full prefill: A sequence which is scheduled for a full prefill -> We can use prefill or chunked prefill kernel.
        # 2. new partial prefill: A sequence that its first chunk is scheduled for prefill -> We can use prefill or chunked prefill kernel.
        # 3. subsequent partial prefill: A sequence that its subsequent chunks are scheduled for prefill -> We should use chunked prefill kernel.
        # 4. decode: A sequence that has been completely prefilled already => We should use decode or chunked prefill kernel.
        # NOTE(pooyam): If God helps and we benchmark that ragged attention is superior in every scenario, we can simplify all these and get rid of this misery.

        new_full_prefill_seqs = []
        new_partial_prefill_seqs = []
        subsequent_partial_prefill_seqs = []
        decoding_seqs = []

        # NOTE(pooyam): The lesson learned was that `num_prompt_tokens` is not a reliable field to decide prefilling state based on solely.
        # The reason is that once a request is preempated and added back later, `num_prompt_tokens` still points to original number of prompts.
        # However, once the request is added back, we need to prefill previous prompt tokens AND previous output tokens again.
        # We should use `num_scheduled_tokens`, `num_computed_tokens` and **`num_tokens`** in most of the logics.
        for seq in (scheduler_output.scheduled_new_reqs +
                    scheduler_output.scheduled_cached_reqs):
            index = self.input_batch.req_id_to_index[seq.req_id]
            num_tokens = self.input_batch.num_tokens[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                seq.req_id]
            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[
                index]

            assert num_computed_tokens + num_scheduled_tokens <= num_tokens

            if num_computed_tokens + num_scheduled_tokens < num_tokens:
                if num_computed_tokens == 0:
                    new_partial_prefill_seqs.append(seq)
                else:
                    subsequent_partial_prefill_seqs.append(seq)
            else:
                if num_computed_tokens == 0:
                    new_full_prefill_seqs.append(seq)
                elif num_scheduled_tokens != 1:
                    subsequent_partial_prefill_seqs.append(seq)
                elif num_scheduled_tokens == 1:
                    # We need to distinguish the following cases:
                    # Case 1: not preempted at all:  prompt: 10 | output tokens: 4 | num_computed: 13 | num scheduled: 1 | num tokens: 14 => decoding and it has written sth before => write to position i=4 in output cache.
                    # Case 2: preempted and added back:  prompt: 10 | output tokens: 4 | num_computed: 13 | num_scheduled: 1 | num tokens: 14 => prefilling and it has not written sth before => write to position 0 in output cache.
                    # As you see, metadata related to Case 1 and Case 2 are exactly the same. Then how to distinguish? One idea is to use `resumed_from_preemption` field.
                    is_cached_req = isinstance(seq, CachedRequestData)
                    is_resumed_from_preemption = is_cached_req and seq.resumed_from_preemption
                    if not is_resumed_from_preemption:
                        if num_computed_tokens < self.input_batch.num_prompt_tokens[
                                index]:
                            subsequent_partial_prefill_seqs.append(seq)
                        else:
                            decoding_seqs.append(seq)
                    else:
                        subsequent_partial_prefill_seqs.append(seq)
                else:
                    raise ValueError("This should not happen.")

        return new_full_prefill_seqs, new_partial_prefill_seqs, subsequent_partial_prefill_seqs, decoding_seqs

    def _prepare_chunked_prefill(self,
                                 scheduler_output: VllmSchedulerOutput) -> Any:
        block_size = self.cache_config.block_size
        # in vLLMs scheduler output, scheduled_cached_reqs can mean two things: Subsequent prefill of an already seen request / or decode.

        new_full_prefill_seqs, new_partial_prefill_seqs, subsequent_partial_prefill_seqs, decoding_seqs = self._get_prefill_and_decode_seqs(
            scheduler_output)
        num_decode_seqs = len(decoding_seqs)
        num_prefill_seqs = len(new_full_prefill_seqs) + len(
            new_partial_prefill_seqs) + len(subsequent_partial_prefill_seqs)
        assert num_prefill_seqs > 0
        assert num_prefill_seqs <= MAX_PREFILL_SEQS_PER_TOKEN_BATCH

        # NOTE(pooyam): It's possible to remove this loop by approximating by upper bound.
        num_tokens_scheduled = pad_to_multiple(
            num_decode_seqs, block_size) if num_decode_seqs > 0 else 0
        for seq in new_full_prefill_seqs + new_partial_prefill_seqs + subsequent_partial_prefill_seqs:
            num_tokens_scheduled += pad_to_multiple(
                scheduler_output.num_scheduled_tokens[seq.req_id], block_size)
        num_tokens_scheduled = pad_to_multiple(
            num_tokens_scheduled,
            self.scheduler_config.chunked_prefill_tokens_padding)

        if num_decode_seqs > 0:
            decode_input_token_indices = np.full((num_tokens_scheduled, ),
                                                 -1,
                                                 dtype=np.int32)
        else:
            decode_input_token_indices = None

        do_sampling = False
        decode_seq_lens = self.decode_seq_lens
        decode_seq_lens[num_decode_seqs:] = 0
        decode_block_indices = self.decode_block_indices
        input_positions = np.zeros((1, num_tokens_scheduled), dtype=np.int32)
        decode_kv_cache_write_indices = np.full((num_tokens_scheduled, ),
                                                -1,
                                                dtype=np.int32)
        temperatures = np.full((num_tokens_scheduled, ), 1.0, dtype=np.float32)
        top_ps = np.full((num_tokens_scheduled, ), 1.0, dtype=np.float32)
        top_ks = np.full((num_tokens_scheduled, ), 1, dtype=np.int32)
        running_indices = np.full((num_tokens_scheduled, ), -1, dtype=np.int32)
        output_token_indices = np.full((num_tokens_scheduled, ),
                                       -1,
                                       dtype=np.int32)

        # Fill the token batch with decode tokens first.
        for i, seq in enumerate(decoding_seqs):
            seq_index = self.input_batch.req_id_to_index[seq.req_id]
            seq_len = self.input_batch.num_computed_tokens_cpu[seq_index] + 1

            num_blocks = self.input_batch.block_table.block_tables[
                0].num_blocks_per_row[seq_index]
            block_table = self.input_batch.block_table.block_tables[
                0].block_table_cpu[seq_index, :num_blocks]

            position = seq_len - 1

            running_indices[i] = seq_index
            decode_input_token_indices[
                i] = self.input_batch.num_computed_tokens_cpu[
                    seq_index] - self.input_batch.num_prompt_tokens[seq_index]
            assert decode_input_token_indices[i] >= 0
            input_positions[:, i] = position
            decode_block_indices[i][:len(block_table)] = block_table

            decode_seq_lens[i] = seq_len
            assert position // block_size == len(block_table) - 1
            block_id = -1
            block_offset = position % block_size
            decode_kv_cache_write_indices[i] = (
                block_table[block_id] * block_size + block_offset)
            temperatures[i] = self.input_batch.temperature_cpu[seq_index]
            top_ps[i] = self.input_batch.top_p_cpu[seq_index]
            top_ks[i] = self.input_batch.top_k_cpu[seq_index]
            output_token_indices[i] = decode_input_token_indices[i] + 1
            do_sampling = determine_do_sampling(top_ks[i], temperatures[i])

        token_offset = num_decode_seqs
        if num_decode_seqs > 0 and num_prefill_seqs > 0:
            # Add padding tokens so that prefill segments are paged aligned
            if num_decode_seqs % block_size != 0:
                token_offset = pad_to_multiple(num_decode_seqs, block_size)

        # Then fill the token batch with prefill tokens.
        prefill_seq_lens = self.prefill_seq_lens
        prefill_seq_lens[num_prefill_seqs:] = 0
        prefill_block_indices = self.prefill_block_indices
        prefill_query_start_offsets = self.prefill_query_start_offsets
        # One cache update index per page for prefill.
        assert num_tokens_scheduled % block_size == 0
        prefill_kv_cache_write_indices = np.full(
            (num_tokens_scheduled // block_size, ), -1, dtype=np.int32)
        prefill_input_ids = np.zeros((1, num_tokens_scheduled), dtype=np.int32)
        for i, seq in enumerate(new_full_prefill_seqs +
                                new_partial_prefill_seqs +
                                subsequent_partial_prefill_seqs):
            seq_index = self.input_batch.req_id_to_index[seq.req_id]
            num_tokens = self.input_batch.num_tokens[seq_index]
            prefill_len = scheduler_output.num_scheduled_tokens[seq.req_id]
            num_prefilled_tokens = self.input_batch.num_computed_tokens_cpu[
                seq_index]
            # TODO(pooyam): How to make sure we are not reading beyond prefill?
            prefill_input_ids[:, token_offset:token_offset +
                              prefill_len] = (self.input_batch.token_ids_cpu[
                                  seq_index,
                                  num_prefilled_tokens:num_prefilled_tokens +
                                  prefill_len])
            input_positions[:, token_offset:token_offset + prefill_len] = list(
                range(
                    num_prefilled_tokens,
                    num_prefilled_tokens + prefill_len,
                ))
            prefill_seq_lens[i] = pad_to_multiple(
                num_prefilled_tokens + prefill_len, block_size)
            prefill_query_start_offsets[i] = token_offset
            num_blocks = self.input_batch.block_table.block_tables[
                0].num_blocks_per_row[seq_index]
            block_table = self.input_batch.block_table.block_tables[
                0].block_table_cpu[seq_index, :num_blocks]

            prefill_block_indices[i][:len(block_table)] = block_table
            prefill_kv_cache_write_indices[
                token_offset // block_size:math.ceil(
                    (token_offset + prefill_len) /
                    block_size)] = block_table[num_prefilled_tokens //
                                               block_size:math.ceil(
                                                   (num_prefilled_tokens +
                                                    prefill_len) / block_size)]

            assert num_prefilled_tokens + prefill_len <= num_tokens

            if num_prefilled_tokens + prefill_len == num_tokens:
                # only in this case, a new decode token will be generated
                last_prefill_token_idx = token_offset + prefill_len - 1
                running_indices[last_prefill_token_idx] = seq_index

                # Hex-LLM equivalent: output_token_indices[last_prefill_token_idx] = seq.get_decoded_len()
                # NOTE(pooyam): Is there a case where this shouldn't be 0?
                output_token_indices[last_prefill_token_idx] = 0

                temperatures[
                    last_prefill_token_idx] = self.input_batch.temperature_cpu[
                        seq_index]
                top_ps[last_prefill_token_idx] = self.input_batch.top_p_cpu[
                    seq_index]
                top_ks[last_prefill_token_idx] = self.input_batch.top_k_cpu[
                    seq_index]

            do_sampling = determine_do_sampling(
                self.input_batch.top_k_cpu[seq_index],
                self.input_batch.temperature_cpu[seq_index])
            # Add padding tokens so that prefill segments are paged aligned
            token_offset = pad_to_multiple(token_offset + prefill_len,
                                           block_size)
        prefill_query_start_offsets[num_prefill_seqs:] = token_offset

        # Concat the kv cache write indices for decode tokens and prefill tokens
        kv_cache_write_indices = np.concatenate(
            (decode_kv_cache_write_indices, prefill_kv_cache_write_indices))
        num_decode_seqs_arr = np.array([num_decode_seqs], np.int32)
        num_prefill_seqs_arr = np.array([num_prefill_seqs], np.int32)

        (
            prefill_input_ids,
            decode_input_token_indices,
            input_positions,
            temperatures,
            top_ps,
            top_ks,
            kv_cache_write_indices,
            running_indices,
            output_token_indices,
            decode_seq_lens,
            decode_block_indices,
            num_decode_seqs_arr,
            prefill_seq_lens,
            prefill_block_indices,
            prefill_query_start_offsets,
            num_prefill_seqs_arr,
        ) = self._device_array((
            prefill_input_ids,
            decode_input_token_indices,
            input_positions,
            temperatures,
            top_ps,
            top_ks,
            kv_cache_write_indices,
            running_indices,
            output_token_indices,
            decode_seq_lens,
            decode_block_indices,
            num_decode_seqs_arr,
            prefill_seq_lens,
            prefill_block_indices,
            prefill_query_start_offsets,
            num_prefill_seqs_arr,
        ))
        # Merge decode tokens with prefill tokens.
        if num_decode_seqs == 0:
            input_ids = prefill_input_ids
        else:
            decode_input_ids = jnp.swapaxes(
                self.read_outputs(self.output_cache, running_indices,
                                  decode_input_token_indices),
                0,
                1,
            )
            input_ids = jnp.where(
                jnp.arange(num_tokens_scheduled) < num_decode_seqs,
                decode_input_ids,
                prefill_input_ids,
            )
        kv_cache_position_indices = None
        evict_write_indices = None
        replacement_write_indices = None
        eviction_score_mask = None
        return (
            False,  # when chunked prefill is enabled, `is_prefill` is just a dummy value.
            do_sampling,
            self.kv_caches,
            input_ids,
            AttentionMetadata(
                input_positions=input_positions,
                seq_lens=None,  # use decode_lengths / prefill_lengths instead
                block_indices=
                None,  # use decode_page_indices / prefill_page_indices instead
                kv_cache_write_indices=kv_cache_write_indices,
                chunked_prefill_enabled=True,
                decode_lengths=decode_seq_lens,
                decode_page_indices=decode_block_indices,
                num_decode_seqs=num_decode_seqs_arr,
                prefill_lengths=prefill_seq_lens,
                prefill_page_indices=prefill_block_indices,
                prefill_query_start_offsets=prefill_query_start_offsets,
                num_prefill_seqs=num_prefill_seqs_arr,
            ),
            temperatures,
            top_ps,
            top_ks,
            kv_cache_position_indices,
            evict_write_indices,
            replacement_write_indices,
            eviction_score_mask,
        ), (
            running_indices,
            output_token_indices,
        )


def pad_to_multiple(x: int,
                    multiple: int = 8,
                    max_limit: Optional[int] = None,
                    keep_one: bool = False) -> int:
    assert x > 0
    if keep_one and x == 1:
        return x
    x = x + (-x % multiple)
    if max_limit is not None:
        x = min(x, max_limit)
    return x


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
