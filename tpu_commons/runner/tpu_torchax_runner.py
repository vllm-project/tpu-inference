# ruff: noqa
# isort: skip_file
# yapf: disable
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import bisect
import gc
import os
import time
from typing import TYPE_CHECKING, Optional, cast
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from torch.utils import _pytree as pytree
import vllm.envs as envs

import jax
import jax.numpy as jnp
import torchax
from tpu_commons.models.torchax.torchax_wrapper import (
    get_cpu_tensor_from_torchax_tensor, wrap_model, wrap_model_func)
from tpu_commons.distributed.tpu_distributed_utils import (
    create_torchax_kv_cache, create_torchax_tensor_with_partition_spec)

from vllm.model_executor.models.interfaces import supports_transcription
from vllm.model_executor.models.interfaces_base import (
    is_pooling_model, is_text_generation_model)
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
from vllm.config import ParallelConfig, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from tpu_commons.logger import init_logger
from tpu_commons.runner.utils import (get_padded_num_reqs_with_upper_limit,
                                      get_padded_token_len, get_req_paddings,
                                      get_token_paddings, MIN_NUM_SEQS)
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.model_executor.model_loader import get_model_loader

from tpu_commons.models.torchax.tpu import TPUModelLoader

from vllm.sequence import IntermediateTensors
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType, cdiv,
                        is_pin_memory_available)

from tpu_commons.attention.backends.pallas_torchax import (
    PallasAttentionBackend, PallasMetadata)

from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheConfig, KVCacheSpec,
                                        SlidingWindowSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, LogprobsTensors,
                             ModelRunnerOutput)
from vllm.v1.sample.tpu.metadata import TPUSupportedSamplingMetadata
from vllm.v1.sample.tpu.sampler import Sampler as TPUSampler
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.tpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.utils import (bind_kv_cache,
                                  initialize_kv_cache_for_kv_sharing)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

INVALID_TOKEN_ID = -1


#########################################################
# Ways to avoid recompilation
#########################################################
#
# The model executor has two primary components:
# 1. preparing the model and sampler inputs
# 2. executing the model and sampler.
# The core idea is to avoid any TPU computation during input preparation. For
# better compilation tracking and increased flexibility, the model execution and
# sampler are divided into several distinct components.
#
# Below are the detailed steps:
#
# Step 1
# It is recommended to avoid TPU operations when preparing the model and sampler
# inputs. CPU tensors can be prepared and transferred to the XLA device using
# cpu_tensor.to(xla_device), which only triggers CPU to TPU transfers and avoids
# compilation.
#
# Step 2
# The TPU execution should be decomposed into subgraphs (4 at the moment):
# 1. the main model
# 2. selecting hidden states for each request
# 3. sampler
# 4. encoder.
# Each subgraph should be decorated in a torch.compile. This is used to make
# sure that we have the same subgraph topology in both dummy_run and
# xecute_model. The results from these subgraphs should either be passed to
# other subgraphs, or transferred from TPU to CPU using xla_tensor.cpu() for
# subsequent processing on the CPU.
#
# Step 3
# The dummy_run should be comprehensive, ensuring all potential input shapes and
# branch predictions are included as subgraph inputs to facilitate
# pre-compilation.
class TPUModelRunner(LoRAModelRunnerMixin):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        original_parallel_config: Optional[ParallelConfig] = None,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.original_parallel_config = original_parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.device_config = vllm_config.device_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.check_recompilation = envs.VLLM_XLA_CHECK_RECOMPILATION

        # SPMD Related
        self.use_spmd = envs.VLLM_XLA_USE_SPMD
        self.mesh = None
        if self.use_spmd:
            devices = jax.devices()
            self.mesh = jax.sharding.Mesh(devices, axis_names=('x', ))

        self.enforce_eager = model_config.enforce_eager

        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]
        self._hidden_states_dtype = self.dtype

        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        # InputBatch needs to work with sampling tensors greater than padding
        # to avoid dynamic shapes. Also, avoid suboptimal alignment.
        self.max_num_reqs = max(scheduler_config.max_num_seqs, MIN_NUM_SEQS)
        self.num_tokens_paddings = get_token_paddings(
            min_token_size=16,
            max_token_size=scheduler_config.max_num_batched_tokens,
            padding_gap=envs.VLLM_TPU_BUCKET_PADDING_GAP)
        # In case `max_num_tokens < max(num_tokens_paddings)` use the actual
        # padded max value to pre-allocate data structures and pre-compile.
        self.max_num_tokens = self.num_tokens_paddings[-1]

        # Model-related.
        self.num_attn_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_query_heads = model_config.get_num_attention_heads(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_size = model_config.get_head_size()
        self.hidden_size = model_config.get_hidden_size()
        self.vocab_size = model_config.get_vocab_size()

        # Torchax runner doesn't support lora now
        assert self.lora_config is None

        # Lazy initialization
        self.model: nn.Module  # Set after load_model
        self.kv_caches: list[torch.Tensor] = []
        self.kv_caches_dict: dict[str, torch.Tensor] = dict()
        self.model_func = None
        self.compute_logits_func = None
        self.torchax_env = torchax.default_env()

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}

        # Initialize input batch early to avoid AttributeError in _update_states
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
        )

        # Cached torch/numpy tensor
        # The pytorch tensor and numpy array share the same buffer.
        # Sometimes the numpy op is faster so we create both.
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu")

        self.positions_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu")
        self.positions_np = self.positions_cpu.numpy()

        self.block_table_cpu = torch.zeros(
            (self.max_num_reqs, self.max_num_blocks_per_req),
            dtype=torch.int32,
            device="cpu")

        self.query_start_loc_cpu = torch.zeros(self.max_num_tokens + 1,
                                               dtype=torch.int32,
                                               device="cpu",
                                               pin_memory=self.pin_memory)
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()

        self.seq_lens_cpu = torch.zeros(self.max_num_tokens,
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=self.pin_memory)
        self.seq_lens_np = self.seq_lens_cpu.numpy()

        # Range tensor with values [0 .. self.max_num_tokens - 1].
        # Used to initialize positions / context_lens / seq_lens
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(self.max_num_tokens, dtype=np.int64)
        self.num_reqs_paddings = get_req_paddings(
            min_req_size=MIN_NUM_SEQS, max_req_size=self.max_num_reqs)

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}

    def _update_states(self, scheduler_output: "SchedulerOutput") -> bool:
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
            assert new_req_data.sampling_params is not None,\
                "Pooling is not supported in TPU yet"
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                pooling_params=None,
                generator=None,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_data.resumed_from_preemption[i]

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens
            if not resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                for block_ids, new_ids in zip(req_state.block_ids,
                                              new_block_ids):
                    block_ids.extend(new_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)
            self.input_batch.block_table.append_row(new_block_ids, req_index)

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

    def get_model(self) -> nn.Module:
        return self.model

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return ("generate", )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_name, attn_module in layers.items():
            if (kv_tgt_layer :=
                    attn_module.kv_sharing_target_layer_name) is not None:
                # The layer doesn't need its own KV cache and will use that of
                # the target layer. We skip creating a KVCacheSpec for it, so
                # that KV cache management logic will act as this layer does
                # not exist, and doesn't allocate KV cache for the layer. This
                # enables the memory saving of cross-layer kv sharing, allowing
                # a given amount of memory to accommodate longer context lengths
                # or enable more requests to be processed simultaneously.
                self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                continue

            if attn_module.attn_type == AttentionType.DECODER:
                if attn_module.sliding_window is not None:
                    kv_cache_spec[layer_name] = SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        sliding_window=attn_module.sliding_window,
                        use_mla=False,
                    )
                else:
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        use_mla=False,
                    )
            elif attn_module.attn_type in (AttentionType.ENCODER,
                                           AttentionType.ENCODER_ONLY):
                # encoder-only attention does not need KV cache.
                continue
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown attention type: {attn_module.attn_type}")

        return kv_cache_spec

    def _get_slot_mapping_metadata(self, num_reqs,
                                   num_scheduled_tokens_per_req):
        """
        Computes metadata for mapping slots to blocks in the key-value (KV)
        cache for a batch of requests.

        This function determines, for each request in the batch, how the
        scheduled tokens are distributed across memory blocks, and generates
        metadata needed to map slices of tokens to their corresponding positions
        in the KV cache.

        Args:
            num_reqs (int): Number of requests in the current batch.
            num_scheduled_tokens_per_req (int or np.ndarray): Number of tokens
            to be scheduled for each request.

        Returns:
            np.ndarray: A 2D array of shape (total_block_len, 3), where each row
            contains:
                - kv_cache_start_index (int): The starting index in the KV cache
                    for the corresponding slice.
                - new_kv_start_index (int): The starting index in the new KV
                    cache for the corresponding slice.
                - slice_len (int): The length of the slice.
        """
        slices_start_np = self.input_batch.num_computed_tokens_cpu[:num_reqs]
        slices_end_np = self.input_batch.num_computed_tokens_cpu[:num_reqs] + \
            num_scheduled_tokens_per_req
        local_block_start_idx_np = slices_start_np // self.block_size
        local_block_end_idx_np = (slices_end_np - 1) // self.block_size
        no_repeat_req_indices_np = self.arange_np[:num_reqs]
        global_block_start_idx_np = (
            no_repeat_req_indices_np * self.max_num_blocks_per_req +
            local_block_start_idx_np)
        block_lens_np = local_block_end_idx_np - local_block_start_idx_np + 1
        global_block_start_idx_np = np.repeat(global_block_start_idx_np,
                                              block_lens_np)
        slice_arange_np = np.concatenate(
            [self.arange_np[:n] for n in block_lens_np])
        global_block_indices_np = global_block_start_idx_np + slice_arange_np
        block_table_cpu = self.input_batch.block_table[0].get_cpu_tensor()
        block_numbers_np = block_table_cpu.flatten(
        )[global_block_indices_np].numpy()
        total_block_len = np.sum(block_lens_np)
        slot_mapping_slices_np = np.repeat(np.array([[0, self.block_size]],
                                                    dtype=np.int32),
                                           total_block_len,
                                           axis=0)
        cu_block_lens_np = np.zeros(len(block_lens_np) + 1, dtype=np.int32)
        np.cumsum(block_lens_np, out=cu_block_lens_np[1:])
        for req_idx in range(num_reqs):
            slot_mapping_slices_np[cu_block_lens_np[req_idx]][
                0] = slices_start_np[req_idx] % self.block_size
            slot_mapping_slices_np[
                cu_block_lens_np[req_idx + 1] -
                1][1] = (slices_end_np[req_idx] - 1) % self.block_size + 1
        slice_lens_np = slot_mapping_slices_np[:,
                                               1] - slot_mapping_slices_np[:,
                                                                           0]
        cu_slices_lens_np = np.zeros(len(slice_lens_np) + 1, dtype=np.int32)
        np.cumsum(slice_lens_np, out=cu_slices_lens_np[1:])
        kv_cache_start_indices_np = slot_mapping_slices_np[:, 0] + \
            (block_numbers_np * self.block_size)
        new_kv_start_indices_np = cu_slices_lens_np[:-1]
        slot_mapping_metadata_np = np.stack([
            kv_cache_start_indices_np, new_kv_start_indices_np, slice_lens_np
        ],
                                            axis=1)
        return slot_mapping_metadata_np

    def _prepare_inputs(self, scheduler_output: "SchedulerOutput"):
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0
        # torchax runner doesn't support lora.
        assert self.lora_config is None

        # Get the number of scheduled tokens for each request.
        num_scheduled_tokens_per_req = []
        max_num_scheduled_tokens_all_reqs = 0
        for req_id in self.input_batch.req_ids[:num_reqs]:
            assert req_id is not None
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens_per_req.append(num_tokens)
            max_num_scheduled_tokens_all_reqs = max(
                max_num_scheduled_tokens_all_reqs, num_tokens)
        num_scheduled_tokens_per_req = np.array(num_scheduled_tokens_per_req,
                                                dtype=np.int32)
        assert max_num_scheduled_tokens_all_reqs > 0

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        # For each scheduled token, what are the corresponding req index.
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens_per_req)

        # Get batched arange.
        # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # For each scheduled token, what is its position in corresponding req.
        arange = np.concatenate(
            [self.arange_np[:n] for n in num_scheduled_tokens_per_req])

        # Get positions.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens_per_req,
                  out=self.query_start_loc_np[1:num_reqs + 1])
        self.query_start_loc_np[num_reqs + 1:] = 1

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens_per_req)

        # Do the padding and copy the tensors to the TPU.
        padded_total_num_scheduled_tokens = get_padded_token_len(
            self.num_tokens_paddings, total_num_scheduled_tokens)
        # Zero out to avoid spurious values from prev iteration (last cp chunk)
        self.input_ids_cpu[
            total_num_scheduled_tokens:padded_total_num_scheduled_tokens] = 0

        # Calculate the slot mapping.
        slot_mapping_metadata_np = self._get_slot_mapping_metadata(
            num_reqs, num_scheduled_tokens_per_req)
        num_slices = slot_mapping_metadata_np.shape[0]
        padded_num_slices = _get_padded_num_kv_cache_update_slices(
            padded_total_num_scheduled_tokens, self.max_num_reqs,
            self.block_size)
        slot_mapping_metadata_np = np.pad(
            slot_mapping_metadata_np,
            [[0, padded_num_slices - len(slot_mapping_metadata_np)], [0, 0]],
            constant_values=0)
        slot_mapping_metadata_np = np.transpose(slot_mapping_metadata_np)

        self.input_ids = create_torchax_tensor_with_partition_spec(
            self.input_ids_cpu[:padded_total_num_scheduled_tokens], self.mesh)
        self.position_ids = create_torchax_tensor_with_partition_spec(
            self.positions_cpu[:padded_total_num_scheduled_tokens], self.mesh)
        slot_mapping = create_torchax_tensor_with_partition_spec(
            torch.from_numpy(slot_mapping_metadata_np), self.mesh)
        block_tables = self.block_table_cpu[:self.max_num_reqs]
        block_tables[:num_reqs, :self.max_num_blocks_per_req] = (
            self.input_batch.block_table[0].get_cpu_tensor()[:num_reqs])
        block_tables = create_torchax_tensor_with_partition_spec(
            block_tables, self.mesh)
        query_start_loc = create_torchax_tensor_with_partition_spec(
            self.query_start_loc_cpu[:self.max_num_reqs + 1], self.mesh)
        seq_lens = create_torchax_tensor_with_partition_spec(
            self.seq_lens_cpu[:self.max_num_reqs], self.mesh)

        num_seqs = create_torchax_tensor_with_partition_spec(
            torch.tensor([num_reqs], dtype=torch.int32), self.mesh)
        num_slices = create_torchax_tensor_with_partition_spec(
            torch.tensor([num_slices], dtype=torch.int32), self.mesh)
        attn_metadata = PallasMetadata(
            slot_mapping=slot_mapping.jax(),
            block_tables=block_tables.jax(),
            context_lens=seq_lens.jax(),
            query_start_loc=query_start_loc.jax(),
            num_seqs=num_seqs.jax(),
            num_slices=num_slices.jax(),
        )
        # NOTE(woosuk): Due to chunked prefills, there can be at most 1 partial
        # request in the batch. While we should not sample any token from this
        # partial request, we do so for simplicity. We will ignore the sampled
        # token from the partial request.
        # TODO: Support prompt logprobs.
        padded_num_reqs = get_padded_num_reqs_with_upper_limit(
            num_reqs, self.max_num_reqs)
        # Indices at which we sample (positions of last token in the sequence).
        # Padded to avoid recompiling when `num_reqs` varies.
        logits_indices = self.query_start_loc_cpu[1:padded_num_reqs + 1] - 1
        logits_indices = create_torchax_tensor_with_partition_spec(
            logits_indices, self.mesh).jax()

        layer_names = get_layers_from_vllm_config(self.vllm_config,
                                                  Attention).keys()
        per_layer_attn_metadata = {
            layer_name: attn_metadata
            for layer_name in layer_names
        }
        return per_layer_attn_metadata, logits_indices, padded_num_reqs

    def _get_model_inputs(self, input_ids: torch.Tensor,
                          mm_embeds: list[torch.Tensor]):
        # For text-only models, we use token ids as input.
        # While it is possible to use embeddings as input just like the
        # multimodal models, it is not desirable for performance since
        # then the embedding layer is not included in the CUDA graph.
        return input_ids, None

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> ModelRunnerOutput:
        # Multi-model is not supported
        assert intermediate_tensors is None, "Multi-model is not supported"
        assert scheduler_output.grammar_bitmask is None, \
            "Structured decoding is not supported."

        # Update cached state
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOutput if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT

        mm_embeds = []
        # Prepare inputs
        attn_metadata, logits_indices, padded_num_reqs = self._prepare_inputs(
            scheduler_output)
        input_ids, inputs_embeds = self._get_model_inputs(
            self.input_ids, mm_embeds)
        num_reqs = self.input_batch.num_reqs
        # Run the decoder
        input_args = (input_ids.jax(), self.position_ids.jax())
        num_scheduled_tokens_padded = input_ids.shape[0]
        hidden_states, new_kv_caches = self.model_func(
            self.params_and_buffers_jax, input_args, self.kv_caches_dict,
            attn_metadata, num_scheduled_tokens_padded)
        # Set the new KV caches to the static forward context.
        static_forward_context = self.vllm_config.compilation_config.\
                                        static_forward_context
        for layer_name, kv_cache in new_kv_caches.items():
            # NOTE: Use list because of virtual engine.
            static_forward_context[layer_name].kv_cache = [kv_cache]
            self.kv_caches_dict[layer_name] = kv_cache[0]
        with torchax.default_env():
            hidden_states = self.select_hidden_states(hidden_states,
                                                      logits_indices)
            logits = self.compute_logits_func(self.params_and_buffers_jax,
                                              hidden_states, None)
            tpu_sampling_metadata = TPUSupportedSamplingMetadata.\
                from_input_batch(self.input_batch, padded_num_reqs, self.device)
            selected_token_ids = self.sample_from_logits(logits)
            selected_token_ids = torchax.tensor.Tensor(selected_token_ids,
                                                       self.torchax_env)

            # Remove padding on cpu and keep dynamic op outside of xla graph.
            if self.use_spmd:
                selected_token_ids = get_cpu_tensor_from_torchax_tensor(
                    selected_token_ids)
                selected_token_ids = selected_token_ids[:num_reqs]
            else:
                selected_token_ids = selected_token_ids.cpu()[:num_reqs]

        # Update the cache state concurrently. Code above will not block until
        # we use `selected_token_ids`. Add mark_step if post-processing changes
        request_seq_lens: list[tuple[int, CachedRequestState, int]] = []
        discard_sampled_tokens_req_indices = []
        for i, req_id in zip(range(num_reqs), self.input_batch.req_ids):
            assert req_id is not None
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len >= req_state.num_tokens:
                request_seq_lens.append((i, req_state, seq_len))
            else:
                # Ignore the sampled token from the partial request.
                # Rewind the generator state as if the token was not sampled.
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    # This relies on cuda-specific torch-internal impl details
                    generator.set_offset(generator.get_offset() - 4)

                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)

        assert all(
            req_id is not None for req_id in
            self.input_batch.req_ids[:num_reqs]), "req_ids contains None"
        req_ids = cast(list[str], self.input_batch.req_ids[:num_reqs])

        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}
        for req_id in self.input_batch.req_ids[:num_reqs]:
            prompt_logprobs_dict[req_id] = None

        max_gen_len = selected_token_ids.shape[-1]
        if max_gen_len == 1:
            valid_sampled_token_ids = selected_token_ids.tolist()

            # Mask out the sampled tokens that should not be sampled.
            # TODO: Keep in sync with gpu_model_runner.py, in particular
            #       the "else" case here
            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()

            # Append sampled tokens
            for i, req_state, seq_len in request_seq_lens:
                token_id = valid_sampled_token_ids[i][0]
                self.input_batch.token_ids_cpu[i, seq_len] = token_id
                req_state.output_token_ids.append(token_id)
                self.input_batch.num_tokens[i] += 1

        else:
            valid_mask = selected_token_ids != INVALID_TOKEN_ID
            gen_lens = valid_mask.sum(dim=1).tolist()
            valid_sampled_token_ids = [
                seq.tolist()
                for seq in selected_token_ids[valid_mask].split(gen_lens)
            ]
            self.input_batch.num_tokens[:num_reqs] += gen_lens
            for i, req_state, seq_len in request_seq_lens:
                target_slice = slice(seq_len - gen_lens[i] + 1, seq_len + 1)
                self.input_batch.token_ids_cpu[
                    i, target_slice] = valid_sampled_token_ids[i]
                req_state.output_token_ids.extend(valid_sampled_token_ids[i])

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
        )

        # Check there are no new graphs compiled - all the graphs should be
        # captured and compiled during warm up.
        return model_runner_output

    def load_model(self) -> None:
        self.device = self.device_config.device

        # NOTE(woosuk): While the executor assigns the TP ranks to the worker
        # process, the ranks can be different from the ranks internally assigned
        # by the JAX runtime. Therefore, there is a mismatch in the rank
        # assignment between the gloo (cpu) runtime and the JAX (tpu) runtime.
        # This is not a problem in linear layers because all-reduce is
        # rank-agnostic. However, it matters for all-gather as the ranks
        # determine the order of concatenating the output tensors.
        # As a workaround, we use the JAX rank assignment only when loading
        # the embedding weights.
        jax_tp_rank = jax.process_index()

        tpu_loader = TPUModelLoader(load_config=self.vllm_config.load_config)
        model = tpu_loader.load_model(
            vllm_config=self.vllm_config,
            model_config=self.vllm_config.model_config,
            mesh=self.mesh)

        # Extract all params and buffers for functional call.
        torchax.enable_globally()
        # with self.torchax_env:
        from torchax.interop import extract_all_buffers
        params, buffers = extract_all_buffers(model)
        # Need to explicitly move to jax device, because `model.to('jax')`
        # won't move tensors on python attributes to jax device.
        self.params_and_buffers = {**params, **buffers}
        for name, tensor in self.params_and_buffers.items():
            if not isinstance(tensor, torchax.tensor.Tensor):
                self.params_and_buffers[name] = \
                    create_torchax_tensor_with_partition_spec(tensor,
                                                                self.mesh,
                                                                ())
        self.params_and_buffers_jax = \
            pytree.tree_map_only(torch.Tensor, lambda x : x.jax(),
                                self.params_and_buffers)

        # Create a function for model.forward
        static_forward_context = \
            self.vllm_config.compilation_config.static_forward_context
        wrapped_model_forward = wrap_model(
            model,
            self.vllm_config,
            static_forward_context,
        )
        self.model_func = wrapped_model_forward

        # Create a function for model.compute_logits
        self.compute_logits_func = wrap_model_func(model, "compute_logits")

        torchax.disable_globally()

        if not hasattr(self, "model"):
            self.model = model

    @torch.no_grad()
    def _dummy_run(self, num_tokens: int) -> None:
        input_ids = create_torchax_tensor_with_partition_spec(
            torch.zeros((num_tokens), dtype=torch.int32), self.mesh)
        inputs_embeds = None
        actual_num_reqs = min(num_tokens, self.max_num_reqs)
        position_ids = create_torchax_tensor_with_partition_spec(
            torch.zeros(num_tokens, dtype=torch.int32), self.mesh)
        padded_num_slices = _get_padded_num_kv_cache_update_slices(
            num_tokens, self.max_num_reqs, self.block_size)
        slot_mapping = create_torchax_tensor_with_partition_spec(
            torch.zeros((3, padded_num_slices), dtype=torch.int64), self.mesh)
        block_tables = create_torchax_tensor_with_partition_spec(
            torch.zeros((self.max_num_reqs, self.block_table_cpu.shape[1]),
                        dtype=torch.int32), self.mesh)
        query_lens = [1] * self.max_num_reqs
        query_start_loc = create_torchax_tensor_with_partition_spec(
            torch.cumsum(torch.tensor([0] + query_lens, dtype=torch.int32),
                         dim=0,
                         dtype=torch.int32), self.mesh)
        context_lens = create_torchax_tensor_with_partition_spec(
            torch.ones((self.max_num_reqs, ), dtype=torch.int32), self.mesh)
        num_seqs = create_torchax_tensor_with_partition_spec(
            torch.tensor([actual_num_reqs], dtype=torch.int32), self.mesh)
        num_slices = create_torchax_tensor_with_partition_spec(
            torch.tensor([padded_num_slices], dtype=torch.int32), self.mesh)
        attn_metadata = PallasMetadata(
            slot_mapping=slot_mapping.jax(),
            block_tables=block_tables.jax(),
            context_lens=context_lens.jax(),
            query_start_loc=query_start_loc.jax(),
            num_seqs=num_seqs.jax(),
            num_slices=num_slices.jax(),
        )

        layer_names = get_layers_from_vllm_config(self.vllm_config,
                                                  Attention).keys()
        per_layer_attn_metadata = {
            layer_name: attn_metadata
            for layer_name in layer_names
        }

        input_args = (input_ids.jax(), position_ids.jax())
        out, new_kv_caches = self.model_func(self.params_and_buffers_jax,
                                             input_args, self.kv_caches_dict,
                                             per_layer_attn_metadata,
                                             num_tokens)
        # Set the new KV caches to the static forward context.
        static_forward_context = \
            self.vllm_config.compilation_config.static_forward_context
        for layer_name, kv_cache in new_kv_caches.items():
            # NOTE: Use list because of virtual engine.
            static_forward_context[layer_name].kv_cache = [kv_cache]
            self.kv_caches_dict[layer_name] = kv_cache[0]
        self._hidden_states_dtype = torchax.ops.mappings.j2t_dtype(out.dtype)

    def _precompile_backbone(self) -> None:
        logger.info("Compiling the model with different input shapes.")
        start = time.perf_counter()
        for num_tokens in self.num_tokens_paddings:
            logger.info("  -- num_tokens: %d", num_tokens)
            self._dummy_run(num_tokens)
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)

    def _precompile_select_hidden_states(self) -> None:
        # Compile hidden state selection function for bucketed
        # n_tokens x max_num_reqs. Graph is really small so this is fine.
        logger.info(
            "Compiling select_hidden_states with different input shapes.")
        start = time.perf_counter()
        hsize = self.model_config.get_hidden_size()
        for num_tokens in self.num_tokens_paddings:
            dummy_hidden = create_torchax_tensor_with_partition_spec(
                torch.zeros((num_tokens, hsize),
                            dtype=self._hidden_states_dtype), self.mesh).jax()
            for num_reqs in self.num_reqs_paddings:
                indices = create_torchax_tensor_with_partition_spec(
                    torch.zeros(num_reqs, dtype=torch.int32), self.mesh).jax()
                self.select_hidden_states(dummy_hidden, indices)
                logger.info("  -- num_tokens: %d, num_seqs: %d", num_tokens,
                            num_reqs)
                # Requests can't be more than tokens. But do compile for the
                # next bigger value in case num_tokens uses bucketed padding.
                if num_reqs >= min(num_tokens, self.max_num_reqs):
                    break
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)

    def _precompile_compute_logits(self) -> None:
        logger.info("Compiling compute_logits with different input shapes.")
        start = time.perf_counter()
        hsize = self.model_config.get_hidden_size()
        for num_reqs in self.num_reqs_paddings:
            dummy_hidden = create_torchax_tensor_with_partition_spec(
                torch.zeros((num_reqs, hsize),
                            dtype=self._hidden_states_dtype), self.mesh).jax()
            self.compute_logits_func(self.params_and_buffers_jax, dummy_hidden,
                                     None)
            logger.info("  -- num_seqs: %d", num_reqs)
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)

    def _precompile_sample_from_logits(self) -> None:
        logger.info(
            "Compiling sample_from_logits with different input shapes.")
        start = time.perf_counter()
        for num_reqs in self.num_reqs_paddings:
            dummy_logits = create_torchax_tensor_with_partition_spec(
                torch.zeros((num_reqs, self.vocab_size),
                            dtype=self._hidden_states_dtype), self.mesh,
                ()).jax()
            self.sample_from_logits(dummy_logits)
            logger.info("  -- num_seqs: %d", num_reqs)
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)

    def capture_model(self) -> None:
        """
        Precompile all the subgraphs with possible input shapes.
        """
        self._precompile_backbone()
        self._precompile_select_hidden_states()
        self._precompile_compute_logits()
        self._precompile_sample_from_logits()

    def profile_run(
        self,
        num_tokens: int,
    ) -> None:
        # Bind empty KV cache tensors to the model.
        kv_caches: dict[str, torch.Tensor] = {}
        kv_cache_spec = self.get_kv_cache_spec()
        for layer_name, layer_spec in kv_cache_spec.items():
            if isinstance(layer_spec, AttentionSpec):
                dtype = layer_spec.dtype
                # This is a workaround for torchax to create tensor on TPU
                # instead of CPU. Within torchax.enable_globally(), it will
                # create on CPU if we set device to be 'jax'.
                # Also torch.tensor(..., device='jax') will create tensor on CPU
                # instaled of TPU, need to call .to('jax')
                with torchax.default_env():
                    # Use an empty tensor instead of `None`` to force Dynamo to pass
                    # it by reference, rather by specializing on the value ``None``.
                    tpu_kv_cache = torch.tensor([], dtype=dtype)
                    tpu_kv_cache = create_torchax_tensor_with_partition_spec(
                        tpu_kv_cache, self.mesh, ())
                kv_caches[layer_name] = tpu_kv_cache
            else:
                raise NotImplementedError(
                    f"Unsupported KV cache spec '{type(layer_spec)}'")

        runner_kv_caches: list[torch.Tensor] = []
        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            runner_kv_caches)
        self.kv_caches_dict = kv_caches
        self.kv_caches_dict = pytree.tree_map_only(torch.Tensor,
                                                   lambda x: x.jax(),
                                                   self.kv_caches_dict)

        # Profile with multimodal encoder & encoder cache.
        # TODO: handle encoder-decoder models once we support them.
        # Trigger compilation for general shape.
        self._dummy_run(num_tokens)

        gc.collect()

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        torchax.enable_globally()
        if len(kv_cache_config.kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")

        if kv_cache_config.kv_cache_groups[
                0].kv_cache_spec.block_size != self.block_size:
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=self.max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=[
                    kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
                ],
            )
        # Verify dtype compatibility between block_table_cpu and input_batch
        assert self.block_table_cpu.dtype == self.input_batch.block_table[
            0].get_cpu_tensor().dtype

        kv_cache_sizes = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            assert len(kv_cache_tensor.shared_by) == 1, (
                "KV cache tensor shared by multiple layers is not supported in "
                "TPU.")
            kv_cache_sizes[kv_cache_tensor.shared_by[0]] = kv_cache_tensor.size

        kv_caches: dict[str, torch.Tensor] = {}
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                tensor_size = kv_cache_sizes[layer_name]
                assert tensor_size % kv_cache_spec.page_size_bytes == 0
                num_blocks = tensor_size // kv_cache_spec.page_size_bytes  # noqa
                if isinstance(kv_cache_spec, AttentionSpec):
                    if self.use_spmd:
                        num_kv_heads = kv_cache_spec.num_kv_heads
                        assert self.original_parallel_config is not None
                        tp_size = \
                            self.original_parallel_config.tensor_parallel_size
                        # TODO: Handle kv cache duplication under SPMD mode.
                        assert num_kv_heads % tp_size == 0, (
                            f"num_kv_heads {num_kv_heads} must be divisible by "
                            f"tp_size {tp_size} under SPMD mode")
                    kv_cache_shape = PallasAttentionBackend.get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype

                    tpu_kv_cache = torch.zeros(kv_cache_shape, dtype=dtype)

                    partition_spec = ()
                    if self.use_spmd:
                        partition_spec = (None, None, 'x', None)
                        # Use torchax tensor to support SPMD sharding.
                    tpu_kv_cache = create_torchax_kv_cache(
                        kv_cache_shape, dtype, self.mesh, partition_spec)
                    kv_caches[layer_name] = tpu_kv_cache.jax()
                else:
                    raise NotImplementedError

        # Setup `kv_cache_config` and `kv_caches` for models
        # with cross-layer KV sharing
        if self.shared_kv_cache_layers:
            initialize_kv_cache_for_kv_sharing(
                self.shared_kv_cache_layers,
                kv_cache_config.kv_cache_groups,
                kv_caches,
            )

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches,
        )
        self.kv_caches_dict = kv_caches
        torchax.disable_globally()

    def reset_dynamo_cache(self):
        compiled_model = self.model.model
        if isinstance(compiled_model, TorchCompileWrapperWithCustomDispatcher):
            logger.info("Clear dynamo cache and cached dynamo bytecode.")
            torch._dynamo.eval_frame.remove_from_cache(
                compiled_model.original_code_object)
            compiled_model.compiled_codes.clear()

    @staticmethod
    @jax.jit
    def select_hidden_states(hidden_states, indices_do_sample):
        return hidden_states[indices_do_sample]

    def compute_logits(self,
                       sample_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(sample_hidden_states, None)

    @staticmethod
    @jax.jit
    def sample_from_logits(logits):
        return jnp.argmax(logits, axis=-1, keepdims=True)


def _get_padded_num_kv_cache_update_slices(num_tokens: int, max_num_reqs: int,
                                           page_size: int) -> int:
    """Calculates the padded number of KV cache update slices to avoid
    recompilation."""
    padded_num_slices = 2 * max_num_reqs + num_tokens // page_size
    padded_num_slices = min(padded_num_slices, num_tokens)
    return padded_num_slices
