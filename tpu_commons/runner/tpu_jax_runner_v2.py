# Here we try to bring as much code as possible from Hex-LLM, instead of `tpu_torch_xla_runner.py` -> jax conversion.
import functools
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import bind_kv_cache

logger = init_logger(__name__)


class TPUModelRunner():

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: Any,
    ):
        self.vllm_config = vllm_config
        self.kv_caches = None
        self._init_mesh()

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
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        model_config = self.vllm_config.model_config
        parallel_config = self.vllm_config.parallel_config
        for i in range(model_config.get_num_layers(parallel_config)):
            kv_cache_spec[f"layers.{i}"] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=model_config.get_num_attention_heads(
                    parallel_config),
                head_size=model_config.get_head_size(),
                dtype=jnp.bfloat16,
                use_mla=False,
            )

        return kv_cache_spec

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        kv_caches: dict[str, Tuple[jax.Array, jax.Array]] = {}

        kv_cache_groups = kv_cache_config.kv_cache_groups
        assert len(
            kv_cache_groups) == 1, "Only full attention layer is supported now"

        kv_cache_spec = kv_cache_groups[0].kv_cache_spec
        layer_names = kv_cache_groups[0].layer_names
        cache_dtype = kv_cache_spec.dtype
        # TODO: vllm's pallas backend uses a different shape:
        # (num_blocks, block_size, num_kv_heads * 2, head_size)
        # need to figure out the performance diff.
        cache_shape = (
            kv_cache_spec.num_kv_heads,
            kv_cache_config.num_blocks,
            kv_cache_spec.block_size,
            kv_cache_spec.head_dim,
        )
        # Shard the num_kv_heads dim along the 'model' axis.
        sharding = NamedSharding(self.mesh, PartitionSpec("model"))

        def _allocate() -> Any:
            return jnp.empty(
                shape=cache_shape,
                dtype=cache_dtype,
            )

        sharded_allocate = jax.jit(_allocate, out_shardings=sharding)
        for layer_name in layer_names:
            k_cache = sharded_allocate()
            v_cache = sharded_allocate()
            kv_caches[layer_name] = (k_cache, v_cache)

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)

    def capture_model(self) -> None:
        pass

    def execute_model(
        self,
        scheduler_output: "VllmSchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> ModelRunnerOutput:
        req_id_to_index = {}
        req_ids = []
        prompt_logprobs_dict = {}

        # At first step of this implementation, let's do prefill and decode seperately for maximum code reuse.
        if len(scheduler_output.scheduled_new_reqs):
            _ = self._prepare_prefill(scheduler_output)

        all_reqs = scheduler_output.scheduled_new_reqs + scheduler_output.scheduled_cached_reqs

        for i, seq in enumerate(all_reqs):
            req_id_to_index[seq.req_id] = i
            req_ids.append(seq.req_id)
            prompt_logprobs_dict[seq.req_id] = None

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            prompt_logprobs_dict=prompt_logprobs_dict,
            logprobs=None,
            spec_token_ids=None,
            sampled_token_ids=[[0] for _ in range(len(req_ids))],
        )

    def _init_mesh(self) -> None:
        mesh_shape = [1, len(jax.devices())]
        # TODO: use global constants for mesh axis names.
        axis_names = ["data", "model"]
        self.mesh = Mesh(
            devices=mesh_utils.create_device_mesh(mesh_shape),
            axis_names=axis_names,
        )
        logger.info(f"Init mesh | mesh={self.mesh}")

    def _device_array(self, *args, sharding=None, **kwargs) -> jax.Array:
        if sharding is None:
            sharding = NamedSharding(self.mesh, PartitionSpec(None))
        return jax.device_put(*args, device=sharding, **kwargs)

    def _prepare_prefill(self, scheduler_output: VllmSchedulerOutput) -> Any:
        block_size = self.vllm_config.cache_config.block_size
        sliding_window = self.vllm_config.model_config.get_sliding_window()

        # TODO: pad batch_size
        batch_size = len(scheduler_output.scheduled_new_reqs)

        # Full prompt length.
        max_prompt_len = max([
            len(seq.prompt_token_ids)
            for seq in scheduler_output.scheduled_new_reqs
        ])

        # Unfilled prompt length.
        # TODO: Fix this for prefix caching.
        max_unfilled_prompt_len = max_prompt_len

        # Padded full prompt length.
        # TODO: Fix padding.
        padded_prompt_len = max_prompt_len

        # Padded unfilled prompt length.
        # TODO: Fix padding.
        padded_unfilled_prompt_len = max_unfilled_prompt_len

        images_flattened = []

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
            block_table = seq.block_ids
            assert len(block_table) <= padded_num_blocks
            block_indices[i][:len(block_table)] = block_table
            # Unfilled prompt associated block indices.
            num_unfilled_blocks = len(
                block_table) - num_effective_cached_blocks
            assert num_unfilled_blocks <= padded_num_unfilled_blocks
            kv_cache_write_indices[i][:num_unfilled_blocks] = block_table[
                num_effective_cached_blocks:]
            temperatures[i] = seq.sampling_params.temperature
            top_ps[i] = seq.sampling_params.top_p
            top_ks[i] = seq.sampling_params.top_k

            # TODO(pooyam): How to get this?
            running_indices[i] = 0  # seq.running_id

            # TODO(pooyam): double check this.
            output_token_indices[i] = seq.num_computed_tokens

            if seq.sampling_params.top_k != 1:
                do_sampling = True
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


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "input_positions",
        "seq_lens",
        "block_indices",
        "kv_cache_write_indices",
        "decode_lengths",
        "decode_page_indices",
        "num_decode_seqs",
        "prefill_lengths",
        "prefill_page_indices",
        "prefill_query_start_offsets",
        "num_prefill_seqs",
    ],
    meta_fields=["chunked_prefill_enabled"],
)
@dataclass
class AttentionMetadata(object):
    input_positions: jax.Array
    # If mix attention, this is a list of len 2
    seq_lens: Union[jax.Array, List[jax.Array]]
    # If mix attention, this is a list of len 2
    block_indices: Union[jax.Array, List[jax.Array]]
    # If mix attention, this is a list of len 2
    kv_cache_write_indices: Union[jax.Array, List[jax.Array]]
    # The following fields are set only when chunked prefill is enabled
    chunked_prefill_enabled: bool = False
    decode_lengths: jax.Array = None  # [max_num_decode_seqs]
    decode_page_indices: jax.Array = None  # [max_num_decode_seqs, pages_per_sequence]
    num_decode_seqs: jax.Array = None  # [1]
    prefill_lengths: jax.Array = None  # [max_num_prefill_seqs]
    prefill_page_indices: jax.Array = None  # [max_num_prefill_seqs, pages_per_sequence]
    prefill_query_start_offsets: jax.Array = None  # [max_num_prefill_seqs + 1]
    num_prefill_seqs: jax.Array = None  # [1]
