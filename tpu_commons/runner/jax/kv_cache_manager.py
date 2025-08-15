import functools
from typing import TYPE_CHECKING, List, Dict

import jax
import jax.numpy as jnp
import torch
from jax.sharding import NamedSharding, PartitionSpec
from vllm.attention.backends.abstract import AttentionType
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec, SlidingWindowSpec)

from tpu_commons import utils as common_utils
from tpu_commons.logger import init_logger
from tpu_commons.runner import utils as runner_utils
from tpu_commons.runner.jax.input_batch_jax import (CachedRequestState,
                                                    InputBatch)

if TYPE_CHECKING:
    from vllm.v1.request import Request

    from tpu_commons.runner.jax.tpu_jax_runner import TPUModelRunner

logger = init_logger(__name__)


class KVCacheManager:

    def __init__(self, runner: "TPUModelRunner"):
        self.runner = runner
        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}

    def get_kv_cache_spec1(self):
        # TODO(xiang): this hack tricks engine core to init successfully
        block_size = self.runner.vllm_config.cache_config.block_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        model_config = self.runner.vllm_config.model_config
        parallel_config = self.runner.vllm_config.parallel_config

        # Pad num_kv_heads to multiple of TP size.
        num_kv_heads = common_utils.get_padded_num_heads(
            model_config.get_total_num_kv_heads(),
            self.runner.mesh.shape["model"])

        # Pad head_dim to multiple of 128.
        head_size = model_config.get_head_size()
        head_size = common_utils.get_padded_head_dim(head_size)

        for i in range(model_config.get_num_layers(parallel_config)):
            kv_cache_spec[f"layers.{i}"] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=torch.bfloat16,
                use_mla=False,
            )

        return kv_cache_spec
    
    def get_kv_cache_spec(self):
        # TODO(xiang): this hack tricks engine core to init successfully
        block_size = self.runner.cache_config.block_size
        use_mla = self.runner.model_config.use_mla
        kv_cache_spec: dict[str, KVCacheSpec] = {}

        # If use pure jax (MODEL_IMPL_TYPE=flax_nnx), we don't register
        # attention into compilation config.
        # Use FullAttentionSpec for each layer
        if len(self.runner.vllm_config.compilation_config.static_forward_context
               ) == 0:
            model_config = self.runner.model_config
            parallel_config = self.runner.parallel_config
            # Pad num_kv_heads to multiple of TP size.
            num_kv_heads = common_utils.get_padded_num_heads(
                model_config.get_total_num_kv_heads(),
                self.mesh.shape["model"])
            for i in range(model_config.get_num_layers(parallel_config)):
                kv_cache_spec[f"layer.{i}"] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=num_kv_heads,
                    head_size=model_config.get_head_size(),
                    dtype=torch.bfloat16,
                    use_mla=use_mla,
                )
        else:
            # Else propagate attention info from compilation config.
            from tpu_commons.models.vllm.sharding import AttentionInfo
            for layer_name, attention_info in self.runner.vllm_config.compilation_config.static_forward_context.items(
            ):
                if not isinstance(attention_info, AttentionInfo):
                    continue
                if (kv_tgt_layer := attention_info.kv_sharing_target_layer_name
                    ) is not None:
                    # The layer doesn't need its own KV cache and will use that of
                    # the target layer. We skip creating a KVCacheSpec for it, so
                    # that KV cache management logic will act as this layer does
                    # not exist, and doesn't allocate KV cache for the layer. This
                    # enables the memory saving of cross-layer kv sharing, allowing
                    # a given amount of memory to accommodate longer context lengths
                    # or enable more requests to be processed simultaneously.
                    self.runner.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                    continue
                if attention_info.attn_type == AttentionType.DECODER:
                    if attention_info.sliding_window is not None:
                        kv_cache_spec[layer_name] = SlidingWindowSpec(
                            block_size=block_size,
                            num_kv_heads=attention_info.num_kv_heads,
                            head_size=attention_info.head_size,
                            dtype=torch.bfloat16,
                            sliding_window=attention_info.sliding_window,
                            use_mla=use_mla)
                    else:
                        kv_cache_spec[layer_name] = FullAttentionSpec(
                            block_size=block_size,
                            num_kv_heads=attention_info.num_kv_heads,
                            head_size=attention_info.head_size,
                            dtype=torch.bfloat16,
                            use_mla=use_mla,
                        )
                elif attention_info.attn_type in (AttentionType.ENCODER,
                                                  AttentionType.ENCODER_ONLY):
                    # encoder-only attention does not need KV cache.
                    continue
                elif attention_info.attn_type == AttentionType.ENCODER_DECODER:
                    raise NotImplementedError
                else:
                    raise ValueError(
                        f"Unknown attention type: {attention_info.attn_type}")
        return kv_cache_spec
    
    def maybe_reinitialize_input_batch(self,
                                       kv_cache_config: KVCacheConfig) -> None:
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
        ]
        if block_sizes != [self.runner.cache_config.block_size]:
            assert self.cache_config.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details.")
            self.runner.input_batch = InputBatch(
                max_num_reqs=self.runner.max_num_reqs,
                max_model_len=self.runner.max_model_len,
                max_num_batched_tokens=self.runner.max_num_tokens,
                pin_memory=False,
                vocab_size=self.runner.model_config.get_vocab_size(),
                block_sizes=block_sizes,
            )

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        self.runner.kv_caches: List[jax.Array] = []
        self.maybe_reinitialize_input_batch(kv_cache_config)

        # kv_cache_groups = kv_cache_config.kv_cache_groups
        # if len(kv_cache_groups) > 1:
        #     raise NotImplementedError(
        #         "Hybrid models with more than one KV cache type are not "
        #         "supported yet.")

        # kv_cache_spec = kv_cache_groups[0].kv_cache_spec
        # layer_names = kv_cache_groups[0].layer_names

        # # NOTE: we'll multiply the num_kv_heads by 2 in the function
        # self.runner.kv_caches = runner_utils.create_kv_caches(
        #     num_blocks=kv_cache_config.num_blocks,
        #     block_size=kv_cache_spec.block_size,
        #     num_kv_heads=kv_cache_spec.num_kv_heads,
        #     head_size=kv_cache_spec.head_size,
        #     mesh=self.runner.mesh,
        #     layer_names=layer_names,
        #     devices=self.runner.devices,
        # )

        # if has_kv_transfer_group():
        #     get_kv_transfer_group().register_runner(self.runner)

        # uniform page size.
        representative_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        page_size_bytes = representative_spec.page_size_bytes
        self.runner.layer_name_to_kvcache_index: Dict[str, int] = {}
        # kv_caches_list: List[jax.Array] = []
        for i, kv_cache_tensor in enumerate(kv_cache_config.kv_cache_tensors):
            assert kv_cache_tensor.size % page_size_bytes == 0
            kv_cache = runner_utils.create_kv_caches(
                num_blocks=kv_cache_tensor.size // page_size_bytes,
                block_size=representative_spec.block_size,
                num_kv_heads=representative_spec.num_kv_heads,
                head_size=representative_spec.head_size,
                mesh=self.mesh,
                layer_names=['kv_cache_tensor.{i}'],
                devices=self.devices)[0]
            self.runner.kv_caches.append(kv_cache)
            for layer_name in kv_cache_tensor.shared_by:
                self.runner.layer_name_to_kvcache_index[layer_name] = i
        
        if self.shared_kv_cache_layers:
            for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
                self.runner.layer_name_to_kvcache_index[layer_name] = self.runner.layer_name_to_kvcache_index[target_layer_name]

        if has_kv_transfer_group():
            get_kv_transfer_group().register_runner(self.runner)

    @staticmethod
    @functools.partial(jax.jit)
    def _jitted_gather_kv_cache(
            kv_caches: List[jax.Array],
            block_ids: jax.Array) -> List[jax.Array]:
        """
        JIT-compiled function to gather KV cache slices for all layers at once.
        This uses jax.tree.map to apply the operation across all layers.
        """
        # Define the function to apply to each layer's cache.
        # The 'block_ids' array is captured from the outer scope.
        gather_and_reshape = lambda layer_kv_cache: (
            layer_kv_cache.at[block_ids]
            .get()
            .reshape(-1, *layer_kv_cache.shape[2:])
        )
        return jax.tree.map(gather_and_reshape, kv_caches)

    @staticmethod
    @functools.partial(
        jax.jit,
        static_argnames=("block_size"),
        donate_argnames=("kv_caches"),
    )
    def _jitted_insert_kv_cache(
        block_size,
        kv_caches: List[jax.Array],
        kv_cache_slices: List[jax.Array],
        block_numbers: jax.Array,
    ) -> List[jax.Array]:
        """
        JIT-compiled function to insert KV cache slices into the physical
        cache for all layers at once. This fuses the pad, reshape, and scatter
        operations into a single efficient kernel.
        """
        new_kv_caches = []
        # Assuming block numbers are non-negative and sorted.
        for i, layer_kv_cache_slices in enumerate(kv_cache_slices):
            padded_seq_len, packing_div, packing, head_dim = layer_kv_cache_slices.shape
            padding_config = ((0, block_numbers.shape[0] * block_size -
                               padded_seq_len), (0, 0), (0, 0), (0, 0))
            layer_kv_cache_slices = jnp.pad(layer_kv_cache_slices,
                                            pad_width=padding_config)
            layer_kv_cache_slices = layer_kv_cache_slices.reshape(
                -1, block_size, packing_div, packing, head_dim)
            updated_cache = kv_caches[i].at[block_numbers].set(
                layer_kv_cache_slices)
            new_kv_caches.append(updated_cache)
        return new_kv_caches

    def get_kv_cache_for_block_ids(
        self,
        block_ids: List[int],
    ) -> List[jax.Array]:
        """
        Extracts the KV cache slices for a given list of block IDs.
        This assumes all provided blocks are full.

        Args:
            block_ids: A list of block IDs to extract KV cache for.

        Returns:
            A list of JAX arrays, with each array representing the KV cache
            slices for a layer, concatenated for all blocks.
        """
        block_ids = jnp.array(block_ids)
        with runner_utils.LatencyTracker("BatchedGatherKVSlices-for-blocks"):
            batched_kv_cache_per_layer = self._jitted_gather_kv_cache(
                self.runner.kv_caches, block_ids)

        return batched_kv_cache_per_layer

    def transfer_kv_cache(self,
                          kv_cache_slices: List[jax.Array]) -> List[jax.Array]:
        """
        Transfers KV cache slices to the runner's mesh.

        This is used when a KV cache generated on one runner (e.g., a prefill
        runner) needs to be used on another runner (e.g., a decode runner)
        with a different device mesh. The transfer is asynchronous.

        Args:
            kv_cache_slices: A list of JAX arrays, where each array contains
                the KV cache slices for a specific layer. The shape of each
                slice is expected to be (num_tokens, num_kv_heads * 2, head_size).

        Returns:
            A new list of JAX arrays representing the KV cache slices, sharded
            across the runner's device mesh.
        """
        # The KV cache slices have a shape of (num_tokens, num_kv_heads * 2, head_size).
        # We shard along the num_kv_heads dimension (axis=1), which corresponds
        # to the "model" axis of the mesh for tensor parallelism.
        sharding = NamedSharding(self.runner.mesh, PartitionSpec())
        transferred_kv_cache = jax.device_put(kv_cache_slices, sharding)
        for cache in transferred_kv_cache:
            cache.block_until_ready()
        return transferred_kv_cache

    def insert_request_with_kv_cache(
        self,
        request: "Request",
        kv_cache_slices: List[jax.Array],
        block_ids: List[List[int]],
    ):
        """
        Inserts a request and its KV cache into the runner. This is used to
        transfer a request from a prefill runner to a decode runner.

        The provided KV cache slices are copied into the physical blocks
        allocated for the request. The runner's internal state is then updated
        to include the request.

        Args:
            request: The vLLM request object, containing the state after prefill.
            kv_cache_slices: The KV cache for the request, already transferred
                to this runner's mesh. This is a list of JAX arrays, one per layer.
            block_ids: The physical block numbers allocated for this request on
                this runner. This is a list of lists, for each KV cache group.
        """
        # Assume one KV cache group for now, which is consistent with current setup.
        if len(block_ids) > 1:
            raise NotImplementedError(
                "Inserting KV cache for models with multiple KV cache groups "
                "is not supported yet.")
        block_numbers = block_ids[0]
        num_blocks = len(block_numbers)

        # Pad the number of blocks to static bucket sizes to avoid recompilation.
        block_buckets = [
            1, 2, 4, 8, 16, 32, 64, self.runner.max_num_blocks_per_req
        ]
        import bisect
        bucket_index = bisect.bisect_left(block_buckets, num_blocks)
        padded_num_blocks = block_buckets[bucket_index]
        padding_size = padded_num_blocks - num_blocks

        # Pad target_block_numbers. Pad with the max_num_blocks + 1 to avoid writing
        # to unintended blocks. JAX would ignore as it's beyond the number of blocks
        # in the cache.
        max_num_blocks = self.runner.vllm_config.cache_config.num_gpu_blocks
        assert max_num_blocks is not None and max_num_blocks != 0
        block_numbers.extend([0] * padding_size)

        padded_block_numbers = jnp.array(block_numbers, dtype=jnp.int32)

        # Call the JIT-compiled function to perform the scatter operation
        # for all layers in a single, fused kernel.
        with runner_utils.LatencyTracker(
                f"JittedInsertKVCache-b{padded_num_blocks}"):
            self.runner.kv_caches = KVCacheManager._jitted_insert_kv_cache(
                self.runner.block_size,
                self.runner.kv_caches,
                kv_cache_slices,
                padded_block_numbers,
            )

        logger.debug(
            f"Updated kv cache entries cnt={len(self.runner.kv_caches)}")

        # Update runner's internal state to track the new request.
        req_id = request.request_id
        if req_id in self.runner.requests:
            logger.warning(
                f"Request {req_id} already exists in the runner. Overwriting.")

        # Create a CachedRequestState object to add to the input batch.
        req_state = CachedRequestState(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            output_token_ids=[request.all_token_ids[-1]],
            sampling_params=request.sampling_params,
            block_ids=tuple(block_ids),
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            mm_kwargs=getattr(request, "mm_kwargs", []),
            mm_positions=getattr(request, "mm_positions", []),
            mm_hashes=getattr(request, "mm_hashes", []),
            pooling_params=getattr(request, "pooling_params", None),
            generator=None,
        )

        self.runner.requests[req_id] = req_state
        self.runner.input_batch.add_request(req_state)
