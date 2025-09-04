import functools
from typing import TYPE_CHECKING, Dict, List

import jax
import jax.numpy as jnp
import torch
from jax.sharding import NamedSharding, PartitionSpec
from vllm.attention import Attention
from vllm.attention.backends.abstract import AttentionType
from vllm.config import get_layers_from_vllm_config
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec, SlidingWindowSpec)

from tpu_commons import utils
from tpu_commons import utils as common_utils
from tpu_commons.logger import init_logger
from tpu_commons.runner import utils as runner_utils
from tpu_commons.runner.input_batch_jax import CachedRequestState, InputBatch
from tpu_commons.runner.kv_cache import create_kv_caches

if TYPE_CHECKING:
    from vllm.v1.request import Request

    from tpu_commons.runner.tpu_jax_runner import TPUModelRunner

logger = init_logger(__name__)


class KVCacheManager:

    def __init__(self, runner: "TPUModelRunner"):
        self.runner = runner
        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}

    def get_kv_cache_spec(self):
        # TODO(xiang): this hack tricks engine core to init successfully
        block_size = self.runner.cache_config.block_size
        use_mla = self.runner.model_config.use_mla
        kv_cache_spec: dict[str, KVCacheSpec] = {}

        # If use pure jax (MODEL_IMPL_TYPE=flax_nnx), we don't register
        # attention into compilation config.
        # Use FullAttentionSpec for each layer
        if len(self.runner.vllm_config.compilation_config.
               static_forward_context) == 0:
            model_config = self.runner.model_config
            parallel_config = self.runner.parallel_config
            # Pad num_kv_heads to multiple of TP size.
            num_kv_heads = common_utils.get_padded_num_heads(
                model_config.get_total_num_kv_heads(),
                self.runner.mesh.shape["model"])
            head_size = common_utils.get_padded_head_dim(
                model_config.get_head_size())
            for i in range(model_config.get_num_layers(parallel_config)):
                kv_cache_spec[f"layer.{i}"] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    dtype=torch.bfloat16,
                    use_mla=use_mla,
                )
        else:
            # Else propagate attention modules from compilation config.
            layers = get_layers_from_vllm_config(self.runner.vllm_config,
                                                 Attention)
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
                            num_kv_heads=common_utils.get_padded_num_heads(
                                attn_module.num_kv_heads,
                                self.runner.mesh.shape["model"]),
                            head_size=common_utils.get_padded_head_dim(
                                attn_module.head_size),
                            dtype=torch.bfloat16,
                            sliding_window=attn_module.sliding_window,
                            use_mla=use_mla)
                    else:
                        kv_cache_spec[layer_name] = FullAttentionSpec(
                            block_size=block_size,
                            num_kv_heads=common_utils.get_padded_num_heads(
                                attn_module.num_kv_heads,
                                self.runner.mesh.shape["model"]),
                            head_size=common_utils.get_padded_head_dim(
                                attn_module.head_size),
                            dtype=torch.bfloat16,
                            use_mla=use_mla,
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

    def maybe_reinitialize_input_batch(self,
                                       kv_cache_config: KVCacheConfig) -> None:
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
        ]
        if block_sizes != [self.runner.cache_config.block_size]:
            assert self.runner.cache_config.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details.")
            new_input_batch = InputBatch(
                max_num_reqs=self.runner.max_num_reqs,
                max_model_len=self.runner.max_model_len,
                max_num_batched_tokens=self.runner.max_num_tokens,
                pin_memory=False,
                vocab_size=self.runner.model_config.get_vocab_size(),
                block_sizes=block_sizes,
            )
            self.runner.input_batch = new_input_batch
            self.runner.persistent_batch_manager.input_batch = new_input_batch

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        self.maybe_reinitialize_input_batch(kv_cache_config)

        # uniform page size.
        representative_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        page_size_bytes = representative_spec.page_size_bytes
        self.runner.layer_name_to_kvcache_index: Dict[str, int] = {}
        kv_caches = self.runner.kv_caches
        num_blocks_list = []
        for i, kv_cache_tensor in enumerate(kv_cache_config.kv_cache_tensors):
            assert kv_cache_tensor.size % page_size_bytes == 0
            num_blocks = kv_cache_tensor.size // page_size_bytes
            # NOTE: we'll multiply the num_kv_heads by 2 in the function
            kv_cache = create_kv_caches(
                num_blocks=num_blocks,
                block_size=representative_spec.block_size,
                num_kv_heads=representative_spec.num_kv_heads,
                head_size=representative_spec.head_size,
                mesh=self.runner.mesh,
                layer_names=[f'kv_cache_tensor.{i}'])[0]
            kv_caches.append(kv_cache)
            num_blocks_list.append(num_blocks)
            for layer_name in kv_cache_tensor.shared_by:
                self.runner.layer_name_to_kvcache_index[layer_name] = i

        if self.shared_kv_cache_layers:
            for layer_name, target_layer_name in self.shared_kv_cache_layers.items(
            ):
                self.runner.layer_name_to_kvcache_index[
                    layer_name] = self.runner.layer_name_to_kvcache_index[
                        target_layer_name]

        logger.info(
            f"Init kv-cache | "
            f"num_layers={len(kv_caches)} | "
            f"shape=(num_blocks, {kv_caches[0].shape[1:]}) | "
            f"num_blocks={num_blocks_list} | "
            f"sharding={kv_caches[0].sharding} | "
            f"dtype={kv_caches[0].dtype} | "
            f"hbm={utils.hbm_usage_gb(self.runner.mesh.devices.flatten())}Gb")

    @staticmethod
    @functools.partial(jax.jit)
    def _jitted_gather_kv_cache(kv_caches: List[jax.Array],
                                block_ids: jax.Array) -> List[jax.Array]:
        """
        JIT-compiled function to gather KV cache slices for all layers at once.
        This uses jax.tree.map to apply the operation across all layers.
        """

        def gather_and_reshape(layer_kv_cache):
            return layer_kv_cache.at[block_ids].get().reshape(
                -1, *layer_kv_cache.shape[2:])

        return jax.tree.map(gather_and_reshape, kv_caches)

    @staticmethod
    @functools.partial(
        jax.jit,
        static_argnames=("len_block"),
    )
    def _jitted_gather_continuous_kv_cache(kv_caches: List[jax.Array],
                                           start_block,
                                           len_block) -> List[jax.Array]:
        """
        JIT-compiled function to gather KV cache slices for all layers at once.
        This uses jax.tree.map to apply the operation across all layers.
        """

        def gather_and_reshape(layer_kv_cache):
            shape = layer_kv_cache.shape
            return jax.lax.dynamic_slice_in_dim(layer_kv_cache,
                                                start_block,
                                                len_block,
                                                axis=0).reshape(
                                                    -1, *shape[2:])

        return jax.tree.map(gather_and_reshape, kv_caches)

    @staticmethod
    @functools.partial(
        jax.jit,
        static_argnames=("block_size"),
        donate_argnames=(
            "kv_caches",
            "kv_cache_slices",
        ),
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

        def _update_layer(cache, slices):
            """The function to apply to each layer's cache and slices."""
            reshaped_slices = slices.reshape(-1, 1, block_size,
                                             *slices.shape[1:])
            for (i, block_idx) in enumerate(block_numbers):
                cache = jax.lax.dynamic_update_slice_in_dim(cache,
                                                            reshaped_slices[i],
                                                            block_idx,
                                                            axis=0)
            return cache

        return jax.tree.map(_update_layer, kv_caches, kv_cache_slices)

    @staticmethod
    @functools.partial(
        jax.jit,
        static_argnames=("block_size"),
        donate_argnames=(
            "kv_caches",
            "kv_cache_slices",
        ),
    )
    def _jitted_insert_continuous_kv_cache(
        block_size,
        kv_caches: List[jax.Array],
        kv_cache_slices: List[jax.Array],
        start_block,
    ) -> List[jax.Array]:
        """
        JIT-compiled function to insert KV cache slices into continuous blocks.
        Makes use of dynamic_update_slice_in_dim.
        """

        def _update_layer(cache, slices):
            """The function to apply to each layer's cache and slices."""
            reshaped_slices = slices.reshape(-1, block_size, *slices.shape[1:])

            return jax.lax.dynamic_update_slice_in_dim(cache,
                                                       reshaped_slices,
                                                       start_block,
                                                       axis=0)

        return jax.tree.map(_update_layer, kv_caches, kv_cache_slices)

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
        if block_ids == list(range(block_ids[0],
                                   block_ids[0] + len(block_ids))):
            with runner_utils.LatencyTracker(
                    "BatchedGatherKVSlices-for-blocks"):
                batched_kv_cache_per_layer = self._jitted_gather_continuous_kv_cache(
                    self.runner.kv_caches, block_ids[0], len(block_ids))

        else:
            with runner_utils.LatencyTracker(
                    "BatchedGatherKVSlices-for-blocks"):
                batched_kv_cache_per_layer = self._jitted_gather_kv_cache(
                    self.runner.kv_caches, jnp.array(block_ids))
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
        logger.debug(
            f"Transferring kv cache shape {len(kv_cache_slices)} * {kv_cache_slices[0].shape} sharding {kv_cache_slices[0].sharding} size {kv_cache_slices[0].nbytes * len(kv_cache_slices)/1024/1024} Mbytes"
        )
        sharding = NamedSharding(self.runner.mesh,
                                 PartitionSpec(None, None, "model"))
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
        if block_numbers == list(
                range(block_numbers[0],
                      block_numbers[0] + len(block_numbers))):
            # For continuous blocks we use slice instead of scatter.
            start_block = block_numbers[0]
            with runner_utils.LatencyTracker(
                    f"JittedInsertContinuousKVCache-b{len(block_numbers)}"):
                logger.debug(f"inserting to continuous blocks {block_numbers}")
                self.runner.kv_caches = KVCacheManager._jitted_insert_continuous_kv_cache(
                    self.runner.block_size,
                    self.runner.kv_caches,
                    kv_cache_slices,
                    start_block,
                )
        else:
            with runner_utils.LatencyTracker(
                    f"JittedInsertKVCache-b{len(block_numbers)}"):
                logger.debug(
                    f"inserting to non continuous blocks {block_numbers}")
                self.runner.kv_caches = KVCacheManager._jitted_insert_kv_cache(
                    self.runner.block_size,
                    self.runner.kv_caches,
                    kv_cache_slices,
                    jnp.array(block_numbers),
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
