# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING, List

import jax
import jax.numpy as jnp
import numpy as np
import vllm.envs as envs
from jax.sharding import NamedSharding, PartitionSpec
from torchax.ops.mappings import t2j_dtype
from vllm.config import get_layers_from_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.mla import MLAAttention
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec, MambaSpec,
                                        MLAAttentionSpec, SlidingWindowSpec)

from tpu_inference import utils
from tpu_inference import utils as common_utils
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.runner import utils as runner_utils
from tpu_inference.runner.input_batch import CachedRequestState, InputBatch
from tpu_inference.runner.kv_cache import (KVCacheMetadata, create_kv_caches,
                                           get_attention_page_size_bytes)

if TYPE_CHECKING:
    from vllm.v1.request import Request

    from tpu_inference.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


class KVCacheManager:

    def __init__(self, runner: "TPUModelRunner"):
        self.runner = runner
        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}
        self.use_mla = self.runner.model_config.use_mla

    def _create_attention_spec(
            self,
            block_size: int,
            num_kv_heads: int,
            head_size: int,
            sliding_window: bool | None = None) -> KVCacheSpec:
        if self.use_mla:
            page_size_bytes = get_attention_page_size_bytes(
                self.runner.mesh, block_size, num_kv_heads, head_size,
                self.runner.kv_cache_dtype, True)
            return MLAAttentionSpec(block_size=block_size,
                                    num_kv_heads=1,
                                    head_size=head_size,
                                    dtype=self.runner.kv_cache_dtype,
                                    cache_dtype_str=self.runner.vllm_config.
                                    cache_config.cache_dtype,
                                    page_size_padded=int(page_size_bytes))
        else:
            page_size_bytes = get_attention_page_size_bytes(
                self.runner.mesh, block_size, num_kv_heads, head_size,
                self.runner.kv_cache_dtype, False)
            if sliding_window is not None:
                return SlidingWindowSpec(block_size=block_size,
                                         num_kv_heads=num_kv_heads,
                                         head_size=head_size,
                                         dtype=self.runner.kv_cache_dtype,
                                         sliding_window=sliding_window,
                                         page_size_padded=int(page_size_bytes))
            else:
                return FullAttentionSpec(block_size=block_size,
                                         num_kv_heads=num_kv_heads,
                                         head_size=head_size,
                                         dtype=self.runner.kv_cache_dtype,
                                         page_size_padded=int(page_size_bytes))

    def get_kv_cache_spec(self):
        # TODO(xiang): this hack tricks engine core to init successfully
        block_size = self.runner.cache_config.block_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}

        tp_axis_name = ShardingAxisName.ATTN_HEAD
        model_cnt = common_utils.get_mesh_shape_product(
            self.runner.mesh, tp_axis_name)
        # If use pure jax (MODEL_IMPL_TYPE=flax_nnx), we don't register
        # attention into compilation config.
        # Use FullAttentionSpec for each layer
        # TODO(pooyam): Is it possible to merge the logic for vllm and non-vllm models?
        model_config = self.runner.model_config
        if self.use_mla:
            # Individually pad the RopE and latents
            qk_rope_head_dim = getattr(model_config.hf_text_config,
                                       "qk_rope_head_dim", 0)
            padded_kv_lora_rank = common_utils.align_to(
                model_config.hf_text_config.kv_lora_rank, 128)
            padded_qk_rope_head_dim = common_utils.align_to(
                qk_rope_head_dim, 128)
            mla_head_size = padded_kv_lora_rank + padded_qk_rope_head_dim

        if len(self.runner.vllm_config.compilation_config.
               static_forward_context) == 0:
            parallel_config = self.runner.parallel_config
            text_config = getattr(model_config, "hf_text_config",
                                  getattr(model_config, "hf_config", None))
            base_num_kv_heads = model_config.get_total_num_kv_heads()
            base_head_size = model_config.get_head_size()

            for i in range(model_config.get_num_layers(parallel_config)):
                if self.use_mla:
                    kv_cache_spec[f"layer.{i}"] = self._create_attention_spec(
                        block_size, 1, mla_head_size)
                else:
                    # TODO(kwang3939): unify the hybrid kv cache of jax path and tochax path.
                    layer_type = "full_attention"
                    if hasattr(text_config, "layer_types") and i < len(
                            text_config.layer_types):
                        layer_type = text_config.layer_types[i]

                    is_sliding = layer_type == "sliding_attention"
                    if not is_sliding:
                        num_kv_heads = getattr(text_config,
                                               "num_global_key_value_heads",
                                               base_num_kv_heads)
                        head_size = getattr(text_config, "global_head_dim",
                                            base_head_size)
                    else:
                        num_kv_heads = getattr(text_config,
                                               "num_key_value_heads",
                                               base_num_kv_heads)
                        head_size = getattr(text_config, "head_dim",
                                            base_head_size)
                    # Pad num_kv_heads to multiple of TP size.
                    num_kv_heads = common_utils.get_padded_num_heads(
                        num_kv_heads, model_cnt)
                    head_size = common_utils.get_padded_head_dim(head_size)
                    # TODO(kwang3939): Re-enable sliding_window once mixed dims with sliding_window is supported.
                    sliding_window = None
                    kv_cache_spec[f"layer.{i}"] = self._create_attention_spec(
                        block_size,
                        num_kv_heads,
                        head_size,
                        sliding_window=sliding_window)

            if self.runner.speculative_config and self.runner.speculative_config.method == "eagle3":
                draft_model_config = self.runner.speculative_config.draft_model_config
                hf_config = draft_model_config.hf_config
                num_kv_heads = common_utils.get_padded_num_heads(
                    hf_config.num_key_value_heads, model_cnt)
                head_size = common_utils.get_padded_head_dim(
                    hf_config.hidden_size // hf_config.num_attention_heads)
                # Eagle3 has only 1 layer
                for i in range(1):
                    if self.use_mla:
                        kv_cache_spec[
                            f"draft_layer.{i}"] = self._create_attention_spec(
                                block_size, 1, mla_head_size)
                    else:
                        kv_cache_spec[
                            f"draft_layer.{i}"] = self._create_attention_spec(
                                block_size, num_kv_heads, head_size)
        else:
            # Else propagate attention modules from compilation config.
            layers = {}
            from vllm.model_executor.layers.mamba.abstract import MambaBase
            attention_types = [Attention, MLAAttention, MambaBase]

            for attn_cls in attention_types:
                # Get the layers for the current class
                new_layers = get_layers_from_vllm_config(
                    self.runner.vllm_config, attn_cls)

                # Add them to the main dictionary (equivalent to your | operator)
                layers.update(new_layers)

            # TODO(yuyanpeng): enable sliding windows once mixed dims support
            # Currently, with sliding windows, there is
            # shared_kv_cache_layers among each group.
            # The shared kv_cache_layers do not support mixed dims for
            # TPU for now. If share kv_cache, the attention kernel would
            # throw exception for non-matched dimension between kv_cache
            # and actual dims. Disable sliding window for workaround.
            head_size_set = {
                common_utils.get_padded_head_dim(attn_module.head_size)
                for attn_module in layers.values()
                if not isinstance(attn_module, MambaBase)
            }
            disable_sliding_window = len(head_size_set) > 1

            logger.warning(f"Compilation num_layers = {len(layers)}")

            for layer_name, attn_module in layers.items():
                if isinstance(attn_module, MambaBase):
                    spec = attn_module.get_kv_cache_spec(
                        self.runner.vllm_config)
                    if spec is not None:
                        kv_cache_spec[layer_name] = spec
                    continue

                if disable_sliding_window:
                    attn_module.sliding_window = None

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
                    num_kv_heads = common_utils.get_padded_num_heads(
                        attn_module.num_kv_heads,
                        self.runner.mesh.shape["model"])
                    head_size = common_utils.get_padded_head_dim(
                        attn_module.head_size)

                    if attn_module.sliding_window is not None:
                        kv_cache_spec[
                            layer_name] = self._create_attention_spec(
                                block_size,
                                num_kv_heads,
                                head_size,
                                sliding_window=attn_module.sliding_window)
                    elif self.use_mla:
                        kv_cache_spec[
                            layer_name] = self._create_attention_spec(
                                block_size, 1, mla_head_size)
                    else:
                        kv_cache_spec[
                            layer_name] = self._create_attention_spec(
                                block_size, num_kv_heads, head_size)
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
            assert self.runner.vllm_config.offload_config.uva.cpu_offload_gb == 0, (
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
            self.runner.block_tables_cpu = [
                np.zeros((self.runner.max_num_reqs,
                          cdiv(self.runner.max_model_len, block_size)),
                         dtype=np.int32) for block_size in block_sizes
            ]

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        self.maybe_reinitialize_input_batch(kv_cache_config)

        # There will be no KV cache for pooling models.
        if not kv_cache_config.kv_cache_groups:
            return

        layer_name_to_spec = {}
        for group in kv_cache_config.kv_cache_groups:
            group_spec = group.kv_cache_spec
            if hasattr(group_spec, 'kv_cache_specs'):
                for layer_name in group.layer_names:
                    layer_name_to_spec[layer_name] = group_spec.kv_cache_specs[
                        layer_name]
            else:
                for layer_name in group.layer_names:
                    layer_name_to_spec[layer_name] = group.kv_cache_spec

        kv_caches = self.runner.kv_caches
        num_blocks_list = []
        # Mapping between KV cache type and the associated metadata, needed for logging
        # about KV cache
        metadata = {
            "mamba": KVCacheMetadata(),
            "regular_attn": KVCacheMetadata()
        }
        # If this is true, then we'll initialize a new KV cache for each layer in "shared_by"
        # instead of the default behavior of initializing a single KV cache for each of the
        # shared layers
        duplicate_shared_layers = False
        if not duplicate_shared_layers:
            for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
                if any(
                        isinstance(layer_name_to_spec[layer_name], MambaSpec)
                        for layer_name in kv_cache_tensor.shared_by):
                    # TODO (jacobplatin): we should not be replicating the kv cache for each layer and instead
                    # should follow the native GPU/Torch approach where every group of layers (shared_by)
                    # shares the same underlying raw tensor.
                    logger.warning_once(
                        "MambaSpec does not support shared layers for now, defaulting to single KV cache per layer..."
                    )
                    duplicate_shared_layers = True
                    # assert that each kv_cache_tensor in kv_cache_config.kv_cache_tensors has the same number of shared layers
                    # This is needed for models like Qwen3.5 where every 4 layers share the same KV cache (3 linear attn and 1 full attn)
                    num_shared_layers = len(
                        kv_cache_config.kv_cache_tensors[0].shared_by)
                    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
                        assert len(
                            kv_cache_tensor.shared_by
                        ) == num_shared_layers, f"Expected all kv_cache_tensors to have the same number of shared layers {num_shared_layers}, but found {len(kv_cache_tensor.shared_by)}"
                    break

        for i, kv_cache_tensor in enumerate(kv_cache_config.kv_cache_tensors):
            for j, layer_name in enumerate(kv_cache_tensor.shared_by):
                layer_spec = layer_name_to_spec[layer_name]

                page_size_bytes = layer_spec.page_size_bytes
                assert kv_cache_tensor.size % page_size_bytes == 0
                num_blocks = kv_cache_tensor.size // page_size_bytes
                dp_size = self.runner.vllm_config.sharding_config.total_dp_size
                # num_blocks must be a multiple of dp_size
                num_blocks = (num_blocks // dp_size) * dp_size

                if isinstance(layer_spec, MambaSpec):
                    mamba_states = []
                    for state_index, (shape, dtype) in enumerate(
                            zip(layer_spec.shapes, layer_spec.dtypes)):
                        jax_dtype = t2j_dtype(dtype)
                        cache_shape = (num_blocks, *shape)
                        if state_index == 0:
                            # conv_state: [num_blocks, conv_kernel_size, intermediate_size]
                            spec = PartitionSpec(None, None,
                                                 ShardingAxisName.ATTN_HEAD)
                        elif state_index == 1:
                            # ssm_state: [num_blocks, num_heads, head_dim, state_size]
                            spec = PartitionSpec(None,
                                                 ShardingAxisName.ATTN_HEAD,
                                                 None, None)
                        else:
                            spec = PartitionSpec(
                                None, *([None] * (len(cache_shape) - 1)))

                        sharding = NamedSharding(self.runner.mesh, spec)

                        # NOTE: conv state will always be BF16 and SSM state will always be FP32
                        # regardless of the `kv-cache-dtype` (as is in upstream vLLM)
                        def _allocate_mamba(c_shape=cache_shape,
                                            c_dtype=jax_dtype):
                            return jnp.empty(shape=c_shape, dtype=c_dtype)

                        mamba_allocate = jax.jit(_allocate_mamba,
                                                 out_shardings=sharding)
                        mamba_states.append(mamba_allocate())

                    metadata["mamba"].count += 1
                    if metadata["mamba"].shape is None:
                        # Mamba is a tuple of arrays, so we store a tuple of their metadata
                        metadata["mamba"].shape = tuple(s.shape
                                                        for s in mamba_states)
                        metadata["mamba"].dtype = tuple(s.dtype
                                                        for s in mamba_states)
                        metadata["mamba"].sharding = tuple(
                            s.sharding for s in mamba_states)

                    kv_caches.append(tuple(mamba_states))
                else:
                    # We should only init a new kv cache for the first layer in shared_by
                    # if duplicate_shared_layers is False.  Otherwise, if duplicate_shared_layers
                    # is True, we should init a new kv cache for each layer in shared_by
                    if j == 0 or duplicate_shared_layers:
                        # NOTE: we'll multiply the num_kv_heads by 2 in the function
                        if self.use_mla:
                            head_size = self.runner.model_config.hf_config.kv_lora_rank + \
                                self.runner.model_config.hf_config.qk_rope_head_dim
                        else:
                            head_size = layer_spec.head_size
                        kv_cache = create_kv_caches(
                            num_blocks=num_blocks,
                            block_size=layer_spec.block_size,
                            num_kv_heads=layer_spec.num_kv_heads,
                            head_size=head_size,
                            mesh=self.runner.mesh,
                            layer_names=[f'kv_cache_tensor.{i}'],
                            cache_dtype=t2j_dtype(layer_spec.dtype),
                            use_mla=self.use_mla,
                        )[0]
                        kv_caches.append(kv_cache)

                        # Update Regular Attention Metadata
                        metadata["regular_attn"].count += 1
                        if metadata["regular_attn"].shape is None:
                            metadata["regular_attn"].shape = kv_cache.shape
                            metadata["regular_attn"].dtype = kv_cache.dtype
                            metadata[
                                "regular_attn"].sharding = kv_cache.sharding
                # We should only add the blocks for the first layer in shared_by
                # if duplicate_shared_layers is False.  Otherwise, if duplicate_shared_layers
                # is True, we should add the blocks for each layer in shared_by.
                if j == 0 or duplicate_shared_layers:
                    num_blocks_list.append(num_blocks)
                layer_idx = (i * num_shared_layers
                             ) + j if duplicate_shared_layers else i
                self.runner.layer_name_to_kvcache_index[layer_name] = layer_idx
        if self.shared_kv_cache_layers:
            for layer_name, target_layer_name in self.shared_kv_cache_layers.items(
            ):
                self.runner.layer_name_to_kvcache_index[
                    layer_name] = self.runner.layer_name_to_kvcache_index[
                        target_layer_name]

        log_parts = [
            "Init kv-cache", f"num_total_layers={len(kv_caches)}",
            f"num_blocks={num_blocks_list}"
        ]

        if metadata["regular_attn"].count > 0:
            log_parts.append(
                f"regular_attn_layers={metadata['regular_attn'].count} | "
                f"regular_attn_shape=(num_blocks, {metadata['regular_attn'].shape[1:]}) | "
                f"regular_attn_sharding={metadata['regular_attn'].sharding} | "
                f"regular_attn_dtype={metadata['regular_attn'].dtype}")

        if metadata["mamba"].count > 0:
            log_parts.append(f"mamba_layers={metadata['mamba'].count} | "
                             f"mamba_shape={metadata['mamba'].shape} | "
                             f"mamba_sharding={metadata['mamba'].sharding} | "
                             f"mamba_dtype={metadata['mamba'].dtype}")

        log_parts.append(
            f"hbm={utils.hbm_usage_gb(self.runner.mesh.devices.flatten())}Gb")

        logger.info(" | ".join(log_parts))

    def delete_kv_cache(self) -> None:
        """Delete KV cache JAX arrays to free HBM.
        This explicitly deletes all KV cache JAX arrays, clearing the HBM
        they occupy.

        1. Avoid serving stale KV cache values from a previous model version
           (since prefix cache keys remain constant but values become invalid
           after weight updates).
        2. Free HBM to reduce memory fragmentation during the HBM-heavy
           resharding operation, allowing higher --gpu-memory-utilization
           settings.
        After calling this method, ``reinitialize_kv_cache`` must be called
        to reallocate the KV cache before the next inference step.
        """
        kv_caches = self.runner.kv_caches
        if not kv_caches:
            logger.info("delete_kv_cache: No KV cache to delete.")
            return

        num_layers = len(kv_caches)
        logger.info(
            f"Deleting kv-cache | "
            f"num_layers={num_layers} | "
            f"hbm_before="
            f"{utils.hbm_usage_gb(self.runner.mesh.devices.flatten())}Gb")

        # Explicitly delete each JAX array to release HBM.
        for kv_cache in kv_caches:
            kv_cache.delete()
        self.runner.kv_caches.clear()
        self.runner.layer_name_to_kvcache_index.clear()

        logger.info(
            f"KV cache delete complete | "
            f"hbm_after="
            f"{utils.hbm_usage_gb(self.runner.mesh.devices.flatten())}Gb")

    def reinitialize_kv_cache(self) -> None:
        """Reinitialize KV cache from the stored configuration.
        This reallocates fresh (empty) KV cache arrays using the
        ``KVCacheConfig`` that was saved during the initial
        ``initialize_kv_cache`` call.  It is intended to be called after
        ``delete_kv_cache`` (and typically after a weight-sync / resharding
        step) so that inference can resume with a clean cache.
        Raises:
            RuntimeError: If ``initialize_kv_cache`` was never called (i.e.
                there is no stored ``kv_cache_config``).
        """
        kv_cache_config = getattr(self.runner, 'kv_cache_config', None)
        if kv_cache_config is None:
            raise RuntimeError(
                "Cannot reinitialize KV cache: no kv_cache_config found. "
                "initialize_kv_cache must be called first.")

        logger.info(
            f"Reinitializing kv-cache | "
            f"hbm_before="
            f"{utils.hbm_usage_gb(self.runner.mesh.devices.flatten())}Gb")

        self.initialize_kv_cache(kv_cache_config)

    @staticmethod
    @jax.jit
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
    @jax.jit(static_argnames=("len_block"))
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

    def _jitted_insert_kv_cache(
        block_size,
        kv_caches: List[jax.Array],
        kv_cache_slices: List[jax.Array],
        block_numbers: List[int],
    ) -> List[jax.Array]:
        """
        Iteratively call continuous KV cache insertion for each contiguous block sub-array.
        """
        if not block_numbers:
            return kv_caches

        start_idx = 0
        for i in range(1, len(block_numbers) + 1):
            if i == len(block_numbers
                        ) or block_numbers[i] != block_numbers[i - 1] + 1:
                start_block = block_numbers[start_idx]

                chunk_size = i - start_idx
                token_start = start_idx * block_size
                with runner_utils.LatencyTracker(
                        f"insert_continuous_kv_cache_from_slice {start_block} {chunk_size}"
                ):
                    kv_caches = KVCacheManager._jitted_insert_continuous_kv_cache_from_slice(
                        block_size,
                        chunk_size,
                        kv_caches,
                        kv_cache_slices,
                        token_start,
                        start_block,
                    )
                start_idx = i

        return kv_caches

    @staticmethod
    @jax.jit(
        static_argnames=("block_size", "chunk_size"),
        donate_argnames=("kv_caches", ),
    )
    def _jitted_insert_continuous_kv_cache_from_slice(
        block_size: int,
        chunk_size: int,
        kv_caches: List[jax.Array],
        kv_cache_slices: List[jax.Array],
        token_start: int,
        start_block: int,
    ) -> List[jax.Array]:
        """
        JIT-compiled function that dynamically slices the required tokens from KV cache
        slices and inserts them into continuous physical blocks.
        """

        def _update_layer(cache, slices):
            """The function to apply to each layer's cache and slices."""

            # Dynamically slice exactly chunk_size * block_size tokens starting at token_start
            extracted_slices = jax.lax.dynamic_slice_in_dim(slices,
                                                            token_start,
                                                            chunk_size *
                                                            block_size,
                                                            axis=0)
            reshaped_slices = extracted_slices.reshape(-1, block_size,
                                                       *slices.shape[1:])

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
            batched_kv_cache_per_layer = self._jitted_gather_continuous_kv_cache(
                self.runner.kv_caches, block_ids[0], len(block_ids))

        else:
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
        sharding = NamedSharding(
            self.runner.mesh, PartitionSpec(None, ShardingAxisName.ATTN_HEAD))
        if envs.VLLM_TPU_USING_PATHWAYS:
            from pathwaysutils.experimental import \
                reshard as experimental_reshard

            def get_sharding(x):
                return sharding

            sharding_spec_pytree = jax.tree.map(get_sharding, kv_cache_slices)
            transferred_kv_cache = experimental_reshard.reshard(
                kv_cache_slices,
                sharding_spec_pytree,
                donate=False,
            )
        else:
            transferred_kv_cache = jax.device_put(kv_cache_slices, sharding)

        jax.block_until_ready(transferred_kv_cache)
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
                self.runner.kv_caches = KVCacheManager._jitted_insert_continuous_kv_cache_from_slice(
                    self.runner.block_size,
                    len(block_numbers),
                    self.runner.kv_caches,
                    kv_cache_slices,
                    0,
                    start_block,
                )
                jax.block_until_ready(self.runner.kv_caches)
        else:
            with runner_utils.LatencyTracker(
                    f"JittedInsertKVCache-b{len(block_numbers)}"):
                logger.debug(
                    f"inserting to non continuous blocks {block_numbers}")
                self.runner.kv_caches = KVCacheManager._jitted_insert_kv_cache(
                    self.runner.block_size,
                    self.runner.kv_caches,
                    kv_cache_slices,
                    block_numbers,
                )
                jax.block_until_ready(self.runner.kv_caches)

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
            mm_features=getattr(request, "mm_features", []),
            pooling_params=getattr(request, "pooling_params", None),
            generator=None,
        )

        self.runner.requests[req_id] = req_state
        self.runner.input_batch.add_request(req_state)
