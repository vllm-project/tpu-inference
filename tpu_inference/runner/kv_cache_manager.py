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
import dataclasses
from typing import TYPE_CHECKING, List

import jax
import jax.numpy as jnp
import vllm.envs as envs
from jax.sharding import NamedSharding, PartitionSpec
from torchax.ops.mappings import t2j_dtype
from vllm.config import get_layers_from_vllm_config, set_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mla import MLAAttention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.utils import (get_kv_cache_layout,
                                              set_kv_cache_layout)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec, MambaSpec,
                                        MLAAttentionSpec, SlidingWindowSpec)

from tpu_inference import utils
from tpu_inference import utils as common_utils
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.offload.utils import get_kv_connector_cache_layout
from tpu_inference.runner import utils as runner_utils
from tpu_inference.runner.input_batch import CachedRequestState, InputBatch
from tpu_inference.runner.kv_cache import (KVCacheMetadata, create_kv_caches,
                                           get_attention_page_size_bytes)

if TYPE_CHECKING:
    from vllm.v1.request import Request

    from tpu_inference.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)

# default layout (order) used by kv cache manager
# N=num_blocks, H=num_heads and D=head_size
DEFAULT_KV_CACHE_LAYOUT = "NHD"


class KVCacheManager:

    def __init__(self, runner: "TPUModelRunner"):
        self.runner = runner
        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}
        self.use_mla = self.runner.model_config.use_mla
        # Set by `update_mamba_page_size_padded` for hybrid attention+mamba
        # models. When set, every attention layer spec reports this as its
        # `page_size_padded` so vLLM sees a uniform page size across groups
        # and computes `num_blocks` that matches what each layer actually gets
        # on the TPU side (where we duplicate the shared tensor per layer
        # because mamba and attention caches have different shapes).
        self._hybrid_uniform_page_size_bytes: int | None = None
        # Per-mamba-layer leading-dim cap for hybrid models. Mamba state is
        # recurrent (one slot per *active* request), so the `num_blocks`
        # slots vLLM hands out are mostly idle. Set by
        # `_maybe_set_compact_mamba_num_blocks_override`; when set,
        # `initialize_kv_cache` allocates this many slots per mamba layer
        # while attention layers keep the full `num_blocks` from the
        # (correspondingly enlarged) attention pool. The GDN op indexes
        # mamba state with `attn_metadata.mamba_state_indices` (in
        # `[0, max_num_reqs]`) so it stays in-bounds for the smaller arrays.
        self._mamba_num_blocks: int | None = None

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
            page_size_padded = (self._hybrid_uniform_page_size_bytes
                                if self._hybrid_uniform_page_size_bytes
                                is not None else int(page_size_bytes))
            return MLAAttentionSpec(block_size=block_size,
                                    num_kv_heads=1,
                                    head_size=head_size,
                                    dtype=self.runner.kv_cache_dtype,
                                    cache_dtype_str=self.runner.vllm_config.
                                    cache_config.cache_dtype,
                                    page_size_padded=page_size_padded)
        else:
            page_size_bytes = get_attention_page_size_bytes(
                self.runner.mesh, block_size, num_kv_heads, head_size,
                self.runner.kv_cache_dtype, False)
            page_size_padded = (self._hybrid_uniform_page_size_bytes
                                if self._hybrid_uniform_page_size_bytes
                                is not None else int(page_size_bytes))
            if sliding_window is not None:
                return SlidingWindowSpec(block_size=block_size,
                                         num_kv_heads=num_kv_heads,
                                         head_size=head_size,
                                         dtype=self.runner.kv_cache_dtype,
                                         sliding_window=sliding_window,
                                         page_size_padded=page_size_padded)
            else:
                return FullAttentionSpec(block_size=block_size,
                                         num_kv_heads=num_kv_heads,
                                         head_size=head_size,
                                         dtype=self.runner.kv_cache_dtype,
                                         page_size_padded=page_size_padded)

    def update_mamba_page_size_padded(
            self, layers: dict[str, AttentionLayerBase]) -> None:
        """Pad attention and mamba page sizes so vLLM's num_blocks matches
        what the TPU allocates per layer.

        For hybrid attention+mamba models, vLLM groups a tensor's memory so
        that one `KVCacheTensor` is `shared_by` one layer from each kv-cache
        group (e.g., Qwen3.5: 1 full-attn + 3 linear-attn per shared_by).
        vLLM's scheduler assumes these layers share a single physical
        tensor at the byte level — each layer's block_table indexes into
        disjoint slots of the same backing allocation, and device kernels
        reinterpret the bytes as attention KV or mamba state depending on
        which layer is accessing the slot.

        TPU `jax.Array`s are strongly typed, so we cannot overlay an
        attention tensor and a mamba tensor on the same bytes.
        `initialize_kv_cache` therefore allocates one physical array per
        layer in the `shared_by` group, carving the group's byte budget
        into separate per-layer tensors. Without the compensation done
        here, vLLM's block pool would hold `num_shared_layers`× more
        block IDs than each per-layer array has slots — the scheduler
        would hand out block IDs beyond a layer's leading dimension,
        JAX's indexed writes would silently clip them, and multiple
        requests' mamba recurrent states would collapse onto the same
        slot (corrupted state → gibberish generation).

        The fix: set every layer's reported `page_size_padded` equal to the
        full per-`shared_by` footprint — `num_attn_groups × attn_page +
        num_mamba_groups × mamba_unpadded`, where `attn_page` is the
        TPU-actual per-block bytes (from `get_attention_page_size_bytes`,
        which accounts for dtype packing like fp8) and `mamba_unpadded` is
        the natural `prod(shape) × dtype_size`. vLLM then computes a
        smaller `num_blocks` that exactly matches what we allocate per layer
        on the TPU side. HBM usage is unchanged; only the block-ID
        accounting lines up.

        Args:
            layers: A dictionary mapping layer names to their corresponding
                attention module instances (e.g., `MambaBase`, `Attention`).
        """
        attn_modules = [
            module for module in layers.values()
            if isinstance(module, Attention)
        ]
        if not attn_modules:
            return

        first_attn_module = attn_modules[0]
        for module in attn_modules:
            assert module.num_kv_heads == first_attn_module.num_kv_heads
            assert module.head_size == first_attn_module.head_size

        num_kv_heads = common_utils.get_padded_num_heads(
            first_attn_module.num_kv_heads,
            common_utils.get_mesh_shape_product(self.runner.mesh,
                                                ShardingAxisName.ATTN_HEAD))
        head_size = common_utils.get_padded_head_dim(
            first_attn_module.head_size)
        attn_page_size_bytes = get_attention_page_size_bytes(
            self.runner.mesh, self.runner.cache_config.block_size,
            num_kv_heads, head_size, self.runner.kv_cache_dtype, False)

        mamba_modules = [
            module for module in layers.values()
            if isinstance(module, MambaBase)
        ]
        if not mamba_modules:
            # Not hybrid; set `mamba_page_size_padded` to the attention
            # page size as a no-op default (vLLM's platform interface sets
            # this too when it detects hybrid). No layer duplication will
            # happen without mamba layers, so no block-ID mismatch to fix.
            logger.debug(
                "Setting padded mamba page size in cache config to %d",
                attn_page_size_bytes)
            self.runner.cache_config.mamba_page_size_padded = (
                attn_page_size_bytes)
            return

        # Compute the unpadded mamba page size from an actual mamba module's
        # spec (shapes × dtype-size), ignoring any existing padding.
        first_mamba_spec = mamba_modules[0].get_kv_cache_spec(
            self.runner.vllm_config)
        assert isinstance(first_mamba_spec, MambaSpec)
        unpadded_mamba_page_size = dataclasses.replace(
            first_mamba_spec, page_size_padded=None).page_size_bytes

        # Derive vLLM's kv-cache group layout. vLLM splits each type into
        # equal-sized groups of `group_size` layers, then allocates
        # `group_size` `KVCacheTensor`s, each `shared_by` one layer from
        # every group — so each tensor covers `num_attn_groups +
        # num_mamba_groups` layers.
        #
        # Choosing `group_size` trades off padding vs. number of groups:
        #   * group_size = max_count → fewer groups (often 1 per type),
        #     but the smaller side pads its group up to max_count layers
        #     (wastes space if max ≫ min).
        #   * group_size = min_count → no padding, but the larger side
        #     splits into `ceil(max/min)` groups.
        # vLLM's rule: pick max_count only when counts are close enough
        # that the padding is minor (max < 1.5 × min), else min_count.
        #   e.g. 12 sliding-window + 13 full-attn → max (1 group each)
        #   e.g. 10 full-attn      + 30 mamba     → min (1 attn + 3 mamba)
        #
        # This duplicates the heuristic from
        # `vllm/v1/core/kv_cache_utils.py::_get_kv_cache_groups_uniform_page_size`.
        # We can't call it directly because vLLM's grouping needs a fully
        # populated spec dict, while we need the group layout *before* we
        # can finish creating the specs (padding depends on grouping,
        # spec creation depends on padding). Keep in sync if that
        # heuristic ever changes — it has been stable since the hybrid
        # allocator landed.
        num_attn = len(attn_modules)
        num_mamba = len(mamba_modules)
        min_count = min(num_attn, num_mamba)
        max_count = max(num_attn, num_mamba)
        # Match vLLM exactly: float comparison, no int() truncation (matters
        # at e.g. min=3, max=4, where 4 < 4.5 but 4 < int(4.5)==4 differs).
        if max_count < min_count * 1.5:
            group_size = max_count
        else:
            group_size = min_count
        num_attn_groups = (num_attn + group_size - 1) // group_size
        num_mamba_groups = (num_mamba + group_size - 1) // group_size

        uniform_page_size_bytes = (num_attn_groups * attn_page_size_bytes +
                                   num_mamba_groups * unpadded_mamba_page_size)

        logger.info(
            "Hybrid KV cache: padding every layer spec to %d bytes "
            "(num_attn_groups=%d × attn_page=%d + "
            "num_mamba_groups=%d × mamba_unpadded=%d). This makes vLLM's "
            "num_blocks match per-layer TPU allocation when mamba layers "
            "cannot be truly shared.", uniform_page_size_bytes,
            num_attn_groups, attn_page_size_bytes, num_mamba_groups,
            unpadded_mamba_page_size)

        self._hybrid_uniform_page_size_bytes = int(uniform_page_size_bytes)
        self.runner.cache_config.mamba_page_size_padded = int(
            uniform_page_size_bytes)

        # Cap each mamba layer at `max_num_reqs+1` slots and grow the
        # attention pool with the freed HBM. See
        # `_maybe_set_compact_mamba_num_blocks_override`.
        self._maybe_set_compact_mamba_num_blocks_override(
            attn_page_size_bytes, int(unpadded_mamba_page_size),
            num_attn_groups, num_mamba_groups, num_attn, num_mamba, group_size)

    def _maybe_set_compact_mamba_num_blocks_override(
            self, attn_page_size_bytes: int,
            unpadded_mamba_page_size_bytes: int, num_attn_groups: int,
            num_mamba_groups: int, num_attn_layers: int, num_mamba_layers: int,
            group_size: int) -> None:
        """Cap mamba layers at `max_num_reqs+1` slots and pin
        `num_gpu_blocks_override` so the freed HBM grows the attention pool.

        Tradeoff vs. the uniform num_blocks layout
        ------------------------------------------
        Mamba state is recurrent: one slot per *active* request, regardless
        of context length. The uniform layout gives every layer the same
        `num_blocks`, leaving `num_blocks − max_num_reqs` mamba slots idle
        forever. The compact layout caps mamba at `max_num_reqs + 1` (the
        `+1` is vLLM's null block), which is strictly better for any model
        where `num_blocks > max_num_reqs` — i.e. all production hybrid
        configs we run.
        Cost: a small bookkeeping invariant in the GDN op (it must index
        mamba state by per-request slot id rather than by `block_tables`,
        since the mamba leading dim is now smaller than the attn pool).
        See `gdn_attention_op.gdn_attention_core_tpu`.
        Skipped if the user pinned `num_gpu_blocks_override` or
        `hbm_usage_bytes` cannot read HBM (e.g. CPU-only tests). In those
        cases vLLM keeps its uniform sizing; the page-size padding done in
        the caller still keeps the per-layer block-ID range correct.

        Sizing math
        -----------
        Each kv-cache tensor is shared across `num_attn_groups +
        num_mamba_groups` layers; there are `group_size` such tensors.
        Per-tensor budget `B = avail / group_size`. With
        `N_mamba = max_num_reqs + 1`,
            N_attn = floor((B − num_mamba_groups × N_mamba × mamba_unpadded)
                            / (num_attn_groups × attn_page))
        rounded down to the sharding divisor.

        Args:
            attn_page_size_bytes: TPU-actual bytes per block per attention
                layer (accounts for dtype packing like fp8).
            unpadded_mamba_page_size_bytes: bytes per slot per mamba layer
                (`prod(shape) × dtype_size`, no padding).
            num_attn_groups: # vLLM kv-cache groups holding attention layers.
            num_mamba_groups: # vLLM kv-cache groups holding mamba layers.
            num_attn_layers: total attention layers (logging only).
            num_mamba_layers: total mamba layers (logging only).
            group_size: layers per kv-cache group; equals the # of
                `KVCacheTensor`s vLLM allocates per kv-cache group.

        Returns:
            None. On success: sets `cache_config.num_gpu_blocks_override`
            and `_mamba_num_blocks` so `initialize_kv_cache` allocates the
            smaller mamba arrays. On any preconditions-fail path: leaves
            both unset.
        """
        cache_config = self.runner.cache_config
        if cache_config.num_gpu_blocks_override is not None:
            return

        devices = self.runner.mesh.devices.flatten()
        try:
            hbm_usage = utils.hbm_usage_bytes(devices)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Compact-mamba sizing skipped: hbm_usage_bytes failed (%s). "
                "vLLM will size the block pool from the uniform spec; "
                "page-size padding keeps per-layer block IDs in range.", exc)
            return

        total_limit = sum(limit for _, limit in hbm_usage)
        total_used = sum(used for used, _ in hbm_usage)
        gpu_mem_util = cache_config.gpu_memory_utilization
        avail = int(total_limit * gpu_mem_util - total_used)
        if avail <= 0:
            return

        # Sharding divisor: per-layer num_blocks must be a multiple of the
        # per-axis device count so JAX shardings don't round up. MLA shards
        # over MLP_TENSOR (unless DP attention is on); other layouts shard
        # over ATTN_DATA. Match `initialize_kv_cache` exactly so what we
        # compute here is what gets allocated.
        if self.use_mla and not self.runner.vllm_config.additional_config.get(
                "sharding", {}).get("sharding_strategy", {}).get(
                    "enable_dp_attention", False):
            divisor = common_utils.get_mesh_shape_product(
                self.runner.mesh, ShardingAxisName.MLP_TENSOR)
        else:
            divisor = common_utils.get_mesh_shape_product(
                self.runner.mesh, ShardingAxisName.ATTN_DATA)
        # `get_mesh_shape_product` returns 1 for an absent axis, but be
        # defensive against an empty mesh shape that produces 0.
        divisor = max(divisor, 1)

        # Mamba slot budget: one per persistent-batch slot plus the null
        # block, rounded up to the sharding divisor. `runner.max_num_reqs`
        # already includes the DP multiplier
        # (= `dp_size × scheduler_config.max_num_seqs`).
        mamba_num_blocks = self.runner.max_num_reqs + 1
        mamba_num_blocks = (
            (mamba_num_blocks + divisor - 1) // divisor) * divisor

        # Attention block count: fits into HBM left after mamba.
        # `attn_page_size_bytes` is per-block per-attention-layer; the
        # per-tensor cost is `num_attn_groups × N_attn × attn_page` because
        # one tensor backs `num_attn_groups` attention layers.
        avail_per_tensor = avail // group_size
        mamba_per_tensor = (num_mamba_groups * mamba_num_blocks *
                            unpadded_mamba_page_size_bytes)
        attn_per_tensor_avail = avail_per_tensor - mamba_per_tensor
        if attn_per_tensor_avail <= 0:
            # Mamba already saturates the budget — pathological
            # configuration (e.g., mamba_unpadded × max_num_reqs alone
            # exceeds gpu_memory_utilization × total_hbm). Skip the
            # override and let vLLM fall back to its uniform sizing; if
            # the model genuinely doesn't fit, vLLM will OOM with the
            # uniform layout too and we want that signal to surface.
            logger.warning(
                "Compact-mamba sizing skipped: mamba alone (mamba_num_blocks="
                "%d × num_mamba_groups=%d × mamba_unpadded=%d) exceeds "
                "per-tensor budget %d. Lower `gpu_memory_utilization` or "
                "`max_num_seqs`.", mamba_num_blocks, num_mamba_groups,
                unpadded_mamba_page_size_bytes, avail_per_tensor)
            return

        attn_num_blocks = attn_per_tensor_avail // (num_attn_groups *
                                                    attn_page_size_bytes)
        attn_num_blocks = (attn_num_blocks // divisor) * divisor
        if attn_num_blocks <= 0:
            logger.warning(
                "Compact-mamba sizing skipped: attn_num_blocks=0 after "
                "rounding to divisor=%d (avail_per_tensor=%d, "
                "mamba_per_tensor=%d). Lower `gpu_memory_utilization` or "
                "`max_num_seqs`.", divisor, avail_per_tensor, mamba_per_tensor)
            return

        cache_config.num_gpu_blocks_override = int(attn_num_blocks)
        self._mamba_num_blocks = int(mamba_num_blocks)

        attn_bytes = num_attn_layers * attn_num_blocks * attn_page_size_bytes
        mamba_bytes = (num_mamba_layers * mamba_num_blocks *
                       unpadded_mamba_page_size_bytes)
        logger.info(
            "Compact-mamba KV cache: num_gpu_blocks_override=%d (attn), "
            "_mamba_num_blocks=%d. HBM split: attn=%d layers × %d blocks "
            "× %d B = %.2f GiB; mamba=%d layers × %d slots × %d B = "
            "%.2f GiB; total=%.2f GiB / avail=%.2f GiB.", attn_num_blocks,
            mamba_num_blocks, num_attn_layers, attn_num_blocks,
            attn_page_size_bytes, attn_bytes / (2**30), num_mamba_layers,
            mamba_num_blocks, unpadded_mamba_page_size_bytes,
            mamba_bytes / (2**30), (attn_bytes + mamba_bytes) / (2**30),
            avail / (2**30))

    def get_kv_cache_spec(self):
        # TODO(xiang): this hack tricks engine core to init successfully
        block_size = self.runner.cache_config.block_size
        block_size *= self.runner.vllm_config.parallel_config.decode_context_parallel_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}

        tp_axis_name = ShardingAxisName.ATTN_HEAD
        model_cnt = common_utils.get_mesh_shape_product(
            self.runner.mesh, tp_axis_name)
        # If use pure jax (MODEL_IMPL_TYPE=flax_nnx), we don't register
        # attention into compilation config.
        # Use FullAttentionSpec for each layer
        # TODO(pooyam): Is it possible to merge the logic for vllm and non-vllm models?
        model_config = self.runner.model_config
        text_config = getattr(model_config, "hf_text_config",
                              getattr(model_config, "hf_config", None))
        if self.use_mla:
            # Individually pad the RopE and latents
            qk_rope_head_dim = getattr(text_config, "qk_rope_head_dim", 0)
            padded_kv_lora_rank = common_utils.align_to(
                text_config.kv_lora_rank, 128)
            padded_qk_rope_head_dim = common_utils.align_to(
                qk_rope_head_dim, 128)
            mla_head_size = padded_kv_lora_rank + padded_qk_rope_head_dim

        if len(self.runner.vllm_config.compilation_config.
               static_forward_context) == 0:
            parallel_config = self.runner.parallel_config
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
            layers = get_layers_from_vllm_config(
                self.runner.vllm_config, (Attention, MLAAttention, MambaBase))

            has_attention = any(
                isinstance(attn_module, (Attention))
                for attn_module in layers.values())
            has_mamba = any(
                isinstance(attn_module, MambaBase)
                for attn_module in layers.values())

            # Cache config update for hybrid attention models with mamba and
            # full attention layers.
            is_hybrid_mamba_attention = has_attention and has_mamba
            if is_hybrid_mamba_attention:
                # Unify the page sizes of mamba and full attention layers to
                # enable use of shared kv cache, vLLM also expects page sizes to
                # be unified.
                self.update_mamba_page_size_padded(layers)

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

    def get_kv_cache_layout(self):
        # return the layout (mostly "NHD" or "HND") of kv cache
        return get_kv_cache_layout()

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
            num_speculative_tokens = 0
            if self.runner.vllm_config.speculative_config:
                num_speculative_tokens = self.runner.vllm_config.speculative_config.num_speculative_tokens
            new_input_batch = InputBatch(
                max_num_reqs=self.runner.max_num_reqs,
                max_model_len=self.runner.max_model_len,
                max_num_batched_tokens=self.runner.max_num_tokens,
                pin_memory=False,
                vocab_size=self.runner.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                num_speculative_tokens=num_speculative_tokens,
            )
            self.runner.input_batch = new_input_batch
            self.runner.persistent_batch_manager.input_batch = new_input_batch

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

        # set the kv cache layout which is needed by kv connectors
        # NOTE(jcgu): please update the default value when the order changes
        set_kv_cache_layout(DEFAULT_KV_CACHE_LAYOUT)
        # verify kv cache layout is matched between the cache manager and
        # the kv connector (if configured)
        _required_kv_layout = get_kv_connector_cache_layout()
        if (_required_kv_layout
                and _required_kv_layout != DEFAULT_KV_CACHE_LAYOUT):
            raise ValueError(
                f"KV cache layout ({DEFAULT_KV_CACHE_LAYOUT}) does not match with the "
                f"kv_connector's required layout ({_required_kv_layout})")

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
                    non_mtp_tensors = [
                        t for t in kv_cache_config.kv_cache_tensors
                        if not any("mtp" in name for name in t.shared_by)
                    ]
                    if non_mtp_tensors:
                        # assert that each kv_cache_tensor in kv_cache_config.kv_cache_tensors has the same number of shared layers
                        # This is needed for models like Qwen3.5 where every 4 layers share the same KV cache (3 linear attn and 1 full attn)
                        num_shared_layers = len(non_mtp_tensors[0].shared_by)
                        for kv_cache_tensor in non_mtp_tensors:
                            assert len(
                                kv_cache_tensor.shared_by
                            ) == num_shared_layers, f"Expected all non-MTP kv_cache_tensors to have the same number of shared layers {num_shared_layers}, but found {len(kv_cache_tensor.shared_by)}"
                    break

        for i, kv_cache_tensor in enumerate(kv_cache_config.kv_cache_tensors):
            if duplicate_shared_layers:
                total_group_page_size = 0
                for name in kv_cache_tensor.shared_by:
                    spec = layer_name_to_spec[name]
                    # Use the per-layer *TPU-actual* per-block bytes so the
                    # sum equals the `page_size_padded` that
                    # `update_mamba_page_size_padded` installed on every
                    # spec (== attn_page + N × mamba_unpadded). For
                    # attention, the TPU-actual size includes dtype-
                    # specific packing (e.g., fp8 KV packs 4 elements per
                    # 32-bit word) which `spec.real_page_size_bytes`
                    # doesn't account for — on fp8 models they differ by
                    # 2×, which would break the num_blocks match here.
                    if isinstance(spec, MambaSpec):
                        total_group_page_size += dataclasses.replace(
                            spec, page_size_padded=None).page_size_bytes
                    else:
                        total_group_page_size += get_attention_page_size_bytes(
                            self.runner.mesh, spec.block_size,
                            spec.num_kv_heads, spec.head_size, spec.dtype,
                            self.use_mla)
                num_blocks = kv_cache_tensor.size // total_group_page_size
            else:
                # If sharing KV cache, compute `num_blocks` using the page size
                # of the first layer.
                page_size_bytes = layer_name_to_spec[
                    kv_cache_tensor.shared_by[0]].page_size_bytes
                assert kv_cache_tensor.size % page_size_bytes == 0
                num_blocks = kv_cache_tensor.size // page_size_bytes

            # Default KV cache is sharded over (BATCH=(dp, attn_dp))
            divisor = common_utils.get_mesh_shape_product(
                self.runner.mesh, ShardingAxisName.BATCH)

            # num_blocks must be a multiple of the sharding divisor
            num_blocks = (num_blocks // divisor) * divisor

            # When compact-mamba sizing succeeded (set by
            # `_maybe_set_compact_mamba_num_blocks_override`), mamba layers
            # allocate `_mamba_num_blocks` (= max_num_reqs + 1) slots while
            # attention layers keep the full `num_blocks`. When that override
            # didn't run (CPU tests, user-pinned `num_gpu_blocks_override`),
            # mamba and attn use the same `num_blocks` — vLLM's uniform sizing.
            mamba_num_blocks = (self._mamba_num_blocks
                                if self._mamba_num_blocks is not None else
                                num_blocks)

            for j, layer_name in enumerate(kv_cache_tensor.shared_by):
                layer_spec = layer_name_to_spec[layer_name]
                if isinstance(layer_spec, MambaSpec):
                    mamba_states = []
                    for state_index, (shape, dtype) in enumerate(
                            zip(layer_spec.shapes, layer_spec.dtypes)):
                        jax_dtype = t2j_dtype(dtype)
                        cache_shape = (mamba_num_blocks, *shape)
                        if state_index == 0:
                            # conv_state: [num_blocks, conv_kernel_size, intermediate_size]
                            spec = PartitionSpec(ShardingAxisName.ATTN_DATA,
                                                 None,
                                                 ShardingAxisName.ATTN_HEAD)
                        elif state_index == 1:
                            # ssm_state: [num_blocks, num_heads, head_dim, state_size]
                            spec = PartitionSpec(ShardingAxisName.ATTN_DATA,
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
                    model_config = self.runner.model_config
                    text_config = getattr(
                        model_config, "hf_text_config",
                        getattr(model_config, "hf_config", None))
                    if j == 0 or duplicate_shared_layers:
                        # NOTE: we'll multiply the num_kv_heads by 2 in the function
                        if self.use_mla:

                            head_size = text_config.kv_lora_rank + \
                                text_config.qk_rope_head_dim
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
                    num_blocks_list.append(mamba_num_blocks if isinstance(
                        layer_spec, MambaSpec) else num_blocks)
                layer_idx = (i * num_shared_layers
                             ) + j if duplicate_shared_layers else i
                self.runner.layer_name_to_kvcache_index[layer_name] = layer_idx
        if self.shared_kv_cache_layers:
            for layer_name, target_layer_name in self.shared_kv_cache_layers.items(
            ):
                self.runner.layer_name_to_kvcache_index[
                    layer_name] = self.runner.layer_name_to_kvcache_index[
                        target_layer_name]

        logger.info(
            "Hybrid KV cache layout: num_kv_cache_groups=%d, "
            "num_kv_cache_tensors=%d, kv_cache_config.num_blocks=%d, "
            "duplicate_shared_layers=%s", len(kv_cache_config.kv_cache_groups),
            len(kv_cache_config.kv_cache_tensors), kv_cache_config.num_blocks,
            duplicate_shared_layers)

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

        with set_current_vllm_config(self.runner.vllm_config):
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
