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

import copy
import functools
import inspect
import time
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, List, Optional, Tuple
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn
import torchax
import vllm.envs as vllm_envs
from flax.typing import PRNGKey
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import TORCH_DTYPE_TO_JAX
from torchax.ops.ops_registry import register_torch_function_op
from vllm.config import VllmConfig, set_current_vllm_config, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.pooler import Pooler
from vllm.model_executor.model_loader import get_model as vllm_get_model
from vllm.model_executor.models import supports_lora, supports_multimodal
from vllm.model_executor.models.interfaces_base import is_pooling_model
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

from tpu_inference.distributed.jax_parallel_state import \
    get_pp_group as jax_get_pp_group
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm import ops as patch_ops
from tpu_inference.layers.vllm.mla_attention import \
    VllmTPUMultiHeadLatentAttentionWrapper
from tpu_inference.layers.vllm.process_weights.cleanup_sharding import \
    shard_model_to_tpu
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.logger import init_logger
from tpu_inference.models.common.interface import PoolerFunc
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context, set_vllm_model_wrapper_context)
from tpu_inference.runner.lora_utils import replace_lora_metadata

logger = init_logger(__name__)


class _VllmRunner(torch.nn.Module):

    def __init__(self, vllm_model: torch.nn.Module):
        super().__init__()
        self.vllm_model = vllm_model

        has_pooler = is_pooling_model(vllm_model)
        self.pooler = vllm_model.pooler if has_pooler else None

    def forward(self, **kwargs) -> torch.Tensor:
        if "hidden_state" in kwargs:
            return self.compute_logits(kwargs["hidden_state"])
        elif "call_method" in kwargs:
            method_name = kwargs["call_method"]
            call_args = kwargs.get("call_args", tuple())
            call_kwargs = kwargs.get("call_kwargs", {})
            method = getattr(self.vllm_model, method_name)
            return method(*call_args, **call_kwargs)
        else:
            return self.compute_hidden_state(
                kwargs["input_ids"],
                kwargs["positions"],
                kwargs["intermediate_tensors"],
                kwargs["inputs_embeds"],
            )

    def compute_hidden_state(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden_state = self.vllm_model(input_ids, positions,
                                       intermediate_tensors, inputs_embeds)
        return hidden_state

    def compute_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.vllm_model.compute_logits(hidden_state)


class VllmModelWrapper:
    """ Wraps a vLLM Pytorch model and let it run on the JAX engine. """

    rng: PRNGKey
    mesh: Mesh
    model: _VllmRunner

    def __init__(self, vllm_config: VllmConfig, rng: PRNGKey, mesh: Mesh):
        self.vllm_config = vllm_config
        self.rng = rng
        self.mesh = mesh

        self.vllm_config.quant_config = get_tpu_quantization_config(
            self.vllm_config, self.mesh)
        self._apply_pp_patch()
        self._patch_vllm_ops()

        from vllm.model_executor.custom_op import op_registry_oot
        if MultiHeadLatentAttentionWrapper.__name__ not in op_registry_oot:
            MultiHeadLatentAttentionWrapper.register_oot(
                VllmTPUMultiHeadLatentAttentionWrapper)

    def _patch_sdpa(self):
        from torchax.ops.jtorch import register_function

        from tpu_inference.layers.common.attention_interface import \
            sharded_flash_attention

        @register_function(
            torch.nn.functional.scaled_dot_product_attention,
            is_jax_function=True,
            needs_env=False,
        )
        def patched_sdpa(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            enable_gqa=False,
        ):
            if dropout_p != 0.0:
                raise NotImplementedError(
                    "patched_sdpa does not support dropout_p")
            if enable_gqa is not False:
                raise NotImplementedError(
                    "patched_sdpa does not support enable_gqa")

            # Q, K, V shapes: (batch, num_heads, seq_len, head_dim)
            batch = query.shape[0]
            num_heads = query.shape[1]
            q_seq_len = query.shape[2]
            kv_seq_len = key.shape[2]

            # padding due to the requirement of sharded_flash_attention
            q_pad = (128 - (q_seq_len % 128)) % 128
            kv_pad = (128 - (kv_seq_len % 128)) % 128

            if q_pad > 0:
                query = jnp.pad(query, ((0, 0), (0, 0), (0, q_pad), (0, 0)))
            if kv_pad > 0:
                key = jnp.pad(key, ((0, 0), (0, 0), (0, kv_pad), (0, 0)))
                value = jnp.pad(value, ((0, 0), (0, 0), (0, kv_pad), (0, 0)))

            # Prevent nan while using -inf
            mask_value = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
            ab = jnp.zeros((batch, num_heads, q_seq_len, kv_seq_len),
                           dtype=jnp.float32)
            if attn_mask is not None:
                # attn_mask shape: (batch, num_heads, q_len, kv_len)
                if attn_mask.dtype == jnp.bool_:
                    ab = jnp.where(attn_mask, ab, mask_value)
                else:
                    ab += attn_mask

            if q_pad > 0 or kv_pad > 0:
                ab = jnp.pad(
                    ab,
                    ((0, 0), (0, 0), (0, q_pad), (0, kv_pad)),
                    mode="constant",
                    constant_values=mask_value,
                )

            attn_fn = sharded_flash_attention(self.mesh,
                                              causal=is_causal,
                                              sm_scale=scale,
                                              use_attention_bias=True)
            out = attn_fn(query, key, value, ab, None)

            if q_pad > 0:
                out = out[:, :, :q_seq_len, :]

            return out

    def _patch_vllm_ops(self):
        # Caution: there is no public api for restore the ops.
        # It need to patched again if the ops are jitted and mesh is change.
        # The overwritten ops should not be called after the end of model wrapper.

        # Import the registered ops at first and then we can overwrite them.
        import torchax.ops.jtorch  # noqa: F401

        # Patch sdpa from torch ops to flash attention to prevent OOM
        register_torch_function_op(
            torch.nn.functional.scaled_dot_product_attention,
            functools.partial(patch_ops.scaled_dot_product_attention,
                              mesh=self.mesh),
            is_jax_function=True,
            needs_env=False,
        )

        @register_function(
            torch.ops.vllm.torch_sdpa_wrapper,
            is_jax_function=True,
            needs_env=False,
        )
        def patched_vllm_vit_sdpa(
            query,
            key,
            value,
            scale=None,
            cu_seqlens=None,
            enable_gqa=False,
        ):

            # Inputs are JAX arrays in shape [B, S, N, D]
            # Rearrange to [B, N, S, D]
            query = jnp.swapaxes(query, 1, 2)
            key = jnp.swapaxes(key, 1, 2)
            value = jnp.swapaxes(value, 1, 2)

            batch = query.shape[0]
            num_heads = query.shape[1]
            q_seq_len = query.shape[2]
            kv_seq_len = key.shape[2]

            # Padding due to requirement of sharded_flash_attention
            q_pad = (128 - (q_seq_len % 128)) % 128
            kv_pad = (128 - (kv_seq_len % 128)) % 128

            if q_pad > 0:
                query = jnp.pad(query, ((0, 0), (0, 0), (0, q_pad), (0, 0)))
            if kv_pad > 0:
                key = jnp.pad(key, ((0, 0), (0, 0), (0, kv_pad), (0, 0)))
                value = jnp.pad(value, ((0, 0), (0, 0), (0, kv_pad), (0, 0)))

            if cu_seqlens is not None:
                # Convert cu_seqlens to SegmentIds
                cu_seqlens_arr = jnp.array(cu_seqlens)
                lens = cu_seqlens_arr[1:] - cu_seqlens_arr[:-1]
                num_segs = lens.shape[0]

                # Real segments
                q_real_seg = jnp.repeat(jnp.arange(num_segs), lens, total_repeat_length=q_seq_len)
                kv_real_seg = q_real_seg  # Assuming Q and KV sequence lengths are same for ViT self attention

                if q_pad > 0:
                    q_pad_seg = jnp.full((q_pad,), num_segs)
                    q_seg = jnp.concatenate([q_real_seg, q_pad_seg])
                else:
                    q_seg = q_real_seg

                if kv_pad > 0:
                    kv_pad_seg = jnp.full((kv_pad,), num_segs)
                    kv_seg = jnp.concatenate([kv_real_seg, kv_pad_seg])
                else:
                    kv_seg = kv_real_seg

                q_seg = jnp.broadcast_to(q_seg, (batch, q_seg.shape[0]))
                kv_seg = jnp.broadcast_to(kv_seg, (batch, kv_seg.shape[0]))

                from tpu_inference.kernels.flash_attention.kernel import SegmentIds
                seg_ids = SegmentIds(q=q_seg, kv=kv_seg)
            else:
                seg_ids = None

            attn_fn = sharded_flash_attention(
                self.mesh,
                causal=False,
                sm_scale=scale,
                use_attention_bias=False
            )

            out = attn_fn(query, key, value, seg_ids)

            if q_pad > 0:
                out = out[:, :, :q_seq_len, :]

            # Rearrange back [B, N, S, D] -> [B, S, N, D]
            out = jnp.swapaxes(out, 1, 2)

            return out

    def _apply_pp_patch(self):
        # patch `get_pp_group` in vLLM to jax's get_pp_group.
        import sys

        import vllm.distributed as vllm_dist
        import vllm.distributed.parallel_state as vllm_ps

        vllm_ps.get_pp_group = jax_get_pp_group
        vllm_dist.get_pp_group = jax_get_pp_group

        for module_name, module in sys.modules.items():
            if module_name.startswith("vllm.model_executor.models"):
                if hasattr(module, "get_pp_group"):
                    setattr(module, "get_pp_group", jax_get_pp_group)

    def _apply_qwen3_vl_patches(self, vllm_model):
        """
        Apply Qwen3-VL specific monkey-patches for stateless Deepstack support.
        This allows passing intermediate vision embeddings through JIT boundaries
        by packing them into inputs_embeds.
        """
        if not hasattr(vllm_model, "config") or "qwen" not in getattr(vllm_model.config, "model_type", "").lower():
            return

        if not getattr(vllm_model, "use_deepstack", False):
            return

        from torchax.interop import jax_view, torch_view

        from tpu_inference.distributed.jax_parallel_state import \
            get_pp_group as jax_get_pp_group

        logger.info("Applying Qwen3-VL stateless Deepstack patches")

        # 1. Override setter to avoid in-place mutation error
        orig_set_deepstack = getattr(vllm_model, "_set_deepstack_input_embeds", None)
        if orig_set_deepstack is not None:
            def patched_set_deepstack(deepstack_input_embeds):
                vllm_model._deepstack_tensors = {}
                if isinstance(deepstack_input_embeds, dict):
                    vllm_model._deepstack_tensors.update(deepstack_input_embeds)
                elif isinstance(deepstack_input_embeds, (list, tuple)):
                    indexes = getattr(vllm_model.config.vision_config, "deepstack_visual_indexes", [])
                    for idx, v in enumerate(deepstack_input_embeds):
                        key = f"deepstack_input_embeds_{indexes[idx]}" if idx < len(indexes) else f"deepstack_input_embeds_{idx}"
                        vllm_model._deepstack_tensors[key] = v
            vllm_model._set_deepstack_input_embeds = patched_set_deepstack

        orig_get_deepstack = getattr(vllm_model, "_get_deepstack_input_embeds", None)
        if orig_get_deepstack is not None:
            from vllm.sequence import IntermediateTensors
            
            def patched_get_deepstack(num_tokens: int):
                orig_output = orig_get_deepstack(num_tokens)
                converted = {}
                for k, v in orig_output.items():
                    if not v.__class__.__module__.startswith("torchax"):
                        try:
                            # If it's a standard PyTorch Tensor (static), convert it via numpy
                            # to a JAX array, then wrap in torch_view for JAX compatibility.
                            import jax
                            val_f32 = v.detach().cpu().float().numpy()
                            jax_arr = jax.device_put(val_f32).astype(jax.numpy.bfloat16)
                            v = torch_view(jax_arr)
                        except Exception:
                            # If it fails, it might be dynamic/tracer, we fall back to jax_view 
                            # (which will fail if it's not torchax but at least it fails loudly if it's an unsupported case).
                            v = torch_view(jax_view(v))
                    converted[k] = v
                return IntermediateTensors(converted)
            vllm_model._get_deepstack_input_embeds = patched_get_deepstack

        # 2. Patch embed_input_ids to pack state
        orig_embed_input_ids = getattr(vllm_model, "embed_input_ids", None)
        if orig_embed_input_ids is not None:
            def patched_embed_input_ids(*args, **kwargs):
                inputs_embeds = orig_embed_input_ids(*args, **kwargs)
                deepstack_input_embeds = getattr(vllm_model, "deepstack_input_embeds", None)
                if deepstack_input_embeds is not None:
                    if torch.is_tensor(deepstack_input_embeds):
                        packed = deepstack_input_embeds.transpose(0, 1).reshape(inputs_embeds.size(0), -1)
                        inputs_embeds = torch.cat([inputs_embeds, packed], dim=-1)
                return inputs_embeds
            vllm_model.embed_input_ids = patched_embed_input_ids

        # 3. Patch forward to unpack state
        orig_forward = vllm_model.forward
        def patched_forward(input_ids, positions, intermediate_tensors, inputs_embeds=None, **kwargs):
            if inputs_embeds is not None and jax_get_pp_group().is_first_rank:
                if getattr(vllm_model, "use_deepstack", False) and inputs_embeds.shape[-1] > vllm_model.visual_dim:
                    packed_dim = inputs_embeds.shape[-1] - vllm_model.visual_dim
                    deepstack_packed = inputs_embeds[..., vllm_model.visual_dim:]
                    inputs_embeds = inputs_embeds[..., :vllm_model.visual_dim]
                    
                    deepstack_input_embeds = {}
                    num_levels = getattr(vllm_model, "deepstack_num_level", 1)
                    per_level_dim = packed_dim // num_levels
                    indexes = getattr(vllm_model.config.vision_config, "deepstack_visual_indexes", [])
                    
                    for idx, layer_idx in enumerate(indexes):
                        start = idx * per_level_dim
                        end = (idx + 1) * per_level_dim
                        sliced = deepstack_packed[..., start:end]
                        if not isinstance(sliced, torch.Tensor) and "torchax.tensor" not in str(type(sliced)):
                            sliced = torch_view(jax_view(sliced))
                        deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"] = sliced

                    vllm_model._set_deepstack_input_embeds(deepstack_input_embeds)

            return orig_forward(input_ids=input_ids, positions=positions, 
                                intermediate_tensors=intermediate_tensors, 
                                inputs_embeds=inputs_embeds, **kwargs)
        vllm_model.forward = patched_forward

    def load_weights(self):
        loading_start = time.time()
        # Set up to load the model into CPU first.
        # Cache device slice config since device config cannot be deepcopied
        modified_slice_config = False
        if hasattr(
                self.vllm_config.device_config,
                'slice') and self.vllm_config.device_config.slice is not None:
            slice_config = self.vllm_config.device_config.slice
            modified_slice_config = True
            self.vllm_config.device_config.slice = None
        self.vllm_config.compilation_config.static_forward_context.clear()

        vllm_config_for_load = copy.deepcopy(self.vllm_config)
        if modified_slice_config:
            self.vllm_config.device_config.slice = slice_config
        assert self.vllm_config.model_config.dtype in TORCH_DTYPE_TO_JAX, "The model_config.dtype must be a PyTorch dtype."
        vllm_config_for_load.device_config.device = "cpu"
        # Remove the dynamically added sharding_config attribute to avoid errors
        # when vLLM's replace() function checks for dataclass fields.
        # This is safe because vllm_config_for_load is only used for model loading
        # which doesn't need sharding_config, and self.vllm_config still has it.
        if hasattr(vllm_config_for_load, 'sharding_config'):
            delattr(vllm_config_for_load, 'sharding_config')
        # Clearing the cached compilation config, otherwise vllm model init will fail

        # When expert parallelism is enabled, vLLM loads weight in sharding
        # aware manner. Since tpu-inference has its own sharding logic, this
        # may casue errors. Therefore, we disable it during weight loading.
        vllm_config_for_load.parallel_config.enable_expert_parallel = False

        use_random_weights = (
            vllm_config_for_load.load_config.load_format == "dummy")
        if use_random_weights:
            logger.info(
                "Initializing vLLM model with random weights, weight loading skipped."
            )
        # The DummyModelLoader in vLLM calls torch._sync for torch_xla path when
        # it detects the tpu platform, but we don't need it and it causes crash
        # without proper setup.
        load_context = patch(
            "torch._sync",
            return_value=None) if use_random_weights else nullcontext()

        # By default load weights to the CPU device first. If we are running
        # under Pathways, this would cause weights to be loaded on a CPU-only
        # node, so we'll need to remove this context.
        jax_context = jax.default_device(
            jax.devices("cpu")
            [0]) if not vllm_envs.VLLM_TPU_USING_PATHWAYS else nullcontext()
        # Load the vLLM model and wrap it into a new model whose forward
        # function can calculate the hidden_state and logits.
        with load_context, jax_context:
            with set_current_vllm_config(vllm_config_for_load):
                vllm_model = vllm_get_model(vllm_config=vllm_config_for_load)
                self._apply_qwen3_vl_patches(vllm_model)
        lora_manager = None
        if vllm_config_for_load.lora_config is not None:
            # Replace layers in the model with LoRA layers.
            with torchax.default_env():
                # Argument "device" in load_lora_model is used to set the device
                # used in punica wrapper.
                lora_manager, vllm_model = load_lora_model(
                    vllm_model, vllm_config_for_load, device="jax")
            replace_set_lora(vllm_model)

        static_forward_context = vllm_config_for_load.compilation_config.static_forward_context
        self.vllm_config.compilation_config.static_forward_context = static_forward_context
        self.vllm_config.compilation_config.static_all_moe_layers = vllm_config_for_load.compilation_config.static_all_moe_layers

        self.model = _VllmRunner(vllm_model)
        params_and_buffers = shard_model_to_tpu(self.model, self.mesh)

        self._pooler: Pooler | None = self.model.pooler

        loading_end = time.time()
        total_loading_time = loading_end - loading_start
        logger.info(
            f"Total time to load model weights from storage to TPU: {total_loading_time:.2f} seconds."
        )
        # Returning to the jax land, so we need to wrap it into a JaxValue.
        return jax_view(params_and_buffers), lora_manager

    def jit_step_func(self):

        @jax.jit(
            donate_argnames=("kv_caches", ),
            out_shardings=(
                None,  # kv_caches - keep original sharding
                NamedSharding(self.mesh,
                              PartitionSpec(ShardingAxisName.ATTN_DATA, None)),
                None,  # empty list
            ),
            compiler_options={
                "xla_tpu_all_gather_collective_matmul_mode":
                "post_spmd_conservative",
                "xla_tpu_reduce_scatter_collective_matmul_mode":
                "post_spmd_conservative"
            },
            static_argnames=(
                "layer_name_to_kvcache_index",
                "is_first_rank",
                "is_last_rank",
            ),
        )
        def step_fun(
            params_and_buffers,  # This has been wrapped into torchax TorchValue
            kv_caches: List[jax.Array],
            input_ids: jax.Array,
            attn_metadata: AttentionMetadata,
            input_embeds: jax.Array,
            input_positions: jax.Array,
            layer_name_to_kvcache_index: Sequence[Tuple[str, int]],
            lora_metadata,
            intermediate_tensors: JaxIntermediateTensors = None,
            is_first_rank: bool = True,
            is_last_rank: bool = True,
            *args,
        ) -> Tuple[List[jax.Array], jax.Array]:
            layer_name_to_kvcache_index = dict(layer_name_to_kvcache_index)
            lora_metadata = torch_view(lora_metadata)
            with torchax.default_env(), set_vllm_model_wrapper_context(
                    kv_caches=kv_caches,
                    mesh=self.mesh,
                    layer_name_to_kvcache_index=layer_name_to_kvcache_index
            ), set_forward_context(attn_metadata=attn_metadata,
                                   vllm_config=self.vllm_config):
                # We need to wrap args from jax land into TorchValue with
                # torch_view in order to call the Torch function.
                original_lora_metadata = replace_lora_metadata(
                    self.model, lora_metadata, self.vllm_config.lora_config)
                if not is_first_rank:
                    intermediate_tensors = intermediate_tensors.to_torch()
                output_from_torch = torch.func.functional_call(
                    self.model,
                    torch_view(params_and_buffers),
                    kwargs={
                        "input_ids": torch_view(input_ids),
                        "positions": torch_view(input_positions),
                        "intermediate_tensors": intermediate_tensors,
                        "inputs_embeds": torch_view(input_embeds),
                    },
                    tie_weights=False,
                )
                replace_lora_metadata(self.model, original_lora_metadata,
                                      self.vllm_config.lora_config)
                vllm_model_wrapper_context = get_vllm_model_wrapper_context()
                new_kv_caches = vllm_model_wrapper_context.kv_caches
            # Wrap the output(hidden states or intermediate tensor)
            # from torch land into a JaxValue for the jax code to consume.
            if not is_last_rank:
                output = JaxIntermediateTensors.from_torch(output_from_torch)
            else:
                output = jax_view(output_from_torch)
            return new_kv_caches, output, []

        return step_fun

    def wrap_embed_multimodal_func(self):
        if not self.vllm_config.model_config.is_multimodal_model:
            return None

        # The function cannot be JITted directly due to its dynamic implementation
        def embed_multimodal_func(
            params_and_buffers: Any,
            image_grid_thw: Any,
            **kwargs,
        ) -> Any:
            inner_model = getattr(self.model, "vllm_model", self.model)
            method = getattr(inner_model, "embed_multimodal", None)
            sig = inspect.signature(method) if method else None
            has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()) if sig else False
            supports_image_grid_thw = sig and ("image_grid_thw" in sig.parameters or has_var_keyword)

            # Delete image_grid_thw if it's not supported to avoid passing it to models that don't want it.
            if not supports_image_grid_thw:
                del image_grid_thw

            with torchax.default_env():
                call_kwargs = {}
                if supports_image_grid_thw:
                    call_kwargs["image_grid_thw"] = torch.tensor(
                        image_grid_thw, dtype=torch.long)
                for k, v in kwargs.items():
                    if isinstance(v, jax.Array):
                        call_kwargs[k] = torch_view(v)
                    elif isinstance(v, np.ndarray):
                        # The "pixel_values" need to be a torch.Tensor.
                        # Cast it back to torch.Tensor.
                        call_kwargs[k] = torch_view(jnp.array(v))
                    else:
                        call_kwargs[k] = v

                output_from_torch = torch.func.functional_call(
                    self.model,
                    torch_view(params_and_buffers),
                    kwargs={
                        "call_method": "embed_multimodal",
                        "call_args": (),
                        "call_kwargs": call_kwargs,
                    },
                    tie_weights=False,
                )

                return jax_view(output_from_torch)

        return embed_multimodal_func

    def _wrap_generic_embed_input_ids_func(self):
        # The function cannot be JITted directly due to its dynamic implementation
        def embed_input_ids_func(
            params_and_buffers: Any,
            input_ids: jax.Array,
            mm_embeds: list[jax.Array] | jax.Array | None = None,
            *,
            is_multimodal: jax.Array | None = None,
        ) -> jax.Array:
            with torchax.default_env():
                if mm_embeds is not None:
                    if isinstance(mm_embeds, list):
                        torch_mm_embeds = [torch_view(x) for x in mm_embeds]
                    else:
                        torch_mm_embeds = torch_view(mm_embeds)
                    call_args = (torch_view(input_ids), torch_mm_embeds)
                else:
                    call_args = (torch_view(input_ids), )

                call_kwargs = {
                    "is_multimodal": torch_view(is_multimodal),
                }
                if handle_oov_mm_token:
                    call_kwargs["handle_oov_mm_token"] = handle_oov_mm_token

                output_from_torch = torch.func.functional_call(
                    self.model,
                    torch_view(params_and_buffers),
                    kwargs={
                        "call_method": "embed_input_ids",
                        "call_args": call_args,
                        "call_kwargs": call_kwargs,
                    },
                    tie_weights=False,
                )

                return jax_view(output_from_torch)

        return embed_input_ids_func

    def _wrap_qwen3_vl_embed_input_ids_func(self):
        # Specific fix for Qwen3-VL where multimodal embeds must always be a list or tuple of Tensors.
        def embed_input_ids_func(
            params_and_buffers: Any,
            input_ids: jax.Array,
            mm_embeds: list[jax.Array] | jax.Array | None = None,
            *,
            is_multimodal: jax.Array | None = None,
            handle_oov_mm_token: bool = False,
        ) -> jax.Array:
            with torchax.default_env():
                if mm_embeds is not None:
                    if isinstance(mm_embeds, list):
                        torch_mm_embeds = [torch_view(x) for x in mm_embeds]
                    else:
                        torch_mm_embeds = [torch_view(mm_embeds)] # Fix: always pass a list of length 1
                    call_args = (torch_view(input_ids), torch_mm_embeds)
                else:
                    call_args = (torch_view(input_ids), )

                call_kwargs = {
                    "is_multimodal": torch_view(is_multimodal),
                }
                if handle_oov_mm_token:
                    call_kwargs["handle_oov_mm_token"] = handle_oov_mm_token

                output_from_torch = torch.func.functional_call(
                    self.model,
                    torch_view(params_and_buffers),
                    kwargs={
                        "call_method": "embed_input_ids",
                        "call_args": call_args,
                        "call_kwargs": call_kwargs,
                    },
                    tie_weights=False,
                )

                return jax_view(output_from_torch)

        return embed_input_ids_func

    def wrap_embed_input_ids_func(self):
        if not self.vllm_config.model_config.is_multimodal_model:
            return None

        is_qwen3_vl = type(self.model.vllm_model).__name__ == "Qwen3VLForConditionalGeneration"
        if is_qwen3_vl:
            return self._wrap_qwen3_vl_embed_input_ids_func()
        else:
            return self._wrap_generic_embed_input_ids_func()

    def jit_compute_logits_func(self):

        # TODO(gxd3): revisit if the sharding below is the best way to shard the
        # output logits.
        @jax.jit(out_shardings=(NamedSharding(
            self.mesh,
            PartitionSpec(ShardingAxisName.MLP_DATA,
                          ShardingAxisName.MLP_TENSOR))))
        def compute_logits_func(
            params_and_buffers: Any,
            hidden_states: jax.Array,
            lora_metadata,
        ) -> jax.Array:
            lora_metadata = torch_view(lora_metadata)
            with torchax.default_env(), set_vllm_model_wrapper_context(
                    kv_caches=None, mesh=self.mesh):
                original_lora_metadata = replace_lora_metadata(
                    self.model, lora_metadata, self.vllm_config.lora_config)
                logits = torch.func.functional_call(
                    self.model,
                    torch_view(params_and_buffers),
                    kwargs={
                        "hidden_state": torch_view(hidden_states),
                    },
                    tie_weights=False,
                )
                replace_lora_metadata(self.model, original_lora_metadata,
                                      self.vllm_config.lora_config)
            return jax_view(logits)

        return compute_logits_func

    def build_pooler_func(self) -> PoolerFunc:

        def compute_pooler_output(
            hidden_states: jax.Array,
            pooling_metadata: PoolingMetadata,
            seq_lens: np.ndarray,
        ) -> PoolerOutput:
            assert self._pooler is not None, "Model does not support pooling"

            torch_states: torch.Tensor = torch_view(hidden_states)
            with torchax.default_env():
                torch_states = torch_states.to('cpu', non_blocking=True)
                pooling_metadata.build_pooling_cursor(
                    seq_lens,
                    torch.tensor(seq_lens),
                    device=torch_states.device,
                )
                outputs: list[torch.Tensor] = self._pooler(
                    torch_states,
                    pooling_metadata,
                )
                return outputs

        return compute_pooler_output


def load_lora_model(model: torch.nn.Module, vllm_config: VllmConfig,
                    device: str) -> torch.nn.Module:
    if not supports_lora(model):
        raise ValueError(
            f"{model.__class__.__name__} does not support LoRA yet.")

    if supports_multimodal(model):
        logger.warning("Regarding multimodal models, vLLM currently "
                       "only supports adding LoRA to language model.")

    # Add LoRA Manager to the Model Runner
    lora_manager = LRUCacheWorkerLoRAManager(
        vllm_config,
        device,
        model.embedding_modules,
    )
    return lora_manager, lora_manager.create_lora_manager(model)


# The reason why replace the method is that the set_lora and reset_lora need to
# run under torchax env.
def replace_set_lora(model):

    def _tpu_set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
    ):
        with torchax.default_env():
            self._original_set_lora(index, lora_a, lora_b)

    def _tpu_reset_lora(self, index: int):
        with torchax.default_env():
            self._original_reset_lora(index)

    for _, module in model.named_modules():
        if isinstance(module, BaseLayerWithLoRA):
            module._original_set_lora = module.set_lora
            module._original_reset_lora = module.reset_lora
            module.set_lora = _tpu_set_lora.__get__(module, module.__class__)
            module.reset_lora = _tpu_reset_lora.__get__(
                module, module.__class__)
