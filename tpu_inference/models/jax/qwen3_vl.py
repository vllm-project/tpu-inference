# ruff: noqa: F821, F722
# Copyright 2026 Google LLC
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
from __future__ import annotations

from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from jax.sharding import Mesh
from jaxtyping import Array, Float, Int
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig, Qwen3VLVisionConfig)
from vllm.config import VllmConfig

from tpu_inference import utils as utils
from tpu_inference.layers.jax.layers import FlaxUtils
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.qwen2_5_vl import (Qwen2_5_VisionAttention,
                                                 Qwen2_5_VisionRotaryEmbedding)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()
modeling_flax_utils = FlaxUtils()


class Qwen3_VisionPatchEmbed(nnx.Module):

    def __init__(
        self,
        rngs: nnx.Rngs,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nnx.Conv(in_features=in_channels,
                             out_features=hidden_size,
                             kernel_size=kernel_size,
                             strides=kernel_size,
                             use_bias=True,
                             param_dtype=dtype,
                             kernel_init=nnx.with_partitioning(
                                 init_fn, (None, None, None, None, "model")),
                             bias_init=nnx.with_partitioning(
                                 init_fn, ("model", )),
                             rngs=rngs)

    def __call__(
        self, x: Float[Array, "seq_len flat_patch_dim"]
    ) -> Float[Array, "seq_len hidden_size"]:
        L, dim = x.shape
        C = dim // (self.temporal_patch_size * self.patch_size *
                    self.patch_size)
        x = x.reshape(L, C, self.temporal_patch_size, self.patch_size,
                      self.patch_size)
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = self.proj(x)
        x = x.reshape(L, self.hidden_size)
        return x


class Qwen3_VisionMLP(nnx.Module):

    def __init__(self, config: Qwen3VLVisionConfig, dtype: jnp.dtype,
                 rngs: nnx.Rngs):
        in_features = config.hidden_size
        hidden_features = config.intermediate_size

        act_name = config.hidden_act
        if act_name == "gelu_pytorch_tanh":
            act_fn = partial(nnx.gelu, approximate=True)
        else:
            act_fn = modeling_flax_utils.ACT2FN[act_name]

        # Qwen3 Vision uses standard two-layer MLP (no gated GLU)
        self.linear_fc1 = nnx.Linear(
            in_features,
            hidden_features,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model", )),
            rngs=rngs,
        )
        self.linear_fc2 = nnx.Linear(
            hidden_features,
            in_features,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rngs,
        )
        self.act_fn = act_fn

    def __call__(
        self, x: Float[Array, "seq_len hidden_size"]
    ) -> Float[Array, "seq_len hidden_size"]:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3_VisionBlock(nnx.Module):

    def __init__(self, config: Qwen3VLConfig, norm_eps: float,
                 dtype: jnp.dtype, rngs: nnx.Rngs, mesh: Mesh):
        vision_config = config.vision_config
        dim = vision_config.hidden_size
        norm_layer = partial(
            nnx.LayerNorm,
            epsilon=norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            bias_init=nnx.with_partitioning(init_fn, (None, )))

        self.norm1 = norm_layer(dim, dtype=dtype, rngs=rngs)
        self.norm2 = norm_layer(dim, dtype=dtype, rngs=rngs)
        self.attn = Qwen2_5_VisionAttention(config=config,
                                            dtype=dtype,
                                            rngs=rngs,
                                            mesh=mesh)
        self.mlp = Qwen3_VisionMLP(config=vision_config,
                                   dtype=dtype,
                                   rngs=rngs)

    def __call__(
            self,
            x: Float[Array, "seq_len 1 hidden_size"],
            rotary_pos_emb: Float[Array, "seq_len rot_dim"],
            cu_window_seqlens: Optional[Int[Array, "num_windows"]] = None,
            use_fullattn: bool = True
    ) -> Float[Array, "seq_len 1 hidden_size"]:

        x = x + self.attn(self.norm1(x), rotary_pos_emb, cu_window_seqlens,
                          use_fullattn)
        x = x + self.mlp(self.norm2(x))

        return x


class Qwen3_VisionPatchMerger(nnx.Module):

    def __init__(self, d_model: int, context_dim: int, norm_layer: Callable,
                 spatial_merge_size: int, use_postshuffle_norm: bool,
                 dtype: jnp.dtype, rngs: nnx.Rngs):
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm

        self.norm = norm_layer(
            self.hidden_size if use_postshuffle_norm else context_dim,
            param_dtype=dtype,
            dtype=dtype,
            rngs=rngs,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            bias_init=nnx.with_partitioning(init_fn, (None, )))
        self.mlp_fc1 = nnx.Linear(
            self.hidden_size,
            self.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model", )),
            rngs=rngs)
        self.mlp_act = modeling_flax_utils.ACT2FN["gelu"]
        self.mlp_fc2 = nnx.Linear(
            self.hidden_size,
            d_model,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rngs)

    def __call__(
        self, x: Float[Array, "seq_len hidden_size"]
    ) -> Float[Array, "token_len out_hidden_size_cat"]:
        if self.use_postshuffle_norm:
            x = self.norm(x.reshape(-1, self.hidden_size))
        else:
            x = self.norm(x).reshape(-1, self.hidden_size)
        x = self.mlp_fc1(x)
        x = self.mlp_act(x)
        x = self.mlp_fc2(x)
        return x


class Qwen3_VisionTransformer(nnx.Module):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rngs: nnx.Rngs,
                 mesh: Mesh,
                 norm_eps: float = 1e-6):
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        vision_config = hf_config.vision_config
        dtype = utils.to_jax_dtype(model_config.dtype)

        self.config = vision_config
        self.dtype = dtype

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads

        # args for sequence partitioning
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.deepstack_visual_indexes = getattr(vision_config,
                                                "deepstack_visual_indexes", [])

        self.patch_embed = Qwen3_VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size,
            dtype=dtype,
            rngs=rngs)

        # Absolute position embedding table
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)
        self.pos_embed = nnx.Param(
            jax.random.uniform(
                rngs.params(),
                (self.num_position_embeddings, self.hidden_size),
                dtype=dtype))

        head_dim = vision_config.hidden_size // vision_config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nnx.List([
            Qwen3_VisionBlock(
                config=hf_config,
                norm_eps=norm_eps,
                dtype=dtype,
                rngs=rngs,
                mesh=mesh,
            ) for _ in range(vision_config.depth)
        ])

        norm_layer = partial(nnx.LayerNorm, epsilon=norm_eps)

        self.merger = Qwen3_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=vision_config.hidden_size,
            norm_layer=norm_layer,
            spatial_merge_size=vision_config.spatial_merge_size,
            use_postshuffle_norm=False,
            dtype=dtype,
            rngs=rngs)

        self.deepstack_merger_list = nnx.List([
            Qwen3_VisionPatchMerger(
                d_model=vision_config.out_hidden_size,
                context_dim=vision_config.hidden_size,
                norm_layer=norm_layer,
                spatial_merge_size=vision_config.spatial_merge_size,
                use_postshuffle_norm=True,
                dtype=dtype,
                rngs=rngs) for _ in range(len(self.deepstack_visual_indexes))
        ])

    def compute_aux_arrays(
        self, grid_thw: tuple[tuple[int, int, int], ...]
    ) -> tuple[Int[Array, "seq_len"], Float[Array, "seq_len rot_dim"], Float[
            Array, "seq_len hidden_size"], Int[Array, "num_seqlens_plus_1"]]:
        """Computes grid/position auxiliary arrays (RoPE, pos_embeds, seqlens).

        OPTIMIZATION NOTE:
        Deliberately written in pure NumPy. Since this runs eagerly on CPU (not JIT'ed due to
        dynamic loop), using JAX eagerly would introduce heavy dispatch overhead and trigger
        dynamic compilation of JAX's internal helper ops (meshgrid, linspace, etc.) for different
        image sizes. NumPy executes natively in C, avoiding all JAX CPU eager/compilation bottlenecks.
        Converted back to JAX array once at the final eagerly executed boundary.
        """

        def get_rope_by_thw_np(t: int, h: int, w: int):
            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size
            window_index_thw = np.arange(t * llm_h * llm_w)

            # rotary_pos_emb_thw
            hpos_ids, wpos_ids = np.indices((h, w))
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).transpose(0, 2, 1, 3).flatten()
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).transpose(0, 2, 1, 3).flatten()
            pos_ids = np.stack([hpos_ids, wpos_ids], axis=-1)
            pos_ids = np.tile(pos_ids, (t, 1))

            max_size = max(h, w)
            # rotary_pos_emb_full
            inv_freq = 1.0 / (self.rotary_pos_emb.theta**(
                np.arange(0, self.rotary_pos_emb.dim, 2, dtype=np.float32) /
                self.rotary_pos_emb.dim))
            seq = np.arange(max_size, dtype=np.float32)
            rotary_pos_emb_full = np.outer(seq, inv_freq)

            rotary_pos_emb_thw = rotary_pos_emb_full[pos_ids].reshape(
                pos_ids.shape[0], -1)
            rotary_pos_emb_thw = rotary_pos_emb_thw.reshape(
                rotary_pos_emb_thw.shape[0] // self.spatial_merge_unit,
                self.spatial_merge_unit, -1)

            rotary_pos_emb_thw = rotary_pos_emb_thw[window_index_thw, :, :]
            rotary_pos_emb_thw = rotary_pos_emb_thw.reshape(
                -1, rotary_pos_emb_thw.shape[-1])
            cu_seqlens_thw = np.full(t, h * w, dtype=np.int32)

            return rotary_pos_emb_thw, window_index_thw, cu_seqlens_thw

        def pos_embed_interpolate_np(t: int, h: int, w: int):
            embed_weight = np.array(self.pos_embed[...])
            hidden_dim = embed_weight.shape[1]
            m_size = self.spatial_merge_size
            num_grid_per_side = self.num_grid_per_side

            h_idxs = np.linspace(0, num_grid_per_side - 1, h)
            w_idxs = np.linspace(0, num_grid_per_side - 1, w)

            h_floor = h_idxs.astype(np.int32)
            w_floor = w_idxs.astype(np.int32)
            h_ceil = np.clip(h_floor + 1, 0, num_grid_per_side - 1)
            w_ceil = np.clip(w_floor + 1, 0, num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            dh_grid, dw_grid = np.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = np.meshgrid(h_floor,
                                                     w_floor,
                                                     indexing="ij")
            h_ceil_grid, w_ceil_grid = np.meshgrid(h_ceil,
                                                   w_ceil,
                                                   indexing="ij")

            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1.0 - dh_grid - w01

            h_grid = np.stack(
                [h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
            w_grid = np.stack(
                [w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
            h_grid_idx = h_grid * num_grid_per_side

            indices = (h_grid_idx + w_grid).reshape(4, -1)
            weights = np.stack([w00, w01, w10, w11], axis=0).reshape(4, -1, 1)

            embeds = embed_weight[indices]
            embeds *= weights
            combined = embeds.sum(axis=0)

            combined = combined.reshape(h // m_size, m_size, w // m_size,
                                        m_size, hidden_dim)
            combined = np.transpose(combined,
                                    (0, 2, 1, 3, 4)).reshape(-1, hidden_dim)
            repeated = np.tile(combined, (t, 1))

            return repeated

        num_grids = len(grid_thw)

        rotary_pos_emb = []
        pos_embeds = []
        window_index = []
        cu_seqlens = []

        window_index_id = 0
        for i in range(num_grids):
            t, h, w = grid_thw[i]
            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size

            # 1. get_rope_by_thw
            rotary_pos_emb_thw, window_index_thw, cu_seqlens_thw = get_rope_by_thw_np(
                t, h, w)

            # 2. pos_embed_interpolate
            repeated = pos_embed_interpolate_np(t, h, w)

            # Append outputs
            window_index.append(window_index_thw + window_index_id)
            window_index_id += (t * llm_h * llm_w)

            rotary_pos_emb.append(rotary_pos_emb_thw)
            pos_embeds.append(repeated)
            cu_seqlens.append(cu_seqlens_thw)

        rotary_pos_emb = np.concatenate(rotary_pos_emb, axis=0)
        pos_embeds = np.concatenate(pos_embeds, axis=0)
        window_index = np.concatenate(window_index, axis=0)

        cu_seqlens = np.concatenate(cu_seqlens, axis=0)
        cu_seqlens = np.cumsum(cu_seqlens, axis=0, dtype=np.int32)
        cu_seqlens = np.pad(cu_seqlens, ((1, 0), ),
                            mode='constant',
                            constant_values=0)

        # Pad arrays using NumPy on Host before converting to JAX Array
        num_patches = rotary_pos_emb.shape[0]
        bucket_num_patches = 1 << (num_patches - 1).bit_length()
        num_tokens = window_index.shape[0]
        bucket_num_tokens = bucket_num_patches // self.spatial_merge_unit

        rotary_pos_emb_padded = np.pad(rotary_pos_emb,
                                       ((0, bucket_num_patches - num_patches),
                                        (0, 0)))
        pos_embeds_padded = np.pad(pos_embeds,
                                   ((0, bucket_num_patches - num_patches),
                                    (0, 0)))
        window_index_padded = np.concatenate([
            window_index,
            np.arange(num_tokens, bucket_num_tokens, dtype=np.int32)
        ])

        L = cu_seqlens.shape[0]
        bucket_num_seqlens = 1 << (L - 1 - 1).bit_length()
        target_len = bucket_num_seqlens + 1
        pad_size = target_len - L
        cu_seqlens_padded = np.pad(cu_seqlens, (0, pad_size), mode='edge')
        cu_seqlens_padded[-1] = bucket_num_patches

        # Convert to JAX Array once at the final eagerly executed boundary with static shapes
        return (jnp.array(window_index_padded),
                jnp.array(rotary_pos_emb_padded, dtype=jnp.bfloat16),
                jnp.array(pos_embeds_padded, dtype=self.pos_embed.dtype),
                jnp.array(cu_seqlens_padded))

    def compute_hidden_states(
        self, x: Float[Array, "seq_len hidden_size"],
        window_index: Int[Array,
                          "seq_len"], rotary_pos_emb: Float[Array,
                                                            "seq_len rot_dim"],
        pos_embeds: Float[Array, "seq_len hidden_size"],
        cu_seqlens: Int[Array, "num_seqlens"]
    ) -> Float[Array, "token_len out_hidden_size_cat"]:
        hidden_states = self.patch_embed(x)

        # Add absolute positional embeddings exactly like PyTorch!
        hidden_states = hidden_states + pos_embeds

        seq_len = x.shape[0]

        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        hidden_states = jnp.expand_dims(hidden_states, axis=1)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            # All Qwen3-VL blocks execute full attention
            hidden_states = blk(hidden_states,
                                rotary_pos_emb=rotary_pos_emb,
                                cu_window_seqlens=cu_seqlens,
                                use_fullattn=True)

            # Qwen3-VL DeepStack implementation
            if layer_num in self.deepstack_visual_indexes:
                deepstack_merger_idx = self.deepstack_visual_indexes.index(
                    layer_num)
                squeezed = jnp.squeeze(hidden_states, axis=1)
                deepstack_feature = self.deepstack_merger_list[
                    deepstack_merger_idx](squeezed)
                deepstack_feature_lists.append(deepstack_feature)

        # Final layer merger
        squeezed = jnp.squeeze(hidden_states, axis=1)
        hidden_states = self.merger(squeezed)

        # Concatenate DeepStack features along channel dimension
        if len(deepstack_feature_lists) > 0:
            hidden_states = jnp.concatenate([hidden_states] +
                                            deepstack_feature_lists,
                                            axis=-1)

        # Reorder back from window attention order to original sequence order
        reverse_indices = jnp.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states

    @jax.jit
    def encode_padded_jit(
        self, x_padded: Float[Array, "padded_patch_len hidden_size"],
        window_index: Int[Array, "padded_token_len"],
        rotary_pos_emb: Float[Array, "padded_patch_len rot_dim"],
        pos_embeds: Float[Array, "padded_patch_len hidden_size"],
        cu_seqlens: Int[Array, "padded_seqlens"]
    ) -> Float[Array, "padded_token_len out_hidden_size_cat"]:
        return self.compute_hidden_states(x_padded, window_index,
                                          rotary_pos_emb, pos_embeds,
                                          cu_seqlens)

    def __call__(
        self, x_padded: Float[Array, "padded_patch_len hidden_size"],
        grid_thw: tuple[tuple[int, int, int], ...]
    ) -> Float[Array, "padded_token_len out_hidden_size_cat"]:
        window_index, rotary_pos_emb, pos_embeds, cu_seqlens = self.compute_aux_arrays(
            grid_thw)

        hidden_states = self.encode_padded_jit(x_padded, window_index,
                                               rotary_pos_emb, pos_embeds,
                                               cu_seqlens)
        return hidden_states


def copy_weights_to_jax_vision_tower(
    torch_visual: torch.nn.Module,
    jax_visual: Qwen3_VisionTransformer,
) -> None:
    """Safely aligns and copies weights from PyTorch visual tower to JAX visual tower using a clean mapping structure."""
    from jax.sharding import NamedSharding

    from tpu_inference.models.jax.utils.weight_utils import shard_put

    # 1. Declarative weight projection dictionary with custom tensor transformation lambdas
    mappings = {
        "patch_embed.proj.weight":
        ("patch_embed.proj.kernel", lambda w: w.transpose(2, 3, 4, 1, 0)),
        "patch_embed.proj.bias": ("patch_embed.proj.bias", None),
        "pos_embed.weight": ("pos_embed", None),
        "merger.norm.weight": ("merger.norm.scale", None),
        "merger.norm.bias": ("merger.norm.bias", None),
        "merger.linear_fc1.weight": ("merger.mlp_fc1.kernel", lambda w: w.T),
        "merger.linear_fc1.bias": ("merger.mlp_fc1.bias", None),
        "merger.linear_fc2.weight": ("merger.mlp_fc2.kernel", lambda w: w.T),
        "merger.linear_fc2.bias": ("merger.mlp_fc2.bias", None),
    }

    # 2. Dynamically register Deepstack layers
    for idx in range(len(jax_visual.deepstack_visual_indexes)):
        mappings[f"deepstack_merger_list.{idx}.norm.weight"] = (
            f"deepstack_merger_list.{idx}.norm.scale", None)
        mappings[f"deepstack_merger_list.{idx}.norm.bias"] = (
            f"deepstack_merger_list.{idx}.norm.bias", None)
        mappings[f"deepstack_merger_list.{idx}.linear_fc1.weight"] = (
            f"deepstack_merger_list.{idx}.mlp_fc1.kernel", lambda w: w.T)
        mappings[f"deepstack_merger_list.{idx}.linear_fc1.bias"] = (
            f"deepstack_merger_list.{idx}.mlp_fc1.bias", None)
        mappings[f"deepstack_merger_list.{idx}.linear_fc2.weight"] = (
            f"deepstack_merger_list.{idx}.mlp_fc2.kernel", lambda w: w.T)
        mappings[f"deepstack_merger_list.{idx}.linear_fc2.bias"] = (
            f"deepstack_merger_list.{idx}.mlp_fc2.bias", None)

    # 3. Dynamically register Block layers (Attn & MLP)
    for idx in range(len(jax_visual.blocks)):
        pfx = f"blocks.{idx}"
        mappings[f"{pfx}.norm1.weight"] = (f"{pfx}.norm1.scale", None)
        mappings[f"{pfx}.norm1.bias"] = (f"{pfx}.norm1.bias", None)
        mappings[f"{pfx}.norm2.weight"] = (f"{pfx}.norm2.scale", None)
        mappings[f"{pfx}.norm2.bias"] = (f"{pfx}.norm2.bias", None)

        mappings[f"{pfx}.attn.qkv.weight"] = (f"{pfx}.attn.qkv_proj.kernel",
                                              lambda w: w.T)
        mappings[f"{pfx}.attn.qkv.bias"] = (f"{pfx}.attn.qkv_proj.bias", None)
        mappings[f"{pfx}.attn.proj.weight"] = (f"{pfx}.attn.proj.kernel",
                                               lambda w: w.T)
        mappings[f"{pfx}.attn.proj.bias"] = (f"{pfx}.attn.proj.bias", None)

        mappings[f"{pfx}.mlp.linear_fc1.weight"] = (
            f"{pfx}.mlp.linear_fc1.kernel", lambda w: w.T)
        mappings[f"{pfx}.mlp.linear_fc1.bias"] = (f"{pfx}.mlp.linear_fc1.bias",
                                                  None)
        mappings[f"{pfx}.mlp.linear_fc2.weight"] = (
            f"{pfx}.mlp.linear_fc2.kernel", lambda w: w.T)
        mappings[f"{pfx}.mlp.linear_fc2.bias"] = (f"{pfx}.mlp.linear_fc2.bias",
                                                  None)

    # 4. Get the JAX model's state representation
    jax_state = nnx.state(jax_visual)

    # 5. Run dynamic transfer mapping loop
    for t_name, t_param in torch_visual.named_parameters():
        if t_name in mappings:
            j_name, transform_fn = mappings[t_name]
            import torchax
            with torchax.default_env():
                t_val = t_param.detach().to(torch.float32).cpu().numpy()
            if transform_fn is not None:
                t_val = transform_fn(t_val)
            j_val = jnp.array(t_val)

            # Traverse JAX state dict to find target JAX Param
            keys = j_name.split(".")
            jax_param = jax_state
            for key in keys:
                if key.isdigit():
                    jax_param = jax_param[int(key)]
                else:
                    jax_param = jax_param[key]

            # Deduce sharding specs and assign safely
            sharding_spec = jax_param.get_metadata().get("sharding", ())
            if isinstance(sharding_spec, NamedSharding):
                sharding_spec = sharding_spec.spec

            # Copy sharded parameters into target slot
            jax_param[...] = shard_put(j_val, sharding_spec)

    # 6. Warm update the state in the live JAX model
    nnx.update(jax_visual, jax_state)
