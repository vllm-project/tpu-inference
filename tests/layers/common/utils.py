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
from typing import Optional

import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F
from jax.sharding import AxisType

from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  MESH_AXIS_NAMES_2D)


def get_spmd_mesh(num_devices: int = 1, enable_attn_dp: bool = False):
    devices = sorted(jax.devices(), key=lambda d: d.id)[0:num_devices]

    if enable_attn_dp:
        if num_devices < 2:
            raise ValueError(
                f"enable_attn_dp requires at least 2 devices, got {num_devices}"
            )
        axis_names = MESH_AXIS_NAMES
        attn_dp_size = 2
        model_size = num_devices // attn_dp_size
        mesh_shape = (1, attn_dp_size, 1, 1, model_size)
        return jax.make_mesh(mesh_shape,
                             axis_names,
                             axis_types=(AxisType.Auto, ) * len(axis_names),
                             devices=devices)
    else:
        axis_names = MESH_AXIS_NAMES_2D
        mesh_shape = (1, len(devices))
        return jax.make_mesh(mesh_shape,
                             axis_names,
                             axis_types=(AxisType.Auto, ) * len(axis_names),
                             devices=devices)


def find_all_layer_type(module: torch.nn.Module, layer_type: torch.nn.Module):
    ret = []
    for name, child in module.named_children():
        if isinstance(child, layer_type):
            ret.append(child)
        else:
            ret.extend(find_all_layer_type(child, layer_type))
    return ret


# TODO(kyuyeunk): Consolidate all reference implementation used for unit tests
# into a single file.
def ref_moe(x: torch.Tensor,
            router_logits: torch.Tensor,
            w1: torch.Tensor,
            w2: torch.Tensor,
            w1_bias: Optional[torch.Tensor],
            w2_bias: Optional[torch.Tensor],
            top_k: int,
            renormalize: bool,
            activation: str,
            scoring_func: str = "softmax") -> torch.Tensor:
    """
    Reference implementation of MoE forward pass in Torch.

    Args:
        x: Input tensor of shape [tokens, dim].
        router_logits: Tensor of shape [tokens, dim].
        w1: Tensor of shape [num_experts, intermediate_size * 2, dim].
        w2: Tensor of shape [num_experts, dim, intermediate_size].
        w1_bias: Tensor of shape [num_experts, intermediate_size * 2]
        w2_bias: Tensor of shape [num_experts, dim]
        top_k: Number of top-k experts to use.
        renormalize: Whether to renormalize the expert weights.
        activation: Activation function to use.
        scoring_func: Scoring function to use.

    Returns:
        Output tensor of shape [tokens, dim].
    """

    match scoring_func:
        case "softmax":
            expert_weights = F.softmax(router_logits, dim=-1)
        case "sigmoid":
            expert_weights = F.sigmoid(router_logits)
        case _:
            raise NotImplementedError(
                f"No reference implementation for {scoring_func} scoring")

    expert_weights, expert_indices = torch.topk(expert_weights, top_k, dim=-1)
    if renormalize:
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True)

    x = torch.einsum("ti,eoi->teo", x, w1)
    if w1_bias is not None:
        x += w1_bias.unsqueeze(0)

    match activation:
        case "silu":
            x1, x3 = x.chunk(chunks=2, dim=-1)
            x = F.silu(x1) * x3
        case "swigluoai":
            x1, x3 = x[..., ::2], x[..., 1::2]
            x1 = x1.clamp(min=None, max=7.0)
            x3 = x3.clamp(min=-7.0, max=7.0)
            gated_activation = x1 * torch.sigmoid(x1 * 1.702)
            x = gated_activation * (x3 + 1)
        case _:
            raise NotImplementedError(
                f"No reference implementation for {activation} activation")

    x = torch.einsum("teo,eio->tei", x, w2)
    if w2_bias is not None:
        x += w2_bias.unsqueeze(0)

    seq_indexes = torch.arange(x.shape[0]).unsqueeze(1)
    x = x[seq_indexes, expert_indices]

    return torch.einsum("tai,ta->ti", x, expert_weights)


def ref_moe_jax(x: jax.Array,
                router_logits: jax.Array,
                w1: jax.Array,
                w2: jax.Array,
                w1_bias: Optional[jax.Array],
                w2_bias: Optional[jax.Array],
                top_k: int,
                renormalize: bool,
                activation: str,
                scoring_func: str = "softmax") -> jax.Array:
    """
    Reference implementation of MoE forward pass in JAX.

    Args:
        x: Input array of shape [tokens, dim].
        router_logits: array of shape [tokens, dim].
        w1: array of shape [num_experts, intermediate_size * 2, dim].
        w2: array of shape [num_experts, dim, intermediate_size].
        w1_bias: array of shape [num_experts, intermediate_size * 2]
        w2_bias: array of shape [num_experts, dim]
        top_k: Number of top-k experts to use.
        renormalize: Whether to renormalize the expert weights.
        activation: Activation function to use.
        scoring_func: Scoring function to use.

    Returns:
        Output array of shape [tokens, dim].
    """

    match scoring_func:
        case "softmax":
            expert_weights = jax.nn.softmax(router_logits, axis=-1)
        case "sigmoid":
            expert_weights = jax.nn.sigmoid(router_logits)
        case _:
            raise NotImplementedError(
                f"No reference implementation for {scoring_func} scoring")

    expert_weights, expert_indices = jax.lax.top_k(expert_weights, top_k)
    if renormalize:
        expert_weights /= expert_weights.sum(axis=-1, keepdims=True)

    x = jnp.einsum("ti,eoi->teo", x, w1)
    if w1_bias is not None:
        x += w1_bias[None, ...]

    match activation:
        case "silu":
            x1, x3 = jnp.split(x, 2, axis=-1)
            x = jax.nn.silu(x1) * x3
        case "swigluoai":
            x1, x3 = x[..., ::2], x[..., 1::2]
            x1 = jnp.minimum(x1, 7.0)
            x3 = jnp.clip(x3, -7.0, 7.0)
            gated_activation = x1 * jax.nn.sigmoid(x1 * 1.702)
            x = gated_activation * (x3 + 1)
        case _:
            raise NotImplementedError(
                f"No reference implementation for {activation} activation")

    x = jnp.einsum("teo,eio->tei", x, w2)
    if w2_bias is not None:
        x += w2_bias[None, ...]

    x = jnp.take_along_axis(x, expert_indices[..., None], axis=1)

    return jnp.einsum("tai,ta->ti", x, expert_weights)
