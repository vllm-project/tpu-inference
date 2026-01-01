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

import jax
import torch
import torch.nn.functional as F

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
        mesh_shape = (1, attn_dp_size, 1, model_size)
        return jax.make_mesh(mesh_shape, axis_names, devices=devices)
    else:
        axis_names = MESH_AXIS_NAMES_2D
        mesh_shape = (1, len(devices))
        return jax.make_mesh(mesh_shape, axis_names, devices=devices)


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
def ref_moe(x, router_logits, w1, w2, w1_bias, w2_bias, top_k, renormalize,
            activation):

    expert_weights = F.softmax(router_logits, dim=-1)
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
