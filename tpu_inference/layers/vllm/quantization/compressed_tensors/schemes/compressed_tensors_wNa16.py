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

from collections.abc import Callable
from typing import Optional
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import torch
from compressed_tensors.quantization import ActivationOrdering
from jax.sharding import PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,)
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter,
                                           RowvLLMParameter)

from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.layers.common.quantization import ct_u32_unpack_u4
from tpu_inference.layers.common.utils import \
    slice_sharded_tensor_for_concatenation
from tpu_inference.layers.vllm.quantization.configs import \
    VllmQuantLinearConfig
from tpu_inference.logger import init_logger
from tpu_inference.utils import t2j

P = PartitionSpec
logger = init_logger(__name__)

WNA16_SUPPORTED_BITS = [4, 8]

def _unpack_int4_to_bf16_pallas(w_packed_int32):
    """Native bitwise unpacking inside TPU SRAM."""
    shifts = jax.lax.iota(jnp.uint32, 8) * 4
    w_expanded = jnp.expand_dims(w_packed_int32.astype(jnp.uint32), axis=1)
    shifts_expanded = jnp.expand_dims(jnp.expand_dims(shifts, axis=0), axis=-1)

    w_unpacked = (w_expanded >> shifts_expanded) & 0xF
    w_unpacked = w_unpacked.reshape(-1, w_packed_int32.shape[-1])

    return w_unpacked.astype(jnp.bfloat16) - 8.0

def pallas_wna16_fused_matmul_kernel(
    x_ref, w_packed_ref, w_scale_ref, out_ref,
    *, BM, BK, BN, group_size, pack_factor
):
    # x_ref is passed in as [BM, K]
    # w_packed_ref is passed in as [K // pack_factor, BN]
    # w_scale_ref is passed in as [scale_K, BN]
    
    acc = jnp.zeros((BM, BN), dtype=jnp.float32)
    num_k_blocks = x_ref.shape[1] // BK
    packed_BK = BK // pack_factor

    def body_fn(k_idx, acc_carry):
        # We only need to slide along the K dimension inside the kernel
        x_tile = x_ref[..., pl.ds(k_idx * BK, BK)]

        w_packed_tile = w_packed_ref[
            pl.ds(k_idx * packed_BK, packed_BK), ...]

        if group_size == -1:
            # Channel-wise scale: shape is [1, BN], load it entirely
            w_scale_tile = w_scale_ref[...]
        else:
            # Group-wise scale: Load the packed block and repeat it
            w_scale_block = w_scale_ref[
                pl.ds(k_idx * packed_BK, packed_BK), ...]
            w_scale_tile = jnp.repeat(w_scale_block, pack_factor, axis=0)

        w_bf16_unpacked = _unpack_int4_to_bf16_pallas(w_packed_tile)
        w_bf16 = w_bf16_unpacked * w_scale_tile

        acc_carry += jax.lax.dot(
            x_tile.astype(jnp.bfloat16), w_bf16,
            preferred_element_type=jnp.float32)
        return acc_carry

    acc = jax.lax.fori_loop(0, num_k_blocks, body_fn, acc)

    out_ref[...] = acc.astype(jnp.bfloat16)

@functools.partial(jax.jit, static_argnames=['BM', 'BK', 'BN', 'group_size', 'pack_factor'])
def apply_pallas_fused_linear(x, w_packed, w_scale, BM=32, BK=128, BN=128, group_size=-1, pack_factor=8):
    """Wrapper using BlockSpec to strictly control TPU local memory allocations."""
    M, K = x.shape
    N = w_packed.shape[1]
    
    pad_m = (BM - (M % BM)) % BM
    pad_k = (BK - (K % BK)) % BK
    pad_n = (BN - (N % BN)) % BN
    
    # 1. Pad Activations
    if pad_m > 0 or pad_k > 0:
        x = jnp.pad(x, ((0, pad_m), (0, pad_k)))
        
    # 2. Pad Packed Weights
    pad_k_packed = pad_k // pack_factor
    if pad_k_packed > 0 or pad_n > 0:
        w_packed = jnp.pad(w_packed, ((0, pad_k_packed), (0, pad_n)))
        
    # 3. Expand and Pad Scales
    if group_size != -1:
        repeat_factor = group_size // pack_factor
        w_scale = jnp.repeat(w_scale, repeat_factor, axis=0)
        if pad_k_packed > 0 or pad_n > 0:
            w_scale = jnp.pad(w_scale, ((0, pad_k_packed), (0, pad_n)))
        scale_k_dim = w_packed.shape[0]
    else:
        if pad_n > 0:
            w_scale = jnp.pad(w_scale, ((0, 0), (0, pad_n)))
        scale_k_dim = 1

    padded_M = x.shape[0]
    padded_K = x.shape[1]
    padded_N = w_packed.shape[1]
    
    grid = (padded_M // BM, padded_N // BN)

    # Use BlockSpec to map the (M, N) grid coordinates to the exact HBM slices
    out = pl.pallas_call(
        functools.partial(
            pallas_wna16_fused_matmul_kernel,
            BM=BM, BK=BK, BN=BN, group_size=group_size, pack_factor=pack_factor
        ),
        out_shape=jax.ShapeDtypeStruct((padded_M, padded_N), jnp.bfloat16),
        in_specs=[
            # x_ref gets [BM, padded_K]
            pl.BlockSpec(block_shape=(BM, padded_K), index_map=lambda i, j: (i, 0)),
            # w_packed_ref gets [padded_K // pack_factor, BN]
            pl.BlockSpec(block_shape=(padded_K // pack_factor, BN), index_map=lambda i, j: (0, j)),
            # w_scale_ref gets [scale_K, BN]
            pl.BlockSpec(block_shape=(scale_k_dim, BN), index_map=lambda i, j: (0, j)),
        ],
        # out_ref maps to [BM, BN]
        out_specs=pl.BlockSpec(block_shape=(BM, BN), index_map=lambda i, j: (i, j)),
        grid=grid
    )(x, w_packed, w_scale)
    
    return out[:M, :N]

def _torch_unpack_from_int32(packed: torch.Tensor,
                             num_bits: int) -> torch.Tensor:
    """Unpack int32 values into individual n-bit values stored as uint8.

    Each int32 contains (32 // num_bits) packed values.  The least-significant
    bits correspond to the first logical element in the packed dimension.
    """
    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    shifts = torch.arange(0, 32, num_bits, dtype=torch.int32,
                          device=packed.device)
    unpacked = torch.bitwise_and(
        torch.bitwise_right_shift(packed.unsqueeze(-1), shifts), mask)
    out_shape = list(packed.shape)
    out_shape[-1] *= pack_factor
    return unpacked.reshape(out_shape).to(torch.uint8)


def _torch_repack_to_int32(unpacked: torch.Tensor,
                           num_bits: int) -> torch.Tensor:
    """Repack individual n-bit values (uint8) back into int32 containers.

    Inverse of ``_torch_unpack_from_int32``.  Uses bitwise OR and left-shift
    so no precision is lost.
    """
    pack_factor = 32 // num_bits
    *leading, last = unpacked.shape
    assert last % pack_factor == 0, (
        f"Last dim {last} not divisible by pack_factor {pack_factor}")
    grouped = unpacked.reshape(*leading, last // pack_factor,
                               pack_factor).to(torch.int32)
    shifts = torch.arange(0, 32, num_bits, dtype=torch.int32,
                          device=unpacked.device)
    packed = torch.zeros(*leading, last // pack_factor,
                         dtype=torch.int32, device=unpacked.device)
    for i in range(pack_factor):
        packed = torch.bitwise_or(
            packed,
            torch.bitwise_left_shift(grouped[..., i], shifts[i]))
    return packed

class VllmCompressedTensorsWNA16(CompressedTensorsScheme):

    def __init__(
        self,
        strategy: str,
        num_bits: int,
        linear_config: VllmQuantLinearConfig,
        group_size: int | None = None,
        symmetric: bool = True,
        actorder: ActivationOrdering | None = None,
    ):
        self.pack_factor = 32 // num_bits
        self.num_bits = num_bits
        self.strategy = strategy
        self.symmetric = symmetric
        self.group_size = -1 if group_size is None else group_size
        self.has_g_idx = actorder == ActivationOrdering.GROUP
        self.linear_config = linear_config

        if num_bits not in WNA16_SUPPORTED_BITS:
            raise ValueError(
                f"Unsupported num_bits = {num_bits}. "
                f"Supported num_bits = {WNA16_SUPPORTED_BITS}")

        if self.group_size == -1 and self.strategy != "channel":
            raise ValueError(
                "WNA16 requires group quantization or channelwise "
                "quantization, but found no group size and strategy "
                "is not channelwise.")

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        output_size: int,
        input_size: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        group_size = self.group_size if self.group_size != -1 else input_size
        row_parallel = input_size != input_size_per_partition
        partition_scales = row_parallel and self.group_size != -1

        scales_and_zp_size = input_size // group_size
        if partition_scales:
            assert input_size_per_partition % group_size == 0
            scales_and_zp_size = input_size_per_partition // group_size

        weight = PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            packed_factor=self.pack_factor,
            packed_dim=1,
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
        )
        layer.register_parameter("weight_packed", weight)

        weight_scale_args = {
            "weight_loader": weight_loader,
            "data": torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            ),
        }
        if not partition_scales:
            weight_scale = ChannelQuantScaleParameter(
                output_dim=0, **weight_scale_args)
        else:
            weight_scale = GroupQuantScaleParameter(
                output_dim=0, input_dim=1, **weight_scale_args)
        layer.register_parameter("weight_scale", weight_scale)

        weight_shape = BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_shape", weight_shape)

        if not self.symmetric:
            zeros_args = {
                "weight_loader": weight_loader,
                "data": torch.zeros(
                    output_size_per_partition // self.pack_factor,
                    scales_and_zp_size,
                    dtype=torch.int32,
                ),
            }
            if not partition_scales:
                qzeros = PackedColumnParameter(
                    output_dim=0,
                    packed_dim=0,
                    packed_factor=self.pack_factor,
                    **zeros_args,
                )
            else:
                qzeros = PackedvLLMParameter(
                    input_dim=1,
                    output_dim=0,
                    packed_dim=0,
                    packed_factor=self.pack_factor,
                    **zeros_args,
                )
            layer.register_parameter("weight_zero_point", qzeros)

        if self.has_g_idx:
            weight_g_idx = RowvLLMParameter(
                data=torch.empty(input_size_per_partition, dtype=torch.int32),
                input_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_g_idx", weight_g_idx)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # ----- 1. Pre-shuffle actorder weights (Unpack-Shuffle-Repack) -----
        if hasattr(layer, "weight_g_idx"):
            g_idx = layer.weight_g_idx.data
            if (g_idx >= 0).all():
                input_size = g_idx.shape[0]
                effective_gs = (input_size if self.group_size == -1
                                else self.group_size)
                expected = (torch.arange(input_size, dtype=g_idx.dtype,
                                         device=g_idx.device)
                            // effective_gs)

                if not torch.equal(g_idx, expected):
                    q_perm = torch.argsort(g_idx)

                    # Unpack packed int32 → individual n-bit values (uint8)
                    unpacked = _torch_unpack_from_int32(
                        layer.weight_packed.data, self.num_bits)
                    # Shuffle columns so column k ∈ group k//group_size
                    unpacked = unpacked[:, q_perm]
                    # Repack back into int32 containers
                    repacked = _torch_repack_to_int32(unpacked, self.num_bits)
                    layer.weight_packed.data.copy_(repacked)
                    del unpacked, repacked

                    # Store q_perm so apply_weights can permute activations
                    # to match the shuffled column order.
                    layer.register_buffer(
                        "q_perm", q_perm.to(torch.int32))


            delattr(layer, "weight_g_idx")
            self.has_g_idx = False

        if hasattr(layer, "weight_shape"):
            delattr(layer, "weight_shape")

        # ----- 2. Convert to JAX arrays -----
        w_packed = t2j(layer.weight_packed, use_dlpack=False)
        w_scale = t2j(layer.weight_scale, use_dlpack=False)
        delattr(layer, "weight_packed")
        delattr(layer, "weight_scale")

        bias = None
        if layer.bias is not None and not layer.skip_bias_add:
            bias = t2j(layer.bias, use_dlpack=False)
            delattr(layer, "bias")

        # ----- 3. Shard across the TPU mesh -----
        @jax.jit
        def shard_and_prepare(weight, scale, bias):
            lin_weights = LinearWeights(
                weight=weight,
                weight_scale=scale,
                zero_point=None,
                bias=bias,
            )
            return shard_linear_weights(
                lin_weights,
                mesh=self.linear_config.mesh,
                weight_p_spec=self.linear_config.weight_sharding,
                bias_p_spec=self.linear_config.bias_sharding,
            )

        sharded_weights = shard_and_prepare(w_packed, w_scale, bias)

        # ----- 4. Re-register as PyTorch Parameters -----
        if self.linear_config.fuse_matmuls:
            layer.weight_packed = Parameter(
                torch_view(sharded_weights.weight), requires_grad=False)
            layer.weight_scale = Parameter(
                torch_view(sharded_weights.weight_scale),
                requires_grad=False)
            if bias is not None:
                layer.bias = Parameter(
                    torch_view(sharded_weights.bias), requires_grad=False)
        else:
            layer.weight_packed = to_parameter_list(
                torch_view(sharded_weights.weight))
            layer.weight_scale = to_parameter_list(
                torch_view(sharded_weights.weight_scale))
            if bias is not None:
                layer.bias = to_parameter_list(
                    torch_view(sharded_weights.bias))

    def _dequantize_on_the_fly_fallback(self, layer: torch.nn.Module) -> jax.Array:
        """Standard JAX fallback for when Pallas cannot be used (e.g. actorder)."""
        w_packed = jax_view(layer.weight_packed)
        w_scale = jax_view(layer.weight_scale)
        
        weight = ct_u32_unpack_u4(w_packed)

        input_size = weight.shape[1]
        effective_group_size = input_size if self.group_size == -1 else self.group_size
        num_groups = input_size // effective_group_size
        weight = weight.reshape((weight.shape[0], num_groups, effective_group_size))

        scales = jnp.expand_dims(w_scale, -1)
        weight_f = weight.astype(jnp.bfloat16)

        if not self.symmetric and hasattr(layer, "weight_zero_point"):
            zp_packed = jax_view(layer.weight_zero_point)
            zero_point = ct_u32_unpack_u4(zp_packed)
            zp = jnp.expand_dims(zero_point.astype(jnp.bfloat16), -1)
            weight_deq = (weight_f - zp) * scales
        else:
            offset = jnp.array(1 << (self.num_bits - 1), dtype=jnp.bfloat16)
            weight_deq = (weight_f - offset) * scales

        weight_deq = weight_deq.reshape((weight_deq.shape[0], -1))

        return weight_deq

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        with jax.named_scope(layer._get_name()):
            if self.linear_config.fuse_matmuls:
                out = self._apply_fused(layer, x, bias)
            else:
                out = self._apply_split(layer, x, bias)
        return out

    def _apply_fused(self, layer: torch.nn.Module, x: torch.Tensor,
                     bias: Optional[torch.Tensor]) -> torch.Tensor:
        x_jax = jax_view(x)
        bias_jax = jax_view(bias) if bias is not None else None

        # If weights were pre-shuffled (actorder), permute activations to
        # match the new column order so the matmul remains correct.
        if hasattr(layer, 'q_perm'):
            x_jax = x_jax[:, jax_view(layer.q_perm)]

        # --- PALLAS KERNEL PATH ---
        if (not self.has_g_idx
            and self.symmetric
            and self.num_bits == 4):
            
            w_packed = jax_view(layer.weight_packed)
            # Transpose packed weights because PyTorch linear loads them as [Out, In], 
            # but matmul expects [In, Out].
            w_packed_t = w_packed.T
            w_scale = jax_view(layer.weight_scale).T
            
            # Execute ultra-fast local memory kernel
            outs = apply_pallas_fused_linear(
                x_jax, w_packed_t, w_scale,
                BM=16, BK=128, BN=128,
                group_size=self.group_size,
                pack_factor=self.pack_factor
            )
            
            if bias_jax is not None and not layer.skip_bias_add:
                outs += bias_jax

        # --- JAX FALLBACK PATH ---
        else:
            weight_deq = self._dequantize_on_the_fly_fallback(layer)
            lin_weights = process_linear_weights(
                LinearWeights(
                    weight=weight_deq,
                    weight_scale=None,
                    zero_point=None,
                    bias=bias_jax,
                ),
                fused=True,
                output_sizes=self.linear_config.output_sizes,
                reorder_size=self.linear_config.n_shards,
            )
            outs = jnp.einsum("bd,fd->bf", x_jax, lin_weights.weight)
            if lin_weights.bias is not None and not layer.skip_bias_add:
                outs += lin_weights.bias

        # Handle post-processing (TPU sharding/concatenation)
        outs = slice_sharded_tensor_for_concatenation(
            outs, self.linear_config.output_sizes, self.linear_config.n_shards)
        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)

    def _apply_split(self, layer: torch.nn.Module, x: torch.Tensor,
                     bias: Optional[torch.Tensor]) -> torch.Tensor:
        x_jax = jax_view(x)
        bias_jax = jax_view(bias) if bias is not None else None

        if hasattr(layer, 'q_perm'):
            x_jax = x_jax[:, jax_view(layer.q_perm)]

        weight_deq = self._dequantize_on_the_fly_fallback(layer)

        lin_weights = process_linear_weights(
            LinearWeights(
                weight=weight_deq,
                weight_scale=None,
                zero_point=None,
                bias=bias_jax,
            ),
            fused=False,
            output_sizes=self.linear_config.output_sizes,
            reorder_size=self.linear_config.n_shards,
        )

        outs = []
        for i, weight in enumerate(lin_weights.weight):
            out = jnp.einsum("bd,fd->bf", x_jax, weight)
            if lin_weights.bias is not None and not layer.skip_bias_add:
                out += lin_weights.bias[i]
            outs.append(out)

        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)