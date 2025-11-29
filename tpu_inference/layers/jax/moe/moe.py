from dataclasses import InitVar, dataclass
from functools import partial
from typing import Optional, Tuple
import math
import enum
from unittest import mock
import contextlib

import jax
import jax.numpy as jnp
from jax.experimental import xla_metadata, scheduling_groups
#from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm as megablox_gmm
from tpu_inference.kernels.megablox.gmm import gmm as megablox_gmm
from flax import nnx
from flax.typing import Sharding
from jax.sharding import PartitionSpec
from jaxtyping import Float
from qwix._src.core.ragged_dot import ragged_dot as qwix_ragged_dot
from qwix._src.core.qarray import QArray
from qwix._src.providers import ptq

from tpu_inference.kernels.fused_moe.v1.kernel import fused_ep_moe
from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.layers import FlaxUtils
from tpu_inference.layers.vllm.fused_moe import fused_moe_func
from tpu_inference.models.jax.utils.quantization.quantization_utils import (
    manually_quantize_qwix_activation, manually_quantize_qwix_weight)

modeling_flax_utils = FlaxUtils()
set_xla_metadata = xla_metadata.set_xla_metadata
xla_metadata_call = scheduling_groups.xla_metadata_call

def mosaic_fusion_group(group_id: str):
  """Groups operations for Mosaic fusion.

  Args:
    group_id: A string identifier for the fusion group.

  Returns:
    A call to `xla_metadata_call` with the specified group_id.
  """
  return xla_metadata_call(mosaic_fusion_group=group_id)


@contextlib.contextmanager
def intercept_ragged_dot_general():
  """Context manager to intercept jax.lax.ragged_dot_general calls."""
  original_fn = jax.lax.ragged_dot_general

  def handler(*args, **kwargs):
    lhs, rhs = args[:2]
    assert len(lhs.shape) == 2, f"expectiing 2D lhs, got {lhs.shape}"
    m = lhs.shape[0]
    assert len(rhs.shape) == 3, f"expectiing 3D rhs, got {rhs.shape}"
    k, n = rhs.shape[-2:]
    tiling = (min(m, 512), k, n)
    print(f"setting tiling: {tiling}")
    with set_xla_metadata(ragged_dot_tiling=",".join([str(t) for t in tiling])):
      return original_fn(*args, **kwargs)

  with mock.patch.object(jax.lax, "ragged_dot_general", handler):
    yield

def round_up_to_multiple_of_128_within_limit(x: int, limit: int) -> int:
    """
    Rounds the given integer `x` up to the nearest multiple of 128, without
    exceeding the specified `limit`.

    If `x` is less than or equal to 128, returns 128.
    If `x` is less than `limit`, returns the smallest multiple of 128 greater
    than or equal to `x`.
    If `x` is greater than or equal to `limit`, searches for the largest
    multiple of 128 less than or equal to `limit` (down to 512) that divides `x`
    evenly, and returns it.
    If no such candidate is found, returns `limit`.

    Args:
        x (int): The integer to round up.
        limit (int): The upper bound (must be a multiple of 128).

    Returns:
        int: The rounded value according to the rules above.

    Raises:
        AssertionError: If `limit` is less than 128 or not a multiple of 128.
    """
    assert limit >= 128 and limit % 128 == 0
    if x <= 128:
        return 128
    if x < limit:
        return (x + 127) // 128 * 128
    for candidate in range(limit, 511, -128):
        if x % candidate == 0:
            return candidate
    return limit

@dataclass(kw_only=True)
class Router(nnx.Module):
    """Router module for Mixture-of-Experts (MoE) layers.

    This module determines which experts each token should be routed to based on the input.

    Attributes:
    """
    dtype: jnp.dtype
    hidden_size: int
    num_experts: int
    num_experts_per_tok: int
    router_act: str
    rngs: InitVar[nnx.Rngs]
    activation_ffw_td: Sharding
    ed_sharding: Sharding
    random_init: bool = False
    use_moe_kernel: bool = False

    def __call__(self, x_TD: Float):
        """Routes tokens to experts.

        Args:
            x_TD: Input array of shape (sequence_length, d_model).

        Returns:
            A tuple containing:
                - normalized_weights_TX: Normalized weights for selected experts, shape (sequence_length, num_experts_per_tok).
                - selected_experts_TX: Indices of selected experts, shape (sequence_length, num_experts_per_tok).
        """
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, self.activation_ffw_td)
        router_act = modeling_flax_utils.ACT2FN[self.router_act]
        router_logits_TE = jnp.einsum('TD,DE -> TE', x_TD,
                                      self.kernel_DE.value)
        if self.use_moe_kernel:
            return router_logits_TE
        else:
            weights_TX, selected_experts_TX = jax.lax.top_k(
                router_logits_TE, self.num_experts_per_tok)
            if self.router_act != "sigmoid":  # sigmoid does not accept axis argument.
                normalized_weights_TX = router_act(weights_TX.astype(self.dtype),
                                                axis=-1)
            else:
                normalized_weights_TX = router_act(weights_TX.astype(self.dtype))
            return normalized_weights_TX, selected_experts_TX

    def __post_init__(self, rngs: nnx.Rngs):
        """Generates the router kernel (weights) for routing."""
        shape = (self.hidden_size, self.num_experts)
        self.kernel_DE = create_param(rngs,
                                      shape=shape,
                                      dtype=self.dtype,
                                      sharding=self.ed_sharding,
                                      random_init=self.random_init)


@dataclass(kw_only=True)
class MoE(nnx.Module):
    """Mixture-of-Experts (MoE) Routed MLP Layer.

    This module implements a MoE layer with a router and multiple expert MLPs.

    Attributes:
        router: The Router module.
    """
    dtype: jnp.dtype
    num_local_experts: int
    hidden_size: int
    intermediate_size_moe: int
    hidden_act: str
    rngs: InitVar[nnx.Rngs]
    router: nnx.Module
    mesh: jax.sharding.Mesh
    
    # --- Sharding Config ---
    activation_ffw_td: Sharding
    activation_ffw_ted: Sharding
    edf_sharding: Sharding
    efd_sharding: Sharding
    e2df_sharding: Sharding = ()

    # --- Flags & Configs ---
    apply_expert_weight_before_computation: bool
    random_init: bool = False
    use_fused_moe_kernel: bool = False
    use_vllm_moe_kernel: bool = False
    
    # --- Sparse MoE Specific Attributes ---
    use_sparse_moe: bool = False
    num_experts_per_tok: int = 1  # Required for Sparse, optional/derived for Dense
    tile_size: tuple[int, int, int] = (128, 128, 128)
    use_megablox: bool = False
    quantized_dtype: Optional[jnp.dtype] = None

    # --- MoE Kernel Specific Attributes ---
    renormalize: bool = True

    def __call__(self, x_TD: Float):
        """Performs the forward pass of the MoE layer.

        Args:
            x_TD: Input array of shape (sequence_length, d_model).

        Returns:
            Output array of shape (sequence_length, d_model) after passing through MoE.
        """
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, self.activation_ffw_td)
        if self.use_fused_moe_kernel:
            router_logits_TE = self.router(x_TD)
            block_size = {
                "bt": 32,
                "bf": 512,
                "bd1": 512,
                "bd2": 512,
                "btc": 64,
                "bfc": 256,
                "bd1c": 256,
                "bd2c": 256,
            }
            ep_axis_name = self.efd_sharding[0]
            output_TD = fused_ep_moe(
                mesh=self.mesh,
                tokens=x_TD,
                w1=self.kernel_gating_upproj_E2DF.value,
                w2=self.kernel_down_proj_EFD.value,
                gating_output=router_logits_TE,
                top_k=self.router.num_experts_per_tok,
                ep_axis_name=ep_axis_name,
                renormalize_topk_logits=self.renormalize,
                act_fn=self.hidden_act,
                **block_size,
            )
            return output_TD
        elif self.use_vllm_moe_kernel:
            router_logits_TE = self.router(x_TD)
            output_TD = fused_moe_func(
                hidden_states=x_TD,
                w1=self.kernel_gating_upproj_EFD.value,
                w2=self.kernel_down_proj_EDF.value,
                w1_bias=self.w1_bias,
                w2_bias=self.w2_bias,
                gating_output=router_logits_TE,
                topk=self.router.num_experts_per_tok,
                renormalize=self.renormalize,
                mesh=self.mesh,
                use_ep=self.num_expert_parallelism>1,
                activation=self.hidden_act,
            )
            return output_TD
        else:
            weights_TX, indices_TX = self.router(x_TD)

            if self.use_sparse_moe:
                if self.quantized_dtype:
                    
                    #gate_up_scale_channel_sharding = self.efd_sharding[2] if   else None
                    gating_up_proj_spec = (PartitionSpec(*self.edf_sharding), PartitionSpec(self.edf_sharding[0], self.edf_sharding[1], self.edf_sharding[2]))
                    down_proj_spec = (PartitionSpec(*self.efd_sharding), PartitionSpec(self.efd_sharding[0], None, self.efd_sharding[2]))
                else:
                    gating_up_proj_spec = PartitionSpec(*self.edf_sharding)
                    down_proj_spec = PartitionSpec(*self.efd_sharding)
                in_specs = (
                    PartitionSpec(),  # Replicated `self`
                    PartitionSpec(*self.activation_ffw_td),  # Sharded x_TD
                    PartitionSpec(),  # Replicated router_weights_TX
                    PartitionSpec(),  # Replicated selected_experts_TX
                    gating_up_proj_spec,  # Sharded gating kernel
                    gating_up_proj_spec,  # Sharded up-projection kernel
                    down_proj_spec,  # Sharded down-projection kernel
                )
                out_specs = PartitionSpec(*self.activation_ffw_td)

                mapped_moe_fwd = partial(jax.experimental.shard_map.shard_map,
                                        mesh=self.mesh,
                                        in_specs=in_specs,
                                        out_specs=out_specs,
                                        check_rep=False)(
                                            MoE._distributed_sparse_moe_fwd)

                kernel_gating_EDF = self._process_weight_for_qwix('kernel_gating_EDF', self.kernel_gating_EDF, channelwise_axes=[0, 2], tiled_axes={1: 1792})
                kernel_up_proj_EDF = self._process_weight_for_qwix('kernel_up_proj_EDF', self.kernel_up_proj_EDF, channelwise_axes=[0, 2], tiled_axes={1: 1792})
                kernel_down_proj_EFD = self._process_weight_for_qwix('kernel_down_proj_EFD', self.kernel_down_proj_EFD, channelwise_axes=[0, 2], tiled_axes={1: 2048})

                return mapped_moe_fwd(self, x_TD, weights_TX,
                                    indices_TX, kernel_gating_EDF,
                                    kernel_up_proj_EDF, kernel_down_proj_EFD)
            
            # Dense Matmul        
            else:
                one_hot_indices_TXE = jax.nn.one_hot(
                    indices_TX, num_classes=self.num_local_experts, dtype=self.dtype)
                full_weights_TE = jnp.sum(one_hot_indices_TXE * weights_TX[..., None],
                                        axis=1)
                # Some models use the routing scores to weight the data instead of
                # weighting the expert outputs.
                if self.apply_expert_weight_before_computation:
                    with jax.named_scope("pre_computing_weight"):
                        return self._moe_fwd_preapply_router_weights(
                            x_TD, full_weights_TE)
                else:
                    return self._moe_fwd(x_TD, full_weights_TE)        

    def __post_init__(self, rngs: nnx.Rngs):
        """Generates the kernels (weights) for the router and experts (gating, up-projection, and down-projection layers)."""
        E = self.num_local_experts
        D = self.hidden_size
        F = self.intermediate_size_moe

        if self.use_fused_moe_kernel:
            shape_gating_up = (E, 2, D, F)
            if self.edf_sharding:
                self.e2df_sharding = (self.edf_sharding[0], None, self.edf_sharding[1], self.edf_sharding[2])
            self.kernel_gating_upproj_E2DF = create_param(rngs,
                                              shape=(E, 2, D, F),
                                              dtype=self.dtype,
                                              sharding=self.e2df_sharding,
                                              random_init=self.random_init)
            self.kernel_down_proj_EFD = create_param(rngs,
                                                    shape=(E, F, D),
                                                    dtype=self.dtype,
                                                    sharding=self.efd_sharding,
                                                    random_init=self.random_init)
        elif self.use_vllm_moe_kernel:
            shape_gating_up = (E, 2 * F, D)
            self.kernel_gating_upproj_EFD = create_param(rngs,
                                              shape=(E, 2 * F, D),
                                              dtype=self.dtype,
                                              sharding=self.efd_sharding,
                                              random_init=self.random_init)
            self.kernel_down_proj_EDF = create_param(rngs,
                                                    shape=(E, D, F),
                                                    dtype=self.dtype,
                                                    sharding=self.edf_sharding,
                                                    random_init=self.random_init)
        else:
            #shape_gating = (self.num_local_experts, D, F)
            #shape_up = (self.num_local_experts, D, F)

            self.kernel_gating_EDF = create_param(rngs,
                                                shape=(E, D, F),
                                                dtype=self.dtype,
                                                sharding=self.edf_sharding,
                                                random_init=self.random_init)
            self.kernel_up_proj_EDF = create_param(rngs,
                                                shape=(E, D, F),
                                                dtype=self.dtype,
                                                sharding=self.edf_sharding,
                                                random_init=self.random_init)
            self.kernel_down_proj_EFD = create_param(rngs,
                                                    shape=(E, F, D),
                                                    dtype=self.dtype,
                                                    sharding=self.efd_sharding,
                                                    random_init=self.random_init)

        # Default MoE has no bias vectors
        self.w1_bias, self.w2_bias = (None, None)

        self.expert_axis_name = self.edf_sharding[0]
        if self.expert_axis_name is None:
            self.num_expert_parallelism = 1
        else:
            if isinstance(self.expert_axis_name, str):
                self.num_expert_parallelism =self.mesh.shape[self.expert_axis_name]
            else:
                self.num_expert_parallelism = math.prod(self.mesh.shape[axis] for axis in self.expert_axis_name)
        # Derive if data is sharded by expert
        self.data_axis_name = self.activation_ffw_td[0]
        self.is_batch_sharded_by_expert = (
            self.expert_axis_name is not None) and (self.expert_axis_name == self.data_axis_name)

    def _moe_fwd_preapply_router_weights(self, x_TD: jax.Array, weights_TE):
        """Performs the forward pass of the MoE experts with router weights pre-applied to the inputs.

        Args:
            x_TD: Input array for the experts, shape (sequence_length, hidden_size).
            weights_TE: Router weights, shape (sequence_length, num_experts).

        Returns:
            Output array of shape (sequence_length, d_model).
        """
        # Data needs to be replicated since it will be weighted by the router
        # scores before being passed to each expert.
        num_experts = weights_TE.shape[-1]
        x_TED = jnp.repeat(x_TD[:, None, :], num_experts, 1)
        weights_TED = weights_TE[..., None]
        x_TED = jnp.asarray(x_TED, self.dtype)

        with jax.named_scope("activation_expert_weighting"):
            x_TED = x_TED * weights_TED

        x_TED = nnx.with_sharding_constraint(x_TED, self.activation_ffw_ted)
        with jax.named_scope("gating"):
            gating_TEF = jnp.einsum('TED,EDF -> TEF', x_TED,
                                    self.kernel_gating_EDF.value)
            activated_gating_TEF = modeling_flax_utils.ACT2FN[self.hidden_act](
                gating_TEF)
        with jax.named_scope("up_projection"):
            up_proj_TEF = jnp.einsum('TED,EDF -> TEF', x_TED,
                                     self.kernel_up_proj_EDF.value)

        fuse_TEF = activated_gating_TEF * up_proj_TEF

        with jax.named_scope("down_projection"):
            down_proj_TED = jnp.einsum('TEF,EFD -> TED', fuse_TEF,
                                       self.kernel_down_proj_EFD.value)
        with jax.named_scope("sum"):
            output_TD = down_proj_TED.sum(axis=1)
        return output_TD.astype(self.dtype)

    def _moe_fwd(self, x_TD: Float, weights):
        """Performs the basic forward pass of the MoE experts without dropping or megablocks.

        Args:
            x_TD: Input array for the experts, shape (sequence_length, d_model).
            weights: Weights for combining expert outputs, shape (sequence_length, num_experts).

        Returns:
            Output array of shape (sequence_length, d_model).
        """
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, self.activation_ffw_td)
        with jax.named_scope("gating"):
            gating_TEF = jnp.einsum('TD,EDF -> TEF', x_TD,
                                    self.kernel_gating_EDF.value)
            activated_gating_TEF = modeling_flax_utils.ACT2FN[self.hidden_act](
                gating_TEF)
        with jax.named_scope("up_projection"):
            up_proj_TEF = jnp.einsum('TD,EDF -> TEF', x_TD,
                                     self.kernel_up_proj_EDF.value)

        fuse_TEF = activated_gating_TEF * up_proj_TEF

        with jax.named_scope("down_projection"):
            down_proj_TED = jnp.einsum('TEF,EFD -> TED', fuse_TEF,
                                       self.kernel_down_proj_EFD.value)
        with jax.named_scope("sum"):
            output_TD = jnp.einsum('TED,TE -> TD', down_proj_TED, weights)
        return output_TD.astype(self.dtype)

    def _sort_activations(self, inputs: jax.Array,
                          sort_indices: jax.Array) -> jax.Array:
        """Sorts activations(inputs) by `sort_indices` for the forward pass."""
        return inputs[sort_indices, ...]

    @staticmethod
    def get_all_to_all_params(
        all_shards_group_sizes,
        shard_id,
        num_expert_parallelism,
        is_batch_sharded=True,
    ):
        """Generates params for ragged_all_to_all communication."""

        class TransformStrategy(enum.Enum):
            INPUT_OFFSET = enum.auto()
            SEND_SIZE = enum.auto()
            OUTPUT_OFFSET = enum.auto()
            RECV_SIZE = enum.auto()

        def transform_array(input_array, shard_id, strategy, is_batch_sharded):
            if is_batch_sharded:
                if strategy == TransformStrategy.INPUT_OFFSET:
                    local_array = input_array[shard_id]
                    return jnp.concatenate(
                        (jnp.array([0]), jnp.cumsum(local_array)[:-1]))
                elif strategy == TransformStrategy.SEND_SIZE:
                    return input_array[shard_id]
                elif strategy == TransformStrategy.OUTPUT_OFFSET:
                    zero_row = jnp.zeros((1, ) + input_array.shape[1:],
                                         dtype=input_array.dtype)
                    array_with_zeros = jnp.concatenate((zero_row, input_array),
                                                       axis=0)
                    cumulated_array = jnp.cumsum(array_with_zeros,
                                                 axis=0,
                                                 dtype=input_array.dtype)
                    return cumulated_array[shard_id]
                elif strategy == TransformStrategy.RECV_SIZE:
                    return input_array[:, shard_id]
                else:
                    raise ValueError(
                        f"Unknown transform array strategy: {strategy}")
            else:
                if strategy == TransformStrategy.INPUT_OFFSET:
                    return jnp.zeros(num_expert_parallelism,
                                     dtype=input_array.dtype)
                elif strategy == TransformStrategy.SEND_SIZE:
                    return jnp.repeat(input_array[shard_id],
                                      num_expert_parallelism)
                elif strategy == TransformStrategy.OUTPUT_OFFSET:
                    output_offset = jnp.concatenate(
                        (jnp.array([0]),
                         jnp.cumsum(input_array[:-1])))[shard_id]
                    return jnp.repeat(output_offset, num_expert_parallelism)
                elif strategy == TransformStrategy.RECV_SIZE:
                    return input_array
                else:
                    raise ValueError(
                        f"Unknown transform array strategy: {strategy}")

        input_offsets = transform_array(all_shards_group_sizes, shard_id,
                                        TransformStrategy.INPUT_OFFSET,
                                        is_batch_sharded)
        send_sizes = transform_array(all_shards_group_sizes, shard_id,
                                     TransformStrategy.SEND_SIZE,
                                     is_batch_sharded)
        output_offsets = transform_array(all_shards_group_sizes, shard_id,
                                         TransformStrategy.OUTPUT_OFFSET,
                                         is_batch_sharded)
        recv_sizes = transform_array(all_shards_group_sizes, shard_id,
                                     TransformStrategy.RECV_SIZE,
                                     is_batch_sharded)
        return input_offsets, send_sizes, output_offsets, recv_sizes

    def _local_permute(
        self,
        inputs,
        global_group_sizes,
        local_expert_size,
        shard_index,
        is_offset=False,
        global_sorted_experts=None,
    ):
        """Permutes tokens locally within an expert shard."""
        # global_group_sizes: (tokens parallelism, num_total_experts)
        # all_shard_local_sizes: (tokens parallelism, num local experts in the shard)
        all_shard_local_sizes = jax.lax.dynamic_slice_in_dim(
            global_group_sizes,
            shard_index * local_expert_size,
            local_expert_size,
            axis=1,
        )
        local_sizes = all_shard_local_sizes.reshape(-1)

        # local_group_size: (tokens parallelism, )
        local_group_size = jnp.sum(all_shard_local_sizes, axis=0)

        # When token replicated in devices
        if is_offset:
            global_sorted_shard_assignments = jnp.floor_divide(
                global_sorted_experts, local_expert_size)
            expert_indices = jnp.where(
                global_sorted_shard_assignments == shard_index,
                jnp.mod(global_sorted_experts, local_expert_size),
                local_expert_size,
            )

        # When token sharded in devices
        else:
            base_indices = jnp.mod(jnp.arange(local_sizes.shape[0]),
                                   local_expert_size)
            expert_indices = jnp.repeat(base_indices,
                                        local_sizes,
                                        total_repeat_length=inputs.shape[0])

        sorted_indices = jnp.argsort(expert_indices)
        # sort the inputs based on the local expert_indices
        sorted_inputs = self._sort_activations(inputs, sorted_indices)
        # sortted local expert id from 0 to local expert size
        sorted_experts_ids = expert_indices[sorted_indices]
        return (
            sorted_inputs,
            sorted_indices,
            local_group_size,
            sorted_experts_ids,
        )

    def _permute(self, inputs_TD: Float, selected_experts_TX: jax.Array):
        """Global permute: Sorts tokens by assigned expert."""
        # suffix t = T * X = total_assignments for the local tokens(T) on this device.
        total_tokens = inputs_TD.shape[0]
        flat_expert_indices = selected_experts_TX.flatten()
        sort_indices_t = jnp.argsort(flat_expert_indices)

        replicated_inputs_tD = jnp.repeat(inputs_TD,
                                          self.num_experts_per_tok,
                                          axis=0)
        sorted_inputs_tD = self._sort_activations(replicated_inputs_tD,
                                                  sort_indices_t)

        # number of tokens assigned to each expert
        group_sizes_E = jnp.bincount(flat_expert_indices,
                                     length=self.num_local_experts)

        expert_ids = jnp.arange(self.num_local_experts)
        total_assignments = total_tokens * self.num_experts_per_tok
        sorted_expert_assignments_t = jnp.repeat(
            expert_ids,
            repeats=group_sizes_E,
            total_repeat_length=total_assignments)

        return (
            sorted_inputs_tD,
            sort_indices_t,
            group_sizes_E,
            sorted_expert_assignments_t,
        )

    def _unpermute(self, processed_tokens: jax.Array, sort_indices: jax.Array,
                   router_weights_TX: jax.Array):
        """Unsorts tokens to their original order and combines expert outputs with router's weight."""
        with jax.named_scope("unpermute"):
            unsorted_tokens_tD = self._sort_activations(
                processed_tokens, jnp.argsort(sort_indices))
            D = unsorted_tokens_tD.shape[-1]
            reshaped_tokens_TXD = unsorted_tokens_tD.reshape(
                -1, self.num_experts_per_tok, D)
        with jax.named_scope("combine_weights"):
            output_TD = jnp.einsum(
                "TXD,TX -> TD",
                reshaped_tokens_TXD.astype(jnp.float32),
                router_weights_TX.astype(jnp.float32),
                precision='float32',
            )

        return output_TD.astype(self.dtype)
    @mosaic_fusion_group("qwix_quant")    
    def _gmm(self, inputs, kernel, group_sizes):
        """Performs Grouped Matrix Multiply."""
        num_rows = inputs.shape[0]
        pad_amount = (self.tile_size[0] -
                      num_rows % self.tile_size[0]) % self.tile_size[0]
        if pad_amount > 0:
            inputs = jnp.pad(inputs, ((0, pad_amount), (0, 0)))

        if self.use_megablox:
            final_kernel = kernel

            if self.quantized_dtype:
                kernel_qvalue, kernel_scale = kernel
                #kernel_qvalue = jnp.swapaxes(kernel_qvalue, 1, 2)
                kernel_scale = jnp.expand_dims(kernel_scale, 2)
            else:
                #kernel_qvalue = jnp.swapaxes(kernel, 1, 2)
                kernel_qvalue = kernel
                kernel_scale = None

            m, g, k, n = inputs.shape[0], *kernel_qvalue.shape
            tm = round_up_to_multiple_of_128_within_limit(m, 512)
            tk = round_up_to_multiple_of_128_within_limit(k, 2048)
            tn = round_up_to_multiple_of_128_within_limit(n, 2048)

            output = megablox_gmm(
                lhs=inputs,
                rhs=kernel_qvalue,
                rhs_scale=kernel_scale,
                group_sizes=group_sizes,
                #_rhs=True,
                preferred_element_type=self.dtype,
                tiling=(tm, tk, tn),
            )

        else:
            inputs = manually_quantize_qwix_activation(
                inputs, "ragged_dot", jnp.float8_e4m3fn, [0], {},
                "absmax") if self.quantized_dtype else inputs
            ragged_dot_func = qwix_ragged_dot if self.quantized_dtype else jax.lax.ragged_dot
            final_kernel = kernel

            if self.quantized_dtype:
                kernel_qvalue, kernel_scale = kernel

                final_kernel = QArray(
                    qvalue=kernel_qvalue, 
                    scale=kernel_scale, 
                    qtype=self.quantized_dtype
                )

            tiling = self.tile_size
            with intercept_ragged_dot_general():
                output = ragged_dot_func(
                    lhs=inputs,
                    rhs=final_kernel,
                    group_sizes=group_sizes,
                    preferred_element_type=self.dtype,
                )

        if pad_amount > 0:
            output = output[:num_rows, :]
        return output

    @staticmethod
    def _distributed_sparse_moe_fwd(
        self,
        x_TD: jax.Array,
        router_weights_TX: jax.Array,
        selected_experts_TX: jax.Array,
        kernel_gating: jax.Array,
        kernel_up_proj: jax.Array,
        kernel_down_proj: jax.Array,
    ):
        """
        The sparse MoE forward pass with fully distributed logic.
        This assumes it is running within a distributed TPU.
        """

        # 1. Global Permute, perpute all tokens across shards
        (
            sorted_inputs,
            global_sort_indices,
            global_group_sizes,
            global_sorted_experts,
        ) = self._permute(x_TD, selected_experts_TX)

        # TODO: update to 'expert' after we enable expert parallelism, currently experts are sharded along model axis
        # or we sould derive it from the model init

        if self.num_expert_parallelism > 1:
            expert_shard_id = jax.lax.axis_index(self.expert_axis_name)
            local_expert_size = self.num_local_experts // self.num_expert_parallelism
            if self.is_batch_sharded_by_expert:
                # When token sharded in devices
                # In this path, we assume the data(tokens) are fully sharded on expert, namely data_axis_name == expert_axis_name

                # 2a. Send Tokens To Experts (All-to-All)
                # Gather group sizes from all data shards
                # all_shards_group_sizes: (data parallelism = expert parallelism, number of total experts )
                all_shards_group_sizes = jax.lax.all_gather(
                    global_group_sizes, axis_name=self.data_axis_name)

                # all_shards_group_sizes_per_expert_shard[i][j] = # tokens on shard[i] to be sent to expert shard[j]
                all_shards_group_sizes_per_expert_shard = jnp.sum(
                    all_shards_group_sizes.reshape(
                        self.num_expert_parallelism,  # data parallelism
                        self.num_expert_parallelism,  # expert parallelism
                        local_expert_size  # Experts per shard
                    ),
                    axis=2)
                input_offsets, send_sizes, output_offsets, recv_sizes = self.get_all_to_all_params(
                    all_shards_group_sizes_per_expert_shard, expert_shard_id,
                    self.num_expert_parallelism)
                # Estimate buffer size
                local_total_assignments = x_TD.shape[
                    0] * self.num_experts_per_tok
                global_total_assignments = local_total_assignments * self.num_expert_parallelism
                output_shape_est = jnp.zeros(
                    (global_total_assignments, self.hidden_size),
                    dtype=sorted_inputs.dtype)

                inputs_after_all2all = jax.lax.ragged_all_to_all(
                    sorted_inputs,
                    output_shape_est,
                    input_offsets,
                    send_sizes,
                    output_offsets,
                    recv_sizes,
                    axis_name=self.expert_axis_name)

                # 3a. Local Permute
                # Get full group sizes from all shards
                full_global_group_sizes = jax.lax.all_gather(
                    global_group_sizes, axis_name=self.expert_axis_name)
                (
                    compute_inputs,
                    local_sorted_indices,
                    compute_group_sizes,
                    compute_expert_ids,
                ) = self._local_permute(
                    inputs_after_all2all,
                    full_global_group_sizes,
                    local_expert_size,
                    shard_index=expert_shard_id,
                    is_offset=False,
                )

            else:
                # When token replicated in devices

                # 2. No send all-to-all needed, as the tokens are sorted and replicated on all devices
                # 3b. Local "Permute"
                (
                    compute_inputs,
                    local_sorted_indices,
                    compute_group_sizes,
                    compute_expert_ids,
                ) = self._local_permute(
                    sorted_inputs,
                    global_group_sizes[None, :],
                    local_expert_size,
                    shard_index=expert_shard_id,
                    is_offset=True,
                    global_sorted_experts=global_sorted_experts,
                )

                # Calculate group sizes for return all-to-all
                reshaped_group_sizes = jnp.sum(global_group_sizes.reshape(
                    -1, local_expert_size),
                                               axis=1)
                mask = compute_expert_ids < local_expert_size
                compute_inputs = compute_inputs * mask[..., None]

        else:
            # --- NO EXPERT PARALLELISM ---
            compute_inputs = sorted_inputs
            compute_group_sizes = global_group_sizes
            compute_expert_ids = global_sorted_experts
            local_sorted_indices = jnp.arange(sorted_inputs.shape[0])

        # 4. Compute: Apply experts using Grouped Matrix Multiply
        with jax.named_scope("gating"):
            # compute_inputs: (local total assignments, D)
            gating_TEF = self._gmm(compute_inputs, kernel_gating,
                                   compute_group_sizes)
            activated_gating_TEF = modeling_flax_utils.ACT2FN[self.hidden_act](
                gating_TEF)

        with jax.named_scope("up_projection"):
            up_proj_TEF = self._gmm(compute_inputs, kernel_up_proj,
                                    compute_group_sizes)

        fuse_TEF = activated_gating_TEF * up_proj_TEF

        with jax.named_scope("down_projection"):
            # intermediate_output: (local total assignments, D)
            intermediate_output = self._gmm(fuse_TEF, kernel_down_proj,
                                            compute_group_sizes)

        # 5. Return Results (All-to-All)
        if self.num_expert_parallelism > 1:
            local_total_assignments = x_TD.shape[0] * self.num_experts_per_tok
            output_shape = jnp.zeros(
                (local_total_assignments, self.hidden_size),
                dtype=intermediate_output.dtype)

            if self.is_batch_sharded_by_expert:
                # When token sharded in devices
                # Unsort locally before sending back
                local_output = self._sort_activations(
                    intermediate_output, jnp.argsort(local_sorted_indices))

                input_offsets, send_sizes, output_offsets, recv_sizes = self.get_all_to_all_params(
                    jnp.transpose(all_shards_group_sizes),
                    expert_shard_id,
                    self.num_expert_parallelism,
                )
                final_intermediate_output = jax.lax.ragged_all_to_all(
                    local_output,
                    output_shape,
                    input_offsets,
                    send_sizes,
                    output_offsets,
                    recv_sizes,
                    axis_name=self.expert_axis_name)
            else:
                # When token replicated in devices
                input_offsets, send_sizes, output_offsets, recv_sizes = self.get_all_to_all_params(
                    reshaped_group_sizes,
                    expert_shard_id,
                    self.num_expert_parallelism,
                    is_batch_sharded=False,
                )
                final_intermediate_output = jax.lax.ragged_all_to_all(
                    intermediate_output,
                    output_shape,
                    input_offsets,
                    send_sizes,
                    output_offsets,
                    recv_sizes,
                    axis_name=self.expert_axis_name)
        else:
            final_intermediate_output = intermediate_output

        # 6. Global Unpermute (on the data shard)
        with jax.named_scope("unpermute"):
            output_TD = self._unpermute(final_intermediate_output,
                                        global_sort_indices, router_weights_TX)

        return output_TD

    def _process_weight_for_qwix(self, name, weight_param, channelwise_axes=[], tiled_axes={}):
        """
        Extracts weight value, applies quantization if needed, 
        and returns the underlying array.
        """
        weight = weight_param.value

        if self.quantized_dtype:
            if not isinstance(weight, ptq.WithAux):
                weight = manually_quantize_qwix_weight(
                    name,
                    weight, 
                    self.quantized_dtype, 
                    channelwise_axes, 
                    tiled_axes, 
                    "absmax"
                )

            #TODO swap the scale per mgblx kernels
            #return (weight.array.qvalue,jnp.swapaxes(weight.array.scale , 1, 2))
            return (weight.array.qvalue, weight.array.scale)
        
        return weight