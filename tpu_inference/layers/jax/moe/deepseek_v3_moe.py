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

import enum
from dataclasses import InitVar, dataclass
from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Sharding
from jax.sharding import PartitionSpec
from jaxtyping import Float
from qwix._src.core.ragged_dot import ragged_dot as qwix_ragged_dot
from qwix._src.providers import ptq

from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.layers import FlaxUtils
from tpu_inference.layers.jax.moe.moe import CombineExperts, MoE
from tpu_inference.models.jax.utils.qwix.qwix_utils import (
    manually_quantize_qwix_activation, manually_quantize_qwix_weight)

modeling_flax_utils = FlaxUtils()


@dataclass
class DeepSeekV3Router(nnx.Module):
    """Router module for Mixture-of-Experts (MoE) layers.

    This module determines which experts each token should be routed to based on the input.

    """

    hidden_size: int
    num_experts: int
    num_experts_per_tok: int
    n_groups: int
    topk_groups: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    dtype: jnp.dtype
    rngs: InitVar[nnx.Rngs]

    # Sharding Attributes
    activation_ffw_td: Sharding = ()
    ed_sharding: Sharding = ()
    e_sharding: Sharding = ()

    random_init: bool = False

    router_bias_dtype: jnp.dtype = jnp.float32

    def get_topk_indices(self, scores_TE: Float) -> Float:
        """Get the topk indices of the scores.

        Args:
            scores_TE: The scores to get the topk indices of. Shape (sequence, num_experts).

        Returns:
            The topk indices of the scores. Shape (sequence, num_experts_per_tok).
        """

        scores_TE = scores_TE + self.bias_E
        if self.n_groups > 1:
            experts_per_group = self.num_experts // self.n_groups
            group_scores_TGM = jnp.reshape(
                scores_TE, (-1, self.n_groups, experts_per_group))
            group_scores_TG2 = jax.lax.top_k(group_scores_TGM, k=2)[0]
            group_scores_TG = jnp.sum(group_scores_TG2, axis=-1)
            indices = jax.lax.top_k(group_scores_TG, k=self.topk_groups)[1]

            mask_TG = jnp.any(jnp.arange(
                self.n_groups)[:, None] == indices[..., None, :],
                              axis=-1)
            mask_TE = jnp.repeat(mask_TG,
                                 scores_TE.shape[-1] // mask_TG.shape[-1], -1)
            scores_TE = jnp.where(mask_TE, scores_TE, 0.0)

        indices_TX = jax.lax.top_k(scores_TE, k=self.num_experts_per_tok)[1]

        return indices_TX

    def __call__(self, x_TD: Float) -> Tuple[Float, Float]:
        """Routes tokens to top k experts.

        Args:
            x_TD: Input array of shape (sequence, d_model).

        Returns:
            A tuple containing:
                - weights: Normalized weights for selected experts, shape (sequence, num_experts_per_tok).
                - indices: Indices of selected experts, shape (sequence, num_experts_per_tok).
        """
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, self.activation_ffw_td)

        scores_TE = jnp.einsum("TD,DE -> TE", x_TD, self.kernel_DE.value)
        scores_TE = nnx.sigmoid(scores_TE)

        original_scores_TE = scores_TE
        topk_indices_TX = self.get_topk_indices(scores_TE)
        weights_TX = jnp.take_along_axis(original_scores_TE,
                                         topk_indices_TX,
                                         axis=-1)

        if self.norm_topk_prob:
            weights_TX /= jnp.sum(weights_TX, axis=-1)[..., None] + 1e-20

        weights_TX *= self.routed_scaling_factor

        return weights_TX, topk_indices_TX

    def __post_init__(self, rngs: nnx.Rngs):
        """Generates the router kernel (weights and bias) for routing."""
        D = self.hidden_size
        E = self.num_experts
        self.kernel_DE = create_param(rngs,
                                      shape=(D, E),
                                      dtype=self.dtype,
                                      sharding=self.ed_sharding,
                                      random_init=self.random_init)
        self.bias_E = create_param(rngs,
                                   shape=(E, ),
                                   dtype=self.router_bias_dtype,
                                   sharding=self.e_sharding,
                                   random_init=self.random_init)


@dataclass(kw_only=True)
class SparseMoE(MoE):
    """Mixture-of-Experts (MoE) Routed MLP Layer.

    This module implements a Sparse MoE layer with a router and multiple expert MLPs.

    Attributes:
        num_experts_per_tok: The number of experts each token is routed to.
        tile_size: A tuple (batch, activation_dim, weight_dim) for GMM tiling.
        use_megablox: If True, uses the MegaBlox GMM kernel.
        mesh: The device mesh.
        # TODO: need to redesign this I/O for parallelism
        num_expert_parallelism: The size of the 'expert' mesh dimension.
        # TODO: determine if we get it from external or extrat it in MoE class
        is_batch_sharded_by_expert: True if batch is sharded over 'expert' dim.
    """
    num_experts_per_tok: int
    #TODO: tile size is (tile_batch_seq, tile_activation_dim, tile_weight_dim,) from MaxText
    tile_size: tuple[int, int, int] = (128, 64, 128)
    use_megablox: bool = False
    mesh: jax.sharding.Mesh
    # This should be set if and only if you have quantized your model (via Qwix)
    quantized_dtype: Optional[jnp.dtype] = None

    def __post_init__(self, rngs: nnx.Rngs):
        super().__post_init__(rngs)
        self.combine_experts = CombineExperts(dtype=self.dtype)

        # Derive the expert sharding
        self.expert_axis_name = self.edf_sharding[0]
        if self.expert_axis_name is None:
            self.num_expert_parallelism = 1
        else:
            self.num_expert_parallelism = self.mesh.shape[
                self.expert_axis_name]

        # Derive if data is sharded by expert
        self.data_axis_name = self.activation_ffw_td[0]
        self.is_batch_sharded_by_expert = (
            self.expert_axis_name is not None) and (self.expert_axis_name
                                                    == self.data_axis_name)

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
            reshaped_tokens_TXD = unsorted_tokens_tD.reshape(
                -1, self.num_experts_per_tok, self.hidden_size)
        return self.combine_experts(reshaped_tokens_TXD, router_weights_TX)

    def _gmm(self, inputs, kernel, group_sizes):
        """Performs Grouped Matrix Multiply."""
        num_rows = inputs.shape[0]
        pad_amount = (self.tile_size[0] -
                      num_rows % self.tile_size[0]) % self.tile_size[0]
        if pad_amount > 0:
            inputs = jnp.pad(inputs, ((0, pad_amount), (0, 0)))

        if self.use_megablox:
            #TODO: megablox is used in MaxText, keep a placeholder here for future implement
            raise NotImplementedError(
                "MegaBlox kernel call is not implemented.")
        else:
            inputs = manually_quantize_qwix_activation(
                inputs, "ragged_dot", jnp.float8_e4m3fn, [0], {},
                "absmax") if self.quantized_dtype else inputs
            ragged_dot_func = qwix_ragged_dot if self.quantized_dtype else jax.lax.ragged_dot
            output = ragged_dot_func(
                lhs=inputs,
                rhs=kernel,
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
        expert_shard_id = jax.lax.axis_index(self.expert_axis_name)
        local_expert_size = self.num_local_experts // self.num_expert_parallelism

        if self.num_expert_parallelism > 1:
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

    def __call__(self, x_TD: Float):
        """Performs the forward pass of the Sparse MoE layer."""
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, self.activation_ffw_td)
        router_weights_TX, selected_experts_TX = self.router(x_TD)

        in_specs = (
            PartitionSpec(),  # Replicated `self`
            PartitionSpec(*self.activation_ffw_td),  # Sharded x_TD
            PartitionSpec(),  # Replicated router_weights_TX
            PartitionSpec(),  # Replicated selected_experts_TX
            PartitionSpec(*self.edf_sharding),  # Sharded gating kernel
            PartitionSpec(*self.edf_sharding),  # Sharded up-projection kernel
            PartitionSpec(
                *self.efd_sharding),  # Sharded down-projection kernel
        )
        out_specs = PartitionSpec(*self.activation_ffw_td)

        mapped_moe_fwd = partial(jax.shard_map,
                                 mesh=self.mesh,
                                 in_specs=in_specs,
                                 out_specs=out_specs,
                                 check_vma=False)(
                                     SparseMoE._distributed_sparse_moe_fwd)

        kernel_gating_EDF = self.kernel_gating_EDF.value
        kernel_up_proj_EDF = self.kernel_up_proj_EDF.value
        kernel_down_proj_EFD = self.kernel_down_proj_EFD.value

        if self.quantized_dtype:
            if not isinstance(kernel_gating_EDF, ptq.WithAux):
                kernel_gating_EDF = manually_quantize_qwix_weight(
                    kernel_gating_EDF, self.quantized_dtype, [0, 2], {},
                    "absmax")
            if not isinstance(kernel_up_proj_EDF, ptq.WithAux):
                kernel_up_proj_EDF = manually_quantize_qwix_weight(
                    kernel_up_proj_EDF, self.quantized_dtype, [0, 2], {},
                    "absmax")
            if not isinstance(kernel_down_proj_EFD, ptq.WithAux):
                kernel_down_proj_EFD = manually_quantize_qwix_weight(
                    kernel_down_proj_EFD, self.quantized_dtype, [0, 1], {},
                    "absmax")
            kernel_gating_EDF = kernel_gating_EDF.array
            kernel_up_proj_EDF = kernel_up_proj_EDF.array
            kernel_down_proj_EFD = kernel_down_proj_EFD.array

        return mapped_moe_fwd(self, x_TD, router_weights_TX,
                              selected_experts_TX, kernel_gating_EDF,
                              kernel_up_proj_EDF, kernel_down_proj_EFD)
