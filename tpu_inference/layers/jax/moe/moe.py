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

from dataclasses import InitVar, dataclass
from typing import Any, Iterable, Iterator, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Sharding
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import Float

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import cpu_mesh_context
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.layers import FlaxUtils
from tpu_inference.layers.jax.moe.utils import select_moe_backend
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import (
    jax_array_from_reshaped_torch, shard_put)

modeling_flax_utils = FlaxUtils()
logger = init_logger(__name__)


@dataclass(kw_only=True)
class CombineExperts(nnx.Module):
    """Combines expert outputs with router weights.

    Supports `TED,TE -> TD` when passed expert outputs, using float32
    accumulation for numerical stability, then casting back to the target
    dtype.
    """

    dtype: jnp.dtype

    def __call__(self, expert_outputs_TED: Float, weights_TE: Float) -> Float:
        with jax.named_scope("combine_experts"):
            output_TD = jnp.einsum(
                "TED,TE -> TD",
                expert_outputs_TED.astype(jnp.float32),
                weights_TE.astype(jnp.float32),
                precision="float32",
            )

        return output_TD.astype(self.dtype)


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
    moe_backend: MoEBackend = MoEBackend.DENSE_MAT
    mesh: Optional[jax.sharding.Mesh] = None

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
        x_TD = jax.lax.with_sharding_constraint(
            x_TD,
            NamedSharding(self.mesh, PartitionSpec(*self.activation_ffw_td)))
        router_act = modeling_flax_utils.ACT2FN[self.router_act]
        router_logits_TE = jnp.einsum('TD,DE -> TE', x_TD,
                                      self.kernel_DE.value)

        #TODO: Refactor the Router so that it will always only return router_logits_TE
        if self.moe_backend in MoEBackend.fused_moe_backends():
            return router_logits_TE
        else:
            weights_TX, selected_experts_TX = jax.lax.top_k(
                router_logits_TE, self.num_experts_per_tok)
            if self.router_act != "sigmoid":  # sigmoid does not accept axis argument.
                normalized_weights_TX = router_act(weights_TX.astype(
                    self.dtype),
                                                   axis=-1)
            else:
                normalized_weights_TX = router_act(
                    weights_TX.astype(self.dtype))
            return normalized_weights_TX, selected_experts_TX

    def __post_init__(self, rngs: nnx.Rngs):
        """Generates the router kernel (weights) for routing."""
        shape = (self.hidden_size, self.num_experts)
        self.kernel_DE = create_param(rngs,
                                      shape=shape,
                                      dtype=self.dtype,
                                      sharding=self.ed_sharding,
                                      random_init=self.random_init)


# --- Main Class for MoE ---
# TODO(#3041): Remove once we JaxRoutedExperts is fully ready.
@dataclass(kw_only=True)
class JaxMoE(JaxModule):
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
    expert_axis_name: str
    num_expert_parallelism: int
    random_init: bool = False
    moe_backend: MoEBackend = MoEBackend.DENSE_MAT
    scoring_func: str = "softmax"

    # --- Sparse MoE Specific Attributes ---
    num_experts_per_tok: int = 1  # Required for Sparse, optional/derived for Dense
    tile_size: tuple[int, int, int] = (128, 128, 128)
    # NOTE: this is only needed for SparseMoE
    qwix_quantized_weight_dtype: Optional[jnp.dtype] = None

    # --- MoE Kernel Specific Attributes ---
    renormalize: bool = True

    # ---- Quantization Specific Attributes ----
    quant_config: Optional[QuantizationConfig] = None
    prefix: str = ""
    enable_return_routed_experts: bool = False

    def __call__(
        self,
        x_TD: jax.Array,
        router_logits: Optional[jax.Array] = None
    ) -> tuple[jax.Array, Optional[jax.Array]]:
        """Performs the forward pass of the MoE layer.

        Args:
            x_TD: Input array of shape (sequence_length, d_model).
            router_logits: Optional pre-computed router logits. If not provided, logits will be computed using the router.

        Returns:
            Output array of shape (sequence_length, d_model) after passing through MoE.
            If `enable_return_routed_experts` is True, also returns the indices of the selected experts.
        """
        if self.quant_method is None:
            raise ValueError("Expected quant_method to be set!")
        if router_logits is None:
            router_logits = self.router(x_TD)
        x_TD = self.quant_method.apply_jax(self,
                                           x_TD,
                                           router_logits=router_logits)
        if self.enable_return_routed_experts:
            if self.moe_backend in MoEBackend.fused_moe_backends():
                _, selected_experts_TX = jax.lax.top_k(
                    router_logits, self.num_experts_per_tok)
            else:
                _, selected_experts_TX = router_logits
            return x_TD, selected_experts_TX
        else:
            return x_TD, None

    def __post_init__(self, rngs: nnx.Rngs):
        """Generates the kernels (weights) for the router and experts (gating, up-projection, and down-projection layers)."""
        E = self.num_local_experts
        D = self.hidden_size
        F = self.intermediate_size_moe

        self.kernel_gating_EDF = create_param(rngs,
                                              shape=(E, D, F),
                                              dtype=self.dtype,
                                              sharding=self.edf_sharding,
                                              random_init=self.random_init)
        self.kernel_gating_EDF.set_metadata(
            _weights_to_load=[None for _ in range(E)])
        self.kernel_up_proj_EDF = create_param(rngs,
                                               shape=(E, D, F),
                                               dtype=self.dtype,
                                               sharding=self.edf_sharding,
                                               random_init=self.random_init)
        self.kernel_up_proj_EDF.set_metadata(
            _weights_to_load=[None for _ in range(E)])
        self.kernel_down_proj_EFD = create_param(
            rngs,
            shape=(E, F, D),
            dtype=self.dtype,
            sharding=self.efd_sharding if self.moe_backend
            not in MoEBackend.fused_moe_backends() else self.edf_sharding,
            random_init=self.random_init)
        self.kernel_down_proj_EFD.set_metadata(
            _weights_to_load=[None for _ in range(E)])

        # Derive if data is sharded by expert
        self.data_axis_name = self.activation_ffw_td[0]
        self.is_batch_sharded_by_expert = (
            self.expert_axis_name is not None) and (self.expert_axis_name
                                                    == self.data_axis_name)

        self.top_k = self.router.num_experts_per_tok
        self.use_ep = self.num_expert_parallelism > 1
        self.activation = self.hidden_act
        self.scoring_func = self.scoring_func

        if self.quant_config is None:
            self.quant_method = None
        elif (quant_method :=
              self.quant_config.get_quant_method(self, prefix=self.prefix)):
            assert isinstance(quant_method, QuantizeMethodBase)
            self.quant_method = quant_method
            self.quant_method.create_weights_jax(self, rngs=rngs)
        else:
            self.quant_method = None

    def named_parameters(self, *args, **kwargs) -> Iterator[tuple[str, Any]]:
        for name, param in super().named_parameters(*args, **kwargs):
            # Weight loader relies on this function to check if all parameters are loaded.
            # We put router/gating param in JaxMoE because we fuse all kinds of MoE into one.
            # However, router/gating param does not belong to "experts" but "mlp" in HF checkpoint,
            # so we skip them in the named_parameters of JaxMoE to avoid confusion for weight loading completeness check.
            if "router" in name:
                continue
            yield name, param

    def load_weights(self, weights: Iterable):
        """Used by JaxAutoWeightLoader to load HF weights into the layer."""
        if self.quant_method is None or not hasattr(self.quant_method,
                                                    "load_weights"):
            return self._load_weights(weights)

        return self.quant_method.load_weights(
            layer=self,
            original_load_weights_fn=self._load_weights,
            weights=weights)

    def _load_weights(self,
                      weights: Iterable,
                      *,
                      mesh: jax.sharding.Mesh | None = None):
        """Load HF weights into the layer.

        self.quant_method might reuse this method if the quantization method has specific logic for loading weights.
        """

        cnt = 0
        for param_name, torch_weight in weights:
            cnt += 1
            param_name: str = param_name.split(
                self.prefix)[-1]  # ".0.down_proj.weight" for example
            names = param_name.split(".")
            assert len(
                names
            ) == 3, f"Expected param name to be .<expert_id>.<param_name>.weight, got {param_name}"
            expert_id, param_type, _ = names
            expert_id = int(expert_id)
            jax_param = None
            if param_type.endswith("up_proj"):
                jax_param = self.kernel_up_proj_EDF
            elif param_type.endswith("down_proj"):
                jax_param = self.kernel_down_proj_EFD
            elif param_type.endswith("gate_proj"):
                jax_param = self.kernel_gating_EDF
            else:
                raise ValueError(
                    f"Unexpected param type in {param_name}, expected up_proj, down_proj, gate_proj"
                )

            assert isinstance(jax_param, nnx.Param)

            jax_weight = jax_array_from_reshaped_torch(
                torch_weight, reshape_dims=(1, ) +
                torch_weight.shape)  # add expert dim for concatenation later
            jax_param._weights_to_load[expert_id] = jax_weight

        logger.debug(f"Loaded {cnt} weights for {self.prefix} MoE layer.")

        loaded_names = set()
        # This function could be called more than once, if the weights for moe layer is spread
        # across multiple safetensor files. Here we use counter to track the completion of weight loading, and only perform the fusion and sharding after all weights are loaded.
        for param_name, param in {
                "kernel_gating_EDF": self.kernel_gating_EDF,
                "kernel_up_proj_EDF": self.kernel_up_proj_EDF,
                "kernel_down_proj_EFD": self.kernel_down_proj_EFD
        }.items():
            weights_to_load = param._weights_to_load
            if all(w is not None for w in weights_to_load):
                with cpu_mesh_context():
                    weights = jnp.concatenate(param._weights_to_load, axis=0)
                try:
                    param.value = shard_put(weights, param.out_sharding, mesh)
                    loaded_names.add(param_name)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load weights for {param_name} with {weights.shape=} {param.value.shape=}"
                    ) from e

        return loaded_names


class JaxRoutedExperts(JaxModule):
    """Expert-only MoE module analogous to vllm's RoutedExperts.

    Decouples expert computation from routing; the caller is responsible for
    computing router_logits before calling __call__.  use_ep and moe_backend
    are derived from the vLLM parallel config at init time so the EP/TP
    backend selection matches the torchax path.

    When quant_config is None, UnquantizedConfig is used so that
    process_weights_after_loading (sharding/fusion) always runs.
    """

    def __init__(
        self,
        *,
        dtype: jnp.dtype,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size_moe: int,
        hidden_act: str,
        rngs: nnx.Rngs,
        mesh: jax.sharding.Mesh,
        top_k: int,
        scoring_func: str = "softmax",
        renormalize: bool = True,
        random_init: bool = False,
        qwix_quantized_weight_dtype: Optional[jnp.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        enable_return_routed_experts: bool = False,
    ):
        self.dtype = dtype
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.intermediate_size_moe = intermediate_size_moe
        self.hidden_act = hidden_act
        self.mesh = mesh
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.random_init = random_init
        self.qwix_quantized_weight_dtype = qwix_quantized_weight_dtype
        self.prefix = prefix
        self.enable_return_routed_experts = enable_return_routed_experts

        E = num_local_experts
        D = hidden_size
        F = intermediate_size_moe

        # Weights are initially unsharded; quant method shards them in
        # process_weights_after_loading via shard_moe_weights.
        self.kernel_gating_EDF = create_param(rngs,
                                              shape=(E, D, F),
                                              dtype=dtype,
                                              random_init=random_init)
        self.kernel_gating_EDF.set_metadata(_weights_to_load=[None] * E)
        self.kernel_up_proj_EDF = create_param(rngs,
                                               shape=(E, D, F),
                                               dtype=dtype,
                                               random_init=random_init)
        self.kernel_up_proj_EDF.set_metadata(_weights_to_load=[None] * E)
        self.kernel_down_proj_EFD = create_param(rngs,
                                                 shape=(E, F, D),
                                                 dtype=dtype,
                                                 random_init=random_init)
        self.kernel_down_proj_EFD.set_metadata(_weights_to_load=[None] * E)

        # Derive use_ep from the vLLM parallel config (same formula as torchax).
        self.use_ep = self._compute_use_ep()
        self.moe_backend = select_moe_backend(self.use_ep)
        # Needed by apply_jax for the input sharding constraint.
        self.activation = hidden_act
        self.activation_ffw_td = (ShardingAxisName.MLP_DATA, None)
        self.num_experts_per_tok = top_k

        if quant_config is None:
            # Imported locally to avoid an import cycle.
            from tpu_inference.layers.jax.quantization.unquantized import \
                UnquantizedConfig
            quant_config = UnquantizedConfig({})
        self.quant_config = quant_config

        if (qm := quant_config.get_quant_method(self, prefix=prefix)):
            assert isinstance(qm, QuantizeMethodBase)
            self.quant_method = qm
            self.quant_method.create_weights_jax(self, rngs=rngs)
        else:
            raise ValueError("Expected quant_method to be set!")

    @staticmethod
    def _compute_use_ep() -> bool:
        # Replicate vLLM logic
        # https://github.com/vllm-project/vllm/blob/36bbecd6436d0dd4c7a27fbb09a787e00534d647/vllm/model_executor/layers/fused_moe/config.py#L1190-L1193
        from vllm.config import get_current_vllm_config
        pc = get_current_vllm_config().parallel_config
        return (pc.data_parallel_size * pc.prefill_context_parallel_size *
                pc.tensor_parallel_size) > 1 and pc.enable_expert_parallel

    def __call__(
        self,
        x_TD: jax.Array,
        router_logits: jax.Array,
    ) -> tuple[jax.Array, Optional[jax.Array]]:
        assert self.quant_method is not None
        x_TD = self.quant_method.apply_jax(self,
                                           x_TD,
                                           router_logits=router_logits)
        if self.enable_return_routed_experts:
            _, selected_experts_TX = jax.lax.top_k(router_logits, self.top_k)
            return x_TD, selected_experts_TX
        return x_TD, None

    def _load_weights(self,
                      weights: Iterable,
                      *,
                      mesh: jax.sharding.Mesh | None = None) -> set:
        """Accumulate per-expert tensors; concatenate and shard when complete."""
        cnt = 0
        for param_name, torch_weight in weights:
            rel_name = param_name.split(self.prefix)[-1]
            names = rel_name.split(".")
            assert len(names) == 3, (
                f"Expected .<expert_id>.<param_name>.weight, got {rel_name}")
            expert_id, param_type, _ = names
            expert_id = int(expert_id)
            if param_type.endswith("up_proj"):
                jax_param = self.kernel_up_proj_EDF
            elif param_type.endswith("down_proj"):
                jax_param = self.kernel_down_proj_EFD
            elif param_type.endswith("gate_proj"):
                jax_param = self.kernel_gating_EDF
            else:
                raise ValueError(f"Unexpected param type in {rel_name}, "
                                 "expected gate_proj, up_proj, or down_proj")
            assert isinstance(jax_param, nnx.Param)
            jax_param._weights_to_load[
                expert_id] = jax_array_from_reshaped_torch(torch_weight,
                                                           reshape_dims=(1, ) +
                                                           torch_weight.shape)
            cnt += 1

        logger.debug(f"Loaded {cnt} weights for {self.prefix} MoE layer.")

        loaded_names = set()
        for name, param in {
                "kernel_gating_EDF": self.kernel_gating_EDF,
                "kernel_up_proj_EDF": self.kernel_up_proj_EDF,
                "kernel_down_proj_EFD": self.kernel_down_proj_EFD,
        }.items():
            if all(w is not None for w in param._weights_to_load):
                with cpu_mesh_context():
                    concatenated = jnp.concatenate(param._weights_to_load,
                                                   axis=0)
                param.value = shard_put(concatenated, param.out_sharding, mesh)
                loaded_names.add(name)
        return loaded_names

    def load_weights(self, weights: Iterable):
        """Used by JaxAutoWeightLoader to load HF weights into the layer."""
        if not hasattr(self.quant_method, "load_weights"):
            return self._load_weights(weights)
        return self.quant_method.load_weights(
            layer=self,
            original_load_weights_fn=self._load_weights,
            weights=weights)
