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

import jax
import jax.numpy as jnp
import qwix
import qwix.pallas as qpl
from flax import nnx
from jax.sharding import Mesh, PartitionSpec

from tpu_inference import envs
from tpu_inference.kernels.fused_rmsnorm_fp8 import kernel as fused_rmsnorm_fp8
from tpu_inference.layers.common import sharding as sharding_lib
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig


def _get_token_partition_spec(mesh: Mesh | None,
                              ndim: int = 2) -> PartitionSpec:
    """Returns PartitionSpec(token_axes, None, ...) matching ATTN_DATA on mesh."""
    if mesh is None or ndim < 2:
        return PartitionSpec(*((None, ) * max(ndim, 1)))
    attn_data = sharding_lib.ShardingAxisName.ATTN_DATA
    raw_axes = (attn_data, ) if isinstance(attn_data,
                                           str) else tuple(attn_data)
    valid_axes = tuple(ax for ax in raw_axes if ax in mesh.axis_names)
    if not valid_axes:
        return PartitionSpec(*((None, ) * ndim))
    first_axis = valid_axes[0] if len(valid_axes) == 1 else valid_axes
    return PartitionSpec(first_axis, *((None, ) * (ndim - 1)))


def sharded_fused_rmsnorm_fp8_quant(
    x: jax.Array,
    weight: jax.Array,
    *,
    epsilon: float = 1e-6,
    residual: jax.Array | None = None,
    return_residual: bool = False,
    preferred_block_m: int | None = None,
    block_k: int | None = None,
    mesh: Mesh | None = None,
) -> qwix.QArray | tuple[qwix.QArray, jax.Array]:
    """Executes fused RMSNorm (+ optional residual) + FP8 dynamic quant across mesh."""
    if preferred_block_m is None:
        preferred_block_m = envs.FUSED_RMSNORM_BLOCK_M

    if mesh is None:
        sharding = getattr(x, "sharding", None)
        if isinstance(sharding, jax.sharding.NamedSharding) and isinstance(
                sharding.mesh, Mesh) and not sharding.mesh.empty:
            mesh = sharding.mesh

    def _run_local(
        x_local: jax.Array,
        w_local: jax.Array,
        res_local: jax.Array | None = None,
    ):
        return fused_rmsnorm_fp8.fused_rmsnorm_fp8(
            x_local,
            w_local,
            residual=res_local,
            eps=epsilon,
            preferred_block_m=preferred_block_m,
            block_k=block_k,
            return_residual=return_residual,
        )

    if mesh is None or x.ndim < 2:
        return _run_local(x, weight, residual)

    def _run_local_tuples(
        x_local: jax.Array,
        w_local: jax.Array,
        res_local: jax.Array | None = None,
    ):
        local_out = _run_local(x_local, w_local, res_local)
        if return_residual:
            q_arr, updated_res = local_out
            return q_arr.qvalue, q_arr.scale, updated_res
        return local_out.qvalue, local_out.scale

    token_spec = _get_token_partition_spec(mesh, x.ndim)
    weight_spec = PartitionSpec(None)
    out_specs = ((token_spec, token_spec, token_spec) if return_residual else
                 (token_spec, token_spec))

    if residual is None:
        sharded_fn = jax.shard_map(
            lambda x_l, w_l: _run_local_tuples(x_l, w_l, None),
            mesh=mesh,
            in_specs=(token_spec, weight_spec),
            out_specs=out_specs,
            check_vma=False,
        )
        res = sharded_fn(x, weight)
    else:
        sharded_fn = jax.shard_map(
            _run_local_tuples,
            mesh=mesh,
            in_specs=(token_spec, weight_spec, token_spec),
            out_specs=out_specs,
            check_vma=False,
        )
        res = sharded_fn(x, weight, residual)

    if return_residual:
        out_q, out_scale, updated_res = res
        return qwix.QArray(qvalue=out_q, scale=out_scale), updated_res
    out_q, out_scale = res
    return qwix.QArray(qvalue=out_q, scale=out_scale)


class JaxRmsNorm(nnx.RMSNorm, JaxModule):
    """RmsNorm layer for JAX."""

    def __init__(self,
                 *args,
                 quant_config: QuantizationConfig | None = None,
                 prefix: str = "",
                 fuse_fp8_quant: bool = False,
                 **kwargs):
        # nnx.RMSNorm uses `param_dtype` for parameter initialization dtype.
        # Accept `dtype` as an alias for backward compatibility, but forward it
        # as `param_dtype` so that weights are created with the correct dtype.
        if "dtype" in kwargs and "param_dtype" not in kwargs:
            kwargs["param_dtype"] = kwargs.pop("dtype")
        nnx.RMSNorm.__init__(self, *args, **kwargs)
        # For compatibility. HF model use 'weight' as name suffix, we alias `self.scale` to
        # `self.weight` such that `named_parameters()` can match the names in HF models. We also
        # apply transpose here to match HF weight layout.
        self.weight = self.scale
        delattr(self, 'scale')

        self.prefix = prefix
        self.fuse_fp8_quant = fuse_fp8_quant
        self.mesh = getattr(quant_config, "mesh", None)
        self.quant_method = None
        if quant_config is not None:
            quant_method = quant_config.get_quant_method(self, prefix=prefix)
            if quant_method is not None:
                assert isinstance(quant_method, QuantizeMethodBase)
                self.quant_method = quant_method
                quant_method.create_weights_jax(self)

    def __getattr__(self, name: str):
        if name == "scale":
            # nnx.RMSNorm needs to access self.scale
            return self.weight

    def _should_fuse_fp8_quant(self) -> bool:
        if getattr(self, "disable_quant_stats_update", False):
            return False
        if self.fuse_fp8_quant:
            return True
        if not envs.USE_FUSED_RMSNORM_QUANT:
            return False
        if not self.prefix.endswith(
            ("input_layernorm", "post_attention_layernorm")):
            return False
        rule = qpl.get_current_rule("einsum")
        if rule is None or rule.act_qtype is None or rule.act_static_scale:
            return False
        return jnp.dtype(rule.act_qtype) == jnp.float8_e4m3fn

    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None = None,
    ) -> jax.Array | qwix.QArray:
        if self.quant_method is None:
            if mask is None and self._should_fuse_fp8_quant():
                return sharded_fused_rmsnorm_fp8_quant(
                    x,
                    self.weight.value,
                    epsilon=self.epsilon,
                    mesh=self.mesh,
                )
            return nnx.RMSNorm.__call__(self, x, mask=mask)
        return self.quant_method.apply_jax(self, x, mask=mask)


class JaxLayerNorm(nnx.LayerNorm, JaxModule):
    """LayerNorm layer that inherits from JaxModule and maps scale to weight for compatibility."""

    def __init__(self, *args, **kwargs):
        # nnx.LayerNorm uses `param_dtype` for parameter initialization dtype.
        # Accept `dtype` as an alias for backward compatibility.
        if "dtype" in kwargs and "param_dtype" not in kwargs:
            kwargs["param_dtype"] = kwargs.pop("dtype")
        nnx.LayerNorm.__init__(self, *args, **kwargs)

        # For compatibility, alias scale to weight
        self.weight = self.scale
        delattr(self, 'scale')

    def __getattr__(self, name: str):
        if name == "scale":
            return self.weight
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'")
