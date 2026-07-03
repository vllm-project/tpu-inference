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
"""JAX native implementation of Qwen3-Next (hybrid gated delta net linear
attention + gated full attention, sparse MoE with a shared expert)."""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.gdn_attention import run_jax_gdn_attention
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import \
    reorder_concatenated_tensor_for_sharding
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()

# Indirection over the fused Pallas conv1d + GDN kernel so CPU tests can
# substitute a pure JAX reference implementation.
_gdn_core = run_jax_gdn_attention


class Conv1dWeight(JaxModule):
    """Holds the depthwise causal conv weight under a `weight` param so its
    parameter name lines up with the HF checkpoint (`...conv1d.weight`)."""

    def __init__(self, conv_dim: int, kernel_size: int, dtype: jnp.dtype,
                 rng: nnx.Rngs):
        self.weight = create_param(rng, (conv_dim, 1, kernel_size),
                                   ("model", None, None), dtype)


class Qwen3NextGatedDeltaNet(JaxModule):
    """Gated delta net linear attention block.

    The projections, unpacking, output norm and gate run as plain JAX ops;
    the depthwise causal conv1d and the gated delta rule scan run in the
    fused GDN kernel, which also reads and writes the per request
    (conv_state, recurrent_state) cache slots."""

    def __init__(self,
                 config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 quant_config,
                 prefix: str = ""):
        self.mesh = mesh
        self.n_kq = config.linear_num_key_heads
        self.n_v = config.linear_num_value_heads
        self.d_k = config.linear_key_head_dim
        self.d_v = config.linear_value_head_dim
        self.kernel_size = config.linear_conv_kernel_dim
        self.key_dim = self.n_kq * self.d_k
        self.value_dim = self.n_v * self.d_v
        self.conv_dim = 2 * self.key_dim + self.value_dim
        hidden_size = config.hidden_size

        self.in_proj_qkvz = JaxEinsum(
            "TD,DP->TP",
            (hidden_size, 2 * self.key_dim + 2 * self.value_dim),
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".in_proj_qkvz",
        )
        self.in_proj_ba = JaxEinsum(
            "TD,DP->TP",
            (hidden_size, 2 * self.n_v),
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".in_proj_ba",
        )
        self.conv1d = Conv1dWeight(self.conv_dim, self.kernel_size, dtype, rng)
        self.A_log = create_param(rng, (self.n_v, ), ("model", ), jnp.float32)
        self.dt_bias = create_param(rng, (self.n_v, ), ("model", ),
                                    jnp.float32)
        # Gated RMSNorm over d_v; standard scale (not the zero centered
        # variant the rest of the model uses). The silu(z) gate is applied
        # outside, matching HF's norm-before-gate ordering.
        self.norm = JaxRmsNorm(
            self.d_v,
            epsilon=config.rms_norm_eps,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".norm",
        )
        self.out_proj = JaxEinsum(
            "TP,PD->TD",
            (self.value_dim, hidden_size),
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".out_proj",
        )

    def _split_qkvz_ba(self, qkvz: jax.Array, ba: jax.Array):
        """Unpacks the interleaved per key head projection layout, matching
        HF's fix_query_key_value_ordering."""
        t = qkvz.shape[0]
        r = self.n_v // self.n_kq
        qkvz = qkvz.reshape(t, self.n_kq, 2 * self.d_k + 2 * r * self.d_v)
        q, k, v, z = jnp.split(
            qkvz, [self.d_k, 2 * self.d_k, 2 * self.d_k + r * self.d_v],
            axis=-1)
        v = v.reshape(t, self.n_v, self.d_v)
        z = z.reshape(t, self.n_v, self.d_v)
        ba = ba.reshape(t, self.n_kq, 2 * r)
        b, a = jnp.split(ba, [r], axis=-1)
        b = b.reshape(t, self.n_v)
        a = a.reshape(t, self.n_v)
        return q, k, v, z, b, a

    def __call__(
        self,
        kv_cache: Tuple[jax.Array, jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[Tuple[jax.Array, jax.Array], jax.Array]:
        md = attention_metadata
        t = x.shape[0]
        q, k, v, z, b, a = self._split_qkvz_ba(self.in_proj_qkvz(x),
                                               self.in_proj_ba(x))
        mixed_qkv = jnp.concatenate([
            q.reshape(t, self.key_dim),
            k.reshape(t, self.key_dim),
            v.reshape(t, self.value_dim),
        ],
                                    axis=-1)

        # The kernel runs per TP shard; interleave the blocked [Q|K|V]
        # layout so each shard receives its own contiguous [Q|K|V] slice.
        tp_size = get_mesh_shape_product(self.mesh, ShardingAxisName.ATTN_HEAD)
        split_sizes = [self.key_dim, self.key_dim, self.value_dim]
        if tp_size > 1:
            mixed_qkv = reorder_concatenated_tensor_for_sharding(
                mixed_qkv, split_sizes, tp_size, 1)
            conv_weight = reorder_concatenated_tensor_for_sharding(
                self.conv1d.weight.value, split_sizes, tp_size, 0)
        else:
            conv_weight = self.conv1d.weight.value

        conv_state, recurrent_state = kv_cache
        # With speculative decoding the conv cache carries extra rows; the
        # kernel consumes exactly kernel_size - 1.
        conv_state_tail = None
        if conv_state.shape[1] > self.kernel_size - 1:
            conv_state_tail = conv_state[:, self.kernel_size - 1:, :]
            conv_state = conv_state[:, :self.kernel_size - 1, :]

        (new_conv_state, new_recurrent_state), core_out = _gdn_core(
            mixed_qkv,
            b,
            a,
            conv_state,
            recurrent_state,
            conv_weight,
            None,
            self.A_log.value,
            self.dt_bias.value,
            md.mamba_state_indices,
            md.query_start_loc,
            md.request_distribution,
            md.seq_lens,
            n_kq=self.n_kq,
            n_v=self.n_v,
            d_k=self.d_k,
            d_v=self.d_v,
            kernel_size=self.kernel_size,
            mesh=self.mesh,
        )

        if conv_state_tail is not None:
            new_conv_state = jnp.concatenate([new_conv_state, conv_state_tail],
                                             axis=1)

        core_out = core_out.reshape(t, self.n_v, self.d_v)
        gated = self.norm(core_out.astype(jnp.float32)) * jax.nn.silu(
            z.astype(jnp.float32))
        out = self.out_proj(gated.reshape(t, self.value_dim).astype(x.dtype))
        return (new_conv_state, new_recurrent_state), out
