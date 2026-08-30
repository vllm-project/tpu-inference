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
"""TPU lowering for vLLM's Mamba2 mixer (MambaMixer2).

vLLM's MambaMixer2.forward calls `torch.ops.vllm.mamba_mixer2`, a custom op with
no torchax lowering. We OOT-override the layer (same mechanism as the GDN op) so
its forward runs conv1d + SSD in JAX. Importing this module registers the
override; it is imported from `custom_ops/__init__.py`.

Used by hybrid Mamba2-Transformer-MoE models such as Nemotron-H.
"""
import jax
import jax.numpy as jnp
import torch
from torchax.interop import jax_view, torch_view
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2

from tpu_inference.layers.common.mamba2_ssd import run_jax_mamba2
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context
from tpu_inference.utils import get_mesh_shape_product

logger = init_logger(__name__)


def mamba2_core_tpu(mixed_xBC: torch.Tensor, dt: torch.Tensor,
                    output: torch.Tensor, layer_name: str,
                    mesh: jax.sharding.Mesh) -> None:
    """Bridge between PyTorch (torchax) and JAX for the Mamba2 conv1d + SSD core.

    Reads the ragged mamba metadata and the paged conv/ssm caches, runs conv1d +
    the SSD scan under shard_map, writes the caches back, and copies the result
    into `output` (an in-place update, mirroring the GDN core op).
    """
    fc = get_forward_context()
    md = fc.attn_metadata[layer_name]
    layer = fc.no_compile_layers[layer_name]
    vctx = get_vllm_model_wrapper_context()

    tp = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)
    n_heads = layer.num_heads // tp
    head_dim = layer.head_dim
    n_groups = layer.n_groups // tp
    ssm_n = layer.ssm_state_size
    kernel_size = layer.conv_kernel_size
    inter = layer.intermediate_size  # full
    groups_ssm = layer.groups_ssm_state_size  # full
    conv_dim = inter + 2 * groups_ssm

    # Split [x | B | C] at the x/B/C boundaries and feed them to the SSD as
    # SEPARATE shard_map args. The concatenated mixed_xBC has layout
    # [x_full | B_full | C_full] where x is head-sharded but B/C are group-
    # sharded; handing the concat to one P(ATTN_HEAD) split would slice it into
    # contiguous chunks that cross the x/B/C boundaries, so a shard's x-heads and
    # B/C-groups would belong to different logical quarters. Splitting first
    # (each part contiguously shardable) and re-concatenating per shard inside
    # `_run_local` keeps each shard's heads and groups aligned. Split by a
    # fraction of each tensor's own dim so it works whether the dim is logical-
    # full (conv_dim) or per-shard (conv_dim / tp).
    def _split3(t, axis):
        d = t.shape[axis]
        n1 = d * inter // conv_dim
        n2 = d * groups_ssm // conv_dim
        sl = [slice(None)] * t.ndim

        def part(a, b):
            sl[axis] = slice(a, b)
            return t[tuple(sl)]

        return part(0, n1), part(n1, n1 + n2), part(n1 + n2, d)

    xb, bb, cb_xbc = _split3(mixed_xBC, -1)
    x_l, B_l, C_l = jax_view(xb), jax_view(bb), jax_view(cb_xbc)
    j_dt = jax_view(dt)
    cwx, cwb, cwc = _split3(layer.conv1d.weight, 0)  # (conv_dim/tp, 1, kernel)
    cw_x, cw_B, cw_C = jax_view(cwx), jax_view(cwb), jax_view(cwc)
    if layer.conv1d.bias is not None:
        cbx, cbb, cbc = _split3(layer.conv1d.bias, 0)
        cb_x, cb_B, cb_C = jax_view(cbx), jax_view(cbb), jax_view(cbc)
    else:
        cb_x = cb_B = cb_C = None
    j_A = jax_view(layer.A)  # already -exp(A_log)
    j_dt_bias = jax_view(layer.dt_bias)
    j_D = jax_view(layer.D)

    layer_idx = vctx.layer_name_to_kvcache_index[layer_name]
    conv_state, ssm_state = vctx.kv_caches[layer_idx]

    pad = md.padded_num_reqs
    dp = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_DATA)
    state_indices = md.mamba_state_indices.astype(jnp.int32)[:pad]
    qsl = md.query_start_loc[:pad + dp]
    seq_lens = md.seq_lens[:pad]

    (new_conv, new_ssm), y = run_jax_mamba2(x_l,
                                            B_l,
                                            C_l,
                                            j_dt,
                                            conv_state,
                                            ssm_state,
                                            cw_x,
                                            cw_B,
                                            cw_C,
                                            cb_x,
                                            cb_B,
                                            cb_C,
                                            j_A,
                                            j_dt_bias,
                                            j_D,
                                            qsl,
                                            state_indices,
                                            md.request_distribution,
                                            seq_lens,
                                            n_groups=n_groups,
                                            n_heads=n_heads,
                                            head_dim=head_dim,
                                            ssm_n=ssm_n,
                                            kernel_size=kernel_size,
                                            mesh=mesh)

    vctx.kv_caches[layer_idx] = (new_conv, new_ssm)
    output.copy_(torch_view(y.reshape(output.shape)))


@MambaMixer2.register_oot
class VllmMambaMixer2(MambaMixer2):

    def forward(self, hidden_states: torch.Tensor,
                mup_vector: torch.Tensor | None = None) -> torch.Tensor:
        mesh = get_vllm_model_wrapper_context().mesh

        # 1. in_proj, then split [gate | x|B|C | dt] (per-shard local sizes).
        projected_states, _ = self.in_proj(hidden_states)
        if mup_vector is not None:
            projected_states = projected_states * mup_vector
        gate = projected_states[..., :self.tped_intermediate_size]
        rest = projected_states[..., self.tped_intermediate_size:]
        mixed_xBC = rest[..., :self.tped_conv_size]
        dt = rest[..., self.tped_conv_size:]

        # 2. conv1d + SSD core (custom op).
        ssm_out = torch.zeros(
            (hidden_states.shape[0], self.tped_intermediate_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device)
        mamba2_core_tpu(mixed_xBC, dt, ssm_out, self.prefix, mesh)

        # 3. gated RMSNorm (applies SiLU to the gate internally) + out_proj.
        hidden_states = self.norm(ssm_out, gate)
        out, _ = self.out_proj(hidden_states)
        return out


logger.info_once("Registered TPU OOT MambaMixer2 (Mamba2 mixer JAX lowering)")
