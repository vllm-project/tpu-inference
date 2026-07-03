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
"""Numerics tests for the JAX native Qwen3-Next model against the HF
transformers implementation. These run on CPU: the fused Pallas conv1d + GDN
kernel is substituted with the pure JAX reference implementations."""

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from transformers.models.qwen3_next.configuration_qwen3_next import \
    Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import \
    Qwen3NextGatedDeltaNet as HFGatedDeltaNet

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.jax import qwen3_next


def tiny_config() -> Qwen3NextConfig:
    return Qwen3NextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        partial_rotary_factor=0.25,
        rope_theta=10000000,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        vocab_size=128,
        rms_norm_eps=1e-6,
        full_attention_interval=4,
        max_position_embeddings=256,
    )


def single_device_mesh() -> jax.sharding.Mesh:
    devices = np.asarray(jax.devices()[:1])
    return jax.sharding.Mesh(devices.reshape((1, 1, 1, 1)),
                             ('data', 'attn_dp', 'model', 'expert'))


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _l2norm(x, eps=1e-6):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)


def reference_gdn_core(mixed_qkv, b, a, conv_state, recurrent_state,
                       conv_weight, conv_bias, a_log, dt_bias, state_indices,
                       query_start_loc, distribution, seq_lens, *, n_kq, n_v,
                       d_k, d_v, kernel_size, mesh):
    """Token by token NumPy substitute for run_jax_gdn_attention, ported
    directly from HF's torch_causal_conv1d_update and
    torch_recurrent_gated_delta_rule (with qk l2 norm and query scaling)."""
    del mesh
    assert conv_bias is None
    mixed = np.asarray(mixed_qkv, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    new_conv_state = np.array(conv_state, dtype=np.float32)
    new_recurrent_state = np.array(recurrent_state, dtype=np.float32)
    w = np.asarray(conv_weight, dtype=np.float32)[:, 0, :]  # (dim, K)
    a_log = np.asarray(a_log, dtype=np.float32)
    dt_bias = np.asarray(dt_bias, dtype=np.float32)
    query_start_loc = np.asarray(query_start_loc)
    seq_lens = np.asarray(seq_lens)
    state_indices = np.asarray(state_indices)

    key_dim = n_kq * d_k
    r = n_v // n_kq
    out = np.zeros((mixed.shape[0], n_v * d_v), dtype=np.float32)
    num_seqs = int(np.asarray(distribution)[2])
    for s in range(num_seqs):
        start, end = int(query_start_loc[s]), int(query_start_loc[s + 1])
        slot = int(state_indices[s])
        state = new_recurrent_state[slot].copy()  # (n_v, d_k, d_v)
        conv_buf = new_conv_state[slot].copy()  # (K - 1, dim)
        if int(seq_lens[s]) - (end - start) == 0:
            state[:] = 0.0
            conv_buf[:] = 0.0
        for t in range(start, end):
            window = np.concatenate([conv_buf, mixed[t][None, :]], axis=0)
            conv_out = (window * w.T).sum(axis=0)
            conv_buf = window[1:]
            h = _silu(conv_out)
            q = h[:key_dim].reshape(n_kq, d_k)
            k = h[key_dim:2 * key_dim].reshape(n_kq, d_k)
            v = h[2 * key_dim:].reshape(n_v, d_v)
            q = np.repeat(q, r, axis=0)
            k = np.repeat(k, r, axis=0)
            q = _l2norm(q) / np.sqrt(d_k)
            k = _l2norm(k)
            beta = 1.0 / (1.0 + np.exp(-b[t]))  # (n_v,)
            g = -np.exp(a_log) * np.log1p(np.exp(a[t] + dt_bias))
            state = state * np.exp(g)[:, None, None]
            kv_mem = (state * k[:, :, None]).sum(axis=1)  # (n_v, d_v)
            delta = (v - kv_mem) * beta[:, None]
            state = state + k[:, :, None] * delta[:, None, :]
            out[t] = (state * q[:, :, None]).sum(axis=1).reshape(-1)
        new_recurrent_state[slot] = state
        new_conv_state[slot] = conv_buf
    return (jnp.asarray(new_conv_state),
            jnp.asarray(new_recurrent_state)), jnp.asarray(out)


def build_gdn_block(cfg, mesh) -> qwen3_next.Qwen3NextGatedDeltaNet:
    with jax.set_mesh(mesh):
        return qwen3_next.Qwen3NextGatedDeltaNet(config=cfg,
                                                 dtype=jnp.float32,
                                                 rng=nnx.Rngs(0),
                                                 mesh=mesh,
                                                 quant_config=None,
                                                 prefix="linear_attn")


def copy_hf_gdn_weights(block, hf):
    block.in_proj_qkvz.weight.value = jnp.asarray(
        hf.in_proj_qkvz.weight.detach().numpy().T)
    block.in_proj_ba.weight.value = jnp.asarray(
        hf.in_proj_ba.weight.detach().numpy().T)
    block.conv1d.weight.value = jnp.asarray(hf.conv1d.weight.detach().numpy())
    block.A_log.value = jnp.asarray(hf.A_log.detach().numpy())
    block.dt_bias.value = jnp.asarray(hf.dt_bias.detach().numpy())
    block.norm.scale.value = jnp.asarray(hf.norm.weight.detach().numpy())
    block.out_proj.weight.value = jnp.asarray(
        hf.out_proj.weight.detach().numpy().T)


def prefill_metadata(seq_len, max_reqs=2):
    query_start_loc = np.zeros(max_reqs + 1, dtype=np.int32)
    query_start_loc[1:] = seq_len
    seq_lens = np.zeros(max_reqs, dtype=np.int32)
    seq_lens[0] = seq_len
    state_indices = np.zeros(max_reqs, dtype=np.int32)
    state_indices[0] = 1
    return AttentionMetadata(
        input_positions=jnp.arange(seq_len),
        seq_lens=jnp.asarray(seq_lens),
        query_start_loc=jnp.asarray(query_start_loc),
        request_distribution=jnp.asarray([0, 1, 1], dtype=jnp.int32),
        mamba_state_indices=jnp.asarray(state_indices),
    )


def decode_metadata(context_len, max_reqs=2):
    query_start_loc = np.zeros(max_reqs + 1, dtype=np.int32)
    query_start_loc[1:] = 1
    seq_lens = np.zeros(max_reqs, dtype=np.int32)
    seq_lens[0] = context_len + 1
    state_indices = np.zeros(max_reqs, dtype=np.int32)
    state_indices[0] = 1
    return AttentionMetadata(
        input_positions=jnp.asarray([context_len]),
        seq_lens=jnp.asarray(seq_lens),
        query_start_loc=jnp.asarray(query_start_loc),
        request_distribution=jnp.asarray([1, 1, 1], dtype=jnp.int32),
        mamba_state_indices=jnp.asarray(state_indices),
    )


def empty_gdn_state(cfg, num_blocks=3):
    conv_dim = (2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim +
                cfg.linear_num_value_heads * cfg.linear_value_head_dim)
    conv_state = jnp.zeros(
        (num_blocks, cfg.linear_conv_kernel_dim - 1, conv_dim),
        dtype=jnp.float32)
    recurrent_state = jnp.zeros(
        (num_blocks, cfg.linear_num_value_heads, cfg.linear_key_head_dim,
         cfg.linear_value_head_dim),
        dtype=jnp.float32)
    return conv_state, recurrent_state


class TestGatedDeltaNet:

    def test_prefill_matches_hf(self, monkeypatch):
        monkeypatch.setattr(qwen3_next, "_gdn_core", reference_gdn_core)
        cfg = tiny_config()
        torch.manual_seed(0)
        hf = HFGatedDeltaNet(cfg, layer_idx=0).float().eval()
        mesh = single_device_mesh()
        block = build_gdn_block(cfg, mesh)
        copy_hf_gdn_weights(block, hf)

        seq_len = 6
        x = torch.randn(1, seq_len, cfg.hidden_size)
        with torch.no_grad():
            want = hf(hidden_states=x)

        state = empty_gdn_state(cfg)
        _, got = block(state, jnp.asarray(x[0].numpy()),
                       prefill_metadata(seq_len))
        np.testing.assert_allclose(np.asarray(got),
                                   want[0].numpy(),
                                   rtol=2e-3,
                                   atol=2e-3)

    def test_decode_matches_hf_prefill_tail(self, monkeypatch):
        """Runs prefill for T tokens then one decode step and checks the
        decode output equals HF's output for position T when HF processes
        all T+1 tokens in one pass. This exercises the recurrent and conv
        state carry between calls."""
        monkeypatch.setattr(qwen3_next, "_gdn_core", reference_gdn_core)
        cfg = tiny_config()
        torch.manual_seed(1)
        hf = HFGatedDeltaNet(cfg, layer_idx=0).float().eval()
        mesh = single_device_mesh()
        block = build_gdn_block(cfg, mesh)
        copy_hf_gdn_weights(block, hf)

        seq_len = 6
        x = torch.randn(1, seq_len + 1, cfg.hidden_size)
        with torch.no_grad():
            want_full = hf(hidden_states=x)

        state = empty_gdn_state(cfg)
        state, _ = block(state, jnp.asarray(x[0, :seq_len].numpy()),
                         prefill_metadata(seq_len))
        _, got_last = block(state, jnp.asarray(x[0, seq_len:].numpy()),
                            decode_metadata(seq_len))
        np.testing.assert_allclose(np.asarray(got_last),
                                   want_full[0, seq_len:].numpy(),
                                   rtol=2e-3,
                                   atol=2e-3)
