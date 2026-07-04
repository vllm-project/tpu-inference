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
import pytest
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
        # 128 so get_padded_head_dim is the identity and numerics line up
        # with HF without padding effects.
        head_dim=128,
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
    # Axis order matches the `mesh` fixture in conftest.py so the block-level
    # tests and the checkpoint round-trip test (which uses the fixture)
    # exercise the same layout.
    devices = np.asarray(jax.devices()[:1])
    return jax.sharding.Mesh(devices.reshape((1, 1, 1, 1)),
                             ('data', 'attn_dp', 'expert', 'model'))


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


def fake_attention(kv_cache, q, k, v, attention_metadata, mesh,
                   head_dim_original, **kwargs):
    """NumPy causal softmax attention with GQA, standing in for the ragged
    paged attention TPU kernel."""
    q_np = np.asarray(q, dtype=np.float32)
    k_np = np.asarray(k, dtype=np.float32)
    v_np = np.asarray(v, dtype=np.float32)
    seq_len, num_heads, _ = q_np.shape
    rep = num_heads // k_np.shape[1]
    k_np = np.repeat(k_np, rep, axis=1)
    v_np = np.repeat(v_np, rep, axis=1)
    scores = np.einsum('tnh,snh->nts', q_np, k_np)
    scores = scores * (head_dim_original**-0.5)
    causal = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    scores = np.where(causal[None, :, :], -np.inf, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    out = np.einsum('nts,snh->tnh', probs, v_np)
    return kv_cache, jnp.asarray(out)


class TestFullAttention:

    def _build(self, cfg, mesh):
        with jax.set_mesh(mesh):
            return qwen3_next.Qwen3NextAttention(config=cfg,
                                                 dtype=jnp.float32,
                                                 rng=nnx.Rngs(0),
                                                 mesh=mesh,
                                                 kv_cache_dtype="auto",
                                                 quant_config=None,
                                                 prefix="self_attn")

    def _copy_weights(self, block, hf, cfg):
        num_heads = cfg.num_attention_heads
        head_dim = cfg.head_dim
        hidden = cfg.hidden_size
        num_kv = cfg.num_key_value_heads
        # HF q_proj weight is (num_heads * 2 * head_dim, hidden) with the
        # query and gate halves fused per head.
        q_w = hf.q_proj.weight.detach().numpy().reshape(
            num_heads, 2 * head_dim, hidden)
        block.q_proj.weight.value = jnp.asarray(q_w.transpose(2, 0, 1))
        k_w = hf.k_proj.weight.detach().numpy().reshape(
            num_kv, head_dim, hidden)
        block.k_proj.weight.value = jnp.asarray(k_w.transpose(2, 0, 1))
        v_w = hf.v_proj.weight.detach().numpy().reshape(
            num_kv, head_dim, hidden)
        block.v_proj.weight.value = jnp.asarray(v_w.transpose(2, 0, 1))
        o_w = hf.o_proj.weight.detach().numpy().reshape(
            hidden, num_heads, head_dim)
        block.o_proj.weight.value = jnp.asarray(o_w.transpose(1, 2, 0))
        # Zero centered norms: fold the +1 into the scale.
        block.q_norm.scale.value = jnp.asarray(
            hf.q_norm.weight.detach().numpy()) + 1.0
        block.k_norm.scale.value = jnp.asarray(
            hf.k_norm.weight.detach().numpy()) + 1.0

    def test_matches_hf(self, monkeypatch):
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            Qwen3NextAttention, Qwen3NextRotaryEmbedding)
        monkeypatch.setattr(qwen3_next, "attention", fake_attention)
        cfg = tiny_config()
        cfg._attn_implementation = "eager"
        torch.manual_seed(2)
        hf = Qwen3NextAttention(cfg, layer_idx=3).float().eval()
        mesh = single_device_mesh()
        block = self._build(cfg, mesh)
        self._copy_weights(block, hf, cfg)

        seq_len = 5
        x = torch.randn(1, seq_len, cfg.hidden_size)
        rotary = Qwen3NextRotaryEmbedding(cfg)
        position_ids = torch.arange(seq_len)[None]
        cos, sin = rotary(x, position_ids)
        causal_mask = torch.full((1, 1, seq_len, seq_len),
                                 torch.finfo(torch.float32).min).triu(1)
        with torch.no_grad():
            want, _ = hf(x,
                         position_embeddings=(cos, sin),
                         attention_mask=causal_mask)

        _, got = block(None, jnp.asarray(x[0].numpy()),
                       prefill_metadata(seq_len))
        np.testing.assert_allclose(np.asarray(got),
                                   want[0].numpy(),
                                   rtol=2e-3,
                                   atol=2e-3)


class TestSparseMoeBlock:

    def _build(self, cfg, mesh):
        from unittest.mock import MagicMock

        from tpu_inference.layers.jax.quantization.unquantized import \
            UnquantizedConfig
        vllm_config = MagicMock()
        vllm_config.model_config.hf_text_config = cfg
        vllm_config.model_config.dtype = jnp.float32
        vllm_config.quant_config = UnquantizedConfig({})
        with jax.set_mesh(mesh):
            return qwen3_next.Qwen3NextSparseMoeBlock(vllm_config=vllm_config,
                                                      rng=nnx.Rngs(0),
                                                      mesh=mesh,
                                                      prefix="mlp")

    def _copy_weights(self, block, hf):
        block.gate.weight.value = jnp.asarray(
            hf.gate.weight.detach().numpy().T)
        # HF stores fused 3D expert tensors: gate_up_proj (E, 2F, D) with
        # the gate rows first, down_proj (E, D, F).
        gate_up = hf.experts.gate_up_proj.detach().numpy()
        intermediate = gate_up.shape[1] // 2
        block.experts.kernel_gating_EDF.value = jnp.asarray(
            gate_up[:, :intermediate, :].transpose(0, 2, 1))
        block.experts.kernel_up_proj_EDF.value = jnp.asarray(
            gate_up[:, intermediate:, :].transpose(0, 2, 1))
        block.experts.kernel_down_proj_EFD.value = jnp.asarray(
            hf.experts.down_proj.detach().numpy().transpose(0, 2, 1))
        block.shared_expert.gate_proj.weight.value = jnp.asarray(
            hf.shared_expert.gate_proj.weight.detach().numpy().T)
        block.shared_expert.up_proj.weight.value = jnp.asarray(
            hf.shared_expert.up_proj.weight.detach().numpy().T)
        block.shared_expert.down_proj.weight.value = jnp.asarray(
            hf.shared_expert.down_proj.weight.detach().numpy().T)
        block.shared_expert_gate.weight.value = jnp.asarray(
            hf.shared_expert_gate.weight.detach().numpy().T)

    def test_matches_hf(self, monkeypatch):
        from transformers.models.qwen3_next.modeling_qwen3_next import \
            Qwen3NextSparseMoeBlock
        monkeypatch.setenv("USE_DENSE_MOE", "1")
        cfg = tiny_config()
        torch.manual_seed(3)
        hf = Qwen3NextSparseMoeBlock(cfg).float().eval()
        # The fused expert tensors are created with torch.empty and only
        # initialized through from_pretrained; fill them explicitly.
        with torch.no_grad():
            hf.experts.gate_up_proj.normal_(0.0, 0.05)
            hf.experts.down_proj.normal_(0.0, 0.05)
            hf.gate.weight.normal_(0.0, 0.05)
        mesh = single_device_mesh()
        block = self._build(cfg, mesh)
        self._copy_weights(block, hf)

        x = torch.randn(6, cfg.hidden_size)
        with torch.no_grad():
            want = hf(x[None])

        with jax.set_mesh(mesh):
            got, expert_ids = block(jnp.asarray(x.numpy()))
        assert expert_ids is not None
        np.testing.assert_allclose(np.asarray(got),
                                   want[0].numpy(),
                                   rtol=2e-3,
                                   atol=2e-3)

    def test_fused_expert_weights_load(self, monkeypatch):
        """The recursive loader hands the experts module relative names, so
        a fused checkpoint arrives as bare "gate_up_proj" / "down_proj".
        transformers 5.x stores experts this way, so drive that path
        directly (save_pretrained in the round-trip test emits per-expert
        tensors and never exercises it)."""
        monkeypatch.setenv("USE_DENSE_MOE", "1")
        cfg = tiny_config()
        mesh = single_device_mesh()
        block = self._build(cfg, mesh)
        experts = block.experts

        e = cfg.num_experts
        d = cfg.hidden_size
        f = cfg.moe_intermediate_size
        rs = np.random.RandomState(0)
        gate_up = rs.standard_normal((e, 2 * f, d)).astype(np.float32)
        down = rs.standard_normal((e, d, f)).astype(np.float32)

        with jax.set_mesh(mesh):
            loaded = experts._load_weights([
                ("gate_up_proj", torch.from_numpy(gate_up)),
                ("down_proj", torch.from_numpy(down)),
            ])

        assert loaded == {
            "kernel_gating_EDF", "kernel_up_proj_EDF", "kernel_down_proj_EFD"
        }
        # Dense backend orients kernels as (E, D, F) / (E, F, D).
        np.testing.assert_allclose(np.asarray(experts.kernel_gating_EDF.value),
                                   gate_up[:, :f, :].transpose(0, 2, 1),
                                   rtol=1e-6,
                                   atol=1e-6)
        np.testing.assert_allclose(np.asarray(
            experts.kernel_up_proj_EDF.value),
                                   gate_up[:, f:, :].transpose(0, 2, 1),
                                   rtol=1e-6,
                                   atol=1e-6)
        np.testing.assert_allclose(np.asarray(
            experts.kernel_down_proj_EFD.value),
                                   down.transpose(0, 2, 1),
                                   rtol=1e-6,
                                   atol=1e-6)


class TestCheckpointRoundTrip:

    def test_tiny_checkpoint_load_and_forward(self, tmp_path, monkeypatch, rng,
                                              mesh, mock_vllm_config):
        """Saves a tiny random HF Qwen3-Next checkpoint, loads it through
        the real vLLM loader path into the JAX model and compares logits
        for a short prefill against HF transformers. This exercises every
        weight mapping rule (zero centered norm folding, doubled q_proj,
        GDN projections, fused 3D experts, shared expert, lm_head)."""
        import jax as jax_module
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader import get_model_loader

        from tpu_inference.distributed.jax_parallel_state import \
            init_pp_distributed_environment
        from tpu_inference.layers.jax.quantization import \
            get_tpu_quantization_config
        from tpu_inference.models.jax.qwen3_next import Qwen3NextForCausalLM

        monkeypatch.setattr(qwen3_next, "_gdn_core", reference_gdn_core)
        monkeypatch.setattr(qwen3_next, "attention", fake_attention)
        monkeypatch.setenv("USE_DENSE_MOE", "1")

        from transformers.models.qwen3_next.modeling_qwen3_next import \
            Qwen3NextForCausalLM as HFModel
        cfg = tiny_config()
        cfg._attn_implementation = "eager"
        torch.manual_seed(4)
        hf = HFModel(cfg).float().eval()
        with torch.no_grad():
            for layer in hf.model.layers:
                layer.mlp.experts.gate_up_proj.normal_(0.0, 0.05)
                layer.mlp.experts.down_proj.normal_(0.0, 0.05)
                layer.mlp.gate.weight.normal_(0.0, 0.05)
        hf.save_pretrained(tmp_path / "ckpt")

        from unittest.mock import MagicMock
        vllm_config = mock_vllm_config(str(tmp_path / "ckpt"), "auto")
        vllm_config.model_config.dtype = jnp.float32
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.enable_expert_parallel = False
        vllm_config.parallel_config.enable_ep_weight_filter = False
        init_pp_distributed_environment(
            ip="",
            rank=0,
            world_size=1,
            device=jax_module.devices()[0],
            need_pp=False,
        )
        vllm_config.quant_config = get_tpu_quantization_config(vllm_config)

        with jax_module.set_mesh(mesh):
            model = Qwen3NextForCausalLM(vllm_config, rng, mesh)
            loader = get_model_loader(vllm_config.load_config)
            with set_current_vllm_config(vllm_config):
                loader.load_weights(model, vllm_config.model_config)

        seq_len = 5
        input_ids = torch.arange(seq_len)[None] % cfg.vocab_size
        with torch.no_grad():
            want_logits = hf(input_ids).logits[0]

        kv_caches = []
        for layer_type in cfg.layer_types:
            if layer_type == "linear_attention":
                kv_caches.append(empty_gdn_state(cfg))
            else:
                kv_caches.append(None)

        with jax_module.set_mesh(mesh):
            _, hidden, _, _ = model(
                kv_caches,
                jnp.asarray(input_ids[0].numpy()),
                prefill_metadata(seq_len),
            )
            got_logits = model.compute_logits(hidden)
        got = np.asarray(got_logits)[:, :cfg.vocab_size]
        np.testing.assert_allclose(got,
                                   want_logits.numpy(),
                                   rtol=5e-3,
                                   atol=5e-3)


@pytest.mark.skipif(
    jax.default_backend() != "tpu",
    reason="exercises the fused GDN and ragged paged attention TPU kernels")
class TestTpuSmoke:

    def test_forward_with_real_kernels(self, rng, mesh, mock_vllm_config):
        """Builds a shrunken Qwen3-Next (4 layers: 3 GDN + 1 full attention)
        with dummy weights and runs one prefill through the real kernels,
        checking output shapes and that the GDN states were written."""
        from unittest.mock import MagicMock

        from tpu_inference.distributed.jax_parallel_state import \
            init_pp_distributed_environment
        from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
            get_kv_cache_shape
        from tpu_inference.layers.jax.quantization import \
            get_tpu_quantization_config
        from tpu_inference.models.jax.qwen3_next import Qwen3NextForCausalLM

        vllm_config = mock_vllm_config("Qwen/Qwen3-Next-80B-A3B-Instruct",
                                       "auto")
        hf_config = vllm_config.model_config.hf_config
        hf_config.num_hidden_layers = 4
        hf_config.layer_types = hf_config.layer_types[:4]
        hf_config.num_experts = 16
        vllm_config.load_config.load_format = "jax_dummy"
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.enable_expert_parallel = False
        init_pp_distributed_environment(
            ip="",
            rank=0,
            world_size=1,
            device=jax.devices()[0],
            need_pp=False,
        )
        vllm_config.quant_config = get_tpu_quantization_config(vllm_config)

        with jax.set_mesh(mesh):
            model = Qwen3NextForCausalLM(vllm_config, rng, mesh)
            # The dummy loader fills every param and runs the per module
            # post processing that fuses the MoE kernels for the GMM
            # backend.
            from vllm.config import set_current_vllm_config
            from vllm.model_executor.model_loader import get_model_loader
            loader = get_model_loader(vllm_config.load_config)
            with set_current_vllm_config(vllm_config):
                loader.load_weights(model, vllm_config.model_config)

        seq_len = 16
        max_reqs = 8
        num_blocks = 8
        block_size = 32
        conv_dim = (
            2 * hf_config.linear_num_key_heads * hf_config.linear_key_head_dim
            +
            hf_config.linear_num_value_heads * hf_config.linear_value_head_dim)
        attn_cache_shape = get_kv_cache_shape(num_blocks, block_size,
                                              hf_config.num_key_value_heads,
                                              hf_config.head_dim, jnp.bfloat16)

        kv_caches = []
        for layer_type in hf_config.layer_types:
            if layer_type == "linear_attention":
                kv_caches.append((
                    jnp.zeros((max_reqs + 1,
                               hf_config.linear_conv_kernel_dim - 1, conv_dim),
                              dtype=jnp.bfloat16),
                    jnp.zeros((max_reqs + 1, hf_config.linear_num_value_heads,
                               hf_config.linear_key_head_dim,
                               hf_config.linear_value_head_dim),
                              dtype=jnp.bfloat16),
                ))
            else:
                kv_caches.append(
                    jnp.zeros(attn_cache_shape, dtype=jnp.bfloat16))

        query_start_loc = np.zeros(max_reqs + 1, dtype=np.int32)
        query_start_loc[1:] = seq_len
        seq_lens = np.zeros(max_reqs, dtype=np.int32)
        seq_lens[0] = seq_len
        state_indices = np.zeros(max_reqs, dtype=np.int32)
        state_indices[0] = 1
        metadata = AttentionMetadata(
            input_positions=jnp.arange(seq_len),
            block_tables=jnp.zeros(max_reqs * num_blocks, dtype=jnp.int32),
            seq_lens=jnp.asarray(seq_lens),
            query_start_loc=jnp.asarray(query_start_loc),
            request_distribution=jnp.asarray([0, 1, 1], dtype=jnp.int32),
            mamba_state_indices=jnp.asarray(state_indices),
        )

        input_ids = jnp.arange(seq_len, dtype=jnp.int32)
        with jax.set_mesh(mesh):
            new_kv_caches, hidden, _, _ = model(kv_caches, input_ids, metadata)
            logits = model.compute_logits(hidden)

        assert hidden.shape == (seq_len, hf_config.hidden_size)
        assert logits.shape[0] == seq_len
        new_conv_state, new_recurrent_state = new_kv_caches[0]
        assert not np.allclose(np.asarray(new_recurrent_state[1]), 0.0)
        assert not np.allclose(np.asarray(new_conv_state[1]), 0.0)
