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
"""The fused decode loop must run hybrid models' per-layer metadata dict.

Hybrid models (>1 kv-cache group: attention + linear-attention layers)
carry `attn_metadata` as a dict of per-layer `AttentionMetadata` whose
entries share every per-step field and differ only in `block_tables`.
`continue_decode` used to assume a single object and crashed on
`init_state.attn_metadata.seq_lens` during its own precompile warmup
(`'dict' object has no attribute 'seq_lens'`), which alone kept the loop
unusable on any hybrid model. The stub model below asserts the dict shape
survives into every step and that per-group block_tables stay distinct.
"""

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.runner.decode_loop import (TpuSamplingState,
                                              _first_group_metadata,
                                              _with_step_dynamics,
                                              continue_decode)

_BATCH = 4
_VOCAB = 11


def _metadata(block_fill: int) -> AttentionMetadata:
    return AttentionMetadata(
        input_positions=jnp.zeros((_BATCH, ), dtype=jnp.int32),
        block_tables=jnp.full((_BATCH, 2), block_fill, dtype=jnp.int32),
        seq_lens=jnp.ones((_BATCH, ), dtype=jnp.int32),
        query_start_loc=jnp.arange(_BATCH + 1, dtype=jnp.int32),
        request_distribution=jnp.array([_BATCH, 0, _BATCH], dtype=jnp.int32),
        mamba_state_indices=jnp.arange(_BATCH, dtype=jnp.int32),
        has_initial_state=jnp.ones((_BATCH, ), dtype=jnp.int32),
    )


def test_with_step_dynamics_updates_only_positions_and_seq_lens():
    template = {"attn": _metadata(7), "linear": _metadata(9)}
    pos = jnp.full((_BATCH, ), 5, dtype=jnp.int32)
    sl = jnp.full((_BATCH, ), 6, dtype=jnp.int32)
    stepped = _with_step_dynamics(template, pos, sl)
    assert set(stepped) == {"attn", "linear"}
    for name, group in stepped.items():
        np.testing.assert_array_equal(group.input_positions, pos)
        np.testing.assert_array_equal(group.seq_lens, sl)
        # Everything else is loop-invariant, including the per-group part.
        np.testing.assert_array_equal(group.block_tables,
                                      template[name].block_tables)
    ref = _first_group_metadata(template)
    assert ref is next(iter(template.values()))


def _stub_model_fn(state,
                   kv_caches,
                   tokens,
                   attn_metadata,
                   inputs_embeds,
                   positions,
                   layer_name_to_kvcache_index,
                   lora_metadata,
                   intermediate_tensors,
                   is_first_rank,
                   is_last_rank,
                   shared_attention_metadata=None):
    # The hybrid contract: the loop must hand the model the SAME dict shape
    # the runner primed it with, with distinct per-group block_tables.
    assert isinstance(attn_metadata, dict)
    assert set(attn_metadata) == {"attn", "linear"}
    delta = (attn_metadata["linear"].block_tables[0, 0] -
             attn_metadata["attn"].block_tables[0, 0])
    hidden = (positions.astype(jnp.float32)[:, None] +
              delta.astype(jnp.float32) * 0.0) * jnp.ones((1, 3))
    return kv_caches, hidden, [], None


def _stub_compute_logits_fn(state, hidden, _):
    # Deterministic argmax = floor(position) + 1, capped inside the vocab.
    idx = jnp.clip(hidden[:, 0].astype(jnp.int32) + 1, 0, _VOCAB - 1)
    return jax.nn.one_hot(idx, _VOCAB, dtype=jnp.float32)


def _stub_sample_fn(rng, mesh, logits, sampling_metadata):
    return jnp.argmax(logits, axis=-1).astype(jnp.int32), logits


def test_continue_decode_runs_a_hybrid_metadata_dict():
    if jax.default_backend() == "cpu":
        import pytest
        pytest.skip("the decode loop's jit carries TPU compiler options")
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("x", ))
    template = {"attn": _metadata(7), "linear": _metadata(9)}
    init_state = TpuSamplingState(
        current_tokens=jnp.zeros((_BATCH, ), dtype=jnp.int32),
        active_mask=jnp.ones((_BATCH, ), dtype=bool),
        attn_metadata=template,
        step_counter=jnp.array(0, dtype=jnp.int32),
    )
    steps = 3
    (token_buffer, kv_caches, final_state, _, expert_indices,
     logprobs) = continue_decode(
         state={},
         model_fn=_stub_model_fn,
         compute_logits_fn=_stub_compute_logits_fn,
         sample_fn=_stub_sample_fn,
         init_state=init_state,
         kv_caches=[jnp.zeros((2, 2), dtype=jnp.float32)],
         max_decode_steps=jnp.array(steps, dtype=jnp.int32),
         static_max_decode_steps=steps,
         eos_token_id=(_VOCAB - 1, ),
         padding_token_id=0,
         rng=jax.random.key(0),
         mesh=mesh,
         sampling_metadata=None,
     )
    assert token_buffer.shape == (steps, _BATCH)
    assert expert_indices is None and logprobs is None
    # Every step ran: positions advanced by `steps` for every active row, and
    # the final metadata kept the dict shape with per-group tables intact.
    assert isinstance(final_state.attn_metadata, dict)
    np.testing.assert_array_equal(
        _first_group_metadata(final_state.attn_metadata).input_positions,
        np.full((_BATCH, ), steps, dtype=np.int32))
    np.testing.assert_array_equal(
        final_state.attn_metadata["linear"].block_tables,
        np.full((_BATCH, 2), 9, dtype=np.int32))
