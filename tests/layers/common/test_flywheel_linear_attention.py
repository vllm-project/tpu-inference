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
"""GDN dispatch onto the selected linear attention backend, with the kernels
stubbed."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flywheel_tpu.linear_attention.gdn import wrapper as flywheel_gdn
from jax.sharding import Mesh

from tpu_inference.kernels.gdn.v3 import wrapper as v3_gdn
from tpu_inference.layers.common import attention_backend
from tpu_inference.layers.common import \
    flywheel_linear_attention as flywheel_linear
from tpu_inference.layers.common import gdn_attention
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES

NUM_TOKENS, NUM_HEADS, HEAD_DIM, KERNEL_SIZE = 8, 2, 128, 4
QKV_DIM = 3 * NUM_HEADS * HEAD_DIM
NUM_STATE_SLOTS, WRITE_SLOT, READ_SLOT = 4, 2, 1
V3_MARKER, FLYWHEEL_MARKER = 1, 2


@pytest.fixture
def mesh() -> Mesh:
    """One CPU device over every axis the shard_map specs name."""
    devices = np.array([jax.devices("cpu")[0]])
    return Mesh(devices.reshape((1, ) * len(MESH_AXIS_NAMES)), MESH_AXIS_NAMES)


def make_inputs(mesh: Mesh) -> dict[str, object]:
    """One request of NUM_TOKENS tokens on a cached prefix, resuming from
    READ_SLOT and checkpointing into WRITE_SLOT. Each slot holds its own
    index, so a state shows which slot it came from."""
    slot_ids = jnp.arange(NUM_STATE_SLOTS, dtype=jnp.float32)
    conv_state = jnp.broadcast_to(
        slot_ids[:, None, None],
        (NUM_STATE_SLOTS, KERNEL_SIZE - 1, QKV_DIM)).astype(jnp.bfloat16)
    recurrent_state = jnp.broadcast_to(
        slot_ids[:, None, None,
                 None], (NUM_STATE_SLOTS, NUM_HEADS, HEAD_DIM, HEAD_DIM))
    return {
        "j_mixed_qkv": jnp.ones((NUM_TOKENS, QKV_DIM), jnp.bfloat16),
        "j_b": jnp.ones((NUM_TOKENS, NUM_HEADS), jnp.bfloat16),
        "j_a": jnp.ones((NUM_TOKENS, NUM_HEADS), jnp.bfloat16),
        "conv_state": conv_state,
        "recurrent_state": recurrent_state,
        "j_conv_weight": jnp.ones((QKV_DIM, 1, KERNEL_SIZE), jnp.float32),
        "j_conv_bias": None,
        "j_A_log": jnp.ones((NUM_HEADS, ), jnp.float32),
        "j_dt_bias": jnp.ones((NUM_HEADS, ), jnp.float32),
        "state_indices": jnp.array([WRITE_SLOT], jnp.int32),
        "query_start_loc": jnp.array([0, NUM_TOKENS], jnp.int32),
        "distribution": jnp.array([0, 0, 1], jnp.int32),
        "seq_lens": jnp.array([2 * NUM_TOKENS], jnp.int32),
        "n_kq": NUM_HEADS,
        "n_v": NUM_HEADS,
        "d_k": HEAD_DIM,
        "d_v": HEAD_DIM,
        "kernel_size": KERNEL_SIZE,
        "mesh": mesh,
        "read_state_indices": jnp.array([READ_SLOT], jnp.int32),
    }


def stub_kernel(marker: int, calls: list[dict]):
    """A kernel that records its static options, copies the read slot plus
    marker into the write slot, and outputs marker."""

    def run(qkv, b, a, conv_state, recurrent_state, conv_weight, conv_bias,
            a_log, dt_bias, query_start_loc, state_indices, distribution,
            seq_lens, read_state_indices, **options):
        calls.append(options)
        new_conv_state = conv_state.at[state_indices].set(
            conv_state[read_state_indices] + marker)
        new_recurrent_state = recurrent_state.at[state_indices].set(
            recurrent_state[read_state_indices] + marker)
        output = jnp.full((qkv.shape[0], options["n_v"] * options["d_v"]),
                          marker, qkv.dtype)
        return (new_conv_state, new_recurrent_state), output

    return run


@pytest.mark.parametrize(("backend_cls", "marker"), [
    (None, V3_MARKER),
    (attention_backend.GDNLinearAttentionBackend, V3_MARKER),
    (flywheel_linear.FlywheelLinearAttentionBackend, FLYWHEEL_MARKER),
])
def test_gdn_runs_the_selected_backend_kernel(mesh, monkeypatch, backend_cls,
                                              marker):
    calls: list[dict] = []
    monkeypatch.setattr(v3_gdn, "fused_conv1d_gdn",
                        stub_kernel(V3_MARKER, calls))
    monkeypatch.setattr(flywheel_gdn, "fused_conv1d_gdn",
                        stub_kernel(FLYWHEEL_MARKER, calls))
    kernel = backend_cls().conv1d_gdn_kernel() if backend_cls else None

    states, output = gdn_attention.run_jax_gdn_attention(
        **make_inputs(mesh), conv1d_gdn_kernel=kernel)

    np.testing.assert_array_equal(output.astype(jnp.float32), marker)
    for state in states:
        np.testing.assert_array_equal(
            np.asarray(state[WRITE_SLOT], np.float32), READ_SLOT + marker)
        np.testing.assert_array_equal(np.asarray(state[0], np.float32), 0)
    [call] = calls
    assert call["n_kq"] == call["n_v"] == NUM_HEADS
