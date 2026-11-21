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

from tpu_inference.layers.common.attention_metadata import (AttentionMetadata,
                                                            MambaMetadata)


def test_mamba_state_indices_alias():
    positions = jnp.zeros((4, ), dtype=jnp.int32)
    assert AttentionMetadata(
        input_positions=positions).mamba_state_indices is None

    state_indices = jnp.array([3, 7], dtype=jnp.int32)
    metadata = AttentionMetadata(
        input_positions=positions,
        mamba=MambaMetadata(state_indices=state_indices))
    assert metadata.mamba_state_indices is state_indices


def test_mamba_state_indices_alias_is_not_a_pytree_leaf():
    state_indices = jnp.array([3, 7], dtype=jnp.int32)
    metadata = AttentionMetadata(
        input_positions=jnp.zeros((4, ), dtype=jnp.int32),
        mamba=MambaMetadata(state_indices=state_indices))

    leaves, treedef = jax.tree_util.tree_flatten(metadata)
    assert len(leaves) == 2  # input_positions and mamba.state_indices

    rebuilt = jax.tree_util.tree_unflatten(treedef, [x + 1 for x in leaves])
    assert jnp.array_equal(rebuilt.mamba_state_indices, state_indices + 1)
