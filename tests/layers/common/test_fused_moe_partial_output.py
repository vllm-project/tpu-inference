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
"""Tests for `partial_output_to_add` in the fused MoE path.

`partial_output_to_add` lets a caller fold a per-device partial sum (e.g. a
TP-sharded shared expert) into the routed-expert partial *before* the MoE
all-reduce, so both are reduced by one collective instead of two.

These tests run on CPU: the grouped matmul and the collective are mocked, as
in `tests/kernels/experimental/small_collectives/test_small_all_reduce.py`,
so only the arithmetic and the plumbing are checked here.
"""
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.layers.common import fused_moe_gmm
from tpu_inference.layers.common.fused_moe_gmm import (_add_partial_output,
                                                       moe_gmm_local)


def test_add_partial_output_same_shape():
    out = jnp.ones((4, 8), jnp.bfloat16)
    partial = jnp.full((4, 8), 2.0, jnp.float32)
    res = _add_partial_output(out, partial)
    assert res.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(res, np.float32), 3.0)


def test_add_partial_output_lane_padded_hidden():
    # The kernel may run on a lane-padded hidden size; the partial covers only
    # the real columns and the padding is left untouched.
    out = jnp.ones((4, 16), jnp.float32)
    partial = jnp.full((4, 10), 2.0, jnp.float32)
    res = np.asarray(_add_partial_output(out, partial))
    np.testing.assert_array_equal(res[:, :10], 3.0)
    np.testing.assert_array_equal(res[:, 10:], 1.0)


def test_add_partial_output_none_is_identity():
    out = jnp.ones((4, 8), jnp.float32)
    assert _add_partial_output(out, None) is out


def test_add_partial_output_rejects_bad_shapes():
    out = jnp.ones((4, 8), jnp.float32)
    with pytest.raises(ValueError):
        _add_partial_output(out, jnp.ones((5, 8), jnp.float32))
    with pytest.raises(ValueError):
        _add_partial_output(out, jnp.ones((4, 9), jnp.float32))


def _moe_gmm_local_inputs():
    num_tokens = 8
    topk = 2
    batch_size = num_tokens * topk
    hidden_size = 128
    inter_size = 64
    global_num_experts = 8
    local_group_size = 2
    x = jnp.ones((batch_size, hidden_size), dtype=jnp.bfloat16)
    w1 = jnp.ones((local_group_size, hidden_size, inter_size * 2),
                  dtype=jnp.bfloat16)
    w2 = jnp.ones((local_group_size, inter_size, hidden_size),
                  dtype=jnp.bfloat16)
    kwargs = dict(
        group_sizes=jnp.full((global_num_experts, ), 2, dtype=jnp.int32),
        group_offset=jnp.array([0], dtype=jnp.int32),
        topk_argsort_revert_indices=jnp.arange(batch_size, dtype=jnp.int32),
        topk_weights=jnp.full((num_tokens, topk), 0.5, dtype=jnp.float32),
    )
    return x, w1, w2, num_tokens, hidden_size, topk, kwargs


@pytest.mark.parametrize("parallelism", ["tp", "ep"])
def test_moe_gmm_local_adds_partial_before_reduce(parallelism):
    """out(with partial) - out(without) == partial, and the partial is summed
    *inside* the reduction (the mocked psum sees it)."""
    x, w1, w2, num_tokens, hidden_size, topk, kwargs = _moe_gmm_local_inputs()

    def fake_gmm(lhs, rhs, *args, **unused):
        return jnp.ones((lhs.shape[0], hidden_size), dtype=jnp.bfloat16)

    seen_by_psum = []

    def fake_psum(val, axis_name):
        seen_by_psum.append(val)
        return val

    partial = (jnp.arange(num_tokens * hidden_size, dtype=jnp.float32).reshape(
        num_tokens, hidden_size) % 7).astype(jnp.bfloat16)

    def run(partial_output):
        seen_by_psum.clear()
        with patch.object(jax.lax, "psum", side_effect=fake_psum), \
             patch.object(fused_moe_gmm, "gmm_wrapper", side_effect=fake_gmm):
            out = moe_gmm_local(
                x,
                w1,
                None,
                None,
                w2,
                None,
                None,
                kwargs["group_sizes"],
                kwargs["group_offset"],
                kwargs["topk_argsort_revert_indices"],
                kwargs["topk_weights"],
                partial_output,
                activation="silu",
                topk=topk,
                parallelism=parallelism,
                onehot_moe_permute_threshold=1024,
                defer_all_reduce=False,
            )
        assert len(seen_by_psum) == 1
        return np.asarray(out, np.float32), np.asarray(
            seen_by_psum[0], np.float32)

    base, base_reduced = run(None)
    with_partial, with_partial_reduced = run(partial)
    np.testing.assert_allclose(with_partial - base,
                               np.asarray(partial, np.float32))
    # The addend went in before the collective, not after.
    np.testing.assert_allclose(with_partial_reduced - base_reduced,
                               np.asarray(partial, np.float32))


def test_moe_gmm_local_rejects_partial_on_scatter_paths():
    x, w1, w2, _, _, topk, kwargs = _moe_gmm_local_inputs()
    partial = jnp.zeros((8, 128), jnp.bfloat16)
    for flags in (dict(scatter_results=True), dict(enable_rs_kernel=True)):
        with pytest.raises(NotImplementedError):
            moe_gmm_local(
                x,
                w1,
                None,
                None,
                w2,
                None,
                None,
                kwargs["group_sizes"],
                kwargs["group_offset"],
                kwargs["topk_argsort_revert_indices"],
                kwargs["topk_weights"],
                partial,
                activation="silu",
                topk=topk,
                parallelism="ep",
                onehot_moe_permute_threshold=1024,
                **flags,
            )


def test_fused_moe_func_validates_partial_output():
    hidden = jnp.ones((8, 128), jnp.bfloat16)
    w1 = jnp.ones((8, 128, 128), jnp.bfloat16)
    w2 = jnp.ones((8, 64, 128), jnp.bfloat16)
    gating = jnp.ones((8, 8), jnp.float32)
    # Shape mismatch is rejected before anything is dispatched.
    with pytest.raises(ValueError):
        fused_moe_gmm.fused_moe_func(
            hidden,
            w1,
            w2,
            None,
            None,
            None,
            None,
            gating_output=gating,
            topk=2,
            partial_output_to_add=jnp.ones((4, 128), jnp.bfloat16),
        )
    # Reduce-scatter style paths cannot absorb a partial.
    for flags in (dict(scatter_results=True), dict(enable_rs_kernel=True),
                  dict(use_gmm_fused_rs_kernel=True)):
        with pytest.raises(NotImplementedError):
            fused_moe_gmm.fused_moe_func(
                hidden,
                w1,
                w2,
                None,
                None,
                None,
                None,
                gating_output=gating,
                topk=2,
                partial_output_to_add=jnp.ones((8, 128), jnp.bfloat16),
                **flags,
            )
