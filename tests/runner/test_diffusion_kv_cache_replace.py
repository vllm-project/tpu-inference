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

import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np


def _load_module():
    path = (pathlib.Path(__file__).resolve().parents[2] / "tpu_inference" /
            "layers" / "common" / "kv_cache_replace.py")
    spec = importlib.util.spec_from_file_location("kv_cache_replace_under_test",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kv_cache_replace = _load_module()


def test_replace_cached_kv_overwrites_only_requested_active_positions():
    cache = jnp.zeros((4, 4, 2, 1, 2), dtype=jnp.float32)
    keys = jnp.array([
        [[1.0, 2.0]],
        [[3.0, 4.0]],
        [[50.0, 60.0]],
        [[70.0, 80.0]],
    ])
    values = keys + 10.0

    replace = jax.jit(kv_cache_replace.replace_cached_kv)
    updated = replace(
        cache,
        keys,
        values,
        jnp.array([5, 6, 1, 2], dtype=jnp.int32),
        jnp.array([[2, 1], [3, 0]], dtype=jnp.int32),
        jnp.array([True, False]),
    )

    np.testing.assert_array_equal(updated[1, 1, 0, 0], [1.0, 2.0])
    np.testing.assert_array_equal(updated[1, 1, 1, 0], [11.0, 12.0])
    np.testing.assert_array_equal(updated[1, 2, 0, 0], [3.0, 4.0])
    np.testing.assert_array_equal(updated[1, 2, 1, 0], [13.0, 14.0])
    assert np.count_nonzero(np.asarray(updated[0])) == 0
    assert np.count_nonzero(np.asarray(updated[2])) == 0
    assert np.count_nonzero(np.asarray(updated[3])) == 0


def test_replace_cached_kv_matches_bfloat16_head_packing():
    cache = jnp.zeros((1, 4, 2, 2, 2), dtype=jnp.bfloat16)
    keys = jnp.array([[[1, 2], [3, 4]]], dtype=jnp.bfloat16)
    values = jnp.array([[[11, 12], [13, 14]]], dtype=jnp.bfloat16)

    updated = kv_cache_replace.replace_cached_kv(
        cache,
        keys,
        values,
        jnp.array([2], dtype=jnp.int32),
        jnp.array([[0]], dtype=jnp.int32),
        jnp.array([True]),
    )

    np.testing.assert_array_equal(
        updated[0, 2],
        np.array([[[1, 2], [11, 12]], [[3, 4], [13, 14]]]),
    )
