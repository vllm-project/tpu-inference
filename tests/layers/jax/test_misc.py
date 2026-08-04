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
"""`layers/jax/misc.shard_put` -- the Llama-4 weight placement helper.

It is a second copy of the placement logic in
`models/jax/utils/weight_utils.shard_put`, and it has to make the same
multi-host choice: under the Ray backend a process addresses only its own
devices, so `jax.device_put(x, NamedSharding(full_mesh, ...))` is rejected and
the array has to be assembled from process-local shards instead.
"""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tpu_inference import envs
from tpu_inference.layers.jax.misc import shard_put


def _mesh(num_devices):
    devices = jax.devices()[:num_devices]
    if len(devices) < num_devices:
        pytest.skip(f"needs {num_devices} devices, have {len(devices)}")
    return Mesh(np.array(devices).reshape(1, num_devices), ("data", "model"))


def test_multihost_placement_uses_the_process_local_api():
    """Without this, a multi-host Llama-4 load dies the same way the MXFP4
    expert placement did: device_put over a mesh whose devices this process
    does not all address."""
    mesh = _mesh(2)
    x = jnp.arange(16, dtype=jnp.float32).reshape(2, 8)
    with mock.patch.object(envs, "TPU_MULTIHOST_BACKEND", "ray"):
        with mock.patch("jax.make_array_from_callback",
                        side_effect=jax.make_array_from_callback) as spy:
            out = shard_put(x, (None, "model"), mesh)
    spy.assert_called_once()
    assert spy.call_args[0][1].mesh is mesh
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x))


def test_single_host_placement_is_unchanged():
    """Anti-vacuity for the above, and the compatibility claim: with the
    backend unset the process-local path is not taken at all."""
    mesh = _mesh(2)
    x = jnp.arange(16, dtype=jnp.float32).reshape(2, 8)
    with mock.patch.object(envs, "TPU_MULTIHOST_BACKEND", ""):
        with mock.patch("jax.make_array_from_callback",
                        side_effect=jax.make_array_from_callback) as spy:
            out = shard_put(x, (None, "model"), mesh)
    spy.assert_not_called()
    assert out.sharding.spec == P(None, "model")
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x))


def test_both_paths_agree_on_values_and_sharding():
    mesh = _mesh(2)
    x = jnp.arange(32, dtype=jnp.float32).reshape(4, 8)
    with mock.patch.object(envs, "TPU_MULTIHOST_BACKEND", ""):
        single = shard_put(x, (None, "model"), mesh)
    with mock.patch.object(envs, "TPU_MULTIHOST_BACKEND", "ray"):
        multi = shard_put(x, (None, "model"), mesh)
    assert single.sharding == multi.sharding
    np.testing.assert_array_equal(np.asarray(single), np.asarray(multi))


def test_single_device_mesh_keeps_its_shortcut():
    """A 1-device mesh still bypasses sharding entirely (the comment in the
    helper: naming a sharding there triggers a recursive-jit error)."""
    mesh = _mesh(1)
    x = jnp.arange(8, dtype=jnp.float32)
    with mock.patch.object(envs, "TPU_MULTIHOST_BACKEND", "ray"):
        with mock.patch("jax.make_array_from_callback") as spy:
            out = shard_put(x, ("model", ), mesh)
    spy.assert_not_called()
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x))
