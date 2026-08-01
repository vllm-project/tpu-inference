# Copyright 2025 Google LLC
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

import math
from typing import Tuple

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.utils import general_device_put


# TODO(xiang): move this to weight_utils.py
def shard_put(x: jax.Array, sharding_names: Tuple[str, ...] | P,
              mesh: jax.sharding.Mesh) -> jax.Array:
    # Single device sharding requires this special handling
    # to avoid the recursive jit error.
    if math.prod(mesh.axis_sizes) == 1:
        return jax.device_put(x, mesh.devices.flatten()[0])
    # Not `jax.device_put(x, NamedSharding(mesh, ...))`: under the Ray
    # multi-host backend a process addresses only its own devices, so naming a
    # sharding over the whole mesh is rejected ("must be a Device or a
    # Sharding which represents addressable devices"). `general_device_put`
    # assembles the global array from each process's addressable shards, and
    # falls through to a plain `device_put` when there is a single process --
    # so single-host behaviour is unchanged.
    source_mesh = (x.sharding.mesh if isinstance(getattr(x, "sharding", None),
                                                 NamedSharding) else None)
    return general_device_put(x,
                              NamedSharding(mesh, P(*sharding_names)),
                              source_mesh=source_mesh)
