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
"""The mesh the small-collective kernel tests run on."""

import jax
import numpy as np
import pytest
from jax.sharding import Mesh

from .oracle import AXIS, N


@pytest.fixture(scope="module")
def mesh():
    """The eight cores of one host. A fixture, not a `skipif`: asking JAX for
    devices at collection would claim the chip for the pytest parent."""
    if jax.devices()[0].platform != "tpu" or jax.device_count() < N:
        pytest.skip("needs the 8 TPU cores of one host")
    return Mesh(np.array(jax.devices()[:N]), (AXIS, ))
