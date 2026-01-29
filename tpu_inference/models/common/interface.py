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

from typing import Protocol

import jax
import numpy as np
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata


class PoolerFunc(Protocol):
    """The wrapped pooler interface.

    Accept hidden-state, pooling-metadata and sequence lengths.
    Returns pooler output as a list of tensors, one per request.

    The contract is dependent on vLLM lib.
    """

    def __call__(
        self,
        hidden_states: jax.Array,
        pooling_metadata: PoolingMetadata,
        seq_lens: np.ndarray,
    ) -> PoolerOutput:
        ...
