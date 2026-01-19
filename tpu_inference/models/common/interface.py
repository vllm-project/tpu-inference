# SPDX-License-Identifier: Apache-2.0

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
