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

from typing import Iterable, Protocol

from flax import nnx
from vllm.distributed.utils import get_pp_indices

from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.jax import JaxModule
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


class PPMissingLayer(JaxModule):
    """
    A placeholder layer for missing layers in a pipeline parallel model.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        """Return the first arg from args or the first value from kwargs."""
        return args[0] if args else next(iter(kwargs.values()))

    def load_weights(self, weights: Iterable, *args, **kwargs) -> set[str]:
        """Consume and drop weights owned by other pipeline stages.

        Returns an empty set: nothing is loaded on this rank. Returning a
        set (instead of None) tells vLLM's AutoWeightsLoader that the
        weights were handled, so it does not emit a per-layer
        "Unable to collect loaded parameters" warning during loading.
        """
        num_dropped = sum(1 for _ in weights)
        logger.debug(
            "[pp] PPMissingLayer dropped %d weights owned by another "
            "pipeline stage", num_dropped)
        return set()


class LayerFn(Protocol):

    def __call__(self, layer_index: int) -> nnx.Module:
        ...


def get_start_end_layer(num_hidden_layers: int, rank: int,
                        world_size: int) -> tuple[int, int]:
    return get_pp_indices(num_hidden_layers, rank, world_size)


def make_layers(
    num_hidden_layers: int,
    layer_fn: LayerFn,
) -> tuple[int, int, nnx.List]:
    start_layer, end_layer = get_start_end_layer(num_hidden_layers,
                                                 get_pp_group().rank_in_group,
                                                 get_pp_group().world_size)

    layers = [PPMissingLayer() for _ in range(start_layer)] \
        + [layer_fn(i) for i in range(start_layer, end_layer)] \
        + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]

    return start_layer, end_layer, nnx.List(layers)
