from typing import List, Protocol

from flax import nnx
from vllm.distributed import get_pp_group
from vllm.distributed.utils import get_pp_indices


class PPMissingLayer(nnx.Module):
    """
    A placeholder layer for missing layers in a pipeline parallel model.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        """Return the first arg from args or the first value from kwargs."""
        return args[0] if args else next(iter(kwargs.values()))


class LayerFn(Protocol):

    def __call__(self) -> nnx.Module:
        ...


def make_layers(
    num_hidden_layers: int,
    layer_fn: LayerFn,
) -> tuple[int, int, List[nnx.Module]]:
    start_layer, end_layer = get_pp_indices(num_hidden_layers,
                                            get_pp_group().rank_in_group,
                                            get_pp_group().world_size)

    layers = [PPMissingLayer() for _ in range(start_layer)] \
        + [layer_fn() for _ in range(start_layer, end_layer)] \
        + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]

    return start_layer, end_layer, layers
