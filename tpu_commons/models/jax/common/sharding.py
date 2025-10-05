import numpy as np
from jax.sharding import Mesh


def build_mesh(devices, strategy: dict[str, int]) -> Mesh:
    """Constructs a JAX device mesh from a sharding strategy.

    This method creates a logical grid of devices based on the parallelism
    degrees defined in the strategy. The logical axis names ('dp', 'ep',
    'sp', 'tp') are used to map tensor dimensions to the physical device grid.

    Args:
        strategy: A dictionary from upper level config.

    Returns:
        A JAX `Mesh` object.
    """

    axis_order = {
        "data": strategy.get("data_parallelism", 1),
        "expert": strategy.get("expert_parallelism", 1),
        "seq": strategy.get("sequence_parallelism", 1),
        "model": strategy.get("tensor_parallelism", 1),
    }
    # TODO: add logic to infer axis when the degree is -1
    mesh_axis_names = []
    mesh_shape = []
    for axis, dim in axis_order.items():
        mesh_axis_names.append(axis)
        mesh_shape.append(dim)

    if not mesh_shape:
        mesh_shape = [1]
        mesh_axis_names = [
            'data'
        ]  # default to data parallelism if no other strategy is specified

    devices = np.asarray(devices).reshape(mesh_shape)
    return Mesh(devices, axis_names=tuple(mesh_axis_names))
