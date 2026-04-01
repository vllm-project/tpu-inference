"""Simplified serving checkpoint format for export (lazy mode only)."""

import datetime
import hashlib
import json
import os
from typing import Any

from absl import logging
import jax
from jax import export as jax_export
import jaxtyping

CURRENT_CHECKPOINT_VERSION = 1
MODEL_PARAMS_SHAPE_DTYPE_STRUCT_FILE_NAME = "model_params_shape_dtype_struct.jax_exported"
METADATA_FILE_NAME = "metadata.json"

def _calculate_md5(file_path: str) -> str:
    """Calculate the MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            md5.update(block)
    return md5.hexdigest()


def _save_checkpoint_abstract_info(
    params: jaxtyping.PyTree[jax.typing.ArrayLike],
    output_dir: str,
    params_sharding_override: jax.sharding.Sharding | None = None,
) -> None:
    """Saves the checkpoint abstract info (shape, dtype, sharding) to the output directory."""

    @jax.jit
    def _dummy_fn(
        params: jaxtyping.PyTree[jax.typing.ArrayLike],
    ) -> jaxtyping.PyTree[jax.typing.ArrayLike]:
        return params

    if params_sharding_override is not None:
        def _make_struct(p, s):
            return jax.ShapeDtypeStruct(shape=p.shape, dtype=p.dtype, sharding=s)
        export_inputs = jax.tree.map(_make_struct, params, params_sharding_override)
    else:
        export_inputs = params

    model_params_shape_dtype_struct = jax_export.export(_dummy_fn)(export_inputs)
    
    os.makedirs(output_dir, exist_ok=True)
    bin_path = os.path.join(output_dir, MODEL_PARAMS_SHAPE_DTYPE_STRUCT_FILE_NAME)
    with open(bin_path, "wb") as f:
        f.write(model_params_shape_dtype_struct.serialize())


def save(
    params: jaxtyping.PyTree[jax.typing.ArrayLike],
    output_dir: str,
    abstract_mesh: jax.sharding.AbstractMesh,
    *,
    description: str = "",
    params_sharding_override: jaxtyping.PyTree[jax.sharding.Sharding] | None = None,
) -> None:
    """Saves JAX parameter tree abstract info with version and content tracking."""
    
    os.makedirs(output_dir, exist_ok=True)

    _save_checkpoint_abstract_info(params, output_dir, params_sharding_override)
    
    abstract_mesh_str = tuple(
        (name, size)
        for name, size in zip(abstract_mesh.axis_names, abstract_mesh.axis_sizes)
    )
    abstract_mesh_str = json.dumps(abstract_mesh_str)

    bin_path = os.path.join(output_dir, MODEL_PARAMS_SHAPE_DTYPE_STRUCT_FILE_NAME)
    shape_dtype_struct_hash = _calculate_md5(bin_path)

    metadata = {
        "version": CURRENT_CHECKPOINT_VERSION,
        "timestamp": datetime.datetime.now().isoformat(),
        "description": description,
        "fingerprint": {
            "shape_dtype_struct_hash": shape_dtype_struct_hash,
        },
        "abstract_mesh": abstract_mesh_str,
    }

    metadata_path = os.path.join(output_dir, METADATA_FILE_NAME)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load(
    output_dir: str,
    *,
    mesh: jax.sharding.Mesh | None = None,
    lazy_mode: bool = True, # Default to lazy_mode in export lib
) -> jaxtyping.PyTree[jax.typing.ArrayLike]:
    """Loads JAX parameter tree abstract info from a saved servingcheckpoint."""
    
    metadata_path = os.path.join(output_dir, METADATA_FILE_NAME)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    abstract_mesh_str = metadata["abstract_mesh"]
    shape_tuple = json.loads(abstract_mesh_str)
    
    # Reconstruct AbstractMesh
    axis_names = tuple(item[0] for item in shape_tuple)
    axis_sizes = tuple(item[1] for item in shape_tuple)
    abstract_mesh = jax.sharding.AbstractMesh(axis_sizes, axis_names)

    if mesh is not None:
        if not (mesh.axis_names == abstract_mesh.axis_names and mesh.axis_sizes == abstract_mesh.axis_sizes):
            raise ValueError(f"Mesh setup must match abstract mesh. mesh: {mesh}, abstract_mesh: {abstract_mesh}")
    else:
        # If no mesh provided, we can't create NamedSharding easily without real devices,
        # but for lazy load (abstract) we might just create a dummy mesh if we can.
        # Let's assume mesh IS provided or we are okay with un-sharded ShapeDtypeStructs if possible.
        # If we need sharded ShapeDtypeStructs, we need a mesh.
        # Let's assume mesh is provided for now if we want shardings.
        pass

    bin_path = os.path.join(output_dir, MODEL_PARAMS_SHAPE_DTYPE_STRUCT_FILE_NAME)
    with open(bin_path, "rb") as f:
        exported = jax_export.deserialize(bytearray(f.read()))

    in_avals = exported.in_avals
    
    # If no mesh, we can't get shardings in the same way.
    # exported.in_shardings_jax(mesh) requires mesh.
    if mesh is not None:
        in_shardings = exported.in_shardings_jax(mesh)
    else:
        in_shardings = [None] * len(in_avals) # Or leave as None

    if lazy_mode:
        def _make_struct(aval, sharding):
            return jax.ShapeDtypeStruct(aval.shape, aval.dtype, sharding=sharding)
        
        flat_shape_dtype_struct = jax.tree.map(_make_struct, in_avals, in_shardings)
        args_and_kwargs = jax.tree.unflatten(exported.in_tree, flat_shape_dtype_struct)
    else:
        raise NotImplementedError("Non-lazy mode is not supported in this simplified library.")

    args = args_and_kwargs[0]
    arg = args[0]
    return arg
