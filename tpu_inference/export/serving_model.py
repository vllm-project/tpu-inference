"""Simplified serving model format utilities for export."""

import json
import os
from typing import Any

from absl import logging
import jax
from jax import export as jax_export

def save_jax_exported(
    exp: jax_export.Exported,
    bin_file_path: str,
    *,
    vjp_order: int = 0,
) -> None:
    """Serializes a `jax.export.Exported` object and saves it to disk."""
    if os.path.exists(bin_file_path):
        raise ValueError(f"File {bin_file_path} already exists.")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(bin_file_path), exist_ok=True)
    
    with open(bin_file_path, "wb") as f:
        f.write(exp.serialize(vjp_order=vjp_order))


def load_jax_exported(bin_file_path: str) -> jax_export.Exported:
    """Loads a `jax.export.Exported` object from disk."""
    if not os.path.exists(bin_file_path):
        raise ValueError(f"File {bin_file_path} does not exist.")
    with open(bin_file_path, "rb") as f:
        return jax_export.deserialize(bytearray(f.read()))


def check_calling_convention_version_supported(version: int) -> None:
    """Checks if the calling convention is supported."""
    if version < jax_export.minimum_supported_calling_convention_version:
        raise ValueError(
            f"Calling convention version {version} is not supported. Minimum is {jax_export.minimum_supported_calling_convention_version}."
        )
    if version > jax_export.maximum_supported_calling_convention_version:
        raise ValueError(
            f"Calling convention version {version} is not supported. Maximum is {jax_export.maximum_supported_calling_convention_version}."
        )


def load_native_model(nativemodel_path: str) -> dict[str, jax_export.Exported]:
    """Load the native model from the nativemodel_path."""
    model_fn_dir = os.path.join(nativemodel_path, "model_fn")
    model_fn_metadata_file = os.path.join(model_fn_dir, "metadata.json")
    if not os.path.exists(model_fn_metadata_file):
        raise ValueError(f"Model path {model_fn_metadata_file} does not exist.")
    with open(model_fn_metadata_file, "r") as f:
        model_fn_metadata = json.load(f)
    model_fn_map: dict[str, jax_export.Exported] = {}
    for method_key, method_metadata in model_fn_metadata.items():
        calling_convention_version = method_metadata["calling_convention_version"]
        check_calling_convention_version_supported(calling_convention_version)
        model_fn_map[method_key] = load_jax_exported(
            os.path.join(model_fn_dir, method_metadata["file_path"])
        )
    return model_fn_map


def save_native_model(nativemodel_path: str,
                      model_fn_map: dict[str, jax_export.Exported]) -> None:
    """Saves the native model to the nativemodel_path."""
    model_fn_dir = os.path.join(nativemodel_path, "model_fn")
    os.makedirs(model_fn_dir, exist_ok=True)

    metadata = {}
    for method_key, exported in model_fn_map.items():
        file_name = f"{method_key}.pb"
        file_path = os.path.join(model_fn_dir, file_name)
        save_jax_exported(exported, file_path)
        metadata[method_key] = {
            "file_path": file_name,
            "calling_convention_version": exported.calling_convention_version
        }

    metadata_path = os.path.join(model_fn_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

