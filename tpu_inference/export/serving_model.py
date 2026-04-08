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
"""Simplified serving model format utilities for export."""

import json
import logging
import os

from jax import export as jax_export


def save_jax_exported(
    exp: jax_export.Exported,
    bin_file_path: str,
    *,
    vjp_order: int = 0,
) -> None:
    """Serializes a `jax.export.Exported` object and saves it to disk."""
    if os.path.exists(bin_file_path):
        logging.warning(f"File {bin_file_path} already exists.")

    dirname = os.path.dirname(bin_file_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

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


def load(nativemodel_path: str) -> dict[str, jax_export.Exported]:
    """Load from the nativemodel_path."""
    model_fn_dir = os.path.join(nativemodel_path, "model_fn")
    model_fn_metadata_file = os.path.join(model_fn_dir, "metadata.json")
    if not os.path.exists(model_fn_metadata_file):
        raise ValueError(
            f"Model path {model_fn_metadata_file} does not exist.")
    with open(model_fn_metadata_file, "r") as f:
        model_fn_metadata = json.load(f)
    model_fn_map: dict[str, jax_export.Exported] = {}
    for method_key, method_metadata in model_fn_metadata.items():
        calling_convention_version = method_metadata[
            "calling_convention_version"]
        check_calling_convention_version_supported(calling_convention_version)
        model_fn_map[method_key] = load_jax_exported(
            os.path.join(model_fn_dir, method_metadata["file_path"]))
    return model_fn_map


def save(nativemodel_path: str,
         model_fn_map: dict[str, jax_export.Exported]) -> None:
    """Saves to the nativemodel_path."""
    model_fn_dir = os.path.join(nativemodel_path, "model_fn")
    os.makedirs(model_fn_dir, exist_ok=True)

    metadata = {}
    for method_key, exported in model_fn_map.items():
        file_name = f"{method_key}.jax_exported"
        file_path = os.path.join(model_fn_dir, file_name)
        save_jax_exported(exported, file_path)
        metadata[method_key] = {
            "file_path": file_name,
            "calling_convention_version": exported.calling_convention_version
        }

    metadata_path = os.path.join(model_fn_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)


def save_deduplicated_functions(
        export_path: str, model_fn_map: dict[str,
                                             jax_export.Exported]) -> None:
    """Saves exported functions to subfolders, deduplicating them by hash."""
    import hashlib
    import shutil

    logging.info("Saving unique exported model functions to subfolders in %s",
                 export_path)
    os.makedirs(export_path, exist_ok=True)
    seen_hashes = {}

    for name, exp in model_fn_map.items():
        try:
            # Attempt serialization to compute hash for deduplication
            serialized_bytes = exp.serialize()
            h = hashlib.md5(serialized_bytes).hexdigest()

            if h in seen_hashes:
                logging.info(
                    f"Function {name} is a duplicate of {seen_hashes[h]}, skipping."
                )
                continue

            seen_hashes[h] = name

            sub_path = os.path.join(export_path, name)
            if os.path.exists(sub_path):
                shutil.rmtree(sub_path)
            os.makedirs(sub_path, exist_ok=True)

            save(sub_path, {name: exp})
            logging.info(f"Successfully saved function {name}")
        except Exception as e:
            logging.error(f"Failed to process/save function {name}: {e}")
            if isinstance(
                    e, TypeError
            ) and "Serialization is supported only for dictionaries with string keys" in str(
                    e):
                logging.error(
                    "To fix this properly, please identify the data structure containing the integer keys "
                    "and register it using `jax.export.register_pytree_node_serialization`. "
                    "This error usually happens when a standard Python dict with integer keys is part of the PyTree. "
                    "Consider wrapping it in a custom class and registering that class for serialization."
                )
