"""Export manager to orchestrate model and checkpoint export."""

import json
import os
import logging
from typing import Callable, Any
import jax

from tpu_inference.export import serving_model
from tpu_inference.export import serving_checkpoint

logger = logging.getLogger(__name__)

def export_model(name: str, fn: Callable, *args, **kwargs):
    """Exports the model function using jax.export and saves it along with checkpoint info.

    Args:
        name: Name of the function being exported (e.g., 'prefill', 'decode').
        fn: The function to export.
        *args: positional arguments to pass to the function for tracing.
        **kwargs: keyword arguments. Can include 'mesh' to save checkpoint info.
    """
    export_path = os.environ.get("GOOGLE_EXPORT_MODEL_PATH")
    if not export_path:
        logger.warning("GOOGLE_EXPORT_MODEL_PATH not set, skipping export.")
        return

    logger.info(f"Exporting model {name} to {export_path}")

    # Ensure the directory exists
    os.makedirs(export_path, exist_ok=True)

    try:
        # Export the function
        exported = jax.export.export(fn)(*args)
        
        # Save the exported model
        model_dir = os.path.join(export_path, "model_fn")
        os.makedirs(model_dir, exist_ok=True)
        
        bin_file_name = f"{name}.pb"
        bin_file_path = os.path.join(model_dir, bin_file_name)
        serving_model.save_jax_exported(exported, bin_file_path)
        
        # Save metadata.json as expected by load_native_model
        metadata_path = os.path.join(model_dir, "metadata.json")
        metadata = {
            name: {
                "file_path": bin_file_name,
                "calling_convention_version": exported.calling_convention_version,
            }
        }
        
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                try:
                    existing_metadata = json.load(f)
                    existing_metadata.update(metadata)
                    metadata = existing_metadata
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode existing metadata.json at {metadata_path}, overwriting.")
                
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Successfully exported model function {name}")

        # Save checkpoint info (abstract info)
        mesh = kwargs.get("mesh")
        if len(args) > 0 and mesh is not None:
             state = args[0]
             checkpoint_dir = os.path.join(export_path, "checkpoint")
             # Use the first arg as params/state for checkpoint info
             serving_checkpoint.save(state, checkpoint_dir, mesh, description=f"Exported from {name}")
             logger.info(f"Successfully exported checkpoint abstract info to {checkpoint_dir}")
        else:
             logger.warning(f"Skipping checkpoint export for {name}. args length: {len(args)}, mesh provided: {mesh is not None}")

    except Exception as e:
        logger.exception(f"Failed to export model {name}: {e}")
        raise e
