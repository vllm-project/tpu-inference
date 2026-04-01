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

import os
from typing import Any, Dict, Optional, Union

import jax
import orbax.checkpoint as ocp
from flax import nnx

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def _maybe_register_colocated_python_handlers(
    enable_colocated_python: bool,
    enable_single_replica: bool = False,
) -> None:
    """Register colocated Python array handlers for Pathways checkpointing."""
    if not enable_colocated_python:
        return
    logger.info("Registering colocated python array handler for checkpointing")
    checkpointing_impl = ocp.pathways.CheckpointingImpl.from_options(
        use_colocated_python=True,
    )
    ocp.pathways.register_type_handlers(
        use_single_replica_array_handler=enable_single_replica,
        checkpointing_impl=checkpointing_impl,
    )


def save_checkpoint(
    state: Any,
    path: str,
    step: int = 0,
    use_checkpoint_manager: bool = False,
    enable_colocated_python: bool = False,
    enable_single_replica: bool = False,
):
    """
    Saves the model state (JAX PyTree or nnx.State) to an Orbax checkpoint.

    Args:
        state: The model state to save. Can be a JAX PyTree (dict of arrays) 
            or a flax.nnx.State.
        path: The directory to save the checkpoint to.
        step: The current step number (used if use_checkpoint_manager is True).
        use_checkpoint_manager: Whether to use CheckpointManager for versioning.
        enable_colocated_python: Whether to use colocated Python checkpointing
            optimization (for Pathways / single controller).
        enable_single_replica: Whether to use single replica array handler.
    """
    logger.info(f"Saving checkpoint to {path} (step {step})...")

    _maybe_register_colocated_python_handlers(
        enable_colocated_python, enable_single_replica)

    # Ensure the path exists
    os.makedirs(path, exist_ok=True)

    # Convert nnx.State to a plain dict/PyTree for Orbax if needed.
    # StandardCheckpointer handles PyTrees.
    if isinstance(state, nnx.State):
        save_state = state.to_pure_dict()
    else:
        save_state = state

    if use_checkpoint_manager:
        options = ocp.CheckpointManagerOptions(max_to_keep=3)
        mngr = ocp.CheckpointManager(path, options=options, item_names=('state',))
        mngr.save(step, args=ocp.args.Composite(state=ocp.args.StandardSave(save_state)))
        mngr.wait_until_finished()
    else:
        checkpointer = ocp.StandardCheckpointer()
        save_path = os.path.join(path, f"checkpoint_{step}")
        checkpointer.save(save_path, save_state, force=True)
        checkpointer.wait_until_finished()

    logger.info(f"Checkpoint saved successfully to {path}")


def load_checkpoint(
    path: str,
    abstract_state: Any,
    step: Optional[int] = None,
    use_checkpoint_manager: bool = False,
    enable_colocated_python: bool = False,
    enable_single_replica: bool = False,
) -> Any:
    """
    Loads the model state (JAX PyTree or nnx.State) from an Orbax checkpoint.

    Args:
        path: The directory to load the checkpoint from.
        abstract_state: An abstract version of the state (containing shapes/dtypes)
            to guide the restoration. Can be a PyTree of jax.ShapeDtypeStruct 
            or an abstract flax.nnx.State.
        step: The step number to load (if None and using manager, loads the latest).
        use_checkpoint_manager: Whether to use CheckpointManager.
        enable_colocated_python: Whether to use colocated Python checkpointing
            optimization (for Pathways / single controller).
        enable_single_replica: Whether to use single replica array handler.

    Returns:
        The restored model state. If abstract_state was nnx.State, returns nnx.State.
    """
    logger.info(f"Loading checkpoint from {path}...")

    _maybe_register_colocated_python_handlers(
        enable_colocated_python, enable_single_replica)

    is_nnx = isinstance(abstract_state, nnx.State)
    if is_nnx:
        # StandardRestore expects a PyTree of ShapeDtypeStruct
        restore_abstract = abstract_state.to_pure_dict()
    else:
        restore_abstract = abstract_state

    if use_checkpoint_manager:
        mngr = ocp.CheckpointManager(path)
        if step is None:
            step = mngr.latest_step()
        if step is None:
            raise ValueError(f"No checkpoints found in {path}")
        restored = mngr.restore(
            step, args=ocp.args.Composite(state=ocp.args.StandardRestore(restore_abstract)))
        restored_state = restored.state
    else:
        checkpointer = ocp.StandardCheckpointer()
        if step is not None:
            load_path = os.path.join(path, f"checkpoint_{step}")
        else:
            load_path = path

        restored_state = checkpointer.restore(
            load_path, target=restore_abstract)

    if is_nnx:
        # Wrap back into nnx.State
        restored_state = nnx.State(restored_state)

    logger.info(f"Checkpoint loaded successfully from {path}")
    return restored_state
