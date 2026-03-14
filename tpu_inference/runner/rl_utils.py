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

"""Utilities for RL (Reinforcement Learning) integration.

These helpers provide **direct, in-process** access to the model runner's
state (weights) from an ``LLM`` instance.  They bypass ``collective_rpc``
entirely, so JAX PyTrees are never serialized.

Requirements
------------
* ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` – the engine must run in the same
  process so that ``model_executor`` is available on the ``LLMEngine``.

Typical RL weight-sync loop::

    from vllm import LLM, SamplingParams
    from tpu_inference.runner.rl_utils import get_weights, update_weights

    llm = LLM(model=..., ...)
    weights = get_weights(llm)          # nnx.State PyTree
    # ... train / modify weights ...
    update_weights(llm, new_weights)    # install new weights
    llm.generate(prompts, params)       # serve with updated weights
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jaxtyping

if TYPE_CHECKING:
    from vllm import LLM

    from tpu_inference.runner.tpu_runner import TPUModelRunner


def _get_model_runner(llm: "LLM") -> "TPUModelRunner":
    """Return the :class:`TPUModelRunner` from an ``LLM`` instance.

    This traverses the object graph:
    ``llm.llm_engine.model_executor.driver_worker.worker.model_runner``

    Raises
    ------
    RuntimeError
        If ``model_executor`` is not available (e.g. multiprocessing is
        enabled).
    """
    engine = llm.llm_engine
    executor = getattr(engine, "model_executor", None)
    if executor is None:
        raise RuntimeError(
            "model_executor is not available on the LLMEngine. "
            "RL weight-sync utilities require in-process mode. "
            "Set VLLM_ENABLE_V1_MULTIPROCESSING=0 before creating "
            "the LLM instance."
        )
    # executor.driver_worker is a WorkerWrapperBase whose .worker is
    # the actual TPUWorker, which owns the model_runner.
    return executor.driver_worker.worker.model_runner


def get_weights(llm: "LLM") -> jaxtyping.PyTree:
    """Return the current model weights (``nnx.State``) from *llm*.

    The returned object is the **live** state used by the model runner –
    it is *not* a copy.  Mutating the leaves in-place will affect
    subsequent inference calls.
    """
    return _get_model_runner(llm).state


def update_weights(llm: "LLM", new_weights: jaxtyping.PyTree) -> None:
    """Replace the model weights on *llm* with *new_weights*.

    *new_weights* must be a PyTree with the exact same structure,
    leaf shapes, and shardings as the current ``model_runner.state``.
    No key-mapping, transposing, or resharding is performed – the
    caller is responsible for providing weights in the correct layout.

    Parameters
    ----------
    llm:
        The ``LLM`` instance whose weights should be updated.
    new_weights:
        A complete state PyTree (e.g. ``nnx.State``) to install.

    Raises
    ------
    ValueError
        If *new_weights* is ``None``, or if its structure, shapes, or
        shardings do not match the current state.
    """
    _get_model_runner(llm)._update_weights(new_weights)
