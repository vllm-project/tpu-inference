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

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Iterator, Optional

from flax import nnx

if TYPE_CHECKING:
    import jax
    from jax.sharding import Mesh
    from vllm.config import VllmConfig


class JaxModule(nnx.Module):
    """Base module for JAX layers, extending flax.nnx.Module.
    """

    @classmethod
    def create_model(
        cls,
        vllm_config: "VllmConfig",
        rng: "jax.Array",
        mesh: "Mesh",
    ) -> Optional["JaxModule"]:
        """Create and return a fully initialized model, ready for inference.

        Override this classmethod in model subclasses that need full control
        over model creation — for example, models that manage their own
        sharding internally and do not need the default ``@jax.jit`` wrapper
        provided by the loader.

        When overridden, the returned model should be fully sharded, have
        weights loaded (or random-initialized for dummy mode), and have any
        quantization / cache initialization already applied.

        The default implementation returns ``None``, which signals the loader
        to use its standard creation pipeline.

        Args:
            vllm_config: The VllmConfig for this model.
            rng: The JAX PRNG key.
            mesh: The device mesh.

        Returns:
            A fully initialized ``nnx.Module`` ready for inference, or
            ``None`` to fall back to the default loader pipeline.
        """
        del vllm_config, rng, mesh  # unused in default implementation
        return None

    def _get_name(self) -> str:
        return self.__class__.__name__

    def named_parameters(self,
                         prefix: str = "",
                         recurse=True) -> Iterator[tuple[str, nnx.Param]]:
        """Yields the named parameters of the module.
        
        Arguments:
            prefix: Prefix to add to the parameter names.
            recurse: If True, then yields parameters of this module and all submodules.
                Otherwise, yields only parameters that are direct members of this module.

        Yields:
            (string, Param): Tuple containing a name and parameter
        """
        for name, param in self.__dict__.items():
            if isinstance(param, nnx.Param):
                yield (f"{prefix}.{name}" if prefix else name), param

        if not recurse:
            return

        for name, child in self.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_parameters(prefix=child_prefix,
                                              recurse=True)

    def named_children(
            self) -> Iterator[tuple[str, "JaxModule | JaxModuleList"]]:
        """Returns an iterator over immediate children modules.
        
        Yields:
            (string, Module | list): Tuple containing a name and child module
        """
        for name, value in self.__dict__.items():
            if isinstance(value, JaxModule):
                yield name, value
            elif isinstance(value, list) or isinstance(value, nnx.List):
                yield name, JaxModuleList(value)


class JaxModuleList(nnx.List):
    """A list container for JaxModule objects."""

    def __init__(self, modules: Iterable[JaxModule]):
        """Initializes the JaxModuleList.

        Args:
            modules: An optional list of JaxModule objects to initialize the list.
        """
        super().__init__()
        for module in modules:
            self.append(module)

    def _get_name(self) -> str:
        return self.__class__.__name__

    def named_parameters(self,
                         prefix: str = "",
                         recurse=True) -> Iterator[tuple[str, nnx.Param]]:
        """Yields the named parameters of all modules in the list."""

        for idx, module in enumerate(self):
            module_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            yield from module.named_parameters(prefix=module_prefix,
                                               recurse=recurse)

    def named_children(
            self) -> Iterator[tuple[str, "JaxModule | JaxModuleList"]]:
        """Returns an iterator over the modules in the list with their indices as names.

        Yields:
            (string, JaxModule): Tuple containing the index as a string and the module
        """
        for idx, item in enumerate(self):
            if isinstance(item, JaxModule):
                yield str(idx), item
            elif isinstance(item, list):
                yield str(idx), JaxModuleList(item)
