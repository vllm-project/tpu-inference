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
"""Transitional compatibility package aliasing ``vllm_jax`` to ``tpu_inference``.

While physical source files remain under ``tpu_inference/`` during Stage 2,
importing ``vllm_jax`` (or any ``vllm_jax.*`` submodule) resolves to the exact
same module object in ``sys.modules`` as ``tpu_inference`` so module-level
state, plugin registration, and XLA/JAX compilation caches are never duplicated.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from types import ModuleType

_CANONICAL = "tpu_inference"
_ALIAS = "vllm_jax"


class _SingletonAliasLoader(importlib.abc.Loader):
    """Loader that binds an alias module name to its canonical singleton."""

    def __init__(self, alias_name: str, target_name: str) -> None:
        self._alias_name = alias_name
        self._target_name = target_name
        self._saved_attrs: tuple[object, ...] | None = None

    def create_module(self,
                      spec: importlib.machinery.ModuleSpec) -> ModuleType:
        mod = importlib.import_module(self._target_name)
        self._saved_attrs = (
            getattr(mod, "__name__", None),
            getattr(mod, "__package__", None),
            getattr(mod, "__spec__", None),
            getattr(mod, "__loader__", None),
            getattr(mod, "__path__", None),
        )
        sys.modules[self._alias_name] = mod
        return mod

    def exec_module(self, module: ModuleType) -> None:
        # CPython's module_from_spec overwrites __name__, __package__,
        # __loader__, __path__, and __spec__ on the returned object with the
        # alias spec's attributes before calling exec_module(). Restore the
        # canonical attributes so relative imports and coverage attribution
        # inside tpu_inference continue to see their canonical identity.
        if self._saved_attrs is not None:
            name, pkg, spec, loader, path = self._saved_attrs
            if name is not None:
                module.__name__ = name  # type: ignore[assignment]
            if pkg is not None:
                module.__package__ = pkg  # type: ignore[assignment]
            if spec is not None:
                module.__spec__ = spec  # type: ignore[assignment]
            if loader is not None:
                module.__loader__ = loader  # type: ignore[assignment]
            if path is not None:
                module.__path__ = path  # type: ignore[assignment]
        sys.modules[self._alias_name] = module
        sys.modules[self._target_name] = module


class _VllmJaxAliasFinder(importlib.abc.MetaPathFinder):
    """MetaPathFinder redirecting ``vllm_jax.*`` imports to ``tpu_inference.*``."""

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != _ALIAS and not fullname.startswith(f"{_ALIAS}."):
            return None
        target_name = f"{_CANONICAL}{fullname[len(_ALIAS):]}"
        target_spec = importlib.util.find_spec(target_name)
        if target_spec is None:
            return None
        is_pkg = target_spec.submodule_search_locations is not None
        spec = importlib.machinery.ModuleSpec(
            name=fullname,
            loader=_SingletonAliasLoader(fullname, target_name),
            origin=target_spec.origin,
            is_package=is_pkg,
        )
        if is_pkg and target_spec.submodule_search_locations is not None:
            spec.submodule_search_locations = list(
                target_spec.submodule_search_locations)
        return spec


def _install_alias() -> ModuleType:
    if not any(
            type(f).__name__ == "_VllmJaxAliasFinder" for f in sys.meta_path):
        sys.meta_path.insert(0, _VllmJaxAliasFinder())
    root = importlib.import_module(_CANONICAL)
    sys.modules[_ALIAS] = root
    return root


_install_alias()
