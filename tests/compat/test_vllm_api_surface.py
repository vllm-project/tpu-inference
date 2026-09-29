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
"""Guards against upstream vLLM API drift.

The integration pipeline builds against vLLM main, so what usually stalls LKG
promotion is not a perf or accuracy regression -- it is a Python-level API
break: a symbol that moved, an ABC that grew a method, a module attribute that
was renamed. Any one of those takes out every downstream TPU test at once, so
an hour of TPU time is an expensive way to learn that an import is dead.

These checks run on CPU in seconds. They enumerate what tpu_inference actually
contains rather than listing the classes that happened to break before, so an
upstream refactor nobody has seen yet is caught the same way.

They are a fast first line, not a replacement for the TPU suite: a break that
only manifests once a kernel runs still needs a real device.
"""

import ast
import importlib
import inspect
import pkgutil
from pathlib import Path

import pytest

import tpu_inference

# Modules that genuinely cannot be imported standalone on a CPU host. Every
# entry is a hole in the sweep, so prefer fixing the module over adding one.
_UNIMPORTABLE_PREFIXES = ()


def _module_names():
    for info in pkgutil.walk_packages(tpu_inference.__path__,
                                      prefix="tpu_inference."):
        if info.name.startswith(_UNIMPORTABLE_PREFIXES):
            continue
        yield info.name


@pytest.fixture(scope="module")
def swept_modules():
    """Import every tpu_inference module, collecting failures rather than raising.

    One dead import usually means dozens of dead imports; reporting them all in
    a single run is the difference between one fix-up PR and a dozen.
    """
    imported = {}
    failures = []
    for name in _module_names():
        try:
            imported[name] = importlib.import_module(name)
        except BaseException as exc:  # noqa: BLE001 - report, don't stop
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    return imported, failures


def _declaring_class(cls, attr):
    """Return the class in cls.__mro__ that declares `attr`, or None."""
    for base in cls.__mro__:
        if attr in vars(base):
            return base
    return None


def _owned_classes(modules):
    """Every class defined by tpu_inference itself, deduplicated."""
    seen = {}
    for module in modules.values():
        for obj in vars(module).values():
            if (inspect.isclass(obj) and getattr(
                    obj, "__module__", "").startswith("tpu_inference")):
                seen[id(obj)] = obj
    return list(seen.values())


def test_every_module_imports(swept_modules):
    _, failures = swept_modules
    assert not failures, (
        "tpu_inference modules that no longer import against "
        "this vLLM:\n  " + "\n  ".join(failures))


def test_no_vllm_abstract_method_left_unimplemented(swept_modules):
    """Catch upstream adding an abstract method to a base class we subclass.

    Only abstract methods *declared by vLLM* count. An abstract method that
    tpu_inference declares on its own intermediate base is intentional, and
    Python populates ``__abstractmethods__`` identically for both, so the
    declaring class is what tells them apart.
    """
    imported, _ = swept_modules
    offenders = []
    for cls in _owned_classes(imported):
        for attr in sorted(getattr(cls, "__abstractmethods__", ())):
            owner = _declaring_class(cls, attr)
            if owner is None or not owner.__module__.startswith("vllm"):
                continue
            offenders.append(
                f"{cls.__module__}.{cls.__qualname__} does not implement "
                f"{owner.__module__}.{owner.__qualname__}.{attr}")

    assert not offenders, ("vLLM abstract methods with no TPU implementation "
                           "(instantiating these raises TypeError):\n  " +
                           "\n  ".join(sorted(offenders)))


def test_vllm_module_attributes_still_exist(swept_modules):
    """Catch ``vllm_module.some_helper`` reads of a symbol upstream removed.

    A plain import sweep misses these: the module still imports, and the
    AttributeError only surfaces when the line runs -- often deep in a
    quantization path that needs an eight-chip host to reach.

    Module aliases are resolved from the imported namespace rather than from
    the import statement, so the mapping is exact. The AST only supplies the
    attribute names.
    """
    imported, _ = swept_modules
    missing = set()

    for name, module in imported.items():
        source_path = getattr(module, "__file__", None)
        if not source_path or not source_path.endswith(".py"):
            continue

        aliases = {
            alias: value
            for alias, value in vars(module).items() if inspect.ismodule(value)
            and value.__name__.split(".")[0] == "vllm"
        }
        if not aliases:
            continue

        try:
            tree = ast.parse(Path(source_path).read_text())
        except (OSError, SyntaxError):
            continue

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)):
                continue
            target = aliases.get(node.value.id)
            if target is not None and not hasattr(target, node.attr):
                missing.add(f"{name}:{node.lineno} reads "
                            f"{target.__name__}.{node.attr}, which this vLLM "
                            f"no longer defines")

    assert not missing, ("vLLM module attributes that no longer exist:\n  " +
                         "\n  ".join(sorted(missing)))
