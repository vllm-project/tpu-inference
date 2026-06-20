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
"""Isolated workspace per candidate evaluation.

Why this exists: a broken diff could leave the source tree in an
unimportable state, corrupting other concurrent / subsequent candidates.
The worktree approach gives each candidate its own copy of the target
module, imported as a temporary uniquely-named module so the live
production kernel under ``tpu_inference`` is never modified.

Two strategies are supported, chosen by the caller:

* ``import_candidate_module`` — fastest. Writes the mutated source to a
  ``tempfile.NamedTemporaryFile`` and imports it via ``importlib.util``
  under a unique module name. No git worktree, no path manipulation. Best
  when the diff touches a *single* file and the file is self-contained.

* ``materialize_in_tmp_tree`` — slower but more robust. Mirrors the
  target file's directory tree under a temp dir with the mutated file
  replacing the original at the correct relative path, so cross-imports
  within the module work. Cleaned up on context exit.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from types import ModuleType
from typing import Iterator


@contextlib.contextmanager
def import_candidate_module(
    mutated_source: str,
    *,
    name_hint: str = "evolved",
) -> Iterator[ModuleType]:
    """Import ``mutated_source`` as a unique module; clean up afterward.

    The module's ``__name__`` is namespaced under ``__evolve_candidates__``
    to keep it well-separated from production modules.
    """
    suffix = uuid.uuid4().hex[:10]
    mod_name = f"__evolve_candidates__.{name_hint}_{suffix}"
    parent_name = "__evolve_candidates__"

    # Register the namespace package if it isn't already.
    if parent_name not in sys.modules:
        parent = ModuleType(parent_name)
        parent.__path__ = []  # marker — namespace package
        sys.modules[parent_name] = parent

    tmp_dir = tempfile.mkdtemp(prefix="evolve_")
    try:
        tmp_path = Path(tmp_dir) / f"{name_hint}_{suffix}.py"
        tmp_path.write_text(mutated_source)
        spec = importlib.util.spec_from_file_location(mod_name, tmp_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"importlib could not load spec for {tmp_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(mod_name, None)
        shutil.rmtree(tmp_dir, ignore_errors=True)


@contextlib.contextmanager
def materialize_in_tmp_tree(
    *,
    repo_root: str | os.PathLike,
    rel_path: str,
    mutated_source: str,
    extra_sources: dict[str, str] | None = None,
) -> Iterator[Path]:
    """Copy the relevant subtree to a temp dir with ``mutated_source`` swapped in.

    ``rel_path`` is the path of the target file relative to ``repo_root``.
    ``extra_sources`` is an optional mapping of additional rel-paths to
    contents (useful for multi-file mutations, though they're rare).

    Yields the path to the temp dir. Caller is responsible for prepending
    it to ``sys.path`` if they want to import from it.
    """
    repo_root = Path(repo_root)
    tmp_root = Path(tempfile.mkdtemp(prefix="evolve_tree_"))
    try:
        target_rel = Path(rel_path)
        # Mirror just the immediate package directory of the target file.
        src_pkg = repo_root / target_rel.parent
        dst_pkg = tmp_root / target_rel.parent
        if src_pkg.exists():
            shutil.copytree(src_pkg, dst_pkg, dirs_exist_ok=True)
        else:
            dst_pkg.mkdir(parents=True, exist_ok=True)
        (tmp_root / target_rel).write_text(mutated_source)
        if extra_sources:
            for rel, content in extra_sources.items():
                p = tmp_root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)
        yield tmp_root
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
