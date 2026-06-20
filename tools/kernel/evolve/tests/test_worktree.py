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
"""Tests for candidate worktree isolation."""

import sys

import pytest

from tools.kernel.evolve.worktree import (import_candidate_module,
                                          materialize_in_tmp_tree)


def test_import_candidate_module_evaluates_source():
    src = "X = 42\ndef f(y):\n    return X + y\n"
    with import_candidate_module(src, name_hint="t1") as mod:
        assert mod.X == 42
        assert mod.f(8) == 50


def test_import_candidate_module_cleans_up_after_exit():
    src = "X = 1\n"
    with import_candidate_module(src, name_hint="t2") as mod:
        mod_name = mod.__name__
        assert mod_name in sys.modules
    assert mod_name not in sys.modules


def test_import_candidate_module_unique_namespace_per_candidate():
    src = "X = 1\n"
    with import_candidate_module(src, name_hint="t3") as m1:
        with import_candidate_module(src, name_hint="t3") as m2:
            assert m1.__name__ != m2.__name__  # uuid suffixes differ
            assert m1.X == m2.X == 1


def test_import_candidate_module_propagates_syntax_error():
    src = "def f(:\n  pass\n"
    with pytest.raises(SyntaxError):
        with import_candidate_module(src, name_hint="bad"):
            pass


def test_materialize_in_tmp_tree_creates_file_at_rel_path(tmp_path):
    repo = tmp_path / "repo"
    pkg = repo / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "target.py").write_text("ORIGINAL = True\n")
    with materialize_in_tmp_tree(repo_root=repo,
                                 rel_path="pkg/target.py",
                                 mutated_source="MUTATED = True\n") as root:
        new_file = root / "pkg" / "target.py"
        assert new_file.read_text() == "MUTATED = True\n"
        # __init__.py should be preserved.
        assert (root / "pkg" / "__init__.py").exists()
