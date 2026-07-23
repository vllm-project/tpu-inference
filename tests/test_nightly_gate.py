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
"""Unit tests for the BVT_ONLY collection gate defined in tests/conftest.py.

The gate is the riskiest logic in the per-push/nightly split, so it is tested
directly against the real conftest functions with synthetic collection items.
No TPU/JAX is executed, but importing the conftest does ``import jax``, so this
test requires jax to be importable (always true in the jax CI step).
"""

import importlib.util
import pathlib
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_safe

# Import the real conftest module by path so we exercise the shipped gate logic.
_CONFTEST_PATH = pathlib.Path(__file__).parent / "conftest.py"
_spec = importlib.util.spec_from_file_location("_tests_conftest_under_test",
                                               _CONFTEST_PATH)
_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conftest)


class _FakeItem:

    def __init__(self, nodeid, bvt):
        self.nodeid = nodeid
        self._bvt = bvt

    def get_closest_marker(self, name):
        if name == "bvt" and self._bvt:
            return object()
        return None


class _FakeHook:

    def __init__(self):
        self.deselected = []

    def pytest_deselected(self, items):
        self.deselected.extend(items)


def test_partition_keeps_only_bvt_in_marked_module():
    # module "f" declares bvt -> only its bvt items survive;
    # module "g" has no bvt marker -> kept entirely.
    items = [
        _FakeItem("f::a", True),
        _FakeItem("f::b", False),
        _FakeItem("g::c", False),
    ]
    keep, drop = _conftest.partition_bvt_only(items)
    assert [it.nodeid for it in keep] == ["f::a", "g::c"]
    assert [it.nodeid for it in drop] == ["f::b"]


def test_partition_no_bvt_anywhere_keeps_all():
    items = [_FakeItem("f::a", False), _FakeItem("g::b", False)]
    keep, drop = _conftest.partition_bvt_only(items)
    assert [it.nodeid for it in keep] == ["f::a", "g::b"]
    assert drop == []


def test_modifyitems_default_runs_everything(monkeypatch):
    monkeypatch.delenv("BVT_ONLY", raising=False)
    items = [_FakeItem("f::a", True), _FakeItem("f::b", False)]
    config = SimpleNamespace(hook=_FakeHook())
    _conftest.pytest_collection_modifyitems(config, items)
    assert [it.nodeid for it in items] == ["f::a", "f::b"]
    assert config.hook.deselected == []


def test_modifyitems_bvt_only_filters_in_place(monkeypatch):
    monkeypatch.setenv("BVT_ONLY", "1")
    items = [
        _FakeItem("f::a", True),
        _FakeItem("f::b", False),
        _FakeItem("g::c", False),
    ]
    config = SimpleNamespace(hook=_FakeHook())
    _conftest.pytest_collection_modifyitems(config, items)
    assert [it.nodeid for it in items] == ["f::a", "g::c"]
    assert len(config.hook.deselected) == 1
