from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.cpu_safe

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / ".buildkite" / "scripts" / "select_kube_tests.py"
_SPEC = importlib.util.spec_from_file_location("select_kube_tests", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_RULES = _MODULE.load_ownership(
    _REPO_ROOT / ".buildkite" / "kubernetes" / "test_ownership.json")


def test_docs_only_selects_nothing():
    plan = _MODULE.select_tests(["docs/cache.md", "README.md"], _RULES,
                                event="pull_request")

    assert plan["mode"] == "docs-only"
    assert plan["cpu_steps"] == []
    assert plan["tpu_steps"] == []


def test_owned_change_selects_targeted_step_and_cpu():
    plan = _MODULE.select_tests(
        ["tests/e2e/test_speculative_decoding.py"],
        _RULES,
        event="pull_request",
    )

    assert plan["mode"] == "targeted"
    assert plan["cpu_steps"] == ["kube_cpu_safe_tests"]
    assert plan["tpu_steps"] == ["kube_e2e_speculative_decoding"]


def test_unowned_code_widens_to_default_matrix():
    plan = _MODULE.select_tests(["tpu_inference/new_area/module.py"], _RULES,
                                event="pull_request")

    assert plan["mode"] == "default-fallback"
    assert plan["unowned_files"] == ["tpu_inference/new_area/module.py"]
    assert set(plan["tpu_steps"]) == _MODULE.DEFAULT_TPU_STEPS


@pytest.mark.parametrize("event", ["main", "nightly", "release"])
def test_protected_events_select_full_matrix(event):
    plan = _MODULE.select_tests(["tests/test_envs.py"], _RULES, event=event)

    assert plan["mode"] == "full"
    assert plan["full_matrix"] is True
    assert set(plan["tpu_steps"]) == (
        _MODULE.DEFAULT_TPU_STEPS | _MODULE.FULL_ONLY_STEPS)


def test_ci_change_selects_full_matrix():
    plan = _MODULE.select_tests([".buildkite/pipeline_kube.yaml"], _RULES,
                                event="pull_request")

    assert plan["mode"] == "full"
    assert plan["full_matrix"] is True
