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
"""BVT unit tests for the ``vllm_jax`` <-> ``tpu_inference`` singleton alias
and transitional empty meta-package definitions."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_fresh_python(code: str,
                      extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}"
                         if existing_pythonpath else str(REPO_ROOT))
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.fail(f"Subprocess failed (exit {proc.returncode}):\n"
                    f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


@pytest.mark.bvt
def test_import_vllm_jax_first_preserves_singleton_identity() -> None:
    _run_fresh_python("""
        import sys
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import vllm_jax
            from vllm_jax import envs as vj_envs
            from vllm_jax.logger import init_logger as vj_init_logger
            import tpu_inference
            from tpu_inference import envs as ti_envs
            from tpu_inference.logger import init_logger as ti_init_logger

        assert vllm_jax is tpu_inference
        assert sys.modules["vllm_jax"] is sys.modules["tpu_inference"]
        assert vj_envs is ti_envs
        # `from vllm_jax import envs` reads the existing attribute and
        # registers no sys.modules entry; import the submodule to check it.
        import vllm_jax.envs
        assert sys.modules["vllm_jax.envs"] is sys.modules["tpu_inference.envs"]
        assert vj_init_logger is ti_init_logger
        assert vllm_jax.__name__ == "tpu_inference"
        assert vj_envs.__name__ == "tpu_inference.envs"

        alias_warnings = [
            w for w in caught
            if "tpu_inference" in str(w.message) or "vllm_jax" in str(w.message)
        ]
        assert not alias_warnings, alias_warnings
    """)


@pytest.mark.bvt
def test_import_tpu_inference_first_preserves_singleton_identity() -> None:
    _run_fresh_python("""
        import sys

        import tpu_inference
        from tpu_inference import envs as ti_envs
        import vllm_jax
        from vllm_jax import envs as vj_envs

        assert tpu_inference is vllm_jax
        assert sys.modules["vllm_jax"] is sys.modules["tpu_inference"]
        assert ti_envs is vj_envs
        # `from vllm_jax import envs` reads the existing attribute and
        # registers no sys.modules entry; import the submodule to check it.
        import vllm_jax.envs
        assert sys.modules["vllm_jax.envs"] is sys.modules["tpu_inference.envs"]
    """)


@pytest.mark.bvt
def test_find_spec_and_deep_submodule_import() -> None:
    _run_fresh_python("""
        import importlib.util
        import sys

        assert "vllm_jax" not in sys.modules
        assert "tpu_inference" not in sys.modules

        root_spec = importlib.util.find_spec("vllm_jax")
        assert root_spec is not None

        sub_spec = importlib.util.find_spec("vllm_jax.envs")
        assert sub_spec is not None

        import vllm_jax.envs
        assert sys.modules["vllm_jax"] is sys.modules["tpu_inference"]
        assert sys.modules["vllm_jax.envs"] is sys.modules["tpu_inference.envs"]
    """)


@pytest.mark.bvt
def test_transitional_meta_packages_declare_empty_wheels() -> None:
    _run_fresh_python(
        """
        import runpy
        import sys
        import types
        from unittest.mock import patch

        calls = []
        fake_setuptools = types.SimpleNamespace(
            setup=lambda **kw: calls.append(kw))
        with patch.dict(sys.modules, {"setuptools": fake_setuptools}):
            runpy.run_path("packaging/tpu_inference_meta/setup.py")
            runpy.run_path("packaging/vllm_tpu_meta/setup.py")

        assert len(calls) == 2
        core_meta, combined_meta = calls

        assert core_meta["name"] == "tpu-inference"
        assert core_meta["version"] == "0.99.0rc1"
        assert core_meta["packages"] == []
        assert core_meta["install_requires"] == ["vllm-jax-core==0.99.0rc1"]
        assert "entry_points" not in core_meta

        assert combined_meta["name"] == "vllm-tpu"
        assert combined_meta["version"] == "0.99.0rc1"
        assert combined_meta["packages"] == []
        assert combined_meta["install_requires"] == ["vllm-jax==0.99.0rc1"]
        assert "entry_points" not in combined_meta
        """,
        extra_env={"VLLM_VERSION_OVERRIDE": "0.99.0rc1"},
    )
