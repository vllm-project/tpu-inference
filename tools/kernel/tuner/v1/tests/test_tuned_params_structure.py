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
"""Tests that every tuned_params.py referenced by kernel_autotune_mapping
has the expected structure required by the autotuning pipeline.

Each target file must contain:
  - A function named ``get_tuned_params``
  - A module-level dictionary named ``tuned_params_mapping``

These invariants are relied on by the shell-level monkey-patching in
.buildkite/benchmark/scripts/kernel_autotune.sh (the update_all_tuned_params_py
function).
"""

import ast
import importlib.util
import tempfile
import textwrap
from pathlib import Path

from absl.testing import absltest

from tools.kernel.tuner.v1.autotune.autotune_result_processing import \
    KernelAutoTuneResultProcessor
from tools.kernel.tuner.v1.autotune.kernel_autotune_config import \
    kernel_autotune_mapping

# The docker workspace prefix used in kernel_autotune_mapping paths.
_DOCKER_PREFIX = "/workspace/tpu_inference/"

# Resolve the repository root (the directory that contains 'tpu_inference/' and 'tools/').
_REPO_ROOT = Path(__file__).resolve().parents[5]


def _resolve_local_path(docker_path: str) -> Path:
    """Convert a docker-internal path to a local repo-relative path."""
    if docker_path.startswith(_DOCKER_PREFIX):
        return _REPO_ROOT / docker_path[len(_DOCKER_PREFIX):]
    return _REPO_ROOT / docker_path


class TunedParamsStructureTest(absltest.TestCase):
    """Validates that all tuned_params.py files have the required structure."""

    def test_kernel_autotune_mapping_is_not_empty(self):
        """The mapping must have at least one entry."""
        self.assertGreater(
            len(kernel_autotune_mapping), 0,
            "kernel_autotune_mapping is empty — no tuned_params files to validate."
        )

    def test_all_target_files_exist(self):
        """Every path in the mapping must resolve to an existing file."""
        for tuner_name, docker_path in kernel_autotune_mapping.items():
            local_path = _resolve_local_path(docker_path)
            self.assertTrue(
                local_path.is_file(),
                f"Target file for '{tuner_name}' does not exist: {local_path}")

    def test_all_target_files_have_get_tuned_params_function(self):
        """Every target file must define a function called ``get_tuned_params``."""
        for tuner_name, docker_path in kernel_autotune_mapping.items():
            local_path = _resolve_local_path(docker_path)
            if not local_path.is_file():
                self.skipTest(f"File not found, skipping: {local_path}")

            source = local_path.read_text()
            tree = ast.parse(source, filename=str(local_path))

            func_names = [
                node.name for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            ]
            self.assertIn(
                "get_tuned_params", func_names,
                f"'{tuner_name}' target file {local_path.name} is missing "
                f"'def get_tuned_params(...)'. Found functions: {func_names}")

    def test_all_target_files_have_tuned_params_mapping_dict(self):
        """Every target file must define a module-level variable called
        ``tuned_params_mapping``."""
        for tuner_name, docker_path in kernel_autotune_mapping.items():
            local_path = _resolve_local_path(docker_path)
            if not local_path.is_file():
                self.skipTest(f"File not found, skipping: {local_path}")

            source = local_path.read_text()
            tree = ast.parse(source, filename=str(local_path))

            # Collect all top-level assignment targets.
            top_level_names = set()
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            top_level_names.add(target.id)
                elif isinstance(node, ast.AnnAssign) and isinstance(
                        node.target, ast.Name):
                    top_level_names.add(node.target.id)

            self.assertIn(
                "tuned_params_mapping", top_level_names,
                f"'{tuner_name}' target file {local_path.name} is missing "
                f"'tuned_params_mapping' dictionary. "
                f"Found top-level names: {sorted(top_level_names)}")

    def test_no_underscore_prefixed_get_tuned_params(self):
        """The target files must NOT already contain a function called
        ``_get_tuned_params`` — that name is created by the autotuning
        monkey-patch and its presence would indicate a corrupted file."""
        for tuner_name, docker_path in kernel_autotune_mapping.items():
            local_path = _resolve_local_path(docker_path)
            if not local_path.is_file():
                self.skipTest(f"File not found, skipping: {local_path}")

            source = local_path.read_text()
            tree = ast.parse(source, filename=str(local_path))

            func_names = [
                node.name for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            ]
            self.assertNotIn(
                "_get_tuned_params", func_names,
                f"'{tuner_name}' target file {local_path.name} already "
                f"contains 'def _get_tuned_params(...)'. This function is "
                f"created by the autotuning pipeline's monkey-patch — its "
                f"presence in the source file suggests a corrupted commit.")

    def test_write_tuned_params_mapping_round_trips(self):
        """The mapping writer should preserve existing entries and add new ones."""
        processor = KernelAutoTuneResultProcessor()
        module_source = textwrap.dedent("""
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class TuningKey:
                model: str
                block: int

            @dataclass(frozen=True)
            class TunableParams:
                enabled: bool

            def get_tuned_params():
                return None

            tuned_params_mapping: dict[TuningKey, TunableParams] = {
                TuningKey(model="baseline", block=1): TunableParams(enabled=True),
            }
        """)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample_tuned_params.py"
            path.write_text(module_source, encoding="utf-8")

            spec = importlib.util.spec_from_file_location(
                "sample_tuned_params_module", path)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            module.tuned_params_mapping.update({
                module.TuningKey(model="new", block=2):
                module.TunableParams(enabled=False),
            })

            processor._write_tuned_params_mapping(path, module)
            rewritten_source = path.read_text(encoding="utf-8")
            self.assertIn("tuned_params_mapping", rewritten_source)
            ast.parse(rewritten_source, filename=str(path))

            reloaded_spec = importlib.util.spec_from_file_location(
                "sample_tuned_params_module_reloaded", path)
            self.assertIsNotNone(reloaded_spec)
            self.assertIsNotNone(reloaded_spec.loader)
            reloaded_module = importlib.util.module_from_spec(reloaded_spec)
            reloaded_spec.loader.exec_module(reloaded_module)

            expected = {
                reloaded_module.TuningKey(model="baseline", block=1):
                reloaded_module.TunableParams(enabled=True),
                reloaded_module.TuningKey(model="new", block=2):
                reloaded_module.TunableParams(enabled=False),
            }
            self.assertEqual(reloaded_module.tuned_params_mapping, expected)

    def test_metric_evaluation_helper(self):
        """The metric evaluation helper should capture delta and verdicts."""
        processor = KernelAutoTuneResultProcessor()
        lower_is_better_metrics = {"MedianITL"}
        threshold = 0.004

        cases = [
            ("Throughput", 100.0, 120.0, 0.2, "IMPROVED"),
            ("MedianITL", 100.0, 95.0, 0.05, "IMPROVED"),
            ("Throughput", 100.0, 90.0, -0.1, "REGRESSION"),
            ("Throughput", 0.0, 10.0, None, "BASELINE_ZERO"),
            ("Throughput", 1e-12, 10.0, None, "BASELINE_ZERO"),
            ("Throughput", 100.0, None, None, "MISSING"),
            ("Throughput", 100.0, 100.3, 0.003, "NEUTRAL"),
        ]

        for metric, baseline, tuned, expected_delta, expected_verdict in cases:
            with self.subTest(metric=metric, baseline=baseline, tuned=tuned):
                delta, verdict = processor._evaluate_metric_result(
                    baseline, tuned, metric, threshold,
                    lower_is_better_metrics)
                if expected_delta is None:
                    self.assertIsNone(delta)
                else:
                    self.assertAlmostEqual(delta, expected_delta)
                self.assertEqual(verdict, expected_verdict)

    def test_should_create_pr_gate(self):
        """The PR gate should only allow PR creation under the expected conditions."""
        processor = KernelAutoTuneResultProcessor()
        cases = [
            (True, False, False, True),
            (False, False, False, False),
            (True, True, False, False),
            (True, False, True, False),
        ]
        for monitor_improved, has_regression, hard_blocker, expected in cases:
            with self.subTest(monitor_improved=monitor_improved,
                              has_regression=has_regression,
                              hard_blocker=hard_blocker):
                self.assertEqual(
                    processor._should_create_pr(monitor_improved,
                                                has_regression, hard_blocker),
                    expected)


if __name__ == "__main__":
    absltest.main()
