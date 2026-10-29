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
"""Unit tests for .buildkite/scripts/bootstrap.sh and pipeline_config.sh.

Runs the real bootstrap.sh against stub `buildkite-agent`, `git`, `curl` and
`sleep` binaries, in a scratch working directory that holds a symlink to the
repo's `.buildkite/`, so nothing is written into the checkout. `jq` and PyYAML
are real: the GitHub label extraction uses `jq`, and the benchmark pipeline
generator imports `yaml`.

tests/conftest.py imports jax, so locally run this with --noconftest:

    python3 -m pytest tests/scripts/test_bootstrap.py --noconftest
"""

import json
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

import pytest

try:
    import yaml
except ImportError:  # pragma: no cover - environment-dependent
    yaml = None

pytestmark = [
    pytest.mark.skipif(yaml is None,
                       reason="PyYAML is required: pip install pyyaml"),
    pytest.mark.skipif(shutil.which("jq") is None,
                       reason="jq is required on PATH"),
]

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BOOTSTRAP = _REPO_ROOT / ".buildkite" / "scripts" / "bootstrap.sh"
_REAL_GIT = shutil.which("git") or "/usr/bin/git"
_FAKE_SHA = "0123456789abcdef0123456789abcdef01234567"

_FAKE_AGENT = """#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "${TEST_AGENT_LOG}/calls.log"
if [[ "$1" == "pipeline" && "$2" == "upload" ]]; then
  shopt -s nullglob
  if [[ "${3:-}" == "--dry-run" ]]; then
    dfiles=("${TEST_AGENT_LOG}"/dry_run_*.yml)
    cat > "${TEST_AGENT_LOG}/dry_run_${#dfiles[@]}.yml"
    echo '{"steps": [{"label": "stub", "agents": {"queue": "kube"}}]}'
  else
    ufiles=("${TEST_AGENT_LOG}"/upload_*.yml)
    cat > "${TEST_AGENT_LOG}/upload_${#ufiles[@]}.yml"
  fi
elif [[ "$1" == "meta-data" && "$2" == "set" && $# -eq 3 ]]; then
  # Value piped on stdin (changed_files): drain it, or the writer takes
  # SIGPIPE and bootstrap's pipefail turns that into exit 141.
  cat > /dev/null
fi
"""

# fetch is a no-op; every diff names one source file so the docs-only skip
# does not fire; rev-parse (benchmark CODE_HASH) and ls-remote (integration
# pipeline) are canned because the scratch cwd is not a git checkout and the
# test must not reach the network.
_FAKE_GIT = f"""#!/usr/bin/env bash
if [[ "${{1:-}}" == "fetch" ]]; then
  exit 0
fi
if [[ "${{1:-}}" == "diff" || "${{1:-}}" == "diff-tree" ]]; then
  echo "tpu_inference/runner/tpu_runner.py"
  exit 0
fi
if [[ "${{1:-}}" == "rev-parse" ]]; then
  echo "{_FAKE_SHA}"
  exit 0
fi
if [[ "${{1:-}}" == "ls-remote" ]]; then
  echo "{_FAKE_SHA}	HEAD"
  exit 0
fi
exec {_REAL_GIT} "$@"
"""

# Logs its invocation, then writes a GitHub pull-request JSON whose labels
# come from TEST_PR_LABELS (comma separated) to the path after -o.
_FAKE_CURL = """#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "${TEST_AGENT_LOG}/curl.log"
out=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "-o" ]]; then out="$2"; shift; fi
  shift
done
labels_json=""
IFS=',' read -ra labels <<< "${TEST_PR_LABELS:-}"
for l in "${labels[@]}"; do
  [[ -z "$l" ]] && continue
  labels_json+="${labels_json:+,}{\\"name\\":\\"$l\\"}"
done
echo "{\\"labels\\":[${labels_json}]}" > "$out"
"""

_FAKE_SLEEP = """#!/usr/bin/env bash
exit 0
"""


class BootstrapRun:
    """What one bootstrap.sh run did, as seen by the stubs."""

    def __init__(self, tmp_path: pathlib.Path,
                 proc: subprocess.CompletedProcess):
        self.returncode = proc.returncode
        self.stdout = proc.stdout
        self.stderr = proc.stderr
        self.calls = self._lines(tmp_path / "calls.log")
        self.curl_calls = self._lines(tmp_path / "curl.log")
        self.uploads = [
            p.read_text() for p in sorted(tmp_path.glob("upload_*.yml"))
        ]
        self.dry_runs = [
            p.read_text() for p in sorted(tmp_path.glob("dry_run_*.yml"))
        ]

    @staticmethod
    def _lines(path: pathlib.Path) -> list[str]:
        return path.read_text().splitlines() if path.exists() else []


def run_bootstrap(labels: str = "",
                  check: bool = True,
                  **env: str) -> BootstrapRun:
    """Run bootstrap.sh against stub binaries.

    `labels` is the PR's label list, served both as
    BUILDKITE_PULL_REQUEST_LABELS (read by determine_job_priority) and by the
    stub GitHub API (read for `ready`).
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for name, script in (
            ("buildkite-agent", _FAKE_AGENT),
            ("git", _FAKE_GIT),
            ("curl", _FAKE_CURL),
            ("sleep", _FAKE_SLEEP),
        ):
            path = bin_dir / name
            path.write_text(script)
            path.chmod(path.stat().st_mode | stat.S_IEXEC)
        # The benchmark generator runs `python3` from PATH and needs PyYAML:
        # make that the interpreter running this test.
        (bin_dir / "python3").symlink_to(sys.executable)

        # Scratch cwd: bootstrap.sh addresses .buildkite/ by relative path and
        # writes its scratch files (notification YAML, GitHub response) to cwd.
        work = tmp_path / "work"
        work.mkdir()
        (work / ".buildkite").symlink_to(_REPO_ROOT / ".buildkite")

        # Provide a clean base environment so host Buildkite vars never leak in.
        run_env = {
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "HOME": str(tmp_path),
            "TEST_AGENT_LOG": str(tmp_path),
            "TEST_PR_LABELS": labels,
            "BUILDKITE_COMMIT": "HEAD",
            "BUILDKITE_BRANCH": "feature-branch",
            "BUILDKITE_PULL_REQUEST": "false",
            "BUILDKITE_PULL_REQUEST_LABELS": labels,
            "BUILDKITE_PIPELINE_SLUG": "tpu-inference-ci",
            "BUILDKITE_SOURCE": "webhook",
            "BUILDKITE_MESSAGE": "",
            # Empty: the creator email is the concurrency group's fallback
            # key, so setting it would cap every PR test's uploads.
            "BUILDKITE_BUILD_CREATOR_EMAIL": "",
            "CI_FLEET": "bare",
            **env,
        }
        proc = subprocess.run(
            ["bash", str(_BOOTSTRAP)],
            cwd=work,
            env=run_env,
            capture_output=True,
            text=True,
            check=check,
        )
        return BootstrapRun(tmp_path, proc)


def _nightly_gated(uploads: list[str]) -> list[str]:
    """The uploads that gate steps on NIGHTLY (pipeline_jax.yml, nightly_verify.yml)."""
    return [u for u in uploads if "NIGHTLY" in u]


class TestBootstrap(unittest.TestCase):
    """Verify priority classes, label gating and per-author concurrency cap."""

    def test_ready_pr_is_pre_merge_and_keeps_nightly_unset(self) -> None:
        run = run_bootstrap(labels="ready", BUILDKITE_PULL_REQUEST="123")
        self.assertTrue(run.uploads)
        self.assertEqual(run.dry_runs, [])
        self.assertIn("meta-data set JOB_PRIORITY 5", run.calls)
        self.assertIn("meta-data set WORKLOAD_PRIORITY pre-merge", run.calls)
        for content in run.uploads:
            self.assertTrue(content.startswith("priority: 5\n"))
        gated = _nightly_gated(run.uploads)
        self.assertTrue(gated)
        for content in gated:
            self.assertIn('build.env("NIGHTLY")', content)
        self.assertFalse(any("annotate" in c for c in run.calls))

    def test_pr_without_ready_label_fails_after_notification_only(
            self) -> None:
        run = run_bootstrap(labels="",
                            BUILDKITE_PULL_REQUEST="123",
                            check=False)
        self.assertEqual(run.returncode, 1)
        self.assertIn("Missing 'ready' label on PR. Failing build.",
                      run.stdout)
        self.assertEqual(len(run.curl_calls), 1)
        # The notification fragment goes up before the label check; nothing else.
        self.assertEqual(len(run.uploads), 1)
        parsed = yaml.safe_load(run.uploads[0])
        self.assertEqual(set(parsed), {"priority", "notify"})
        self.assertEqual(run.dry_runs, [])

    def test_oncall_fix_pr_is_oncall_priority(self) -> None:
        run = run_bootstrap(labels="ready,oncall-fix",
                            BUILDKITE_PULL_REQUEST="123")
        self.assertTrue(run.uploads)
        self.assertIn("meta-data set JOB_PRIORITY 12", run.calls)
        self.assertIn("meta-data set WORKLOAD_PRIORITY oncall-fix", run.calls)
        self.assertTrue(any("--context oncall-fix" in c for c in run.calls))
        for content in run.uploads:
            self.assertTrue(content.startswith("priority: 12\n"))

    def test_pr_with_author_caps_tpu_concurrency(self) -> None:
        run = run_bootstrap(
            labels="ready",
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_BUILD_AUTHOR_EMAIL="Alice@example.com",
        )
        self.assertTrue(run.uploads)
        # Every upload went through the dry run and came back as capped JSON.
        self.assertEqual(len(run.dry_runs), len(run.uploads))
        for raw_yaml in run.dry_runs:
            self.assertTrue(raw_yaml.startswith("priority: 5\n"))
        for content in run.uploads:
            doc = json.loads(content)
            step = doc["steps"][0]
            self.assertEqual(step["concurrency_group"],
                             "tpu-pr/alice@example.com")
            self.assertEqual(step["concurrency"], 40)
            self.assertEqual(step["concurrency_method"], "eager")

    def test_integration_pipeline_is_integration_class_without_label_fetch(
        self, ) -> None:
        run = run_bootstrap(
            BUILDKITE_PIPELINE_SLUG="tpu-vllm-integration",
            BUILDKITE_BRANCH="main",
            BUILDKITE_PULL_REQUEST="false",
        )
        self.assertTrue(run.uploads)
        self.assertIn("meta-data set JOB_PRIORITY 3", run.calls)
        self.assertIn("meta-data set WORKLOAD_PRIORITY integration", run.calls)
        self.assertIn(f"meta-data set VLLM_COMMIT_HASH {_FAKE_SHA}", run.calls)
        self.assertEqual(run.curl_calls, [])
        for content in run.uploads:
            self.assertTrue(content.startswith("priority: 3\n"))

    def test_post_merge_main_branch_is_post_merge_class(self) -> None:
        run = run_bootstrap(
            BUILDKITE_BRANCH="main",
            BUILDKITE_PULL_REQUEST="false",
        )
        self.assertTrue(run.uploads)
        self.assertIn("meta-data set JOB_PRIORITY 10", run.calls)
        self.assertIn("meta-data set WORKLOAD_PRIORITY post-merge", run.calls)
        for content in run.uploads:
            self.assertTrue(content.startswith("priority: 10\n"))

    def test_scheduled_nightly_keeps_nightly_priority(self) -> None:
        run = run_bootstrap(
            BUILDKITE_BRANCH="main",
            BUILDKITE_PULL_REQUEST="false",
            BUILDKITE_SOURCE="schedule",
            NIGHTLY="1",
        )
        self.assertTrue(run.uploads)
        self.assertIn("meta-data set JOB_PRIORITY 0", run.calls)
        self.assertFalse(any("WORKLOAD_PRIORITY" in c for c in run.calls))
        gated = _nightly_gated(run.uploads)
        self.assertTrue(gated)
        for content in run.uploads:
            self.assertTrue(content.startswith("priority: 0\n"))
        for content in gated:
            self.assertIn('build.env("NIGHTLY")', content)


if __name__ == "__main__":
    unittest.main()
