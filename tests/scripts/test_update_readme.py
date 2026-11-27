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
"""Unit tests for scripts/update_readme.py: which tables a run rewrites.

The script reads the CSVs under support_matrices/ and writes README.md and
docs/includes, all relative to the working directory, so each test runs in a
temporary directory with a copy of the CSVs and a README that holds only the
marker blocks.
"""

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "update_readme.py"

_SPEC = importlib.util.spec_from_file_location("update_readme", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_STALE = "STALE"


def _snippet(type_key, section_key):
    """The snippet of a table, under the name the pages in docs/ include."""
    name = section_key if type_key == "release" else f"{type_key}_{section_key}"
    return Path("docs") / "includes" / f"{name}.md"


def _states(type_key):
    """The states of the tables of `type_key`: {"stale"}, {"fresh"}, or more."""
    readme = Path("README.md").read_text(encoding="utf-8")
    states = set()
    for section_key in _MODULE.CSV_MAP[type_key]:
        marker = f"{type_key}_{section_key}"
        block = re.search(
            f"<!-- START: {marker} -->(.*?)<!-- END: {marker} -->", readme,
            re.DOTALL)
        assert block is not None, f"README.md lost its {marker} block"
        block_stale = block.group(1) == _STALE
        snippet_stale = _snippet(
            type_key, section_key).read_text(encoding="utf-8") == _STALE
        if block_stale and snippet_stale:
            states.add("stale")
        elif not block_stale and not snippet_stale:
            states.add("fresh")
        else:
            states.add(f"{marker}: its README block and snippet differ")
    return states


@pytest.fixture
def stale_tables(tmp_path, monkeypatch):
    """The working directory of a run in which every table is stale."""
    shutil.copytree(_REPO_ROOT / "support_matrices",
                    tmp_path / "support_matrices")
    monkeypatch.chdir(tmp_path)
    Path("docs", "includes").mkdir(parents=True)
    blocks = ["*Last Updated: never*"]
    for type_key, sections in _MODULE.CSV_MAP.items():
        for section_key in sections:
            marker = f"{type_key}_{section_key}"
            blocks.append(
                f"<!-- START: {marker} -->{_STALE}<!-- END: {marker} -->")
            _snippet(type_key, section_key).write_text(_STALE,
                                                       encoding="utf-8")
    Path("README.md").write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    assert _states("release") == _states("nightly") == {"stale"}


@pytest.mark.parametrize("only, other", [("release", "nightly"),
                                         ("nightly", "release")])
def test_one_type_leaves_the_other_untouched(stale_tables, only, other):
    _MODULE.update_readme([only])

    assert _states(only) == {"fresh"}
    assert _states(other) == {"stale"}


def test_default_rewrites_both_types(stale_tables):
    _MODULE.update_readme()

    assert _states("release") == _states("nightly") == {"fresh"}


def test_command_line(stale_tables):

    def run(*args):
        # The script prints an emoji, which needs a UTF-8 stdout.
        return subprocess.run(
            [sys.executable, str(_SCRIPT), *args],
            capture_output=True,
            env={
                **os.environ, "PYTHONIOENCODING": "utf-8"
            },
            check=False,
        ).returncode

    assert run("--only", "release") == 0
    assert _states("release") == {"fresh"}
    assert _states("nightly") == {"stale"}

    assert run("--only", "both") != 0
    assert _states("nightly") == {"stale"}

    # No argument: both, as update_readme.yml runs it.
    assert run() == 0
    assert _states("nightly") == {"fresh"}
