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
"""Tests for the diff parser/applier and AST validator."""

import textwrap

import pytest

from tools.kernel.evolve.mutator.diff_applier import (apply_diff, extract_diff,
                                                      validate_python)

_BASELINE = textwrap.dedent("""\
    def f(x):
        # baseline impl
        y = x * 2
        return y + 1
    """)


def test_extract_fenced_diff_block():
    raw = "Some prose.\n```diff\n--- a/x\n+++ b/x\n```\nMore prose."
    diff = extract_diff(raw)
    assert "--- a/x" in diff
    assert "+++ b/x" in diff


def test_extract_handles_missing_lang_tag():
    raw = "Prose.\n```\n--- a/x\n+++ b/x\n```"
    diff = extract_diff(raw)
    assert "--- a/x" in diff


def test_extract_raises_when_no_diff_block():
    with pytest.raises(ValueError, match="no fenced"):
        extract_diff("Just prose, no diff here.")


def test_apply_single_hunk_replace_line():
    diff = textwrap.dedent("""\
        --- a/f.py
        +++ b/f.py
        @@ -2,2 +2,2 @@ def f(x):
             # baseline impl
        -    y = x * 2
        +    y = x * 3
        """)
    result = apply_diff(_BASELINE, diff)
    assert result.success
    assert "y = x * 3" in result.new_source
    assert "y = x * 2" not in result.new_source


def test_apply_tolerates_line_offset_drift():
    """LLMs often miscount line numbers by ±1-3; the applier should locate
    by context."""
    # Hunk header claims old_start=10 but the actual context appears at line 2.
    diff = textwrap.dedent("""\
        --- a/f.py
        +++ b/f.py
        @@ -10,2 +10,2 @@
             # baseline impl
        -    y = x * 2
        +    y = x * 4
        """)
    result = apply_diff(_BASELINE, diff)
    assert result.success
    assert "y = x * 4" in result.new_source


def test_apply_rejects_context_mismatch():
    diff = textwrap.dedent("""\
        --- a/f.py
        +++ b/f.py
        @@ -1,1 +1,1 @@
        -nonexistent_baseline_line
        +replacement
        """)
    result = apply_diff(_BASELINE, diff)
    assert not result.success
    assert "did not match" in result.error


def test_apply_handles_insertion_only_hunk():
    diff = textwrap.dedent("""\
        --- a/f.py
        +++ b/f.py
        @@ -4,1 +4,2 @@
             return y + 1
        +    # appended comment
        """)
    result = apply_diff(_BASELINE, diff)
    assert result.success
    assert "# appended comment" in result.new_source


def test_apply_multi_hunk():
    diff = textwrap.dedent("""\
        --- a/f.py
        +++ b/f.py
        @@ -1,1 +1,1 @@
        -def f(x):
        +def f(x, y):
        @@ -3,1 +3,1 @@
        -    y = x * 2
        +    y = x * 5
        """)
    result = apply_diff(_BASELINE, diff)
    assert result.success
    assert "def f(x, y):" in result.new_source
    assert "y = x * 5" in result.new_source
    assert result.hunks_applied == 2


def test_apply_rejects_empty_diff():
    result = apply_diff(_BASELINE, "")
    assert not result.success


def test_validate_python_accepts_valid_source():
    ok, err = validate_python("x = 1\ndef f():\n    return x\n")
    assert ok
    assert err is None


def test_validate_python_rejects_syntax_error():
    ok, err = validate_python("def f(:\n    return 1\n")
    assert not ok
    assert "SyntaxError" in err
