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
"""Parse, apply, and validate unified diffs emitted by the mutator.

The mutator returns free-form text with a fenced ```diff block. This module:

1. ``extract_diff(raw)`` — pulls the first fenced ```diff block out of the
   LLM's prose. Tolerates the LLM omitting the language tag.
2. ``apply_diff(baseline, diff)`` — applies the diff to ``baseline``.
   Supports the standard unified format (``--- a/...`` / ``+++ b/...`` /
   ``@@ -X,Y +Z,W @@``) and is lenient about context-line drift (the LLM
   sometimes miscounts hunk offsets by a small number of lines).
3. ``validate_python(source)`` — ``ast.parse`` smoke check. Rejects sources
   that can't be parsed before we spend a TPU run on them.

The applier deliberately does NOT shell out to ``patch`` or ``git apply`` —
both are sensitive to trailing-whitespace differences that LLMs commonly
introduce, and we want a deterministic, dependency-free implementation.
"""

from __future__ import annotations

import ast
import dataclasses
import re


@dataclasses.dataclass
class DiffResult:
    success: bool
    new_source: str | None = None
    error: str | None = None
    hunks_applied: int = 0


_FENCED_RE = re.compile(
    r"```(?:diff|patch)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_diff(raw: str) -> str:
    """Pull the first fenced diff block out of the LLM response.

    Raises ``ValueError`` if no fenced block is present.
    """
    match = _FENCED_RE.search(raw)
    if not match:
        # As a fallback, accept the entire response IF it already looks like
        # a unified diff (starts with `---` line).
        s = raw.strip()
        if s.startswith("---") or s.startswith("diff --git"):
            return s
        raise ValueError("no fenced ```diff block in LLM output")
    return match.group(1).strip()


_HUNK_HEADER_RE = re.compile(
    r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@.*$")


@dataclasses.dataclass
class _Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]  # raw hunk body lines (' ', '+', '-' prefixed)


def _parse_hunks(diff: str) -> list[_Hunk]:
    """Parse the body of a unified diff into hunks.

    Skips file-header lines (``--- a/...``, ``+++ b/...``, ``diff --git``,
    ``index ...``) — we only care about the hunk content because we're
    applying to a single in-memory source string.
    """
    hunks: list[_Hunk] = []
    current: _Hunk | None = None
    for raw_line in diff.splitlines():
        if (raw_line.startswith("--- ") or raw_line.startswith("+++ ")
                or raw_line.startswith("diff --git")
                or raw_line.startswith("index ")
                or raw_line.startswith("similarity ")
                or raw_line.startswith("rename ")
                or raw_line.startswith("new file ")
                or raw_line.startswith("deleted file ")):
            current = None
            continue
        m = _HUNK_HEADER_RE.match(raw_line)
        if m:
            current = _Hunk(
                old_start=int(m.group(1)),
                old_count=int(m.group(2) or "1"),
                new_start=int(m.group(3)),
                new_count=int(m.group(4) or "1"),
                lines=[],
            )
            hunks.append(current)
            continue
        if current is None:
            # Lines before the first hunk header — ignore.
            continue
        # Empty lines in a diff are valid context (single space then EOL),
        # but LLMs commonly drop the leading space. Treat a fully empty
        # line as a blank context line.
        if raw_line == "":
            current.lines.append(" ")
            continue
        prefix = raw_line[0]
        if prefix in (" ", "+", "-"):
            current.lines.append(raw_line)
        elif raw_line.startswith("\\ No newline at end of file"):
            # Standard ``\ No newline at end of file`` marker — ignore.
            continue
        else:
            # Tolerate unprefixed lines as context (LLMs sometimes drop the
            # leading space on otherwise-blank-looking lines).
            current.lines.append(" " + raw_line)
    return hunks


def _split_lines_keep(text: str) -> list[str]:
    """Split into lines without losing the trailing-newline marker."""
    return text.splitlines(keepends=True)


def _hunk_body_old(hunk: _Hunk) -> list[str]:
    """Lines as they should appear in the OLD source for this hunk."""
    out: list[str] = []
    for line in hunk.lines:
        if line.startswith("+"):
            continue
        # context (' ') or deletion ('-')
        out.append(line[1:])
    return out


def _hunk_body_new(hunk: _Hunk) -> list[str]:
    """Lines as they should appear in the NEW source for this hunk."""
    out: list[str] = []
    for line in hunk.lines:
        if line.startswith("-"):
            continue
        out.append(line[1:])
    return out


def _try_locate(baseline_lines: list[str], expected_old: list[str],
                hinted_start: int) -> int | None:
    """Find the 0-based index where ``expected_old`` matches.

    Tries the hint first, then a window around it, then a full-file scan.
    Returns ``None`` if no match is found.
    """
    n = len(baseline_lines)
    m = len(expected_old)
    if m == 0:
        # An "insert-only" hunk; just trust the hint.
        return max(0, min(n, hinted_start))
    if hinted_start + m <= n and baseline_lines[hinted_start:hinted_start +
                                                m] == expected_old:
        return hinted_start
    # Window search ± 50 lines around the hint.
    window = 50
    for delta in range(1, window + 1):
        for cand in (hinted_start - delta, hinted_start + delta):
            if 0 <= cand and cand + m <= n:
                if baseline_lines[cand:cand + m] == expected_old:
                    return cand
    # Full-file scan as a last resort.
    for cand in range(0, n - m + 1):
        if baseline_lines[cand:cand + m] == expected_old:
            return cand
    return None


def apply_diff(baseline: str, diff: str) -> DiffResult:
    """Apply the diff to ``baseline`` and return the new source.

    Tolerates small hunk-offset drift; rejects context mismatches that can't
    be located anywhere in the baseline.
    """
    try:
        hunks = _parse_hunks(diff)
    except (ValueError, IndexError) as err:
        return DiffResult(success=False, error=f"diff parse failed: {err}")
    if not hunks:
        return DiffResult(success=False, error="no hunks in diff")
    baseline_lines = _split_lines_keep(baseline)
    # Sort hunks by old_start so we can apply them in order with a running
    # offset between old-line-positions and new-line-positions.
    hunks.sort(key=lambda h: h.old_start)
    output: list[str] = []
    cursor = 0  # next unread index in baseline_lines
    applied = 0
    for hunk in hunks:
        # Expected old body (with trailing newlines normalized to the
        # baseline's existing line endings).
        old_body = _hunk_body_old(hunk)
        new_body = _hunk_body_new(hunk)
        # Strip newline endings on hunk lines (the unified diff lines never
        # carry a trailing newline in their text); we'll re-attach them
        # using the baseline's existing whitespace style.
        # Convert to baseline-style by adding "\n" to each line.
        old_body_lines = [line + "\n" for line in old_body]
        new_body_lines = [line + "\n" for line in new_body]
        # Hunk hint: 1-based; convert to 0-based.
        hint = max(0, hunk.old_start - 1)
        # Search starting at cursor so earlier hunks aren't re-applied.
        if hint < cursor:
            hint = cursor
        found = _try_locate(baseline_lines, old_body_lines, hint)
        if found is None:
            # Last-chance: try with the existing line endings preserved
            # (some baselines use CRLF).
            old_body_no_nl = old_body
            for i, _bl in enumerate(baseline_lines):
                if (i + len(old_body_no_nl) <= len(baseline_lines) and [
                        bl.rstrip("\r\n")
                        for bl in baseline_lines[i:i + len(old_body_no_nl)]
                ] == old_body_no_nl):
                    found = i
                    break
        if found is None:
            return DiffResult(
                success=False,
                error=(f"hunk @@ -{hunk.old_start},{hunk.old_count} "
                       f"+{hunk.new_start},{hunk.new_count} @@ did not "
                       f"match baseline context anywhere "
                       f"(searched the full file)"),
                hunks_applied=applied,
            )
        # Pass through unchanged lines up to the hunk start.
        output.extend(baseline_lines[cursor:found])
        # Emit the new body.
        output.extend(new_body_lines)
        cursor = found + len(old_body_lines)
        applied += 1
    # Flush remainder.
    output.extend(baseline_lines[cursor:])
    return DiffResult(success=True,
                      new_source="".join(output),
                      hunks_applied=applied)


def validate_python(source: str) -> tuple[bool, str | None]:
    """``ast.parse`` smoke check on the mutated source."""
    try:
        ast.parse(source)
        return True, None
    except SyntaxError as err:
        return False, f"SyntaxError: {err}"
