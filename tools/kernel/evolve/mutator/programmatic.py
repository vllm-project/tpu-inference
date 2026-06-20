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
"""Deterministic, API-free mutator that generates real source diffs.

Implements ``LLMClient`` so it drops directly into the existing
orchestrator. The "chat" call samples a transformation from a library of
TPU-perf rewrites and emits a unified diff against the kernel's current
source. No tokens spent, no API key needed — useful for:

* Validating the evolve loop produces real measured wins on real TPU.
* Reproducible benchmarking of the orchestrator's classification machinery.
* CI smoke tests.

Two transformation families ship:

* ``LiteralRewriteRule`` — replace a module-level ``NAME = VALUE`` literal
  with another value from a candidate list. Targets things like
  ``BLOCK_M = 128 -> 256`` or ``ACCUM_DTYPE = jnp.float32 -> jnp.bfloat16``.
* ``LineRewriteRule`` — replace a specific source line with a fixed
  alternative. Targets structural one-liner changes (e.g. adding a
  ``donate_argnames`` annotation).

The mutator never repeats the same (source, rule, target_value) tuple, so
the orchestrator's archive de-dup is sufficient to avoid wasted runs.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LiteralRewriteRule:
    """Rewrite a module-level ``NAME = LITERAL`` assignment.

    Matches the first line whose stripped form is ``NAME = anything`` and
    replaces it with ``NAME = chosen_value``. Comments and trailing
    whitespace on the line are preserved.
    """
    name: str
    values: list[str]
    description: str = ""


@dataclasses.dataclass
class LineRewriteRule:
    """Rewrite a specific line matched by a regex.

    Use when the target isn't a literal assignment — e.g. swapping a kwarg
    or adding a decorator.
    """
    pattern: str
    replacements: list[str]
    description: str = ""


_LINE_ASSIGNMENT_RE_FMT = r"^(?P<indent>\s*){name}\s*=\s*(?P<rhs>[^#\n]+?)(?P<trail>\s*(?:#.*)?)$"


def _find_literal_line(source_lines: list[str],
                       name: str) -> tuple[int, str, str] | None:
    """Return ``(index, current_rhs, trailing_comment)`` for ``NAME = ...``.

    Searches top-down for the first match. Returns ``None`` if not found.
    """
    pat = re.compile(_LINE_ASSIGNMENT_RE_FMT.format(name=re.escape(name)))
    for i, line in enumerate(source_lines):
        m = pat.match(line)
        if m is None:
            continue
        return i, m.group("rhs").strip(), m.group("trail").rstrip("\n")
    return None


def _format_unified_diff(
    *,
    path: str,
    line_index: int,
    old_line: str,
    new_line: str,
) -> str:
    """Build a minimal one-hunk unified diff."""
    # Drop trailing newlines for the body; the unified format is
    # line-oriented with no embedded newlines on the hunk body.
    old = old_line.rstrip("\n")
    new = new_line.rstrip("\n")
    return ("--- a/{p}\n"
            "+++ b/{p}\n"
            "@@ -{n},1 +{n},1 @@\n"
            "-{old}\n"
            "+{new}\n").format(p=path, n=line_index + 1, old=old, new=new)


class ProgrammaticMutator:
    """``LLMClient``-shaped mutator that yields real kernel-source diffs.

    Sampling: each ``chat`` call picks (with a seeded RNG) a rule from the
    available pool, then a target value from that rule that hasn't been
    proposed against the current source. Falls back to whitespace tweaks
    (always-applicable no-ops) if the rule pool is exhausted — this keeps
    the orchestrator from stalling.
    """

    def __init__(
        self,
        *,
        baseline_path: str,
        literal_rules: Iterable[LiteralRewriteRule] = (),
        line_rules: Iterable[LineRewriteRule] = (),
        seed: int = 0,
        model_id: str = "programmatic",
    ) -> None:
        self.baseline_path = baseline_path
        self.literal_rules = list(literal_rules)
        self.line_rules = list(line_rules)
        self._rng = np.random.default_rng(seed)
        self._proposed: set[tuple[str, str, str]] = set()
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def chat(self, *, system: str, user: str, max_tokens: int = 4096) -> str:
        """Return an LLM-shaped response: hypothesis + fenced ```diff block.

        Picks the next viable (rule, value) pair given the current source.
        The source is extracted from the user prompt; the orchestrator
        always includes a fenced ``python`` block with the baseline.
        """
        del system, max_tokens  # unused — mutator is deterministic
        baseline_source = _extract_source_from_prompt(user)
        if baseline_source is None:
            return self._no_diff_response(
                "could not parse baseline from prompt")
        source_lines = baseline_source.splitlines(keepends=True)

        # Try literal rules first.
        rules = list(self.literal_rules)
        self._rng.shuffle(rules)
        for rule in rules:
            found = _find_literal_line(source_lines, rule.name)
            if found is None:
                continue
            idx, current_rhs, trail = found
            values = list(rule.values)
            self._rng.shuffle(values)
            for v in values:
                if v == current_rhs:
                    continue
                key = (rule.name, current_rhs, v)
                if key in self._proposed:
                    continue
                self._proposed.add(key)
                old_line = source_lines[idx]
                # Preserve indent + trailing comment by surgically replacing
                # only the RHS expression.
                new_line = re.sub(
                    _LINE_ASSIGNMENT_RE_FMT.format(name=re.escape(rule.name)),
                    rf"\g<indent>{rule.name} = {v}\g<trail>",
                    old_line,
                )
                diff = _format_unified_diff(
                    path=self.baseline_path,
                    line_index=idx,
                    old_line=old_line,
                    new_line=new_line,
                )
                hypothesis = (
                    f"Hypothesis: rewriting `{rule.name}` from "
                    f"`{current_rhs}` to `{v}` should change kernel "
                    f"performance ({rule.description or 'tunable literal'}).")
                return f"{hypothesis}\n\n```diff\n{diff}```\n"

        # Then line rules.
        rules2 = list(self.line_rules)
        self._rng.shuffle(rules2)
        for line_rule in rules2:
            pat = re.compile(line_rule.pattern, re.MULTILINE)
            for m in pat.finditer(baseline_source):
                line_no = baseline_source.count("\n", 0, m.start()) + 1
                idx = line_no - 1
                if idx >= len(source_lines):
                    continue
                old_line = source_lines[idx]
                replacements = list(line_rule.replacements)
                self._rng.shuffle(replacements)
                for repl in replacements:
                    if repl == old_line.rstrip("\n"):
                        continue
                    key = (line_rule.pattern, old_line, repl)
                    if key in self._proposed:
                        continue
                    self._proposed.add(key)
                    new_line = repl + "\n"
                    diff = _format_unified_diff(
                        path=self.baseline_path,
                        line_index=idx,
                        old_line=old_line,
                        new_line=new_line,
                    )
                    hypothesis = (
                        f"Hypothesis: line rewrite — "
                        f"{line_rule.description or 'structural change'}.")
                    return f"{hypothesis}\n\n```diff\n{diff}```\n"

        return self._no_diff_response("rule pool exhausted")

    def _no_diff_response(self, reason: str) -> str:
        return f"No mutation available ({reason}). [no diff]"


def _extract_source_from_prompt(user: str) -> str | None:
    """Pull the ```python source block out of the orchestrator's user prompt."""
    m = re.search(r"```python\s*\n(.*?)```", user, flags=re.DOTALL)
    if m is None:
        return None
    return m.group(1)
