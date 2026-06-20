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
"""Composable mutations — chain N atomic rules into one diff.

Why this matters: single-knob mutations have a ceiling (you can't find a
win that requires TWO simultaneous changes — e.g. larger block + lower
accumulator dtype). Chained mutations vastly expand the surface but
multiply the failure rate, so the critic step becomes load-bearing.

``ChainingMutator`` wraps any base ``ProgrammaticMutator``:
1. Generate a primary diff from the inner mutator.
2. With probability ``chain_prob``, apply it to the source, then ask the
   inner mutator for a SECOND diff against the mutated source.
3. Merge the two diffs into a single combined unified diff and emit.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from tools.kernel.evolve.mutator.diff_applier import apply_diff, extract_diff
from tools.kernel.evolve.mutator.programmatic import ProgrammaticMutator

logger = logging.getLogger(__name__)


class ChainingMutator:
    """``LLMClient``-shaped wrapper that may stack 2-3 base mutations."""

    def __init__(
        self,
        inner: ProgrammaticMutator,
        *,
        chain_prob: float = 0.5,
        max_chain_length: int = 3,
        seed: int = 0,
        model_id: str | None = None,
    ) -> None:
        self.inner = inner
        self.chain_prob = chain_prob
        self.max_chain_length = max_chain_length
        self._rng = np.random.default_rng(seed)
        self._model_id = model_id or f"chained:{inner.model_id}"

    @property
    def model_id(self) -> str:
        return self._model_id

    def chat(self, *, system: str, user: str, max_tokens: int = 4096) -> str:
        # First mutation off the inner.
        r1 = self.inner.chat(system=system, user=user, max_tokens=max_tokens)
        try:
            d1 = extract_diff(r1)
        except ValueError:
            return r1  # nothing to chain
        m = re.search(r"```python\s*\n(.*?)```", user, flags=re.DOTALL)
        if m is None:
            return r1
        source = m.group(1)
        applied = apply_diff(source, d1)
        if not applied.success or applied.new_source is None:
            return r1
        chain_n = 1
        accumulated_diff = d1
        current_source = applied.new_source
        while (chain_n < self.max_chain_length
               and self._rng.random() < self.chain_prob):
            # Ask inner for another diff against the mutated source.
            mutated_user = re.sub(
                r"```python\s*\n.*?```",
                f"```python\n{current_source}```",
                user,
                count=1,
                flags=re.DOTALL,
            )
            r2 = self.inner.chat(system=system,
                                 user=mutated_user,
                                 max_tokens=max_tokens)
            try:
                d2 = extract_diff(r2)
            except ValueError:
                break
            applied2 = apply_diff(current_source, d2)
            if not applied2.success or applied2.new_source is None:
                break
            # Combine the diffs by producing a fresh unified diff between
            # ORIGINAL and applied2.new_source — uses ``difflib`` for safety.
            accumulated_diff = _build_combined_diff(source,
                                                    applied2.new_source)
            current_source = applied2.new_source
            chain_n += 1

        hypothesis = (
            f"Hypothesis: composed mutation (chain length={chain_n}) — "
            f"combines independent transformations into one diff.")
        return f"{hypothesis}\n\n```diff\n{accumulated_diff}\n```\n"


def _build_combined_diff(original: str, mutated: str) -> str:
    """Produce a unified diff between ``original`` and ``mutated``."""
    import difflib
    orig_lines = original.splitlines(keepends=True)
    new_lines = mutated.splitlines(keepends=True)
    diff = difflib.unified_diff(
        orig_lines,
        new_lines,
        fromfile="a/source",
        tofile="b/source",
        n=3,
    )
    return "".join(diff)
