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
"""Adversarial pre-filter for candidate diffs.

A second LLM call (typically a cheaper model — Haiku 4.5 is the natural
default) is asked to *refute* the candidate diff. Rejects with a clear
``VERDICT: likely_broken`` skip the TPU run. ``unsure`` and
``likely_correct`` proceed.

The critic is optional — orchestrator config gates it behind
``use_critic=True`` (default). For budget-constrained sweeps the user can
turn it off and rely on the numerics gate alone.
"""

from __future__ import annotations

import dataclasses
import logging
import re

from tools.kernel.evolve.mutator.llm_client import LLMClient
from tools.kernel.evolve.mutator.prompts import build_critic_prompts

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Critique:
    verdict: str  # 'likely_correct' | 'likely_broken' | 'unsure'
    reason: str
    raw_response: str


_VERDICT_RE = re.compile(
    r"VERDICT\s*:\s*(likely_correct|likely_broken|unsure)\s*(.*)",
    re.IGNORECASE,
)


def _parse_verdict(text: str) -> Critique:
    """Best-effort parse of a critic response.

    If we can't find a ``VERDICT:`` line, default to ``unsure`` so the
    candidate proceeds (false-negative bias).
    """
    m = _VERDICT_RE.search(text)
    if m is None:
        return Critique(
            verdict="unsure",
            reason=(text.strip()[:200] if text else "no parse"),
            raw_response=text,
        )
    verdict = m.group(1).lower()
    reason = m.group(2).strip(" .\n")
    return Critique(verdict=verdict, reason=reason, raw_response=text)


def critique_diff(
    llm: LLMClient,
    *,
    diff: str,
    kernel_name: str,
    baseline_path: str,
    baseline_source: str,
    excerpt_window: int = 80,
    max_tokens: int = 512,
) -> Critique:
    """Run the critic LLM on a candidate diff.

    To keep the critic prompt short, we pass only the source lines that the
    diff touches, expanded by ``excerpt_window`` lines on each side. Saves
    most of the token cost vs sending the whole file.
    """
    excerpt = _excerpt_around_diff(diff, baseline_source, excerpt_window)
    system, user = build_critic_prompts(
        diff=diff,
        kernel_name=kernel_name,
        baseline_path=baseline_path,
        baseline_source_snippet=excerpt,
    )
    try:
        raw = llm.chat(system=system, user=user, max_tokens=max_tokens)
    except Exception as err:  # pragma: no cover - flaky API path
        logger.warning("critic LLM call failed (%s); treating as unsure", err)
        return Critique(
            verdict="unsure",
            reason=f"critic call failed: {err}",
            raw_response="",
        )
    return _parse_verdict(raw)


def _excerpt_around_diff(diff: str, source: str, window: int) -> str:
    """Return source lines touching the diff, expanded by ``window``."""
    # Find @@ -X,Y headers and collect (start_line, length) per hunk.
    hunk_re = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+\d+(?:,\d+)?\s+@@",
                         re.MULTILINE)
    ranges: list[tuple[int, int]] = []
    for m in hunk_re.finditer(diff):
        start = int(m.group(1))
        length = int(m.group(2) or "1")
        ranges.append((start, length))
    if not ranges:
        # Couldn't locate hunks; fall back to the first ``window * 4`` lines.
        lines = source.splitlines()
        return "\n".join(lines[:window * 4])
    lines = source.splitlines()
    n = len(lines)
    keep: set[int] = set()
    for start, length in ranges:
        a = max(0, start - 1 - window)
        b = min(n, start - 1 + length + window)
        keep.update(range(a, b))
    if not keep:
        return ""
    out: list[str] = []
    last = -1
    for idx in sorted(keep):
        if idx > last + 1:
            out.append(f"# ... ({idx - last - 1} lines omitted)")
        out.append(f"{idx + 1:4d}: {lines[idx]}")
        last = idx
    return "\n".join(out)
