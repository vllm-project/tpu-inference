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
"""Build a single mergeable diff that adds a batch of new tuned entries.

The result is suitable for ``gh pr create`` or ``git apply`` directly. The
diff inserts each entry under its appropriate ``q_*_kv_*`` head block,
respecting alphabetic ordering when possible.
"""

from __future__ import annotations

import dataclasses
import re
import textwrap
from pathlib import Path

from tools.kernel.evolve.mutator.diff_applier import validate_python


@dataclasses.dataclass
class TunedWin:
    """One verified winner ready to be persisted."""
    device: str
    page_size: int
    dtype_key: str
    head_key: str
    extra_key: str  # e.g. 'max_model_len-1024-sw-None'
    bkv_p: int
    bq: int
    fitness_ns: float
    speedup_vs_baseline: float | None = None
    referencing_models: list[str] = dataclasses.field(default_factory=list)


def build_auto_pr(
    *,
    wins: list[TunedWin],
    tuned_path: Path,
) -> tuple[str, str]:
    """Apply all wins to tuned_block_sizes.py; return (new_source, PR body).

    Each win is inserted as a new ``max_model_len-...-sw-...`` line inside
    its head_key block. If the head_key block doesn't exist, a fresh block
    is created next to a sibling block under the same dtype/page/device.

    Returns ``(new_source, pr_body_markdown)``.
    """
    src = tuned_path.read_text()
    lines = src.splitlines(keepends=True)
    new_lines = list(lines)

    inserted_count = 0
    skipped: list[str] = []
    for win in sorted(
            wins,
            key=lambda w:
        (w.device, w.page_size, w.dtype_key, w.head_key, w.extra_key)):
        # Try to insert into an existing head block.
        success = _insert_into_existing_block(new_lines, win)
        if success:
            inserted_count += 1
            continue
        # Otherwise insert a fresh block next to a sibling.
        success = _insert_new_head_block(new_lines, win)
        if success:
            inserted_count += 1
            continue
        skipped.append(f"{win.device}/{win.page_size}/{win.dtype_key}/"
                       f"{win.head_key}/{win.extra_key}")

    new_source = "".join(new_lines)
    ok, parse_err = validate_python(new_source)
    if not ok:
        raise RuntimeError(
            f"auto-PR source fails to parse — bailing out: {parse_err}")

    pr_body = _format_pr_body(wins=wins,
                              inserted=inserted_count,
                              skipped=skipped)
    return new_source, pr_body


def _insert_into_existing_block(
    lines: list[str],
    win: TunedWin,
) -> bool:
    """Find the existing ``head_key: {`` block and insert a new entry."""
    head_pat = f"'{win.head_key}'"
    # Locate the block; we expect it to follow the dtype_key + page_size
    # block markers but a naive textual search suffices when the head_key
    # is sufficiently specific. Confirm parents.
    for i, line in enumerate(lines):
        if head_pat in line and ":" in line:
            # Confirm enclosing dtype + page_size by walking back.
            if not _block_matches(lines, i, win):
                continue
            # Find the closing brace of the head block.
            depth = 0
            for j in range(i, len(lines)):
                depth += lines[j].count("{") - lines[j].count("}")
                if depth == 0 and j > i:
                    # Insert just before the closing brace line.
                    indent = _detect_indent(lines[i + 1] if i +
                                            1 < j else lines[i])
                    insert_line = (
                        f"{indent}'{win.extra_key}': ({win.bkv_p}, {win.bq}),\n"
                    )
                    lines.insert(j, insert_line)
                    return True
            break
    return False


def _insert_new_head_block(lines: list[str], win: TunedWin) -> bool:
    """Insert a brand new head_key block; place next to a sibling head."""
    # Find a sibling block under the same device/page/dtype.
    target_dtype_pat = f"'{win.dtype_key}'"
    target_device_pat = f"'{win.device}'"
    in_device = False
    in_page = False
    in_dtype = False
    insert_index = None
    indent = "                "
    for i, line in enumerate(lines):
        if target_device_pat in line and "{" in line:
            in_device = True
            continue
        if in_device and f"{win.page_size}: " in line and "{" in line:
            in_page = True
            continue
        if in_page and target_dtype_pat in line and "{" in line:
            in_dtype = True
            continue
        if in_dtype:
            # Find an existing q_head- block to anchor on.
            if "'q_head-" in line and ":" in line and "{" in line:
                # Insert before this line.
                indent = _detect_indent(line)
                insert_index = i
                break
            # If we hit a closing brace at this depth, the dtype block ended
            # without an anchor — insert just before it.
            if line.strip().startswith("},") and lines[i - 1].endswith("\n"):
                insert_index = i
                break
    if insert_index is None:
        return False
    body_indent = indent + "    "
    block = (f"{indent}'{win.head_key}': {{\n"
             f"{body_indent}'{win.extra_key}': ({win.bkv_p}, {win.bq}),\n"
             f"{indent}}},\n")
    lines.insert(insert_index, block)
    return True


def _block_matches(lines: list[str], i: int, win: TunedWin) -> bool:
    """Walk back from line i to verify enclosing dtype_key + page_size +
    device matches the winner."""
    saw_dtype = False
    saw_page = False
    saw_device = False
    target_dtype = f"'{win.dtype_key}'"
    target_device = f"'{win.device}'"
    for j in range(i - 1, -1, -1):
        line = lines[j]
        if not saw_dtype and target_dtype in line:
            saw_dtype = True
            continue
        if saw_dtype and not saw_page and f"{win.page_size}: " in line:
            saw_page = True
            continue
        if saw_page and target_device in line:
            saw_device = True
            break
        # If we crossed a sibling dtype/page/device level without matching,
        # bail out (this head_key is under a different parent).
        if saw_dtype and "': {" in line and target_dtype not in line:
            return False
    return saw_dtype and saw_page and saw_device


def _detect_indent(line: str) -> str:
    m = re.match(r"^(\s*)", line)
    return m.group(1) if m else "                "


def _format_pr_body(
    *,
    wins: list[TunedWin],
    inserted: int,
    skipped: list[str],
) -> str:
    rows = []
    for w in sorted(wins, key=lambda x: -(x.speedup_vs_baseline or 0)):
        spd = (f"{w.speedup_vs_baseline:.3f}x"
               if w.speedup_vs_baseline is not None else "?")
        models = (", ".join(w.referencing_models[:3]) +
                  (f" (+{len(w.referencing_models)-3})"
                   if len(w.referencing_models) > 3 else ""))
        rows.append(
            f"| {w.device} | {w.page_size} | {w.dtype_key} | {w.head_key} | "
            f"{w.extra_key} | ({w.bkv_p}, {w.bq}) | {w.fitness_ns/1e3:.1f}us | "
            f"{spd} | {models} |")
    skipped_block = ""
    if skipped:
        skipped_block = ("\n\n### Skipped (could not place automatically)\n" +
                         "\n".join(f"- `{s}`" for s in skipped))
    return textwrap.dedent("""\
        ### Auto-PR: tuned-block-size entries discovered by evolve

        This PR adds {inserted} verified entries to ``tuned_block_sizes.py``
        for shapes that the v3 kernel was missing. Every entry was verified
        against the eager reference (dtype-tier allclose + cosine + anti-cheat)
        and measured on a real TPU.

        | device | page | dtype | head | extra | (bkv_p, bq) | latency | speedup | models |
        |---|---|---|---|---|---|---|---|---|
        {table}{skipped}
        """).format(
        inserted=inserted,
        table="\n        ".join(rows),
        skipped=skipped_block,
    )
