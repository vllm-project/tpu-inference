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
"""Tests for sweep utilities (missing entries + auto-PR)."""

import textwrap
from pathlib import Path

from tools.kernel.evolve.sweep.auto_pr import TunedWin, build_auto_pr
from tools.kernel.evolve.sweep.missing_entries import (ModelSpec,
                                                       find_missing_entries)


def _write_table(path: Path, contents: dict) -> None:
    path.write_text("TUNED_BLOCK_SIZES = " + repr(contents) + "\n")


def test_find_missing_returns_empty_when_table_complete(tmp_path):
    path = tmp_path / "tab.py"
    table = {
        "TPU v7": {
            128: {
                "q_bfloat16_kv_bfloat16": {
                    "q_head-16_kv_head-8_head-128": {
                        "max_model_len-1024-sw-None": (8, 32),
                    }
                }
            }
        }
    }
    _write_table(path, table)
    missing = find_missing_entries(
        tuned_block_sizes_path=path,
        models=[ModelSpec("Qwen3-0.6B", 16, 8, 128)],
        context_lengths=[1024],
    )
    assert missing == []


def test_find_missing_detects_absent_entry(tmp_path):
    path = tmp_path / "tab.py"
    table = {"TPU v7": {128: {"q_bfloat16_kv_bfloat16": {}}}}
    _write_table(path, table)
    missing = find_missing_entries(
        tuned_block_sizes_path=path,
        models=[ModelSpec("Qwen3-0.6B", 16, 8, 128)],
        context_lengths=[1024, 2048],
    )
    assert len(missing) == 2
    assert all(m.head_key == "q_head-16_kv_head-8_head-128" for m in missing)
    assert {m.extra_key
            for m in missing} == {
                "max_model_len-1024-sw-None",
                "max_model_len-2048-sw-None",
            }
    # Each missing entry should record the model that needs it.
    for m in missing:
        assert "Qwen3-0.6B" in m.referencing_models


def test_build_auto_pr_inserts_new_head_block(tmp_path):
    # Minimal table with a sibling block under the right parent.
    src = textwrap.dedent("""\
        TUNED_BLOCK_SIZES = {
            'TPU v7': {
                128: {
                    'q_bfloat16_kv_bfloat16': {
                        'q_head-16_kv_head-2_head-128': {
                            'max_model_len-1024-sw-None': (8, 32),
                        },
                    },
                },
            },
        }
        """)
    path = tmp_path / "tab.py"
    path.write_text(src)
    win = TunedWin(
        device="TPU v7",
        page_size=128,
        dtype_key="q_bfloat16_kv_bfloat16",
        head_key="q_head-16_kv_head-8_head-128",
        extra_key="max_model_len-1024-sw-None",
        bkv_p=8,
        bq=32,
        fitness_ns=1000.0,
        speedup_vs_baseline=1.14,
        referencing_models=["Qwen3-0.6B"],
    )
    new_src, pr_body = build_auto_pr(wins=[win], tuned_path=path)
    assert "q_head-16_kv_head-8_head-128" in new_src
    assert "(8, 32)" in new_src
    assert "1.14" in pr_body  # speedup recorded in PR body
