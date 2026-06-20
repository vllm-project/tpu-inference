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
"""Tests for the deterministic programmatic mutator."""

import textwrap

from tools.kernel.evolve.mutator.diff_applier import (apply_diff, extract_diff,
                                                      validate_python)
from tools.kernel.evolve.mutator.programmatic import (LineRewriteRule,
                                                      LiteralRewriteRule,
                                                      ProgrammaticMutator)

_BASELINE = textwrap.dedent("""\
    import jax.numpy as jnp

    BLOCK_M = 128
    BLOCK_N = 128
    ACCUM_DTYPE = jnp.float32


    def matmul(x, y):
        return jnp.matmul(x, y)
    """)


def _user_prompt(src: str) -> str:
    return f"Source:\n```python\n{src}```"


def test_emits_valid_diff_for_literal_rewrite():
    mut = ProgrammaticMutator(
        baseline_path="m.py",
        literal_rules=[
            LiteralRewriteRule("BLOCK_M",
                               values=["256", "512"],
                               description="row tile"),
        ],
        seed=0,
    )
    response = mut.chat(system="s", user=_user_prompt(_BASELINE))
    diff = extract_diff(response)
    result = apply_diff(_BASELINE, diff)
    assert result.success
    assert "BLOCK_M = 256" in result.new_source or "BLOCK_M = 512" in result.new_source
    assert "BLOCK_M = 128" not in result.new_source
    ok, _ = validate_python(result.new_source)
    assert ok


def test_skips_current_value():
    mut = ProgrammaticMutator(
        baseline_path="m.py",
        literal_rules=[
            LiteralRewriteRule("BLOCK_M", values=["128", "256"]),
        ],
        seed=0,
    )
    response = mut.chat(system="s", user=_user_prompt(_BASELINE))
    diff = extract_diff(response)
    result = apply_diff(_BASELINE, diff)
    assert result.success
    # Must pick 256, not the current value 128.
    assert "BLOCK_M = 256" in result.new_source


def test_does_not_repeat_proposals():
    mut = ProgrammaticMutator(
        baseline_path="m.py",
        literal_rules=[
            LiteralRewriteRule("BLOCK_M", values=["256"]),  # only one option
        ],
        seed=0,
    )
    r1 = mut.chat(system="s", user=_user_prompt(_BASELINE))
    r2 = mut.chat(system="s", user=_user_prompt(_BASELINE))
    # First call yields a diff; the second has nothing new to propose.
    assert "```diff" in r1
    assert "```diff" not in r2
    assert "rule pool exhausted" in r2


def test_handles_dtype_string_rhs():
    mut = ProgrammaticMutator(
        baseline_path="m.py",
        literal_rules=[
            LiteralRewriteRule(
                "ACCUM_DTYPE",
                values=["jnp.bfloat16", "jnp.float32"],
                description="accumulator dtype",
            ),
        ],
        seed=0,
    )
    response = mut.chat(system="s", user=_user_prompt(_BASELINE))
    diff = extract_diff(response)
    result = apply_diff(_BASELINE, diff)
    assert result.success
    assert "ACCUM_DTYPE = jnp.bfloat16" in result.new_source
    assert "ACCUM_DTYPE = jnp.float32" not in result.new_source


def test_line_rewrite_rule_applies_when_pattern_matches():
    src = textwrap.dedent("""\
        def f(x):
            return x * 2
        """)
    mut = ProgrammaticMutator(
        baseline_path="f.py",
        line_rules=[
            LineRewriteRule(
                pattern=r"^    return x \* 2$",
                replacements=["    return x + x"],
                description="x*2 -> x+x",
            ),
        ],
        seed=0,
    )
    response = mut.chat(system="s", user=_user_prompt(src))
    diff = extract_diff(response)
    result = apply_diff(src, diff)
    assert result.success
    assert "return x + x" in result.new_source


def test_falls_back_when_no_rules_apply():
    mut = ProgrammaticMutator(
        baseline_path="m.py",
        literal_rules=[
            LiteralRewriteRule("DOES_NOT_EXIST", values=["1", "2"]),
        ],
        seed=0,
    )
    response = mut.chat(system="s", user=_user_prompt(_BASELINE))
    assert "```diff" not in response
    assert "rule pool exhausted" in response


def test_preserves_trailing_comments():
    src = "BLOCK_M = 128  # trailing comment\n"
    mut = ProgrammaticMutator(
        baseline_path="m.py",
        literal_rules=[
            LiteralRewriteRule("BLOCK_M", values=["256"]),
        ],
        seed=0,
    )
    response = mut.chat(system="s", user=_user_prompt(src))
    diff = extract_diff(response)
    result = apply_diff(src, diff)
    assert result.success
    assert "BLOCK_M = 256" in result.new_source
    assert "# trailing comment" in result.new_source
