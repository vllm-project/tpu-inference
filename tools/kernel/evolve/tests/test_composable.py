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
"""Tests for ChainingMutator."""

import textwrap

from tools.kernel.evolve.mutator.composable import ChainingMutator
from tools.kernel.evolve.mutator.diff_applier import apply_diff, extract_diff
from tools.kernel.evolve.mutator.programmatic import (LiteralRewriteRule,
                                                      ProgrammaticMutator)

_BASELINE = textwrap.dedent("""\
    import jax.numpy as jnp

    BLOCK_M = 128
    BLOCK_N = 128
    ACCUM_DTYPE = jnp.float32


    def matmul(x, y):
        return jnp.matmul(x, y)
    """)


def _prompt(src: str) -> str:
    return f"```python\n{src}```"


def test_chained_mutator_can_emit_single_diff_when_chain_prob_zero():
    inner = ProgrammaticMutator(
        baseline_path="m.py",
        literal_rules=[
            LiteralRewriteRule("BLOCK_M", values=["256", "512"]),
            LiteralRewriteRule("BLOCK_N", values=["256", "512"]),
        ],
        seed=0,
    )
    chained = ChainingMutator(inner,
                              chain_prob=0.0,
                              max_chain_length=3,
                              seed=0)
    response = chained.chat(system="", user=_prompt(_BASELINE))
    diff = extract_diff(response)
    r = apply_diff(_BASELINE, diff)
    assert r.success
    # Only one of BLOCK_M / BLOCK_N changed.
    changed_m = "BLOCK_M = 128" not in r.new_source
    changed_n = "BLOCK_N = 128" not in r.new_source
    assert changed_m ^ changed_n


def test_chained_mutator_stacks_two_changes_when_chain_prob_high():
    inner = ProgrammaticMutator(
        baseline_path="m.py",
        literal_rules=[
            LiteralRewriteRule("BLOCK_M", values=["256"]),
            LiteralRewriteRule("BLOCK_N", values=["256"]),
        ],
        seed=0,
    )
    chained = ChainingMutator(inner,
                              chain_prob=1.0,
                              max_chain_length=2,
                              seed=0)
    response = chained.chat(system="", user=_prompt(_BASELINE))
    diff = extract_diff(response)
    r = apply_diff(_BASELINE, diff)
    assert r.success
    # Both BLOCK_M and BLOCK_N should have changed.
    assert "BLOCK_M = 256" in r.new_source
    assert "BLOCK_N = 256" in r.new_source


def test_chained_mutator_handles_inner_returning_no_diff():
    """If the inner can't propose a diff, the wrapper passes through."""
    inner = ProgrammaticMutator(
        baseline_path="m.py",
        literal_rules=[
            LiteralRewriteRule("DOES_NOT_EXIST", values=["1"]),
        ],
        seed=0,
    )
    chained = ChainingMutator(inner, chain_prob=1.0)
    response = chained.chat(system="", user=_prompt(_BASELINE))
    # No fenced diff in response.
    assert "```diff" not in response
