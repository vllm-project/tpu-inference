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
"""End-to-end evaluator test against a synthetic in-memory kernel.

Verifies that the evaluator correctly classifies:
* baseline: VERIFIED
* good diff: VERIFIED with fitness ≤ baseline (or comparable)
* bad-diff that breaks compile: FAILED_COMPILE
* numerics-wrong diff: FAILED_NUMERICS
* unparseable mutation: FAILED_DIFF
* anti-cheat catches input-aliased output: FAILED_ANTI_CHEAT
"""

import textwrap
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from tools.kernel.evolve.evaluator import evaluate_genome
from tools.kernel.evolve.genome import Genome, GenomeStatus
from tools.kernel.tuner.v1.verifier.anti_cheat import AntiCheatGuard

_BASELINE_SRC = textwrap.dedent("""\
    import jax.numpy as jnp

    def kernel(x):
        return x * 2.0
    """)

_GOOD_DIFF = textwrap.dedent("""\
    --- a/k.py
    +++ b/k.py
    @@ -3,1 +3,1 @@
    -def kernel(x):
    +def kernel(x):  # mutated noop
    """)

_BAD_COMPILE_DIFF = textwrap.dedent("""\
    --- a/k.py
    +++ b/k.py
    @@ -3,2 +3,2 @@
    -def kernel(x):
    -    return x * 2.0
    +def kernel(x)
    +    return undefined_var + x
    """)

_NUMERICS_WRONG_DIFF = textwrap.dedent("""\
    --- a/k.py
    +++ b/k.py
    @@ -4,1 +4,1 @@
    -    return x * 2.0
    +    return x * 5.0
    """)

_INPUT_ALIAS_DIFF = textwrap.dedent("""\
    --- a/k.py
    +++ b/k.py
    @@ -4,1 +4,1 @@
    -    return x * 2.0
    +    return x
    """)


class _StubOracle:

    def compute(self, inputs):
        return inputs["x"] * 2.0

    def dtype_tolerance(self, dtype):
        return 1e-4, 1e-4


class _StubHost:
    kernel_name = "synthetic_mul"
    kernel_symbol = "kernel"

    def __init__(self):
        rng = np.random.default_rng(0)
        self._x = jnp.asarray(rng.normal(0, 1, size=(64, )).astype(np.float32))
        self.inputs = {"x": self._x}

    @property
    def baseline_path(self) -> str:
        return "k.py"

    def read_baseline_source(self) -> str:
        return _BASELINE_SRC

    def build_kernel_fn(self, module) -> Callable[[], Any]:
        fn = module.kernel
        x = self._x

        def call():
            return fn(x)

        return call

    def get_oracle(self):
        return _StubOracle()

    def anti_cheat_skip_keys(self) -> tuple[str, ...]:
        return ()


def _genome(diff: str) -> Genome:
    return Genome.new(diff=diff,
                      baseline_path="k.py",
                      parent_ids=["baseline"],
                      generation=1,
                      island_id=0)


def test_baseline_evaluates_verified():
    host = _StubHost()
    baseline = Genome.baseline(baseline_path="k.py")
    result = evaluate_genome(baseline, host, warmup=1, iters=3)
    assert result.status == GenomeStatus.VERIFIED
    assert result.fitness < float("inf")
    assert result.bench is not None


def test_good_diff_verified():
    host = _StubHost()
    result = evaluate_genome(_genome(_GOOD_DIFF), host, warmup=1, iters=3)
    assert result.status == GenomeStatus.VERIFIED
    assert result.fitness < float("inf")


def test_bad_diff_fails_compile():
    host = _StubHost()
    result = evaluate_genome(_genome(_BAD_COMPILE_DIFF),
                             host,
                             warmup=1,
                             iters=3)
    # SyntaxError surfaces as FAILED_DIFF (ast.parse step).
    assert result.status == GenomeStatus.FAILED_DIFF
    assert result.fitness == float("inf")


def test_numerics_wrong_diff_fails_verifier():
    host = _StubHost()
    result = evaluate_genome(_genome(_NUMERICS_WRONG_DIFF),
                             host,
                             warmup=1,
                             iters=3)
    assert result.status == GenomeStatus.FAILED_NUMERICS
    assert result.fitness == float("inf")
    assert result.numerics is not None
    assert not result.numerics.passed


def test_input_alias_caught_by_anti_cheat():
    host = _StubHost()

    # Without anti-cheat, this would fail numerics (output=x vs ref=2x).
    # But we want to test the anti-cheat layer specifically: use a "trivial"
    # reference (x → x) so numerics passes and anti-cheat is the only gate.
    class _PassThroughOracle(_StubOracle):

        def compute(self, inputs):
            return inputs["x"]

    host.get_oracle = lambda: _PassThroughOracle()
    guard = AntiCheatGuard()
    result = evaluate_genome(_genome(_INPUT_ALIAS_DIFF),
                             host,
                             warmup=1,
                             iters=3,
                             anti_cheat=guard)
    assert result.status == GenomeStatus.FAILED_ANTI_CHEAT
    assert "input" in (result.error or "")


def test_genome_with_unapplyable_diff_fails_diff():
    host = _StubHost()
    bad_diff = textwrap.dedent("""\
        --- a/k.py
        +++ b/k.py
        @@ -1,1 +1,1 @@
        -nonexistent_line
        +replacement
        """)
    result = evaluate_genome(_genome(bad_diff), host, warmup=1, iters=3)
    assert result.status == GenomeStatus.FAILED_DIFF
