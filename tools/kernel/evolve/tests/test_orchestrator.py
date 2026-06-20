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
"""End-to-end orchestrator test using StubClient (no API calls)."""

import textwrap
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from tools.kernel.evolve.archive import Archive
from tools.kernel.evolve.genome import Genome, GenomeStatus
from tools.kernel.evolve.mutator.llm_client import StubClient
from tools.kernel.evolve.orchestrator import EvolutionConfig, Orchestrator

_BASELINE_SRC = textwrap.dedent("""\
    import jax.numpy as jnp

    def kernel(x):
        return x * 2.0
    """)

_DIFF_GOOD = textwrap.dedent("""\
    Hypothesis: a comment-only rename helps the cache layout (noop).

    ```diff
    --- a/k.py
    +++ b/k.py
    @@ -3,1 +3,1 @@
    -def kernel(x):
    +def kernel(x):  # mutated
    ```
    """)

_DIFF_NUMERICS_BAD = textwrap.dedent("""\
    Hypothesis: scale up by 5x (wrong, will fail verifier).

    ```diff
    --- a/k.py
    +++ b/k.py
    @@ -4,1 +4,1 @@
    -    return x * 2.0
    +    return x * 5.0
    ```
    """)

_DIFF_NO_FENCE = "Just prose, no diff. The mutator forgot."

_DIFF_UNAPPLYABLE = textwrap.dedent("""\
    Hypothesis: replace something that's not there.

    ```diff
    --- a/k.py
    +++ b/k.py
    @@ -1,1 +1,1 @@
    -nonexistent_line
    +replacement
    ```
    """)


class _StubOracle:

    def compute(self, inputs):
        return inputs["x"] * 2.0

    def dtype_tolerance(self, dtype):
        return 1e-4, 1e-4


class _Host:
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

    def anti_cheat_skip_keys(self):
        return ()


_CRITIC_PASS = "VERDICT: likely_correct reason: looks fine."


def _stub_mutator():
    """Cycles through a mix of good/bad diffs so the loop exercises every
    classification path."""
    return StubClient([
        _DIFF_GOOD,
        _DIFF_NUMERICS_BAD,
        _DIFF_NO_FENCE,
        _DIFF_UNAPPLYABLE,
        _DIFF_GOOD,
    ])


def _stub_critic():
    return StubClient([_CRITIC_PASS] * 100)


def test_orchestrator_runs_and_finds_winner(tmp_path):
    host = _Host()
    archive = Archive(
        baseline=Genome.baseline(baseline_path="k.py"),
        num_islands=2,
        island_cap=8,
        persist_path=tmp_path / "archive.jsonl",
    )
    cfg = EvolutionConfig(
        num_islands=2,
        population_cap=8,
        candidates_per_island_per_gen=3,
        generations=2,
        migration_freq=1,
        migration_top_k=1,
        use_critic=True,
        warmup_iters=1,
        bench_iters=3,
        seed=0,
    )
    orch = Orchestrator(
        host=host,
        mutator=_stub_mutator(),
        archive=archive,
        config=cfg,
        critic_llm=_stub_critic(),
    )
    arc = orch.run()
    # Baseline must be verified.
    assert arc.baseline.status == GenomeStatus.VERIFIED
    # Some candidates must have been evaluated.
    assert arc.size() > 1
    # At least one VERIFIED candidate.
    verified = [g for g in arc.all_finite()]
    assert len(verified) >= 1
    # And at least one failure-class genome with finite count but failed.
    statuses = {g.status for isl in arc.islands for g in isl.members}
    assert GenomeStatus.FAILED_NUMERICS in statuses


def test_orchestrator_raises_if_baseline_fails(tmp_path):
    """If the host's baseline doesn't match its own oracle, the run must
    bail out — there's no point evolving against a broken comparison."""

    class _BadHost(_Host):

        def get_oracle(self):

            class _O:

                def compute(self, inputs):
                    return inputs["x"] * 100.0  # wildly disagrees w/ baseline

                def dtype_tolerance(self, dtype):
                    return 1e-4, 1e-4

            return _O()

    host = _BadHost()
    archive = Archive(
        baseline=Genome.baseline(baseline_path="k.py"),
        num_islands=1,
        persist_path=tmp_path / "archive.jsonl",
    )
    cfg = EvolutionConfig(num_islands=1,
                          candidates_per_island_per_gen=1,
                          generations=1,
                          use_critic=False,
                          warmup_iters=1,
                          bench_iters=2,
                          seed=0)
    orch = Orchestrator(host=host,
                        mutator=_stub_mutator(),
                        archive=archive,
                        config=cfg)
    import pytest
    with pytest.raises(RuntimeError, match="Baseline failed verifier"):
        orch.run()


def test_orchestrator_persists_archive(tmp_path):
    host = _Host()
    path = tmp_path / "archive.jsonl"
    archive = Archive(
        baseline=Genome.baseline(baseline_path="k.py"),
        num_islands=1,
        persist_path=path,
    )
    cfg = EvolutionConfig(num_islands=1,
                          candidates_per_island_per_gen=2,
                          generations=1,
                          use_critic=False,
                          warmup_iters=1,
                          bench_iters=2,
                          seed=0)
    orch = Orchestrator(host=host,
                        mutator=_stub_mutator(),
                        archive=archive,
                        config=cfg)
    orch.run()
    assert path.exists()
    # Reloading should populate the same island.
    archive2 = Archive(
        baseline=Genome.baseline(baseline_path="k.py"),
        num_islands=1,
        persist_path=path,
    )
    assert archive2.size() >= 2  # baseline + ≥1 candidate
