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
"""AlphaEvolve-style LLM-mutation kernel optimizer for TPU Pallas.

System architecture:

* ``genome`` — a candidate solution is a ``Genome`` carrying a unified diff
  against a baseline kernel source, fitness (latency), parent ids, island id,
  generation, and an evaluation status.
* ``archive`` — island-based population storage with JSONL persistence,
  tournament selection, top-K elitism, and periodic migration.
* ``mutator.llm_client`` — pluggable LLM backend (Anthropic / Gemini stub /
  pre-canned responses). The mutator emits a unified diff inside a fenced
  block.
* ``mutator.diff_applier`` — applies LLM-emitted diffs to the baseline source
  and AST-validates the result.
* ``mutator.critic`` — optional adversarial pre-filter: a second LLM call
  asked to refute the candidate before TPU evaluation.
* ``worktree`` — isolates each candidate in a temp directory so a broken
  diff cannot corrupt the working tree or other candidates.
* ``evaluator`` — runs a candidate through the Phase 0+1 verifier + bench
  stack and returns ``EvaluationResult``.
* ``orchestrator`` — island GA loop: select parents, mutate, evaluate,
  insert, migrate.

The verifier and bench infrastructure live under
``tools.kernel.tuner.v1.{verifier,bench}`` and are reused unchanged — they
are the moat that prevents the Sakana-class "fake speedup" failure modes
(buffer reuse, dropped layers, async-incomplete returns, numerical drift).
"""

from tools.kernel.evolve.archive import Archive, Island
from tools.kernel.evolve.evaluator import EvaluationResult, evaluate_genome
from tools.kernel.evolve.genome import Genome, GenomeStatus
from tools.kernel.evolve.orchestrator import EvolutionConfig, Orchestrator

__all__ = [
    "Archive",
    "EvaluationResult",
    "EvolutionConfig",
    "Genome",
    "GenomeStatus",
    "Island",
    "Orchestrator",
    "evaluate_genome",
]
