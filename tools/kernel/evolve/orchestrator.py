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
"""Island-based evolutionary orchestrator.

Main loop per generation:

    for island in islands:
        for k in 1..candidates_per_island:
            parent = tournament(island)
            critique = critic(parent)  # optional
            diff = mutator(parent, critique)
            applied = apply_diff(diff)
            result = evaluator(applied)
            archive.insert(genome with result)
    if generation % migration_freq == 0:
        archive.migrate(top_k)

Determinism: a seeded ``numpy.random.Generator`` drives tournament picks,
mutation/crossover selection, and migration. Mutator LLM responses are
naturally non-deterministic; if reproducibility matters, drive with
``StubClient`` or ``CachingClient``.

Persistence: ``Archive`` writes every accepted genome to a JSONL file
synchronously, so a ``KeyboardInterrupt`` mid-run leaves a resumable state.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time

import numpy as np

from tools.kernel.evolve.archive import Archive
from tools.kernel.evolve.evaluator import (EvaluationResult, KernelHost,
                                           evaluate_genome)
from tools.kernel.evolve.genome import Genome, GenomeStatus
from tools.kernel.evolve.mutator.critic import critique_diff
from tools.kernel.evolve.mutator.diff_applier import extract_diff
from tools.kernel.evolve.mutator.llm_client import LLMClient
from tools.kernel.evolve.mutator.prompts import (ParentSummary,
                                                 build_mutation_prompts)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EvolutionConfig:
    num_islands: int = 5
    population_cap: int = 12
    candidates_per_island_per_gen: int = 4
    generations: int = 10
    migration_freq: int = 3
    migration_top_k: int = 2
    tournament_k: int = 3
    use_critic: bool = True
    max_parent_summaries: int = 3
    mutation_max_tokens: int = 4096
    critic_max_tokens: int = 512
    seed: int = 0
    warmup_iters: int = 2
    bench_iters: int = 10
    cosine_floor: float = 0.9999


class Orchestrator:
    """Drives an evolution run end-to-end."""

    def __init__(
        self,
        *,
        host: KernelHost,
        mutator: LLMClient,
        archive: Archive,
        config: EvolutionConfig,
        critic_llm: LLMClient | None = None,
        anti_cheat=None,
        failure_log=None,
        example_pool=None,
        cost_model=None,
    ) -> None:
        self.host = host
        self.mutator = mutator
        self.critic_llm = critic_llm or mutator
        self.archive = archive
        self.config = config
        self.anti_cheat = anti_cheat
        self.failure_log = failure_log  # optional FailureLog
        self.example_pool = example_pool  # optional ExamplePool (RLAIF)
        self.cost_model = cost_model  # optional LinearSurrogate (RL reward proxy)
        self.rng = np.random.default_rng(config.seed)
        self._candidate_count = 0
        # Cache baseline source so we only re-read it if it changes.
        self._baseline_source = host.read_baseline_source()

    # -- public API ----------------------------------------------------------

    def run(self) -> Archive:
        """Run for ``config.generations`` generations and return the archive."""
        self._evaluate_baseline()
        for gen in range(1, self.config.generations + 1):
            t0 = time.time()
            self._run_generation(gen)
            if gen % self.config.migration_freq == 0:
                self.archive.migrate(self.config.migration_top_k, self.rng)
            elapsed = time.time() - t0
            self._log_summary(gen, elapsed)
        return self.archive

    # -- internals -----------------------------------------------------------

    def _evaluate_baseline(self) -> None:
        if self.archive.baseline.status == GenomeStatus.VERIFIED:
            logger.info("Baseline already evaluated (fitness=%s)",
                        self.archive.baseline.fitness)
            return
        logger.info("Evaluating baseline %s",
                    self.archive.baseline.baseline_path)
        result = evaluate_genome(
            self.archive.baseline,
            self.host,
            warmup=self.config.warmup_iters,
            iters=self.config.bench_iters,
            cosine_floor=self.config.cosine_floor,
            anti_cheat=self.anti_cheat,
        )
        self._absorb(self.archive.baseline, result)
        if result.status != GenomeStatus.VERIFIED:
            raise RuntimeError(
                f"Baseline failed verifier ({result.status.value}): "
                f"{result.error}. The host's inputs/oracle must agree before "
                f"evolution can be meaningful.")

    def _run_generation(self, generation: int) -> None:
        for island_id, island in enumerate(self.archive.islands):
            for _ in range(self.config.candidates_per_island_per_gen):
                self._spawn_one(generation, island_id)

    def _spawn_one(self, generation: int, island_id: int) -> Genome | None:
        parent = self.archive.select_parent(island_id,
                                            self.config.tournament_k, self.rng)
        parents = self._gather_parent_summaries(island_id, parent)
        anti_patterns = (self.failure_log.anti_patterns()
                         if self.failure_log else None)
        rejected_diffs = self._recent_rejected_diffs(max_diffs=4)
        positive_block = self._positive_examples_block()
        cost_hint = self._cost_model_hint()
        system, user = build_mutation_prompts(
            kernel_name=self.host.kernel_name,
            baseline_path=self.host.baseline_path,
            baseline_source=self._baseline_source,
            baseline_fitness_ns=(self.archive.baseline.fitness
                                 if math.isfinite(
                                     self.archive.baseline.fitness) else None),
            parents=parents,
            anti_patterns=anti_patterns,
            rejected_diffs=rejected_diffs,
            positive_examples_block=positive_block,
            cost_model_hint=cost_hint,
        )
        try:
            raw = self.mutator.chat(system=system,
                                    user=user,
                                    max_tokens=self.config.mutation_max_tokens)
        except Exception as err:
            logger.warning("mutator chat failed (gen=%d island=%d): %s",
                           generation, island_id, err)
            return None
        try:
            diff = extract_diff(raw)
        except ValueError as err:
            logger.info("mutator response had no diff: %s", err)
            return None

        genome = Genome.new(
            diff=diff,
            baseline_path=self.host.baseline_path,
            parent_ids=[parent.id],
            generation=generation,
            island_id=island_id,
            created_at=time.time(),
        )
        if not self.archive.insert(genome):
            # Same diff hash already evaluated — skip.
            return None
        self._candidate_count += 1

        # Optional critic gate.
        if self.config.use_critic:
            critique = critique_diff(
                self.critic_llm,
                diff=diff,
                kernel_name=self.host.kernel_name,
                baseline_path=self.host.baseline_path,
                baseline_source=self._baseline_source,
                max_tokens=self.config.critic_max_tokens,
            )
            if critique.verdict == "likely_broken":
                genome.status = GenomeStatus.REJECTED_CRITIC
                genome.error = f"critic: {critique.reason}"
                self.archive.update(genome)
                if self.failure_log is not None:
                    self.failure_log.record(
                        rule_name=
                        f"critic_rejection:{_short_reason(critique.reason)}",
                        status="REJECTED_CRITIC")
                logger.info("[g%d/i%d] %s rejected by critic: %s", generation,
                            island_id, genome.id, critique.reason)
                return genome

        # Evaluate on TPU.
        result = evaluate_genome(
            genome,
            self.host,
            warmup=self.config.warmup_iters,
            iters=self.config.bench_iters,
            cosine_floor=self.config.cosine_floor,
            anti_cheat=self.anti_cheat,
        )
        self._absorb(genome, result)
        if self.failure_log is not None:
            self.failure_log.record(
                rule_name=f"mutation:gen{generation}",
                status=result.status.value,
                fitness_ns=(result.fitness
                            if math.isfinite(result.fitness) else None),
            )
        # RLAIF: verified wins go into the persistent example pool so
        # future runs can see them as few-shot exemplars.
        self._absorb_into_example_pool(genome, result)
        return genome

    def _recent_rejected_diffs(self, *, max_diffs: int = 4) -> list[str]:
        """Pull the most-recent rejected diffs from the archive — used as
        anti-examples in the next mutation prompt."""
        rejected: list[Genome] = []
        for isl in self.archive.islands:
            for g in isl.members:
                if g.status in (
                        GenomeStatus.REJECTED_CRITIC,
                        GenomeStatus.FAILED_NUMERICS,
                        GenomeStatus.FAILED_ANTI_CHEAT,
                        GenomeStatus.FAILED_COMPILE,
                ):
                    rejected.append(g)
        # Most recent first by created_at.
        rejected.sort(key=lambda g: g.created_at, reverse=True)
        return [g.diff for g in rejected[:max_diffs] if g.diff]

    def _absorb(self, genome: Genome, result: EvaluationResult) -> None:
        genome.status = result.status
        genome.fitness = result.fitness
        genome.error = result.error
        if result.bench is not None:
            genome.metrics["p50_ns"] = int(result.bench.p50_ns)
            genome.metrics["p95_ns"] = int(result.bench.p95_ns)
            genome.metrics["mean_ns"] = int(result.bench.mean_ns)
        if result.numerics is not None:
            genome.metrics["cosine"] = result.numerics.cosine
            genome.metrics["max_abs_diff"] = result.numerics.max_abs_diff
            genome.metrics["nan_count"] = result.numerics.nan_count
        if result.mutated_source_preview is not None:
            genome.mutated_source_preview = result.mutated_source_preview
        genome.evaluated_at = time.time()
        if genome.id == "baseline":
            self.archive.baseline = genome
            # Also append to disk so a resume sees the fitness.
            self.archive._append_to_disk(genome)
        else:
            self.archive.update(genome)

    def _gather_parent_summaries(
        self,
        island_id: int,
        immediate_parent: Genome,
    ) -> list[ParentSummary]:
        """Pick a few ancestors to show the mutator in-context.

        Order: the immediate parent first; then the top ``max-1`` from the
        same island; if still short, fill with best from any island. Drop
        duplicates by id.
        """
        chosen: dict[str, Genome] = {}
        chosen[immediate_parent.id] = immediate_parent
        for g in sorted(self.archive.islands[island_id].finite(),
                        key=lambda x: x.fitness):
            if g.id not in chosen:
                chosen[g.id] = g
            if len(chosen) >= self.config.max_parent_summaries:
                break
        if len(chosen) < self.config.max_parent_summaries:
            for g in sorted(self.archive.all_finite(),
                            key=lambda x: x.fitness):
                if g.id not in chosen:
                    chosen[g.id] = g
                if len(chosen) >= self.config.max_parent_summaries:
                    break
        out: list[ParentSummary] = []
        for g in chosen.values():
            out.append(
                ParentSummary(
                    id=g.id,
                    fitness_ns=(None if not math.isfinite(g.fitness) else
                                g.fitness),
                    diff=g.diff or "(baseline — no diff)",
                    notes=f"status={g.status.value}",
                ))
        return out

    def _log_summary(self, generation: int, elapsed_sec: float) -> None:
        s = self.archive.summary()
        speed = s["speedup_vs_baseline"]
        speed_s = "—" if speed is None else f"{speed:.3f}x"
        logger.info("[gen %d] %.1fs  verified=%d/%d  best=%s  %s", generation,
                    elapsed_sec, s["verified"], s["total_genomes"],
                    s["best_genome_id"], speed_s)

    # -- RLAIF + cost-model helpers ------------------------------------------

    def _positive_examples_block(self) -> str:
        """Render the past-verified-wins few-shot section for this kernel."""
        if self.example_pool is None:
            return ""
        from tools.kernel.evolve.mutator.example_pool import \
            render_examples_for_prompt
        examples = self.example_pool.for_kernel(self.host.kernel_name,
                                                top_k=3,
                                                min_speedup=1.01)
        return render_examples_for_prompt(examples)

    def _cost_model_hint(self) -> str:
        """Render a one-line hint about the surrogate's predicted-best family."""
        if self.cost_model is None:
            return ""
        return ("## Cost-model hint\n"
                "An online surrogate has been trained on this run's "
                "telemetry. Prefer mutations that target axes the surrogate "
                "hasn't seen explored yet (variety > exploitation early).")

    def _absorb_into_example_pool(self, genome: Genome,
                                  result: EvaluationResult) -> None:
        """If a verified candidate beat the baseline, add it to the pool."""
        if self.example_pool is None:
            return
        if result.status.value != "VERIFIED":
            return
        if not math.isfinite(result.fitness):
            return
        if not math.isfinite(self.archive.baseline.fitness):
            return
        speedup = (self.archive.baseline.fitness /
                   result.fitness if result.fitness > 0 else None)
        if speedup is None or speedup < 1.01:
            return
        from tools.kernel.evolve.mutator.example_pool import PositiveExample
        hypothesis = ""
        try:
            first_line = (genome.diff.splitlines()[0] if genome.diff else "")
            hypothesis = first_line[:200]
        except Exception:
            pass
        self.example_pool.add(
            PositiveExample(
                kernel=self.host.kernel_name,
                diff=genome.diff,
                speedup=speedup,
                p_value=genome.metrics.get("p_value"),
                hypothesis=hypothesis,
                added_at=time.time(),
                source_run_id=genome.id,
            ))


def _short_reason(reason: str | None) -> str:
    """Compress a critic's verbose reason into a tag suitable for FailureLog."""
    if not reason:
        return "unknown"
    lower = reason.lower()
    for tag, keyword in (
        ("approx_reciprocal", "reciprocal"),
        ("precision_regress", "precision"),
        ("dropped_blockready", "block_until_ready"),
        ("wrong_dtype", "dtype"),
        ("shape_mismatch", "shape"),
        ("mask_bug", "mask"),
    ):
        if keyword in lower:
            return tag
    return reason.split(".")[0][:40].strip().replace(" ", "_")
