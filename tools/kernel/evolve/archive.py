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
"""Island-based archive with persistent JSONL storage.

Why islands: a single population converges quickly to a local minimum
(documented for both AutoTVM and AlphaEvolve). Multiple semi-isolated
islands explore different basins; periodic migration of the top-K crosses
their best findings without fully merging the populations. The mechanism is
the same as the canonical FunSearch / AlphaEvolve archive.
"""

import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Iterator

from tools.kernel.evolve.genome import Genome


@dataclasses.dataclass
class Island:
    """One isolated sub-population."""
    id: int
    members: list[Genome] = dataclasses.field(default_factory=list)
    cap: int = 16

    def insert(self, genome: Genome) -> None:
        self.members.append(genome)
        # Cap retention by fitness — only keep ``cap`` best when overflowing.
        if len(self.members) > self.cap:
            self.members.sort(key=lambda g: g.fitness)
            self.members = self.members[:self.cap]

    def finite(self) -> list[Genome]:
        return [g for g in self.members if math.isfinite(g.fitness)]

    def best(self) -> Genome | None:
        fin = self.finite()
        if not fin:
            return None
        return min(fin, key=lambda g: g.fitness)

    def top_k(self, k: int) -> list[Genome]:
        fin = self.finite()
        return sorted(fin, key=lambda g: g.fitness)[:k]


class Archive:
    """Multi-island archive with JSONL persistence.

    The on-disk format is a single JSONL file with one genome per line; the
    baseline appears first. Re-loading reconstructs the islands by
    ``island_id``. Persistence is append-only during a run for crash safety,
    then a final compaction pass dedupes by genome ``id``.
    """

    def __init__(
        self,
        *,
        baseline: Genome,
        num_islands: int = 5,
        island_cap: int = 16,
        persist_path: str | os.PathLike | None = None,
    ) -> None:
        self.baseline = baseline
        self.islands: list[Island] = [
            Island(id=i, cap=island_cap) for i in range(num_islands)
        ]
        self.persist_path = (Path(persist_path)
                             if persist_path is not None else None)
        if self.persist_path is not None:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        # Track unique genome ids to avoid double-counting after restart.
        self._seen: set[str] = set()
        if self.persist_path is not None and self.persist_path.exists():
            self._load_from_disk()

    def insert(self, genome: Genome) -> bool:
        """Insert a genome. Returns False if already present (by id)."""
        if genome.id in self._seen:
            return False
        self._seen.add(genome.id)
        self.islands[genome.island_id].insert(genome)
        self._append_to_disk(genome)
        return True

    def update(self, genome: Genome) -> None:
        """Update an in-place genome (e.g. after evaluation)."""
        # _seen already contains the id; just replace in its island.
        island = self.islands[genome.island_id]
        for i, g in enumerate(island.members):
            if g.id == genome.id:
                island.members[i] = genome
                break
        else:
            island.members.append(genome)
        # The on-disk JSONL is append-only; updates append a new line. A
        # compaction pass at the end dedupes by id keeping the latest.
        self._append_to_disk(genome)

    def best_genome(self) -> Genome | None:
        candidates: list[Genome] = []
        for isl in self.islands:
            b = isl.best()
            if b is not None:
                candidates.append(b)
        if not candidates:
            return None
        return min(candidates, key=lambda g: g.fitness)

    def baseline_fitness(self) -> float:
        return self.baseline.fitness

    def select_parent(self, island_id: int, tournament_k: int, rng) -> Genome:
        """Tournament selection within an island.

        Falls back to the baseline if the island has no verified genomes
        yet (cold-start of generation 0).
        """
        pool = self.islands[island_id].finite()
        if not pool:
            return self.baseline
        k = min(tournament_k, len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        picked = [pool[int(i)] for i in idx]
        return min(picked, key=lambda g: g.fitness)

    def migrate(self, top_k: int, rng) -> None:
        """Copy the top-K from each island to a random other island.

        Migration is non-destructive: the source genome stays in its home
        island; a clone (with a fresh ``island_id``) joins the destination.
        """
        n = len(self.islands)
        if n < 2 or top_k < 1:
            return
        for src in self.islands:
            top = src.top_k(top_k)
            if not top:
                continue
            dst_idx = int(rng.integers(0, n))
            while dst_idx == src.id:
                dst_idx = int(rng.integers(0, n))
            for g in top:
                clone = dataclasses.replace(g, island_id=dst_idx)
                # Don't duplicate via _seen; allow same id to be in multiple
                # islands so the destination still benefits even if it had
                # ranked the genome itself.
                self.islands[dst_idx].insert(clone)
                self._append_to_disk(clone)

    def all_finite(self) -> list[Genome]:
        out: list[Genome] = []
        for isl in self.islands:
            out.extend(isl.finite())
        return out

    def __iter__(self) -> Iterator[Genome]:
        yield self.baseline
        for isl in self.islands:
            yield from isl.members

    def size(self) -> int:
        return 1 + sum(len(isl.members) for isl in self.islands)

    def _append_to_disk(self, genome: Genome) -> None:
        if self.persist_path is None:
            return
        with self.persist_path.open("a") as f:
            f.write(json.dumps(genome.to_dict()) + "\n")

    def _load_from_disk(self) -> None:
        with self.persist_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    g = Genome.from_dict(d)
                except (json.JSONDecodeError, KeyError, ValueError):
                    # Skip corrupt lines (best-effort resume).
                    continue
                if g.id == "baseline":
                    self.baseline = g
                    self._seen.add("baseline")
                    continue
                if g.id in self._seen:
                    # Update path (latest write wins).
                    self.islands[g.island_id].insert(g)
                    continue
                self._seen.add(g.id)
                self.islands[g.island_id].insert(g)

    def summary(self) -> dict:
        """Compact dict for human-readable logging."""
        best = self.best_genome()
        return {
            "num_islands":
            len(self.islands),
            "total_genomes":
            self.size(),
            "verified":
            sum(len(i.finite()) for i in self.islands),
            "baseline_fitness_ns":
            (None if not math.isfinite(self.baseline.fitness) else
             self.baseline.fitness),
            "best_fitness_ns": (None if best is None else best.fitness),
            "best_genome_id":
            None if best is None else best.id,
            "speedup_vs_baseline":
            (None if (best is None or not math.isfinite(self.baseline.fitness)
                      or not math.isfinite(best.fitness) or best.fitness == 0)
             else self.baseline.fitness / best.fitness),
        }
