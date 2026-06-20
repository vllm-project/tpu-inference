# Architecture rationale

This document explains *why* the design decisions were made. The
`README.md` covers *how* to use the system.

## Why island-based GA (vs. single population)

A single population converges quickly to a local optimum, especially
when mutation-quality variance is high (every LLM call lands somewhere
on a wide quality distribution). Islands + periodic migration is the
canonical AlphaEvolve / FunSearch escape: each island explores a
distinct basin, top-K migration prevents premature consensus.

## Why diff-based candidates (vs. whole-file rewrites)

Three reasons:

1. **Attribution.** A single-hunk diff makes wins traceable to a
   specific change. Whole-file rewrites lose this.
2. **Revertability.** `git apply -R` reverses a bad ship. Whole-file
   rewrites require manual surgery.
3. **Context efficiency.** A 1900-LOC kernel sent every turn is ~50k
   tokens. Asking for a diff (rather than a full rewrite) cuts the
   *output* token cost ~100× and reduces the surface where the model can
   accidentally drift the unrelated parts of the file.

## Why three mutator backends

* `ProgrammaticMutator`: deterministic, free, exhaustive over a defined
  parametric surface. Wins when the search is *within* a smooth
  parameter landscape (block sizes, dtypes).
* `VertexAnthropicClient` / `AnthropicClient`: structural mutation — the
  kind of refactor a human engineer would write. Wins are bigger but
  rarer, and you pay $1/call.
* `LocalLlmClient`: vLLM-hosted open-source models. Cheap, on-prem,
  matches Anthropic for parametric tasks but lower-quality on
  structural reasoning.

The orchestrator's `Mutator` parameter is `LLMClient`-typed (Protocol),
so all three drop in interchangeably. Production runs use programmatic
* Claude in alternation; the `ChainingMutator` composes them into one
call.

## Why three independent verifier moats

Single-moat failures are documented:

* Sakana CUDA Engineer: numerical-only moat (allclose at `atol=1e-2`)
  was bypassed by output-buffer-reuse exploits — passed unit tests with
  >100× "speedups" that collapsed under audit.
* DeepReinforce CUDA-L1: numerical-only moat let through 33% of
  "speedups" that were harness-state exploits.
* Every academic LLM-kernel-opt paper without a critic gate has shipped
  >10× false-positive wins.

Three independent moats — math, semantics, statistics — give us a
multiplicative reduction in false-positive probability. The Vertex run
on RPA v3 (5 critic-rejections in a row, all on the same wrong idea)
shows the semantic moat catching what numerics-with-tight-tolerance
*would* have passed for the random test inputs the bench harness uses.

## Why subprocess-per-TPU-core (vs. threads or multi-host)

JAX/XLA compilation state is process-global. Two concurrent runs of two
different kernel variants in the same process *will* clobber each
other's compile caches and produce nondeterministic results. Subprocess
isolation costs ~200ms per spawn — amortized over multi-second kernel
runs, this is free. Multi-host (cross-VM) would require a coordinator;
not worth the complexity for current scale.

## Why JSONL telemetry (vs. SQLite or a database)

Append-only JSONL survives crashes losslessly, lets `tail -f` work for
live monitoring, and the analysis pass is `jq` (or `summarize()`) one-
liners. A real DB would also need a migration story for the schema as
fields evolve — not worth it until we hit ~10⁶ events.

## Why the `global` Vertex region

Anthropic-on-Vertex's `global` region is the documented auto-routing
endpoint. Specific regions (`us-east5` etc.) work, but each one
requires per-region model enablement in Model Garden — more brittle.
`global` Just Works as long as the model is enabled at the project
level.

## Why FailureLog feeds back into the prompt

Real Vertex runs produce the same wrong idea repeatedly when the LLM
falls into a cognitive attractor (e.g. "use approximate reciprocal").
Showing the previous rejected diffs *in the prompt* gives the LLM the
chance to recognize the pattern and try a different family. Without
this, the same $1 per call is spent re-proposing the same dead-end
mutation 5+ times.

## Why paired t-tests on per-round samples (vs. mean comparison)

Single-shot mean comparisons confuse measurement noise with real
effects. Per-round throughput on Qwen3-0.6B has ~10% std-dev across
rounds, even after warmup. A claimed +5% improvement has 95% CI
[-10%, +20%] — *consistent with no change*. The paired t-test
explicitly tests "is the improvement larger than noise?" and reports a
p-value. We added this AFTER documenting the prior turn's "wins" were
actually within noise — see `stats/qwen3_stats_bench.py` output.

## Why a critic at all (vs. trusting the verifier)

The numerical verifier compares outputs on specific random inputs. It
can be wrong in two ways:

* False negatives: the wrong-on-paper kernel happens to be right on
  *these* inputs. Random inputs from `gen_random(seed=...)` are weak
  protection against adversarial inputs.
* False positives: the right-on-paper kernel hits numerical-precision
  edge cases the verifier didn't anticipate.

The critic provides *semantic* understanding orthogonal to numerical
checks. The Vertex production run demonstrated this: the critic
correctly identified "lax.div was deliberately preserved here for fp32
precision" — reasoning a numerical check could never produce.
