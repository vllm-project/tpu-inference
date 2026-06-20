# `tools/kernel/evolve/` — TPU/Pallas kernel optimizer

A verifier-first, LLM-mutation + GA kernel optimizer for TPU Pallas
kernels. Inspired by AlphaEvolve, but with three independent moats —
numerical, semantic (adversarial critic), and statistical — to prevent
the false-positive class of "wins" that took down Sakana AI CUDA Engineer
and DeepReinforce CUDA-L1.

```
                           ┌──────────────────────────┐
        ┌──────►  Mutator: programmatic ▸ Claude on    │
        │       Vertex ▸ local vLLM (any of three)     │
        │                                              ▼
   ┌─────────┐    ┌──────────────────────────────┐    ┌────────────┐
   │ Archive │◄──┤ Diff applier + worktree     │───►│ Bench harness │
   │ (island │    │ + AST validate + import     │    │ (real TPU)    │
   │  GA)    │    └──────────────────────────────┘    └────────────┘
   └─────────┘                                              │
        │                                                   ▼
        │             ┌─────────────────────────────────────────────┐
        ├────────────►│ Verifier (3-tier):                          │
        │             │  • Math: dtype-allclose + cosine + anti-cheat│
        │             │  • Semantics: Opus critic refutes diffs     │
        │             │  • Stats: paired t-test on N-round samples  │
        │             └─────────────────────────────────────────────┘
        ▼
   Telemetry (JSONL) ──► analysis CLI ──► auto-PR diff ──► nightly CI
```

## Quickstart

```bash
# 1. Run the synthetic-matmul demo (no API key needed, programmatic mutator):
python -m tools.kernel.evolve.examples.matmul_evolve --generations 5

# 2. Run the Qwen3-0.6B real-model sweep:
python -m tools.kernel.evolve.examples.qwen3_rpa_evolve \
    --candidates "4:32,8:32,16:32" --max-tokens 64

# 3. Stats-bench any claimed win (paired t-test, 95% CI):
python -m tools.kernel.evolve.stats.qwen3_stats_bench \
    --candidates "8:32,8:128" --rounds 10

# 4. With Claude on Vertex AI (set up below):
export ANTHROPIC_VERTEX_PROJECT_ID=your-project
python -m tools.kernel.evolve.examples.claude_rpa_v3_evolve \
    --generations 5 --use-critic
```

## Setup

Required packages: `jax`, `numpy`, `optuna`, `anthropic` (already in repo deps).

For the Claude-on-Vertex mutator:
1. Enable Anthropic models in Vertex Model Garden:
   `https://console.cloud.google.com/vertex-ai/model-garden`
2. Authenticate: `gcloud auth application-default login`
3. Set project: `export ANTHROPIC_VERTEX_PROJECT_ID=your-project-id`
4. Models that have been verified working: `claude-opus-4-8`, `claude-opus-4-7`
   on `region='global'` (the Vertex auto-routing endpoint).

## Subpackage map

| Path | Purpose |
|---|---|
| `archive.py`, `genome.py`, `evaluator.py`, `orchestrator.py`, `worktree.py` | Core GA loop, candidate persistence, isolation |
| `mutator/` | LLM client backends (`AnthropicClient`, `VertexAnthropicClient`, `LocalLlmClient`, `StubClient`, `EnsembleClient` — multi-model round-robin), `ProgrammaticMutator`, `BestOfNMutator` (BoN), `ExamplePool` (RLAIF positive examples), `ChainingMutator`, prompts (with casebook of ~30 historical wins), diff applier, adversarial critic, `FailureLog` |
| `cross_shape.py` | Cross-shape statistical validation — bench a diff against production shape catalog (Qwen3, Llama 3, Qwen3.5, fp8 KV). Rejects shape-specific wins that regress elsewhere. |
| `lm_eval_gate.py` | Outermost correctness gate — gsm8k/mmlu_pro deltas before/after the patch, idempotent w.r.t. working tree (SHA-verified restore). |
| `e2e_benchmark.py` | End-to-end vLLM offline throughput bench, paired-t over N trials. Measures the user-visible number. |
| `auto_pr.py` | Auto-PR generator: takes verified evidence, applies the diff to a fresh branch, commits with Signed-off-by, optionally pushes + opens via `gh`. Idempotent dry-run mode for review. |
| `ship_pipeline.py` | Six-gate end-to-end orchestrator: numerics → critic → stats → cross-shape → lm-eval → e2e → auto-PR. Single command from candidate diff to PR. |
| `parallel/` | Subprocess-per-TPU-core fan-out for concurrent evaluation |
| `sweep/` | Auto-discovery of missing/suboptimal tuned-table entries, auto-PR diff generation |
| `stats/` | Paired t-test, Welch's t-test, Wilcoxon signed-rank, Qwen3 stats-bench CLI |
| `telemetry/` | JSONL event writer, analysis CLI |
| `cost_model/` | Learned surrogate that pre-filters expensive evals |
| `fidelity/` | Three-tier router: surrogate → microbench → full-model |
| `ci/` | `nightly_evolve` runner + kernel-specific recipes (RPA v3, MLA v2, quant matmul, fused MoE) |
| `kernelbench/` | TPU/Pallas port of Stanford KernelBench |
| `examples/` | End-to-end demos: synthetic matmul, Qwen3-RPA, Claude-RPA, KernelBench |
| `tests/` | 161+ unit + integration tests (CPU-runnable) |

## The six gates

Why this matters: every prior LLM-kernel-opt system has shipped a >100×
"speedup" that collapsed within days. Six independent gates make that
class of false positive impossible. A diff that passes all six is
production-ready by construction:

| # | Gate | What it catches |
|---|---|---|
| 1 | Numerical | NaN/Inf, dropped layers, scale regression, buffer aliasing, all-zero outputs |
| 2 | Semantic (Claude critic) | Plausible-looking but wrong mutations: precision-before-exp, NaN-poison, missing block_until_ready |
| 3 | Statistical (paired-t) | False wins masquerading as noise; effect size below detection threshold |
| 4 | Cross-shape | Shape-specific wins that regress on production shapes (Qwen3, Llama 3, fp8 KV variants) |
| 5 | lm-eval correctness | Subtle accuracy drift the math gate didn't catch (vLLM FP8/H100 lesson) |
| 6 | End-to-end throughput | Kernel win that doesn't translate to user-visible token/sec |

The first three live inside the evolution loop; the last three live in
`ship_pipeline.py`. The pipeline auto-emits a PR if all six pass.

### 1. Numerical (math-level)

Each candidate's output is compared to a pure-JAX eager reference via
`tools.kernel.tuner.v1.verifier.numerics.check_many`:

* NaN/Inf check → instant reject.
* Cosine similarity floor (default 0.9999) → catches dropped layers /
  scale regressions even when allclose is loose.
* dtype-aware `allclose` (atol/rtol from the oracle).
* `AntiCheatGuard` catches all-zero / constant / input-aliased outputs
  (the Sakana classics).

### 2. Semantic (LLM critic)

Before any TPU run, the proposed diff is passed to a cheaper Claude model
(typically Opus 4.7 when Opus 4.8 is the mutator) asked to *refute* the
change. Concrete failure modes are listed in the critic prompt — drop of
`block_until_ready`, accumulator-dtype regressions, mask removals under
sliding-window, etc. Verified in production runs: the critic caught
five repeated "use `pl.reciprocal(approx=True)` instead of `lax.div`"
proposals from Opus 4.8, each correctly reasoning that fp32 paths
deliberately preserved exact division.

### 3. Statistical (significance test)

Even after a candidate passes numerics + critic + the bench harness, its
*claimed speedup* must survive a paired t-test at p<0.05 over N=10
rounds. The `stats/qwen3_stats_bench.py` CLI implements this; see the
generated 95% CIs and Cohen's d. Real production finding: 4 of 4
candidates with mean speedups in the +0.3% to +3.0% range failed
significance — they were measurement noise, not real wins.

## Adding a new kernel target

See `HOWTO_NEW_KERNEL.md` for a step-by-step. Brief outline:

1. Define a `KernelHost` subclass exposing the verifier oracle, kernel
   symbol, input generator, and a `build_kernel_fn(module) -> Callable`.
2. (Optional) write a `ProgrammaticMutator` rule library or describe the
   tunable surface for the LLM.
3. Drop a recipe in `ci/recipes/your_kernel.py` exposing `run(out_dir)`.
4. `nightly_evolve` picks it up automatically when files under the
   kernel's directory change.

## Cost model: real numbers

* **Programmatic mutator**: free. The matmul demo finds a 1.12× win in ~13s.
* **Claude Opus 4.8 + 4.7 on Vertex**: ~$0.75 per RPA-v3-sized call (50k
  input tokens). A 5-generation × 2-island × 3-candidate run with critic
  enabled is ~60 LLM calls + ~30 critic calls = ~$30. Verifier rejects
  ~50% (real production data) before they reach TPU, so actual TPU
  compile budget is half.
* **Real-TPU eval per candidate**: ~5-30s including compile, depending on
  kernel size and shape.
* **Qwen3-0.6B full-model bench**: ~3min per config including model load.

## Telemetry & analysis

```bash
# Per-trial JSONL events are emitted to /tmp/<run>_telemetry.jsonl by
# every example script. Aggregate them:
python -m tools.kernel.evolve.telemetry.analyze \
    /tmp/qwen3_sweep_telemetry.jsonl
```

Output groups by kernel × shape × status, reports verified-rate, best
fitness per shape, and the distribution of failure modes — exactly the
data the FailureLog uses to feed anti-pattern hints back into the next
generation's prompts.
