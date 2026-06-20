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
"""Prompts that drive the mutator and the critic.

The mutator prompt is the load-bearing piece. AlphaEvolve and FunSearch both
demonstrated that the quality of the search is dominated by *how the LLM is
asked to mutate* — task framing, context, output format, and prior-art hints.
Concrete things this implementation does:

* States the goal — minimize latency, preserve correctness — and the gate
  the diff will be checked against (dtype-aware allclose + cosine).
* Bakes in the TPU-perf playbook families A–J as a "mental model" the model
  can reason from. Without this, mutations skew toward CUDA idioms.
* Provides parent diffs and their measured fitness as in-context examples,
  so the model can do gradient-style reasoning across generations.
* Constrains output format to a fenced unified diff. The diff applier
  rejects anything that doesn't parse.
* Hard-stops the model with "ONE structural change" — multi-change diffs
  are harder to attribute wins to and harder to revert.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

# Compact summary of the perf-skill playbook (families A–J). Used as context
# in the mutation prompt so the model can reason from established TPU-perf
# patterns rather than CUDA idioms.
_PERF_PLAYBOOK = """\
TPU/Pallas performance playbook (the families a good mutation usually fits):

A. Memory hierarchy & fusion: spend HBM bandwidth and registers like scarce
   resources. Hoist outside the inner loop; fuse adjacent ops; avoid
   round-trips to HBM that VMEM could absorb.
B. Pipelining & overlap: hide latency behind compute. Overlap DMA with MXU
   via `emit_pipeline`/double-buffering. An idle engine is wasted.
C. MXU vs memory-bound: "fewer FLOPs" is the wrong metric if the op is
   memory-bound. Convert gathers to one-hot matmul above a token threshold.
D. Specialize by regime/shape: one branchy kernel pessimizes both regimes.
   Separate decode vs prefill vs mixed paths.
E. Tiling, layout & block-size: derive sizes from problem-shape; respect
   the TPU's (8, 128) lane/sublane layout; avoid bank conflicts.
F. Collectives & sharding: move the least data; trust the mesh; don't split
   below the atomic unit. Fuse all-gather with matmul where it lands.
G. DP attention & scheduling: trade communication for memory; don't let one
   rank stall the rest.
H. Quantization & dtype: dequant in VMEM; keep compute precision separate
   from storage precision (e.g. bf16 store + fp32 accumulate).
I. Host & dispatch overhead: the host must feed the TPU. Donate buffers
   (`donate_argnames`); pre-flatten Python structures used in the trace.
J. Do less work: the cheapest op is the one you skip. Algebraic identities,
   causal-skip, sliding-window early-exit, size-gated fallbacks.
"""

# Compact case library — distilled from ~30 merged tpu-inference perf PRs.
# Each entry: pattern → mechanism → measured perf → family letter. This is
# the LLM's "institutional memory" of what has worked on this codebase.
# Drawn from the tpu-inference-perf casebook; updated as new PRs land.
_CASE_LIBRARY = """\
Past verified TPU/Pallas optimizations on THIS codebase (use as a mental
catalog when proposing changes — pattern names are searchable in git log):

[Memory hierarchy & fusion — family A]
* #2278 (A8, +60% decode E2E): `@jax.jit(donate_argnames=...)` for state
  buffers → XLA writes in place instead of HBM-copying every step.
* #2671 (A4, −25% decode kernel): fold SiLU into the in-VMEM tile instead of
  a separate full HBM round-trip.
* #2671 (A5, same PR): for GQA, do the qk dot on UNREPEATED heads BEFORE
  `jnp.repeat`, so the K-axis reduction runs once not ×repeat_factor.
* #2741 (A1/A2/A3, −20% bt>1 decode): move per-token slicing INSIDE the
  for-loop so the whole bt-block doesn't stay live in VREGs (cuts spill).
* #1928 (A11): fuse `act(gate)*up` inside the GMM epilogue in VMEM so the
  `[tokens, 2*intermediate]` matmul result never reaches HBM.

[Pipelining & overlap — family B]
* #2282 (B2): split flash QK·softmax and PV into separate stages and
  pipeline N's softmax with (N−1)'s PV — keeps MXU + VPU both busy.
* #2083 (B4, F3): chunk a `psum` along token dim and overlap each chunk's
  all-reduce with the next chunk's SparseCore kernel.

[MXU vs memory-bound — family C]
* #2674 (C1, C2, +14% at small N): replace gather-permute with `one_hot @ x`
  on the MXU when token count ≤ threshold; fold gate+mask+topk into one
  matmul on the way back.

[Regime specialization — family D]
* #1820 (D1, ×1.14): split monolithic RPA into decode / prefill / mixed
  pallas_call's with per-regime block sizes.
* #2056 (D1 at launch level): split MLA into two `pallas_call` instances —
  one with `static_q_len=1` (decode), one mixed — turn q_len constant.
* #2394 (D2, D3, ×2.7 prefill): split ragged-conv into dense+boundary-fixup
  prefill vs concat+einsum decode (each token = its own request).

[Tiling, layout & block-size — family E]
* #2282 (E1–E4): drive block sizes from a VMEM cost model — model double/
  triple buffering + transient compute peak + batch, cap at 80% VMEM.
* #2551 (E6, −80% MLA transpose): replace XLA transpose with a tiled,
  double-buffered Pallas transpose; rewire to head-major end-to-end so the
  transpose mostly disappears.
* #2550 (E5): account for LHS+RHS+scale+bias+accumulator+output (×2 buf)
  when picking GMM tile size, not just RHS×2.
* #2653 (E7, ~1.04×): pick KV-reshape strategy by num_kv_heads — drop
  `with_layout_constraint` when XLA's natural layout already works.

[Collectives & sharding — family F]
* #2661 (F7–F9, ×1.96): replicate KV head when TP>num_kv_heads → all-to-all
  in attention disappears.
* #2679 (F1–F5): replace `psum` with `psum + psum_scatter` to remove the
  XLA-injected padding that broke SparseCore divisibility.
* #2174 (F13, −50% gather time): bitcast-pack indices+weights into one
  int32 array → ONE all-gather instead of two.
* #2500 (F14): hierarchical reduce-scatter — intra-chip cores first, then
  log-step hypercube across chips.

[DP attention & scheduling — family G]
* #2577 (G1–G5, +62% c=512): partition mamba slot pools per DP rank with
  global↔local remap; scheduler tracks pending prefill tokens locally.

[Quantization & dtype-for-bandwidth — family H]
* #2503 (H1–H3): keep 4-bit weights packed in HBM, dequant in VMEM right
  before the MXU matmul — 1/4 HBM traffic, dequant cost is free in VMEM.
* #1841 (H4): allow mixed int4_weight × fp8_activation on the MXU;
  accumulator dtype follows the operands.
* #2482 (H7, +15% conc=512): store SSM/mamba state in bf16; upcast to f32
  on VMEM load. Half the HBM bytes; compute precision unchanged.
* #2612 (H6): `jax.clear_caches()` after model loading — JIT caches reserve
  HBM that's stolen from KV otherwise (148→113 GiB).

[Host & dispatch overhead — family I]
* #2615 / #2657 (I1–I3, +79% Gemma-4-31B throughput): pre-flatten the
  nnx.State pytree once at init; pass flat leaves into JIT; reconstruct
  inside the traced fn. Cuts ~7600 Python calls per decode step.

[Do less work — family J]
* #2498 (J1, +2.7–4.5%): algebraic identity for GDN output projection —
  `o = q@(h+k⊗b) = q@h + (q·k)·b`, the rank-1 matmul collapses to a dot.
* #2303 (J4): if `size(x)*packing*2 < vmem*0.6`, skip the SparseCore kernel
  and do plain `x[indices]` on the main core — launch-cost crossover.
* #1996 (J3): stop zero-initializing the GMM output buffer; mask invalid
  rows just before the topk reduction instead.
* #2674 (J2 inferred): RPA causal-skip — stop the KV loop at the causal
  frontier (`end_bkv_idx = cdiv(min(kv_len, processed+bq), bkv)`).

Recurring shapes that have NOT worked here (don't propose):
* `pl.reciprocal(approx=True)` — fp32 division on TPU is fast; the approx
  version trades fp32 precision for nothing.
* bf16 downcast BEFORE `jnp.exp` — precision loss past the verifier.
* Naive `lax.fori_loop` per token without donation — XLA copies state.
* Multi-hunk drive-by reformatting — every hunk must serve ONE hypothesis.
"""

_MUTATION_SYSTEM_PROMPT = """\
You are an expert TPU/Pallas kernel engineer working in an automated
evolutionary loop. A driver compiles your proposed change, measures latency
on a real TPU, and gates it through a strict numerics verifier (dtype-aware
allclose + cosine similarity floor + anti-cheat detectors that catch
zero/constant/input-aliased outputs).

Your job each call: propose ONE structural change to the kernel that should
make it faster while preserving numerical correctness within the verifier's
tolerance.

Hard rules:

1. Output exactly one unified diff inside a single fenced ```diff block.
   The diff MUST apply cleanly to the baseline source. Use the standard
   format: a/<path> and b/<path> headers, `@@` hunks, ` ` context lines,
   `-` deletions, `+` additions.
2. Change as little as possible. Multi-hunk diffs are fine but every hunk
   must serve the SAME hypothesis. Don't refactor while you're at it.
3. Preserve all public function signatures unless your hypothesis is
   specifically about the signature.
4. Don't introduce new imports of modules not already imported in the file.
5. Don't add tracing/print statements.
6. Briefly state your hypothesis in plain text BEFORE the diff. Two or three
   sentences. The driver logs this so a human reviewer can trace the win.

What WILL get your candidate rejected:

* The diff doesn't apply (wrong context lines, wrong line numbers).
* The mutated file fails to parse as Python.
* The kernel raises at compile time (Pallas constraint violation).
* The kernel returns NaN/Inf, or outputs that fail allclose/cosine vs the
  eager reference, or outputs bitwise identical to an input ("returns
  input"), all-zero outputs, or constant outputs.
* The latency is worse than the baseline by more than 5% (we record it but
  the loop down-ranks it).

Performance playbook (use as a mental model, not a checklist):

""" + _PERF_PLAYBOOK + """

""" + _CASE_LIBRARY


@dataclasses.dataclass
class ParentSummary:
    """Compact description of a parent genome shown in-context."""
    id: str
    fitness_ns: float | None
    diff: str
    notes: str = ""


def _format_parent(p: ParentSummary) -> str:
    fit = ("unknown"
           if p.fitness_ns is None else f"{p.fitness_ns / 1000:.1f} us")
    return (f"### Parent {p.id} (measured: {fit})\n"
            f"{p.notes}\n"
            f"```diff\n{p.diff.strip()}\n```\n")


def build_mutation_prompts(
    *,
    kernel_name: str,
    baseline_path: str | Path,
    baseline_source: str,
    baseline_fitness_ns: float | None,
    parents: list[ParentSummary],
    last_critique: str | None = None,
    extra_context: str = "",
    profile_snippet: str | None = None,
    anti_patterns: list[str] | None = None,
    rejected_diffs: list[str] | None = None,
    positive_examples_block: str = "",
    cost_model_hint: str = "",
) -> tuple[str, str]:
    """Return ``(system, user)`` prompts for the mutator.

    ``anti_patterns`` and ``rejected_diffs`` come from the FailureLog +
    archive. They are the system's accumulated *failure memory*: things the
    LLM has tried and had rejected, so the next proposal can move on rather
    than re-attempt the same dead-end. Concretely solves the "Claude tries
    `pl.reciprocal(approx=True)` five times in a row" attractor that real
    Vertex runs exposed.
    """
    baseline_label = str(baseline_path)
    fit_line = ("Baseline latency: unknown (will be measured this generation)."
                if baseline_fitness_ns is None else
                f"Baseline latency: {baseline_fitness_ns / 1000:.1f} us "
                f"(avg over warmup + 10 timed iters).")

    parent_block = "\n".join(_format_parent(p)
                             for p in parents) if parents else (
                                 "_(none — this is the first generation)_")
    critique_block = (
        f"\nThe previous critic flagged the last attempt as: {last_critique}\n"
        if last_critique else "")
    profile_block = (
        f"\nRecent profile:\n```\n{profile_snippet.strip()}\n```\n"
        if profile_snippet else "")
    extra_block = (f"\n{extra_context}\n" if extra_context else "")

    # Anti-patterns + previously-rejected diffs form the failure-memory
    # section. Putting them BEFORE the source keeps them in-context as the
    # model reads the kernel.
    failure_memory = ""
    if anti_patterns:
        lines = "\n".join(f"- {ap}" for ap in anti_patterns[:6])
        failure_memory += (
            "\n## Mutation classes that keep failing (avoid these)\n"
            f"{lines}\n")
    if rejected_diffs:
        rej_lines = []
        for i, d in enumerate(rejected_diffs[:4], start=1):
            preview = "\n".join(d.strip().splitlines()[:8])
            rej_lines.append(
                f"\n### Rejected attempt #{i}\n```diff\n{preview}\n```\n")
        failure_memory += (
            "\n## Recent rejected diffs (don't repeat the same idea)\n" +
            "".join(rej_lines))

    cost_model_block = (f"\n{cost_model_hint}\n" if cost_model_hint else "")

    user = (f"Target kernel: **{kernel_name}** at `{baseline_label}`\n\n"
            f"{fit_line}\n"
            f"{profile_block}"
            f"{extra_block}\n"
            f"## Parent diffs (highest-ranking ancestors)\n"
            f"{parent_block}\n"
            f"{critique_block}"
            f"{failure_memory}"
            f"{positive_examples_block}"
            f"{cost_model_block}\n"
            f"## Baseline source ({baseline_label})\n"
            f"```python\n{baseline_source}\n```\n\n"
            f"State your hypothesis (2–3 sentences), then output ONE unified "
            f"diff in a fenced ```diff block. **If you've been repeating the "
            f"same kind of change (see Rejected attempts above), try a "
            f"DIFFERENT family from the playbook this turn.**")
    return _MUTATION_SYSTEM_PROMPT, user


_CRITIC_SYSTEM_PROMPT = """\
You are an adversarial reviewer of a proposed TPU/Pallas kernel mutation.
Your job: try to *refute* the change before the driver burns TPU time on it.

Ground your refutation in a SPECIFIC failure mode. Examples:

* "The diff drops a `block_until_ready`, so timing measures returns before
  async work completes."
* "The diff changes accumulator dtype from fp32 to bf16, which will fail
  the cosine floor at long contexts."
* "The diff replaces `jnp.einsum` with a manual matmul that doesn't
  preserve the (head, seq, dim) axis order, so the output shape is wrong."
* "The diff removes the causal mask under sliding_window — outputs at
  positions past the window will leak future tokens."

Output exactly one of:

- `VERDICT: likely_broken` followed by a one-sentence reason.
- `VERDICT: likely_correct` followed by a one-sentence reason.
- `VERDICT: unsure` followed by a one-sentence reason (you couldn't find a
  concrete failure mode AND couldn't justify correctness).

Be skeptical. False positives (rejecting a real win) are recoverable; false
negatives (passing a numerics-wrong candidate) waste a TPU run.
"""


def build_critic_prompts(
    *,
    diff: str,
    kernel_name: str,
    baseline_path: str | Path,
    baseline_source_snippet: str,
) -> tuple[str, str]:
    """Return ``(system, user)`` prompts for the critic."""
    user = (f"Target kernel: **{kernel_name}** at `{baseline_path}`.\n\n"
            f"## Proposed diff\n```diff\n{diff.strip()}\n```\n\n"
            f"## Relevant baseline excerpt\n"
            f"```python\n{baseline_source_snippet.strip()}\n```\n\n"
            f"Try to find ONE concrete reason this will fail the verifier or "
            f"crash. If you cannot, say so. Output a single `VERDICT:` line.")
    return _CRITIC_SYSTEM_PROMPT, user
