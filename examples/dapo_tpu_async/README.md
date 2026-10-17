# Collocated GRPO async-overlap reproduction (TPU v7x, single 64-chip slice)

A reproducible reference for the async inter-wave rollout overlap discussed in
issue #3456 (async scheduling under `UniProcExecutor`) and issue #3455
(prompt-logprobs precompile). This extends the sync recipe in
[`examples/dapo_tpu`](../dapo_tpu) with the async-overlap levers and documents
the measured effect on a collocated GRPO rollout+train cycle.

## Setup

- **Hardware:** TPU v7x (Ironwood), 4×4×4, 64 chips, single slice
- **Placement:** collocated — trainer + rollout share all 64 chips in one
  `jax.distributed` session (not two slices)
- **Model / workload:** Gemma-class 2B, decode-light GRPO — global batch 1024,
  8 generations/prompt, prompt cap 512, response cap 64 (≈28 tok mean, EOS-terminated),
  temperature 0.7
- **Rollout mesh:** DP16 × TP8 (per-process DP, single-host `UniProcExecutor`)
- **vLLM:** a current LKG (the pinned `d626108b…` or newer) where
  `UniProcExecutor.supports_async_scheduling()` returns `True` — see the note below.

## The levers

| Lever | How | What it does |
|---|---|---|
| Async inter-wave overlap | `async_scheduling: true` in the engine config | rollout wave N+1 enqueues while wave N's `device_get` is in flight — attacks the rollout dispatch bubble |
| Skip prompt-logprobs precompile | `SKIP_PROMPT_LOGPROBS_PRECOMPILE=1` (see #3455 / PR #3457) | avoids the redundant prompt-logprobs precompile (off the execution path when `prompt_logprobs` is not requested; also avoids a multi-host startup hang) |
| Async collectives | `LIBTPU_INIT_ARGS="--xla_tpu_prefer_async_allgather_to_allreduce=true --xla_enable_async_all_gather=true"` | keeps the decode collectives off the synchronous critical path (also clears a `0x176` ICI decode stall on collocated SPMD) |
| SparseCore collective offload | `LIBTPU_INIT_ARGS+=" --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true"` | offloads rollout collectives off the TensorCore. Compiles cleanly on cold v7x; unlike the latency-hiding-scheduler, which blew up HLO compile here |

> **Important (issue #3456):** on **older** vLLM, `UniProcExecutor.supports_async_scheduling()`
> returned `False`, so the engine silently reset `async_scheduling` `true → false`
> and the runner's async branch was dead code (the rollout sat ~90% device-idle).
> On a **current** vLLM LKG this is already fixed — `supports_async_scheduling()`
> returns `True` and the config's `async_scheduling: true` is honored. So the
> reproduction requirement is simply: **use a current vLLM LKG and set
> `async_scheduling: true`**. No source override is needed. This example exists to
> make that reproducible end-to-end.

## Measured result (TPU v7x, 64 chips, collocated, correctness-gated)

| Config | Warm rollout+advantage chunk | Warm cycle (opt-to-opt) |
|---|---|---|
| Sync (no async overlap) | ~15.7 s | ~99–106 s |
| **+ async overlap (this example)** | **~13.6 s (~15% faster)** | **~94 s median** (best ~84 s) |

Correctness gate (both configs): loss trend, reward distribution, EOS counts, and
mean completion length all healthy and matched vs the sync path under a fixed seed —
async is a reordering, not a workload change.

## Run

See [`config_template.yaml`](./config_template.yaml) for the full set of keys and
[`launch.sh`](./launch.sh) for a one-command launch (adapt the model path and the
launcher to your environment; the async levers themselves are the portable part).
