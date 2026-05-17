# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`tpu-inference` is a vLLM hardware plugin that unifies JAX and PyTorch under a single lowering path on TPU. It is **not** a standalone inference engine — it registers itself with upstream vLLM via the `vllm.general_plugins` entry point (`tpu_inference.layers.vllm:register_layers`, see `setup.py`) and is picked up at vLLM import time when running on TPU.

vLLM is consumed as a normal package dependency. In this workspace it is also checked out as a sibling at `../vllm/` for cross-referencing upstream behavior; do not assume edits there are picked up unless that checkout is installed editably.

## Common commands

```bash
# Install (editable; pulls deps from requirements.txt — JAX 0.10, libtpu, torchax, flax, etc.)
pip install -e .

# Lint / format / static checks (yapf, isort, ruff, clang-format, pymarkdown, shellcheck)
pre-commit install --hook-type pre-commit --hook-type commit-msg
pre-commit run --all-files

# Tests — pytest layout mirrors the package
pytest tests/                          # full suite
pytest tests/core/sched/               # one subtree
pytest tests/runner/test_tpu_runner.py # one file
pytest tests/runner/test_tpu_runner.py::TestName::test_case  # one test

# Serve a model on TPU (the plugin activates automatically inside vLLM)
vllm serve <model> --tensor-parallel-size=<chips>
```

CI runs on Buildkite; pipeline YAMLs live under `.buildkite/`. New models and features are added via the generators in `.buildkite/pipeline_generation/` (see `.buildkite/README.md`) — do not hand-edit the generated YAMLs.

## Architecture

The plugin's job is to make vLLM run on TPU. The flow when `vllm serve` starts:

1. `tpu_inference/__init__.py` runs at import time. It loads `env_override.py` first (must precede any other module), then either takes the Pathways-proxy path (when `JAX_PLATFORMS` contains `proxy`) or the direct-TPU path. The Pathways path eagerly resolves `vllm.current_platform` because lazy resolution races with multi-host topology discovery.
2. vLLM's platform registry picks `TpuPlatform` from `tpu_inference/platforms/tpu_platform.py`, which declares dispatch key `XLA`, the supported quantization set, and shims a `vllm._C` namespace with dummy ops so upstream CUDA-eager imports don't crash.
3. `TpuPlatform` selects an executor:
   - default → `UniProcExecutor` (single host)
   - `TPU_MULTIHOST_BACKEND=ray` → `tpu_inference/executors/ray_distributed_executor.py`
   - There is also `executors/multiproc_executor.py` for the multi-process path.
4. The executor spins up workers (`tpu_inference/worker/tpu_worker.py`), each owning a `TPUModelRunner` (`tpu_inference/runner/tpu_runner.py`). The runner is where the hot path lives — it composes a `PersistentBatchManager`, `KVCacheManager`, `CompilationManager`, `MultimodalManager`, `SpeculativeDecodingManager`, `StructuredDecodingManager`, and `LoraUtils`, all under `tpu_inference/runner/`.
5. Scheduling: upstream vLLM's scheduler is used by default. When attention data-parallelism is enabled (`enable_dp_attention`), the custom DP scheduler at `tpu_inference/core/sched/dp_scheduler.py` is used instead — it spawns one scheduler worker per DP rank over pipes and arbitrates pending requests across ranks. This is what `repro_server.sh` / `repro_bench.sh` at the workspace root exercise; the `DP_SCHED_BATCH_PREFILL` flag is the A/B knob.

### Layered code organization

`layers/` and `models/` each split three ways and the split is load-bearing:
- `common/` — shared between JAX-native and torchax-backed (PyTorch via vLLM) paths
- `jax/` — JAX-native (Flax) implementations
- `vllm/` — torchax-backed paths that reuse upstream vLLM model code

Per `CONTRIBUTING.md`: when adding a model, enable the PyTorch path via torchax first, then optionally add a JAX-native version. Don't break the split — code that's only used by one path should not leak into `common/`.

### Other large subsystems

- `kernels/` — Pallas/JAX kernels (ragged paged attention, MLA, MoE/megablox, GDN, fused_moe, structured_sparse_matmul, sparse_core, etc.). Each subdir is its own kernel family.
- `distributed/` — KV transfer and TPU host-side KV pools; `jax_parallel_state.py` is the JAX-side counterpart to vLLM's parallel state.
- `offload/` — CPU/host offload manager and connector for KV caches.
- `spec_decode/jax/` — JAX implementations of speculative decoding draft models.
- `lora/` — LoRA glue (also see `runner/lora_utils.py`).
- `core/disagg_executor.py` + `distributed/tpu_connector*.py` — prefill/decode disaggregation.
- `envs.py` — all TPU-specific env vars; `env_override.py` sets defaults before anything else imports. Prefer `tpu_inference.envs.<NAME>` over `os.environ` so the validated accessors apply.

### Workspace-specific scripts (root of `dsr1/`, outside this repo)

The shell scripts in the parent directory wire `tpu-inference` to a vLLM serving setup:

- `repro_server.sh` + `repro_bench.sh` — single-host repro for the DP-scheduler stuckness theory on `wyzhang/bug/r1-scheduler` (touches `tpu_inference/core/sched/dp_scheduler.py`).
- `start_ray_master.sh` + `start_ray_slave.sh` + `start_vllm_server.sh` + `run_bench.sh` — multi-host DeepSeek-R1 (TP=16) over Ray. Note: `TPU_PREMAPPED_BUFFER_SIZE` and `TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES` must be set in the raylet's env (not just the driver's) because they are not in vLLM's driver→worker auto-copy list.

## Conventions

- All source files must carry the Apache-2.0 / Google LLC license header (`addlicense` pre-commit hook enforces this).
- Commits need a `Signed-off-by:` trailer (pre-commit `commit-msg` hook adds it automatically from your git config).
- Don't hand-edit `tpu_inference.egg-info/` or buildkite-generated YAMLs.
