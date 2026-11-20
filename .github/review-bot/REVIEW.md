# Review guide: tpu-inference

This repo is the vLLM plugin for TPU, now named vLLM JAX (`vllm-jax`). It runs JAX models (`MODEL_IMPL_TYPE=flax_nnx`) and vLLM's PyTorch models through torchax (`MODEL_IMPL_TYPE=vllm`). Each rule names its source in parentheses. Paths are relative to the repo root.

## 1. How to review

Read `.review/pr.md`, the diff, and each changed file in `./pr` with its callers. Apply every section 3 pass and each section 4 area whose path matches a changed file. Do not stop early: a missing test does not replace checking for crashes and wrong output.

Anchor each finding on a line the PR adds or changes. For a file the PR should have changed but did not (a model missing from `model_loader.py`, a stale doc), anchor on the changed line that causes it and name the file. For missing evidence, anchor on the line that needs it (a new kernel with no test). Problems only in the title or description go in the summary.

Order findings most severe first; only the first ones up to the limit are posted. Before output, drop anything section 5 covers.

These parts of `tpu_inference/` have no written rules yet: `runner/`, `worker/`, `core/`, `executors/`, `platforms/` (except the quantization list), `distributed/`, `spec_decode/`, `lora/`, `rl/`, `offload/`. Review them with the section 3 passes only.

## 2. How to comment

- [B], [S], and [N] give a rule's usual label: blocking, should-fix, or a nit. Label each finding by its own impact. Findings are the only inline comments. "Ask" in a rule means a question for the review summary, unless the code proves the failure; then it is a finding.
- Blocking here means wrong model output, a crash or hang at load or while serving, or a compile while serving.
- One issue per comment, on the exact line, under 1,500 characters.
- You can read only this repo: the base branch at `./` and the PR head at `./pr`. You cannot read vLLM, JAX, Flax, torchax, Qwix, or tokamax sources. Raise a possible mismatch with them only as a question that says what to check.
- Do not flag a design only because another approach exists. Do not claim a change is faster or slower; ask for a timing or a profile (`docs/developers_guide/profiling.md`).
- Write people's names as plain text, never as @-mentions. Never repeat a non-GitHub link from the PR; point to its line.

## 3. Repo-wide rules

### Pass 0: PR text and tests

- [S] **PR text** (`.github/PULL_REQUEST_TEMPLATE.md`). The description says what the PR does, why, and its known shortcomings. The Tests section says how the change was tested, with commands to reproduce. If the body still holds only the template's placeholder text, ask for both once, in the summary.
- [N] **Issue link** (PR template). A PR that fixes a GitHub issue links it on a `FIXES:` line.
- [S] **Tests with features** (`CONTRIBUTING.md`, Testing). A new feature or model comes with unit tests under `tests/`. A new user-facing feature or model also gets a CI config (see Buildkite CI). Ask when either is missing.
- [S] **Docs stay true** (PR template checklist). When the PR changes something a doc describes, it updates that doc: `docs/developers_guide/` (model registration, weight loading, profiling env vars), `docs/getting_started/out-of-tree.md`, `.buildkite/README.md`, `.buildkite/benchmark/README.md`, `tools/kernel/tuner/v1/README.md`.
- [N] **Comments on hard code** (PR template checklist). Code that is hard to follow gets a short comment that says why it is written that way.

### Pass 1: placement and dependencies

- [N] **Common, jax, or vllm** (`CONTRIBUTING.md`, Directory Structure). Under `tpu_inference/layers/` and `tpu_inference/models/`, code used by both JAX models and vLLM (torchax) models goes in `common/`. Code for one kind goes in `jax/` or `vllm/`. Flag new code placed in one kind's folder that only the other kind uses.
- [B] **vLLM pin** (`.buildkite/scripts/bootstrap.sh`, `.buildkite/scripts/configs/pipeline_config.sh`). PR builds install the vLLM commit in `.buildkite/vllm_lkg.version`. Code that relies on a vLLM API, signature, or behavior must work at that commit. When you cannot confirm it from this repo, ask.
- [S] **Generated files** (`.github/CODEOWNERS`, `.github/workflows/update_readme.yml`, `.buildkite/scripts/commit_support_matrices.sh`, `commit_verified_commit_hashes.sh`). Bots and `scripts/update_readme.py` write `support_matrices/**`, `docs/includes/*.md`, the `README.md` tables between `<!-- START: ... -->` and `<!-- END: ... -->`, and `verified_commit_hashes.csv`. Hand edits there get overwritten. Ask the author to change the generator or the CI step instead.
- [S] **Doc includes** (`mkdocs.yml`: `strict: true`, `pymdownx.snippets` with `check_paths: true`). The docs pull in `CONTRIBUTING.md`, `.buildkite/README.md`, `examples/offline_lora_inference.py`, `examples/multi_modal_inference.py`, and `docs/includes/*.md` with `--8<--` lines. Moving or renaming one of them without updating the include breaks the docs build.

### Pass 2: JAX correctness and serving

- [B] **Pure jitted code** (`docs/developers_guide/torchax_model_development.md`, Pure functions and KV Cache). A function under `jax.jit` takes all its data as arguments and returns all its results. An in-place write inside jit (such as a KV cache update) does not reach the caller: return the new array and assign it back. An array read from outside the function is baked into the compiled program as a constant, so it goes stale when the real value changes.
- [B] **No compiles while serving** (`torchax_model_development.md`, TPU JAX worker). The runner compiles every input shape at startup (`tpu_inference/runner/compilation_manager.py`) so that serving never compiles. A new input shape, static argument, or jitted path that serving can reach must be compiled there too. When the PR adds one, ask for a run with `VLLM_XLA_CHECK_RECOMPILATION=1`, which raises an error on any compile after warmup.

## 4. Area rules

### JAX models: `tpu_inference/models/jax/**`, `tpu_inference/models/common/model_loader.py`

- [B] **Register the model** (`docs/developers_guide/jax_model_development.md`, Model Registration). A new JAX model is added to `_MODEL_REGISTRY` in `_get_model_architecture()` in `models/common/model_loader.py`, under the `architectures` name from the checkpoint's Hugging Face config, which vLLM must also accept. Without it, the loader falls back to vLLM's PyTorch model and the new code never runs.
- [S] **Model shape** (`jax_model_development.md`, Code Organization; `docs/getting_started/out-of-tree.md`). A new model class subclasses `JaxModule` and `LoadableWithIterator`, and its constructor takes `(vllm_config, rng_key, mesh)`, the arguments `model_loader.py` passes. It builds from the `layers/jax` blocks (`JaxEmbed`, `JaxLinear`, `JaxEinsum`, `JaxRmsNorm`, `JaxMoE`) instead of raw Flax NNX, and keeps model-specific layers in its own model file.
- [B] **Forward contract** (`jax_model_development.md`, Implementing the forward pass). `__call__(kv_caches, input_ids, attention_metadata, ...)` returns the updated KV caches first, then the hidden states, in the tuple that `runner/tpu_runner.py` unpacks. All other state lives outside the model. The class also defines `compute_logits(hidden_states)`.
- [S] **Reuse weight loading** (`jax_model_development.md`, Weight Loading). Load through `LoadableWithIterator`, which runs `JaxAutoWeightsLoader`, not a new hand-written loop. Unpacking or reshaping one parameter goes in that parameter's `weight_loader` attribute. Fusing or splitting weights (MoE, MLA) goes in the module's `load_weights`.

### vLLM models through torchax: `tpu_inference/models/vllm/**`

- [B] **Weights and caches as inputs** (`torchax_model_development.md`, Pure functions and KV Cache). The jitted step function calls the PyTorch model with `torch.func.functional_call` and passes the weights in. KV caches go in through `vllm_model_wrapper_context` and come back as outputs (`vllm_model_wrapper.py`). Reading weights or caches from module attributes inside jit bakes them in as constants.

### Quantization: `tpu_inference/layers/*/quantization/**`, `tpu_inference/models/jax/utils/qwix/**`, `tpu_inference/platforms/tpu_platform.py`

- [B] **Supported list** (`jax_model_development.md`, Quantization Support). A new checkpoint `quant_method` is added to `TpuPlatform.supported_quantization` in `platforms/tpu_platform.py`. Otherwise vLLM rejects the checkpoint when it validates the config.
- [S] **Requantize after loading** (`jax_model_development.md`, Process weights after loading). Re-quantizing loaded weights to a TPU-friendly block size happens in the quant method's `process_weights_after_loading`, not in the model or the loader.
- [N] **Qwix configs** (`jax_model_development.md`, Code Organization). Qwix rule files (YAML) go in `tpu_inference/models/jax/utils/qwix/configs/`.

### Kernels: `tpu_inference/kernels/**`

- [S] **Kernel tests** (`CONTRIBUTING.md`, Testing; `.buildkite/pipeline_jax.yml`, "JAX unit tests - kernels"). A new kernel, or a new mode or dtype path in one, gets a test under `tests/kernels/`. That step runs per PR only when `tpu_inference/kernels/`, `tests/kernels/`, `tests/conftest.py`, or `requirements.txt` changes, and it skips the files on its `--ignore` list. Ask about a new test added to that list.
- [S] **Multi-chip kernel tests** (`.buildkite/pipeline_jax.yml`, `.buildkite/scripts/bootstrap.sh`). The main kernel step runs on the small queue (v6e-1 or v7x-2). Only `tests/kernels/collectives/` runs on the 8-chip queue, and only when `tpu_inference/kernels/collectives/` or its tests change. Ask where a new kernel test that needs more devices runs.

### Kernel tuner: `tools/kernel/tuner/v1/**`

- [S] **Order means memory** (`tools/kernel/tuner/v1/README.md`, Step 1). `TunableParams.__le__` and `__ge__` compare memory use. After an out-of-memory failure, the tuner skips every config that is `>=` the failed one (`is_oom_expected` in `common/kernel_tuner_base.py`), so a wrong order silently skips good configs. `TuningKey` and `TunableParams` are `@dataclass(frozen=True)`, not ordered.
- [S] **Failures return a status** (README, Step 3). `run()` returns `TuningStatus.FAILED_OOM` for an out-of-memory error and `TuningStatus.UNKNOWN_ERROR` for other failures, so one bad config does not crash the worker.
- [S] **Search space for Bayesian runs** (README, Step 3). With `support_bayesian_optimization=True`, the tuner overrides `get_search_space()`. The base version returns `{}`, which gives the optimizer nothing to sample.

### Tests: `tests/**`

- [S] **New tests must run** (`.buildkite/pipeline_jax.yml`, "JAX unit tests part2"). The per-PR unit step runs all of `tests/` except `tests/kernels/`, `tests/lora/`, `tests/e2e/`, and the files on its `--ignore` list. A test in those places runs only from a Buildkite step that names it. Ask which step runs a new file there.
- [S] **Device count** (`.buildkite/pipeline_jax.yml`). The unit steps run on the small queue (v6e-1 or v7x-2). A test that skips when it lacks devices passes silently there. It needs its own multi-chip step, as `tests/layers/common/test_moe_hierarchical_dispatch.py` has.
- [S] **bvt marker** (`tests/conftest.py`; `pipeline_jax.yml`, E2E speculative decoding step). In a module with any `@pytest.mark.bvt` test (today only `tests/e2e/test_speculative_decoding.py`), per-push builds run only the bvt tests and the rest run nightly. Ask whether a new test in such a module needs the marker.
- [N] **Mirror the code path** (`.github/CODEOWNERS`, "Tests mirror production-code ownership"). Tests for `tpu_inference/<area>/` go in `tests/<area>/`, so the right owners review them.

### Buildkite CI: `.buildkite/**`

- [S] **New model or feature in CI** (`.buildkite/README.md`). Add a model with `.buildkite/pipeline_generation/add_model_to_ci.py`, which writes to `.buildkite/models/`. Add a feature with `add_feature_to_ci.py`, which writes to `.buildkite/features/`, `parallelism/`, `quantization/`, `rl/`, or `kernel_microbenchmarks/`. Files there join the nightly run on their own; no other wiring is needed.
- [S] **Support matrix name** (`.buildkite/scripts/validate_pipeline_metadata.sh`, `CI_TARGET` tip). In `.buildkite/models/`, `CI_TARGET` is the model's full Hugging Face name (`org/model`, with an optional `/multimodal` or `/text-only` suffix). It becomes the row name in the support matrix.
- [S] **Keep soft_fail on bare-metal tests** (`.buildkite/features/kube/README.md`). In the bare-metal spec files, test steps keep `soft_fail: true` so the `record_step_result.sh` step after them still runs and records the result.
- [B] **Kube files hold steps only** (`.buildkite/models/kube/README.md`, `.buildkite/features/kube/README.md`). The uploader joins every file in a kube folder into one pipeline. A top-level `env:` would reach every step of the build, and a YAML anchor cannot be seen from another file.
- [S] **Kube files mirror bare metal** (same READMEs). A kube file has the same name and step keys as its bare-metal file. Its tests do not use `soft_fail`; the record step depends on the test with `allow_failure`. `validate_pipeline_metadata.sh` skips these folders, so check them by hand.

### Benchmark cases: `.buildkite/benchmark/cases/**`

- [S] **Keep established case names** (`.buildkite/benchmark/README.md`, section 3.D). Renaming the `case_name` of a case with history splits its results on the dashboard. A rename needs a database migration; ask whether one is planned.
- [S] **CI dataset paths** (README, section 3.C). For CI, a custom `dataset-path` file starts with `/workspace/tpu_inference/artifacts/dataset/`, where CI syncs the dataset files.
- [S] **Tracking values match** (README, section 3.D). `INPUT_LEN`, `OUTPUT_LEN`, `PREFIX_LEN`, and `EXPECTED_ETEL` are written to the results database, so they must match the case's real arguments and dataset.

### GitHub config: `.github/CODEOWNERS`

- [S] **Order matters** (`.github/CODEOWNERS`, header). The last matching pattern wins. A narrower path goes below the broader one it refines; above it, the broader owners replace it.

### Docs: `docs/**`, `mkdocs.yml`

- [S] **Add pages to nav** (`mkdocs.yml`, `nav`). Every new page gets a `nav` entry; a page left out cannot be reached from the site menu.
- [S] **Edit the source** (`docs/developers_guide/contributing.md`, `buildkite.md`). These two pages only include `CONTRIBUTING.md` and `.buildkite/README.md`. Edit the source file, not the include line.

## 5. Do not flag

- **Formatting and lint**. `.pre-commit-config.yaml` runs on every PR (`.github/workflows/pre-commit.yml`): yapf formatting, isort import order, ruff's default checks (such as unused imports and undefined names), clang-format, pymarkdown, actionlint, shellcheck, spaces in file names, a missing `__init__.py` under `tpu_inference/` (the hook creates it), and license headers (the hook adds one; both the Apache header and the SPDX lines are fine, so never ask to switch).
- **Sign-off**. DCO `Signed-off-by` lines; a commit-msg hook adds them.
- **CI checks**. Buildkite YAML syntax (`bk pipeline validate`). Spec-file metadata that `validate_pipeline_metadata.sh` checks: `CI_TARGET` unique and one per file; `CI_TPU_VERSION`, `CI_STAGE`, and `CI_CATEGORY` present and allowed; step keys and labels start with `${TPU_VERSION}`; record steps depend on the step they record; every step has an allowed `agents.queue`. Benchmark `case_name` uniqueness (`validate_case_names.py`). The CI generator templates (`test_generation.sh`).
- **PR process**. The `ready` label (Buildkite and merge both require it), the template comment that `pre-PR.yml` posts, CI status and failing jobs, requests for a CI run or build that proves a step ran (CI must pass before a PR merges), and missing CI runs on PRs that change only `.md` files, images, `docs/`, `CODEOWNERS`, or support-matrix CSVs (Buildkite skips those).
- **Old problems**. Issues the PR did not create or make worse. Still flag an old bug that the change starts to rely on.
- **Legacy models**. `llama3.py` and `gpt_oss.py` subclass `nnx.Module` directly. The Model shape rule applies to new models.
- **Test file names**. Both `test_*.py` and `*_test.py` (the kernel tests' style) are collected; the repo has no pytest config that changes this.
