# Kernel Tuner v1

A framework for measuring and tuning the latency of TPU kernels. Results are stored either locally (JSON files) or in Google Cloud Spanner.

---

## 1. Implementing a Custom Kernel Tuner

To add a new kernel to the tuning framework, create a new file (e.g. `my_kernel_tuner.py`) and subclass `KernelTunerBase`. You can add kernel specific flags in this file. To avoid name confliction, the flags should be named in the format of {your_kernel_name}\_{flag_name} in this tuner.py and should append KERNEL_TUNING_ as prefix when invoked through Buildkite UI. For example: flag your_kernel_name_flag_name in your tuner.py should result in specifying KERNEL_TUNNING_YOUR_KERNEL_NAME_FLAG_NAME in BuildKite UI.

### Step 1 — Define `TuningKey` and `TunableParams`

`TuningKey` describes the fixed properties of a kernel invocation (shapes, types, etc.).  
`TunableParams` describes the parameters you want to search over (tile sizes, etc.).  Must implement `__ge__(self, other)` and `__le__(self, other)` function.
Both must be `@dataclass` so the framework can serialize/deserialize them.

```python
import dataclasses

@dataclasses.dataclass
class MyTuningKey:
    batch_size: int
    seq_len: int

@dataclasses.dataclass
class MyTunableParams:
    tile_m: int
    tile_n: int
```

### Step 2 — Subclass `KernelTunerBase`

```python
import itertools
import time

from tools.kernel.tuner.v1.common.kernel_tuner_base import (
    KernelTunerBase, TunerConfig, TuningCase, TuningStatus)


class MyKernelTuner(KernelTunerBase):

    def __init__(self, run_config):
        self.tuner_config = TunerConfig(
            tuning_key_class=MyTuningKey,
            tunable_params_class=MyTunableParams,
            kernel_tuner_name="my_kernel_tuner",  # must match KERNEL_TUNER_REGISTRY key
            support_bayesian_optimization=True,   # opt-in to Bayesian optimization
            n_bayesian_trials=50,                 # optuna trials per TuningKey bucket
        )
        self.run_config = run_config
        super().__init__(tuner_config=self.tuner_config, run_config=run_config)
```

`support_bayesian_optimization=True` switches `measure_latency` from a full sweep to
an [optuna](https://optuna.org/)-powered Bayesian search.  Set it to `False` (or omit
it) to keep the original sweep behaviour.  The mode can also be overridden at runtime
with the `--bayesian_optimization` flag (see Section 2).

### Step 3 — Implement the abstract methods

#### `get_search_space(tuning_key) -> dict`  *(required for Bayesian optimization)*

Returns the tunable parameter search space for a given `TuningKey` as a dictionary
mapping each `TunableParams` field name to a list of candidate values.  Both
`generate_cases` (to enumerate all combinations) and the Bayesian optimizer (to
build the optuna search space) call this method, so the search space only needs to
be defined in one place.

The search space can be **static** or **dynamic** (key-dependent):

```python
    def get_search_space(self, tuning_key: MyTuningKey) -> dict:
        # Example: wider tile_n range for larger seq_len
        tile_n_values = [16, 32, 64]
        if tuning_key.seq_len >= 256:
            tile_n_values = [32, 64, 128, 256]
        return {
            'tile_m': [16, 32, 64],
            'tile_n': tile_n_values,
        }
```

Optuna calls `trial.suggest_categorical(param_name, values)` for each entry, so
every suggested combination is guaranteed to map to a pre-generated case ID in the
database.

#### `generate_cases() -> list[TuningCase]`

Returns the full Cartesian search space as a flat list of `TuningCase` objects.
Call `get_search_space` here so the two stay in sync:

```python
    def generate_cases(self) -> list[TuningCase]:
        cases = []
        for bs, sl in itertools.product([1, 2, 4], [128, 256]):
            tuning_key = MyTuningKey(batch_size=bs, seq_len=sl)
            search_space = self.get_search_space(tuning_key)
            for combo in itertools.product(*search_space.values()):
                params = dict(zip(search_space.keys(), combo))
                cases.append(TuningCase(tuning_key, MyTunableParams(**params)))
        return cases
```

It is recommended to prune invalid tuning cases (cases that will OOM, or cases that
do not satisfy alignment requirements) at this stage to reduce the search space.
Results are persisted so re-runs with the same `case_set_id` skip this step.

#### `generate_inputs(tuning_key: MyTuningKey) -> dict`

Prepares the kernel inputs for a given `TuningKey`. The base class caches the
result so inputs are only regenerated when the key changes.

```python
    def generate_inputs(self, tuning_key: MyTuningKey) -> dict:
        if self._tuning_key and tuning_key == self._tuning_key:
            return self._kernel_inputs_cache
        self._tuning_key = tuning_key
        self._kernel_inputs_cache = {
            'x': jnp.ones((tuning_key.batch_size, tuning_key.seq_len)),
        }
        return self._kernel_inputs_cache
```

#### `run(tuning_key, tunable_params, iters) -> tuple[TuningStatus, float, float]`

Runs the kernel `iters` times and returns `(status, avg_latency_ns, total_latency_ns)`.  
Return `TuningStatus.FAILED_OOM` for OOM errors and `TuningStatus.UNKNOWN_ERROR` for other failures so the framework can record them without crashing the worker. A simple example looks like below:

```python
    def run(self, tuning_key: MyTuningKey, tunable_params: MyTunableParams,
            iters: int = 1) -> tuple[TuningStatus, float, float]:
        inputs = self.generate_inputs(tuning_key)
        try:
            start_ns = time.perf_counter_ns()
            for _ in range(iters):
                my_kernel(inputs['x'], tunable_params.tile_m, tunable_params.tile_n).block_until_ready()
            total_ns = time.perf_counter_ns() - start_ns
            return TuningStatus.SUCCESS, total_ns / iters, total_ns
        except Exception as e:
            logger.warning(f"Kernel failed: {e}")
            return TuningStatus.UNKNOWN_ERROR, 0.0, 0.0
```

### Step 4 — Register the tuner

Add your class to `KERNEL_TUNER_REGISTRY` in [kernel_tuner_runner.py](kernel_tuner_runner.py):

```python
from tools.kernel.tuner.v1.my_kernel_tuner import MyKernelTuner

KERNEL_TUNER_REGISTRY = {
    'example_kernel_tuner':    ExampleKernelTuner,
    'rpa_v3_kernel_tuner':  RpaV3KernelTuner,
    'mla_kernel_tuner': MlaKernelTuner,
    'my_kernel_tuner':      MyKernelTuner,   # <-- add this
}
```

### Step 5 — Run it

Locally:

```bash
python -m tools.kernel.tuner.v1.kernel_tuner_runner \
  --kernel_tuner_name=my_kernel_tuner \
  --run_locally=True \
  --case_set_id=my_first_run \
  --case_set_desc="My kernel first tuning run"
```

On Buildkite, set `KERNEL_TUNING_KERNEL_TUNER_NAME=my_kernel_tuner` in the build environment variables (see Section 2).

---

## 2. Running Locally

Install dependencies first:

```bash
pip install absl-py optuna
```

We recommend running the tuner with local storage first to verify the customized
kernel_tuner is set up correctly.

**Bayesian optimization** (default for `example_kernel_tuner`):

```bash
python -m tools.kernel.tuner.v1.kernel_tuner_runner \
  --run_locally \
  --kernel_tuner_name=example_kernel_tuner \
  --case_set_id=my_bayes_run_1 \
  --case_set_desc="Bayesian optimization run" \
  --run_id=0 \
  --tpu_version=tpu7x \
  --tpu_cores=2 \
  --bayesian_optimization=True
```

**Full sweep** (evaluate every pre-generated case):

```bash
python -m tools.kernel.tuner.v1.kernel_tuner_runner \
  --run_locally \
  --kernel_tuner_name=example_kernel_tuner \
  --case_set_id=my_sweep_run_1 \
  --case_set_desc="Full sweep run" \
  --run_id=0 \
  --tpu_version=tpu7x \
  --tpu_cores=2 \
  --bayesian_optimization=False
```

> Each `case_set_id` is persisted and cannot be reused with a different description,
> so use a distinct ID when switching between modes.

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `--kernel_tuner_name` | `example_kernel_tuner` | Which tuner to run. |
| `--run_locally` | `False` | Use local JSON storage instead of Spanner. |
| `--case_set_id` | _(required)_ | Identifier for this set of tuning cases. |
| `--case_set_desc` | `""` | Human-readable description. |
| `--run_id` | `"0"` | Run ID within the case set. |
| `--tpu_version` | `""` | TPU generation, e.g. `tpu6e` or `tpu7x`. |
| `--tpu_cores` | `0` | Number of TPU cores. |
| `--bayesian_optimization` | _(tuner default)_ | `True` = Bayesian optimization (optuna). `False` = full sweep. Omit to use the tuner's `TunerConfig` default. |
| `--debug` | `False` | Print results after each case iteration. |

Local results are written to JSON files under `/tmp/kernel_tuner_run_{case_set_desc}_{timestamp}/`.

---

## 3. Running on TPU VMs via Buildkite

The pipeline is defined in `.buildkite/pipeline_kernel_tuning.yml` and bootstrapped by `.buildkite/scripts/bootstrap_kernel_tuning.sh`.

### Pipeline overview

1. **Bootstrap** (`bootstrap_kernel_tuning.sh`) — uploads the static `pipeline_kernel_tuning.yml`.
2. **Build** — builds and pushes the `vllm-tpu` Docker image.
3. **Generate cases + upload dynamic pipeline** — runs `kernel_tuner_runner` inside Docker with `--generate_buildkite_pipeline=True`. The generated YAML is written to `/tmp/kernel_tuning/generated_pipeline.yml` (shared with the host via a volume mount) and then uploaded to Buildkite with `buildkite-agent pipeline upload`.
4. **Tuning jobs** — the dynamically-uploaded pipeline fans out individual tuning jobs across TPU workers.

### Triggering a build

**Option A(Recommended) — Buildkite UI:**

1. Go to the pipeline page.
2. Click **New Build**.
3. Set your branch.
4. Expand **Environment Variables** and set the variables listed below.
5. Click **Create Build**.

Make sure to include both `KERNEL_TUNING_TPU_VERSION` and `KERNEL_TUNING_TPU_CORES` so the runner can resolve the correct TPU queue.

**Option B — Buildkite REST API:**

```bash
curl -s -X POST \
  -H "Authorization: Bearer $BUILDKITE_API_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.buildkite.com/v2/organizations/tpu-commons/pipelines/tpu-inference-kernel-tuning/builds" \
  -d '{
    "commit": "'"$(git rev-parse HEAD)"'",
    "branch": "'"$(git rev-parse --abbrev-ref HEAD)"'",
    "message": "kernel tuning run",
    "env": {
      "KERNEL_TUNING_TUNER_KERNEL_NAME":    "rpa_v3_kernel_tuner",
      "KERNEL_TUNING_CASE_SET_ID":    "my_case_set_001",
      "KERNEL_TUNING_RUN_ID":         "001",
      "KERNEL_TUNING_CASE_SET_DESC":  "My tuning run description",
      "KERNEL_TUNING_TPU_VERSION":    "tpu7x",
      "KERNEL_TUNING_TPU_CORES":      "8"
    }
  }'
```

### Required environment variables

Set these in the Buildkite **New Build → Environment Variables** section:

| Variable | Example | Description |
|---|---|---|
| `KERNEL_TUNING_KERNEL_TUNER_NAME` | `rpa_v3_kernel_tuner` | Name of the kernel tuner to run. Must match a key in `KERNEL_TUNER_REGISTRY` defined in kernel_tuner_runner.py. |
| `KERNEL_TUNING_CASE_SET_ID` | `gmm_v2_tuning_001` | Unique identifier for this case set. Used as the primary key in Spanner. |
| `KERNEL_TUNING_RUN_ID` | `001` | Run ID within the case set. Increment for re-runs of the same case set. |
| `KERNEL_TUNING_CASE_SET_DESC` | `"Your description about this case set"` | Human-readable description stored alongside results. |
| `KERNEL_TUNING_TPU_VERSION` | `tpu6e` or `tpu7x` | TPU generation. Controls which agent queue and `TPU_VERSION` env var are used. |
| `KERNEL_TUNING_TPU_CORES` | [1, 8] for `tpu6e` or [2, 8, 16] for `tpu7x` | Together with `KERNEL_TUNING_TPU_VERSION`, this controls the TPU config for tuning jobs. For example, `tpu6e` and `8` runs tuning on a tpu6e TPU with 8 cores. |

---

## 4. Inspecting Results

Use the interactive CLI:

```bash
python tools/kernel/tuner/v1/inspect_result_cli.py
```

On startup, select the result source:

```
Select result source:
  1) local   – local JSON files
  2) spanner – Google Cloud Spanner
Enter 1 or 2:
```

Once connected, the prompt shows your current session context (e.g. `inspect|cs=my_case_set|run=001>`).

### Session commands

#### Set session defaults

```
set_case_set_id ID    # avoids typing --case_set_id on every command
set_run_id ID         # avoids typing --run_id on every command
```

#### List case sets

```
list_case_sets [--filter KEYWORD]
```

Shows `case_set_id`, description, status, scan space size, and number of runs. Use `--filter` to narrow by keyword in ID or description.

```
inspect> list_case_sets --filter gmm_v2
case_set_id              description                  status     scan_space  num_runs
-----------------------  ---------------------------  ---------  ----------  --------
gmm_v2_initial_tuning    Initial GMM_V2 Tuning        COMPLETED  48000       1
gmm_v2_tuning_1          GMMv2 Cover All Tuned Blocks COMPLETED  873600      1
```

#### List runs

```
list_runs [--case_set_id ID] [--filter KEYWORD]
```

Shows `run_id`, `case_set_id`, description, and number of buckets.

#### Count buckets

```
count_buckets [--case_set_id ID] [--run_id ID]
```

Total number of work buckets for a given run.

#### List bucket status

```
list_bucket_status [--case_set_id ID] [--run_id ID]
```

Shows how many buckets are `COMPLETED` vs pending — useful for monitoring progress.

```
inspect|cs=testing_tuning_infra_11|run=001> list_bucket_status
  COMPLETED: 4
```

#### Query run status

```
query_run_status [--case_set_id ID] [--run_id ID]
```

Shows timing info: start time, last completed time, and total wall time.

```
inspect|cs=testing_tuning_infra_11|run=001> query_run_status
  case_set_id: testing_tuning_infra_11
  run_id: 001
  start_time: 2026-04-21 06:22:06.582395+00:00
  last_completed_time: 2026-04-21 06:22:49.187110+00:00
  total_completed_time_us: 9706475
  total_completed_time_s: 9.71
```

#### Query minimum latency results

```
query_min_latency [--case_set_id ID] [--run_id ID] [--show FIELD ...] [--show-baseline]
```

For each unique `TuningKey`, shows the best measured latency and the corresponding `TunableParam` configuration. If repeatable --show option is specified, only the FIELDs are shown. Without --show option, all the fields in TuningKey and TunableParams are shown as a table.

```
inspect|cs=mla_tuning_0|run=4> query_min_latency --show max_num_tokens --show actual_num_q_heads --show actual_lkv_dim  --show actual_r_dim  --show decode_batch_size  --show num_kv_pages_per_block --show latency_us
max_num_tokens  actual_num_q_heads  actual_lkv_dim  actual_r_dim  decode_batch_size  num_kv_pages_per_block  latency_us
--------------  ------------------  --------------  ------------  -----------------  ----------------------  ----------
128             128                 512             64            16                 1                       2059  
...
64              128                 512             64            16                 1                       2041  
8               128                 512             64            8                  2                       2035
```

**`--show-baseline`** adds two extra columns that compare the best-tuned result
against the `is_baseline=True` case for each `TuningKey`:

| Extra column | Description |
|---|---|
| `baseline_latency` | Latency of the baseline case (us). `N/A` if no baseline was measured for that key. |
| `latency_improvement%` | `(baseline − best) / baseline × 100`. Positive = faster after tuning. `N/A` if no baseline. |

```
inspect|cs=mla_tuning_0|run=4> query_min_latency --show max_num_tokens --show latency_us --show-baseline
max_num_tokens  latency_us  baseline_latency  latency_improvement%
--------------  ----------  ----------------  --------------------
4               2035        2500              +18.6%
8               2041        2490              +18.0%
128             2059        N/A               N/A
```

#### Query case latency

```
query_case_latency  Query latency for tuning cases with optional field filters
                        (--case_set_id ID --run_id ID [--filter_key FIELD=VALUE ...] [--show FIELD ...] [--show_all] [--show-baseline])
```

FIELD can be any key in tuning_key or tunable_params. --show option behaves the same as above. --show_all includes all cases, even ones where tuning failed.

```
inspect|cs=mla_tuning_0|run=4> query_case_latency --filter_key max_num_tokens=4 --show max_num_tokens --show actual_num_q_heads --show actual_lkv_dim  --show actual_r_dim  --show decode_batch_size  --show num_kv_pages_per_block --show latency_us --show_all
max_num_tokens  actual_num_q_heads  actual_lkv_dim  actual_r_dim  decode_batch_size  num_kv_pages_per_block  latency_us
--------------  ------------------  --------------  ------------  -----------------  ----------------------  ----------
4               128                 512             64            16                 1                       2078  
4               128                 512             64            8                  1                       2111  
...
4               128                 512             64            32                 1                       FAILURE  
```

**`--show-baseline`** adds the same two extra columns as in `query_min_latency`.
For non-`SUCCESS` rows both columns show `N/A`.

```
inspect|cs=mla_tuning_0|run=4> query_case_latency --filter_key max_num_tokens=4 \
    --show latency_us --show decode_batch_size --show-baseline
decode_batch_size  latency_us  baseline_latency  latency_improvement%
-----------------  ----------  ----------------  --------------------
16                 2078        2500              +16.9%
8                  2111        2500              +15.6%
```

The baseline for a `TuningKey` is the `TuningCase` whose `is_baseline` field was set
to `True` when it was created (typically the currently-deployed default parameters).
If no such case exists in the run, both columns show `N/A`.

#### Other

```
help         Print command reference
exit / quit  Exit the CLI
```

---

## 5. Bayesian Optimization

Instead of sweeping every pre-generated case, the framework supports
**Bayesian optimization** via [optuna](https://optuna.org/) (TPE sampler) to
converge to near-optimal configurations in far fewer trials.

### How it works

1. `generate_cases` enumerates all combinations from `get_search_space` and stores
   them in the database as usual.  In Bayesian mode each bucket covers **exactly one
   `TuningKey`** so optuna can optimize per-key independently.
2. `measure_latency` creates an optuna `Study(direction="minimize")` and runs up to
   `n_bayesian_trials` trials.  Each trial calls
   `trial.suggest_categorical(param, values)` for every field in `get_search_space`,
   guaranteeing the suggested combination maps to a pre-stored `case_id`.
3. Every trial result is written back to the database with the correct `case_id`, so
   `query_min_latency` and all other inspection tools work identically for both modes.
4. **OOM early-stop is preserved**: if a smaller `TunableParams` combination already
   produced `FAILED_OOM`, any larger combination suggested by optuna is immediately
   pruned and logged as `SKIPPED`.

### Enabling it

**Per-tuner default** — set in `TunerConfig` inside the tuner's `__init__`:

```python
self.tuner_config = TunerConfig(
    ...
    support_bayesian_optimization=True,
    n_bayesian_trials=50,   # optuna trials per TuningKey bucket
)
```

**Runtime override** — the `--bayesian_optimization` flag overrides the tuner's
default without modifying source code (see Section 2 for full examples):

```bash
# Force Bayesian optimization
python -m tools.kernel.tuner.v1.kernel_tuner_runner ... --bayesian_optimization=True

# Force full sweep
python -m tools.kernel.tuner.v1.kernel_tuner_runner ... --bayesian_optimization=False
```

### Choosing `n_bayesian_trials`

| Search-space size | Suggested `n_bayesian_trials` |
|---|---|
| ≤ 25 combinations | 15 – 20 (≈ 60 – 80 % coverage) |
| 25 – 100 combinations | 30 – 50 |
| > 100 combinations | 50 – 100 |

Optuna's TPE sampler needs roughly 10 – 20 warm-up (random) trials before it starts
exploiting the model, so `n_bayesian_trials` should be at least 20 for effective
optimization.

---

## 6. Future Work

### Warm-Starting from Previous Runs

The runner could start from the best known `TunableParams` per `TuningKey` from previous case sets or manual selected `TunableParams` and seed the new search space around those values, skipping parameter combinations that were historically poor performers.

### Context- and Benchmark-Aware Tuning

Kernel performance is not purely a function of tensor shapes and tile parameters — the **numeric statistics of the input data** can materially affect execution time on TPU. For example, in LLM serving:

- During **prefill**, attention score matrices tend to be dense and spread across a wide value range.
- During **decode**, most KV cache pages are cold/sparse and the active query tensor is a single token with very different sparsity patterns.

The same `TunableParams` that is optimal for prefill inputs may not be optimal for decode inputs, and vice versa.

To support this, the framework could be extended with:

- A **benchmark context** concept attached to each `TuningCase` (e.g. `context="prefill"` vs `context="decode"`), allowing the same `TuningKey` to be tuned independently under different input distributions.
- A `generate_inputs_for_context(tuning_key, context)` method on `KernelTunerBase` that produces realistic JAX arrays whose numeric statistics match the target workload (e.g. drawn from captured activation distributions or synthetic approximations).
- Context-aware result storage and querying in the inspector CLI, so `query_min_latency` can be filtered by context to return the best `TunableParams` per `(TuningKey, context)` pair.
- At serving time, the kernel dispatch layer would select the tuned parameters based on the current inference phase (prefill vs decode), rather than using a single static lookup.

---

## 7. End-to-End Autotuning Pipeline

The v1 tuner framework is integrated into a fully automated Buildkite pipeline that continuously optimizes kernel parameters based on real-world workload traces. The pipeline operates in 5 stages and automatically creates Pull Requests with improved configurations.

### Pipeline Architecture

The E2E pipeline is defined in `.buildkite/pipeline_kernel_autotune_template.yml` and is driven by environment variables. The 5 stages are:

1. **Pre-Autotuning Benchmark (Cases Collection):**
   Runs a standard benchmark run on the `main` branch. During this run, the kernels intercept actual input shapes, serializing them into Spanner as `TuningCase` records. This guarantees we only tune for shapes actually seen in production.
2. **Kernel Tuning Execution:**
   Triggers multiple parallel tuning jobs on Cloud TPUs. Each job claims a "bucket" of generated tuning cases and measures latency for the tunable parameters defined in the kernel's search space. Results are written back to Spanner.
3. **Patch Kernel Tuning Result:**
   Fetches the absolute best-performing configuration for each shape from Spanner. It then uses shell-level AST-like monkey-patching to safely overwrite the `tuned_params_mapping` dictionary in the target python files (e.g. `tpu_inference/kernels/mla/v2/tuned_params.py`), commits the change, and pushes it to a temporary evaluation branch.
4. **Post-Autotuning Benchmark (Evaluation):**
   Re-runs the exact same benchmark suite as Stage 1, but this time executing against the newly patched evaluation branch containing the tuned kernel parameters.
5. **Evaluate and Create PR:**
   Compares the benchmark metrics from Stage 1 (baseline) and Stage 4 (tuned). If there are performance improvements and no significant regressions (threshold = 0.4%), it automatically generates a Pull Request with an HTML summary report detailing the latency improvements.

### Configuration

To include a new kernel in the autotuning pipeline, you must register its path in the shared configuration file `tools/kernel/tuner/v1/autotune/kernel_autotune_config.py`.

```python
kernel_autotune_mapping = {
    'my_kernel_tuner': '/workspace/tpu_inference/kernels/my_kernel/tuned_params.py',
}
```

**Requirements for Target Files:**
The pipeline uses strict validation before patching any Python files. Your target `tuned_params.py` file must contain:
- A `def get_tuned_params(...)` function.
- A `tuned_params_mapping = { ... }` module-level dictionary.
- No existing function named `_get_tuned_params`.
