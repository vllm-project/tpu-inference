# Kernel Tuner v1

A framework for measuring and tuning the latency of TPU kernels. Results are stored either locally (JSON files) or in Google Cloud Spanner. Supports both exhaustive grid searching (full sweep) and adaptive Bayesian Optimization via Optuna to find optimal tile sizes and parameters significantly faster.

Uses a robust **3-process isolation model** ([Runner](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/kernel_tuner_runner.py) -> [Worker](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/kernel_tuner_worker.py) -> [Executor](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/kernel_tuner_executor.py)) to isolate TPU kernel runs and transparently recover from TPU device poisoning (e.g. OOMs or hardware hangs) without losing optimizer or search state.

---

## 1. Architecture & Multi-Process Execution Model

To handle TPU execution instabilities (such as OOM errors or JAX runtime poisoning), the tuning framework architecture decouples job scheduling, optimization search, and TPU kernel execution across three distinct process layers.

```
+-------------------------------------------------------------------------+
|                              Runner Process                             |
|                        (kernel_tuner_runner.py)                         |
|  - Case Generation & Grouping (by TuningKey)                            |
|  - Work Bucketing & Buildkite Pipeline Generation                       |
|  - Bucket Lifecycle & Retry Management (BucketStatus)                   |
+-------------------------------------------------------------------------+
                                     |
                          Spawns per bucket (subprocess)
                                     v
+-------------------------------------------------------------------------+
|                              Worker Process                             |
|                        (kernel_tuner_worker.py)                         |
|  - Database Manager (StorageManager: Local / Cloud Spanner)             |
|  - Search Optimizer (SweepOptimizer / BayesianOptimizer via Optuna)     |
|  - Lightweight Tuner Config & Search Space                              |
+-------------------------------------------------------------------------+
                                     |
                     Manages lifecycle & IPC via stdin/stdout
                                     v
+-------------------------------------------------------------------------+
|                         ExecutorProcessManager                          |
|                     (executor_process_manager.py)                       |
+-------------------------------------------------------------------------+
                                     |
                         Persistent Subprocess (JSON IPC)
                                     v
+-------------------------------------------------------------------------+
|                             Executor Process                            |
|                       (kernel_tuner_executor.py)                        |
|  - Full JAX & TPU Runtime Environment                                   |
|  - In-Memory Input Tensor Cache (generate_inputs)                       |
|  - Kernel Latency Measurement Execution (run)                           |
+-------------------------------------------------------------------------+
```

### Process Roles

1. **Runner Process** ([kernel_tuner_runner.py](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/kernel_tuner_runner.py)):
   - Serves as the top-level orchestrator.
   - Generates and filters tuning cases, grouping cases by `TuningKey` prior to bucketing to maximize input caching reuse across consecutive kernel runs.
   - Divides case sets into work buckets and generates dynamic Buildkite fan-out pipelines.
   - Tracks bucket lifecycle states (`BucketStatus`) and manages bucket retries and job yields.

2. **Worker Process** ([kernel_tuner_worker.py](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/kernel_tuner_worker.py)):
   - Spawned as a separate process by the Runner for each work bucket.
   - Houses the database connection ([storage_manager.py](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/storage_management/storage_manager.py)) and the search optimizer ([SweepOptimizer](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/optimizer/sweep_optimizer.py) or [BayesianOptimizer](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/optimizer/bayesian_optimizer.py)).
   - Creates a lightweight tuner instance to access configuration and search space definitions without initializing full JAX/TPU device context in the worker process.

3. **Executor Process** ([kernel_tuner_executor.py](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/kernel_tuner_executor.py)):
   - Persistent subprocess spawned and managed by [executor_process_manager.py](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/executor_process_manager.py).
   - Initializes the full JAX/TPU environment and retains the input tensor cache (`generate_inputs()`) across multiple tuning cases to minimize allocation overhead.
   - Executes `KernelTunerBase.run()` measurements.
   - Communicates with the Worker via line-buffered JSON requests over stdin/stdout (using `__JSON__` prefix filters to isolate logger output).

### Fault Isolation & TPU Recovery

TPU kernel execution can occasionally trigger OOMs, unrecoverable C++ exceptions, or hardware state poisoning that corrupts the JAX runtime:

- **Isolated Execution**: Because `run()` executes inside the isolated Executor process, a TPU crash or process exit only kills the Executor.
- **Transparent Worker Recovery**: When the Executor process dies or times out, `ExecutorProcessManager` intercepts the pipe error, logs the stack trace, cleans up process groups, and returns `TuningStatus.UNKNOWN_ERROR`.
- **Automatic Process Restart**: On the next trial request, `ExecutorProcessManager` transparently spawns a fresh replacement Executor process. The Worker process, Optuna search state, and database connections remain intact.

### Job Lifecycle Management (`BucketStatus`)

Work buckets follow an explicit lifecycle tracked in database storage:

- `NOT_STARTED`: Bucket created but not yet claimed by a worker.
- `IN_PROGRESS`: Currently being executed by a worker process.
- `COMPLETED`: All cases in the bucket evaluated successfully.
- `FAILED`: Bucket execution failed after exhausting retry attempts.
- `CANCELLED`: Bucket cancelled by runner or user intervention.
- `YIELDED`: Bucket execution voluntarily yielded (e.g. during Buildkite job timeouts or worker preemptions) to be resumed by a subsequent job run.

### Shared Flags & Centralized Factories

- **Centralized Flags** ([kernel_tuner_flags.py](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/kernel_tuner_flags.py)): Defines common ABSL flags for GCP, Spanner, case set IDs, run IDs, TPU configurations, and optimization settings to eliminate flag duplicate definition errors across imported modules.
- **Tuner & Storage Factories** ([kernel_tuner_factory.py](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/kernel_tuner_factory.py)): Provides centralized factory methods (`create_kernel_tuner`, `create_storage_manager`, `run_config_to_json`) for creating tuners and storage managers in lightweight or full execution modes.
- **Subprocess Environment Helper** ([utils.py](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/utils.py)): Exposes `get_subprocess_env()` to propagate TPU/JAX flags and environment variables to worker and executor subprocesses.

---

## 2. Implementing a Custom Kernel Tuner

To add a new kernel to the tuning framework, create a new file (e.g. `my_kernel_tuner.py`) and subclass `KernelTunerBase` ([common/kernel_tuner_base.py](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/common/kernel_tuner_base.py)). You can add kernel-specific flags in this file. To avoid name conflicts, flags should be named in the format `{your_kernel_name}_{flag_name}` in your tuner module and append `KERNEL_TUNING_` as a prefix when invoked through the Buildkite UI.

### Step 1 — Define `TuningKey` and `TunableParams`

- `TuningKey` describes the fixed properties of a kernel invocation (shapes, data types, etc.).
- `TunableParams` describes the parameters you want to search over (tile sizes, block sizes, etc.). Must implement `__ge__(self, other)` and `__le__(self, other)` methods as they are used for OOM early termination.

Both must be `@dataclass(frozen=True)` (frozen, not ordered) so they are hashable and the framework can serialize and deserialize them properly.

```python
import dataclasses

@dataclasses.dataclass(frozen=True)
class MyTuningKey:
    batch_size: int
    seq_len: int

@dataclasses.dataclass(frozen=True)
class MyTunableParams:
    tile_m: int
    tile_n: int

    def __ge__(self, other) -> bool:
        return self.tile_m >= other.tile_m and self.tile_n >= other.tile_n

    def __le__(self, other) -> bool:
        return self.tile_m <= other.tile_m and self.tile_n <= other.tile_n
```

### Step 2 — Subclass `KernelTunerBase`

Subclasses must instantiate a `TunerConfig` and pass both `tuner_config` and `run_config` to `super().__init__()`:

```python
import itertools
import time

from tools.kernel.tuner.v1.common.kernel_tuner_base import KernelTunerBase
from tools.kernel.tuner.v1.common.tuner_datatypes import (
    RunConfig, TunerConfig, TuningCase, TuningStatus)


class MyKernelTuner(KernelTunerBase):

    def __init__(self, run_config: RunConfig):
        self.tuner_config = TunerConfig(
            tuning_key_class=MyTuningKey,
            tunable_params_class=MyTunableParams,
            kernel_tuner_name="my_kernel_tuner",  # must match KERNEL_TUNER_REGISTRY key
            support_bayesian_optimization=True,   # enable Bayesian Optimization support
            n_bayesian_trials=100,                 # max BO trials per tuning key bucket (default 100)
            bayesian_early_stopping_patience=10,   # stop trial if no improvement for 10 trials
            bayesian_early_stopping_min_delta_ratio=0.05, # min 5% relative improvement
            min_cases_for_bayesian=200,            # fallback to full sweep if search space < 200
        )
        self.run_config = run_config
        super().__init__(
            tuner_config=self.tuner_config,
            run_config=self.run_config,
        )
```

### Step 3 — Implement required methods

#### `get_search_space(tuning_key: MyTuningKey) -> dict[str, list]`

Returns a dictionary mapping parameter names to lists of candidate parameter values for a given `TuningKey`. This method is **required** when using Bayesian Optimization (`support_bayesian_optimization=True`) so `BayesianOptimizer` can sample trials, and it also simplifies `generate_cases()`.

```python
    def get_search_space(self, tuning_key: MyTuningKey) -> dict[str, list]:
        return {
            'tile_m': [16, 32, 64],
            'tile_n': [16, 32, 64],
        }
```

#### `generate_cases() -> list[TuningCase]`

Returns the complete list of `TuningCase` objects. Prune invalid tuning cases (e.g., configurations exceeding memory limits or failing alignment requirements) at this stage to minimize search overhead.
This method is called once to populate the initial case set; results are persisted in storage so re-runs with the same `case_set_id` skip regeneration.

```python
    def generate_cases(self) -> list[TuningCase]:
        cases = []
        for bs, sl in itertools.product([1, 2, 4], [128, 256]):
            tuning_key = MyTuningKey(batch_size=bs, seq_len=sl)
            search_space = self.get_search_space(tuning_key)
            for tm, tn in itertools.product(search_space['tile_m'], search_space['tile_n']):
                cases.append(TuningCase(
                    tuning_key,
                    MyTunableParams(tile_m=tm, tile_n=tn),
                ))
        return cases
```

#### `generate_inputs(tuning_key: MyTuningKey) -> dict`

Prepares the kernel inputs for a given `TuningKey`. In the Executor process, the result is cached in memory so inputs are only regenerated when the tuning key changes.

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
Return `TuningStatus.FAILED_OOM` for OOM errors and `TuningStatus.UNKNOWN_ERROR` for other failures so the framework records them without crashing the worker.

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

### Step 4 — Run it

Locally:

```bash
python -m tools.kernel.tuner.v1.kernel_tuner_runner \
  --kernel_tuner_name=my_kernel_tuner \
  --run_locally=True \
  --case_set_id=my_first_run \
  --run_id=001 \
  --case_set_desc="My kernel first tuning run" \
  --use_bayesian_optimization=True
```

On Buildkite, set `KERNEL_TUNING_KERNEL_TUNER_NAME=my_kernel_tuner` and optionally `KERNEL_TUNING_USE_BAYESIAN_OPTIMIZATION=True` in the build environment variables.

---

## 3. Running Locally

### Available built-in tuners

The framework currently includes these built-in tuners:

- `example_kernel_tuner` ([example_kernel_tuner.py](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/example_kernel_tuner.py))
- `rpa_v3_kernel_tuner` ([rpa_v3_kernel_tuner.py](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/rpa_v3_kernel_tuner.py))
- `mla_kernel_tuner` ([mla_kernel_tuner.py](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/mla_kernel_tuner.py))
- `batched_rpa_kernel_tuner` ([batched_rpa_kernel_tuner.py](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/batched_rpa_kernel_tuner.py)) — supports both prefill and decode phase tuning
- `gmm_v2_kernel_tuner` ([gmm_v2_kernel_tuner.py](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/gmm_v2_kernel_tuner.py))
- `flash_attention_kernel_tuner` ([flash_attention_kernel_tuner.py](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/flash_attention_kernel_tuner.py))

### GMM v2 tuner

The `gmm_v2_kernel_tuner` targets the Megablox GMM v2 kernel. It uses a `TuningKey` that captures the fixed problem description from the GMM worker contract (`m`, `k`, `n`, `tg`, `cg`, `ldt`, `rdt`, `q_block`, `fuse_act`) and a `TunableParams` that captures the tile sizes (`tm`, `tk`, `tn`).

```python
@dataclasses.dataclass(frozen=True)
class TuningKey:
    m: int
    k: int
    n: int
    tg: int
    cg: int
    ldt: str
    rdt: str
    q_block: int
    fuse_act: str | None = None

@dataclasses.dataclass(frozen=True)
class TunableParams:
    tm: int
    tk: int
    tn: int
```

Example local run:

```bash
python -m tools.kernel.tuner.v1.kernel_tuner_runner \
  --run_locally \
  --kernel_tuner_name=gmm_v2_kernel_tuner \
  --case_set_id=gmm_v2_local_smoke \
  --run_id=0 \
  --case_set_desc='gmm_v2_local_smoke' \
  --tpu_version=tpu7x \
  --tpu_cores=2
```

Install dependencies first:

```bash
pip install -r tools/kernel/tuner/v1/storage_management/requirements.txt 
```

We recommend running the tuner with local storage first to verify that your custom kernel tuner is set up correctly:

```bash
python -m tools.kernel.tuner.v1.kernel_tuner_runner \
  --kernel_tuner_name=example_kernel_tuner \
  --run_locally=True \
  --case_set_id=my_local_run \
  --run_id=001 \
  --case_set_desc="My local tuning run" \
  --use_bayesian_optimization=True
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--kernel_tuner_name` | `example_kernel_tuner` | Which tuner to run (must be registered in `KERNEL_TUNER_REGISTRY`). |
| `--run_locally` | `False` | Use local JSON storage instead of Cloud Spanner. |
| `--use_bayesian_optimization` | `False` | Enable Optuna Bayesian Optimization instead of full grid sweep. |
| `--min_cases_for_bayesian` | `200` | Minimum cases in search space required to use Bayesian Optimization (fallbacks to sweep if smaller). |
| `--case_set_id` | `""` | Identifier for this set of tuning cases (required). |
| `--run_id` | `""` | Run ID within the case set (required). |
| `--case_set_desc` | `""` | Human-readable description. |
| `--tpu_version` | `""` | TPU generation (`tpu6e` or `tpu7x`). |
| `--tpu_cores` | `0` | TPU core count (e.g. 1, 2, 8, 16). |

Local results are written to JSON files in the directory `/tmp/kernel_tuner_runner_{case_set_desc}`.

---

## 4. Running on TPU VMs via Buildkite

The pipeline is defined in `.buildkite/pipeline_kernel_tuning.yml` and bootstrapped by `.buildkite/scripts/bootstrap_kernel_tuning.sh`.

### Pipeline overview

1. **Bootstrap** (`bootstrap_kernel_tuning.sh`) — uploads the static `pipeline_kernel_tuning.yml`.
2. **Build** — builds and pushes the `vllm-tpu` Docker image.
3. **Generate cases + upload dynamic pipeline** — runs `kernel_tuner_runner` inside Docker with `--generate_buildkite_pipeline=True`. Cases are grouped by `TuningKey` to maximize input caching in workers. The generated YAML is written to `/tmp/kernel_tuning/generated_pipeline.yml` (shared with the host via a volume mount) and uploaded to Buildkite with `buildkite-agent pipeline upload`.
4. **Tuning jobs** — the dynamically-uploaded pipeline fans out individual worker tuning jobs across TPU workers using the multi-process execution framework.

### Triggering a build

**Option A (Recommended) — Buildkite UI:**

1. Go to the pipeline page.
2. Click **New Build**.
3. Set your branch.
4. Expand **Environment Variables** and set the required variables.
5. Click **Create Build**.

Make sure to specify `KERNEL_TUNING_TPU_VERSION` and `KERNEL_TUNING_TPU_CORES` so the runner resolves the correct TPU queue.

**Option B — Buildkite REST API:**

```bash
curl -s -X POST \
  -H "Authorization: Bearer $BUILDKITE_API_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.buildkite.com/v2/organizations/vllm/pipelines/tpu-inference-kernel-tuning/builds" \
  -d '{
    "commit": "'"$(git rev-parse HEAD)"'",
    "branch": "'"$(git rev-parse --abbrev-ref HEAD)"'",
    "message": "kernel tuning run",
    "env": {
      "KERNEL_TUNING_KERNEL_TUNER_NAME":          "batched_rpa_kernel_tuner",
      "KERNEL_TUNING_CASE_SET_ID":                 "my_case_set_001",
      "KERNEL_TUNING_RUN_ID":                      "001",
      "KERNEL_TUNING_CASE_SET_DESC":               "My tuning run description",
      "KERNEL_TUNING_TPU_VERSION":                 "tpu7x",
      "KERNEL_TUNING_TPU_CORES":                   "2",
      "KERNEL_TUNING_USE_BAYESIAN_OPTIMIZATION":   "True",
      "KERNEL_TUNING_MIN_CASES_FOR_BAYESIAN":      "200"
    }
  }'
```

### Required environment variables

| Variable | Example | Description |
|---|---|---|
| `KERNEL_TUNING_KERNEL_TUNER_NAME` | `rpa_v3_kernel_tuner` | Name of the kernel tuner to run. Must match a key in `KERNEL_TUNER_REGISTRY`. |
| `KERNEL_TUNING_CASE_SET_ID` | `gmm_v2_tuning_001` | Unique identifier for this case set. Used as the primary key in Spanner. |
| `KERNEL_TUNING_RUN_ID` | `001` | Run ID within the case set. Increment for re-runs of the same case set. |
| `KERNEL_TUNING_CASE_SET_DESC` | `"Description of this case set"` | Human-readable description stored alongside results. |
| `KERNEL_TUNING_TPU_VERSION` | `tpu6e` or `tpu7x` | TPU generation. Controls agent queue selection. |
| `KERNEL_TUNING_TPU_CORES` | `1`, `8`, `16` | Number of TPU cores for tuning jobs. |
| `KERNEL_TUNING_USE_BAYESIAN_OPTIMIZATION` | `True` or `False` | Set to `True` to use Bayesian Optimization instead of full grid sweep. |
| `KERNEL_TUNING_N_BAYESIAN_TRIALS` | `100` | Number of Bayesian trials to sample per tuning key bucket (overrides tuner default). |
| `KERNEL_TUNING_MIN_CASES_FOR_BAYESIAN` | `200` | Minimum cases in search space required to use Bayesian Optimization (overrides tuner default). |

---

## 5. Optimization Strategies & Bayesian Optimization

The framework decouples search strategies from kernel execution via the `TuningOptimizer` abstraction ([optimizer/base_optimizer.py](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/optimizer/base_optimizer.py)):

1. **`SweepOptimizer`** ([optimizer/sweep_optimizer.py](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/optimizer/sweep_optimizer.py)): Exhaustively iterates through all tuning cases in the search space.
2. **`BayesianOptimizer`** ([optimizer/bayesian_optimizer.py](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/optimizer/bayesian_optimizer.py)): Uses Optuna with Tree-structured Parzen Estimator (TPE) sampling to intelligently sample tile configurations.

### Multi-Process Execution Integration

Optimizers run inside the **Worker Process** and communicate with the persistent **Executor Process** via `ExecutorProcessManager`:

```python
# Executed in Worker Process:
status, avg_ns, total_ns = self.executor_mgr.execute_run(
    tuning_key=case.tuning_key,
    tunable_params=case.tunable_params,
    iters=iters,
)
```

If an evaluated configuration triggers a TPU OOM or hardware hang, the Executor crashes, `ExecutorProcessManager` recovers transparently by restarting the Executor process, and the Optimizer receives `TuningStatus.UNKNOWN_ERROR` to record the failure in Optuna without losing search state.

### Key Capabilities of Bayesian Optimization

- **TPE Sampler with Integer Remapping**: Maps discrete parameter choices to continuous indices, allowing Optuna to learn parameter trends and converge rapidly.
- **Relative Early Stopping**: Automatically stops trial sampling per tuning key if latency does not improve by at least `bayesian_early_stopping_min_delta_ratio` over `bayesian_early_stopping_patience` consecutive trials.
- **Smart Fallback**: Automatically reverts to full sweep (`SweepOptimizer`) if `get_search_space()` returns an empty dictionary, if total search space cases for a key is less than `min_cases_for_bayesian`, or if `support_bayesian_optimization` is disabled in `TunerConfig`.

---

## 6. Inspecting Results

Use the interactive CLI ([inspect_result_cli.py](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/inspect_result_cli.py)):

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

Once connected, the prompt displays your active context (e.g. `inspect|cs=my_case_set|run=001>`).

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

Shows how many buckets are in each `BucketStatus` state (`COMPLETED`, `IN_PROGRESS`, `FAILED`, `YIELDED`, `NOT_STARTED`) — useful for monitoring progress and worker yields.

```
inspect|cs=testing_tuning_infra_11|run=001> list_bucket_status
  COMPLETED: 4
  IN_PROGRESS: 0
  YIELDED: 0
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
query_min_latency [--case_set_id ID] [--run_id ID] [--show FIELD ...]
```

For each unique `TuningKey`, displays the lowest measured latency and the corresponding `TunableParams` configuration. If repeated `--show` options are specified, only those fields are displayed.

```
inspect|cs=mla_tuning_0|run=4> query_min_latency --show max_num_tokens --show actual_num_q_heads --show actual_lkv_dim  --show actual_r_dim  --show decode_batch_size  --show num_kv_pages_per_block --show latency_us
max_num_tokens  actual_num_q_heads  actual_lkv_dim  actual_r_dim  decode_batch_size  num_kv_pages_per_block  latency_us
--------------  ------------------  --------------  ------------  -----------------  ----------------------  ----------
128             128                 512             64            16                 1                       2059
...
64              128                 512             64            16                 1                       2041
8               128                 512             64            8                  2                       2035
```

#### Query case latency

```
query_case_latency [--case_set_id ID] [--run_id ID] [--filter_key FIELD=VALUE ...] [--show FIELD ...] [--show_all]
```

`FIELD` can be any property in `TuningKey` or `TunableParams`. `--show_all` includes failed and skipped cases.

```
inspect|cs=mla_tuning_0|run=4> query_case_latency --filter_key max_num_tokens=4 --show max_num_tokens --show actual_num_q_heads --show actual_lkv_dim  --show actual_r_dim  --show decode_batch_size  --show num_kv_pages_per_block --show latency_us --show_all
max_num_tokens  actual_num_q_heads  actual_lkv_dim  actual_r_dim  decode_batch_size  num_kv_pages_per_block  latency_us
--------------  ------------------  --------------  ------------  -----------------  ----------------------  ----------
4               128                 512             64            16                 1                       2078
4               128                 512             64            8                  1                       2111
...
4               128                 512             64            32                 1                       FAILURE
```

#### Other

```
help         Print command reference
exit / quit  Exit the CLI
```

---

## 7. End-to-End Autotuning Pipeline

The v1 tuner framework is integrated into a fully automated Buildkite pipeline that continuously optimizes kernel parameters based on real-world workload traces. The pipeline operates in 5 stages and automatically creates Pull Requests with improved configurations.

### Pipeline Architecture

The E2E pipeline is defined in `.buildkite/pipeline_kernel_autotune_template.yml` and is driven by environment variables. The 5 stages are:

1. **Pre-Autotuning Benchmark (Cases Collection):**
   Runs a standard benchmark run on the `main` branch. During this run, the kernels intercept actual input shapes, serializing them into Spanner as `TuningCase` records. This guarantees tuning only targets shapes actually seen in production.
2. **Kernel Tuning Execution:**
   Triggers parallel tuning jobs on Cloud TPUs using the multi-process worker architecture. Each job claims a bucket of generated tuning cases and measures latency for the tunable parameters (via full sweep or Bayesian optimization). Results are saved to Spanner.
3. **Patch Kernel Tuning Result:**
   Fetches the lowest latency configuration for each shape from Spanner. It then updates the `tuned_params_mapping` dictionary in target Python files (e.g. `tpu_inference/kernels/mla/v2/tuned_params.py`), commits the change, and pushes it to an evaluation branch.
4. **Post-Autotuning Benchmark (Evaluation):**
   Re-runs the benchmark suite against the newly patched evaluation branch containing the tuned kernel parameters.
5. **Evaluate and Create PR:**
   Compares baseline and tuned metrics. If performance improves without significant regressions (threshold = 0.4%), it automatically opens a Pull Request with an HTML report summarizing latency gains.

### Configuration

Register new kernels in [autotune/kernel_autotune_config.py](https://github.com/vllm-project/tpu-inference/tree/main/tools/kernel/tuner/v1/autotune/kernel_autotune_config.py):

```python
kernel_autotune_mapping = {
    'my_kernel_tuner': '/workspace/tpu_inference/kernels/my_kernel/tuned_params.py',
}
```

**Requirements for Target Files:**
Target `tuned_params.py` files must contain:
- A `def get_tuned_params(...)` function.
- A `tuned_params_mapping = { ... }` module-level dictionary.
- No existing function named `_get_tuned_params`.

---

## 8. Future Work

### Asynchronous & Parallel Optimization Pipelining

Support running multiple workers in parallel on multi-core TPU VMs when a kernel does not require all TPU cores simultaneously.

### Warm-Starting from Previous Runs

The runner can seed new search spaces around historically top-performing `TunableParams` for given `TuningKey` shapes from previous tuning runs, bypassing configurations that consistently performed poorly.

### Context- and Benchmark-Aware Tuning

Kernel latency varies depending on numerical properties and operational context (e.g., **prefill** vs **decode** phases in LLM serving). To support phase-aware autotuning:
- Attach a benchmark context (e.g. `context="prefill"` vs `context="decode"`) to `TuningCase`.
- Implement `generate_inputs_for_context(tuning_key, context)` on `KernelTunerBase` to generate context-specific input tensors.
