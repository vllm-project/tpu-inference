# Kernel Tuner v1

A framework for measuring and tuning the latency of TPU kernels. Results are stored either locally (JSON files) or in Google Cloud Spanner.

---

## 1. Implementing a Custom Kernel Tuner

To add a new kernel to the tuning framework, create a new file (e.g. `my_kernel_tuner.py`) and subclass `KernelTunerBase`.

### Step 1 — Define `TuningKey` and `TunableParams`

`TuningKey` describes the fixed properties of a kernel invocation (shapes, types, etc.).  
`TunableParams` describes the parameters you want to search over (tile sizes, etc.).  
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
    KernelTunerBase, TuningCase, TuningStatus)


class MyKernelTuner(KernelTunerBase):

    def __init__(self, storage_manager):
        super().__init__(
            tuning_key_class=MyTuningKey,
            tunable_params_class=MyTunableParams,
            storage_manager=storage_manager,
            job_bucket_size=50,          # number of cases per distributed worker
            kernel_tuner_name="my_kernel_tuner",  # must match KERNEL_TUNER_REGISTRY key
        )
```

### Step 3 — Implement the three abstract methods

#### `generate_cases() -> list[TuningCase]`

Returns the full Cartesian search space as a flat list of `TuningCase` objects. It's recommend to prune as much as invalid tuning cases, like cases will result in OOO or cases that doesn't satisfy data alignment requirements, at this stage to reduce the searhing cases.
This is called once to populate the case set; results are persisted so re-runs with the same case_set_id will skip this step.

```python
    def generate_cases(self) -> list[TuningCase]:
        cases = []
        for bs, sl, tm, tn in itertools.product(
            [1, 2, 4],    # batch_size values
            [128, 256],   # seq_len values
            [16, 32],     # tile_m values
            [16, 32],     # tile_n values
        ):
            cases.append(TuningCase(
                MyTuningKey(batch_size=bs, seq_len=sl),
                MyTunableParams(tile_m=tm, tile_n=tn),
            ))
        return cases
```

#### `generate_inputs(tuning_key: MyTuningKey) -> dict`

Prepares the kernel inputs for a given `TuningKey`. The base class caches the
result so inputs are only regenerated when the key changes.

```python
    def generate_inputs(self, tuning_key: MyTuningKey) -> dict:
        if self._TUNING_KEY and tuning_key == self._TUNING_KEY:
            return self._KERNEL_INPUTS_CACHE
        self._TUNING_KEY = tuning_key
        self._KERNEL_INPUTS_CACHE = {
            'x': jnp.ones((tuning_key.batch_size, tuning_key.seq_len)),
        }
        return self._KERNEL_INPUTS_CACHE
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

On Buildkite, set `KERNEL_TUNING_KERNEL_NAME=my_kernel_tuner` in the build environment variables (see Section 2).

---

## 2. Running Locally

Install dependencies first:

```bash
pip install absl-py
```

We recomend run the tuner with local storage first to make sure the customized kernel_tuner is setup correctly. Since it's for debug purpose, the `case_set_id` will be auto-generated from the current timestamp if not provided.

```bash
python -m tools.kernel.tuner.v1.kernel_tuner_runner \
  --kernel_tuner_name=example_kernel_tuner \
  --run_locally=True \
  --case_set_id=my_local_run \
  --case_set_desc="My local tuning run"
```

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `--kernel_tuner_name` | `example_kernel_tuner` | Which tuner to run. Available: `example_kernel_tuner` and refer to Section 4 to implement your own tuner. |
| `--run_locally` | `False` | Use local JSON storage instead of Spanner. |
| `--case_set_id` | _(timestamp)_ as str | Identifier for this set of tuning cases. Auto-generated if omitted. Required when run in distributed tuning mode.|
| `--case_set_desc` | `""` | Human-readable description. |
| `--run_id` | `"0"` | Run ID within the case set. |
| `--debug` | `False` | Print results after each case iteration. |

Local results are written to JSON files in the working directory located at /tmp/kernel_tuner_run_{case_set_id}.

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
      "KERNEL_TUNING_KERNEL_NAME":    "rpa_v3_kernel_tuner",
      "KERNEL_TUNING_CASE_SET_ID":    "my_case_set_001",
      "KERNEL_TUNING_RUN_ID":         "001",
      "KERNEL_TUNING_CASE_SET_DESC":  "My tuning run description",
      "KERNEL_TUNING_TPU_VERSION":    "v7"
    }
  }'
```

### Required environment variables

Set these in the Buildkite **New Build → Environment Variables** section:

| Variable | Example | Description |
|---|---|---|
| `KERNEL_TUNING_KERNEL_NAME` | `rpa_v3_kernel_tuner` | Name of the kernel tuner to run. Must match a key in `KERNEL_TUNER_REGISTRY` defined in kernel_tuner_runner.py. |
| `KERNEL_TUNING_CASE_SET_ID` | `gmm_v2_tuning_001` | Unique identifier for this case set. Used as the primary key in Spanner. |
| `KERNEL_TUNING_RUN_ID` | `001` | Run ID within the case set. Increment for re-runs of the same case set. |
| `KERNEL_TUNING_CASE_SET_DESC` | `"Your description about this case set"` | Human-readable description stored alongside results. |
| `KERNEL_TUNING_TPU_VERSION` | `v6` or `v7` | TPU generation. Controls which agent queue and `TPU_VERSION` env var are used. |

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
query_min_latency [--case_set_id ID] [--run_id ID]
```

For each unique `TuningKey`, shows the best measured latency and the corresponding `TunableParam` configuration.

```
inspect|cs=testing_tuning_infra_11|run=001> query_min_latency
  tuning_key={"key1": 1, "key2": 4}  best_latency_us=135516  warmup_us=189835  tunable_params={"param1": 7, "param2": 11}  case_id=1
  tuning_key={"key1": 1, "key2": 5}  best_latency_us=81849   warmup_us=108349  tunable_params={"param1": 7, "param2": 10}  case_id=2
  tuning_key={"key1": 2, "key2": 4}  best_latency_us=79251   warmup_us=180675  tunable_params={"param1": 7, "param2": 10}  case_id=4
  tuning_key={"key1": 2, "key2": 5}  best_latency_us=105251  warmup_us=49820   tunable_params={"param1": 7, "param2": 11}  case_id=7
```

#### Other

```
help         Print command reference
exit / quit  Exit the CLI
```

---

## 5. Future Work

### Online Search Optimizer

The current framework exhaustively sweeps a pre-defined Cartesian search space. A natural next step is an **online search optimizer** that adaptively narrows the search space while jobs are still running.

For example:
- A new `SearchOptimizer` interface could subscribe to completed bucket results from Spanner in real time (or at the end of each round).
- The optimizer could be plugged in as an optional component of `KernelTunerBase`, overriding a default no-op `suggest_next_cases(completed_results) -> list[TuningCase]` method.

This could potentially dramatically reduce the number of cases needed for large search spaces while still converging to near-optimal configurations.

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
