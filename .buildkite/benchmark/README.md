# vLLM TPU Benchmark Framework

This feature provides an automated, scalable framework for benchmarking vLLM on Google Cloud TPUs. It supports both automated execution via Buildkite CI and manual execution on local TPU VMs.

## 1. Execution Workflows

### Buildkite CI Workflow
The CI pipeline dynamically constructs its execution matrix based on the JSON configuration files. The flow is as follows:

1. **Bootstrap (`benchmark_bootstrap.sh`)**: Acts as the entry point. It scans the test cases directory and invokes the pipeline generator.
2. **Pipeline Generation (`generate_bk_pipeline.py`)**: Parses the JSON case files and dynamically outputs Buildkite YAML definitions. It creates step `group`s for each model/case and assigns the corresponding `ci_queue` (e.g., `tpu_v6e_queue`, `tpu_v7x_2_queue`).
3. **Job Wrapper (`run_job.sh`)**: Executed by the Buildkite agent. It maps the TPU queues to physical device types, sets up Docker volume mounts for artifacts/logs, and launches the benchmark within a Docker container.
4. **Execution (`run_bm.sh`)**: The core runner. It downloads datasets via `gsutil`, boots the vLLM server (`vllm serve`) in the background, waits for readiness, and triggers the client command. For performance runs, it performs a binary search on the request rate to meet `EXPECTED_ETEL`.
5. **Reporting (`report_result.sh`)**: Extracts throughput and latency metrics (P99, Median) from logs, uploads logs to GCS and Buildkite artifacts, and performs an atomic UPSERT of the benchmark results into a Google Cloud Spanner database.

### Local TPU Workflow
Developers can bypass the CI/Docker abstraction and run the benchmark directly on a local TPU VM using the core runner:

```bash
bash run_bm.sh <path_to_case.json> [TARGET_CASE_NAME]
```

*Note: `TARGET_CASE_NAME` is only required if the JSON file uses the multi-case structure.*

## 2. Local Execution Prerequisites & Caveats

The `run_bm.sh` script is designed to be environment-aware. When running locally, please ensure the following preconditions and behaviors are noted:

* **Prerequisites**: You must have a pre-configured Python environment with `vllm` and the `tpu-inference` packages already installed.
* **Dataset Handling (`GCS_BUCKET`)**: If `GCS_BUCKET` is *not* set, the script will skip downloading datasets from Google Cloud Storage. It will fallback to using local dataset files. Ensure your dataset exists locally unless the specific benchmark (`lm_eval`, etc.) does not require one.
* **Log Uploads**: If `GCS_BUCKET` is *not* set, the script will silently skip uploading `bm_log.txt` and `vllm_log.txt` to GCS. Logs will remain in the local folder.
* **Database Reporting (`GCP_DATABASE_ID`, etc.)**: If database-related variables (e.g., `GCP_PROJECT_ID`, `GCP_INSTANCE_ID`, `GCP_DATABASE_ID`) are *not* set, `report_result.sh` will skip inserting the benchmark results into the Spanner DB. Results will only be printed to stdout and saved in the local `.result` file.

***Note: The GCP-related environment variables used below point to the Staging DB and Bucket. When officially launched in the future, the settings will be changed to point to the Production DB and Bucket.***

```json
"GCP_PROJECT_ID": "cloud-tpu-inference-test",
"GCS_BUCKET": "vllm-cb-storage2",
"GCP_INSTANCE_ID": "vllm-bm-inst",
"GCP_DATABASE_ID": "vllm-bm-bk-runs"
```

***There are two more modifications that need to be made here:***
1. [`REMOTE_LOG_ROOT`](scripts/report_result.sh#L49) needs to be changed to use `$GSC_BUCKET` as the bucket for storing logs.
2. The migration file [`vllm_bm_20260410.ddl`](sql/vllm_bm_20260410.ddl) needs to be executed in `vllm-bm-runs` (Production Spanner DB).

## 3. Configuration Guide (JSON Cases)

The feature is driven by JSON configuration files. A case file can define a single benchmark or multiple benchmarks using the `benchmark_cases` array.

### Configuration Structure & Parameter Definitions

* `global_env` (or `env` in single-case): Environment variables applied globally.
  * `GCP_PROJECT_ID`, `GCP_INSTANCE_ID`, `GCP_DATABASE_ID`, `GCS_BUCKET`: Cloud routing for logs, datasets, and DB.
  * `MODELTAG`: Identifier tag for the model state (e.g., `NEW`, `PROD`).
  * `EXPECTED_ETEL`: The target goal for End-to-End Latency (P99 in ms). The script adjusts the request rate via binary search to stay within this limit.
  * `EXPECTED_THROUGHPUT`: Target throughput. Evaluated in `report_result.sh` to flag performance regressions.
  * `INPUT_LEN`, `OUTPUT_LEN`, `PREFIX_LEN`: Metadata representing sequence lengths. Used primarily for database tagging.
* `ci_queue`: Array of strings defining which Buildkite agent queues should pick up this case.
* `server_command_options`: Controls the vLLM backend.
  * `command_type`: Must be `vllm_serve`.
  * `args`: Key-value pairs translated into CLI flags (e.g., `model`, `seed`, `max-model-len`).
  * `env`: Server-command-specific environment variables.
* `client_command_options`: Controls the workload generator.
  * `command_type`: Typically `vllm_bench_serve`. Can also be `lm_eval` for accuracy evaluations. When it is `lm_eval`, the dataset must be specified in the `args`, a specific shell script will be executed based on the dataset configuration, and `server_command_options` does not need to be set.
  * `args`: Client-side CLI flags (e.g., `num-prompts`, `request-rate`).
  * `env`: Client-command-specific environment variables.

### Critical Considerations & Advanced Features

#### A. Dynamic Arguments (TPU Auto-Detection)
The `args` within `server_command_options.args` or `client_command_options.args` support dynamic dictionary mapping based on the host's TPU topology.

```json
"tensor-parallel-size": {
  "v6e-1": 1,
  "v6e-8": 8,
  "v7x-2": 2,
  "default": 1
}
```

*Mechanism: `parser_case.py` uses `tpu_info` to detect the local chip type and count. It automatically resolves the correct value before command execution. If a match is not found, it falls back to `default`.*

#### B. Environment Variable Scoping
Variables defined in the root `env` or `global_env` apply to the entire shell session. However, the `env` block nested inside `server_command_options` or `client_command_options` is strictly scoped.
These variables are injected inline via the `env` command right before binary execution (e.g., `env VLLM_USE_V1=1 vllm bench serve ...`), ensuring they do not pollute the global execution context.

#### C. The `dataset-path` Constraint
When supplying a custom dataset file via `dataset-path`:
* **Buildkite CI**: The path *must* be prefixed with `/workspace/tpu_inference/artifacts/dataset/` because the CI mounts the artifacts folder to this specific container path, and the Dataset File will be synced from the Bucket to this folder.
* **Local Execution**: The path should be an absolute path or relative to the directory where `run_bm.sh` is executed.

#### D. Implicit DB Tracking Variables
While variables like `EXPECTED_ETEL`, `INPUT_LEN`, `OUTPUT_LEN`, and `PREFIX_LEN` might not be directly passed to the `vllm` CLI, they are rigorously parsed and injected into the GCP Spanner `RunRecord` table by `report_result.sh` to track historical performance variations. Please ensure that the values of these variables remain consistent with the corresponding parameter settings in the args of the Case JSON file.
