# XLA autotune

One-factor-at-a-time (OFAT) sweep over XLA / libtpu flags, driven from
Buildkite.  For each candidate flag, stand up `vllm serve` on a TPU pod,
benchmark it with `benchmark_serving`, record the target metric (default
`total_token_throughput`).

## Layout

```
.buildkite/xla_autotune/
├── autotuner.py             OFAT driver
├── vllm_test_framework.py   vllm serve + benchmark_serving runner + MODELS registry
├── flags.txt                Candidate flags, one per line
├── pipeline.yml             Buildkite pipeline definition (matrix-sharded)
└── README.md
```

The host wrapper `.buildkite/scripts/run_xla_autotune_shard.sh` runs one
shard: starts the watcher that ships results to Buildkite as they land,
`git clone`s [`kimbochen/bench_serving`][bs] inside docker (same harness as
`tests/e2e/benchmarking/bm_qwen3_coder.sh`), then invokes `autotuner.py`.

[bs]: https://github.com/kimbochen/bench_serving

## Sharding

For a flag list of length `F` over `N` shards:

* Shard `k` (1-based) processes `flags[(k-1) * ceil(F/N) : k * ceil(F/N)]`.
* To run `N` shards, list `N` entries in `matrix` and set
  `AUTOTUNE_TOTAL_SHARDS` to the same `N` in `pipeline.yml` (single source of
  truth — the shard script reads it; `matrix` is the only list to grow).
* Every shard also runs `AUTOTUNE_BASELINE_RUNS` baselines (no extra
  `LIBTPU_INIT_ARGS`) so the per-VM noise floor can be re-estimated.
* Per `(input_len, output_len)` shape, every trial runs `warmup_runs`
  warmup passes (discarded) plus one measured pass — the first-batch
  latency reflects compile / cache transients, not steady-state.

## Trigger

This runs on the existing `tpu-inference-dev` pipeline.  Its bootstrap
(`.buildkite/scripts/bootstrap_dev.sh`) uploads `pipeline_dev.yml` by
default, or the file named in the `DEV_PIPELINE_FILE` build env var — so
the sweep is selected per-build without touching `pipeline_dev.yml` or any
GitHub-check path:

```
curl -s -X POST -H "Authorization: Bearer $BUILDKITE_API_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.buildkite.com/v2/organizations/<org>/pipelines/tpu-inference-dev/builds" \
  -d '{
    "commit": "<full-40-char-sha>",
    "branch": "<branch>",
    "message": "xla autotune",
    "env": {"DEV_PIPELINE_FILE": ".buildkite/xla_autotune/pipeline.yml"},
    "ignore_pipeline_branch_filters": true
  }'
```

From a fork, set `branch` to the namespaced form `<fork-owner>:<branch>`.
Add any of the `AUTOTUNE_*` knobs below to the same `env` block.

## Knobs

| Env var                  | Default                              | Notes                                                            |
|--------------------------|--------------------------------------|------------------------------------------------------------------|
| `AUTOTUNE_MODEL`         | `Qwen/Qwen3.5-397B-A17B-FP8`         | Must be a key in `MODELS` (`vllm_test_framework.py`).            |
| `AUTOTUNE_TARGET_METRIC` | `total_token_throughput`             | Any key inside the benchmark_serving result JSON.                |
| `AUTOTUNE_BASELINE_RUNS` | `2`                                  | ≥2 lets you estimate per-shard noise floor.                      |
| `AUTOTUNE_FLAGS`         | `.buildkite/xla_autotune/flags.txt`  | One `--flag=value` per line; `#` comments allowed.               |
| `AUTOTUNE_SKIP_CANDIDATES` | `0`                                | Drop the first N candidates from this shard (resume).  Baselines always re-run. |
| `AUTOTUNE_CONFIG`        | _(unset)_                            | Optional JSON of `VLLMTestParam` field overrides.                |
| `AUTOTUNE_DRY_RUN`       | _(unset)_                            | If set, passes `--dry-run` (no `vllm serve` or benchmark).       |
| `AUTOTUNE_TOTAL_SHARDS`  | `1`                                  | Shard count; keep equal to the `matrix` length in `pipeline.yml`. |
| `AUTOTUNE_SERVER_STARTUP_TIMEOUT_S` | `7200`                    | Max wait for `vllm serve` to open its port before failing the trial (generous: a new flag set cold-compiles). |

### Resuming after a cancelled build

If a build is aborted partway through, the next build can skip the candidates
already completed by setting `AUTOTUNE_SKIP_CANDIDATES` in the Buildkite
trigger.  Trial indices stay stable, so e.g. setting `12` will resume at
`cand_013`.  Baselines always re-run so the noise floor is fresh.

## Onboarding a new model

Add a `ModelSpec` entry to the `MODELS` dict in `vllm_test_framework.py`
(`serve_args`, `server_env`, `benchmark_shapes`) and set `AUTOTUNE_MODEL`
in `pipeline.yml`.

## JAX compile cache

`run_in_docker.sh` mounts and synchronises a GCS-backed JAX compile cache
keyed on `jax<VERSION>_tpu<TPU>`:

* Before docker starts: `gsutil -m rsync` pulls
  `gs://ullm-ci-cache/jax_cache/<key>/` → `/mnt/disks/persist/tpu_jax_cache/<key>/`.
* The persist-disk path is exposed to the container as `VLLM_XLA_CACHE_PATH`
  and `JAX_COMPILATION_CACHE_DIR`.
* After docker exits: `gsutil -m rsync` pushes back.  Cache entries are
  content-addressed, so concurrent CI builds pushing in parallel is safe.

JAX keys each compiled HLO module on `(LIBTPU_INIT_ARGS bytes × shape)`,
so a new candidate flag invalidates only the small subset of modules it
actually affects — observed +2–4 % init overhead on the 397B model vs
warm baseline (~10 min), not the +90 min of a cold compile.

## Result handoff

While a shard runs, `autotuner.py` writes each trial's record to
`<artifact-dir>/<trial_id>.json` and appends to `summary.jsonl` the moment
the trial finishes; the per-trial log bundle gets a sibling `<bundle>.done`
marker once the autotuner is done writing it.  In parallel, the host
wrapper's watcher loop (every 20 s):

* Re-uploads top-level JSON / `summary.jsonl` whose mtime advanced.
* Uploads each completed log bundle exactly once (driven by `.done`).

A final sweep after the autotuner exits picks up anything written between
the watcher's last tick and the docker exit.  All artifacts land in
Buildkite under `shard_<k>_of_<N>/...`, mirroring the on-disk layout.

## Interpreting results

Always inspect the baselines first.  If their spread is >10 % (per-VM
noise on the same flag set), the candidate signal will be drowned out —
bump `AUTOTUNE_BASELINE_RUNS`, or repeat a flag multiple times in
`flags.txt` to get candidate replicates.
