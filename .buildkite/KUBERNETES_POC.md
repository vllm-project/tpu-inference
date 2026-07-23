# Buildkite Agent Stack for Kubernetes POC

This POC compares the `ci-dev` Agent Stack cluster (`kube` queue) with the
existing TPU v6e bare-metal queues used by `pipeline_jax.yml`. It is limited to
the available GKE node pools:

- `ct6e-standard-1t`: one TPU chip per node, one existing node, autoscaling
  enabled.
- `ct6e-standard-8t`: eight TPU chips per node, one existing node, autoscaling
  enabled. The project currently has only seven chips of spare quota, so this
  pool cannot add a second node.

The Kubernetes pipeline is `.buildkite/pipeline_kube.yaml`. Its default matrix
contains the per-push v6e-compatible tests; setting `KUBE_FULL_MATRIX=1` adds
the feasible nightly tests. Every TPU test is soft-failing for parity with
`pipeline_jax.yml`, followed by a hard validation step so a failed POC test
still makes the build visibly fail.

## Reproducibility and source integrity

Each build first creates an image from the exact `BUILDKITE_COMMIT` and the
vLLM commit in `.buildkite/vllm_lkg.version`. It publishes the immutable alias
`vllm-tpu:kube-${BUILDKITE_COMMIT}`, and all Kubernetes jobs depend on that
build. This replaces the original manually maintained image tag, which allowed
the checked-out tests and installed code to drift apart.

Agent Stack mounts the checkout below `/workspace`, while the image keeps its
baked vLLM checkout below `/tpu-inference/workspace`. Benchmark helpers now use
the live tpu-inference checkout and independently locate the matching baked
vLLM checkout. Compatibility symlinks preserve the legacy `/workspace/vllm`
and `/workspace/tpu_inference` paths used by bare-metal jobs.

The cache PVC is cloned from `tpu-cache-golden-pvc` for every pod. This is not
the same cache model as the long-lived bare-metal disk, so results must be
labelled as cold-clone or warm-cache before using them for performance claims.

## v6e parity

| `pipeline_jax.yml` | Kubernetes step | Matrix | Notes |
|---|---|---|---|
| 0 | `kube_e2e_mlperf_jax` | Default | Same MLPerf command. |
| 1 | `kube_e2e_mlperf_quantized` | Full | Bare-metal nightly only. |
| 2 | `kube_e2e_mlperf_new_models` | Full | Bare-metal nightly only. |
| 3 | `kube_e2e_mlperf_jax_vllm` | Default | Same default model list; the initial POC ran only two models. |
| 4 | `kube_e2e_mlperf_llama4` | Full | Uses the 8-chip pool. |
| 6 | `kube_e2e_speculative_decoding` | Default | Runs the per-push BVT subset. |
| 7_1, 7_2, 7_3 | `kube_jax_unit_tests_part1`, `part2`, `report` | Default | Produces and combines coverage. GKE's injected `TPU_ACCELERATOR_TYPE` is unset only for `tests/test_envs.py`. |
| 8 | `kube_jax_unit_tests_kernels` | Default | Excludes collectives and the same unsupported files as bare metal. |
| 9 | `kube_jax_unit_tests_collectives` | Full | Uses the 8-chip pool. |
| 10_1, 10_2 | `kube_lora_e2e_single_chip`, `kube_lora_adapter_e2e_single_chip` | Full | Bare-metal nightly only. |
| 11 | `kube_e2e_mlperf_jax_vllm_multi_chip` | Full | Sets `USE_V6E8_QUEUE=True`. The bare-metal step selects the 8-chip queue but omits this flag, so it currently selects a one-chip model despite its label. |
| 13 | `kube_lora_e2e_multi_chip` | Default | 8-chip pool. |
| 15 | `kube_lora_unit_tests` | Default | Runs `test_bgmv.py` and `test_layers.py`; the initial POC accidentally ran the E2E test. |
| 16 | `kube_lora_unit_tests_multi_chip` | Default | 8-chip pool. |
| 27, 28 | `kube_runai_streamer_jax`, `kube_runai_streamer_torchax` | Default | One-chip pool. |
| 29 | `kube_runai_streamer_torchax_ray` | Default | 8-chip pool. |
| 33 | `kube_qwen2_5_vl_7b_accuracy` | Default | One-chip pool. |
| 34, 35 | `kube_disagg_single_host`, `kube_mpmd_data_parallelism` | Default | 8-chip pool. The MPMD shell variables are escaped from pipeline-upload interpolation. |

Steps 12, 14, 17-24, 30-32, 36, and 37 require TPU v7x and have no POC
counterpart. Step 25 is multi-host and cannot run with one instance in each
pool. Step 26 is both v7x multi-host and currently disabled upstream.

## Baseline timing

Timing definitions:

- **Dependency**: `runnable_at - created_at`; time waiting for upstream steps.
- **Queue/provision**: `started_at - runnable_at`; includes Buildkite queueing,
  pod scheduling, PVC creation, and node autoscaling.
- **Execution**: `finished_at - started_at`; includes checkout, plugin setup,
  test execution, artifacts, and teardown.
- **End-to-end**: dependency + queue/provision + execution.

### Kubernetes baseline: kube-dev build 32

[Build 32](https://buildkite.com/tpu-commons/kube-dev/builds/32) used the
original static image and is therefore a diagnostic baseline, not the final
apples-to-apples result. Its jobs became runnable about three seconds after
creation.

| Step | Pool | Queue/provision | Execution | Result |
|---|---:|---:|---:|---|
| MLPerf JAX | 1 chip | 3m 02s | 16m 07s | Passed |
| MLPerf JAX + vLLM | 1 chip | 0m 26s | 5m 45s | Passed; only two models |
| Speculative decoding | 1 chip | 1m 05s | 49m 02s | Failed; stale vLLM image |
| Unit tests part 1 | 1 chip | 15m 04s | 24m 09s | Passed |
| Unit tests part 2 | 1 chip | 1m 58s | 93m 31s | Failed; injected TPU env value |
| Kernel unit tests | 1 chip | 4m 58s | 96m 01s | Passed; broader than bare metal |
| LoRA unit tests | 1 chip | 11m 08s | 5m 08s | Passed; wrong test selection |
| RunAI JAX | 1 chip | 10m 17s | 6m 55s | Passed |
| RunAI Torchax | 1 chip | 6m 39s | 3m 54s | Passed |
| Qwen2.5-VL accuracy | 1 chip | 4m 09s | 10m 39s | Passed |
| LoRA E2E multi-chip | 8 chips | 10m 34s | 6m 44s | Passed |
| LoRA unit multi-chip | 8 chips | 26m 07s | 2m 09s | Passed |
| RunAI Torchax Ray | 8 chips | 28m 31s | 4m 39s | Passed |
| Single-host disaggregation | 8 chips | 22m 14s | 3m 23s | Passed |
| MPMD data parallelism | 8 chips | 17m 44s | 4m 01s | False pass; model expanded to empty string |

The one-chip pool scaled out, with starts ranging from 26 seconds to 15
minutes. Every 8-chip job reused the only 8-chip node and ran serially, so its
queue/provision time rose to 28m 31s. Once a pod was available, the Buildkite
agent registered and accepted its job in roughly one second; agent connection
was not the bottleneck.

### Bare-metal baseline: tpu-inference-ci build 22970

[Build 22970](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/22970)
was a passing `main` build. Each test waited about 10m 10s for the shared Docker
build dependency, then about one second for a persistent agent.

| Step | Queue | Execution | End-to-end including image dependency |
|---|---:|---:|---:|
| 0: MLPerf JAX | 1.1s | 8m 46s | 18m 56s |
| 3: MLPerf JAX + vLLM | 1.1s | 7m 46s | 17m 57s |
| 6: Speculative decoding | 1.1s | 27m 12s | 37m 22s |
| 7_1: Unit tests part 1 | 1.1s | 17m 03s | 27m 14s |
| 7_2: Unit tests part 2 | 0.9s | 81m 09s | 91m 20s |
| 13: LoRA E2E multi-chip | 1.0s | 7m 26s | 17m 37s |
| 15: LoRA unit tests | 1.1s | 3m 17s | 13m 28s |
| 16: LoRA unit multi-chip | 1.1s | 6m 07s | 16m 17s |
| 27: RunAI JAX | 0.8s | 3m 29s | 13m 40s |
| 28: RunAI Torchax | 0.8s | 3m 46s | 13m 57s |
| 29: RunAI Torchax Ray | 1.0s | 7m 01s | 17m 12s |
| 33: Qwen2.5-VL accuracy | 1.1s | 6m 17s | 16m 27s |
| 34: Single-host disaggregation | 1.0s | 7m 07s | 17m 18s |
| 35: MPMD data parallelism | 1.0s | 6m 07s | 16m 18s |

Do not calculate a headline speedup from these two historical builds. Build 32
used a different commit and stale vLLM image; several commands differed or
false-passed; and the cache architectures differ. It is still useful for
showing that elastic capacity dominates Kubernetes wait time and that 8-chip
parallelism is quota-bound.

## Running and collecting results

Push the branch, then run the default matrix. Run the expanded matrix only
after the default build is green:

```bash
bk build create --pipeline tpu-commons/kube-dev \
  --branch test-kube --commit "$(git rev-parse HEAD)" \
  --message "Kubernetes Agent Stack POC"

bk build create --pipeline tpu-commons/kube-dev \
  --branch test-kube --commit "$(git rev-parse HEAD)" \
  --env KUBE_FULL_MATRIX=1 \
  --message "Kubernetes Agent Stack POC: full v6e matrix"
```

Retrieve timestamps without the web UI:

```bash
bk job list --pipeline tpu-commons/kube-dev --build BUILD_NUMBER \
  --no-limit --json > kube-jobs.json
bk job list --pipeline tpu-commons/tpu-inference-ci --build BUILD_NUMBER \
  --no-limit --json > bare-metal-jobs.json
```

Record the build URL, commit, vLLM LKG, matrix, cache state, pool sizes, and TPU
quota with every timing table. For a decision-quality comparison, collect at
least three runs per platform and report median and p95 queue/provision,
execution, end-to-end time, and estimated TPU-hours. Separate cold-node from
warm-node runs and compare only identical step commands.

## Success criteria and next improvements

The POC is successful when:

1. The default and expanded feasible-v6e matrices pass against a commit-specific
   image, with skipped tests explicitly explained.
2. Three repeated runs show predictable queue/provision behavior and no
   indefinite pods, image drift, false passes, or unexplained flakes.
3. TPU-hours and wall time are acceptable at expected concurrency, including
   the deliberately serial 8-chip workload.
4. Cache policy is explicit and reproducible rather than an accidental benefit
   of either platform.

Recommended follow-ups:

- Reserve quota for a second 8-chip node or intentionally limit 8-chip job
  concurrency to one and treat its queue time as expected.
- Set a finite `pendingTimeout` after measuring realistic scale-up p95, so quota
  exhaustion fails clearly instead of waiting forever.
- Add build annotations or a small reporting step that emits the timing fields
  and TPU-hours automatically.
- Add image retention for `kube-*` aliases and delete them on a schedule after
  the comparison window.
- Align the bare-metal step 11 environment with its multi-chip label before
  using it as a performance comparator.
- Promote this pipeline from POC only after repeated results establish sensible
  resource requests, cache behavior, timeout, and concurrency limits.
