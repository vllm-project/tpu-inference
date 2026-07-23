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

Buildkite concurrency groups cap this POC at eight simultaneous one-chip jobs
and one eight-chip job across overlapping builds. This matches the currently
usable quota, prevents Agent Stack from creating pods that cannot be scheduled,
and makes scarce-pool waiting visible as an intentional pipeline gate.

The live `kube-dev` pipeline only uploads `pipeline_kube.yaml`, using its
64-core builder through the cluster's `cpu` queue. The POC image step explicitly
uses `queue: cpu` and builds only `TPU_VERSION=tpu6e`; it does not upload the
shared `pipeline_build.yml` or build a tpu7x image. `pipeline_build.yml` keeps
its `cpu_64_core` default for the existing bare-metal pipeline. If kube-dev
ever reuses that shared file, upload it with `CPU_QUEUE=cpu` rather than changing
the shared default.

## Cluster, queue, and storage topology

A GKE cluster cannot place node pools in two regions. Run one Agent Stack
installation per GKE cluster/region and give each installation a distinct
Buildkite queue, for example `kube-sa-west1` and `kube-us-central1`. Do not let
controllers in both regions consume a generic shared queue: either can claim a
job before its TPU or data locality is known. A shared pipeline generator maps
the logical request (`v6e`, one or eight chips; `v7x`, topology) to the regional
queue and pod scheduling rules so feature engineers do not choose queues,
regions, zones, or node selectors themselves.

Within a region, prefer topology profiles over a queue for every zone. A
regional GKE cluster can keep its control plane and ordinary CPU node pool
spread across zones, while each TPU node pool is restricted to the zones where
that TPU type is available. Add a zone-specific queue only when it represents a
separate controller, security boundary, or operational capacity pool; otherwise
the generated pod's node affinity is the authoritative placement control.

For the current v6e cluster, ordinary CPU work that uses no persistent volume
can omit selectors and affinity and let the default node pool scale in any of
`southamerica-west1-a`, `b`, or `c`. CPU cache/model/dataset preparation is
different because it hands a zonal `premium-rwo` volume to a TPU pod that can
run only in `southamerica-west1-a`. Both the PVC and every CPU/TPU consumer must
therefore be constrained to `a`. In particular:

- If the PVC already exists, its volume node affinity should make the scheduler
  choose `a`, but explicit generated affinity makes the contract observable and
  prevents surprises.
- If the storage class delays provisioning until the first consumer, an
  unconstrained CPU prep pod could cause a new volume to be provisioned in `b`
  or `c`; constrain the prep pod and claim topology before it is scheduled.
- Ensure the default CPU pool can autoscale in `a`. The CPU prep pod must fully
  release the `ReadWriteOnce` attachment before the TPU pod starts.
- Use inexpensive zonal scratch volumes for disposable per-job caches. A
  regional persistent disk costs more and does not make a volume portable to a
  different region; use it only when same-region failure tolerance is actually
  required.

Maintain a region/zone-local golden PVC or snapshot generation beside each TPU
pool. Persistent disks and PVCs are not cross-region cache distribution. Keep
the trusted cache delta, model manifests, and dataset manifests in object
storage, then seed immutable local goldens independently in
`southamerica-west1` for v6e and `us-central1` for v7x. Compilation-cache keys
remain based on software version, TPU generation, and topology rather than
region, while the materialized snapshots are region-local. Measure and budget
cross-region object transfer before choosing the bucket locations.

Apply the same locality rule to container images. The POC currently publishes
to `us-central1-docker.pkg.dev`, so a newly scaled v6e node in South America
must pull layers across regions before its Buildkite agent can start. Build the
v6e image once, address it by digest, copy that digest to a regional Artifact
Registry repository beside each TPU cluster, and let the topology profile
choose the regional reference. This avoids duplicate builds while reducing
cold-node pull latency and cross-region transfer.

For a cost-sensitive POC, a zonal cluster in the TPU zone is sufficient. For a
production service, a regional cluster per TPU region is the better default:
the Agent Stack controller and generic CPU work can survive a zonal failure,
while TPU node pools and their zonal cache volumes stay in supported zones.

### Concrete multi-region balancing example

Suppose `v6e-1` exists in both `us-central1` and `us-west1`. Register two
physical execution pools for the one logical topology:

```yaml
v6e-1:
  - region: us-central1
    queue: kube-v6e1-us-central1
    max_parallel: 4
    zone: us-central1-b
    image_repository: us-central1-docker.pkg.dev/PROJECT/ci/vllm-tpu
    golden_pvc: tpu-cache-v6e1-golden
  - region: us-west1
    queue: kube-v6e1-us-west1
    max_parallel: 2
    zone: us-west1-b
    image_repository: us-west1-docker.pkg.dev/PROJECT/ci/vllm-tpu
    golden_pvc: tpu-cache-v6e1-golden
```

If one Agent Stack controller handles every accelerator profile in a region,
the queue names can simply be `kube-us-central1` and `kube-us-west1`; keep the
TPU type in the topology registry. Use accelerator-specific queues only when
controllers, permissions, quotas, or operational ownership are intentionally
separate.

A feature test remains region-neutral:

```toml
key = "jax-unit-part2"
topology = "v6e-1"
models = []
datasets = []
command = ["python3", "-m", "pytest", "tests/"]
```

The CPU planner queries or is given each pool's capacity and queued work,
represents every usable TPU slot by its estimated next-available time, sorts
the new jobs by expected duration, and greedily assigns each job to the pool
whose earliest slot gives the lowest predicted completion score:

```text
score = earliest slot availability
        + new job duration
        + cold-node penalty
        + missing-cache/model penalty
```

For example, suppose central's four slots are next available in
`[35, 55, 70, 80]` minutes and west's two slots in `[10, 45]` minutes. A new
90-minute unit shard has an earliest finish of `35 + 90 = 125m` in central and
`10 + 90 = 100m` in west, so it goes west. West's slot vector becomes
`[45, 100]` before the next job is placed. This balances the whole uploaded
matrix and accounts for a 90-minute shard differently from a five-minute test.

Buildkite does not support `agents.queue: A OR B`. Make the routing decision
before uploading the TPU steps. A small CPU planning step receives the logical
test declarations and uploads the already-routed graph:

```yaml
steps:
  - label: ":traffic_light: Plan regional TPU work"
    agents:
      queue: cpu
    command: |
      python3 .buildkite/scripts/plan_kube_matrix.py \
        --topology v6e-1 \
        --output /tmp/routed-pipeline.yaml
      buildkite-agent pipeline upload /tmp/routed-pipeline.yaml
```

`plan_kube_matrix.py` should emit a full regional bundle, not merely set one
global queue variable: every logical test can be routed independently, while
each test's CPU-prepare, TPU-run, CPU-finalize, and cleanup steps stay together.

The generated TPU step then contains physical details that were absent from the
feature declaration:

```yaml
agents:
  queue: kube-v6e1-us-west1
concurrency: 2
concurrency_group: tpu/v6e-1/us-west1
plugins:
  - kubernetes:
      podSpec:
        nodeSelector:
          cloud.google.com/gke-tpu-accelerator: tpu-v6e-slice
          cloud.google.com/gke-tpu-topology: 1x1
          topology.kubernetes.io/zone: us-west1-b
        containers:
          - name: vllm-tpu-runner
            image: us-west1-docker.pkg.dev/PROJECT/ci/vllm-tpu@sha256:DIGEST
```

Its generated prep and finalizer use the same queue and zone but omit the TPU
accelerator/topology selectors and TPU resource request, allowing ordinary CPU
nodes in `us-west1-b` to run them.

For an initial implementation without live capacity APIs, use weighted
rendezvous hashing over `pull-request + step key`, with four virtual slots for
central and two for west in this example. It provides the intended 2:1 split,
keeps the same PR test in the same region across commits for cache locality,
and avoids reshuffling most jobs when a pool changes. Add live Buildkite queue
and Kubernetes pending-state inputs later; Buildkite Test Engine timing history
can supply the duration estimates.

The planner emits the selected region into the entire CPU-prepare, TPU-run,
CPU-finalize bundle. All three steps use the same physical queue, regional
image, zone, and PVC names. Once preparation creates a regional PVC, do not
move only the TPU step to another region. If the chosen pool reaches a pending
timeout before work starts, clean up and retry the whole bundle in the other
region. Provide an operator override such as `KUBE_REGION=us-west1` for outage
handling and reproducible debugging.

Do not make both regional Agent Stack installations consume one shared queue
to get accidental first-agent-wins balancing. It can work for a stateless
single pod with identical resources in both clusters, but it cannot guarantee
that separate preparation and TPU steps land in the same region, and it hides
which cache, image registry, quota, and PVC namespace the job will use.

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

The source image exports `TPU_INFERENCE_WORKSPACE=/tpu-inference/workspace` as
the canonical baked workspace. Multihost benchmark inputs use the explicitly
separate `/benchmark-data` scratch root; `/workspace/tpu_inference` remains a
compatibility path for workflows that bind-mount a live checkout there.

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
| 6 | `kube_e2e_speculative_decoding` | Default | Runs the per-push BVT subset normally and the full suite when `KUBE_FULL_MATRIX=1`, matching bare-metal nightly behavior. |
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

- **Dependency/gate**: `runnable_at - created_at`; time waiting for upstream
  steps and any Buildkite concurrency gate.
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

### Corrected cold-clone run: kube-dev build 34

[Build 34](https://buildkite.com/tpu-commons/kube-dev/builds/34) uses the exact
branch commit `5d485b4b61f2f68d112e2233619ecda2b1f47511` and vLLM LKG
`191146dba5bb9d99f5efd48e022d340ecdf8fad85d`. Every TPU job had a 7m
50s image dependency.
The cache PVCs were cloned before the golden PVC was refreshed, so this remains
the cold-cache measurement.

| Step | Queue/provision | Execution | Result |
|---|---:|---:|---|
| MLPerf JAX | 3m 59s | 17m 42s | Passed |
| MLPerf JAX + vLLM | 3m 46s | 15m 42s | Passed |
| Speculative decoding BVT | 4m 26s | 2h 32m 54s | Passed, 11 of 11 cases |
| Unit tests part 1 | 14m 07s | 23m 55s | Passed |
| Unit tests part 2 | 3m 50s | 1h 39m 36s | Passed |
| Kernel unit tests | 9m 47s | 1h 28m 26s | Passed |
| LoRA unit tests | 11m 12s | 2m 29s | Passed |
| RunAI JAX | 12m 13s | 6m 57s | Passed |
| RunAI Torchax | 4m 44s | 4m 38s | Passed |
| Qwen2.5-VL accuracy | 7m 09s | 10m 06s | Passed |
| LoRA E2E multi-chip | 24m 32s | 7m 13s | Passed |
| LoRA unit multi-chip | 3m 49s | 2m 08s | Passed |
| RunAI Torchax Ray | 6m 22s | 5m 11s | Passed |
| Single-host disaggregation | 12m 39s | 3m 42s | Passed |
| MPMD data parallelism | 16m 45s | 7m 15s | Passed |

All 15 default TPU counterparts passed. The first coverage-report attempt failed on
the CPU host because its Python 3.14 interpreter was incompatible with the old
Pex bootstrap, and its path mapping omitted Agent Stack's
`/workspace/build/buildkite` checkout. The current pipeline runs the pinned
producer version of coverage.py in Python 3.12 and aliases all source layouts;
combining build 34's two real artifacts locally passes the 68% threshold.

Cold compilation, rather than serving throughput, explains the execution gap.
For example, the first Qwen3.5 MTP speculative case took 50m 44s here versus
5m 58s in the same-time warm bare-metal build 23029. MLPerf post-readiness
benchmark phases were effectively equal; most of its Kubernetes delta was
server startup and compilation. A run cloned after the golden refresh is
required before drawing a platform-performance conclusion.

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

Build 22970 did not run the change-gated kernel shard. Five recent passing
bare-metal `main` executions provide a separate reference: builds 23011,
22995, 22914, 22892, and 22885 took 70m 25s, 74m 33s, 71m 09s, 71m 03s,
and 70m 10s respectively (median 71m 03s). In build 23011, the kernel job also
had a 9m 54s image dependency and 0.9s queue time, for 1h 20m 19s end to end.
These commits differ from the Kubernetes POC, so use the range as diagnostic
context rather than a platform speedup measurement.

### Same-time bare-metal nightly: build 23029

[Build 23029](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/23029)
started at 07:00 UTC alongside Kubernetes build 34 and uses commit
`468b8afc137fda7c262bfcee99d34f8d53b2000e`, the POC branch's main base and
the same vLLM LKG. Its overall build is failing because the nightly contains
many unrelated matrices, but the individual v6e counterparts below passed.
All had a 9m 24s Docker dependency; the table separates subsequent bare-metal
agent queue time from execution.

| `pipeline_jax.yml` step | Queue | Execution |
|---|---:|---:|
| 0: MLPerf JAX | 9m 40s | 8m 36s |
| 1: MLPerf quantized | 12m 04s | 4m 48s |
| 2: MLPerf new models | 12m 37s | 8m 33s |
| 3: MLPerf JAX + vLLM | 12m 38s | 7m 47s |
| 4: MLPerf Llama4 | 44m 40s | 8m 30s |
| 6: Speculative decoding, full nightly suite | 13m 00s | 1h 17m 40s |
| 7_1: Unit tests part 1 | 13m 13s | 19m 06s |
| 7_2: Unit tests part 2 | 13m 14s | 1h 23m 55s |
| 7_3: Combine and report coverage | 0.6s | 11.5s |
| 8: Kernel unit tests | 13m 14s | 1h 11m 01s |
| 9: Collective kernels | 46m 21s | 6m 27s |
| 10_1: LoRA E2E | 13m 15s | 4m 38s |
| 10_2: LoRA adapter E2E | 13m 24s | 6m 05s |
| 11: MLPerf JAX + vLLM multi-chip | 48m 03s | 9m 58s |
| 13: LoRA E2E multi-chip | 48m 39s | 7m 24s |
| 15: LoRA unit tests | 13m 39s | 3m 18s |
| 16: LoRA unit multi-chip | 48m 45s | 6m 04s |
| 27: RunAI JAX | 16m 14s | 3m 25s |
| 28: RunAI Torchax | 16m 54s | 3m 42s |
| 29: RunAI Torchax Ray | 59m 20s | 6m 55s |
| 33: Qwen2.5-VL accuracy | 16m 58s | 6m 57s |
| 34: Single-host disaggregation | 1h 00m 14s | 7m 04s |
| 35: MPMD data parallelism | 1h 00m 23s | 5m 37s |

The 1-chip and especially 8-chip bare-metal queues were busy in this nightly,
so persistent agents do not guarantee negligible queue time. Conversely, its
execution times reflect a warm persistent cache and provide the most useful
same-time reference for the refreshed-golden Kubernetes run.

Do not calculate a headline speedup from these two historical builds. Build 32
used a different commit and stale vLLM image; several commands differed or
false-passed; and the cache architectures differ. It is still useful for
showing that elastic capacity dominates Kubernetes wait time and that 8-chip
parallelism is quota-bound.

## Running and collecting results

Push the branch, then run the default matrix. Run the expanded matrix only
after the default build is green:

```bash
bk build create --yes --pipeline tpu-commons/kube-dev \
  --branch test-kube --commit "$(git rev-parse HEAD)" \
  --message "Kubernetes Agent Stack POC"

bk build create --yes --pipeline tpu-commons/kube-dev \
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

Or render the timing columns directly as Markdown:

```bash
bk job list --pipeline tpu-commons/kube-dev --build BUILD_NUMBER \
  --no-limit --json | python3 .buildkite/scripts/report_job_timings.py
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

## Recommended cache lifecycle

An opt-in implementation of this flow now lives in `pipeline_kube.yaml`. Launch
it with `KUBE_CPU_ONLY=1` and `KUBE_RESOURCE_PREP_POC=1`: the first flag skips
the standard TPU matrix, while the explicit resource-prep flag schedules its
single prepared TPU counterpart. The graph is:

```text
CPU control: create named clone
  -> CPU default-pool pod: hydrate cache + prepare declared models/dataset
  -> v6e-1 pod: run MLPerf using the same named PVC
  -> CPU default-pool pod: publish only new cache entries
  -> CPU control: delete the clone
```

The feature declaration is limited to model identifiers, dataset aliases,
topology, and command. `resource_registry.json` owns the download mechanism and
verified destination; `prepare_kube_resources.py` owns hydration and produces a
resolved manifest on the PVC. The current proof declares the three default
MLPerf models and the `mlperf` dataset. `MLPERF_DATASET_PATH` lets the existing
benchmark consume the staged dataset without knowing how it was downloaded.
The dataset preparation also installs NLTK's `punkt` and `punkt_tab` resources
under the same PVC. The prepared TPU step sets `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`, turning an incomplete model preparation into a clear
handoff failure instead of a hidden TPU-time download.

Set `KUBE_CACHE_WRITE_PREFIX=gs://...` only when intentionally testing cache
publication. If it is absent, the CPU finalizer reports the delta without
uploading. Publication uses create-only object generations, so concurrent jobs
cannot replace an existing entry. A failed TPU step skips publication but still
allows the cleanup step to delete the clone.

Apply `.buildkite/kubernetes/resource-pvc-rbac.yaml` once to the target cluster.
It creates the dedicated `buildkite-resource-pvc-manager` service account with
namespace-scoped `get`, `create`, and `delete` permissions on persistent volume
claims. Only the PVC create/cleanup control pods select that account; TPU and
ordinary CPU-preparation pods continue to use the unprivileged default. The
control step exposes a clear HTTP 403 if that Role/RoleBinding is missing. Do
not wait for the claim to become `Bound` in the control step: `premium-rwo` can use
`WaitForFirstConsumer`, so the zone-pinned CPU preparation pod is what triggers
binding in `southamerica-west1-a`.

Preparation also rejects a directory such as
`/cache/tpu_jax_cache/jax0.10.2_tputpu6e`; the namespace contents must be at
`/cache/tpu_jax_cache` itself. This turns a silent cold-cache layout error into
an actionable CPU-step failure before a TPU is allocated.

Do not place GCS or GCS FUSE on the live JAX cache path. Keep GCS as the
authoritative store and use a cloned block volume as the fast local working
set. An init container in the TPU pod is not sufficient for resource savings:
Kubernetes schedules the whole pod, including its TPU request, before init
containers run. Use three separately scheduled phases instead:

1. A CPU Buildkite step creates a named, per-job PVC from an immutable,
   versioned golden snapshot. It launches a non-TPU Kubernetes Job that mounts
   the PVC and merges the global delta and, for a pull request, that PR's delta
   into the same `/cache/tpu_jax_cache` directory. It can also stage models and
   benchmark datasets and verify their checksums. The CPU pod requests no TPU
   and uses the ordinary node pool, but its generated zone affinity must match
   the target TPU and zonal PVC. It must terminate and release the
   `ReadWriteOnce` volume before the TPU phase starts.
2. The TPU step depends on preparation and mounts the existing named PVC. Both
   `VLLM_XLA_CACHE_PATH` and `JAX_COMPILATION_CACHE_DIR` continue to point to the
   one merged local directory. The TPU job performs no GCS synchronization and
   writes new entries directly into that directory.
3. A CPU finalizer runs after the TPU pod exits, mounts the same PVC, publishes
   only files created by a successful test with no-clobber semantics, uploads
   remaining artifacts, and deletes the per-job PVC. Cache publication is
   measured separately from TPU execution.
4. Main/nightly builds publish to a trusted global delta. Pull-request builds
   read the golden snapshot plus the global delta plus an expiring overlay such
   as `cache/pr/<number>/jax-<version>/tpu-v6e/<topology>/`, and write only to
   their own overlay. A new PR therefore starts with the shared cache, while
   later commits in the PR also reuse that PR's entries.
5. A scheduled CPU seed job compacts the trusted global delta into a staging
   PVC, validates it, and publishes a new immutable snapshot. Retain one or two
   prior generations for rollback.

GCS object lifecycle rules can use object-name prefix conditions (there are no
real directories). For example, objects below `cache/pr/` can expire after 14
days while `cache/global/` uses a longer period such as 90 days. Explicitly
delete a PR prefix when the PR closes and use lifecycle expiry as a safety net.
Object age is based on object creation time; no-clobber hydration does not
refresh it, so retention must be long enough for active PRs or active prefixes
must be refreshed intentionally.

The current `premium-rwo` PVC-clone design is billed by provisioned disk
capacity, not changed blocks. A 500 GiB golden PVC and a 500 GiB job clone
therefore account for 1,000 GiB of provisioned capacity while both exist; 10
GiB written by the job is contained in the clone's 500 GiB allocation rather
than billed as an additional disk. The short-lived clone is prorated for its
lifetime. A lower-cost experiment should test an immutable golden disk mounted
read-only plus a small writable upper volume, presented as one cache directory
through an overlay filesystem. That can approach golden-plus-delta capacity,
but requires validation of GKE multi-reader attachment, mount privileges,
small-file performance, and cleanup before adoption.

For the current JAX comparison, bare metal uses the GCS namespace
`jax0.10.2_tputpu6e`. Because Kubernetes points both cache variables directly
at `/cache/tpu_jax_cache`, that directory must contain the namespace contents,
not an extra `jax0.10.2_tputpu6e` parent directory.

## Reducing TPU-held non-TPU work

In priority order, the Kubernetes pipeline should also:

1. Reuse a v6e environment image keyed by the vLLM LKG plus Docker/dependency
   inputs instead of rebuilding a large image for every tpu-inference commit.
   The Buildkite checkout is already the authoritative tpu-inference source.
2. Classify and move CPU-safe tests out of the broad unit-test shard. Run test
   collection, imports, configuration checks, credential probes, and dataset
   validation on CPU before scheduling TPU jobs.
3. Move model and dataset downloads into the CPU preparation phase. A cache
   miss in `mlperf.sh` currently downloads and verifies OpenOrca while the TPU
   pod is allocated.
4. Consolidate closely related 8-chip tests where useful. They are serial on
   the only 8-chip node, so fewer pods can avoid repeated image pull, volume
   attachment, and framework startup while preserving sensible retry units.
5. Move cache publication, coverage/artifact handling, reporting, and PVC
   deletion to CPU finalizers so the TPU pod can terminate immediately after
   the actual test.
6. Cancel superseded PR builds and select TPU tests from the changed areas.
   Keep the default BVT matrix per commit and reserve the expanded matrix for
   relevant changes, main, and nightly runs.
7. Produce the common compilation cache once after JAX, vLLM, or image changes
   rather than letting many PR jobs compile the same shapes independently.

For bare metal, the analogous quick wins are moving the GCS rsync and Docker
build/prune work out of the TPU-assigned test command. Persistent agents can
refresh caches and pre-pull images periodically while idle instead of doing so
after a Buildkite job has acquired the TPU.

## Change-based and tiered test selection

The first CPU tier is implemented as an explicit shadow allowlist in
`.buildkite/cpu_safe_tests.txt`. `KUBE_CPU_ONLY=1` builds the exact test image,
runs only those files with `JAX_PLATFORMS=cpu`, no TPU device, and Docker
networking disabled, then skips every standard TPU step. The same tests remain
in their TPU shards until repeated CPU-only builds pass; this is deliberate
shadowing rather than an immediate reduction in TPU coverage.

The exact selector commit was validated by kube-dev build 40: 72 tests and 11
subtests passed in 9.64 seconds with Docker networking disabled. Its enclosing
Buildkite job took 14.5 seconds and started 0.6 seconds after the image became
available. The remaining warning is deliberate evidence that
`BUILDKITE_ANALYTICS_TOKEN` still needs to be provisioned before Test Engine
history can be used.

`select_kube_tests.py` and `test_ownership.json` implement the corresponding
change selector in shadow mode. For example:

```bash
python3 .buildkite/scripts/select_kube_tests.py --base origin/main
```

The JSON output includes selected CPU and TPU step keys, ownership reasons,
unowned files, and whether the full matrix is required. Documentation-only
changes select no tests; unknown code widens to the complete default matrix;
CI/image/dependency changes and main/nightly/release events select the full
feasible matrix. The output retains `"shadow": true` until repeated audits show
that targeted plans would not have hidden failures. Selection emits logical
steps only; a later Test Engine integration may split those candidates by
historical timing without changing this ownership policy.

The existing bootstrap already skips documentation-only builds and gates the
kernel and collective-kernel shards from changed paths. Extend that mechanism
through one deterministic selector which consumes the merge-base diff and
emits a machine-readable test plan. Keep policy out of individual feature
pipelines: they should declare test ownership, topology, models, and datasets;
the selector decides the tier and the shared generator expands it into steps.

Recommended tiers are:

1. **CPU presubmit:** pipeline/config validation, collection checks, and an
   explicit CPU-safe unit-test allowlist. Run for every code change.
2. **v6e one-chip presubmit:** speculative BVT plus tests owned by changed
   runtime/model areas. Changes to shared runtime, dependency, Docker, pytest,
   or pipeline infrastructure fall back to the complete one-chip presubmit.
3. **v6e eight-chip targeted:** collective, distributed, LoRA, Ray, disagg, and
   MPMD tests only when their owners or shared distributed code change. Keep
   concurrency at one while the pool has one node.
4. **Full feasible v6e:** main, nightly, release, an explicit PR label, and a
   periodic audit that catches gaps in the change map.

Do not infer CPU safety merely because a test is fast on TPU. Add an explicit
`cpu_safe` pytest marker only after the test passes in the same image with
`JAX_PLATFORMS=cpu`, no TPU device mounted, and network/model access disabled
unless that access is the behavior under test. Initial high-confidence
candidates to validate are `tests/test_envs.py`, `tests/test_nightly_gate.py`,
`tests/offload/tpu_offload_cpu_backend_test.py`,
`tools/kernel/tuner/v1/tests/test_tuned_params_structure.py`, and
`tools/kernel/tuner/v1/tests/test_inspect_result_cli.py`. Keep them in the TPU
shard during the first shadow runs; remove them only after repeated CPU passes.

For easy local and CI use, a shared runner should locate only files containing
the marker, then invoke `pytest -m cpu_safe` on those files. That avoids
importing every TPU-only module during collection while preserving normal
pytest node IDs. The selector should output candidate node IDs; scheduling and
splitting are separate concerns. A future Buildkite Test Engine (`bktec`)
integration can therefore split the selected CPU, one-chip, and eight-chip
candidates by historical timing without changing the ownership map. Publish
test analytics for all tiers and use distinct suite identities where CPU and
TPU durations differ. The collector is already installed through
`requirements_test.txt`, but build 34 reported that
`BUILDKITE_ANALYTICS_TOKEN` was absent in the Kubernetes container. Provision
that token as a managed cluster secret before relying on Test Engine history;
do not place it directly in pipeline YAML.
