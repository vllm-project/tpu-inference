# Kubernetes Buildkite POC summary

Status: lifecycle proof successful; production adoption undecided.

This proof of concept runs the useful single-host counterparts of
`.buildkite/pipeline_jax.yml` on GKE TPU nodes through Buildkite Agent Stack,
Kueue, and MultiKueue. It is intended to establish facts for a decision, not to
presuppose that Kubernetes should replace the existing bare-metal runner.

For the deployable manifests and operator walkthrough, see
[`kubernetes/kueue-poc/README.md`](kubernetes/kueue-poc/README.md). For the
architecture and the objective alternative analysis, see
[`kubernetes/kueue-poc/DESIGN.md`](kubernetes/kueue-poc/DESIGN.md). Current
operational state and next investigations are in
[`kubernetes/kueue-poc/HANDOFF.md`](kubernetes/kueue-poc/HANDOFF.md).
The production-oriented Standard GKE target, Terraform foundation, and gap
analysis are in [`kubernetes/production/README.md`](kubernetes/production/README.md).

## Executive result

The POC demonstrated all of the following:

- one Buildkite queue, `kube`, can submit Agent Stack Jobs to a controller GKE
  cluster;
- a pipeline step can choose a logical TPU profile without choosing a region;
- Kueue can admit the Job and MultiKueue can copy it to a worker cluster;
- a worker ResourceFlavor can bind the Job to a TPU topology, zone, and
  reservation-backed ComputeClass;
- an image can be built once on the existing 64-core `cpu` queue and pulled by
  all TPU Jobs;
- 14 of 15 TPU steps in a representative run completed successfully.

It also found material issues that prevent a production recommendation today:

- Agent Stack's completed `checkout` container makes a running Job appear
  `Ready=False`. Kueue's `waitForPodsReady` therefore does not measure actual
  Job health for this workload. A three-hour timeout is a tested workaround,
  not a durable integration contract.
- Job-local caches begin cold. Compilation-heavy steps can be substantially
  slower than the warm-cache bare-metal baseline.
- MultiKueue does not copy arbitrary Kubernetes Secrets. The Agent Stack secret
  and any workload secrets must be provisioned in every worker cluster.
- PVC topology, regional capacity, cancellation cleanup, quota fairness, and
  controller behavior under sustained parallel builds still need validation.

The appropriate conclusion is to continue the POC and retain bare metal as the
baseline and fallback. A hybrid outcome is as plausible as a full migration.

## POC scope

### Included

- TPU v6e only.
- One-chip (`1x1`) and eight-chip (`2x4`) single-host profiles.
- Controller cluster `ci-test-controller` in `us-central1`.
- Worker cluster `ci-test-southamerica-west1-worker` in
  `southamerica-west1`, with TPU work pinned to `southamerica-west1-a`.
- Buildkite Agent Stack for Kubernetes `v0.46.2`.
- Kueue and MultiKueue `v0.19.0`.
- Buildkite queue `kube`; image build on queue `cpu`.
- A fresh 500 GiB `premium-rwo` generic ephemeral volume per TPU Job, mounted
  at `/cache`.
- Runnable single-host counterparts from `pipeline_jax.yml`.

### Deliberately excluded

- Multi-host TPU tests.
- TPU v7x and dynamic sub-slicing.
- Golden cache snapshots, PR cache overlays, and CPU-side model/dataset/cache
  preparation.
- Change-based test selection and a separate CPU-safe unit-test lane.
- A production availability, disaster recovery, or security design.
- A claim that Kubernetes is faster or cheaper than bare metal.

## User-facing pipeline

The bootstrap pipeline remains intentionally small:

```yaml
steps:
- label: ":pipeline: Upload Kube Pipeline"
  agents:
    queue: cpu
  command: |
    buildkite-agent pipeline upload .buildkite/pipeline_kube.yaml
```

Feature engineers use the normal Buildkite pipeline. They do not specify a GKE
cluster, region, zone, LocalQueue, ResourceFlavor, PVC, or reservation. The
pipeline's reusable plugin profiles translate a test's TPU requirement into
Kubernetes details:

| Logical profile | Job queue label | TPU topology | Current worker placement |
|---|---|---|---|
| v6e one chip | `v6e-1` | `1x1` | `southamerica-west1-a` |
| v6e eight chips | `v6e-8` | `2x4` | `southamerica-west1-a` |

Both kinds of test still use the single Buildkite queue `kube`. The
`kueue.x-k8s.io/queue-name` label is placed on the top-level Kubernetes Job by
the pipeline plugin configuration. It selects a LocalQueue in the Job's own
namespace (`buildkite`). A LocalQueue with the same name in a different
namespace is not visible to that Job.

The default run covers the most useful one-chip and eight-chip counterparts.
Set `KUBE_FULL_MATRIX=1` to include expensive/nightly variants. Set
`KUBE_TARGET_STEP=kube_e2e_mlperf_jax_vllm_multi_chip` to exercise that targeted
multi-chip step without enabling the full matrix.

## Runtime and storage behavior

The image stores source outside `/workspace` because Agent Stack owns and
mounts `/workspace` at runtime:

- installed TPU Inference source: `/tpu-inference/workspace/tpu_inference`;
- installed vLLM source: `/tpu-inference/workspace/vllm`;
- Agent Stack checkout: `/workspace`;
- compatibility links for bare metal: `/workspace/tpu_inference` and
  `/workspace/vllm` in the image before Agent Stack mounts its workspace.

Benchmark scripts resolve the repositories independently instead of assuming
that both live below the same directory.

Each TPU Job currently receives an empty, job-scoped 500 GiB volume:

```text
/cache/hf_home          model and Hugging Face cache
/cache/nltk_data        NLTK data
/cache/datasets         downloaded test datasets
/cache/tpu_jax_cache    JAX and vLLM XLA compilation cache
```

The TPU Job is responsible for all downloads and cache writes in the current
POC. The PVC is deleted with the Job, so it does not accelerate a later build.
This is simple and isolated, but intentionally leaves cache reuse unsolved.

## Pipeline parity

`pipeline_kube.yaml` keeps counterparts for the `pipeline_jax.yml` steps that
can run on the available single-host v6e shapes. It does not split out tests
merely because they might be CPU-safe; the unit-test shards mirror the existing
pipeline grouping so comparisons remain meaningful.

The Kubernetes pipeline adds two implementation-specific steps:

- a CPU image build/push step, which publishes a commit-addressed image;
- a CPU coverage aggregation step, because coverage artifacts are produced by
  independent TPU Jobs.

Individual test Jobs remain soft-failing, matching the existing pipeline's
diagnostic behavior. A final validation step makes the overall POC result
meaningful by propagating failures.

## Observed results

The builds below are historical observations, not controlled benchmark results.
The closest successful bare-metal reference is an ancestor of the Kubernetes
commit, but the Kubernetes branch includes later vLLM LKG and POC changes and
uses a different cache state. Use the results to identify areas to investigate,
not to declare a winner.

### Lifecycle debugging

| Build | Change under test | Result |
|---|---|---|
| 52, 55, 57 | Earlier timeout configurations | Three long Jobs repeatedly exited 137 near 30 minutes |
| 74 | Job active deadline forced to 120 seconds | Job exited 137 after about 1m32 of agent runtime, proving the Job override survives Agent Stack and MultiKueue |
| 75 | Controller Kueue readiness timeout extended | Kernel Job still exited 137 near 15m12 because the worker retained its shorter setting |
| 76 | Controller and worker readiness timeouts both set to 10800 seconds | Lifecycle timeout resolved; 14 of 15 TPU Jobs passed |

In build 76:

- speculative decoding passed in 2h09m45;
- kernel tests passed in 1h31m15;
- unit tests part 2 ran for 1h36m59 and failed normally on
  `test_mesh_devices_sorting`, after 1,563 passed and 340 skipped.

That last failure was an ordinary test failure (`exit 1`), not the former
lifecycle kill (`exit 137`).

### Repeated matrix validation

Builds 77–79 ran the same commit and were intentionally submitted together to
exercise admission and limited-capacity behavior.

| Build | Active TPU result | Overall result at 2026-07-25 19:55 UTC |
|---|---|---|
| [76](https://buildkite.com/tpu-commons/kube-dev/builds/76) | 14 of 15 passed; one ordinary test failure | Failed |
| [77](https://buildkite.com/tpu-commons/kube-dev/builds/77) | 15 of 15 passed | Passed |
| [78](https://buildkite.com/tpu-commons/kube-dev/builds/78) | 15 of 15 passed | Passed |
| [79](https://buildkite.com/tpu-commons/kube-dev/builds/79) | 13 passed; speculative decoding and unit part 2 still running | Running |

The conditionally disabled full-matrix and coverage-replay rows did not start
and are excluded from the counts. Builds 77 and 78 are the strongest functional
result: the active matrix completed twice without the former secret,
admission, or exit-137 lifecycle failures.

### Execution-time comparison

The Kubernetes column is the mean of builds 77–79 for completed samples. The
two still-running build-79 tests use builds 77 and 78 only. Bare metal is
[build 23183](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/23183),
the closest successful main reference. The kernel step had no executed v6e
counterpart in that build and is therefore omitted.

| Step | Kubernetes mean | Bare metal | Observation |
|---|---:|---:|---|
| MLPerf JAX | 14m08s | 10m01s | Kubernetes 41% slower |
| MLPerf JAX + vLLM | 14m01s | 8m04s | Kubernetes 74% slower |
| Speculative decoding | 2h11m25s | 27m25s | Kubernetes 4.8x slower |
| JAX unit part 1 | 20m32s | 20m49s | Similar |
| JAX unit part 2 | 1h33m31s | 1h24m13s | Kubernetes 11% slower |
| LoRA unit single-chip | 1m41s | 3m23s | Kubernetes 50% faster |
| LoRA E2E multi-chip | 6m52s | 8m30s | Kubernetes 19% faster |
| LoRA unit multi-chip | 1m55s | 7m05s | Kubernetes 73% faster |
| RunAI JAX | 5m17s | 4m41s | Kubernetes 13% slower |
| RunAI TorchAX | 3m47s | 4m05s | Similar |
| RunAI Ray | 4m29s | 8m07s | Kubernetes 45% faster |
| Qwen accuracy | 9m31s | 6m43s | Kubernetes 42% slower |
| Disaggregated serving | 3m42s | 7m48s | Kubernetes 53% faster |
| MPMD | 5m52s | 5m57s | Similar |

The first Kubernetes image build in this batch took 6m52s; the next two
same-commit builds took approximately 1m05s after builder-cache reuse. The
bare-metal image build took 9m27s. Building and publishing once per commit is a
clear benefit.

These durations are not complete user-visible latency. Buildkite marks a
Kubernetes job started only after the ephemeral agent connects, so Kueue
admission, node/PVC provisioning, image pull, and Agent Stack startup are in
the pre-agent wait. A bare-metal agent starts almost immediately, while its
command duration includes GCS cache synchronization, Docker startup, and
cleanup. A fair future benchmark must record both phases.

### Capacity and startup observations

For the 15 active Kubernetes jobs, the time from Buildkite `runnable_at` until
the agent started was:

| Build | Median pre-agent wait | Maximum pre-agent wait |
|---|---:|---:|
| 77 | 5m25s | 23m33s |
| 78 | 37m14s | 48m28s |
| 79 | 1h08m02s | 1h44m21s |

Bare-metal jobs in build 23183 generally waited approximately one second on
already-running agents. The increasing Kubernetes wait across three concurrent
builds shows MultiKueue respecting limited capacity, but also shows that the
current one-chip quota and single eight-chip instance cannot provide low
latency for overlapping full matrices. Even the first build has an
approximately five-minute baseline before most agents connect. Buildkite alone
cannot divide that interval into central admission, worker admission, node
startup, PVC binding, image pull, and agent startup; Kubernetes event metrics
are required.

Overall wall time was 2h23m for build 77, 2h43m for build 78, and 1h35m for
bare-metal build 23183. Speculative decoding was the Kubernetes critical path;
contention added further latency to the later concurrent builds.

### Cache evidence

Both speculative-decoding jobs selected exactly 11 tests and deselected 21.
Pytest reported 7,831.95 seconds in Kubernetes versus 1,514.06 seconds on bare
metal. A sampled steady-state inference interval was much closer—approximately
104.1/236.7 input/output tokens per second in Kubernetes versus 107.8/245.1 on
bare metal—so normal TPU execution speed does not explain the 4.8x job-time
difference.

Bare metal explicitly pulled its JAX cache from GCS and wrote it back after a
successful job. Every Kubernetes Job started with an empty ephemeral model and
compilation-cache volume. Kubernetes builds 77 and 78 both took approximately
131–132 minutes for speculative decoding, confirming that one build did not
warm the next. Logs also showed compile stages taking tens of seconds on the
Kubernetes path where equivalent bare-metal stages were near-instant or much
shorter.

This strongly implicates compilation and other cacheable model-startup work,
but the current logs do not isolate JAX compilation, model download, and
initialization precisely enough to assign the entire difference to XLA. The
next controlled comparison should use the same source and vLLM commit, run one
build at a time, compare cold and prepared Kubernetes caches, and record cache
hits, compilation count/time, model download, and Kubernetes lifecycle events.

The data still does not support the statement that either platform is
uniformly faster. Short tests benefit from removing per-job Docker/cache
housekeeping, while compilation-heavy cold-cache tests can regress sharply.

## What the POC changed

- `.buildkite/pipeline_kube.yaml`: commit-addressed image build, logical TPU
  profiles, Kubernetes Jobs, parity test matrix, coverage aggregation, and
  final result validation.
- `docker/Dockerfile`: a source location not shadowed by Agent Stack's
  `/workspace` mount, plus compatibility links for bare metal.
- `.buildkite/scripts/setup_docker_env.sh`: opt-in Docker build-cache retention,
  commit image aliasing, and optional post-push image retention.
- benchmark scripts: explicit TPU Inference and vLLM path discovery.
- `kubernetes/kueue-poc`: controller and worker Kueue/MultiKueue resources,
  reservation-backed ResourceFlavors, and operating documentation.

## Decision gates

Do not promote this POC to the default CI path until the following are met:

1. Run multiple same-commit, same-test comparisons with controlled cache state;
   record queue, admission, provisioning, image pull, setup, compilation, test,
   and cleanup time separately.
2. Replace or explicitly accept the Agent Stack/Kueue readiness workaround.
3. Demonstrate cancellation and cleanup for Buildkite Job, manager Workload,
   worker Job/Pod, PVC, and TPU node.
4. Demonstrate quota/fairness under overlapping builds without oversubscribing
   the reservation or starving eight-chip work.
5. Define secret distribution and rotation for every worker cluster.
6. Implement and measure a cache design that survives Jobs without allowing
   untrusted PRs to overwrite a shared golden cache.
7. Quantify steady-state GKE, persistent disk, Artifact Registry, egress, and
   operational costs against the bare-metal baseline.
8. Prove a second worker location or document that MultiKueue currently adds
   control-plane complexity without regional failover value.

The decision after those gates should compare three choices: retain bare metal,
adopt a hybrid model, or migrate the selected workloads to Kubernetes.
