# Buildkite Kubernetes POC handoff

Last reviewed: 2026-07-25.

This is the operational handoff for the current experiment. Start with
[`README.md`](README.md) to install or reproduce it. Read [`DESIGN.md`](DESIGN.md)
for the architecture and tradeoffs, and the repository-level
[`../../KUBERNETES_POC.md`](../../KUBERNETES_POC.md) for results and decision
gates.

One investigation is tracked separately because it is unresolved and has its
own reproduction: cancelling a build leaves queued Kubernetes Jobs behind, and
they later consume a TPU node before exiting. See
[`CANCELLATION_HANDOFF.md`](CANCELLATION_HANDOFF.md).

## Current state

- Controller: GKE Autopilot cluster `ci-test-controller`, region `us-central1`.
- Worker: GKE Autopilot cluster `ci-test-southamerica-west1-worker`, region
  `southamerica-west1`, TPU work pinned to zone `southamerica-west1-a`.
- Namespace on both clusters: `buildkite`.
- Buildkite Agent Stack: `v0.46.2`, installed on the controller only.
- Kueue/MultiKueue: `v0.19.0`, installed on controller and worker.
- Buildkite agent queue: `kube`.
- CPU image-build queue: `cpu` (the existing worker has 64 cores; there is no
  `cpu_64_core` queue for this POC pipeline).
- Kueue LocalQueues and ClusterQueues: `v6e-1` and `v6e-8`.
- Both ClusterQueues are members of cohort `v6e-pool`.
- Worker reservation reference:
  `cloudtpu-20250327121505-861300654`; verify this against the live project.
- Cache: empty 500 GiB Job-local `premium-rwo` volume, deleted with the Job.
- No golden cache and no CPU preparation step are active.
- Multi-host and v7x tests are out of scope.

## What has been proven

Builds 77 and 78 are the strongest completed results. Each passed all 15 active
TPU Jobs and the final result-validation step. This verifies the tested
Agent Stack/MultiKueue path through multi-hour execution without the earlier
secret, admission, or exit-137 lifecycle failures.

Build 76 passed 14 of 15 TPU Jobs. The remaining unit-test shard ran for
1h36m59 and failed normally on `test_mesh_devices_sorting` after 1,563 passing
and 340 skipped tests. This separated that test failure from the earlier
infrastructure exit 137; the source fix was present in builds 77–79.

Build 74 proved that Agent Stack and MultiKueue preserve the Kubernetes Job
active deadline override: setting it to 120 seconds killed the command after
approximately 1m32 of agent runtime. Build 75 proved that changing only the
manager readiness timeout is insufficient; the worker Kueue configuration must
match.

Builds 77–79 were started together as a scale/admission experiment after
removing Buildkite-side concurrency throttling. Builds 77 and 78 passed. At
2026-07-25 19:55 UTC, build 79 had 13 passing TPU Jobs while speculative
decoding and unit tests part 2 were still running. Its final outcome must still
be checked. The conditionally excluded rows did not execute and are not test
failures for this comparison.

The repeated command durations were stable, but median pre-agent wait increased
from 5m25s in build 77 to 37m14s in build 78 and 1h08m02s in build 79. This
shows admission respecting the limited TPU pools, but the current capacity is
not sufficient for low-latency overlapping full matrices. See
[`../../KUBERNETES_POC.md`](../../KUBERNETES_POC.md) for the complete timing and
bare-metal comparison.

## Primary known issue: Running but NonReady

Do not interpret `Running` plus `Ready=False` as proof that an Agent Stack test
has failed.

Agent Stack creates checkout, agent, and command as regular containers. Checkout
finishes successfully while the command continues. A terminated regular
container is not ready, so Kubernetes includes it in `ContainersNotReady` and
the Pod remains `Ready=False`.

Kueue's Job adapter then does not regard the Job as pods-ready during normal
execution. The default readiness timeout killed long tests. Both checked-in
Kueue ConfigMaps use 10800 seconds as a compatibility workaround.

Confirm the state with:

```bash
WORKER_CONTEXT=gke_cloud-tpu-inference-test_southamerica-west1_ci-test-southamerica-west1-worker
POD=REPLACE_WITH_POD_NAME

kubectl --context "$WORKER_CONTEXT" -n buildkite get pod "$POD" \
  -o jsonpath='{range .status.containerStatuses[*]}{.name}{" ready="}{.ready}{" state="}{.state}{"\n"}{end}'
kubectl --context "$WORKER_CONTEXT" -n buildkite describe pod "$POD"
kubectl --context "$WORKER_CONTEXT" -n buildkite logs "$POD" -c checkout
kubectl --context "$WORKER_CONTEXT" -n buildkite logs "$POD" -c agent --tail=100
kubectl --context "$WORKER_CONTEXT" -n buildkite logs "$POD" -c vllm-tpu-runner --tail=100
```

The command container name comes from `pipeline_kube.yaml`; verify it from the
Pod spec if Agent Stack changes generated naming.

## Root-cause diagnostic sequence

Set contexts:

```bash
MANAGER_CONTEXT=gke_cloud-tpu-inference-test_us-central1_ci-test-controller
WORKER_CONTEXT=gke_cloud-tpu-inference-test_southamerica-west1_ci-test-southamerica-west1-worker
```

### 1. Find the Buildkite Job and manager Workload

```bash
kubectl --context "$MANAGER_CONTEXT" -n buildkite get job,workload -o wide
kubectl --context "$MANAGER_CONTEXT" -n buildkite get job JOB_NAME -o yaml
kubectl --context "$MANAGER_CONTEXT" -n buildkite describe workload WORKLOAD_NAME
```

Check that the Job—not merely its Pod template—has one of these labels:

```text
kueue.x-k8s.io/queue-name: v6e-1
kueue.x-k8s.io/queue-name: v6e-8
```

Also check the generated Job's `activeDeadlineSeconds` and its Workload
admission conditions.

### 2. Follow MultiKueue dispatch

```bash
kubectl --context "$MANAGER_CONTEXT" get multikueuecluster \
  ci-test-southamerica-west1-worker -o yaml
kubectl --context "$MANAGER_CONTEXT" get admissioncheck \
  multikueue-dispatch -o yaml
kubectl --context "$MANAGER_CONTEXT" -n kueue-system logs \
  deployment/kueue-controller-manager --since=30m | \
  grep -E 'multikueue|admit|evict|PodsReady|JOB_NAME|WORKLOAD_NAME'
```

The last command uses `grep` only for portability on an operator workstation;
`rg` is equally suitable when installed.

### 3. Inspect the worker copy

```bash
kubectl --context "$WORKER_CONTEXT" -n buildkite get job,pod,pvc -o wide
kubectl --context "$WORKER_CONTEXT" -n buildkite describe job JOB_NAME
kubectl --context "$WORKER_CONTEXT" -n buildkite describe pod POD_NAME
kubectl --context "$WORKER_CONTEXT" -n buildkite get events \
  --sort-by=.lastTimestamp | tail -100
```

Classify the delay rather than treating all waiting as one number:

- manager queue wait;
- central admission wait;
- MultiKueue dispatch;
- worker admission;
- node provisioning;
- PVC binding;
- image pull;
- checkout/agent startup;
- model or dataset download;
- XLA compilation;
- test execution;
- cleanup.

### 4. Check queue and flavor capacity

```bash
kubectl --context "$MANAGER_CONTEXT" get clusterqueue v6e-1 v6e-8 -o yaml
kubectl --context "$WORKER_CONTEXT" get clusterqueue v6e-1 v6e-8 -o yaml
kubectl --context "$WORKER_CONTEXT" get resourceflavor v6e-1 v6e-8 -o yaml
kubectl --context "$WORKER_CONTEXT" get nodes \
  -L cloud.google.com/gke-tpu-accelerator,cloud.google.com/gke-tpu-topology,topology.kubernetes.io/zone,cloud.google.com/compute-class
```

Messages such as `resource memory unavailable in ClusterQueue` or `resource cpu
unavailable in ClusterQueue` mean the Job requests a covered resource without
sufficient declared ClusterQueue quota. They do not prove a GKE node shortage.
The checked-in queues cover TPU, CPU, memory, and ephemeral storage.

### 5. Check Kueue effective configuration

```bash
kubectl --context "$MANAGER_CONTEXT" -n kueue-system get configmap \
  kueue-manager-config -o yaml
kubectl --context "$WORKER_CONTEXT" -n kueue-system get configmap \
  kueue-manager-config -o yaml
kubectl --context "$MANAGER_CONTEXT" -n kueue-system rollout status \
  deployment/kueue-controller-manager
kubectl --context "$WORKER_CONTEXT" -n kueue-system rollout status \
  deployment/kueue-controller-manager
```

Both should show `manageJobsWithoutQueueName: false` and the same
`waitForPodsReady.timeout`. A ConfigMap update is not active until the
controller deployment has restarted successfully.

## Common failures

### `agent-stack-k8s-secret not found`

Cause: the manager Job references a Secret that MultiKueue does not copy.

POC recovery: create the equivalent Secret in the worker's `buildkite`
namespace from the approved source, then start a fresh Buildkite build. A Job
that already failed may not recover cleanly after the Secret appears.

Production fix: manage the Secret declaratively on every worker through an
external secret provider or synchronization controller, with rotation and
monitoring. Do not depend on manual `kubectl` copying.

### Job is reserved but no Pod runs

Inspect manager and worker Workload conditions independently. Possible causes
include worker connection failure, worker quota, no matching ResourceFlavor,
reservation exhaustion, TPU provisioning, PVC topology, or a missing Secret.

### Exit 137 around a fixed interval

Compare timestamps with all three mechanisms:

- Kubernetes Job `activeDeadlineSeconds`;
- Kueue `waitForPodsReady.timeout` on manager and worker;
- `kueue.x-k8s.io/max-exec-time-seconds` on the Job/Workload.

An exit code alone does not identify which controller initiated termination.
Use events and Kueue logs.

### Test is unexpectedly slow

First separate infrastructure wait from command runtime. Within command time,
inspect model/dataset download and XLA compilation messages. The current PVC is
cold and disposable, so eliminating bare-metal cleanup does not guarantee a
faster test. Build 57 showed a mixed result and compilation evidence strongly
suggested cold cache as the cause of the largest regression.

## Safe cleanup check

Cancel a disposable Buildkite build and verify all layers converge:

```bash
kubectl --context "$MANAGER_CONTEXT" -n buildkite get job,workload
kubectl --context "$WORKER_CONTEXT" -n buildkite get job,workload,pod,pvc
kubectl --context "$WORKER_CONTEXT" get nodes \
  -L cloud.google.com/gke-tpu-accelerator,cloud.google.com/gke-tpu-topology
```

Do not bulk-delete resources to make the output clean; identify ownership labels
and understand which controller failed to reconcile. Record whether any PVC or
TPU node survives after its Job disappears.

## Next investigations, in priority order

1. Determine final outcomes of builds 77–79 and quantify manager/worker
   admission behavior under 45 simultaneously reserved Jobs.
2. Evaluate a supported fix for the Agent Stack/Kueue readiness mismatch,
   including the temporary Kueue feature gate only as an experiment.
3. Repeat same-commit timings with explicit cold/warm cache states and phase
   instrumentation.
4. Exercise cancellation during each lifecycle phase and verify TPU/PVC cleanup.
5. Test overlapping one-chip and eight-chip builds against actual reservation
   limits; adjust nominal quota from intended capacity rather than guesses.
6. Add a second worker region when capacity is available and measure actual
   placement/failure behavior.
7. Prototype transparent CPU-side model/dataset preparation and safe
   golden-plus-PR cache reuse.
8. Design secret distribution, identity, rotation, and untrusted-PR policy.
9. Prototype change-based/tiered selection and identify CPU-safe tests without
   changing the parity baseline; account for possible Buildkite Test Engine
   adoption.

## POC acceptance position

The infrastructure path is ready for further controlled testing. It is not yet
ready to replace `pipeline_jax.yml`. Keep the existing bare-metal pipeline as
the comparison baseline and fallback until the decision gates in
`KUBERNETES_POC.md` are satisfied.
