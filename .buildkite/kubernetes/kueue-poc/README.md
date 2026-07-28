# MultiKueue TPU POC operator guide

This directory contains the manifests needed to reproduce the tested
Buildkite-to-GKE TPU path. It is an experiment, not a production installation
bundle. Read the [architecture and alternatives](DESIGN.md) before extending
it, and use [HANDOFF.md](HANDOFF.md) for current findings and diagnostics.
[CANCELLATION_HANDOFF.md](CANCELLATION_HANDOFF.md) is the open investigation
into cancelled builds leaving Kubernetes Jobs behind.
The production-oriented Standard GKE design is intentionally separate under
[`../production`](../production/README.md).

## Tested topology

| Role | Cluster/context | Location |
|---|---|---|
| Manager | `ci-test-controller` / `gke_cloud-tpu-inference-test_us-central1_ci-test-controller` | `us-central1` |
| Worker | `ci-test-southamerica-west1-worker` / `gke_cloud-tpu-inference-test_southamerica-west1_ci-test-southamerica-west1-worker` | `southamerica-west1`; TPU pinned to `southamerica-west1-a` |

Both are GKE Autopilot clusters registered in the same fleet. The POC used
Kueue `v0.19.0` and Buildkite Agent Stack for Kubernetes `v0.46.2`. Confirm
versions before applying configuration because the APIs and feature gates are
version-sensitive.

The Buildkite Agent Stack controller runs only in the manager cluster, in the
`buildkite` namespace. MultiKueue creates workload Jobs in the worker's
`buildkite` namespace.

## Files

```text
controller/kueue-manager-config.yaml       manager Kueue behavior
controller/manager-southamerica-west1.yaml manager queues and MultiKueue link
worker/kueue-manager-config.yaml           worker Kueue behavior
worker/multikueue-worker-access.yaml       worker service account and RBAC
worker/worker-southamerica-west1.yaml      worker flavors, quota, reservation
jobs/smoke-job.yaml                        direct Kueue/MultiKueue smoke test
```

## Before applying

You need `kubectl` access to both contexts and permission to create cluster-wide
Kueue resources. Confirm the expected objects and reservation:

```bash
kubectl --context gke_cloud-tpu-inference-test_us-central1_ci-test-controller \
  get deploy -n kueue-system kueue-controller-manager \
  -o jsonpath='{.metadata.labels.app\.kubernetes\.io/version}{"\n"}'

kubectl --context gke_cloud-tpu-inference-test_southamerica-west1_ci-test-southamerica-west1-worker \
  get deploy -n kueue-system kueue-controller-manager \
  -o jsonpath='{.metadata.labels.app\.kubernetes\.io/version}{"\n"}'

gcloud compute reservations describe cloudtpu-20250327121505-861300654 \
  --zone southamerica-west1-a \
  --project cloud-tpu-inference-test
```

The reservation name is infrastructure-specific. Verify it against the live
project and update both ComputeClasses in `worker-southamerica-west1.yaml` if it
has changed.

## Install order

Set shell variables only to keep the commands readable:

```bash
MANAGER_CONTEXT=gke_cloud-tpu-inference-test_us-central1_ci-test-controller
WORKER_CONTEXT=gke_cloud-tpu-inference-test_southamerica-west1_ci-test-southamerica-west1-worker
POC_DIR=.buildkite/kubernetes/kueue-poc
```

### 1. Apply worker access and generate its kubeconfig

```bash
kubectl --context "$WORKER_CONTEXT" apply \
  -f "$POC_DIR/worker/multikueue-worker-access.yaml"

WORKER_SERVER=$(kubectl --context "$WORKER_CONTEXT" \
  config view --minify -o jsonpath='{.clusters[0].cluster.server}')
WORKER_CA=$(kubectl --context "$WORKER_CONTEXT" \
  config view --raw --minify -o jsonpath='{.clusters[0].cluster.certificate-authority-data}')
WORKER_TOKEN=$(kubectl --context "$WORKER_CONTEXT" -n kueue-system \
  create token multikueue-sa)
```

Create a temporary kubeconfig containing that server, CA, token, and namespace
`buildkite`, then install it as the manager-side Secret expected by
`MultiKueueCluster`:

```bash
kubectl --context "$MANAGER_CONTEXT" -n kueue-system \
  create secret generic ci-test-southamerica-west1-worker-kubeconfig \
  --from-file=kubeconfig=/path/to/temporary/worker-kubeconfig \
  --dry-run=client -o yaml | \
kubectl --context "$MANAGER_CONTEXT" apply -f -
```

Do not commit the generated kubeconfig. Delete the temporary local file after
the Secret is created. For production, replace this static token flow with a
rotatable identity and secret-management design.

### 2. Apply Kueue manager configuration

```bash
kubectl --context "$MANAGER_CONTEXT" apply \
  -f "$POC_DIR/controller/kueue-manager-config.yaml"
kubectl --context "$WORKER_CONTEXT" apply \
  -f "$POC_DIR/worker/kueue-manager-config.yaml"

kubectl --context "$MANAGER_CONTEXT" -n kueue-system \
  rollout restart deployment/kueue-controller-manager
kubectl --context "$WORKER_CONTEXT" -n kueue-system \
  rollout restart deployment/kueue-controller-manager

kubectl --context "$MANAGER_CONTEXT" -n kueue-system \
  rollout status deployment/kueue-controller-manager
kubectl --context "$WORKER_CONTEXT" -n kueue-system \
  rollout status deployment/kueue-controller-manager
```

The checked-in three-hour `waitForPodsReady` timeout is a tested workaround for
Agent Stack's completed checkout container. It is not a production health
signal; see `DESIGN.md` before changing or accepting it.

### 3. Apply worker and manager resources

Apply worker resources first so dispatched Jobs have valid queues and flavors:

```bash
kubectl --context "$WORKER_CONTEXT" apply \
  -f "$POC_DIR/worker/worker-southamerica-west1.yaml"
kubectl --context "$MANAGER_CONTEXT" apply \
  -f "$POC_DIR/controller/manager-southamerica-west1.yaml"
```

### 4. Provision Agent Stack secrets on the worker

MultiKueue does not copy referenced Secrets. Confirm which Secrets a generated
Agent Stack Job uses in the manager, then create equivalent Secrets in the
worker's `buildkite` namespace using the approved secret source:

```bash
kubectl --context "$MANAGER_CONTEXT" -n buildkite get secrets
kubectl --context "$WORKER_CONTEXT" -n buildkite get secrets
kubectl --context "$WORKER_CONTEXT" -n buildkite \
  get secret agent-stack-k8s-secret
```

Do not use an ad hoc read-and-write pipeline for production secrets. The
production fix is declarative synchronization or an external secret provider.

### 5. Verify Kueue objects

```bash
kubectl --context "$MANAGER_CONTEXT" get multikueuecluster,multikueueconfig,admissioncheck
kubectl --context "$MANAGER_CONTEXT" get resourceflavor,clusterqueue
kubectl --context "$MANAGER_CONTEXT" -n buildkite get localqueue

kubectl --context "$WORKER_CONTEXT" get resourceflavor,clusterqueue
kubectl --context "$WORKER_CONTEXT" -n buildkite get localqueue
```

Expected LocalQueues are `v6e-1` and `v6e-8`, both in namespace `buildkite`.
The older `default` queue is not used by the pipeline and can be investigated
separately before removal.

### 6. Run a direct smoke Job

```bash
kubectl --context "$MANAGER_CONTEXT" apply -f "$POC_DIR/jobs/smoke-job.yaml"
kubectl --context "$MANAGER_CONTEXT" -n buildkite get job,workload -w
```

In another terminal:

```bash
kubectl --context "$WORKER_CONTEXT" -n buildkite get job,pod,pvc -w
```

Delete the smoke Job after recording the result:

```bash
kubectl --context "$MANAGER_CONTEXT" -n buildkite delete job kueue-smoke
```

### 7. Install or verify Agent Stack

The Agent Stack controller belongs in the manager's `buildkite` namespace and
registers Buildkite agent queue `kube`. Its generated Jobs must remain in that
namespace so they can resolve `buildkite/v6e-1` and `buildkite/v6e-8`.

Verify the deployment and inspect a generated Job before starting the full
matrix:

```bash
kubectl --context "$MANAGER_CONTEXT" -n buildkite get deploy,pod
kubectl --context "$MANAGER_CONTEXT" -n buildkite get job \
  -l kueue.x-k8s.io/queue-name -o wide
```

### 8. Exercise the Buildkite pipeline

Configure the `kube-dev` pipeline bootstrap to upload
`.buildkite/pipeline_kube.yaml` on queue `cpu`. Start with the default matrix.
Use `KUBE_FULL_MATRIX=1` only after one-chip admission and cleanup are healthy;
it includes additional/nightly and eight-chip work.

For a targeted eight-chip check, set:

```text
KUBE_TARGET_STEP=kube_e2e_mlperf_jax_vllm_multi_chip
```

## Quick health checks

Manager:

```bash
kubectl --context "$MANAGER_CONTEXT" get multikueuecluster \
  ci-test-southamerica-west1-worker -o yaml
kubectl --context "$MANAGER_CONTEXT" -n buildkite get workload \
  -o custom-columns=NAME:.metadata.name,QUEUE:.spec.queueName,ADMITTED:.status.conditions[-1].type,REASON:.status.conditions[-1].reason
kubectl --context "$MANAGER_CONTEXT" -n buildkite get job -o wide
```

Worker:

```bash
kubectl --context "$WORKER_CONTEXT" -n buildkite get job,pod,pvc -o wide
kubectl --context "$WORKER_CONTEXT" -n buildkite describe pod POD_NAME
kubectl --context "$WORKER_CONTEXT" get nodes \
  -L cloud.google.com/gke-tpu-accelerator,cloud.google.com/gke-tpu-topology,topology.kubernetes.io/zone,cloud.google.com/compute-class
```

A running Agent Stack Pod can show `Running` and `Ready=False` because its
checkout container completed successfully. Inspect individual container states
and command logs before treating that condition as a failed test.
