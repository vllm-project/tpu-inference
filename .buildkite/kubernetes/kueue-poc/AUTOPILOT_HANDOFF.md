# Autopilot MultiKueue handoff

This is the executable handoff for two existing fleet-registered GKE Autopilot
clusters:

- manager/controller: `ci-test-controller`
- manager/controller region: `us-central1`
- first worker: `ci-test-southamerica-west1-worker`
- worker region/TPU zone: `southamerica-west1` / `southamerica-west1-a`
- intended TPU reservation: `cloudtpu-20250327121501-861300654`

The goal of this phase is intentionally narrow: prove that a suspended
`batch/v1 Job` submitted to the controller is admitted by Kueue, copied by
MultiKueue, run exactly once on the South America worker, and reconciled back to
the controller. Do not install Agent Stack or run the TPU test matrix until
that path passes.

Autopilot has a regional control plane. That is acceptable. A TPU Pod and a
zonal Persistent Disk still run in one zone. The worker ResourceFlavor pins
`v6e-1` to `southamerica-west1-a` through
`topology.kubernetes.io/zone`; Kueue applies that label when it admits the
Workload. The first storage test uses a generic ephemeral PVC created by that
same TPU Pod, so `WaitForFirstConsumer` provisions the disk in `a`. It does not
create a PVC in an earlier CPU-only Job.

## Working rules

- Do not operate on the existing `ci-dev` cluster or Buildkite `kube` queue.
- Use explicit `--context` on every `kubectl` command.
- Pin one Kueue release and use the same version on both clusters.
- Do not use an unversioned installation URL.
- Start with `nominalQuota: 1` and one short v6e-1 smoke Job.
- Save the YAML and events for both clusters at every gate.
- Stop at the first failed gate. Do not work around admission or RBAC by granting
  `cluster-admin` to a workload identity.
- Fleet registration does not automatically give MultiKueue access to the
  worker API and does not replicate Secrets, ConfigMaps, PVCs, or images.

## Phase 0: resolve locations and contexts

Set project and location inputs. The controller region is deliberately an
input because the cluster name alone does not reveal it.

```bash
export POC_PROJECT_ID="REPLACE_PROJECT"
export POC_CONTROLLER_REGION="us-central1"
export POC_WORKER_REGION="southamerica-west1"
export POC_CONTROLLER_CLUSTER="ci-test-controller"
export POC_WORKER_CLUSTER="ci-test-southamerica-west1-worker"
export POC_TPU_ZONE="southamerica-west1-a"
export POC_TPU_RESERVATION="cloudtpu-20250327121501-861300654"
```

Confirm both clusters and fleet memberships before changing anything:

```bash
gcloud container clusters list \
  --project "$POC_PROJECT_ID" \
  --filter="name=($POC_CONTROLLER_CLUSTER $POC_WORKER_CLUSTER)" \
  --format="table(name,location,autopilot.enabled,currentMasterVersion,status)"

gcloud container fleet memberships list \
  --project "$POC_PROJECT_ID"
```

Acquire credentials and save stable context names for the rest of the run:

```bash
gcloud container clusters get-credentials "$POC_CONTROLLER_CLUSTER" \
  --region "$POC_CONTROLLER_REGION" \
  --project "$POC_PROJECT_ID"
export POC_CONTROLLER_CONTEXT="$(kubectl config current-context)"

gcloud container clusters get-credentials "$POC_WORKER_CLUSTER" \
  --region "$POC_WORKER_REGION" \
  --project "$POC_PROJECT_ID"
export POC_WORKER_CONTEXT="$(kubectl config current-context)"
```

Print and manually verify the values. They must be different:

```bash
printf 'controller=%s\nworker=%s\n' \
  "$POC_CONTROLLER_CONTEXT" "$POC_WORKER_CONTEXT"

kubectl --context "$POC_CONTROLLER_CONTEXT" cluster-info
kubectl --context "$POC_WORKER_CONTEXT" cluster-info
```

Confirm the operator may install Kueue's CRDs and admission webhooks:

```bash
kubectl --context "$POC_CONTROLLER_CONTEXT" auth can-i create customresourcedefinitions.apiextensions.k8s.io
kubectl --context "$POC_CONTROLLER_CONTEXT" auth can-i create mutatingwebhookconfigurations.admissionregistration.k8s.io
kubectl --context "$POC_WORKER_CONTEXT" auth can-i create customresourcedefinitions.apiextensions.k8s.io
kubectl --context "$POC_WORKER_CONTEXT" auth can-i create mutatingwebhookconfigurations.admissionregistration.k8s.io
```

Gate: both contexts are correct, `ci-test-controller` reports location
`us-central1`, both clusters are `RUNNING` Autopilot clusters, and all four
authorization checks return `yes`.

## Phase 1: select and install Kueue

Select a released Kueue version after checking its official compatibility
documentation against both reported GKE minor versions. The chosen release must
support:

- `batch/v1 Job` integration;
- MultiKueue with `MultiKueueCluster`, `MultiKueueConfig`, and
  `AdmissionCheck`;
- the `kueue.x-k8s.io/v1beta1` resources in this directory, or the manifests
  must be updated to the version actually served;
- GKE Autopilot admission/security constraints; and
- PodsReady timeout/requeue behavior for later capacity-failure testing.

Record the decision instead of substituting `latest`:

```bash
export POC_KUEUE_VERSION="vREPLACE_PINNED_VERSION"
export POC_KUEUE_MANIFEST="https://github.com/kubernetes-sigs/kueue/releases/download/${POC_KUEUE_VERSION}/manifests.yaml"
```

Install the exact same manifest on both clusters:

```bash
kubectl --context "$POC_CONTROLLER_CONTEXT" apply --server-side \
  -f "$POC_KUEUE_MANIFEST"

kubectl --context "$POC_WORKER_CONTEXT" apply --server-side \
  -f "$POC_KUEUE_MANIFEST"
```

Wait for the controller deployment. If the pinned release uses a different
namespace or deployment name, discover it from its manifest and update these
commands; do not guess around a failed wait.

```bash
kubectl --context "$POC_CONTROLLER_CONTEXT" -n kueue-system \
  rollout status deployment/kueue-controller-manager --timeout=5m

kubectl --context "$POC_WORKER_CONTEXT" -n kueue-system \
  rollout status deployment/kueue-controller-manager --timeout=5m
```

Verify APIs and inspect any Autopilot admission adjustments:

```bash
kubectl --context "$POC_CONTROLLER_CONTEXT" api-resources | \
  rg 'clusterqueue|localqueue|workload|multikueue|admissioncheck'

kubectl --context "$POC_WORKER_CONTEXT" api-resources | \
  rg 'clusterqueue|localqueue|workload'

kubectl --context "$POC_CONTROLLER_CONTEXT" -n kueue-system \
  get deployment,pod,event

kubectl --context "$POC_WORKER_CONTEXT" -n kueue-system \
  get deployment,pod,event
```

Gate: both Kueue controllers are Available; all required resource kinds are
served; no webhook, Pod Security, resource-request, or Autopilot rejection is
present. Save controller logs before continuing if either cluster is unhealthy.

## Phase 1.5: prove how Autopilot will consume the TPU reservation

Complete this gate before creating the worker ResourceFlavor or any TPU Job.
Kueue quota and ResourceFlavor do not themselves claim a Compute Engine/Cloud
TPU reservation. A ResourceFlavor can select Kubernetes node labels, including
a supported ComputeClass selector, but the reservation name is not assumed to
be a node label.

Inspect the reservation and save the full result:

```bash
gcloud compute reservations describe "$POC_TPU_RESERVATION" \
  --project "$POC_PROJECT_ID" \
  --zone "$POC_TPU_ZONE" \
  --format=yaml \
  > /tmp/tpu-reservation-before.yaml

sed -n '1,240p' /tmp/tpu-reservation-before.yaml
```

Record at least:

- exact project and zone;
- status;
- `specificReservationRequired`;
- reserved resource/machine properties and count;
- current `inUseCount`;
- sharing policy; and
- whether its shape matches an Autopilot v6e-1 placement.

Then inspect the ComputeClass API served by the actual worker cluster. Do not
copy a ComputeClass example written for a different GKE minor version:

```bash
kubectl --context "$POC_WORKER_CONTEXT" api-resources | rg 'computeclass'
kubectl --context "$POC_WORKER_CONTEXT" get computeclass -o yaml
kubectl --context "$POC_WORKER_CONTEXT" explain computeclass.spec \
  --api-version=cloud.google.com/v1
kubectl --context "$POC_WORKER_CONTEXT" explain computeclass.spec.priorities \
  --api-version=cloud.google.com/v1 --recursive
```

Choose exactly one supported mode and record it:

1. **Automatic matching reservation.** If the reservation can be consumed by
   any matching instance and Autopilot documents/supports that behavior, keep
   the worker ResourceFlavor's TPU/topology/zone labels. The smoke must still
   prove that reservation `inUseCount` increases; Pod success alone is not
   evidence because Autopilot might allocate on-demand capacity.
2. **Specific reservation through ComputeClass.** If this GKE version exposes a
   supported reservation field in ComputeClass for Autopilot TPU workloads,
   create a worker-local ComputeClass referring to
   `cloudtpu-20250327121501-861300654`. Add its documented Pod selector label to
   the worker `v6e-1` ResourceFlavor. Keep the manager ResourceFlavor generic.
   Server-side dry-run the ComputeClass and ResourceFlavor before applying.
3. **Specific reservation through another documented Autopilot selector.** Put
   that selector in the worker ResourceFlavor, not in `pipeline_kube.yaml`, and
   server-side dry-run it before applying.
4. **Unsupported.** If the reservation requires explicit affinity but this
   Autopilot/GKE version offers no supported way to request it for TPU Pods,
   stop. Do not run a potentially on-demand TPU smoke. Use a Standard worker
   node pool configured with specific reservation affinity, or change the
   reservation consumption policy through the infrastructure owner.

Do not add `compute.googleapis.com/reservation-name` to ResourceFlavor merely
because that string appears in Compute Engine reservation-affinity APIs. Those
APIs use it as a NodeConfig affinity key; that does not prove it is a schedulable
Kubernetes node label on Autopilot.

Gate: reservation shape/policy is recorded and one supported consumption mode
is selected. If mode 2 or 3 is selected, update
`worker-southamerica-west1.yaml` with the verified worker-only selector before
Phase 2. If the result is unclear, stop and attach the reservation YAML and
ComputeClass schema to the handoff report.

## Phase 2: configure the worker queue

Apply the one-chip worker resources and the POC-only manager access identity:

```bash
kubectl --context "$POC_WORKER_CONTEXT" apply \
  -f .buildkite/kubernetes/kueue-poc/worker-southamerica-west1.yaml

kubectl --context "$POC_WORKER_CONTEXT" apply \
  -f .buildkite/kubernetes/kueue-poc/multikueue-worker-access.yaml
```

If the pinned Kueue release provides an official MultiKueue worker Role and
kubeconfig-generation helper, prefer that helper. Compare its permissions with
`multikueue-worker-access.yaml`; do not apply both bindings blindly.

Verify the queue and namespace-scoped access:

```bash
kubectl --context "$POC_WORKER_CONTEXT" get resourceflavor,clusterqueue
kubectl --context "$POC_WORKER_CONTEXT" -n buildkite-v6e-1 get localqueue,serviceaccount,role,rolebinding

kubectl --context "$POC_WORKER_CONTEXT" -n buildkite-v6e-1 \
  auth can-i create workloads.kueue.x-k8s.io \
  --as=system:serviceaccount:buildkite-v6e-1:multikueue-manager

kubectl --context "$POC_WORKER_CONTEXT" -n buildkite-v6e-1 \
  auth can-i create jobs.batch \
  --as=system:serviceaccount:buildkite-v6e-1:multikueue-manager
```

Gate: ClusterQueue `v6e-1` is Active, LocalQueue `v6e-1` is Active, and both
impersonated authorization checks return `yes`. Quota remains one chip. Also
verify that the worker ResourceFlavor contains all three labels and that
`southamerica-west1-a` actually supports v6e-1:

```bash
kubectl --context "$POC_WORKER_CONTEXT" get resourceflavor v6e-1 -o yaml
```

The manager ResourceFlavor deliberately has no zone label. A physical worker's
ResourceFlavor owns its local zone; the manager and Buildkite pipeline stay
region-neutral. When another regional worker is added, its local `v6e-1`
ResourceFlavor can use that worker's TPU zone under the same logical queue.
Any verified worker-only reservation/ComputeClass selector belongs in this
ResourceFlavor too; it never belongs in the manager flavor or Buildkite step.

## Phase 3: give the controller scoped worker credentials

Fleet membership alone is insufficient. MultiKueue expects a kubeconfig Secret
on the controller. For this POC, create a bounded service-account token. This is
not the production identity design and must be rotated before it expires.

Create a temporary directory and obtain worker API connection data:

```bash
export POC_AUTH_DIR="$(mktemp -d)"
export POC_WORKER_SERVER="$(kubectl config view --raw --minify \
  --context "$POC_WORKER_CONTEXT" \
  -o jsonpath='{.clusters[0].cluster.server}')"

kubectl config view --raw --minify \
  --context "$POC_WORKER_CONTEXT" \
  -o jsonpath='{.clusters[0].cluster.certificate-authority-data}' | \
  base64 --decode > "$POC_AUTH_DIR/worker-ca.crt"

export POC_WORKER_TOKEN="$(kubectl --context "$POC_WORKER_CONTEXT" \
  -n buildkite-v6e-1 create token multikueue-manager --duration=24h)"
```

Build a minimal kubeconfig containing no developer `exec` plugin:

```bash
kubectl config --kubeconfig="$POC_AUTH_DIR/worker.kubeconfig" \
  set-cluster worker \
  --server="$POC_WORKER_SERVER" \
  --certificate-authority="$POC_AUTH_DIR/worker-ca.crt" \
  --embed-certs=true

kubectl config --kubeconfig="$POC_AUTH_DIR/worker.kubeconfig" \
  set-credentials multikueue-manager --token="$POC_WORKER_TOKEN"

kubectl config --kubeconfig="$POC_AUTH_DIR/worker.kubeconfig" \
  set-context worker \
  --cluster=worker \
  --user=multikueue-manager \
  --namespace=buildkite-v6e-1

kubectl config --kubeconfig="$POC_AUTH_DIR/worker.kubeconfig" \
  use-context worker
```

Test the kubeconfig directly before storing it:

```bash
kubectl --kubeconfig="$POC_AUTH_DIR/worker.kubeconfig" \
  auth can-i get workloads.kueue.x-k8s.io

kubectl --kubeconfig="$POC_AUTH_DIR/worker.kubeconfig" \
  auth can-i create jobs.batch
```

Both must return `yes`. Then create or update the controller Secret:

```bash
kubectl --context "$POC_CONTROLLER_CONTEXT" -n kueue-system \
  create secret generic ci-test-southamerica-west1-worker-kubeconfig \
  --from-file=kubeconfig="$POC_AUTH_DIR/worker.kubeconfig" \
  --dry-run=client -o yaml | \
kubectl --context "$POC_CONTROLLER_CONTEXT" apply -f -
```

Do not commit the kubeconfig, token, Secret YAML, or temporary directory. Record
the token expiry in the POC notes and delete the local directory after the
MultiKueue connection becomes Active.

Gate: the minimal kubeconfig can access only the intended worker namespace and
the Secret exists in the manager Kueue controller namespace.

## Phase 4: configure one-worker MultiKueue

Apply the manager resources prepared for the actual worker name:

```bash
kubectl --context "$POC_CONTROLLER_CONTEXT" apply \
  -f .buildkite/kubernetes/kueue-poc/manager-southamerica-west1.yaml
```

Inspect status and controller logs:

```bash
kubectl --context "$POC_CONTROLLER_CONTEXT" get multikueuecluster
kubectl --context "$POC_CONTROLLER_CONTEXT" describe \
  multikueuecluster ci-test-southamerica-west1-worker
kubectl --context "$POC_CONTROLLER_CONTEXT" describe \
  admissioncheck multikueue-dispatch
kubectl --context "$POC_CONTROLLER_CONTEXT" get clusterqueue v6e-1
kubectl --context "$POC_CONTROLLER_CONTEXT" -n buildkite-v6e-1 \
  get localqueue v6e-1
```

If the worker does not become Active, inspect manager Kueue logs and check:

- Secret name and namespace;
- token expiry;
- worker API reachability from the controller cluster;
- certificate/server values in the kubeconfig;
- worker Role permissions; and
- whether the pinned Kueue release requires additional worker permissions.

Do not widen RBAC until the denied API and verb appear in an error or audit log.

Gate: `ci-test-southamerica-west1-worker` and admission check
`multikueue-dispatch` report Active, and both manager queues are Active.

## Phase 5: run a direct Kubernetes smoke Job

Do not involve Buildkite yet. Replace the placeholder image in `smoke-job.yaml`
with an immutable image digest available to the worker and known to contain
`/bin/sh`. The current v6e POC image is appropriate if Autopilot can pull it.
Do not commit the resolved private image if repository policy prohibits it.

Create the Job on the controller only. The manifest uses `generateName`, so it
must be created rather than applied:

```bash
kubectl --context "$POC_CONTROLLER_CONTEXT" create \
  -f .buildkite/kubernetes/kueue-poc/smoke-job.yaml
```

Watch these in separate terminals:

```bash
kubectl --context "$POC_CONTROLLER_CONTEXT" -n buildkite-v6e-1 \
  get job,workload,pod -w

kubectl --context "$POC_WORKER_CONTEXT" -n buildkite-v6e-1 \
  get job,workload,pod,pvc -w
```

The expected sequence is:

1. the manager Job remains suspended and no manager Pod starts;
2. manager Kueue creates a Workload and reserves manager quota;
3. MultiKueue creates the worker Workload;
4. worker Kueue reserves its one-chip quota;
5. exactly one remote Job and Pod start on the worker;
6. Autopilot provisions a v6e-1 TPU placement if one is not warm;
7. the command completes once;
8. remote completion is reflected on the manager Job.

While the worker Pod is running, query the reservation again:

```bash
gcloud compute reservations describe "$POC_TPU_RESERVATION" \
  --project "$POC_PROJECT_ID" \
  --zone "$POC_TPU_ZONE" \
  --format=yaml \
  > /tmp/tpu-reservation-during.yaml

diff -u /tmp/tpu-reservation-before.yaml /tmp/tpu-reservation-during.yaml || true
```

The reserved usage must increase consistently with the provisioned TPU
resource. If the Pod runs but reservation usage does not change, mark the gate
failed: the workload probably used on-demand capacity or a different
reservation. Capture the selected node YAML and worker events before cleanup.

Collect evidence before deleting anything:

```bash
kubectl --context "$POC_CONTROLLER_CONTEXT" -n buildkite-v6e-1 \
  get job,workload -o yaml > /tmp/multikueue-manager-result.yaml

kubectl --context "$POC_WORKER_CONTEXT" -n buildkite-v6e-1 \
  get job,workload,pod -o yaml > /tmp/multikueue-worker-result.yaml

kubectl --context "$POC_CONTROLLER_CONTEXT" -n buildkite-v6e-1 \
  get events --sort-by=.metadata.creationTimestamp \
  > /tmp/multikueue-manager-events.txt

kubectl --context "$POC_WORKER_CONTEXT" -n buildkite-v6e-1 \
  get events --sort-by=.metadata.creationTimestamp \
  > /tmp/multikueue-worker-events.txt
```

Stop here if a manager Pod starts, the remote command runs twice, completion is
not mirrored, or cancellation leaves the worker Job running.

Gate: one remote execution completes, status returns to the manager,
reservation consumption is demonstrated, and deleting/cancelling the manager
Job removes the remote copy.

## Phase 6: test Autopilot storage in the same TPU Job

After the no-volume smoke passes, add a generic ephemeral volume to the Job's
Pod template. Begin without `dataSource`:

```yaml
volumeMounts:
  - name: cache-volume
    mountPath: /cache
volumes:
  - name: cache-volume
    ephemeral:
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          storageClassName: premium-rwo
          resources:
            requests:
              storage: 500Gi
```

First verify the StorageClass binding behavior:

```bash
kubectl --context "$POC_WORKER_CONTEXT" get storageclass premium-rwo \
  -o jsonpath='{.volumeBindingMode}{"\n"}'
```

Expected: `WaitForFirstConsumer`. The remote TPU Pod is then the first consumer.
Kueue's worker ResourceFlavor pins the Pod to `southamerica-west1-a`, and the
disk is provisioned in that same zone. Confirm the admitted Workload's flavor
assignment, Pod scheduling constraints, PV node affinity, and selected node's
`topology.kubernetes.io/zone` label.

For this POC, cache/model/dataset hydration may run at the beginning of the
same Buildkite command, and cache-delta publication may run before it exits:

```text
TPU Job starts
  -> hydrate GCS cache/model/dataset into /cache
  -> run the TPU test
  -> publish successful cache delta and artifacts
  -> exit; generic ephemeral PVC is garbage-collected
```

This holds the TPU during hydration/publication but removes cross-step,
cross-cluster, and cross-zone PVC coordination. Emit explicit timers for all
three phases. Keep both cache variables pointing at `/cache/tpu_jax_cache`.

Only after this passes should the team test cloning from a golden PVC. A zonal
golden constrains the Job to the golden disk's zone; verify that the requested
TPU type is actually provisionable there.

Gate: PVC and TPU Pod bind in the same zone, `/cache` is writable, cleanup
removes the generic ephemeral claim, and all non-test time is measured.

## Phase 7: attach Buildkite Agent Stack

The intended steady-state Buildkite queue is the existing `kube` queue. That
queue is already associated with the original `ci-dev` Agent Stack. Never let
the old and new controllers consume it simultaneously: Buildkite may let either
controller claim a job before Kueue is involved.

Choose one migration-safe test mode:

1. **Isolated test queue (recommended):** create `kueue-poc`, configure the new
   Agent Stack to consume it, complete the lifecycle tests, then drain it and
   cut over the new stack to `kube`; or
2. **Direct cutover:** scale the old `ci-dev` Agent Stack controller to zero,
   confirm no queued/running `kube` jobs, and configure the new controller to
   consume `kube`.

Install Agent Stack only on `ci-test-controller`, configured to create
Kubernetes Jobs in manager namespace `buildkite-v6e-1`. Do not install an Agent
Stack controller on the worker.

### Exact queue and profile contract

There are three independent selections; do not conflate their names:

| Selection | Current value | Selected by | Purpose |
| --- | --- | --- | --- |
| Buildkite queue | `kube` | `agents.queue` | Sends a Buildkite command job to the one Agent Stack controller. |
| Kueue LocalQueue | `v6e-1` (internal POC name) | top-level Kubernetes Job queue label or LocalQueue defaulting | Maps the namespaced Job to the manager ClusterQueue and MultiKueue admission policy. |
| TPU profile | `v6e-1` | infrastructure-owned PodSpec resource request and selectors | Makes only the compatible worker ResourceFlavor eligible. |

For the current repository, choosing a profile means choosing a shared plugin
anchor, not writing a free-form Pod label:

```yaml
# One chip
plugins:
  - *kube_tpu_v6e_1chip_plugin

# Eight chips
plugins:
  - *kube_tpu_v6e_8chip_plugin
```

Those anchors render the actual scheduling contract. For example, the one-chip
profile produces:

```yaml
nodeSelector:
  cloud.google.com/gke-tpu-accelerator: tpu-v6e-slice
  cloud.google.com/gke-tpu-topology: 1x1
resources:
  requests:
    google.com/tpu: "1"
  limits:
    google.com/tpu: "1"
```

Kueue compares those hard requirements with candidate ResourceFlavors. On the
South America worker, flavor `v6e-1` adds the local zone and any verified
reservation/ComputeClass selector. A flavor whose topology conflicts with the
PodSpec is not eligible. A neutral label such as `ci.example/profile=v6e-1`
could be added for reporting, but Kueue does not interpret arbitrary profile
labels unless a custom admission controller is explicitly written to do so.

Feature engineers should eventually choose `profile: v6e-1` in a test registry
or helper API. A shared pipeline generator then emits the plugin anchor/PodSpec.
Do not put an unsupported `profile` key directly on a Buildkite step; Buildkite
does not know that field. Until that generator exists, selecting the shared
anchor is the concrete interface.

The current MultiKueue POC resources expose only one-chip quota and the `1x1`
flavor. A step using `kube_tpu_v6e_8chip_plugin` will therefore remain
inadmissible. Add and test an eight-chip worker flavor/quota on both manager and
worker before routing eight-chip steps through this stack.

### How Kueue picks up an Agent Stack Job

Kueue manages a `batch/v1 Job` only when it can resolve a LocalQueue. The normal
explicit signal is this top-level Job label:

```yaml
metadata:
  labels:
    kueue.x-k8s.io/queue-name: v6e-1
```

Without that label and without LocalQueue defaulting, Kueue ignores the Job and
the normal Job controller may create a manager-cluster Pod. MultiKueue never
sees it.

To hide Kueue from pipeline authors, enable the pinned Kueue release's
`LocalQueueDefaulting` feature gate and mark `v6e-1` as the default LocalQueue
for namespace `buildkite-v6e-1`:

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: v6e-1
  namespace: buildkite-v6e-1
  labels:
    kueue.x-k8s.io/default-queue: "true"
spec:
  clusterQueue: v6e-1
```

The checked-in manager manifests include this label. Confirm that the pinned
release serves the feature and that it is enabled in the Kueue controller
configuration; its default state has changed across releases. If it is not
available, use one of the explicit-label fallbacks below. The Kueue admission
webhook must add/resolve the queue and suspend the Job during the original API
admission request. Do not assume that the mere existence of one LocalQueue
makes it default.

Prove defaulting before installing/enabling Agent Stack:

1. create an **unsuspended** Job in `buildkite-v6e-1` with no
   `kueue.x-k8s.io/queue-name` label and a TPU request larger than the current
   one-chip quota, so successful Kueue pickup cannot dispatch it;
2. read the stored Job and confirm the queue label was applied and
   `spec.suspend` became `true`;
3. confirm Kueue created a Workload referencing LocalQueue `v6e-1`;
4. confirm no Pod exists on the manager; and
5. delete the test Job and confirm its Workload disappears.

This is the expected stored state:

```yaml
metadata:
  labels:
    kueue.x-k8s.io/queue-name: v6e-1
spec:
  suspend: true
```

If defaulting is unavailable or fails, configure Agent Stack to put the fixed
label on the top-level Job, or use a namespace-scoped mutating admission rule.
Do not continue with an unlabeled Agent Stack Job.

The existing `podSpec.metadata.labels` labels the Pod template and must not be
assumed to label the parent Job. Prove the bridge with an unlabeled, harmless
Job before enabling the Buildkite queue. The Job must be mutated to the queue
and suspended before any Pod is created.

Do not use a Buildkite agent tag as the label bridge. Buildkite tags constrain
agent matching and Agent Stack does not inherently translate them into
Kubernetes Job metadata. For the single-stack POC, namespace LocalQueue
defaulting is the simplest hidden mapping. If multiple internal LocalQueues are
introduced later for policy isolation, use a documented Agent Stack Job-metadata
field or a namespace-scoped admission mutation driven by a neutral logical
profile annotation.

Trigger one Buildkite smoke, then immediately capture its generated Job:

```bash
kubectl --context "$POC_CONTROLLER_CONTEXT" -n buildkite-v6e-1 \
  get jobs --sort-by=.metadata.creationTimestamp

kubectl --context "$POC_CONTROLLER_CONTEXT" -n buildkite-v6e-1 \
  get job REPLACE_JOB -o yaml > /tmp/agent-stack-generated-job.yaml
```

Inspect all referenced dependencies:

```bash
rg 'secretKeyRef|configMapKeyRef|secretName|configMap|persistentVolumeClaim|serviceAccountName' \
  /tmp/agent-stack-generated-job.yaml
```

Every referenced Secret, ConfigMap, service account, PVC, and image must either
be present under the same logical name on the worker or be embedded in a safe,
portable way. A per-job object that exists only on the controller is a hard
blocker because MultiKueue does not copy it.

Set `KUEUE_SMOKE_IMAGE` to the tested immutable digest and upload
`buildkite-smoke.yaml`. Validate success, nonzero exit status, timeout,
cancellation while queued, and cancellation while running.

Gate: Buildkite receives exactly one command result and log stream; its timeout
and cancellation delete manager and worker resources; Agent Stack tolerates the
manager Job remaining suspended during Kueue/MultiKueue wait.

## Phase 8: quota and real-capacity behavior

Keep worker and manager `nominalQuota` at one during the preceding gates.
LocalQueue has no independent quota; it points to ClusterQueue. Kueue
automatically calculates available logical quota as configured quota minus
admitted reservations, but it does not subscribe to GCP TPU reservations or
discover live Autopilot capacity.

After basic dispatch passes, check whether the installed GKE/Kueue combination
exposes provisioning admission APIs:

```bash
kubectl --context "$POC_WORKER_CONTEXT" api-resources | \
  rg 'provisioningrequest|provisioningrequestconfig'
```

If supported for Autopilot v6e, test Kueue's Provisioning Admission Check in a
separate iteration. It can require GKE capacity provisioning before the Job is
unsuspended. It supplements, rather than automatically updates,
`nominalQuota`. Also configure and test Kueue PodsReady timeout/requeue behavior
for capacity that is admitted logically but cannot become Ready physically.

Manage these values from the same Terraform/GitOps source later:

- worker ClusterQueue `nominalQuota`;
- manager aggregate `nominalQuota`;
- allowed TPU shapes;
- any reservation/capacity assignment; and
- maximum accepted concurrency.

Do not raise quota merely because the project quota exists. Raise it only to
the maximum capacity the worker is intended and expected to consume.

## Phase 9: add a second worker

One worker proves remote execution but not balancing or migration. Create a
second regional Autopilot worker in another TPU region, repeat Phases 1 through
3, add its `MultiKueueCluster` name to `v6e-1-workers`, and raise manager quota
to the sum of worker ceilings.

Then test:

- both workers receive jobs;
- saturating one worker sends new admissible work to the other;
- making one worker inactive requires no pipeline change;
- deleting one worker from `MultiKueueConfig` affects only new work; and
- unavailable cloud capacity has a bounded requeue/retry path.

MultiKueue is admission-driven; an exact 50/50 split is not an acceptance
criterion.

## Evidence and handoff report

The executing agent should finish each phase with a short report containing:

```text
Phase:
Commands applied:
Controller context/version:
Worker context/version:
Kueue version:
Objects created:
Observed conditions/events:
Start/end timestamps:
Result: PASS | FAIL | BLOCKED
Cleanup performed:
Next safe action:
```

Do not report the POC successful until direct Job dispatch, remote completion,
cancellation propagation, same-Pod PVC placement, and Agent Stack lifecycle all
pass. With only the current South America worker, report “single-worker
MultiKueue dispatch proven,” not “multi-region balancing proven.”
