# Autopilot MultiKueue handoff

This is the executable handoff for two existing fleet-registered GKE Autopilot
clusters:

- manager/controller: `ci-test-controller`
- first worker: `ci-test-southamerica-west1-worker`

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
export POC_CONTROLLER_REGION="REPLACE_CONTROLLER_REGION"
export POC_WORKER_REGION="southamerica-west1"
export POC_CONTROLLER_CLUSTER="ci-test-controller"
export POC_WORKER_CLUSTER="ci-test-southamerica-west1-worker"
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

Gate: both contexts are correct, both clusters are `RUNNING` Autopilot clusters,
and all four authorization checks return `yes`.

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

Gate: one remote execution completes, status returns to the manager, and
deleting/cancelling the manager Job removes the remote copy.

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

Create a disposable Buildkite queue named `v6e-1`. Install Agent Stack only on
`ci-test-controller`, configured to create Kubernetes Jobs in manager namespace
`buildkite-v6e-1`. Do not install an Agent Stack controller on the worker.

Before sending a TPU command, establish how the generated top-level Job receives
`kueue.x-k8s.io/queue-name: v6e-1`:

1. prefer an Agent Stack value/plugin field that adds metadata to the Job;
2. otherwise use the pinned Kueue release's LocalQueue defaulting feature in
   this dedicated namespace; or
3. use a narrowly scoped admission mutation as a POC fallback.

The existing `podSpec.metadata.labels` labels the Pod template and must not be
assumed to label the parent Job. Prove the bridge with an unlabeled, harmless
Job before enabling the Buildkite queue. The Job must be mutated to the queue
and suspended before any Pod is created.

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
