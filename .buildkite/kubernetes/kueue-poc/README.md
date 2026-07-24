# Kueue and MultiKueue POC for Buildkite TPU jobs

## Recommendation

Use Kueue on each TPU worker cluster and evaluate MultiKueue as the
cross-cluster dispatcher. This is preferable to encoding a regional queue and
selection algorithm in `pipeline_kube.yaml` if the integration gates below
pass. Buildkite should expose one logical queue per resource shape (`v6e-1`,
`v6e-8`, and later `v7x-*`), while Kubernetes owns physical placement.

This design is feasible for independent Kubernetes `Job` objects. It is not
yet proven to be a supported Buildkite Agent Stack configuration, and it does
not make multiple Buildkite steps share a worker cluster or PVC. Treat those
as explicit POC gates rather than assumptions.

It is not unconditionally simpler. MultiKueue adds a manager control plane,
CRDs, cross-cluster credentials, and a resource-replication contract. Its value
is that this operational complexity is centralized and reusable instead of
being repeated in pipeline-generation code.

| Concern | Regional Buildkite queues + planner | Kueue + MultiKueue |
| --- | --- | --- |
| Pipeline location logic | Custom and repository-owned | None after integration |
| Add/drain/migrate a worker | Registry/code/config rollout | Kubernetes control-plane change |
| Local quota and fair sharing | Buildkite concurrency approximation | Native Kueue policy |
| Balance using currently admissible quota | Custom API queries | MultiKueue worker admission |
| Initial infrastructure complexity | Lower | Higher |
| Agent Stack compatibility risk | Low | Must pass Gate 1 and cancellation tests |
| Cross-step PVC locality | Custom bundle routing | Not supplied by MultiKueue |
| Real GCP TPU availability | Must be handled | Still must be handled |

For one or two static clusters, the queue planner is operationally cheaper. If
TPU capacity will move among zones/regions or more execution pools will be
added, MultiKueue is the stronger long-term abstraction—provided the POC proves
Agent Stack lifecycle compatibility.

## Target architecture

```text
                         manager GKE cluster
 Buildkite v6e-1 queue -> Agent Stack controller
                                  |
                         suspended batch/v1 Job
                                  |
                         Kueue + MultiKueue
                           /              \
              worker us-central1-b    worker us-west1-b
              Kueue LocalQueue        Kueue LocalQueue
              CPU + v6e-1 pools       CPU + v6e-1 pools
              local image/golden      local image/golden
```

The manager can be a small non-TPU cluster. It runs the Agent Stack controller
and MultiKueue manager. Worker clusters run Kueue and the copied Buildkite Job,
but do not need another Agent Stack controller. Use a dedicated manager
namespace per logical Buildkite queue, for example `buildkite-v6e-1`. That
makes queue defaulting and admission isolation easy to audit.

Keep different TPU shapes in different namespaces/LocalQueues/ClusterQueues.
For example, a `v6e-1` Job must not become flavor-fungible with a `v6e-8` pool
merely because both request `google.com/tpu`. Give `v6e-8` its own manager
namespace and worker queues, set its ResourceFlavor topology to `2x4`, and make
its quota a multiple of eight chips.

For `v6e-1` in two zones, both workers have a LocalQueue named `v6e-1` backed by
a local ClusterQueue. The manager ClusterQueue represents the aggregate
ceiling; each worker ClusterQueue represents the maximum capacity that its GKE
node-pool autoscaler and cloud quota can actually reach. MultiKueue can then
select a worker that obtains local quota without the pipeline naming it.

Kueue quota should represent autoscalable capacity, not merely nodes running
at this instant. If quota is zero, no Pod appears and the cluster autoscaler has
nothing to react to. If quota exceeds obtainable Google Cloud capacity, Kueue
can admit the Job but its Pod can remain Pending indefinitely. Do not retain the
current `pendingTimeout: 0` as the only production policy. Configure and test
Kueue's PodsReady timeout/requeue behavior on each worker, then keep a longer
Agent Stack pending timeout as a final safety bound. Verify where the pinned
Agent Stack version starts that timer so legitimate Kueue queue wait is not
mistaken for failed GKE provisioning.

## Worker-cluster contract

A Standard execution cluster may be zonal. GKE Autopilot clusters have regional
control planes, but each TPU workload and zonal Persistent Disk still lands in
one zone. Before installing Kueue, verify all of the following:

- Autopilot is allowed to provision the requested TPU shape/topology in at
  least one zone in the worker region;
- the expected GKE accelerator/topology selectors cause Autopilot to provision
  the intended TPU rather than being rejected by admission;
- Kueue quota does not exceed project TPU quota or expected stock;
- the chosen StorageClass uses the intended zonal disk type and, where
  appropriate, `WaitForFirstConsumer`;
- the Buildkite namespace, service accounts, secrets, image-pull access, and
  network egress required by the copied Job exist;
- for the initial Autopilot POC, a `WaitForFirstConsumer` generic ephemeral PVC
  is created by the same TPU Pod rather than by an earlier CPU Job;
- the manager Kueue controller can reach the worker Kubernetes API.

Capture this contract in cluster provisioning rather than `pipeline_kube.yaml`.
The pipeline requests `v6e-1`; a worker declares whether it can satisfy that
request.

## Non-goals and constraints

- MultiKueue does not move a PersistentVolume across regions or clusters.
- MultiKueue does not schedule a Buildkite dependency graph as one unit.
- A CPU prep Job and later TPU Job are independent Workloads and may land on
  different workers even when they use the same logical queue.
- Secrets, service accounts, image-pull credentials, ConfigMaps, StorageClasses,
  golden snapshots/PVCs, and namespaces referenced by a copied Job must already
  exist with compatible names in every eligible worker.
- MultiKueue copies one PodSpec; it does not rewrite an Artifact Registry host
  or PVC name for the selected worker. Use identical logical names. For the POC,
  use one shared image reference and accept cross-region pull cost. Later, use
  a cluster-owned registry mirror or narrowly scoped admission mutation to map
  that logical image reference to an immutable, same-digest regional copy.
- MultiKueue observes Kubernetes quota and admission, not live GCP TPU stock.
- Buildkite concurrency groups can remain as a temporary safety ceiling, but
  Kueue should become the source of truth for physical capacity after the POC.

For the first POC, use a self-contained TPU Job and skip
`KUBE_RESOURCE_PREP_POC`. The Job may hydrate cache/models/datasets and publish
its delta while holding the TPU; that cost must be timed, but it avoids both
cross-cluster and cross-zone handoff. Start with a fresh generic ephemeral PVC,
then add worker-local golden cloning only after its zonal placement is proven.

The concrete handoff for the already-created Autopilot clusters
`ci-test-controller` and `ci-test-southamerica-west1-worker` is in
[`AUTOPILOT_HANDOFF.md`](AUTOPILOT_HANDOFF.md).

## Gate 0: pin and inspect versions

Do this before applying the examples. No unpinned `latest` installation should
be used.

```bash
kubectl version
helm list --all-namespaces
helm get values AGENT_STACK_RELEASE -n AGENT_STACK_NAMESPACE
helm get manifest AGENT_STACK_RELEASE -n AGENT_STACK_NAMESPACE > agent-stack-rendered.yaml
kubectl api-resources | rg 'kueue|multikueue|workload|clusterqueue'
```

Choose a Kueue release whose compatibility documentation explicitly supports:

1. the GKE Kubernetes minor version;
2. `batch/v1 Job` integration;
3. MultiKueue for `batch/v1 Job`;
4. the `MultiKueueCluster`, `MultiKueueConfig`, and `AdmissionCheck` API versions
   used by that release; and
5. LocalQueue defaulting, if that is the chosen Agent Stack bridge; and
6. PodsReady timeout/requeue behavior for remote `batch/v1 Job` workloads.

Install the same pinned Kueue release on the manager and workers using the
upstream release manifests or Helm chart. Enable the MultiKueue controller on
the manager. Do not copy an installation manifest from this directory; only
the POC resources layered on top of Kueue are included here.

## Gate 1: prove Agent Stack creates a Kueue-manageable Job

Before introducing MultiKueue, use one zonal cluster and a disposable Buildkite
queue. Trigger a one-line Agent Stack job and immediately capture the generated
object:

```bash
kubectl get jobs -n buildkite-v6e-1 --sort-by=.metadata.creationTimestamp
kubectl get job JOB_NAME -n buildkite-v6e-1 -o yaml > agent-stack-job.yaml
kubectl get workloads -n buildkite-v6e-1
kubectl get pods -n buildkite-v6e-1
```

Kueue's queue marker must be on the **Job metadata**, not only on
`spec.template.metadata` (the Pod). The resulting Job must be suspended before
a Pod starts, a Workload must be created, and the Job must resume only after
admission.

Inspect the entire Job for dependencies that MultiKueue will not copy:

```bash
kubectl get job JOB_NAME -n buildkite-v6e-1 -o jsonpath='{.spec.template.spec.serviceAccountName}{"\n"}'
kubectl get job JOB_NAME -n buildkite-v6e-1 -o yaml | \
  rg 'secretKeyRef|configMapKeyRef|secretName|configMap|persistentVolumeClaim'
```

Stable references can be provisioned with the same name in every worker. A
per-job Secret or ConfigMap created only in the manager is a hard blocker unless
Agent Stack can be configured to avoid it or a separate, secure replication
controller is introduced. Record owner references and injected volumes too;
the remote Job must not depend on a manager-only object.

Use the first supported bridge available in the pinned versions:

1. configure Agent Stack to add
   `metadata.labels["kueue.x-k8s.io/queue-name"]: v6e-1` to every generated Job;
2. otherwise enable Kueue LocalQueue defaulting and make `v6e-1` the single
   default LocalQueue in the dedicated `buildkite-v6e-1` namespace; or
3. as a POC-only fallback, install a narrowly scoped mutating admission policy
   that adds the fixed Job label only in that namespace.

Do not use a broad rule that manages every unlabeled Job in the cluster; it can
suspend the Agent Stack controller's own maintenance Jobs. Do not proceed if a
Pod can start before Kueue creates and admits its Workload.

Also validate normal Agent Stack semantics while the Job is suspended: build
cancellation, timeout, retry, log streaming, command exit status, and cleanup.
Agent Stack is not itself a Kueue integration, so passing this gate matters more
than the individual APIs merely existing.

## Gate 2: prove Kueue on each worker

Copy `worker-v6e-1.yaml` to each worker and set `nominalQuota` to the real
autoscaling ceiling. The checked-in example is four v6e chips and must be
changed for a worker that can supply only two, one, or another amount.

```bash
kubectl --context WORKER_CONTEXT apply -f worker-v6e-1.yaml
kubectl --context WORKER_CONTEXT get resourceflavor,clusterqueue
kubectl --context WORKER_CONTEXT get localqueue -n buildkite-v6e-1
```

Submit a normal suspended TPU Job directly to each worker, with queue label
`v6e-1`. Confirm that Kueue admits at most the configured chip quota and that
admitted Pending Pods cause the TPU node pool to scale. Confirm the node has
both required labels:

```bash
kubectl --context WORKER_CONTEXT get nodes -L \
cloud.google.com/gke-tpu-accelerator,cloud.google.com/gke-tpu-topology,topology.kubernetes.io/zone
kubectl --context WORKER_CONTEXT describe clusterqueue v6e-1
kubectl --context WORKER_CONTEXT get workloads -A
```

Create an identically named `tpu-cache-golden-pvc` (or its replacement) from a
local snapshot in every worker. A test clone and its consuming Pod must bind in
that worker's only zone. The name and content contract is shared; the disk is
not.

## Gate 3: connect manager to workers

Give the manager Kueue controller a least-privilege kubeconfig for each worker.
Use the credential-generation procedure shipped with the pinned Kueue release;
do not assume that a developer kubeconfig containing the GKE `exec` auth plugin
will work inside the controller Pod. Store each kubeconfig as a Secret in the
namespace where the manager Kueue controller expects MultiKueue credentials.

Verify credentials independently before creating `MultiKueueCluster` objects:

```bash
kubectl --kubeconfig WORKER_SERVICE_ACCOUNT_KUBECONFIG auth can-i get workloads.kueue.x-k8s.io -A
kubectl --kubeconfig WORKER_SERVICE_ACCOUNT_KUBECONFIG auth can-i create jobs.batch -A
kubectl --kubeconfig WORKER_SERVICE_ACCOUNT_KUBECONFIG auth can-i delete jobs.batch -A
```

Edit `manager-v6e-1.yaml` so the two `MultiKueueCluster.spec.kubeConfig.location`
values match the Secret names and the manager `nominalQuota` equals the total
eligible worker ceiling. Apply it and wait for both workers to become Active:

```bash
kubectl --context MANAGER_CONTEXT apply -f manager-v6e-1.yaml
kubectl --context MANAGER_CONTEXT get multikueuecluster
kubectl --context MANAGER_CONTEXT describe admissioncheck multikueue-dispatch
kubectl --context MANAGER_CONTEXT get clusterqueue v6e-1
```

Create `smoke-job.yaml` on the manager. It uses `generateName`, so use
`kubectl create`, not `kubectl apply`. Watch the manager and both workers in
separate terminals:

```bash
kubectl --context MANAGER_CONTEXT create \
  -f .buildkite/kubernetes/kueue-poc/smoke-job.yaml
```

```bash
kubectl --context MANAGER_CONTEXT get job,workload,pod -n buildkite-v6e-1 -w
kubectl --context WORKER_ONE get job,workload,pod -n buildkite-v6e-1 -w
kubectl --context WORKER_TWO get job,workload,pod -n buildkite-v6e-1 -w
```

Expected result: the manager Job remains suspended locally; worker Workloads
compete for local admission; exactly one worker creates/runs the remote Job;
the other worker does not run a duplicate; completion status returns to the
manager Job.

## Gate 4: attach the logical Buildkite queue

Point a disposable Agent Stack controller at Buildkite queue `v6e-1` and have
it create Jobs in manager namespace `buildkite-v6e-1`. Do not install regional
Agent Stack controllers consuming that same queue. Use the label/defaulting
bridge proven in Gate 1.

Set `KUEUE_SMOKE_IMAGE` to an immutable image digest available to every worker,
then upload `buildkite-smoke.yaml` from the existing CPU bootstrap queue. It is
kept separate from `pipeline_kube.yaml` so no production or POC TPU matrix is
silently rerouted before the dispatcher is proven.

The eventual pipeline-facing shape is intentionally small:

```yaml
agents:
  queue: v6e-1
plugins:
  - kubernetes:
      # Use a bounded value after measuring cold node provisioning.
      pendingTimeout: REPLACE_WITH_TESTED_TIMEOUT
      podSpec:
        containers:
          - name: vllm-tpu-runner
            image: SAME_LOGICAL_IMAGE_REFERENCE_IN_EVERY_WORKER
            resources:
              requests:
                google.com/tpu: "1"
              limits:
                google.com/tpu: "1"
```

There is no region, zone, worker name, or balancing code. Once ResourceFlavor
injection is verified, the pipeline can also omit TPU node selectors; the
worker's `v6e-1` ResourceFlavor owns the accelerator and topology labels.

Start with an opt-in one-chip smoke Buildkite step. Then run three classes:

1. a short command that reserves one v6e chip and writes no PVC;
2. one current self-contained `pipeline_kube` test using a worker-local golden;
3. enough parallel sleep Jobs to exceed each individual worker's quota but not
   their combined quota.

Record these timestamps separately:

- Buildkite job becomes runnable;
- Agent Stack creates the manager Job;
- manager Workload obtains quota reservation;
- a worker is selected and its remote Job is created;
- worker Pod becomes Scheduled and Ready (including node autoscaling/image/PVC);
- Buildkite agent starts the command;
- command ends; and
- remote and manager resources are deleted.

Kueue events and Workload conditions are the source of truth for dispatcher
wait; Pod conditions explain GKE provisioning wait. Preserve them as Buildkite
artifacts for the comparison instead of combining all delay into “queue time.”

## Placement and failure experiments

Run all of these before accepting the design:

| Experiment | Expected result |
| --- | --- |
| Submit 20 identical short jobs below combined quota | Both eligible workers are used; no duplicate execution. Distribution need not be round-robin. |
| Saturate worker one local quota | New work can be admitted by worker two without a pipeline change. |
| Set worker one ClusterQueue inactive or remove it from MultiKueueConfig | New work uses worker two; running work is not silently migrated. |
| Add a third worker with the same LocalQueue contract | It receives work without changing `pipeline_kube.yaml`. |
| Exhaust real cloud TPU capacity after Kueue admission | Kueue PodsReady policy releases/requeues the attempt within a bound, with Agent Stack timeout as a longer backstop. Verify whether the pinned MultiKueue release redispatches automatically or requires Buildkite retry. |
| Disconnect one worker API | It becomes inactive; jobs are not duplicated or lost. |
| Restart manager Kueue and Agent Stack controllers | Existing Workload/Job state reconciles and each Buildkite command runs once. |
| Cancel a queued Buildkite job | Manager Workload and all worker copies disappear. |
| Cancel a running Buildkite job | Remote Job/Pod terminate and TPU/PVC resources are released. |
| Reference a missing worker Secret or golden PVC | Failure is visible and bounded; another worker is not selected after the command has begun. |

MultiKueue's choice is admission-driven, not guaranteed round-robin or weighted
distribution. The useful assertion is that capacity is harvested and policy is
centralized, not that 20 jobs split exactly 10/10.

## Acceptance criteria

Adopt MultiKueue over the hand-managed planner only if all are true:

- Pipeline declarations specify logical TPU shape but no physical location.
- A worker can be added, drained, or removed with Kubernetes configuration only.
- No Buildkite command starts before Kueue admission, and no command runs twice.
- Cancellation, timeout, retry, controller restart, and worker disconnect are
  bounded and leave no TPU Pods, Workloads, or ephemeral PVCs behind.
- Combined worker quota is used when one worker is full.
- Median dispatcher overhead (manager Job creation through remote Job creation)
  is measured and acceptably small relative to TPU provisioning and execution.
- Each self-contained job uses only its worker-local image, secret, model,
  dataset, golden cache, and PVC.
- A cloud-capacity miss has a tested timeout/retry path.
- Buildkite timing reports separate dependency/gate, MultiKueue dispatch,
  worker admission, GKE provision, command execution, and cleanup.

Reject or postpone it if Agent Stack cannot safely create suspended/labelled
Jobs, if cancellation leaves remote work behind, or if required Pod fields are
not preserved when MultiKueue copies the Job. In that case, Kueue is still
useful inside each zonal cluster, while Buildkite queue-per-zone remains the
temporary cross-cluster dispatcher.

## Production follow-ups after a successful POC

- Replicate immutable images by digest and golden snapshots to each worker;
  map logical names to local copies in cluster configuration, never by making a
  Buildkite Job infer its region.
- Replace static worker kubeconfig tokens with a managed, rotated identity.
- Add PriorityClasses/WorkloadPriorityClasses for presubmit, main, nightly, and
  interactive work, plus explicit preemption policy.
- Decide whether unused quota should be shared through Kueue cohorts.
- Export Kueue admission/pending metrics and correlate Workload UID, Kubernetes
  Job UID, Buildkite job UUID, physical worker, and TPU topology.
- Design a worker-local workflow for CPU hydration and cache publication. Until
  then, seed local goldens out of band and keep the dispatched TPU Job
  self-contained.
- Remove redundant Buildkite concurrency gates only after Kueue quota and cloud
  quota have been reconciled and load-tested.
