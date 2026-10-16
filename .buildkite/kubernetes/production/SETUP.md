# Production setup runbook

Ordered steps to stand up a production-like Buildkite-on-Kubernetes
environment. [`README.md`](README.md) explains *why* the design looks like
this; this file is what you execute. [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md)
is the gap list you must close before calling it production.

Every step ends in a **Gate**. A gate that does not pass is a stop, not a
warning — most failures here are silent (a Pod that never starts, a cache that
looks warm and misses every lookup, a reservation that quietly bills on-demand).

## Target

| | Manager | Worker |
| --- | --- | --- |
| Project | `cloud-ullm-inference-ci-cd` | `cloud-tpu-inference-test` |
| Location | `us-central1`, regional | `southamerica-west1-a`, zonal |
| Mode | Standard, autoscaling system pool | Standard |
| Runs | Agent Stack, Kueue + MultiKueue manager | Kueue, copied Jobs, TPU Pods |
| TPUs | none | v6e-1 (`1x1`), v6e-8 (`2x4`), from a specific reservation |

Buildkite sees one queue. Kubernetes owns placement: a pipeline step declares a
TPU profile and never a project, region, zone, or cluster name.

---

## 0. Decide and collect

Resolve these before touching Terraform.

**a. Manager cluster mode.** The Terraform builds a **regional Standard**
manager with an autoscaling system pool, and that is the recommended default:
the manager holds the Buildkite agent token and every worker credential, the
node pool is one block you never revisit, and a warm floor of system nodes
keeps Agent Stack from paying cold-start latency on the first job of the day.

Autopilot is a legitimate alternative — the manager runs no TPU workload, and
because MultiKueue keeps the manager Job suspended the *test* Pod never runs
there either, so there is little node-level decision left. The ClusterProfile
auth plugin is delivered by an initContainer, which Autopilot permits. Choose it
if you would rather not own node upgrades at all. It is a rewrite of the
`google_container_cluster.manager` block (`enable_autopilot = true`, drop
`manager_system`), not a variable — GKE cannot convert an existing cluster
either way.

**b. Worker cluster mode is not a choice.** It must be Standard and zonal. TPU
reservation consumption becomes an explicit `reservation_affinity` on the node
pool, verifiable before you spend a TPU hour. On Autopilot it is a ComputeClass
behaviour you can only confirm by watching a reservation counter during a live
run. Zonal also removes any ambiguity about where the TPU Pod and its zonal PD
land.

**c. The TPU reservation name.**

```bash
gcloud compute reservations list --project cloud-tpu-inference-test \
  --filter="zone:southamerica-west1-a" \
  --format="table(name,specificReservationRequired,specificReservation.count,specificReservation.inUseCount)"
```

Record `specificReservation.count` per machine type — it sets both `max_nodes`
and, later, Kueue quota.

**d. Network.** Subnets, secondary Pod/Service ranges, Cloud NAT, and Private
Google Access in both projects. This Terraform consumes them; it does not
create them.

**e. State and secrets**, once, by hand:

```bash
gcloud storage buckets create gs://REPLACE_TF_STATE \
  --project cloud-ullm-inference-ci-cd --location us-central1
gcloud storage buckets update gs://REPLACE_TF_STATE --versioning
```

The Buildkite agent token is added as a Secret Manager *version* outside
Terraform, in step 3. Terraform creates the empty container only — a token in
state is a token in every plan output and every state backup.

**Gate:** reservation name and chip counts recorded; network ranges exist in
both projects; state bucket versioned.

---

## 1. Apply the foundation

```bash
cd .buildkite/kubernetes/production/terraform
cp backend.tf.example backend.tf          # set the state bucket
cp terraform.tfvars.example prod.tfvars   # then edit
terraform init
terraform fmt -check -recursive
terraform validate
terraform plan -var-file=prod.tfvars -out=production.tfplan
terraform show production.tfplan
```

The values that matter for the split-project layout:

```hcl
project_id     = "cloud-ullm-inference-ci-cd"   # fleet host + manager + registry
worker_project = "cloud-tpu-inference-test"     # worker clusters + TPU reservation
```

`worker_project` exists because a reservation is only consumable from the
project that owns it, so the worker cluster has to live beside it. Leave it
`null` for a single-project deployment. The worker still joins the **manager's**
fleet: a cluster can belong to only one fleet, and MultiKueue needs both sides
in the same one.

Apply through the protected infrastructure pipeline using workload identity or
service-account impersonation — never a downloaded key.

**Gate:** apply succeeds; `terraform plan` is then empty.

---

## 2. Verify the node labels and the reservation

Kueue's ResourceFlavors select `tpu-ci.google.com/profile`. GKE applies the TPU
accelerator and topology labels itself. Verify rather than assume — a mismatch
surfaces much later as a Workload admitted on the manager that never runs
anywhere.

```bash
gcloud container clusters get-credentials tpu-ci-southamerica-west1-a \
  --zone southamerica-west1-a --project cloud-tpu-inference-test

gcloud container clusters resize tpu-ci-southamerica-west1-a \
  --node-pool v6e-1 --num-nodes 1 --zone southamerica-west1-a \
  --project cloud-tpu-inference-test --quiet

kubectl get nodes -L tpu-ci.google.com/profile,\
cloud.google.com/gke-tpu-accelerator,cloud.google.com/gke-tpu-topology,topology.kubernetes.io/zone
```

While that node is up:

```bash
gcloud compute reservations describe RESERVATION \
  --project cloud-tpu-inference-test --zone southamerica-west1-a \
  --format="value(specificReservation.inUseCount)"
```

**Gate:** the node reports `profile=v6e-1`, `tpu-v6e-slice`, `1x1`,
`southamerica-west1-a`, **and** `inUseCount` increased. If the node came up but
the count did not, the pool is running on-demand capacity and
`reservation_affinity` is wrong — fix it before anything else. Scale back to 0.

---

## 3. Secrets and the platform stack

1. Add the Buildkite agent token as a Secret Manager version.
2. Install secret synchronization (GKE native sync or External Secrets) so
   `agent-stack-k8s-secret` materializes in the manager **and every worker**.
   MultiKueue copies the PodSpec and nothing else; a Secret that exists only on
   the manager is why a copied Job's init container fails with no useful
   Buildkite-side error.
3. Install the pinned Kueue release on the manager and every worker — same
   version on both.
4. Apply the manager patch in [`kueue/manager-auth-plugin-patch.yaml`](kueue/manager-auth-plugin-patch.yaml)
   with a reviewed, signed digest. The placeholder image must not be deployed.
5. Apply the queue objects in [`kueue/`](kueue).
6. Install Agent Stack on the **manager only**. A second controller on a worker
   would let either one claim a Buildkite job before Kueue is involved.

**Gate:** both Kueue controllers Available; `ClusterQueue` Active on both sides.

---

## 4. Connect manager to worker

The connection model is Fleet `ClusterProfile` + Connect Gateway, not a stored
kubeconfig. Verify the inventory exists before creating `MultiKueueCluster`
objects:

```bash
kubectl --context MANAGER api-resources | grep -i clusterprofile
kubectl --context MANAGER -n kueue-system get clusterprofile
```

If ClusterProfile is not served by your pinned Kueue and GKE versions, stop and
decide deliberately: the fallback is a Secret-based kubeconfig, which is a
bootstrap mechanism, not a production one. If you must use it temporarily,
scope the worker identity to a namespaced Role, give the token a bounded
lifetime, and rotate it automatically. `PRODUCTION_READINESS.md` treats a
static token as a release blocker.

**Gate:** `MultiKueueCluster` and `AdmissionCheck multikueue-dispatch` both
report Active, with no Secret holding a bearer token.

---

## 5. Cache

Read [`cache/`](cache). Two halves, both required:

- **golden PVC → job.** [`cache/golden-cache-pvc.yaml`](cache/golden-cache-pvc.yaml)
  is the warm base each job's ephemeral volume clones. Per-cluster and zonal:
  MultiKueue does not move a PersistentVolume, so every worker needs its own
  under the same name.
- **job → GCS → golden.** [`cache/serviceaccounts.yaml`](cache/serviceaccounts.yaml)
  plus [`cache/golden-refresh-cronjob.yaml`](cache/golden-refresh-cronjob.yaml).

The write-back is the part the POC does not have, and it is not optional at
production scale. A golden holds usable keys but never the full shape matrix:
POC build 35 measured adjacent `num_reqs=16` and `num_reqs=20` shapes at ~115s
of compilation each, on every build, forever, because nothing carried them
back.

Enable it by setting `KUBE_JAX_CACHE_WRITE=1` in the production Agent Stack
environment and adding two commands around each TPU step:

```yaml
commands:
  - bash .buildkite/scripts/kube_jax_cache.sh snapshot
  - <the test command>
  - bash .buildkite/scripts/kube_jax_cache.sh publish
```

`snapshot` records what the golden already had; `publish` uploads only the
difference. Publish is last on purpose: Buildkite aborts the script on the first
failure, so a failed test never reaches it and cannot poison the shared cache —
the same guard `run_in_docker.sh` gets from the container exit code. Uploads use
`if_generation_match=0`, so concurrent publishers of the same content-addressed
entry are idempotent rather than racing.

Both commands are inert unless `KUBE_JAX_CACHE_WRITE=1`, so they are safe to
carry in a shared pipeline.

Set `JAX_CACHE_NAMESPACE` in the refresh CronJob to match the image's JAX
version. `kube_jax_cache.sh` prints the namespace it resolved on every run, so
a JAX bump that misses the CronJob shows up in build logs rather than only as a
gradual slowdown.

Grants are created by [`terraform/cache.tf`](terraform/cache.tf) as direct
Workload Identity Federation principals — no service account, no key —
conditioned to the cache prefix so a test cannot reach the rest of the bucket.

**Gate:** first build logs a non-zero `uploaded`; after a golden refresh, the
next build logs a large baseline and a visibly shorter compile phase. Add a GCS
lifecycle rule on the prefix before the first month of traffic — every build
adds entries and nothing removes them.

---

## 6. Attach the Buildkite queue

Only one Agent Stack controller may consume a queue. Either point the new
controller at a fresh queue and cut over after validation, or stop the existing
controller and confirm no queued or running jobs on that queue first.

**Gate:** one Buildkite job produces exactly one command result and one log
stream; cancelling it, while queued and while running, removes the Job,
Workload, Pod, and ephemeral PVC on **both** clusters.

---

## 7. Before calling it production

Work [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md). The items most likely
to be skipped:

- Kueue quota must equal capacity the worker can actually autoscale into. Zero
  means no Pod is ever created and the autoscaler never reacts; too much means
  Kueue admits work whose Pod stays Pending indefinitely.
- Any Buildkite concurrency limit must agree with that quota. A pipeline that
  allows more concurrent jobs than there are chips just moves the queue.
- `pendingTimeout` must be a measured bound, not `0`. Pair it with Kueue's
  PodsReady/requeue policy so unavailable cloud capacity fails within a bound.
- Every performance comparison must state cold or warm cache. A cold Kubernetes
  run against warm bare metal is not a scheduler regression.
