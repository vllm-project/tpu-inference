# Production-readiness review

Status: gaps identified; production launch not approved.

## Highest-priority findings

### P0: untrusted pipeline code can become cluster control

The pipeline definition lives in the tested checkout. A pull request that can
change Kubernetes plugin configuration may attempt privileged containers,
host networking, host paths, alternate images, service accounts, or unexpected
secrets. Buildkite queue isolation alone is not a Kubernetes security boundary.

Required controls:

- load the trusted Agent Stack/Kubernetes profile from a protected pipeline or
  platform repository, not directly from unreviewed PR content;
- use separate queues, namespaces, service accounts, and credentials for trusted
  and untrusted builds;
- reject privileged mode, host namespaces, host paths, added capabilities,
  writable host mounts, and arbitrary service accounts through admission policy;
- default `automountServiceAccountToken: false` for test Pods;
- enforce approved image registries and immutable image digests;
- keep protected secrets unavailable to fork and untrusted PR builds;
- move Pod Security admission from baseline to restricted after Agent Stack's
  generated containers have compatible security contexts.

This is the largest issue not exercised by the POC.

### P0: static MultiKueue credential has no production lifecycle

The POC creates a token with `kubectl create token`, embeds it in kubeconfig,
and stores it in the manager Secret. That token expires unless an unusually
long duration is accepted, and making it long-lived increases blast radius.
There is no reconciler to rotate it or update the Secret.

Target fix: Fleet ClusterProfile plus the GKE auth plugin, using Workload
Identity Federation and short-lived credentials. Kueue 0.19 marks the feature
alpha, so upgrade and validation are prerequisites. Do not store worker
kubeconfigs in Terraform state as an intermediate production solution.

### P0: Agent Stack readiness is incompatible with Kueue health

The completed checkout regular container keeps a healthy running Pod
`Ready=False`. Extending `waitForPodsReady` to three hours prevents premature
eviction but also weakens stuck-workload detection.

Production requires one of:

- Agent Stack changes checkout to an init-container-compatible lifecycle;
- Kueue supports an integration-specific readiness definition;
- a supported Kueue mode disables this readiness check while independent health
  monitoring and active deadlines provide a reliable replacement.

Deliberately test stuck checkout, stuck image pull, stuck command startup, agent
disconnect, and controller restart. A long timeout alone is not acceptance.

### P0: secret distribution must be automatic and revocable

MultiKueue does not copy referenced Secrets. Manually porting
`agent-stack-k8s-secret` made the POC work but is not a production mechanism.

Create values in Secret Manager outside Terraform and synchronize them to every
cluster. Require rotation, version pin/rollback, missing-secret alerts, audit
logs, and a tested emergency revocation path. Separate Buildkite agent tokens
and model credentials by environment and trust level.

## Capacity and scheduling

### Standard GKE recommendation

Use a regional Standard manager and zonal Standard workers. Each worker has:

- a small autoscaled CPU system pool for Kueue, CSI, secret sync, and platform
  Pods;
- one TPU pool per supported logical profile;
- topology configured through node-pool placement policy;
- reservation affinity configured on the node pool;
- custom profile labels consumed by ResourceFlavors.

This eliminates Autopilot ComputeClasses and explicit zone/reservation selectors
from workloads. It also makes the PVC's zone deterministic.

### Quota must come from one capacity inventory

Kueue nominal quota is not a live subscription to Cloud TPU reservations.
Generate manager and worker quotas from the same reviewed inventory used to
configure node-pool maxima. Continuously compare:

- reservation capacity and utilization;
- node-pool autoscaling maxima;
- manager ClusterQueue quota;
- worker ClusterQueue quota;
- running/admitted TPU requests.

Alert on drift. A Terraform variable is a starting source of truth, not runtime
capacity discovery.

The POC covers CPU, memory, and ephemeral storage with arbitrary generous
quotas. Production examples cover only `google.com/tpu`; Kubernetes schedules
ordinary resources. Add them back only if central admission must explicitly
budget them and the values come from real capacity.

### Do not share a cohort by default across incompatible shapes

The POC puts `v6e-1` and `v6e-8` in cohort `v6e-pool`. Cohort borrowing may
admit chips against quota belonging to a topology that cannot currently satisfy
the Job. Keep the profiles independent unless an experiment proves the cloud
reservation is truly fungible and fragmentation behavior is acceptable. If
borrowing is enabled, define explicit borrowing/lending limits.

### Bound admission wait

`pendingTimeout: 0` allows indefinite waiting. That is useful during capacity
debugging but unsuitable as the only production policy. Define queue-time SLOs,
cancel superseded commits, and emit a clear capacity failure or fallback after
a profile-specific maximum wait.

## Storage and performance

### Right-size the per-Job disk

Every TPU Job currently requests 500 GiB. Fifteen concurrent tests can request
7.5 TiB of provisioned disk before scale experiments are considered. Measure
actual high-water marks and define smaller profile/test-class sizes. Alert on
orphan PVCs and disk quota.

### Build a trusted cache lifecycle

The empty ephemeral PVC isolates Jobs but discards compilation, model, and
dataset data. A production design should provide a read-only trusted generation
plus a writable PR/branch overlay, with CPU-side preparation and explicit
promotion. Untrusted PRs must never mutate the shared golden generation.

Measure cache correctness and performance before adoption. Cache keys must
include TPU shape, JAX/XLA/vLLM versions, model, compile flags, and source
identity. Avoid GCS FUSE for compilation-cache filesystem traffic unless new
measurements overturn the earlier poor result.

### Replicate images near workers

The POC image repository is in `us-central1` while the worker is in
`southamerica-west1`. Cross-region pulls add startup time and may add network
cost. Use Artifact Registry remote/virtual strategy or a regional repository
per worker region, promote the same digest, and make the platform select the
regional mirror. Never rebuild semantically different images per region.

### Remove embedded download credentials

The MLPerf preparation script retains public object-store credentials inherited
from the existing script. Even if they are intentionally public, embedded
credential-shaped values are difficult to audit and rotate. Prefer a public
read-only endpoint without credentials or retrieve a narrowly scoped value from
Secret Manager. Continue checksum verification.

## Networking and security

- Use private nodes, controlled control-plane endpoints, Cloud NAT, and private
  Google access.
- Define egress policy. Tests generally need Artifact Registry, model storage,
  dataset storage, Buildkite, and selected package endpoints—not unrestricted
  east-west access.
- Use NetworkPolicy or a policy engine with Dataplane V2 where appropriate.
- Use Workload Identity Federation; do not use downloaded service-account keys.
- Give node service accounts logging, monitoring, and registry-read roles only.
- Review the official ClusterProfile example's project-wide
  `roles/container.developer` and `roles/gkehub.gatewayEditor`; add IAM
  Conditions or a dedicated project/fleet when resource names are stable.
- Apply binary authorization/signature policy to the test image, Agent Stack,
  Kueue, auth plugin, secret sync, and policy-controller images.
- Scan images and pin all runtime images by digest.
- Encrypt and access-control Terraform remote state; state contains cluster
  metadata even though it must not contain workload secrets.

## Reliability and operations

### Controller availability

Run the manager in a regional cluster with multiple system nodes. Verify Kueue,
Agent Stack, secret sync, policy webhooks, and Fleet inventory behavior during:

- manager node upgrade;
- controller leader change;
- worker control-plane outage;
- Connect Gateway outage;
- expired/denied credentials;
- Kueue and Agent Stack upgrades.

Set PodDisruptionBudgets and topology spread where the charts support replicas.
Confirm which controllers are actually active/standby rather than assuming
replica count equals high availability.

### TPU node upgrades

Reserved TPU pools often cannot create a surge node. The Terraform example uses
zero surge and one unavailable node. Put upgrades in a CI maintenance window,
drain admission first, and verify no Job is evicted without Buildkite reporting
the infrastructure cause. If extra capacity exists, change the upgrade policy
after testing it.

### Cancellation is a release criterion

Cancel during central queueing, dispatch, node scale-up, PVC provisioning, image
pull, model download, compilation, and test execution. Verify deletion of:

- Buildkite Job/agent;
- manager Job and Workload;
- worker Job, Workload, and Pod;
- generic ephemeral PVC/PV;
- unused TPU node.

Record time-to-zero-billing and alert on finalizers or orphan resources.

### Observability

Export and correlate these identifiers: Buildkite build/job, manager Job,
Workload, worker cluster, worker Job/Pod, PVC/PV, node pool, and TPU node.
Measure separate durations for queue, admission, dispatch, scale-up, PVC,
image pull, agent startup, downloads, compile, test, and cleanup.

Define SLOs and alerts for:

- MultiKueueCluster disconnected;
- ClusterProfile missing/stale;
- no admissible flavor or quota drift;
- Job admitted but no worker Pod;
- excessive node provisioning time;
- agent disconnected while Job runs;
- cleanup deadline exceeded;
- cache hit/compile regression;
- Secret sync or rotation failure.

## CI semantics

- Compare the same commit and cache condition against bare metal before drawing
  performance conclusions.
- Stop using blanket `soft_fail` once the migration phase ends. Separate known
  quarantined tests from infrastructure failures.
- Make test selection deterministic and observable before adding change-based
  selection or Buildkite Test Engine.
- Keep full-matrix/nightly capacity policy distinct from PR latency SLOs.
- Cancel superseded builds for the same PR unless a test is deliberately being
  retained for diagnosis.
- Keep bare metal as a fallback until the new path meets the same reliability
  and correctness bar.

## Acceptance gates

Production launch requires all of the following:

1. Trusted/untrusted pipeline and Kubernetes admission boundaries are enforced.
2. No static worker kubeconfig or bearer token is used.
3. Secret synchronization and emergency revocation are tested.
4. Readiness has a correct health signal or an explicitly accepted replacement.
5. Same-commit performance and cost data covers cold and warm cache states.
6. Cancellation reliably removes every resource within a defined SLO.
7. Quota/fairness tests cover simultaneous one-chip and eight-chip builds.
8. At least two worker failure modes are exercised; a second region is tested
   before multi-region resilience is claimed.
9. Cluster, Kueue, Agent Stack, plugin, and policy upgrades have rollback plans.
10. On-call dashboards, alerts, runbooks, ownership, and cost budgets exist.

Until these pass, use a canary queue and keep the production decision open
between bare metal, hybrid, and Kubernetes-first execution.
