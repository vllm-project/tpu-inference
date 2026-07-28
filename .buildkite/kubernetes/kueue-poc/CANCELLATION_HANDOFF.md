# Cancellation cleanup handoff

Cancelling a Buildkite build does not remove the Kubernetes Jobs it created.
This is the executable handoff for diagnosing and fixing that.

`DESIGN.md` §"Agent Stack and Kueue readiness incompatibility" ends by accepting
a long `waitForPodsReady` timeout and compensating with "aggressive cancellation
cleanup". This document is that compensation, and it is currently missing.

## Working rules

- Stop at the first failed gate. Do not proceed on a hypothesis.
- Use explicit `--context` on every command. The two clusters have
  near-identical namespaces and object names.
- Operate on the **manager**, never the worker. Deleting a remote Job directly
  while the manager still believes it is dispatched means MultiKueue recreates
  it — you are racing a controller that will win.
- Cancel in **Buildkite first**, then in Kubernetes. If the Buildkite job is
  still `scheduled`, the controller polls, sees unclaimed work, and creates a
  fresh Kubernetes Job. The delete will look like it silently failed.
- Do not change any timeout in order to fix cancellation. See
  [Why timers are the wrong tool](#why-timers-are-the-wrong-tool).
- Do not disable checkout, and do not add a local patch to Agent Stack without
  recording the exact upstream behaviour that forced it.

## Authoritative identifiers

| Item | Value |
| --- | --- |
| Buildkite pipeline | `tpu-commons/kube-dev` |
| Buildkite queue | `kube` |
| Agent Stack | `v0.46.2`, controller cluster only |
| Kueue / MultiKueue | `v0.19.0`, both clusters |
| Manager context | `gke_cloud-tpu-inference-test_us-central1_ci-test-controller` |
| Worker context | `gke_cloud-tpu-inference-test_southamerica-west1_ci-test-southamerica-west1-worker` |
| Namespace | `buildkite`, both clusters |
| LocalQueue / ClusterQueue | `v6e-1` and `v6e-8`, cohort `v6e-pool` |
| MultiKueueCluster | `ci-test-southamerica-west1-worker` |
| AdmissionCheck | `multikueue-dispatch` |
| Kubernetes Job name | `buildkite-<BUILDKITE_JOB_UUID>` |

Verify the deployed versions before trusting the table — chart version is
intent, image digest is truth, and they diverge the moment someone patches a
Deployment by hand. Phase 0 collects both.

## Observed behaviour

Reported by the operator, not yet reproduced under instrumentation:

1. A Buildkite build is cancelled while several of its jobs are still queued.
2. The corresponding Kubernetes Jobs on the manager remain. They are either
   unadmitted by Kueue, or admitted but waiting on TPU capacity.
3. They are **not** cleaned up.
4. When capacity later frees, Kueue admits one, MultiKueue dispatches it, a TPU
   node scales up, the Pod starts, the agent connects, polls Buildkite,
   discovers the job was cancelled, and exits immediately.

## Mechanism

Buildkite delivers cancellation **to a running agent**, so a job with no agent
has no recipient. Everything else follows from two properties of how Agent Stack
compensates for that.

Operator finding — cite the source file or the observed evidence here before
treating it as settled:

1. **Agent Stack watches Pods, not Jobs**, and its cancellation acts on a
   **Pod**, not the Job. Nothing in that path deletes or suspends the Job
   object.
2. **The check is edge-triggered.** It runs at the moment the build is
   cancelled. It is not re-evaluated afterwards.

Both assumptions are sound in a plain Agent Stack deployment, where the Job and
its Pod are created together, so "there is a Pod now" and "there will ever be a
Pod" are the same statement.

**Kueue breaks that equivalence.** Admission inserts an unbounded delay between
Job creation and Pod creation:

```text
plain Agent Stack     Job created ── Pod created ─────────────► cancel sees a Pod

with Kueue            Job created ── suspended ── ... hours ... ── admitted ── Pod
                                        ▲                                       ▲
                                   cancel fires here                  Pod appears here,
                                   and finds nothing                  unwatched, and runs
```

At cancellation the manager Job is suspended and has no Pod, so the one-shot
check finds nothing to act on and completes successfully. Later, Kueue admits
the Workload, `suspend` flips to false, a Pod is created — and nothing looks at
it again, because the only evaluation already happened.

That is the whole bug. It is not that the controller cannot see the Job; it is
that it only ever looks once, at the one moment the answer is guaranteed to be
"no Pod".

Two consequences worth stating plainly:

- **A Pod-scoped fix would not be enough.** Deleting a Pod leaves the Job, which
  recreates one within `backoffLimit`. Cleanup has to act on the Job or the
  Workload.
- **This is not MultiKueue-specific.** Any delay between Job and Pod creation
  opens the same window, so a plain Kueue install has it too — and even without
  Kueue there is a narrow race if a build is cancelled between Job creation and
  Pod creation. MultiKueue only widens the window from milliseconds to hours.

## Cost

Per cancelled-but-queued job, measured against POC timings:

- a cold TPU node scale-up, roughly 3m24s to 3m51s;
- an image pull and golden-PVC clone;
- one TPU chip held for that whole window;
- and a quota slot taken **ahead of work that is still wanted** — with
  `nominalQuota: 1` on the one-chip flavor, a cancelled job can delay a live one
  by the full provisioning cycle.

The waste scales with how often superseded builds are cancelled, which is every
push to an open PR.

## Why timers are the wrong tool

Cancellation and capacity waiting are different problems. Tuning one to fix the
other makes both worse.

- **Cancellation is a correctness problem.** A cancelled job must never be
  admitted, regardless of how long it has waited. No timeout value expresses
  "this specific job is no longer wanted."
- **Capacity waiting is a resource problem.** A job queued behind a one-chip
  pool is the system working correctly.

Shortening `pendingTimeout` to catch cancelled jobs starts failing healthy
builds during busy periods and still misses a job cancelled ten seconds after
it was queued. The fix must be event-driven.

Record the current bounds before changing anything, so a later change can be
attributed:

| Wait | Bounded by | Current value |
| --- | --- | --- |
| Unadmitted, waiting for quota | nothing in Kueue | unbounded |
| Admitted, Pod not Ready | `waitForPodsReady.timeout` | 10800s |
| Running | `activeDeadlineSeconds`, `max-exec-time-seconds` | 10800s |

Row 1 is the state this document is about. Nothing bounds it today except a
human.

## Phase 0: record the deployed versions

```bash
export MANAGER_CONTEXT=gke_cloud-tpu-inference-test_us-central1_ci-test-controller
export WORKER_CONTEXT=gke_cloud-tpu-inference-test_southamerica-west1_ci-test-southamerica-west1-worker
export NS=buildkite

helm list -n "$NS" --kube-context "$MANAGER_CONTEXT"

kubectl --context "$MANAGER_CONTEXT" -n "$NS" get deployment \
  -o custom-columns=NAME:.metadata.name,IMAGE:.spec.template.spec.containers[*].image

kubectl --context "$MANAGER_CONTEXT" -n "$NS" get pods \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[*].imageID}{"\n"}{end}'

kubectl --context "$MANAGER_CONTEXT" -n "$NS" \
  logs deployment/agent-stack-k8s --tail=100 | head -20
```

If more than one Helm release or Deployment appears, identify which one
advertises Buildkite queue `kube` before reading or changing anything.

Then capture the effective configuration:

```bash
export AGENT_STACK_RELEASE=REPLACE_FROM_HELM_LIST

helm --kube-context "$MANAGER_CONTEXT" get values "$AGENT_STACK_RELEASE" \
  -n "$NS" -a > /tmp/agent-stack-values.yaml

grep -iE 'cancel|poll|stale|grace|ttl|deadline' /tmp/agent-stack-values.yaml
```

**Gate:** exact chart version, image digest, and the full effective values file
are saved. Note whether any cancel-checker or poll-interval key is present, and
whether its value is zero or unset.

## Phase 1: reproduce deterministically

The reproduction must leave a job **unadmitted**, not merely pending on a node,
or it tests the wrong path.

1. Saturate the one-chip quota with a long-running job so the next one cannot be
   admitted.
2. Launch a build whose jobs will queue behind it.
3. Confirm the target Job is suspended and its Workload has no quota
   reservation:

```bash
export JOB_UUID=REPLACE_BUILDKITE_JOB_UUID
export JOB="buildkite-${JOB_UUID}"

kubectl --context "$MANAGER_CONTEXT" -n "$NS" get job "$JOB" \
  -o jsonpath='{.spec.suspend}{"\t"}{.spec.managedBy}{"\n"}'

kubectl --context "$MANAGER_CONTEXT" -n "$NS" get workloads -o json | jq -r \
  --arg JOB "$JOB" '.items[]
    | select(any(.metadata.ownerReferences[]?; .name == $JOB))
    | {name: .metadata.name, admission: .status.admission,
       conditions: [.status.conditions[] | {type, status, reason}]}'
```

4. Confirm no Pod exists on **either** cluster.
5. Cancel the build in Buildkite. Record the wall-clock time.

**Gate:** at cancellation time the Job is `suspend: true`, its Workload has no
admission, and no Pod exists anywhere. Anything else and you are reproducing a
different bug.

## Phase 2: confirm both properties

Run two arms. One alone proves nothing — the point is the contrast between them.

**Arm A — cancel while suspended (the failing case).** Using the Phase 1 setup,
cancel while the Job is suspended with no Pod anywhere, then keep watching well
past the point where quota frees up:

```bash
kubectl --context "$MANAGER_CONTEXT" -n "$NS" \
  logs deployment/agent-stack-k8s --since=15m --follow \
  | grep -iE "$JOB_UUID|cancel" &

watch -n 10 "kubectl --context $MANAGER_CONTEXT -n $NS get job $JOB \
  -o jsonpath='suspend={.spec.suspend} deleted={.metadata.deletionTimestamp}{\"\n\"}'"
```

Expected if the finding holds: a cancellation log line at cancel time, no Job
deletion, then later `suspend` flips to `false`, a Pod appears, and **no second
evaluation occurs**. Record the interval between the cancel log line and the Pod
appearing — that gap is the evidence.

**Arm B — cancel while a Pod is running (the control).** Let a job reach a
running Pod, then cancel. Expected: it is cancelled promptly.

Arm B passing while Arm A fails is what distinguishes "cancellation is broken"
from "cancellation is evaluated once, at the wrong moment". That distinction is
the entire upstream report.

**Gate:** you can state, with timestamps, that the controller evaluated the job
exactly once and that a Pod was created after that evaluation.

## Phase 3: interpret

| Evidence | Owner | Next action |
| --- | --- | --- |
| Arm A: one cancel log line, no re-evaluation, Pod created later | Agent Stack, edge-triggered and Pod-scoped | Confirms the finding. Go to Phase 4. |
| Arm A: no log line at all, even at cancel time | Different bug — the controller is not seeing the build | Check queue/tags and controller polling before assuming anything here applies. |
| Arm B also fails | Cancellation is broken generally, not just for delayed Pods | Stop. This handoff assumes Arm B works; re-scope. |
| Job deleted but worker Job still running | MultiKueue garbage collection | Check `multiKueue.gcInterval` and manager Kueue logs. This is the expensive failure — a TPU is held. |
| `deletionTimestamp` set, Job stuck `Terminating` | Kueue finalizer | Inspect `metadata.finalizers` and Kueue controller health. Capture why before removing one by hand. |

## Phase 4: remediation

Note the ordering change from a normal triage: **configuration is not expected
to help.** No poll interval makes an edge-triggered, Pod-scoped check into a
level-triggered, Job-scoped one. Confirm the effective values from Phase 0 for
the record, then move on rather than tuning.

1. **Report upstream.** This is an integration gap, not a misconfiguration.
   Include: Agent Stack evaluates cancellation once, at cancel time, against
   Pods; Kueue admission decouples Job creation from Pod creation, so the
   evaluation lands in a window where no Pod can exist; `spec.managedBy` on the
   manager Job; and the Arm A / Arm B timestamps. Note that `v0.46.2` already
   consults Buildkite job state before reaping a supposedly empty Job, so the
   capability exists — the question is which code path uses it and when.
   Note also that the race exists without MultiKueue; Kueue only widens it.
2. **Deploy the reconciler.** Phase 5. Given (1) is an upstream code change,
   treat this as the operative fix for as long as that takes, not as a stopgap
   to be tolerated for a week.
3. **RBAC**, only if something wants to act and cannot: widen exactly the verb
   and resource named in the denial, in the namespace it operates in.

## Phase 5: the reconciler

The two properties that make Agent Stack's check insufficient are the two this
must not repeat. It has to be **level-triggered** — reconciling desired against
actual on an interval, so it is correct no matter when the Pod appears — and it
must act on the **Job or Workload**, never the Pod, because deleting a Pod
leaves a Job that creates another one.

Contract:

- List Workloads in namespace `buildkite` on the **manager**.
- Derive the Buildkite job UUID from the owning Job's name
  (`buildkite-<UUID>`).
- Query the Buildkite REST API for each job's state.
- For any job in `canceled` or `canceling`, patch the Workload:

```bash
kubectl --context "$MANAGER_CONTEXT" -n "$NS" patch workload "$WL" \
  --type merge -p '{"spec":{"active":false}}'
```

Why deactivate rather than delete:

- it works regardless of Pod existence, and regardless of *when* it runs
  relative to admission — the two properties Agent Stack's check lacks;
- Kueue evicts the Workload and releases the quota reservation, which is the
  outcome that actually matters;
- the object survives briefly for inspection instead of vanishing.

Requirements: a read-only Buildkite API token from the existing secret path;
RBAC limited to `get`/`list`/`patch` on `workloads` in `buildkite`; a bounded
poll interval; structured logs correlating Buildkite job UUID, Workload name,
and action taken. It must be idempotent — patching an already-inactive Workload
is a no-op.

Test it against both windows, because passing only the first is the failure
mode being fixed:

- cancel **before** admission — the Workload is deactivated and never admitted;
- cancel **after** admission but before the Pod is Ready — the Workload is
  deactivated and the remote Job is torn down;
- cancel a job whose Pod is already running — Agent Stack still handles this, and
  the reconciler must not fight it or double-report.

**Gate:** in all three, quota is released within one poll interval, no TPU node
scales up on behalf of a cancelled job, and Buildkite reports the job once.

## Related but out of scope

Do not fix these here. Record anything you learn and move on.

- **Readiness.** `DESIGN.md` §"Agent Stack and Kueue readiness incompatibility"
  documents that the completed checkout container makes a healthy Pod
  `Ready=False`, and that `waitForPodsReady.timeout: 10800s` is the accepted
  compensation. Note in particular that the Job is reported not-pods-ready
  **for the whole run**, so splitting `timeout` from `recoveryTimeout` does not
  help — `timeout` is the one firing. That section already lists the options.
- **Concurrency versus quota.** The pipeline permits more concurrent one-chip
  jobs than the flavor's `nominalQuota`. That makes the queue longer, which
  makes this bug more expensive, but it is not its cause.
- **`pendingTimeout: 0`.** Unbounded waiting is deliberate for scarce TPU
  capacity. Changing it is a separate decision with its own evidence.

## Report template

```text
Phase:
Agent Stack chart version / image digest:
Kueue version:
Cancel-checker key present / value:
Arm A: cancel log timestamp / Pod creation timestamp / re-evaluated? :
Arm B: cancelled promptly? :
Buildkite build and job UUIDs:
Job state at cancellation (suspend / admission / pods):
Controller log evidence (quote or "absent"):
Phase 3 row matched:
Fix applied (one only):
Time from cancel to quota release:
TPU node scaled on behalf of a cancelled job: yes | no
Result: PASS | FAIL | BLOCKED
Next safe action:
```

Do not report this resolved from configuration alone. Resolution requires a
reproduction in which a cancelled, unadmitted job releases its quota and no TPU
node is ever provisioned for it.
