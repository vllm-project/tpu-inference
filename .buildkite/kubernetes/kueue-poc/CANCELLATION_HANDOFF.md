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

Buildkite delivers cancellation **to a running agent**. A job with no agent has
no recipient. Under MultiKueue this is not an edge case, it is the normal state
for most of a job's life:

```text
Buildkite cancel  ->  agent (Agent API)  ->  process exits
                       ^
                       |
              does not exist yet
```

The manager Job is suspended and podless for its entire life — `spec.managedBy`
tells the built-in Job controller to stand down so MultiKueue owns it. So the
only thing that can act on a cancelled, unstarted job is the controller,
comparing Buildkite state against Kubernetes state.

Agent Stack has machinery for exactly this. The suspected gap is its **trigger
condition**: if the cancel-checker is driven off a Pending Pod, a manager Job
that never has a Pod is invisible to it until MultiKueue dispatches and a Pod
appears on the worker — which matches the observed timing precisely.

**This is a hypothesis, not a finding.** Phase 2 settles it.

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

## Phase 2: determine whether the controller sees it

This is the decisive phase. Start the watch before cancelling if you can.

```bash
kubectl --context "$MANAGER_CONTEXT" -n "$NS" \
  logs deployment/agent-stack-k8s --since=15m --follow \
  | grep -iE "$JOB_UUID|cancel"
```

In parallel, poll the objects for at least three times the cancel-checker poll
interval found in Phase 0, or five minutes if none was found:

```bash
watch -n 10 "kubectl --context $MANAGER_CONTEXT -n $NS get job $JOB \
  -o jsonpath='{.metadata.deletionTimestamp}{\" \"}{.metadata.finalizers}{\"\n\"}'"
```

**Gate:** you can state, with a log line or its absence as evidence, whether the
controller ever evaluated this job after cancellation.

## Phase 3: interpret

| Evidence | Owner | Next action |
| --- | --- | --- |
| No log line mentions the UUID after cancellation | Agent Stack trigger condition | The Pod-gated hypothesis holds. File upstream with the podless-manager-Job explanation, then apply the Phase 5 bridge. |
| Log line says cancelled, Job survives, no `deletionTimestamp` | Controller RBAC | Check its Role for `delete` on `jobs.batch` in `buildkite`. |
| `deletionTimestamp` set, Job stuck `Terminating` | Kueue finalizer | Inspect `metadata.finalizers` and Kueue controller health. Do not remove a finalizer by hand before capturing why it is stuck. |
| Manager Job deleted, worker Job still running | MultiKueue garbage collection | Check `multiKueue.gcInterval` and manager Kueue logs for the remote object. This is the expensive failure — a TPU is held. |
| Cancel-checker key absent or zero in effective values | Configuration | Set a sane poll interval, redeploy through the Helm/GitOps source, repeat Phase 1. |

Record which row matched. Do not apply more than one fix at a time.

## Phase 4: candidate fixes, in order

1. **Configuration.** If a cancel-checker poll interval exists and is unset or
   zero, set it in the Helm values source — never `kubectl edit` — and re-run
   Phase 1. Cheapest possible outcome; try it first.
2. **RBAC.** If the controller wants to delete and cannot, widen only the verb
   and resource named in the denial, in the namespace it operates in.
3. **Upstream.** If the checker never evaluates podless Jobs, this is an
   integration gap between Agent Stack and MultiKueue rather than a
   misconfiguration. Report it with: the podless-by-design explanation,
   `spec.managedBy` on the manager Job, and the Phase 2 log evidence. Note that
   `v0.46.2` already checks Buildkite job state before reaping a supposedly
   empty Job, so the capability exists and the question is which code path
   consults it.
4. **Local bridge.** Phase 5, only while 3 is open.

## Phase 5: the reconciler bridge

Only if Phase 3 shows the controller never evaluates these Jobs. Keep it small,
keep it removable, and record it as temporary.

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

- it works regardless of Pod existence, which is the property Agent Stack's
  checker appears to lack;
- Kueue evicts the Workload and releases the quota reservation, which is the
  outcome that actually matters;
- the object survives briefly for inspection instead of vanishing.

Requirements: a read-only Buildkite API token from the existing secret path;
RBAC limited to `get`/`list`/`patch` on `workloads` in `buildkite`; a bounded
poll interval; structured logs correlating Buildkite job UUID, Workload name,
and action taken. It must be idempotent — patching an already-inactive Workload
is a no-op.

**Gate:** a cancelled, unadmitted job has its quota released within one poll
interval, and no TPU node scales up on its behalf.

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
