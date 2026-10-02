# Workload manifests

Handed to the launcher with `--manifest`, for workloads the built-in one-pod
Job cannot express: several roles that talk to each other, or the hosts of one
multi-host slice.

`workloads/` names what it runs. `qwen3-coder-480b-1p1d.yaml` serves that model
in that topology and nothing else, so it is named for the model and the shape.
One of these belongs to one step.

A manifest that is told what to run instead - the step passing a command it
executes - is a template, named for the bring-up rather than any benchmark,
and several steps can share one. A workload that acquires a second caller has
to be parameterised first, which makes it a template and renames it.

`v7x/` predates both and is neither. No pipeline file names it: it is applied
with kubectl by .buildkite/scripts/daily_run_gke_disagg.sh, which the
tpu-inference-disagg-gke-benchmark pipeline runs nightly against the older GKE
cluster. Grepping the pipelines makes it look dead, so check that script before
changing anything under it.

## What a manifest has to say

Only the hardware it wants, and what is genuinely its own:

- `nodeSelector` with the accelerator and topology, and a `google.com/tpu`
  count. Those three pick the queue, so they cannot come from anywhere else.
- Its containers, and `activeDeadlineSeconds` if the fleet's default is wrong
  for it.

Everything else comes from the fleet, through one annotation on the JobSet:

```yaml
metadata:
  annotations:
    tpu-ci.google.com/defaults: standard
```

That supplies the compilation and model caches with their mounts, the gcsfuse
sidecar settings, the TPU toleration, the service account, `restartPolicy`, the
TTL, the memory request for whichever host the pod lands on, and the retry rules
that let a run survive its node being repaired. It is defined once in ci-infra,
in `kueue/launcher/pod_defaults.yaml`.

The merge is additive, so a role can still declare a volume of its own, or
override anything it needs to differ on.

Roles that hold no chips inherit only the eviction policy: the caches are sized
from a TPU host's memory and a chipless role does not run on one, so it states
what it needs itself. The benchmark client here is one.

Do not add a CPU request. It is a scheduling floor checked against the template
the autoscaler builds for the shape, and one large enough to matter stops the
pool building nodes at all.
