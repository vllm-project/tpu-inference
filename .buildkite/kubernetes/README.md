# TPU steps on Kubernetes

`pipeline_kube.yaml` is `pipeline_jax.yml` run on GKE. Every TPU step goes
through one path: the launcher submits a workload and streams it back.

```yaml
- label: "unit tests"
  agents:
    queue: kube
  plugins:
    - kubernetes:
        podTemplate: tpu-launcher
  command: .buildkite/kubernetes/run.sh v6e-8-2x4 pytest tests/
```

The step names a **profile** and a command. `run.sh` resolves the image and
the tokens, names the environment to forward, picks the manifest, and hands
off to `/opt/launcher/launch`, which lives in the launcher image the cluster
publishes (see `ci-infra`, `terraform/gcp_old/tpu-inference/k8s`).

The split is deliberate. The *shape* of a job lives here, next to the tests it
runs, so changing it is a normal PR. The *facts about the hardware* - chip
count, topology, node labels, queue names, deadlines - come from a profile
registry generated in `ci-infra` from the same tfvars that create the node
pools and the Kueue queues. Manifests reference them as `${CHIPS}`,
`${TOPOLOGY}` and so on, so a profile cannot change region, reservation or chip
count without the queues following, and no placement detail is restated here to
go stale. `/opt/launcher/launch --profile bogus -- true` lists the profiles.

## Why a launcher even for one pod

agent-stack-k8s can only create a `batch/v1` Job, and a Job cannot span hosts,
so multi-host slices and cross-host prefill/decode need a launcher regardless.
Routing single-pod work through it too keeps one code path and, more
importantly, keeps the Buildkite agent *outside* the Kueue workload:

- **No `retry:` for preemption.** Kueue evicts the workload; the agent is on a
  CPU pod it does not manage, so the step log pauses and resumes instead of the
  build failing with a lost agent.
- **No chips held while waiting.** Admission and node scale-up cost a CPU pod,
  not a TPU.

## Manifests

`manifests/test.yaml` is one pod on one host holding every chip on the node.
It mounts the two caches the cluster publishes as PersistentVolumeClaims -
`jax-cache` (compilation cache) at `/cache/jax` and `hf-cache` (model weights)
at the HF hub path - which are GCS FUSE mounts on buckets in the cluster's own
region. A claim named `jax-cache` means the local cache wherever the pod is
dispatched, so the manifest carries no bucket name. The gcsfuse file cache in
front of them is a RAM-backed `emptyDir` sized per machine type
(`${FUSE_CACHE_SIZE}`, from the profile).

`WORKLOAD_MANIFEST` in `run.sh` picks a different manifest for a shape that
needs one; everything else stays the same. **Write your own** - the launcher
submits any `Job` or `JobSet` in this repo. When writing a JobSet, the axis
matters:

| You want | Express it as |
|---|---|
| Distinct roles (prefill, decode, benchmark) | one `replicatedJob` each, `replicas: 1`, `parallelism: 1` |
| N interchangeable independent pods | one `replicatedJob`, `replicas: N`, `parallelism: 1` |
| The hosts of one multi-host TPU slice | one `replicatedJob`, `replicas: 1`, `parallelism: <hosts>`, `completionMode: Indexed` |

`replicas` creates separate Jobs; `parallelism` creates pods *inside* one Job.
Only the last row wants `parallelism > 1`: the pods are hosts of one slice and
their index within the Job is what identifies each host.

What the launcher fills in: `metadata.name`, `namespace`, the Kueue queue
label, correlation labels, the `ownerReference`, and the `args` of the
container named `workload`. Everything else is yours; `${CHIPS}`,
`${TOPOLOGY}`, `${ACCELERATOR_LABEL}`, `${NUM_HOSTS}`, `${MAX_RUNTIME}`,
`${FUSE_CACHE_SIZE}`, `${IMAGE}`, `${WORKLOAD_NAME}` and `${KUEUE_QUEUE}` pull
values from the named profile. The step's own environment is substituted too,
so a manifest can pin something like `${BUILDKITE_COMMIT}`.

Referencing a name is how you require it. The profile names are always defined,
so a `${NAME}` left over after they are applied is a name nothing provides - a
typo, or a variable since removed - and the launcher refuses to submit, listing
what was asked for. (Write `$$` for a literal dollar; a bare `$` is a template
error.) Kubernetes would catch most of these anyway - an unresolved `${NAME}` is
not a valid quantity, integer or label value - but it accepts one in a plain
string like an env `value:` without complaint.

`${ACCELERATOR_LABEL}`, `${TOPOLOGY}` and `${CHIPS}` are required outright: a
manifest that never mentions them is rejected. Unlike the names above,
hardcoding placement fails *silently* - the workload still requests TPU chips,
so Kueue admits it against the profile's quota and the pod lands on whatever
pool has room. That is a different topology than the profile promised, or the
right one by luck, and nothing in the run looks wrong.

Because the manifest is repo-controlled it is validated before submission. The
workload namespace enforces PodSecurity `baseline`, which rejects privileged
containers, `hostPath` volumes and host networking outright; the launcher adds
what admission cannot judge - the manifest may not name a different Kueue
queue than `--profile` resolves to, a workload pod may not run as a service
account the cluster has not published for workloads, and the image must come
from an allow-listed registry.

## Running the suite

`run.sh` is this repo's counterpart to `scripts/run_in_docker.sh`. The mapping:

| bare metal | kubernetes |
|---|---|
| `agents: { queue: tpu_v6e_8_queue }` | `run.sh v6e-8-2x4 ...` |
| `run_in_docker.sh bash <script>` | `bash <script>` - the pod is the container |
| `-e NAME` on `docker run` | `FORWARD` list in `run.sh`, once |
| `vllm-tpu:$BUILDKITE_COMMIT` local tag | the same image from Artifact Registry |

Two things worth knowing:

- **Forwarding is by name.** `run.sh` forwards every `BUILDKITE_*` variable
  (except the command, the plugin list and the agent's local Job API socket)
  plus a named list of test variables; an unset name is skipped rather than
  injected empty. `BUILDKITE_AGENT_ACCESS_TOKEN` is among them, which is what
  lets a workload run `buildkite-agent artifact upload` itself - the agent is
  in a different pod, so `artifact_paths` cannot see test output.
- **Tokens come from Buildkite and Secret Manager, not Kubernetes Secrets.**
  `HF_TOKEN` is the Buildkite cluster secret; the Test Engine analytics token
  is read from Secret Manager. Neither is applied to a cluster.

The image is built once by `build_docker` on the `cpu` queue and its exact
tag handed to the TPU steps through `buildkite-agent meta-data`. Setting
`WORKLOAD_IMAGE` on the build pins one instead and skips the build.

```bash
bk build create --pipeline tpu-commons/kube-dev --branch <branch>            # the per-push suite
bk build create --pipeline tpu-commons/kube-dev --branch <branch> --env NIGHTLY=1
bk build create --pipeline tpu-commons/kube-dev --branch <branch> --env SKIP_PART2=1
```

`VLLM_XLA_CHECK_RECOMPILATION=1` is set on every step, as `run_in_docker.sh`
does: it is what makes JAX persist small and quick compilations, without which
a compilation cache built here plateaus at about twice the warm time. The
cache namespace (`CACHE_NAMESPACE`, default `jax0.11.0_tputpu6e`) is
overridable per build, which is how a cold cache is measured without
disturbing the real one.
