# Model steps on the Kueue fleet

One file per model, named after its bare-metal file in `..`, with the same
step keys, so the two lanes read side by side. A file in `..` with no TPU step
to migrate has no file here; its work sits on cpu behind a TODO.

`upload_kube_lane` in `.buildkite/scripts/bootstrap.sh` uploads every file here
as one pipeline, beside `pipeline_build.yml`, for a kube build with
`CI_LANES=models` (see `ci_fleet.sh`). Like `upload_models_and_features.sh` on
bare metal, it drops each file's `steps:` line and concatenates the rest. So a
file holds steps and comments only: an anchor cannot be seen from another
file, and a top-level `env:` would reach every step of the build and win over
a name passed to `bk build create --env`.

A build is one generation. The defaults are v6e; a v7x build sets
`TPU_VERSION=tpu7x`, `KUBE_SHAPE_SINGLE`/`KUBE_SHAPE_MULTI` to tpu7x shapes and
`TENSOR_PARALLEL_SIZE_SINGLE=2`, the values the bare-metal uploader exports for
its v7x group. Several models run only on v7x: on v6e their steps record
themselves unverified, as they do on bare metal.

Tests are not `soft_fail` here; each record step depends on its test with
`allow_failure` instead. See `../../features/kube/README.md`.

Chips, topology, Kueue queue, caches and deadlines come from the cluster-side
profile registry in ci-infra; a step names only a shape and a command.

`validate_pipeline_metadata.sh` skips this folder: its checks describe the
bare-metal files' labels and record steps, not these.
