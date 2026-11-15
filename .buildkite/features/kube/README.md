# Feature steps on the Kueue fleet

One file per feature, named after its bare-metal file in `..`, with the same
step keys, so the two lanes read side by side. A file in `..` with no TPU step
to migrate has no file here; its work sits on cpu behind a TODO.

`upload_kube_lane` in `.buildkite/scripts/bootstrap.sh` uploads every file here
as one pipeline, beside `pipeline_build.yml`, for a kube build with
`CI_LANES=features` (see `ci_fleet.sh`). Like `upload_models_and_features.sh` on
bare metal, it drops each file's `steps:` line and concatenates the rest. So a
file holds steps and comments only: an anchor cannot be seen from another
file, and a top-level `env:` would reach every step of the build and win over
a name passed to `bk build create --env`.

A build is one generation. The defaults are v6e; a v7x build sets
`TPU_VERSION=tpu7x` and `KUBE_SHAPE_SINGLE`/`KUBE_SHAPE_MULTI` to tpu7x shapes.
The multi-host steps run only in a v7x build, as on bare metal, where the
uploader leaves `features/multi-host.yml` out of the v6e group.

The bare-metal files mark every test `soft_fail` so the record step after it is
still reached. That is not needed here: `allow_failure` on the record step's
own dependency reaches it just the same, and `buildkite-agent step get outcome`
reports hard_failed where it used to report soft_failed, which
`record_step_result.sh` already treats alike. The build goes red either way -
the recorder exits non-zero on anything but a pass - but the test itself now
reads as failed rather than tolerated, which is the question this lane asks.

Chips, topology, Kueue queue, caches and deadlines come from the cluster-side
profile registry in ci-infra; a step names only a shape and a command.

`validate_pipeline_metadata.sh` skips this folder: its checks describe the
bare-metal files' labels and record steps, not these.
