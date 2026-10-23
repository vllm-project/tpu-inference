# tpu-inference-release

One Buildkite pipeline for each tpu-inference release: it settles the version,
builds what the release ships, waits for approval, then publishes it. It builds
the two wheels (`tpu-inference` and the combined `vllm-tpu`) once, checks them,
and publishes those same bytes to PyPI. The Docker image steps are still
placeholders, to be filled in by a follow-up.

The pipeline's settings upload [`pipeline.yml`](pipeline.yml) from the build's
own checkout, and only for builds of `main` and `releases/v*`. A release branch
runs the release tooling (this directory and `.buildkite/scripts`) it was cut
with; a tooling fix a release needs goes to main and is cherry-picked to the
branch, or committed to the branch directly for one-off debugging. A release
branch cut before this tooling existed needs it cherry-picked before its first
release build.

## Cut a release

1. Check that **tpu-inference-ci** passed on the release commit of
   `releases/vX.Y.Z` (all three `MODEL_IMPL_TYPE` runs). The pipeline doesn't
   check this yet.
2. Start a **New Build** of **tpu-inference-release** on `releases/vX.Y.Z`, at
   that commit. For a pre-release set `RELEASE_VERSION=vX.Y.Zrc1`.
3. The build then, without anyone's help:
   - **preflight**: settles the version (branch or `RELEASE_VERSION`, which has
     to be a version of the branch's `X.Y.Z`), records it for every later
     step, and checks PyPI doesn't have it yet;
   - **builds once**: both wheels (the image is a placeholder for now);
   - **checks**: the wheels resolve together from pypi.org for CPython 3.12
     on x86_64 Linux (`check_wheels.sh`);
   - **plans the publish**: uploads the approval block and the publish steps.
4. **Approve** the publish block. PyPI releases can't be replaced.
5. The build publishes what it built and checked: `tpu-inference`, then
   `vllm-tpu`, to PyPI with Trusted Publishing
   (`.buildkite/scripts/publish_wheel.sh`). Publishing the image and checking
   what users get are placeholders for now.

Build env knobs:

| Env | Effect |
|---|---|
| `RELEASE_VERSION=vX.Y.Zrc1` | release a pre-release under that version |
| `RELEASE_APPROVER_TEAMS=a,b` | only these Buildkite teams can approve |

## Pipeline settings (Buildkite, made by hand)

- Repository `https://github.com/vllm-project/tpu-inference.git`.
- GitHub: no builds on push or pull request (manual only).
- Steps:

  ```yaml
  steps:
    - label: ":pipeline: Upload the release pipeline"
      branches: "main releases/v*"
      agents:
        queue: release-tpu-inference
      command: buildkite-agent pipeline upload .buildkite/release/pipeline.yml
  ```

## One-time setup outside this repo

- PyPI: a Google Trusted Publisher on both `tpu-inference` and `vllm-tpu` for
  the service account of the agents on the `release-tpu-inference` queue.
  Without it, `publish_wheel.sh` can't mint a token and uploads nothing.

## Try the pipeline

Approving the publish block uploads the wheels to PyPI, where a version can
never be reused. To try the pipeline without publishing, stop at the approval
block: check that preflight settles the version and that both wheels build and
pass `check_wheels.sh`, then cancel the build instead of approving it. The
steps run only on `releases/vX.Y.Z` branches, so try it on a release branch
that has this directory.
