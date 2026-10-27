# tpu-inference-release

One Buildkite pipeline for each tpu-inference release: it settles the version,
builds what the release ships, waits for approval, then publishes it. The Docker
image and PyPI steps are placeholders for now; follow-ups fill them in.

The pipeline's settings upload [`pipeline.yml`](pipeline.yml) from the build's
own checkout, and only for builds of `releases/v*`. A release branch runs the
release tooling (this directory) it was cut with; a tooling fix a release needs
goes to main and is cherry-picked to the branch.

## Cut a release

1. Check that **tpu-inference-ci** passed on the release commit of
   `releases/vX.Y.Z` (all three `MODEL_IMPL_TYPE` runs). The pipeline doesn't
   check this yet.
2. Start a **New Build** of **tpu-inference-release** on `releases/vX.Y.Z`, at
   that commit. For a pre-release set `RELEASE_VERSION=vX.Y.Zrc1`.
3. The build then, without anyone's help:
   - **preflight**: settles the version (branch or `RELEASE_VERSION`, which has
     to be a version of the branch's `X.Y.Z`) and records it for every later
     step;
   - **builds**: the image and the wheels (placeholders for now);
   - **plans the publish**: uploads the approval block and the publish steps.
4. **Approve** the publish block.
5. The build publishes and checks what users get (placeholders for now).

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
      branches: "releases/v*"
      agents:
        queue: release-tpu-inference
      command: buildkite-agent pipeline upload .buildkite/release/pipeline.yml
  ```

## Try the pipeline

While the build and publish steps are placeholders, a build builds and
publishes nothing, even when approved. Run it on a release branch that has this
directory (cut after it reached main, or with it cherry-picked) and check that
preflight settles the version and that the approval block and the placeholder
steps appear.
