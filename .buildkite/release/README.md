# tpu-inference-release

One Buildkite pipeline for each tpu-inference release: it settles the version,
builds what the release ships, waits for approval, then publishes it. This is
the pipeline's skeleton: the Docker image and PyPI steps are placeholders, to be
filled in by follow-ups (PyPI is still published by
`.github/workflows/release.yml`).

The pipeline's settings upload [`pipeline.yml`](pipeline.yml) **from main**,
whatever branch a build is on, and every step runs main's tooling (this
directory) against the build's checkout. A release branch supplies only the
code being released.

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
    - label: ":pipeline: Upload the release pipeline from main"
      agents:
        queue: release-tpu-inference
      command: |
        git fetch -q origin main
        tools="$$(git rev-parse FETCH_HEAD)"
        buildkite-agent meta-data set release-tools-commit "$$tools"
        git show "$$tools:.buildkite/release/pipeline.yml" | buildkite-agent pipeline upload
  ```

## Try the pipeline

While the build and publish steps are placeholders, a build builds and
publishes nothing, even when approved, so it is safe to run on any release
branch, e.g. `releases/v0.31.0`: check that preflight settles the version and
that the approval block and the placeholder steps appear.

Before this directory is on main, point the settings step at the branch under
review instead (`git fetch -q origin <branch>`), and set it back to `main`
afterwards.
