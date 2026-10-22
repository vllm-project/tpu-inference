# tpu-inference-release

One Buildkite pipeline for each tpu-inference release: it settles the version,
builds what the release ships, waits for approval, then publishes it. The Docker
image `vllm/vllm-tpu` is built once from the release branch, staged in GAR, and
after approval copied by digest to Docker Hub, so users get the exact bytes that
were built. The PyPI steps are placeholders, to be filled in by a follow-up
(PyPI is still published by `.github/workflows/release.yml`).

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
     step; warns if `vX.Y.Z` is already on Docker Hub;
   - **builds**: the image, from the release checkout's `docker/Dockerfile` and
     vLLM pin (`.buildkite/vllm_lkg.version`), staged in GAR as
     `vllm-tpu:release-<version>-build<N>` (CI's own `vllm-tpu` tags are
     `<commit>-<vllm commit>-<tpu>`, so they never collide); the wheels are a
     placeholder for now;
   - **plans the publish**: uploads the approval block and the publish steps.
4. **Approve** the publish block. It names the staged digest and asks whether
   to move `latest` (never for a pre-release; no default) and whether an
   existing Docker Hub tag may be replaced.
5. The build publishes what it built:
   - the staged image, copied by digest to Docker Hub as `vllm/vllm-tpu:vX.Y.Z`
     (and `latest` if chosen);
   - then checks, without credentials, that an anonymous pull of the tag is the
     staged digest.

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

A build stages the image in GAR before anyone approves, and nothing reaches
Docker Hub until the publish block is approved. To try the pipeline on a
release branch, e.g. `releases/v0.31.0`, cancel the build at the approval
block, and delete the staged `vllm-tpu:release-<version>-build<N>` afterwards.

Before a change to this directory is on main, point the settings step at the
branch under review instead (`git fetch -q origin <branch>`), and set it back
to `main` afterwards.

## One-time setup outside this repo

- The `DOCKERHUB_TOKEN` secret (Docker Hub user `tpuinference`) readable by
  this pipeline on `releases/*` branches.
- The agents on the `release-tpu-inference` queue can write to
  `us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/tpu-inference-ci` and
  have `docker buildx` with `imagetools create --prefer-index`.
- That repository's cleanup policy keeps `vllm-tpu:release-*` tags at least
  until they are published: CI pushes many `vllm-tpu` tags a day, so a
  "keep the newest N" rule would otherwise remove a staged release image.
