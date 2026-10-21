# tpu-inference-release

One Buildkite pipeline publishes each tpu-inference release's Docker image: it
builds `vllm/vllm-tpu` once from the release branch, stages it in GAR, waits for
approval, then copies those same bytes to Docker Hub. PyPI is not handled here
yet (`.github/workflows/release.yml` publishes it today); the pipeline has
placeholder steps for it.

The pipeline's settings upload [`pipeline.yml`](pipeline.yml) **from main**,
whatever branch a build is on, and every step runs main's tooling (this
directory) against the build's checkout. A release branch supplies only the
code being released: its `docker/Dockerfile` and `.buildkite/vllm_lkg.version`.

## Cut a release

1. Check that **tpu-inference-ci** passed on the release commit of
   `releases/vX.Y.Z` (all three `MODEL_IMPL_TYPE` runs). The pipeline doesn't
   check this yet.
2. Start a **New Build** of **tpu-inference-release** on `releases/vX.Y.Z`, at
   that commit. For a pre-release set `RELEASE_VERSION=vX.Y.Zrc1`.
3. The build then, without anyone's help:
   - **preflight**: settles the version (branch or `RELEASE_VERSION`, which has
     to be a version of the branch's `X.Y.Z`) and warns if `vX.Y.Z` is already
     on Docker Hub;
   - **builds once**: the image, from the release checkout's Dockerfile and vLLM
     pin, staged in GAR as `vllm-tpu:release-<version>-build<N>` (CI's own
     `vllm-tpu` tags are `<commit>-<vllm commit>-<tpu>`, so they never collide).
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
| `RELEASE_VERSION=vX.Y.Zrc1` | publish a pre-release under that version |
| `RELEASE_APPROVER_TEAMS=a,b` | only these Buildkite teams can approve |

## Pipeline settings (Buildkite, made by hand)

- Repository `https://github.com/vllm-project/tpu-inference.git`.
- GitHub: no builds on push or pull request (manual only).
- Steps:

  ```yaml
  steps:
    - label: ":pipeline: Upload the release pipeline from main"
      agents:
        queue: cpu
      command: |
        git fetch -q origin main
        tools="$$(git rev-parse FETCH_HEAD)"
        buildkite-agent meta-data set release-tools-commit "$$tools"
        git show "$$tools:.buildkite/release/pipeline.yml" | buildkite-agent pipeline upload
  ```

## One-time setup outside this repo

- The `DOCKERHUB_TOKEN` secret (Docker Hub user `tpuinference`) readable by
  this pipeline on `releases/*` branches.
- The agents on the `cpu` queue can write to
  `us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/tpu-inference-ci`
  (as CI's image builders on `cpu_64_core` already do) and have `docker buildx` with
  `imagetools create --prefer-index`.
