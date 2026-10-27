# tpu-inference-release

One Buildkite pipeline publishes the Docker image of each tpu-inference
release, `vllm/vllm-tpu`. It builds the image once from the release branch,
stages it in GAR, waits for approval, then copies those same bytes to Docker
Hub, so users pull exactly the image that was built and approved. It runs on
the `release-tpu-inference` queue.

The pipeline's settings upload [`pipeline.yml`](pipeline.yml) from the build's
own checkout, and only for builds of `releases/v*`. A release branch runs the
release tooling (this directory) it was cut with; a tooling fix a release needs
goes to main and is cherry-picked to the branch.

## Cut a release

1. Check that **tpu-inference-ci** passed on the release commit of
   `releases/vX.Y.Z` (all three `MODEL_IMPL_TYPE` runs).
2. Start a **New Build** of **tpu-inference-release** on `releases/vX.Y.Z`, at
   that commit. For a pre-release set `RELEASE_VERSION=vX.Y.Zrc1`.
3. The build then, without anyone's help:
   - **preflight**: resolves the version (the branch, or `RELEASE_VERSION`,
     which must be a version of the branch's `X.Y.Z`) and warns if `vX.Y.Z`
     is already on Docker Hub;
   - **builds once**: the image, from the release checkout's
     `docker/Dockerfile` and vLLM pin (`.buildkite/vllm_lkg.version`), staged
     in GAR as `vllm-tpu:release-<version>-build<N>`. CI's own `vllm-tpu` tags
     are `<commit>-<vllm commit>-<tpu>`, so the two never collide.
4. **Approve** the publish block. It names the staged digest and asks whether
   to move `latest` (never for a pre-release; no default) and whether an
   existing Docker Hub tag may be replaced.
5. The build publishes what it built:
   - the staged image, copied by digest to Docker Hub as
     `vllm/vllm-tpu:vX.Y.Z` (and `latest` if chosen);
   - then checks, without credentials, that an anonymous request for the tag
     returns the staged digest.

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
      - label: ":pipeline: Upload the release pipeline"
        branches: "releases/v*"
        agents:
          queue: release-tpu-inference
        command: buildkite-agent pipeline upload .buildkite/release/pipeline.yml
    ```

## One-time setup outside this repo

- Buildkite: the `DOCKERHUB_TOKEN` secret (Docker Hub user `tpuinference`)
  readable by this pipeline.
- The `release-tpu-inference` agents can push to
  `us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/tpu-inference-ci` and
  have `docker buildx` with `imagetools create --prefer-index`.
- That repository's cleanup policy keeps `vllm-tpu:release-*` tags at least
  until they are published.
