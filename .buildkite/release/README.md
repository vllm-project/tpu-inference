# tpu-inference-release

One Buildkite pipeline publishes each tpu-inference release: it builds the
`vllm/vllm-tpu` image and the two wheels (`tpu-inference` and the combined
`vllm-tpu`) once from the release branch, checks them, waits for approval, then
publishes those same bytes to Docker Hub and PyPI.

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
     to be a version of the branch's `X.Y.Z`), warns if `vX.Y.Z` is already
     on Docker Hub, and checks PyPI doesn't have the version yet;
   - **builds once**: the image, from the release checkout's Dockerfile and vLLM
     pin, staged in GAR as `vllm-tpu-release:<version>-build<N>`, and both
     wheels;
   - **checks**: the wheels resolve together from pypi.org for CPython 3.12
     on x86_64 Linux (`check_wheels.sh`).
4. **Approve** the publish block. It names the staged digest and asks whether
   to move `latest` (never for a pre-release; no default) and whether an
   existing Docker Hub tag may be replaced.
5. The build publishes what it built and checked:
   - the staged image, copied by digest to Docker Hub as `vllm/vllm-tpu:vX.Y.Z`
     (and `latest` if chosen);
   - `tpu-inference`, then `vllm-tpu`, to PyPI with Trusted Publishing
     (`.buildkite/scripts/publish_wheel.sh`);
   - then checks, without credentials, that an anonymous pull of the tag is the
     staged digest and that `vllm-tpu==X.Y.Z` resolves from pypi.org.

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
      branches: "main releases/v*"
      agents:
        queue: cpu
      command: buildkite-agent pipeline upload .buildkite/release/pipeline.yml
  ```

## One-time setup outside this repo

- The `DOCKERHUB_TOKEN` secret (Docker Hub user `tpuinference`) readable by
  this pipeline on `releases/*` branches.
- The agents on the `cpu` queue can write to
  `us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/tpu-inference-ci`
  (as CI's image builders on `cpu_64_core` already do) and have `docker buildx` with
  `imagetools create --prefer-index`.
- PyPI: a Google Trusted Publisher on both `tpu-inference` and `vllm-tpu` for
  the service account of the agents that run the publish steps. Without it,
  `publish_wheel.sh` can't mint a token and uploads nothing.
