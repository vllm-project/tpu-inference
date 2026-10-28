# vllm-tpu-release

One Buildkite pipeline publishes the Docker image (`vllm/vllm-tpu`) and two
wheels (`tpu-inference` and the combined `vllm-tpu`) of each tpu-inference
release. It builds the image and both wheels once from the release branch,
stages the image in GAR and checks the wheels, waits for approval, then
publishes those same bytes to Docker Hub and PyPI, so users get exactly what
was built and approved. It runs only in the **TPU Release** cluster, on the
`release-tpu-inference` queue.

The pipeline's settings upload [`pipeline.yml`](pipeline.yml) from the build's
own checkout, and only for builds of `releases/v*`. A release branch runs the
release tooling (`.buildkite/release` and `.buildkite/scripts`) it was cut
with; a tooling fix a release needs goes to main and is cherry-picked to the
branch.

## Cut a release

1. Run the full CI on the release commit: a **New Build** of
   **tpu-inference-ci** on `releases/vX.Y.Z` with `NIGHTLY=1`, once for each
   `MODEL_IMPL_TYPE` (`auto`, `flax_nnx`, `vllm`). This is not the scheduled
   nightly, which runs on `main`. Without `NIGHTLY=1` a build runs only the
   smoke tests (BVT) and skips the model and feature tests. All three builds
   must pass; the approver checks them, the pipeline doesn't.
2. Start a **New Build** of **vllm-tpu-release** on `releases/vX.Y.Z`, at
   that commit. For a pre-release set `RELEASE_VERSION=vX.Y.Zrc1`.
3. The build then, without anyone's help:
   - **preflight**: resolves the version (the branch, or `RELEASE_VERSION`,
     which must be a version of the branch's `X.Y.Z`), checks PyPI doesn't have
     the version yet, and warns if `vX.Y.Z` is already on Docker Hub;
   - **builds once**: the image, from the release checkout's
     `docker/Dockerfile` and vLLM pin (`.buildkite/vllm_lkg.version`), staged
     in GAR as `vllm-tpu:release-<version>-build<N>` (CI's own `vllm-tpu` tags
     are `<commit>-<vllm commit>-<tpu>`, so the two never collide), and both
     wheels (`tpu-inference` and `vllm-tpu`);
   - **checks**: the wheels resolve together from pypi.org for CPython 3.12
     on x86_64 Linux (`check_wheels.sh`).
4. **Approve** the publish block. It names the staged digest and asks whether
   to move `latest` (never for a pre-release; no default) and whether an
   existing Docker Hub tag may be replaced. PyPI releases can't be replaced.
5. The build publishes what it built and checked:
   - copies the staged image by digest to Docker Hub as
     `vllm/vllm-tpu:vX.Y.Z` (and `latest` if chosen);
   - uploads `tpu-inference`, then `vllm-tpu`, to PyPI with Trusted Publishing
     (`.buildkite/scripts/publish_wheel.sh`), waiting until each wheel installs
     from PyPI;
   - checks, without credentials, that an anonymous request for the Docker Hub
     tag returns the staged digest and that `vllm-tpu==X.Y.Z` resolves from
     pypi.org.

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
- PyPI: a Google Trusted Publisher on both `tpu-inference` and `vllm-tpu` for
  the service account of the agents on the `release-tpu-inference` queue.
  Without it, `publish_wheel.sh` can't mint a token and uploads nothing.
