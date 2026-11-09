# vllm-tpu-release

One Buildkite pipeline publishes the Docker image (`vllm/vllm-jax`, plus
transitional `vllm/vllm-tpu`), two primary distributions (`vllm-jax-core`, this
repo, and the combined `vllm-jax`), and two transitional empty meta-wheels
(`tpu-inference` and `vllm-tpu`) of each vllm-jax release. The names live in
`lib.sh` (`CORE_PROJECT`, `LEGACY_CORE_PROJECT`, `COMBINED_PROJECT`,
`LEGACY_COMBINED_PROJECT`, `RELEASE_DOCKERHUB_REPO`,
`RELEASE_LEGACY_DOCKERHUB_REPO`); physical source still lives under
`tpu_inference` with `vllm_jax` exposed as a singleton compatibility alias. It
builds the image and all wheels once from the release branch, stages the image
in GAR and checks the wheels, waits for approval, then publishes those same
bytes to Docker Hub and PyPI, so users get exactly what was built and approved.
It runs only in the **TPU Release** cluster, on the `release-tpu-inference`
queue.

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
     the version yet on any of the four projects, and warns if `vX.Y.Z` is
     already on Docker Hub;
   - **builds once**: the image, from the release checkout's
     `docker/Dockerfile` and vLLM pin (`.buildkite/vllm_lkg.version`), staged
     in GAR as `vllm-jax:release-<version>-build<N>` (CI's own `vllm-tpu` tags
     are `<commit>-<vllm commit>-<tpu>`, so the two never collide), plus
     `vllm-jax-core`, `vllm-jax`, and the transitional `tpu-inference` /
     `vllm-tpu` meta-wheels;
   - **checks**: `vllm-jax`, `vllm-tpu`, and `tpu-inference` all resolve cleanly
     from the built wheels plus pypi.org for CPython 3.12 on x86_64 Linux
     (`check_wheels.sh`).
4. **Approve** the publish block. It names the staged digest and asks whether
   to move `latest` (never for a pre-release; no default) and whether an
   existing Docker Hub tag may be replaced. PyPI releases can't be replaced.
5. The build publishes what it built and checked:
   - copies the staged image by digest to Docker Hub as
     `vllm/vllm-jax:vX.Y.Z` and `vllm/vllm-tpu:vX.Y.Z` (and `latest` if chosen);
   - uploads `vllm-jax-core`, then `tpu-inference` (meta) and `vllm-jax`, then
     `vllm-tpu` (meta), to PyPI with Trusted Publishing
     (`.buildkite/scripts/publish_wheel.sh`), waiting until each package
     installs from PyPI;
   - checks, without credentials, that anonymous requests for both Docker Hub
     repos return the staged digest.

Build env knobs:

| Env | Effect |
|---|---|
| `RELEASE_VERSION=vX.Y.Zrc1` | publish a pre-release under that version |
| `RELEASE_APPROVER_TEAMS=a,b` | only these Buildkite teams can approve |

## Pipeline settings (Buildkite, made by hand)

- Repository `https://github.com/vllm-project/vllm-jax.git` (GitHub redirects
  the former `tpu-inference` URL).
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

- Buildkite: the TPU Release cluster secret `DOCKERHUB_TOKEN_TPU_INFERENCE`
  (Docker Hub user `tpuinference`), with an access policy that lets this
  pipeline read it from UI builds on the `release-tpu-inference` queue; that
  user can push to both `vllm/vllm-jax` and `vllm/vllm-tpu`.
- The `release-tpu-inference` agents can push to
  `us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/tpu-inference-ci` and
  have `docker buildx` with `imagetools create --prefer-index`.
- PyPI: a Google Trusted Publisher on `vllm-jax-core`, `tpu-inference`,
  `vllm-jax`, and `vllm-tpu` for the service account of the agents on the
  `release-tpu-inference` queue. Without it, `publish_wheel.sh` can't mint a
  token and uploads nothing.
