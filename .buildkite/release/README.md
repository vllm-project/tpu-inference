# tpu-inference-release

One Buildkite pipeline publishes everything `tpu-inference` ships: the nightly
images, and each release's Docker image and two wheels (`tpu-inference`
and the combined `vllm-tpu`). It runs only in the **TPU Release** cluster,
on the `release-tpu-inference` queue. That queue's VM runs as its own service
account (`ci-release`), refuses jobs from any other pipeline, and is the only
agent that can read the Docker Hub token or mint a PyPI token. No CI agent can.

The pipeline's settings upload [`pipeline.yml`](pipeline.yml) **from main**,
whatever branch a build is on, and every step runs main's tooling (this
directory and `.buildkite/scripts`) against the build's checkout. A release
branch supplies only the code being released.

## Cut a release

1. Get a full CI run of the release commit: a **tpu-inference-ci** build of
   `releases/vX.Y.Z` with `NIGHTLY=1`. It has to pass; the release checks.
2. Start a **New Build** of **tpu-inference-release** on `releases/vX.Y.Z`, at
   that commit. For a pre-release set `RELEASE_VERSION=vX.Y.Zrc1`.
3. The build then, without anyone's help:
   - **preflight**: settles the version (branch or `RELEASE_VERSION`), checks
     the commit's `tpu-inference-ci` statuses on GitHub (passed, with nightly
     steps), and checks PyPI doesn't have the version yet;
   - **builds once**: the `ci`, `dev` and `prod` images, staged in GAR as
     `vllm-tpu-<target>:<version>` and `:<date>-<version>`, and both
     wheels (with the release checkout's own build scripts);
   - **checks**: the wheels resolve together from pypi.org for CPython 3.12
     on x86_64 Linux.
4. **Approve** the publish block. It asks whether to move `latest` (never for a
   pre-release) and whether an existing Docker Hub tag may be replaced.
5. The build publishes what it built and checked:
   - the staged `prod` image, copied by digest to Docker Hub as
     `vllm/vllm-tpu:vX.Y.Z` (and `latest` if chosen);
   - `tpu-inference`, then `vllm-tpu`, to PyPI with Trusted
     Publishing (`.buildkite/scripts/publish_wheel.sh`);
   - then checks, without credentials, that an anonymous pull of the tag is the
     staged digest and that `vllm-tpu==X.Y.Z` resolves from pypi.org.

Build env knobs, for when something is broken and you know why:

| Env | Effect |
|---|---|
| `RELEASE_VERSION=vX.Y.Zrc1` | publish a pre-release under that version |
| `RELEASE_SKIP_CI_GATE=1` | don't check `tpu-inference-ci` (annotated as a warning) |
| `RELEASE_APPROVER_TEAMS=a,b` | only these Buildkite teams can approve |

## Nightly images

A schedule on `main` runs the `nightly_images` step:
`.buildkite/scripts/publish_nightly_images.sh` pushes the `ci`, `dev` and `prod`
images to GAR and the `dev` image to Docker Hub as `nightly`.

## Pipeline settings (Buildkite, made by hand)

- Cluster **TPU Release**; repository
  `git@github.com:vllm-project/tpu-inference.git`.
- GitHub: no builds on push or pull request (manual and scheduled only).
- Schedule: `main`, daily, for the nightly images.
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

## One-time setup outside this repo

- ci-infra: the pipeline's ID in `allowed_pipeline_ids` of
  `ci_release_tpu_inference` (`release_agents.tf`), so the release VM accepts
  its jobs.
- Buildkite: the TPU Release cluster secret `DOCKERHUB_TOKEN_TPU_INFERENCE`
  readable by this pipeline.
- PyPI: a Google Trusted Publisher for
  `ci-release@cloud-ullm-inference-ci-cd.iam.gserviceaccount.com` on both
  `tpu-inference` and `vllm-tpu`.
