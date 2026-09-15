# The image the kube P/D disaggregation benchmark runs, which is the upstream
# nightly with one binary added.
#
# The nightly is what this benchmark has always measured and this file does not
# change that: it is the base, and nothing is installed over it. Two things make
# a copy necessary anyway.
#
# The agent CLI, because on the kube fleet the benchmark runs in a pod in
# another cluster with no Buildkite agent in it, and the pod is deleted the
# moment the JobSet finishes. Uploading its results from inside it while it is
# still alive is the only way they outlive the run.
#
# And the registry, because the launcher accepts a workload image only from this
# project's Artifact Registry - a public repository's pipeline chooses the image
# its own steps run, so an unrestricted one would let a pull request run
# anything on the fleet's chips. Docker Hub is not on that list, so the nightly
# has to be republished here to be usable at all.

ARG BASE_IMAGE=vllm/vllm-tpu:nightly
FROM ${BASE_IMAGE}

COPY --from=buildkite/agent:3 /usr/local/bin/buildkite-agent /usr/local/bin/buildkite-agent
