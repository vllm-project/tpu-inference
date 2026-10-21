# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The upstream nightly, republished with the buildkite-agent CLI added.
#
# The CLI, because the benchmark runs in a pod with no agent in it and the pod
# is deleted when the JobSet finishes - uploading results from inside is the
# only way they outlive the run.
#
# Republished, because the launcher accepts a workload image only from this
# project's Artifact Registry. A public repository's pipeline chooses the image
# its own steps run, so an unrestricted list would let a pull request run
# anything on the fleet's chips.

ARG BASE_IMAGE=vllm/vllm-tpu:nightly

# Pinned, not the `3` channel: the base moving under this image is the point,
# the CLI moving under it is not.
ARG AGENT_IMAGE=buildkite/agent:3.138.0

FROM ${AGENT_IMAGE} AS agent

FROM ${BASE_IMAGE}

COPY --from=agent /usr/local/bin/buildkite-agent /usr/local/bin/buildkite-agent

# What the pods actually run. This image has no repo in it - the agent checks
# the repo out on the host, not in the slice - so a role that execs a path under
# /workspace finds nothing and exits 127. Listed one by one: a script that a
# manifest starts referencing has to be added here deliberately.
COPY disagg_engine.sh disagg_benchmark.sh toy_proxy_server.py /opt/kube/
