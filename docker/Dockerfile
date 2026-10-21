# Copyright 2025 Google LLC
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

ARG BASE_IMAGE="python:3.12-slim-bookworm"

# ==============================================================================
# Stage 1: base (Runtime dependencies + vLLM from pyproject.toml)
# ==============================================================================
FROM ${BASE_IMAGE} AS base

RUN apt-get update && apt-get install -y \
    git \
    libopenmpi-dev \
    libomp-dev \
    procps \
    curl \
    netcat-openbsd \
    rclone \
    wget \
    bc \
    ca-certificates \
    unzip \
    build-essential \
    cmake \
    ninja-build \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Ensure /usr/bin/python3 exists and install uv for dependency resolution.
RUN ln -sf "$(command -v python)" /usr/bin/python3 && \
    pip install --no-cache-dir --upgrade pip uv

WORKDIR /workspace/tpu_inference

# Copy pyproject.toml first so dependency layers stay cached across code edits.
COPY pyproject.toml .

# If VLLM_COMMIT_HASH is passed (e.g., testing upstream HEAD in CI),
# dynamically update pyproject.toml inside the container before installing.
ARG VLLM_COMMIT_HASH=""
RUN if [ -n "$VLLM_COMMIT_HASH" ]; then \
      sed -i -E "s|(git\+https://github\.com/vllm-project/vllm\.git@)[a-zA-Z0-9._-]+|\1${VLLM_COMMIT_HASH}|g" pyproject.toml; \
    fi

# Build vLLM as a pure-Python package without device C++ extensions; TPU runtime
# dependencies are declared directly in pyproject.toml.
ENV CXX=g++
ENV CC=gcc
ENV VLLM_TARGET_DEVICE=empty
ENV UV_TORCH_BACKEND=cpu

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
      -r pyproject.toml \
      --group vllm


# ==============================================================================
# Stage 2: ci (CI/CD system tooling: bktec, buildkite-agent, google-cloud-cli)
# ==============================================================================
FROM base AS ci

COPY --from=buildkite/test-engine-client /usr/local/bin/bktec /usr/local/bin/bktec
COPY --from=buildkite/agent:3 /usr/local/bin/buildkite-agent /usr/local/bin/buildkite-agent

RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsof \
    psmisc \
    && curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/tpu_inference

RUN git config --global --add safe.directory /workspace/tpu_inference


# ==============================================================================
# Stage 3: dev (Developer & CI execution image with test/benchmark dependencies)
# ==============================================================================
FROM ci AS dev

# Pre-install test and benchmarking dependencies before copying the full source
# tree so code changes do not invalidate the dependency layer cache.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
      -r pyproject.toml \
      --group vllm \
      --group test \
      --extra benchmarking

COPY . .

ARG VLLM_COMMIT_HASH=""
RUN if [ -n "$VLLM_COMMIT_HASH" ]; then \
      sed -i -E "s|(git\+https://github\.com/vllm-project/vllm\.git@)[a-zA-Z0-9._-]+|\1${VLLM_COMMIT_HASH}|g" pyproject.toml; \
    fi

ARG RELEASE_VERSION="0.0.0"
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TPU_INFERENCE="${RELEASE_VERSION}" \
    uv pip install --system --no-deps -e .

# Optional local vLLM checkout override staged by docker/build_image.sh -s
ARG VLLM_SOURCE=""
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$VLLM_SOURCE" ] && [ -d /workspace/tpu_inference/vllm ]; then \
      uv pip install --system --no-deps -e /workspace/tpu_inference/vllm; \
    else \
      rm -rf /workspace/tpu_inference/vllm; \
    fi

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD []


# ==============================================================================
# Stage 4: prod (Production release image without CI/test tooling)
# ==============================================================================
FROM base AS prod

WORKDIR /workspace/tpu_inference
COPY . .

ARG VLLM_COMMIT_HASH=""
RUN if [ -n "$VLLM_COMMIT_HASH" ]; then \
      sed -i -E "s|(git\+https://github\.com/vllm-project/vllm\.git@)[a-zA-Z0-9._-]+|\1${VLLM_COMMIT_HASH}|g" pyproject.toml; \
    fi

ARG RELEASE_VERSION="0.0.0"
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TPU_INFERENCE="${RELEASE_VERSION}" \
    uv pip install --system --no-deps .

RUN if [ -z "${VLLM_SOURCE:-}" ] && [ -d /workspace/tpu_inference/vllm ]; then \
      rm -rf /workspace/tpu_inference/vllm; \
    fi

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD []
