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
# The latest main will be used if arg unspecified
ARG VLLM_COMMIT_HASH=""

FROM $BASE_IMAGE

ARG IS_TEST="false"

# Update pip
RUN pip install --upgrade pip

# Install some basic utilities
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
    && rm -rf /var/lib/apt/lists/*

# Build vLLM
WORKDIR /workspace/vllm
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_COMMIT_HASH

RUN git clone $VLLM_REPO . && \
    if [ -n "$VLLM_COMMIT_HASH" ]; then \
        git checkout $VLLM_COMMIT_HASH; \
    fi

RUN --mount=type=cache,target=/root/.cache/pip pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/tpu.txt --retries 3
RUN --mount=type=cache,target=/root/.cache/pip VLLM_TARGET_DEVICE="tpu" pip install -e .

# Install test dependencies
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install -e tests/vllm_test_utils
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install \
    git+https://github.com/thuml/depyf.git \
    pytest-asyncio \
    "lm-eval[api]==0.4.9.2" \
    pytest-cov \
    tblib \
    gdown

# Install tpu_inference
WORKDIR /workspace/tpu_inference
# Install requirements first and cache so we don't need to re-install on code change.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt --retries 3
COPY requirements_benchmarking.txt .
# These are needed for the E2E benchmarking tests (i.e. tests/e2e/benchmarking/mlperf.sh)
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_benchmarking.txt --retries 3
# Install test requirements
COPY requirements_test.txt .
RUN  --mount=type=cache,target=/root/.cache/pip if [ "$IS_TEST" = "true" ]; then \
        pip install -r requirements_test.txt --retries 3; \
    fi
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip pip install -e .

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD []
