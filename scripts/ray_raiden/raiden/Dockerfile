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

# Build custom vllm-tpu image with pre-built tpu-raiden-jax wheel and pinned vllm & tpu-inference
FROM vllm/vllm-tpu:nightly

RUN apt-get update && apt-get install -y build-essential curl git patchelf && rm -rf /var/lib/apt/lists/*

# Checkout pinned vllm commit and install its TPU requirements
RUN git -C /workspace/vllm fetch origin 6185d73882c0cdfd9ee13cea16a9b50d2b5267be && \
    git -C /workspace/vllm checkout 6185d73882c0cdfd9ee13cea16a9b50d2b5267be && \
    pip install -r /workspace/vllm/requirements/tpu.txt --no-deps || true

# Clone pinned tpu-inference commit, install its requirements, and editable install
RUN cd /workspace && \
    rm -rf /workspace/tpu_inference && \
    git clone https://github.com/vllm-project/tpu-inference.git /workspace/tpu_inference && \
    git -C /workspace/tpu_inference checkout 7ed46a69e4bf6900af69ad681e01df40a355be85 && \
    pip install -r /workspace/tpu_inference/requirements.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
    pip install --no-deps -e /workspace/tpu_inference

# Clone pinned tpu-raiden commit
RUN rm -rf /workspace/tpu-raiden && \
    git clone https://github.com/google/tpu-raiden.git /workspace/tpu-raiden && \
    git -C /workspace/tpu-raiden fetch origin 110ebcef084232816c9e30d3ce2ee27f47062055 && \
    git -C /workspace/tpu-raiden checkout 110ebcef084232816c9e30d3ce2ee27f47062055

# Install pre-compiled tpu-raiden-jax wheel from Google Artifact Registry
RUN --mount=type=secret,id=adc,target=/root/.config/gcloud/application_default_credentials.json \
    pip install keyrings.google-artifactregistry-auth && \
    pip install tpu-raiden-jax --extra-index-url https://us-python.pkg.dev/cloud-tpu-inference-test/tpu-raiden/simple/ --no-deps

# Default tpu-raiden tunables & environment variables
ENV RAIDEN_PRELOAD_ENGINE="1"
ENV RAIDEN_NUM_SLOTS="30"
ENV RAIDEN_TRANSPORT_PARALLELISM="32"
ENV TPU_KV_TRANSFER_PORT="9100"
ENV PYTHONPATH="/workspace/tpu-raiden:/workspace/tpu_inference:${PYTHONPATH}"
