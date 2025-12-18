ARG NIGHTLY_DATE="20250714"
ARG BASE_IMAGE="us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.12_tpuvm_$NIGHTLY_DATE"
# The latest main will be used if arg unspecified
ARG VLLM_COMMIT_HASH=""

FROM $BASE_IMAGE

ARG IS_FOR_V7X="false"

# Remove existing versions of dependencies
RUN pip uninstall -y torch torch_xla torchvision

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    git \
    libopenblas-base libopenmpi-dev libomp-dev

# Build vLLM
WORKDIR /workspace/vllm
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_COMMIT_HASH
RUN git clone $VLLM_REPO /workspace/vllm
RUN if [ -n "$VLLM_COMMIT_HASH" ]; then \
        git checkout $VLLM_COMMIT_HASH; \
    fi
RUN --mount=type=cache,target=/root/.cache/pip pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/tpu.txt --retries 3
RUN --mount=type=cache,target=/root/.cache/pip VLLM_TARGET_DEVICE="tpu" pip install -e .

# Install test dependencies
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install -e tests/vllm_test_utils
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install \
    git+https://github.com/thuml/depyf.git \
    pytest-asyncio \
    git+https://github.com/EleutherAI/lm-evaluation-harness.git@206b7722158f58c35b7ffcd53b035fdbdda5126d#egg=lm-eval[api] \
    pytest-cov \
    tblib

# Install tpu_inference
WORKDIR /workspace/tpu_inference
# Install requirements first and cache so we don't need to re-install on code change.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt --retries 3
COPY requirements_benchmarking.txt .
# These are needed for the E2E benchmarking tests (i.e. tests/e2e/benchmarking/mlperf.sh)
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_benchmarking.txt --retries 3
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip pip install -e .

# TODO (jacobplatin): remove when v7x is supported in JAX/Libtpu officially
# NOTE: it's important that this is done after installing tpu_inference above,
# so that the v7x-specific dependencies can override any existing ones.
COPY requirements_v7x.txt .
RUN --mount=type=cache,target=/root/.cache/pip if [ "$IS_FOR_V7X" = "true" ]; then \
        pip install -r requirements_v7x.txt; \
    fi

CMD ["/bin/bash"]
