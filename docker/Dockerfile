ARG NIGHTLY_DATE="20250714"
ARG BASE_IMAGE="us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.12_tpuvm_$NIGHTLY_DATE"

FROM $BASE_IMAGE

# Remove existing versions of dependencies
RUN pip uninstall -y torch torch_xla torchvision

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    git \
    libopenblas-base libopenmpi-dev libomp-dev

# Build vLLM
WORKDIR /workspace/vllm
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_COMMIT_SHA=0f199f197b4e7a835ccc5b4d15363f8faa7824c8
RUN git clone $VLLM_REPO /workspace/vllm
RUN git reset --hard $VLLM_COMMIT_SHA
RUN pip install -r requirements/tpu.txt
RUN VLLM_TARGET_DEVICE="tpu" pip install -e .

# Install test dependencies
RUN python3 -m pip install -e tests/vllm_test_utils
RUN python3 -m pip install --no-cache-dir git+https://github.com/thuml/depyf.git pytest pytest-asyncio tpu-info datasets 'lm_eval[api]==0.4.4'
RUN python3 -m pip install pytest-cov

# Install tpu_commons
WORKDIR /workspace/tpu_commons
# Install requirements first and cache so we don't need to re-install on code change.
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY requirements_benchmarking.txt .
# These are needed for the E2E benchmarking tests (i.e. tests/e2e/benchmarking/mlperf.sh)
RUN pip install -r requirements_benchmarking.txt
COPY . .
RUN pip install -e .

CMD ["/bin/bash"]
