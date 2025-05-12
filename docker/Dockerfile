ARG NIGHTLY_DATE="20250124"
ARG BASE_IMAGE="us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm_$NIGHTLY_DATE"

FROM $BASE_IMAGE

WORKDIR /workspace/tpu_commons
COPY . .

# Remove existing versions of dependencies
RUN pip uninstall -y torch torch_xla torchvision

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    git \
    libopenblas-base libopenmpi-dev libomp-dev

# Install tpu_commons
RUN pip install -r requirements.txt
RUN pip install -e .

# Build vLLM
WORKDIR /workspace/vllm
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_COMMIT_SHA=05a4324f8e3932c25554791ff248e3e0200eef92
RUN git clone $VLLM_REPO /workspace/vllm
RUN git checkout $VLLM_COMMIT_SHA
RUN pip install -r requirements/tpu.txt
RUN VLLM_TARGET_DEVICE="tpu" pip install -e .

# Install test dependencies
RUN python3 -m pip install -e tests/vllm_test_utils
RUN python3 -m pip install --no-cache-dir git+https://github.com/thuml/depyf.git pytest pytest-asyncio tpu-info datasets 'lm_eval[api]==0.4.4'


CMD ["/bin/bash"]
