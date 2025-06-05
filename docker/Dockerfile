ARG NIGHTLY_DATE="20250124"
ARG BASE_IMAGE="us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm_$NIGHTLY_DATE"

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
ARG VLLM_COMMIT_SHA=53a5a0ce30dd623808ebd02947e5183f918b6c2f
RUN git clone $VLLM_REPO /workspace/vllm
RUN git reset --hard $VLLM_COMMIT_SHA
RUN pip install -r requirements/tpu.txt
# This hack disables the pytorch/xla and torchax code path in tpu_commons
# RUN sed -i '/TpuPlatform = TpuCommonsPlatform/d' vllm/platforms/tpu.py
RUN VLLM_TARGET_DEVICE="tpu" pip install -e .

# Install test dependencies
RUN python3 -m pip install -e tests/vllm_test_utils
RUN python3 -m pip install --no-cache-dir git+https://github.com/thuml/depyf.git pytest pytest-asyncio tpu-info datasets 'lm_eval[api]==0.4.4'

# Install tpu_commons
WORKDIR /workspace/tpu_commons
COPY . .
RUN pip install -r requirements.txt
RUN pip install -e .

CMD ["/bin/bash"]
