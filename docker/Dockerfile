ARG NIGHTLY_DATE="20250714"
ARG BASE_IMAGE="us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.12_tpuvm_$NIGHTLY_DATE"

FROM $BASE_IMAGE

# Remove existing versions of dependencies
RUN pip uninstall -y torch torch_xla torchvision

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    git \
    libopenblas-base libopenmpi-dev libomp-dev \
    subversion \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --target=/workspace tpu-inference==0.11.1rc3
RUN pip install --target=/workspace vllm-tpu==0.11.1rc3

# Build vLLM
# WORKDIR /workspace/vllm
# ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
# RUN git clone $VLLM_REPO /workspace/vllm
# RUN git checkout v0.11.1rc1
# RUN pip install -r requirements/tpu.txt
# RUN VLLM_TARGET_DEVICE="tpu" pip install -e .

# Install test dependencies
# RUN python3 -m pip install -e tests/vllm_test_utils
RUN python3 -m pip install --no-cache-dir \
    git+https://github.com/thuml/depyf.git \
    pytest-asyncio \
    git+https://github.com/EleutherAI/lm-evaluation-harness.git@206b7722158f58c35b7ffcd53b035fdbdda5126d#egg=lm-eval[api] \
    pytest-cov \
    tblib \
    numba

WORKDIR /workspace/tpu_inference
COPY scripts/vllm/benchmarking scripts/vllm/benchmarking
COPY tests/e2e/benchmarking tests/e2e/benchmarking

# ENV COMMIT_HASH f26205594d29ada6a259834d9d0a7597904a552b
# ENV SVN_URL https://github.com/vllm-project/tpu-inference/branches/${COMMIT_HASH}/tests/e2e/benchmarking
# ENV TARGET_DIR /workspace/tpu-inference/tests/e2e/benchmarking

# RUN mkdir -p ${TARGET_DIR} && \
#     cd ${TARGET_DIR} && \
#     svn checkout --force ${SVN_URL} . && \
#     rm -rf .svn

# ENV SVN_URL_2 https://github.com/vllm-project/tpu-inference/branches/${COMMIT_HASH}/scripts/vllm/benchmarking
# ENV TARGET_DIR_2 /workspace/tpu-inference/scripts/vllm/benchmarking

# RUN mkdir -p ${TARGET_DIR_2} && \
#     cd ${TARGET_DIR_2} && \
#     svn checkout --force ${SVN_URL_2} . && \
#     rm -rf .svn

# Install tpu_inference
# WORKDIR /workspace/tpu_inference
# # Install requirements first and cache so we don't need to re-install on code change.
# COPY requirements.txt .
# RUN pip install -r requirements.txt
# COPY requirements_benchmarking.txt .
# # These are needed for the E2E benchmarking tests (i.e. tests/e2e/benchmarking/mlperf.sh)
# RUN pip install -r requirements_benchmarking.txt
# COPY . .
# RUN pip install -e .

CMD ["/bin/bash"]
