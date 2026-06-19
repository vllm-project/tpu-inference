FROM vllm/vllm-tpu:latest

# 1. Copy your locally patched tpu-inference repository
WORKDIR /app/tpu-inference
COPY . .

# Patch import in tpu-inference to match new vLLM structure
RUN find /app/tpu-inference -name "*.py" -exec sed -i 's/from vllm.inputs import ProcessorInputs/from vllm.multimodal.processing import ProcessorInputs/g' {} +

# 2. Get vLLM (Use HEAD from main branch)
RUN git clone https://github.com/vllm-project/vllm.git /app/vllm

# 3. Install vLLM first
WORKDIR /app/vllm
# Remove tpu-inference from requirements to avoid pulling from PyPI
RUN sed -i '/tpu-inference/d' requirements/tpu.txt
RUN pip install -r requirements/tpu.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN VLLM_TARGET_DEVICE=tpu pip install -e .

# 4. Install Diagon SDK & the TPU Plugin (Install tpu-inference after)
RUN pip install google-cloud-mldiagnostics google-cloud-logging
WORKDIR /app/tpu-inference
RUN pip install -e . -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Start the server by default
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
