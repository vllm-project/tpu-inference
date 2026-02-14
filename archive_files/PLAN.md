# Coding Plan: Automated vLLM TPU Benchmarking System

This plan outlines the steps to build the automation system for running models and benchmarks on TPU.

## Phase 1: Core Orchestration (TPU Management)
- **Goal**: Script the lifecycle of a TPU VM.
- **Tasks**:
  1. Create `src/tpu_manager.py` using `subprocess` to wrap `gcloud` commands.
  2. Implement `start_tpu()`: Checks if TPU exists, creates it if not.
  3. Implement `stop_tpu()`: Deletes or stops the TPU to save costs.
  4. Implement `get_tpu_ip()`: Retrieves the external/internal IP for SSH access.

## Phase 2: Remote Execution & Setup
- **Goal**: Prepare the TPU VM for vLLM.
- **Tasks**:
  1. Create `src/executor.py` to handle SSH commands (`gcloud compute tpus tpu-vm ssh`).
  2. Implement an environment setup script:
     - Install Docker or required Python packages (vllm, libtpu).
     - Authenticate with Hugging Face (optional).
     - Pull the `vllm-tpu` Docker image.

## Phase 3: Benchmarking Pipeline
- **Goal**: Deploy models and run the benchmarks.
- **Tasks**:
  1. Create `src/benchmarker.py`.
  2. Implement `launch_vllm_server()`: Starts the vLLM server inside the TPU VM (via Docker or native).
  3. Implement `wait_for_server()`: Polls the `/health` endpoint until ready.
  4. Implement `run_benchmark_script()`: Executes `benchmark_serving.py` with specified parameters.
  5. Implement `collect_results()`: SCPs the results back to the local machine.

## Phase 4: Main Controller & Error Handling
- **Goal**: Tie everything together.
- **Tasks**:
  1. Create `run.py` as the entry point.
  2. Add logic to iterate through the `models` list in `config.yaml`.
  3. Implement comprehensive error handling (e.g., TPU creation failures, server crashes).
  4. Add logging to track progress.

## Phase 5: Result Analysis (Optional)
- **Goal**: summarize the data.
- **Tasks**:
  1. Create a script to parse the output JSONs and generate a summary table/graph.
