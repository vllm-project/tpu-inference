# Baseline v2 Benchmarking Suite

This is an automated performance evaluation framework for parameter sweeps on vLLM and TPU-Inference.

## Overview
This orchestrator tests both pure generation endpoints and sequence decodes across massive parameter matrices (batch sizes, input context lengths). 

## Architecture
- `run_sweep_orchestrator.py`: The master sequencer. Reads configuration files and iteratively launches isolated benchmarking processes for every matrix dimension.
- `tpu_profile_vllm.py`: The single-instance benchmarking script. Generates mock input tokens, spins up the vLLM engine, fires the request, cleanly exits, and appends the metric data directly into partitioned `results/` CSVs.
- `configs/`: Contains configuration dictionaries mapping model bounds and parameter matrices.
- `results`: Where results will be stored 

## Features
- Avoids cross-run OOM fragmentation by cleanly isolating TPU processes per test.
- Automatically handles TPU lockfiles and XLA graph compilation overlaps.
- Skips previously completed operations if crash recovery is needed.

## Quickstart Guide

You can run these examples directly on the TPU VM after activating your vLLM environment (e.g., `source /mnt/pd/shen/vllm_env/bin/activate`).

### 1. Run a Single Individual Benchmark
If you just want to test a single configuration quickly without a YAML setup, use the underlying benchmarking script directly. 

This command initializes a Gemma 4 model on 4 TPU cores, feeds it a 1024-token context, generates 128 tokens at a batch size of 4, and logs the throughput:
```bash
python3 tpu_profile_vllm.py \
  --model google/gemma-4-12b-it \
  --input-len 1024 \
  --output-len 128 \
  --batch-size 4 \
  --tensor-parallel-size 4 \
  --dtype bfloat16
```

### 2. Run a Full Parameter Sweep
If you want to run an automated grid search (e.g., testing how throughput scales as context size grows), use the orchestrator script with one of the provided configs. 

This command reads `configs/1_prefill_sweep.yaml` and iteratively launches the single benchmark script for every combination of parameters defined in the matrix, saving all data into a timestamped CSV folder in `results/`:
```bash
python3 run_sweep_orchestrator.py --config configs/1_prefill_sweep.yaml
```

**View Results:**
After the sweep finishes (or while it's running), check the latest timestamped folder in the results directory:
```bash
cat results/*/results.csv
```

**Resume an Interrupted Sweep:**
If an execution crashed (e.g. out of memory on a massive configuration), explicitly pass the failed directory's timestamp. It will safely skip combinations that are already logged in the CSV and resume where it left off:
```bash
python3 run_sweep_orchestrator.py --config configs/1_prefill_sweep.yaml --experiment-id 20261111_001020
```
