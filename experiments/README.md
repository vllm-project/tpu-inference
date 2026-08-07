# Baseline v2 Benchmarking Suite

This is an automated performance evaluation framework for parameter sweeps on vLLM and TPU-Inference.

## Overview
This orchestrator tests both pure generation endpoints and sequence decodes across massive parameter matrices (batch sizes, input context lengths). 

## Architecture
- `run_sweep_orchestrator.py`: The master sequencer. Reads  configuration files and iteratively launches isolated benchmarking processes for every matrix dimension.
- `tpu_profile_vllm.py`: The single-instance benchmarking script. Generates mock input tokens, spins up the vLLM engine, fires the request, cleanly exits, and appends the metric data directly into partitioned `results/` CSVs.
- `configs/`: Contains configuration dictionaries mapping model bounds and parameter matrices. 

## Features
- Avoids cross-run OOM fragmentation by cleanly isolating TPU processes per test.
- Automatically handles TPU lockfiles and XLA graph compilation overlaps.
- Skips previously completed operations if crash recovery is needed.

## Quickstart Guide

**1. Create a Matrix Configuration**
Create a YAML file in `configs/` explicitly defining your sweep boundaries.
```yaml
sweep_matrix:
  model: [meta-llama/Meta-Llama-3.1-8B-Instruct]
  batch_size: [1, 2, 4, 8]
  input_len: [1024]
  output_len: [128]
```

**2. Launch the Orchestrator**
Run the sweep via the central master script passing your config:
```bash
python3 run_sweep_orchestrator.py --config configs/my_experiment.yaml
```

**3. Analyze Chronological Results**
Your metrics will be routed into a newly timestamped output block seamlessly capturing boundaries over all combinations!
```bash
cat results/20261111_001020/results.csv
```

**4. Resume an Interrupted Experiment**
If an execution crashed (e.g. HBM bounds hit), explicitly pass `--experiment-id` routing backward into identical coordinates safely explicitly bypassing logged metrics smoothly:
```bash
python3 run_sweep_orchestrator.py --config configs/my_experiment.yaml --experiment-id 20261111_001020
```
