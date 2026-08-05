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
