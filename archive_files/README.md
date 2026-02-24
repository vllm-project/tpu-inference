# vLLM TPU Automation Benchmarker

This project provides an automated system to provision Google Cloud TPU VMs, deploy vLLM, and run performance benchmarks (throughput and latency) for various LLMs.

## Features
- **Automatic TPU VM Management**: Create and delete TPU VMs (v5e, v6e, etc.) programmatically.
- **Environment Setup**: Automated installation of vLLM or Docker-based deployment.
- **Benchmarking**: Integrated with vLLM's `benchmark_serving.py`.
- **Configurable**: Define model lists, TPU types, and benchmark parameters in a single YAML file.
- **Telemetry**: Collect and store benchmark results (tokens/sec, latency) in JSON/CSV.

## Project Structure
- `configs/`: Directory for benchmark configurations.
- `src/`: Source code for orchestration.
- `results/`: Directory where benchmark outputs are stored.
- `run.py`: Main entry point to start the automated pipeline.

## Prerequisites
- Google Cloud SDK installed and authenticated (`gcloud auth login`).
- Python 3.9+ with `pyyaml` and `paramiko`.
- Project and Zone permissions for TPU VMs.
