# MICROBENCHAMRKING IS EXPERIMENTAL AND NOT SUPPORTED FOR ALL MODELS AND FLEXIBLE WORKLOADS

The Goal of microbenchmarking is to strip the model call from VLLM Dependencies (Scheduler and KV Cache Manager) for efficient debugging and performance optimization of just model call.

The current version is ** working on pinned main **

```
Commit ID 5797c31acb0010cf8c54ba9218bacf96d8a1260e
```

> ⚠️ The microbenchmarking code **does not support all models and features and is currently used for debugging and optimizing static workloads

**Only tested model for microbenchmarking is QWEN3-32B**

## Example command to run Microbenchmark

###
## Decode

```
python examples/microbenchamarking/microbenchmark_app.py --additional_config='{"sharding": {"sharding_strategy": {"tensor_parallelism": 8, "data_parallelism": 1}}}' --model_config='{"model":"Qwen/Qwen3-32B"}' --phase='decode'

```

## Prefill

```
python examples/microbenchamarking/microbenchmark_app.py --additional_config='{"sharding": {"sharding_strategy": {"tensor_parallelism": 8, "data_parallelism": 1}}}' --model_config='{"model":"Qwen/Qwen3-32B"}' --phase='prefill' --max_seq_len=1024 --max_num_seq=2 --max_prefill_len=512

```

Notice that max_num_seq = 2 as maximum of 2 sequences can fit with 512 as max_prefill_len
