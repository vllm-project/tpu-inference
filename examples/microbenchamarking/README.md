# MICROBENCHAMRKING IS EXPERIMENTAL AND NOT SUPPORTED FOR ALL MODELS AND FLEXIBLE WORKLOADS

The Goal of microbenchmarking is to strip the model call from VLLM Dependencies (Scheduler and KV Cache Manager) for efficient debugging and performance optimization of just model call.

The current version is ** working on pinned main **

```
Commit ID 5797c31acb0010cf8c54ba9218bacf96d8a1260e
```

> ⚠️ The microbenchmarking code **does not support all models and features and is currently used for debugging and optimizing static workloads

**Only tested model for microbenchmarking is QWEN3-32B**

## Params needed by microbenchmarking code

### `max_seq_len` -
 max model len this is length of the model including number of prefill and decode tokens

### `phase` -

phase of the model, supported modes are prefill and decode

### `decode_offset_from_prefill` -
used in decode primarily, if the value is 1, it means 1st token after prefill

### `model_hf_config` -
path to json file where HFConfig is saved. We need this because we dont want to download from huggingface.

### `num_block_override` -
number of blocks in KV Cache. This is kept as an override because we need the KV Cache part to be representative, a good value is obtained from
`offline_inference.py` runs.

### `max_prefill_len` -
max length of prefill sequence

### `max_num_sequence` -
is the maximum number of sequence supported by model.

### `Caveats are` :

i) In Prefill phase - `max_num_sequence` = max_seq_len // max_prefill_len

ii) In Decode phase - `max_num_sequence` < `max_seq_len`

### `model_call_steps` -
number of times the model is to be called

### `block_size` -
or same as `page_size` for KV Cache

### `additional_config` -
example of additional config

```
'{"sharding": {"sharding_strategy": {"tensor_parallelism": 8, "data_parallelism": 1}}, "quantization": { "qwix": { "rules": [{ "module_path": ".*", "weight_qtype": "float8_e4m3fn", "act_qtype": "float8_e4m3fn"}]}}}' --model_config='{"model":"Qwen/Qwen3-32B"}'
```

### `model_config` -
--model_config='{"model":"Qwen/Qwen3-32B"}'

### `new_model_design` -
True if microbenchmarking is done for new models like L4 and DeepSeek v3

### `trace_dir` -

local location where traces are stored. Default value is `/tmp/tpu_commons_traces`

## Example command to run Microbenchmark

###
## Decode

```
python examples/microbenchamarking/microbenchmark_app.py --additional_config='{"sharding": {"sharding_strategy": {"tensor_parallelism": 8, "data_parallelism": 1}}}' --model_config='{"model":"Qwen/Qwen3-32B"}' --phase='decode' --max_seq_len=4096 --max_num_seq=2048 --model_hf_config="examples/microbenchamarking/hf_configs/qwen3_32b_hf_config.json"

```

## Prefill

```
python examples/microbenchamarking/microbenchmark_app.py --additional_config='{"sharding": {"sharding_strategy": {"tensor_parallelism": 8, "data_parallelism": 1}}}' --model_config='{"model":"Qwen/Qwen3-32B"}' --phase='prefill' --max_seq_len=1024 --max_num_seq=2 --max_prefill_len=512 --model_hf_config="examples/microbenchamarking/hf_configs/qwen3_32b_hf_config.json"

```

Notice that max_num_seq = 2 as maximum of 2 sequences can fit with 512 as max_prefill_len
