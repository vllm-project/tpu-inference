from vllm.config import VllmConfig


def get_dp_size(vllm_config: VllmConfig) -> int:
    try:
        dp_size = vllm_config.additional_config["sharding"]["sharding_strategy"]["data_parallelism"]
        print("scheduler dp_size from config", dp_size)
    except (KeyError, TypeError, AttributeError):
        dp_size = 1
    
    if hasattr(vllm_config, 'model_config') and hasattr(vllm_config.model_config, 'hf_config'):
        num_kv_heads = vllm_config.model_config.hf_config.num_key_value_heads
        tp = vllm_config.parallel_config.tensor_parallel_size
        attn_dp = max(tp // num_kv_heads, 1)
    else:
        attn_dp = 1
    #hack: 
    attn_dp = 2
    return dp_size, attn_dp, dp_size * attn_dp
