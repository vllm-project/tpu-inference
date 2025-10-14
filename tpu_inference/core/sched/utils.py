from vllm.config import VllmConfig


def get_dp_size(vllm_config: VllmConfig) -> int:
    """Get data parallel size from vllm config.

    Returns:
        dp_size: number of model replicas
        attn_dp: number of attention replicas within each model replica
        total_dp: total data parallel size = dp_size * attn_dp
    """
    try:
        dp_size = vllm_config.additional_config["sharding"]["sharding_strategy"]["data_parallelism"]
    except (KeyError, TypeError, AttributeError):
        dp_size = 1
    
    try:
        tp = vllm_config.parallel_config.tensor_parallel_size
    except (KeyError, TypeError, AttributeError):
        tp = 1

    if hasattr(vllm_config, 'model_config') and hasattr(vllm_config.model_config, 'hf_config'):
        num_kv_heads = vllm_config.model_config.hf_config.num_key_value_heads
        attn_dp = max(tp // num_kv_heads, 1)
    else:
        attn_dp = 1
    
    return dp_size, attn_dp, dp_size * attn_dp
