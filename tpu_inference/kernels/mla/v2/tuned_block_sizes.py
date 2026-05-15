from tpu_inference.logger import init_logger

logger = init_logger(__name__)

tuned_block_sizes = {
'''
    tuning_set_from_log.append([TuningKey(
        max_num_tokens=512,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,

    for each of the above tuningkey, i just want to use their max_num_tokens, actual_num_q_heads, actual_lkv_dim, actual_r_dim to lookup the tunable params for the batched decode case, and ignore the rest of the parameters since they are all the same across different tuning keys in the log (they only differ in max_num_tokens, actual_lkv_dim, actual_r_dim which are already part of the tuning key used for lookup)
'''
'''
    key: (max_num_tokens, actual_num_q_heads, actual_lkv_dim, actual_r_dim, 'batched_decode')
    value: (num_kv_pages_per_block, num_queries_per_block, decode_batch_size)
'''
'''
    the above tuning keys and tunable params are default values that we got from the log. I want to list them out here as a visual comparision.
'''
    # Below are existing default tunable params
    # (4, 128, 512, 64, 'batched_decode'): {'num_kv_pages_per_block': 3, 'num_queries_per_block': 1, 'decode_batch_size': 4},
    # (8, 128, 512, 64, 'batched_decode'): {'num_kv_pages_per_block': 3, 'num_queries_per_block': 1, 'decode_batch_size': 4},
    # (16, 128, 512, 64, 'batched_decode'): {'num_kv_pages_per_block': 3, 'num_queries_per_block': 1, 'decode_batch_size': 4},
    # (32, 128, 512, 64, 'batched_decode'): {'num_kv_pages_per_block': 3, 'num_queries_per_block': 1, 'decode_batch_size': 4},
    # (64, 128, 512, 64, 'batched_decode'): {'num_kv_pages_per_block': 3, 'num_queries_per_block': 1, 'decode_batch_size': 4},
    # (128, 128, 512, 64, 'batched_decode'): {'num_kv_pages_per_block': 3, 'num_queries_per_block': 1, 'decode_batch_size': 4},
    # (160, 128, 512, 64, 'batched_decode'): {'num_kv_pages_per_block': 3, 'num_queries_per_block': 1, 'decode_batch_size': 4},
    # (256, 128, 512, 64, 'batched_decode'): {'num_kv_pages_per_block': 3, 'num_queries_per_block': 1, 'decode_batch_size': 4},
    # (512, 128, 512, 64, 'batched_decode'): {'num_kv_pages_per_block': 3, 'num_queries_per_block': 1, 'decode_batch_size': 4},
    # (max_num_tokens, actual_num_q_heads, actual_lkv_dim, actual_r_dim, 'batched_decode') : (num_kv_pages_per_block, num_queries_per_block, decode_batch_size)
    # below is run local case_set_id mla_tuning_0 and run_id = 4, raw result folder is /tmp/kernel_tuner_run_2026_05_14_21_43_47
    (4, 128, 512, 64, 'batched_decode'): {'num_kv_pages_per_block': 2, 'num_queries_per_block': 1, 'decode_batch_size': 8},
    (8, 128, 512, 64, 'batched_decode'): {'num_kv_pages_per_block': 2, 'num_queries_per_block': 1, 'decode_batch_size': 8},
    (16, 128, 512, 64, 'batched_decode'): {'num_kv_pages_per_block': 2, 'num_queries_per_block': 1, 'decode_batch_size': 8},
    (32, 128, 512, 64, 'batched_decode'): {'num_kv_pages_per_block': 2, 'num_queries_per_block': 1, 'decode_batch_size': 8},
    (64, 128, 512, 64, 'batched_decode'): {'num_kv_pages_per_block': 1, 'num_queries_per_block': 1, 'decode_batch_size': 8},
    (128, 128, 512, 64, 'batched_decode'): {'num_kv_pages_per_block': 2, 'num_queries_per_block': 1, 'decode_batch_size': 8},
    (160, 128, 512, 64, 'batched_decode'): {'num_kv_pages_per_block': 2, 'num_queries_per_block': 1, 'decode_batch_size': 8},
    (256, 128, 512, 64, 'batched_decode'): {'num_kv_pages_per_block': 2, 'num_queries_per_block': 1, 'decode_batch_size': 4},
    (512, 128, 512, 64, 'batched_decode'): {'num_kv_pages_per_block': 2, 'num_queries_per_block': 1, 'decode_batch_size': 8},
}

def lookup_tunable_params(tuning_key):
    if tuning_key not in tuned_block_sizes:
        logger.warning(f"[debug] MLA V2 Tuning key {tuning_key} not found in tuned block sizes. Using default tunable params.")
    return tuned_block_sizes.get(tuning_key, {})