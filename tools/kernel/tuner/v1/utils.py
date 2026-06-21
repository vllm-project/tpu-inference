# Keep in sync with the logic in bootstrap_kernel_tuning.sh:set_jax_envs
def get_tpu_queue_by_version_and_cores(tpu_version:str, tpu_cores:int, tpu_queue_multi:str = None) -> str:
    """Gets and validates TPU queue based on version and core configuration."""
    _queue_by_version_and_cores = {
        ('tpu6e', 1): 'tpu_v6e_queue',
        ('tpu6e', 8): 'tpu_v6e_8_queue',
        ('tpu7x', 2): 'tpu_v7x_2_queue',
        ('tpu7x', 8): 'tpu_v7x_8_queue',
        ('tpu7x', 16): 'tpu_v7x_16_queue',
    }
    assert (
        tpu_version, tpu_cores
    ) in _queue_by_version_and_cores, f'Unsupported combination of TPU version {tpu_version} and cores {tpu_cores}. Supported combinations are: {list(_queue_by_version_and_cores.keys())}'
    expected_queue = _queue_by_version_and_cores[(tpu_version, tpu_cores)]
    assert not tpu_queue_multi or tpu_queue_multi == expected_queue, f'Inconsistent TPU queue {tpu_queue_multi} for version {tpu_version} and cores {tpu_cores}. Expected queue is {expected_queue}. Please check your flags.'
    return tpu_queue_multi or expected_queue

