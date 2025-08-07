import jax
import jax.numpy as jnp
import numpy as np

from tpu_commons.kernels.ragged_kv_cache_update import _dynamic_validate_inputs, kv_cache_update
from jax.sharding import PartitionSpec as P

# Generate test data
page_size = 64
page_num = 512
combined_kv_head_num = 16
head_dim = 128
padded_num_tokens = 128
padded_num_slices = 256