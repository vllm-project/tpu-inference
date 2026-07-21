import jax
import time
import jax.numpy as jnp
from tpu_inference.kernels.experimental.batched_rpa import configs, wrapper, kernel, schedule

# 1. Setup configurations
model_cfgs = configs.ModelConfigs(num_q_heads=4, num_kv_heads=2, head_dim=256, mask_value=-1e9)
max_num_seqs = 128
serve_cfgs = configs.ServingConfigs(
    num_seqs=max_num_seqs, page_size=256, total_q_tokens=128, num_page_indices=128*5,
    dtype_q=jnp.bfloat16, dtype_kv=jnp.float8_e4m3fn, dtype_out=jnp.bfloat16,
)

# 2. Empty distribution: 0 decode, 0 prefill, 0 mixed
empty_dist = jnp.array([0, 0, 0], dtype=jnp.int32)
cu_q_lens = jnp.zeros((max_num_seqs + 1,), dtype=jnp.int32)
kv_lens = jnp.zeros((max_num_seqs,), dtype=jnp.int32)
page_indices = jnp.zeros((128 * 5,), dtype=jnp.int32)

queries = jnp.zeros((128, 4, 256), dtype=jnp.bfloat16)
keys = jnp.zeros((128, 2, 256), dtype=jnp.float8_e4m3fn)
values = jnp.zeros((128, 2, 256), dtype=jnp.float8_e4m3fn)
kv_cache = jnp.zeros((640, 256, 1, 4, 256), dtype=jnp.float8_e4m3fn)

# 3. Compare Different Descriptor Sizes for an EMPTY kernel
test_configs = [
    ('Tiny Descriptor (b2 / q1 / k512)', configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=512, batch_size=2, n_buffer=3)),
    ('Medium Descriptor (b8 / q16 / k1024)', configs.BlockSizes(bq_sz=16, bq_c_sz=4, bkv_sz=1024, batch_size=8, n_buffer=3)),
    ('Heavy Decode Descriptor (b8 / q1 / k3584)', configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=3584, batch_size=8, n_buffer=3)),
]

print("=== EMPTY KERNEL (0 SEQUENCES) PROLOGUE EXPERIMENT ===")
for name, bs in test_configs:
    cfgs = configs.RpaConfigs(block=bs, model=model_cfgs, serve=serve_cfgs, vmem_limit_bytes=128*1024*1024, mode=configs.RpaCase.DECODE)
    
    @jax.jit
    def run_empty(q, k, v, cache):
        q_hbm, new_kv_hbm = wrapper.prepare_inputs(q, k, v, q.dtype, cache.dtype)
        sched = schedule.generate_rpa_metadata(cu_q_lens, kv_lens, empty_dist, cfgs=cfgs, update_kv_cache=True)
        return kernel.rpa_kernel(cu_q_lens, kv_lens, page_indices, sched, q_hbm, new_kv_hbm, cache, cfgs=cfgs)

    # Warmup
    for _ in range(5):
        out = run_empty(queries, keys, values, kv_cache)
    jax.block_until_ready(out)

    # Measure dispatch latency
    start = time.perf_counter()
    iters = 100
    for _ in range(iters):
        out = run_empty(queries, keys, values, kv_cache)
    jax.block_until_ready(out)
    end = time.perf_counter()

    avg_us = (end - start) / iters * 1e6
    print(f"{name:<42} -> Latency: {avg_us:6.2f} us")