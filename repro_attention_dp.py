"""
Reproduction script to compare attention kernel performance and HLO
with and without data parallelism.

Usage:
    python repro_attention_dp.py --mode no_dp
    python repro_attention_dp.py --mode with_dp
"""

import argparse
import os
import time
from pathlib import Path
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax import NamedSharding, shard_map

import sys
sys.path.insert(0, '/home/wenxindong_google_com/tpu-inference')

from tpu_inference.kernels.ragged_paged_attention.v3.kernel_hd64 import (
    ragged_paged_attention_hd64,
    get_kv_cache_shape,
)
from tpu_inference.layers.common.sharding import ShardingAxisName


def create_test_inputs(
    batch_size: int = 256,  
    max_num_tokens: int = 256, 
    num_q_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 64,
    page_size: int = 256,
    pages_per_seq: int = 16,  
    total_num_pages: int = 8282, 
    dp_size: int = 1,
):

    # batch_size = batch_size * dp_size
    total_num_pages = total_num_pages * dp_size
    # Q, K, V tensors - use unique seeds for each DP replica
    q = jnp.concatenate([jax.random.normal(
        jax.random.PRNGKey(0),  # Unique seed per replica
        (max_num_tokens, num_q_heads, head_dim),
        dtype=jnp.bfloat16,
    ) for i in range(dp_size)], axis=0)
    k = jnp.concatenate([jax.random.normal(
        jax.random.PRNGKey(0),  # Unique seed per replica
        (max_num_tokens, num_kv_heads, head_dim),
        dtype=jnp.bfloat16,
    ) for i in range(dp_size)], axis=0)
    v = jnp.concatenate([jax.random.normal(
        jax.random.PRNGKey(0),  # Unique seed per replica
        (max_num_tokens, num_kv_heads, head_dim),
        dtype=jnp.bfloat16,
    ) for i in range(dp_size)], axis=0)
    
    # KV cache
    kv_cache_shape = get_kv_cache_shape(
        total_num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        jnp.bfloat16,
    )
    kv_cache = jnp.zeros(kv_cache_shape, dtype=jnp.bfloat16)
    
    # kv_lens: each DP rank has its own requests
    kv_lens = []
    for dp_rank in range(dp_size):
        kv_lens.append([1024 for _ in range(batch_size)])
    kv_lens = jnp.array(kv_lens, dtype=jnp.int32).reshape(-1)        
    
    # Page indices: each DP rank manages its own pages
    page_indices = jnp.concatenate([jnp.arange(
        batch_size * pages_per_seq, dtype=jnp.int32
    ) for _ in range(dp_size)])
    
    # cu_q_lens: cumulative query lengths PER DP RANK
    cu_q_lens = []
    for _ in range(dp_size):
        cu_q_lens.append([1 * i for i in range(batch_size + 1)])
    cu_q_lens = jnp.array(cu_q_lens, dtype=jnp.int32).reshape(-1)

    # Distribution: per DP rank [decode_end, prefill_end, mixed_end]
    distribution_list = []
    for _ in range(dp_size):
        distribution_list.extend([0, 0, batch_size])
    distribution = jnp.array(distribution_list, dtype=jnp.int32)
    
    # Optional attention sink
    attention_sink = None
    
    sm_scale = head_dim ** -0.5
    
    print(f"\nInput creation (dp_size={dp_size}):")
    print(f"  q: {q.shape}")
    print(f"  kv_cache: {kv_cache.shape}")
    print(f"  kv_lens: {kv_lens.shape} = {kv_lens}")
    print(f"  page_indices: {page_indices.shape}", page_indices)
    print(f"  cu_q_lens: {cu_q_lens.shape} = {cu_q_lens}")
    print(f"  distribution: {distribution.shape} = {distribution}")
    
    return {
        'q': q,
        'k': k,
        'v': v,
        'kv_cache': kv_cache,
        'kv_lens': kv_lens,
        'page_indices': page_indices,
        'cu_q_lens': cu_q_lens,
        'distribution': distribution,
        'attention_sink': attention_sink,
        'sm_scale': sm_scale,
    }

def run(inputs, dump_dir, dp_size=2, num_devices=8, dp=True):
    """Run attention kernel WITH data parallelism."""
    print("\n" + "="*80)
    if dp: 
        print(f"Running WITH Data Parallelism (dp_size={dp_size}, num_devices={num_devices})")
    else:
        print(f"Running WITHOUT Data Parallelism (num_devices={num_devices})")
    print("="*80)
    
    print(f"Input shapes:")
    for key, val in inputs.items():
        if isinstance(val, jax.Array):
            print(f"  {key}: {val.shape} {val.dtype}")
    
    devices = jax.devices()
    total_devices = 8
    if dp: 
        device_array = np.array(devices[:total_devices]).reshape(dp_size, -1)
    else: 
        device_array = np.array(devices[:total_devices//dp_size]).reshape(1, -1)
    mesh = Mesh(
        device_array,
        axis_names=(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD),
    )
    print(f"Mesh: {mesh}")
    print(f"Mesh shape: {mesh.shape}")


    # Define sharding specs (matching the actual implementation)
    qkv_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None)
    kv_cache_spec = P(ShardingAxisName.ATTN_DATA, None, ShardingAxisName.ATTN_HEAD, None, None)
    
    in_specs = (
        qkv_spec,  # q
        qkv_spec,  # k
        qkv_spec,  # v
        kv_cache_spec,  # kv_cache
        P(ShardingAxisName.ATTN_DATA),  # kv_lens
        P(ShardingAxisName.ATTN_DATA),  # page_indices
        P(ShardingAxisName.ATTN_DATA),  # cu_q_lens
        P(ShardingAxisName.ATTN_DATA),  # distribution
    )
    out_specs = (qkv_spec, kv_cache_spec)
    
    print(f"\nSharding specs:")
    print(f"  qkv_spec: {qkv_spec}")
    print(f"  kv_cache_spec: {kv_cache_spec}")
    
    # Create sharded function
    def _ragged_paged_attention(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution):
        return ragged_paged_attention_hd64(
            q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution,
            attention_sink=None,
            sm_scale=inputs['sm_scale'],
        )
    
    sharded_fn = shard_map(
        _ragged_paged_attention,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )
    
    attention_fn = jax.jit(sharded_fn)
    
    # Lower and dump HLO
    print("\nLowering computation...")
    lowered = attention_fn.lower(
        inputs['q'],
        inputs['k'],
        inputs['v'],
        inputs['kv_cache'],
        inputs['kv_lens'],
        inputs['page_indices'],
        inputs['cu_q_lens'],
        inputs['distribution'],
    )
    
    # Dump HLO
    hlo_text = lowered.as_text()
    if dp:  
        hlo_path = Path(dump_dir) / f"with_dp_{dp_size}_hlo.txt"
    else: 
        hlo_path = Path(dump_dir) / f"no_dp_hlo.txt"
    hlo_path.write_text(hlo_text)
    print(f"✓ HLO dumped to: {hlo_path}")
    
    # Compile
    print("Compiling...")
    compiled = lowered.compile()
    
    inputs['q'] = jax.device_put(inputs['q'], NamedSharding(mesh, qkv_spec))
    inputs['k'] = jax.device_put(inputs['k'], NamedSharding(mesh, qkv_spec))
    inputs['v'] = jax.device_put(inputs['v'], NamedSharding(mesh, qkv_spec))
    inputs['kv_cache'] = jax.device_put(inputs['kv_cache'], NamedSharding(mesh, kv_cache_spec))
    inputs['kv_lens'] = jax.device_put(inputs['kv_lens'], NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA)))
    inputs['page_indices'] = jax.device_put(inputs['page_indices'], NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA)))
    inputs['cu_q_lens'] = jax.device_put(inputs['cu_q_lens'], NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA)))
    inputs['distribution'] = jax.device_put(inputs['distribution'], NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA)))
    
    # Warm up
    print("Warming up...")
    for _ in range(3):
        result = attention_fn(
            inputs['q'],
            inputs['k'],
            inputs['v'],
            inputs['kv_cache'],
            inputs['kv_lens'],
            inputs['page_indices'],
            inputs['cu_q_lens'],
            inputs['distribution'],
        )
        jax.block_until_ready(result)
    
    # Benchmark
    print("Benchmarking (10 runs)...")
    times = []
    for i in range(10):
        start = time.time()
        result = attention_fn(
            inputs['q'],
            inputs['k'],
            inputs['v'],
            inputs['kv_cache'],
            inputs['kv_lens'],
            inputs['page_indices'],
            inputs['cu_q_lens'],
            inputs['distribution'],
        )
        jax.block_until_ready(result)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.2f} ms")
    
    avg_time = sum(times) / len(times)
    print(f"\n✓ Average time: {avg_time*1000:.2f} ms")
    print(f"✓ Output shape: {result[0].shape}")
    
    return result, avg_time


def main():
    parser = argparse.ArgumentParser(description="Reproduce attention DP issue")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['no_dp', 'with_dp', 'both'],
        default='both',
        help='Which mode to run'
    )
    parser.add_argument(
        '--dp_size',
        type=int,
        default=2,
        help='Data parallel size for with_dp mode'
    )
    parser.add_argument(
        '--num_devices_no_dp',
        type=int,
        default=4,
        help='Number of devices for no_dp mode (head parallelism only)'
    )
    parser.add_argument(
        '--num_devices_with_dp',
        type=int,
        default=8,
        help='Number of devices for with_dp mode (dp_size * head_parallelism)'
    )
    parser.add_argument(
        '--dump_dir',
        type=str,
        default='/tmp/attention_dp_repro',
        help='Directory to dump HLO files'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size'
    )
    args = parser.parse_args()
    
    # Create dump directory
    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    print(f"HLO dump directory: {dump_dir}")
    
    # Create test inputs
    print("\nCreating test inputs...")
    # Note: We'll create DP-specific inputs when running with_dp mode
    inputs_no_dp = None
    inputs_with_dp = None
    
    print(f"\nJAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Device count: {jax.device_count()}")
    
    results = {}
    
    # Run without DP
    if args.mode in ['no_dp', 'both']:
        print("\nCreating inputs for no_dp mode...")
        inputs_no_dp = create_test_inputs(batch_size=args.batch_size, dp_size=1)
        result, avg_time = run(inputs_no_dp, dump_dir, num_devices=args.num_devices_no_dp, dp=False)
        results['no_dp'] = {'result': result, 'time': avg_time}
    
    # Run with DP
    if args.mode in ['with_dp', 'both']:
        print(f"\nCreating inputs for with_dp mode (dp_size={args.dp_size})...")
        inputs_with_dp = create_test_inputs(batch_size=args.batch_size, dp_size=args.dp_size)
        result, avg_time = run(inputs_with_dp, dump_dir, dp_size=args.dp_size, num_devices=args.num_devices_with_dp, dp=True)
        results['with_dp'] = {'result': result, 'time': avg_time}
    
    # Compare results
    if args.mode == 'both':
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        
        no_dp_time = results['no_dp']['time']
        with_dp_time = results['with_dp']['time']
        speedup = no_dp_time / with_dp_time
        slowdown = with_dp_time / no_dp_time
        
        print(f"No DP time:   {no_dp_time*1000:.2f} ms")
        print(f"With DP time: {with_dp_time*1000:.2f} ms")
        
        if speedup > 1.0:
            print(f"✓ DP is {speedup:.2f}x FASTER")
        else:
            print(f"✗ DP is {slowdown:.2f}x SLOWER")
        
        # Check output consistency
        total = results['no_dp']['result'][0].shape[0]
        for i in range(args.dp_size):
            
            out_no_dp = np.array(results['no_dp']['result'][0], dtype=np.float32)
            out_with_dp = np.array(results['with_dp']['result'][0][total*i : total*(i+1), ...], dtype=np.float32)

            max_diff = np.max(np.abs(out_no_dp - out_with_dp))
            print(f"\nMax output difference for DP rank {i}: {max_diff}")
        
        if max_diff < 1e-2:
            print("✓ Outputs match (within tolerance)")
        else:
            print("✗ Outputs differ significantly!")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"\n1. Compare HLO files in: {dump_dir}")
    print(f"   - no_dp_hlo.txt")
    print(f"   - with_dp_{args.dp_size}_hlo.txt")
    print(f"\n2. Look for:")
    print(f"   - all-reduce, all-gather, collective-permute operations")
    print(f"   - Extra transpose/reshape operations")
    print(f"   - Different memory layouts")
    print(f"\n3. Use diff tool:")
    print(f"   diff {dump_dir}/no_dp_hlo.txt {dump_dir}/with_dp_{args.dp_size}_hlo.txt")
    print(f"\n4. Or use XLA dump for more detail:")
    print(f"   XLA_FLAGS='--xla_dump_to={dump_dir}/xla_dump --xla_dump_hlo_as_text' python {__file__} --mode {args.mode}")
    print()


if __name__ == '__main__':
    main()
