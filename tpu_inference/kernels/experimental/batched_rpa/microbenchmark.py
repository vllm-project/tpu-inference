"""Microbenchmark for Batched Ragged Paged Attention (Batched RPA) kernel.

Adapted for standalone and pytest execution on Cloud TPU VM.
"""

import dataclasses
import time
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.experimental.batched_rpa import configs
from tpu_inference.kernels.experimental.batched_rpa import kernel
from tpu_inference.kernels.experimental.batched_rpa import wrapper


@dataclasses.dataclass(frozen=True)
class ModelParams:
  """Unsharded parameters for LLM models."""

  num_q_heads: int
  num_kv_heads: int
  head_dim: int
  num_devices: list[int]

  def get_sharded_heads(self, num_devices: int) -> tuple[int, int]:
    """Calculates sharded Query and KV head counts for a given TP degree."""
    assert self.num_q_heads % num_devices == 0
    sharded_kv_heads = self.num_kv_heads // num_devices
    if sharded_kv_heads == 0:
      assert num_devices % self.num_kv_heads == 0
      sharded_kv_heads = 1
    return self.num_q_heads // num_devices, sharded_kv_heads

  def __getitem__(self, key: str):
    return getattr(self, key)


MODEL_CONFIGS = {
    'Gemma4-31B': ModelParams(
        num_q_heads=32,
        num_kv_heads=16,
        head_dim=256,
        num_devices=[8],
    ),
    'Qwen3-32B': ModelParams(
        num_q_heads=64,
        num_kv_heads=8,
        head_dim=128,
        num_devices=[2],
    ),
    'Qwen3.5': ModelParams(
        num_q_heads=32,
        num_kv_heads=2,
        head_dim=256,
        num_devices=[8],
    ),
    'Qwen3-Coder-480B': ModelParams(
        num_q_heads=96,
        num_kv_heads=8,
        head_dim=128,
        num_devices=[8],
    ),
}

# Benchmark configurations to run.
BENCHMARK_CONFIGS = [
    # --- Gemma4-31B Benchmark Suites ---
    # 1. Speculative Decoding (MTP: 1 target + 3 draft tokens = 4 tokens scheduled)
    dict(
        model_name='Gemma4-31B',
        mode='spec_decode',
        seq_len=4,
        num_seqs=32,
        page_size=256,
        q_dtype='bfloat16',
        kv_dtype='float8_e4m3fn',
    ),
    # 2. Standard Decode (1 token scheduled)
    dict(
        model_name='Gemma4-31B',
        mode='decode',
        seq_len=1,
        num_seqs=32,
        page_size=256,
        q_dtype='bfloat16',
        kv_dtype='float8_e4m3fn',
    ),
    # 3. Standard Prefill (1024 prompt tokens)
    dict(
        model_name='Gemma4-31B',
        mode='prefill',
        seq_len=1024,
        num_seqs=1,
        page_size=256,
        q_dtype='bfloat16',
        kv_dtype='float8_e4m3fn',
    ),
]


def print_table(header, rows, *, col_width_extra=None):
  """Prints a formatted ASCII/Unicode table to stdout."""
  if col_width_extra is None:
    col_width_extra = {}
  sz = len(header)
  col_width = [(len(str(h)) + 3 + col_width_extra.get(h, 0)) for h in header]
  start_separator = '╒' + '╤'.join('═' * w for w in col_width) + '╕'
  middle_separator = '╞' + '╪'.join('═' * w for w in col_width) + '╡'
  end_separator = '╘' + '╧'.join('═' * w for w in col_width) + '╛'
  fmt = '│' + '│'.join('{{:<{}}}'.format(w) for w in col_width) + '│'
  print(start_separator)
  print(fmt.format(*header))
  for row in rows:
    assert len(row) == sz
    print(middle_separator)
    print(fmt.format(*row))
  print(end_separator)


def get_device_name(num_devices: int | None = None) -> str:
  """Returns the name of the TPU device."""
  try:
    kind = jax.devices()[0].device_kind
  except Exception:
    kind = 'TPU'
  if num_devices is not None:
    kind += f'-{num_devices}'
  return kind


def run_benchmark_and_timing(
    run_fn,
    fn_args: tuple[any, ...],
    verbose: bool = False,
    num_warmup: int = 5,
    num_iters: int = 20,
) -> float:
  """Runs warmup and precise timing via JAX synchronization."""
  def _copy_arg(x):
    if isinstance(x, (jax.Array, np.ndarray)):
      return jnp.copy(x)
    return x

  if verbose:
    print('Warming up JAX kernel...')
  for _ in range(num_warmup):
    out = run_fn(*[_copy_arg(x) for x in fn_args])
    jax.block_until_ready(out)

  if verbose:
    print(f'Timing ({num_iters} iterations)...')
  start = time.perf_counter()
  for _ in range(num_iters):
    copied_args = [_copy_arg(x) for x in fn_args]
    out = run_fn(*copied_args)
    jax.block_until_ready(out)
  end = time.perf_counter()

  t_us = ((end - start) * 1e6) / num_iters
  return t_us


def run_one(
    model_name: str,
    num_devices: int,
    mode: str,
    seq_len: int,
    num_seqs: int,
    page_size: int,
    q_dtype_name: str,
    kv_dtype_name: str,
    bq_sz: int | None = None,
    bkv_sz: int | None = None,
    batch_size: int | None = None,
    n_buffer: int | None = None,
    verbose: bool = False,
) -> tuple[str, ...]:
  """Runs a single microbenchmark configuration and returns a table row."""
  model_config = MODEL_CONFIGS[model_name]
  device_name = get_device_name(num_devices)

  q_dtype = jnp.dtype(q_dtype_name)
  kv_dtype = jnp.dtype(kv_dtype_name)

  custom_bs = None
  if bq_sz is not None and bkv_sz is not None and batch_size is not None:
    custom_bs = configs.BlockSizes(
        bq_sz=bq_sz,
        bq_c_sz=bq_sz,
        bkv_sz=bkv_sz,
        batch_size=batch_size,
        n_buffer=n_buffer or 3,
    )

  actual_num_q_heads, actual_num_kv_heads = model_config.get_sharded_heads(
      num_devices
  )
  actual_head_dim = model_config.head_dim

  if mode == 'prefill':
    total_tokens = num_seqs * seq_len
    cu_q_lens = jnp.arange(num_seqs + 1, dtype=jnp.int32) * seq_len
    kv_lens = jnp.full((num_seqs,), seq_len, dtype=jnp.int32)
    distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)
    pages_per_seq = (seq_len + page_size - 1) // page_size
    rpa_case = configs.RpaCase.MIXED
  elif mode == 'decode':
    total_tokens = num_seqs
    cu_q_lens = jnp.arange(num_seqs + 1, dtype=jnp.int32)
    kv_lens = jnp.full((num_seqs,), 1024 + 1, dtype=jnp.int32)
    distribution = jnp.array([num_seqs, num_seqs, num_seqs], dtype=jnp.int32)
    pages_per_seq = (1024 + 1 + page_size - 1) // page_size
    rpa_case = configs.RpaCase.DECODE
  elif mode in ('spec_decode', 'mixed'):
    total_tokens = num_seqs * seq_len
    cu_q_lens = jnp.arange(num_seqs + 1, dtype=jnp.int32) * seq_len
    kv_lens = jnp.full((num_seqs,), 1024 + seq_len, dtype=jnp.int32)
    distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)
    pages_per_seq = (1024 + seq_len + page_size - 1) // page_size
    rpa_case = configs.RpaCase.MIXED
  else:
    raise ValueError(f'Unknown mode: {mode}')

  max_num_seqs = max(128, num_seqs)
  if num_seqs < max_num_seqs:
    padding = max_num_seqs - num_seqs
    kv_lens = jnp.pad(kv_lens, (0, padding))
    cu_q_lens = jnp.pad(cu_q_lens, (0, padding))

  num_pages = max_num_seqs * pages_per_seq

  q_key, k_key, v_key, cache_key, pi_key = jax.random.split(
      jax.random.key(0), 5
  )

  q = jax.random.uniform(
      q_key, (total_tokens, actual_num_q_heads, actual_head_dim), dtype=q_dtype
  )
  k = jax.random.uniform(
      k_key,
      (total_tokens, actual_num_kv_heads, actual_head_dim),
      dtype=kv_dtype,
  )
  v = jax.random.uniform(
      v_key,
      (total_tokens, actual_num_kv_heads, actual_head_dim),
      dtype=kv_dtype,
  )

  kv_cache_shape = wrapper.get_kv_cache_shape(
      num_pages,
      page_size,
      actual_num_kv_heads,
      actual_head_dim,
      kv_dtype,
  )
  kv_cache = jax.random.uniform(cache_key, kv_cache_shape, dtype=kv_dtype)

  page_indices = jax.random.randint(
      pi_key,
      (max_num_seqs * pages_per_seq,),
      0,
      num_pages,
      dtype=jnp.int32,
  )

  k_scale = 0.5 if q_dtype != kv_dtype else None
  v_scale = 0.5 if q_dtype != kv_dtype else None

  decode_bs = custom_bs if mode == 'decode' else None
  prefill_bs = custom_bs if mode in ('prefill', 'spec_decode', 'mixed') else None

  model_cfgs = configs.ModelConfigs(
      num_q_heads=actual_num_q_heads,
      num_kv_heads=actual_num_kv_heads,
      head_dim=actual_head_dim,
      mask_value=-1e9,
  )
  serve_cfgs = configs.ServingConfigs(
      num_seqs=max_num_seqs,
      page_size=page_size,
      total_q_tokens=total_tokens,
      num_page_indices=num_pages,
      dtype_q=q_dtype,
      dtype_kv=kv_dtype,
      dtype_out=q_dtype,
  )

  vmem_limit = 128 * 1024 * 1024
  try:
    vmem_limit = pltpu.get_tpu_info().vmem_capacity_bytes
  except Exception:
    pass

  if custom_bs is None:
    default_decode, default_prefill, default_spec_decode = wrapper.calculate_block_sizes(
        model_cfgs, serve_cfgs, vmem_limit
    )
    if mode == 'decode':
      eff_block = default_decode
    elif mode in ('spec_decode', 'mixed'):
      eff_block = default_spec_decode
    else:
      eff_block = default_prefill
  else:
    eff_block = custom_bs

  cfgs = configs.RpaConfigs(
      block=eff_block,
      model=model_cfgs,
      serve=serve_cfgs,
      vmem_limit_bytes=vmem_limit,
      mode=rpa_case,
  )

  @jax.jit
  def run_fn(q, k, v, kv_cache):
    out, cache = wrapper.ragged_paged_attention(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        k_scale=k_scale,
        v_scale=v_scale,
        decode_block_sizes=decode_bs,
        prefill_block_sizes=prefill_bs,
    )
    return out, cache

  t_us = run_benchmark_and_timing(
      run_fn,
      (q, k, v, kv_cache),
      verbose=verbose,
  )

  t_us_str = f'{t_us:.2f}' if t_us is not None else '0.0'
  block_size_str = (
      f'q{eff_block.bq_sz}/k{eff_block.bkv_sz}/b{eff_block.batch_size}'
  )

  return (
      device_name,
      model_name,
      mode,
      str(seq_len),
      str(num_seqs),
      block_size_str,
      q_dtype_name,
      kv_dtype_name,
      t_us_str,
  )


def run_all_benchmarks(verbose: bool = True):
  """Runs all benchmarks in BENCHMARK_CONFIGS and prints results."""
  results = []
  header = [
      'device',
      'model',
      'mode',
      'q_len',
      'batch',
      'block_sizes',
      'q_dtype',
      'kv_dtype',
      'latency (us)',
  ]
  col_width_extra = {
      'model': 12,
      'block_sizes': 12,
      'kv_dtype': 6,
      'latency (us)': 6,
  }

  for cfg in BENCHMARK_CONFIGS:
    model_name = cfg['model_name']
    mode = cfg['mode']
    seq_len = cfg['seq_len']
    num_seqs = cfg['num_seqs']
    q_dtype_name = cfg['q_dtype']
    kv_dtype_name = cfg['kv_dtype']

    model_config = MODEL_CONFIGS[model_name]
    num_devices_list = cfg.get('num_devices', model_config.num_devices)
    if isinstance(num_devices_list, int):
      num_devices_list = [num_devices_list]

    page_size = cfg.get('page_size', 256)

    for num_devices in num_devices_list:
      if verbose:
        print(f'\nRunning: {model_name} | mode={mode} | q_len={seq_len} | batch={num_seqs}...')
      row = run_one(
          model_name=model_name,
          num_devices=num_devices,
          mode=mode,
          seq_len=seq_len,
          num_seqs=num_seqs,
          page_size=page_size,
          q_dtype_name=q_dtype_name,
          kv_dtype_name=kv_dtype_name,
          verbose=verbose,
      )
      results.append(row)

  print('\n=== BATCHED RPA MICROBENCHMARK RESULTS ===')
  print_table(header, results, col_width_extra=col_width_extra)
  return results


# PyTest compatibility
def test_microbenchmark():
  results = run_all_benchmarks(verbose=False)
  assert len(results) > 0


if __name__ == '__main__':
  run_all_benchmarks(verbose=True)
