import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
  sys.path.insert(0, _DIR)

from absl.testing import absltest
from absl.testing import parameterized
from jax._src import test_util as jtu
import jax.numpy as jnp
import numpy as np

import benchmark_util
import config
import kernel as compress_store

BENCHMARK_CASES = (
    (
        "hca_prefill_bs512",
        dict(
            case_type="hca",
            batch_size=512,
            hidden_size=7168,
            head_dim=512,
            compress_ratio=128,
            state_block_size=16,
            dtype="bfloat16",
            mode="prefill",
        ),
    ),
    (
        "hca_decode_bs512",
        dict(
            case_type="hca",
            batch_size=512,
            hidden_size=7168,
            head_dim=512,
            compress_ratio=128,
            state_block_size=16,
            dtype="bfloat16",
            mode="decode",
            history_len=1024,
        ),
    ),
    (
        "csa_prefill_bs512",
        dict(
            case_type="csa",
            batch_size=512,
            hidden_size=7168,
            head_dim=512,
            compress_ratio=4,
            state_block_size=16,
            dtype="bfloat16",
            mode="prefill",
        ),
    ),
    (
        "csa_decode_bs512",
        dict(
            case_type="csa",
            batch_size=512,
            hidden_size=7168,
            head_dim=512,
            compress_ratio=4,
            state_block_size=16,
            dtype="bfloat16",
            mode="decode",
            history_len=1024,
        ),
    ),
    (
        "csa_indexer_prefill_bs512",
        dict(
            case_type="csa_indexer",
            batch_size=512,
            hidden_size=7168,
            head_dim=128,
            compress_ratio=4,
            state_block_size=16,
            dtype="bfloat16",
            mode="prefill",
        ),
    ),
    (
        "csa_indexer_decode_bs512",
        dict(
            case_type="csa_indexer",
            batch_size=512,
            hidden_size=7168,
            head_dim=128,
            compress_ratio=4,
            state_block_size=16,
            dtype="bfloat16",
            mode="decode",
            history_len=1024,
        ),
    ),
)


def run_one(
    case_type: str,
    batch_size: int,
    hidden_size: int,
    head_dim: int,
    compress_ratio: int,
    state_block_size: int,
    dtype_name: str,
    mode: str = "prefill",
    history_len: int = 1024,
    boundary_ratio: float | None = None,
) -> tuple[str, ...]:
  """Runs a single microbenchmark configuration and returns a table row."""
  device_name = benchmark_util.get_device_name()
  dtype = jnp.dtype(dtype_name)

  num_tokens = batch_size  # q_len = 1 for decode
  tile_n = 4
  rms_eps = 1e-6

  if boundary_ratio is None:
    boundary_ratio = 1.0 / compress_ratio

  if case_type == "hca":
    config_mode = config.Mode.HCA
    rope_head_dim = 64
    quant_block = 0
  elif case_type == "csa":
    config_mode = config.Mode.CSA
    rope_head_dim = 64
    quant_block = 128
  elif case_type == "csa_indexer":
    config_mode = config.Mode.CSA_INDEXER
    rope_head_dim = 64
    quant_block = 128
  else:
    raise ValueError(f"Unknown case_type: {case_type}")

  cfgs = config.Configs.make(
      config_mode,
      size_n=num_tokens,
      block_size=state_block_size,
      rms_eps=rms_eps,
      tile_n=tile_n,
      head_dim=head_dim,
      rope_head_dim=rope_head_dim,
      compress_ratio=compress_ratio,
      quant_block=quant_block,
  )

  rng = np.random.default_rng(0)

  tokens_per_page = state_block_size

  if mode == "prefill":
    num_boundary_tokens = batch_size // compress_ratio
    assert num_boundary_tokens % 4 == 0
    max_pos = num_boundary_tokens * compress_ratio
    pages_for_state = (max_pos + tokens_per_page - 1) // tokens_per_page

    total_slots_needed = num_boundary_tokens * cfgs.record_slots
    pages_for_kv_cache = (
        total_slots_needed + cfgs.page_size - 1
    ) // cfgs.page_size
    num_pages = pages_for_state + pages_for_kv_cache

    block_table = jnp.array(
        [[i for i in range(pages_for_state)]], dtype=jnp.int32
    )
    boundary_positions = (
        np.arange(num_boundary_tokens) + 1
    ) * compress_ratio - 1
    positions_np = np.pad(
        boundary_positions,
        (0, num_tokens - num_boundary_tokens),
        constant_values=0,
    )
    positions = jnp.array(positions_np, dtype=jnp.int32)
    token_to_req_indices = jnp.zeros((num_tokens,), dtype=jnp.int32)
    kv_slot_mapping_filtered = benchmark_util.generate_kv_slot_mapping(
        num_boundary_tokens, num_pages, cfgs.page_size, cfgs.record_slots
    )
    kv_slot_mapping = jnp.pad(
        kv_slot_mapping_filtered,
        (0, num_tokens - num_boundary_tokens),
        constant_values=-1,
    )

  elif mode == "decode":
    num_boundary_tokens = int(batch_size * boundary_ratio)
    if boundary_ratio > 0 and num_boundary_tokens == 0:
      num_boundary_tokens = 1

    max_pos = history_len
    pages_for_state = (max_pos + tokens_per_page - 1) // tokens_per_page

    total_slots_needed = num_boundary_tokens * cfgs.record_slots
    pages_for_kv_cache = (
        total_slots_needed + cfgs.page_size - 1
    ) // cfgs.page_size
    num_pages = batch_size * pages_for_state + pages_for_kv_cache

    block_table_np = np.zeros((batch_size, pages_for_state), dtype=np.int32)
    for i in range(batch_size):
      block_table_np[i, :] = np.arange(
          i * pages_for_state, (i + 1) * pages_for_state
      )
    block_table = jnp.array(block_table_np)

    max_boundary_idx = history_len // compress_ratio
    if max_boundary_idx <= 0:
      raise ValueError(
          f"history_len {history_len} must be >= compress_ratio"
          f" {compress_ratio}"
      )

    boundary_reqs = rng.choice(
        batch_size, size=num_boundary_tokens, replace=False
    )

    kv_slot_mapping_filtered = benchmark_util.generate_kv_slot_mapping(
        num_boundary_tokens,
        num_pages,
        cfgs.page_size,
        cfgs.record_slots,
    )

    positions_active = np.zeros((num_boundary_tokens,), dtype=np.int32)
    for idx in range(num_boundary_tokens):
      q = rng.integers(0, max_boundary_idx)
      positions_active[idx] = (q + 1) * compress_ratio - 1

    positions_np = np.pad(
        positions_active,
        (0, num_tokens - num_boundary_tokens),
        constant_values=0,
    )
    token_to_req_indices_np = np.pad(
        boundary_reqs,
        (0, num_tokens - num_boundary_tokens),
        constant_values=0,
    )
    kv_slot_mapping_np = np.pad(
        kv_slot_mapping_filtered,
        (0, num_tokens - num_boundary_tokens),
        constant_values=-1,
    )

    positions = jnp.array(positions_np, dtype=jnp.int32)
    kv_slot_mapping = jnp.array(kv_slot_mapping_np, dtype=jnp.int32)
    token_to_req_indices = jnp.array(token_to_req_indices_np, dtype=jnp.int32)
  else:
    raise ValueError(f"Unknown mode: {mode}")

  rms_weight = jnp.array(rng.random(size=(head_dim,), dtype=np.float32)).astype(
      dtype
  )

  if cfgs.dims.rope_head_dim > 0:
    cos_sin_cache = jnp.array(
        rng.random(size=(num_tokens, cfgs.dims.rope_head_dim), dtype=np.float32)
    )
  else:
    cos_sin_cache = None

  # Cache shapes
  cache, rope_cache = benchmark_util.generate_caches(cfgs, num_pages)

  # (decode, chunked-prefill, mixed) request split, as in mla.py. The prefill
  # case is one request; the decode case is one request per token.
  num_reqs = 1 if mode == "prefill" else batch_size
  distribution = jnp.array(
      [0, num_reqs, num_reqs] if mode == "prefill" else [num_reqs] * 3,
      dtype=jnp.int32,
  )

  def run_fn(
      cache,
      rope_cache,
      positions,
      block_table,
      token_to_req_indices,
      kv_slot_mapping,
      rms_weight,
      cos_sin_cache,
      distribution,
  ):
    return compress_store.compress_norm_rope_store(
        cache,
        positions,
        block_table,
        token_to_req_indices,
        kv_slot_mapping,
        rms_weight,
        rope_cache=rope_cache,
        distribution=distribution,
        cos_sin_cache=cos_sin_cache,
        compress_ratio=cfgs.dims.compress_ratio,
        overlap=cfgs.dims.overlap,
        state_block_size=cfgs.dims.block_size,
        quant_block=cfgs.dims.quant_block,
        rms_eps=cfgs.dims.rms_eps,
        # The prefill case's kv slots are consecutive from a page boundary, so
        # a tile is exactly one packed rope row -- what "prefill" expects.
        rope_store_mode=mode,
        name=kernel_name,
    )

  kernel_name = f"compress_store_{case_type}"

  session_id, t_us = benchmark_util.run_benchmark_and_profile(
      run_fn,
      (
          cache,
          rope_cache,
          positions,
          block_table,
          token_to_req_indices,
          kv_slot_mapping,
          rms_weight,
          cos_sin_cache,
          distribution,
      ),
      kernel_name,
      verbose=True,
      profile=True,
  )
  t_us_str = f"{t_us:.1f}" if t_us is not None else "N/A"
  xprof_url = f"http://xprof/?session_id={session_id}" if session_id else "N/A"
  percale_url = (
      f"https://percale.corp.google.com/llo?id={session_id}&name={kernel_name}.1"
      if session_id
      else "N/A"
  )

  config_str = (
      f"{mode}_bs{batch_size}_hs{hidden_size}_cr{compress_ratio}_tn{tile_n}"
  )

  return (
      device_name,
      case_type,
      config_str,
      dtype_name,
      t_us_str,
      xprof_url,
      percale_url,
  )


class CompressorKernelBenchmark(jtu.JaxTestCase):
  """Performance microbenchmarks for the Compressor kernel 2 (compress_store)."""

  _results = []

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    if not cls._results:
      return
    header = [
        "device",
        "case",
        "config",
        "dtype",
        "latency (us)",
        "xprof",
        "percale",
    ]
    col_width_extra = {
        "config": 15,
        "xprof": 55,
        "percale": 75,
    }
    benchmark_util.print_table(
        header, cls._results, col_width_extra=col_width_extra
    )

  @parameterized.named_parameters(*BENCHMARK_CASES)
  def test_benchmark(self, cfg):
    row = run_one(
        case_type=cfg["case_type"],
        batch_size=cfg["batch_size"],
        hidden_size=cfg["hidden_size"],
        head_dim=cfg["head_dim"],
        compress_ratio=cfg["compress_ratio"],
        state_block_size=cfg["state_block_size"],
        dtype_name=cfg["dtype"],
        mode=cfg.get("mode", "prefill"),
        history_len=cfg.get("history_len", 1024),
        boundary_ratio=cfg.get("boundary_ratio", None),
    )
    self._results.append(row)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())