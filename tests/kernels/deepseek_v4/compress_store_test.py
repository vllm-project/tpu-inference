"""Correctness test for Kernel 2 (compress_norm_rope_store)."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
import jax.numpy as jnp
import numpy as np

from google3.experimental.tpu_perf_showcase.vllm.kernels.deepseek_v4.compressor import benchmark_util
from google3.experimental.tpu_perf_showcase.vllm.kernels.deepseek_v4.compressor.compress_store import ref as compress_store_ref
from google3.experimental.tpu_perf_showcase.vllm.kernels.deepseek_v4.compressor.compress_store.tc import config
from google3.experimental.tpu_perf_showcase.vllm.kernels.deepseek_v4.compressor.compress_store.tc import kernel as compress_store
from google3.experimental.tpu_perf_showcase.vllm.kernels.deepseek_v4.compressor.proj_and_save_state import ref as proj_and_save_state_ref


def normalize_fp8_zero_sign(arr):
  is_zero = (arr & 0x7f) == 0
  return jnp.where(is_zero, 0, arr)


def normalize_bf16_zero_sign(arr):
  orig_shape = arr.shape
  arr_2d = arr.reshape(-1, 2)
  even = arr_2d[..., 0]
  odd = arr_2d[..., 1]
  is_zero = (even == 0) & ((odd & 0x7f) == 0)
  new_odd = jnp.where(is_zero, odd & 0x7f, odd)
  normalized = jnp.stack([even, new_odd], axis=-1)
  return normalized.reshape(orig_shape)


class CompressStoreTest(jtu.JaxTestCase):

  def run_compress_store_correctness(
      self,
      num_tokens,
      head_dim,
      rope_head_dim,
      compress_ratio,
      overlap,
      physical_page_size,
      quant_block=64,
      rms_eps=1e-6,
      positions=None,
      token_to_req_indices=None,
      block_table=None,
      kv_slot_mapping=None,
      prefill_len=None,
      interpret=False,
  ):
    # Create config first
    if head_dim == 128:
      mode = config.Mode.CSA_INDEXER
    else:
      mode = config.Mode.CSA if overlap else config.Mode.HCA

    cfgs = config.Configs.make(
        mode,
        size_n=num_tokens,
        physical_page_size=physical_page_size,
        rms_eps=rms_eps,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        quant_block=quant_block,
    )

    state_block_size = cfgs.state_block_size

    state_width = cfgs.state_width
    state_dim = 2 * state_width
    hidden_size = state_dim

    # Setup positions and identify boundary tokens
    if positions is None:
      positions_np = np.arange(num_tokens, dtype=np.int32)
    else:
      positions_np = np.array(positions, dtype=np.int32)

    boundary_mask = ((positions_np + 1) % compress_ratio) == 0
    positions_filtered = positions_np[boundary_mask]
    num_boundary = positions_filtered.shape[0]

    # Calculate num_pages
    run_1_tokens = prefill_len if prefill_len is not None else num_tokens
    tokens_per_page = state_block_size
    pages_for_state = (run_1_tokens + tokens_per_page - 1) // tokens_per_page

    total_bytes_needed = num_boundary * cfgs.record_bytes
    page_bytes = cfgs.physical_page_size * cfgs.row_size_bytes
    pages_for_kv_cache = (total_bytes_needed + page_bytes - 1) // page_bytes
    num_pages = pages_for_state + pages_for_kv_cache

    # 1. Initialize keys
    k = jax.random.key(0)
    k1, k2, k3, k4, k5 = jax.random.split(k, 5)

    # 2. Populate cache
    hidden_states = jax.random.normal(k1, (run_1_tokens, hidden_size))
    wkv_wgate = jax.random.normal(k2, (state_dim, hidden_size))
    ape = jax.random.normal(k3, (compress_ratio, state_width))
    run_1_positions = jnp.arange(run_1_tokens, dtype=jnp.int32)

    slots_per_token = (state_dim * 4) // 128
    run_1_slot_mapping = np.arange(run_1_tokens) * slots_per_token
    run_1_slot_mapping = jnp.array(run_1_slot_mapping, dtype=jnp.int32)

    init_cache = jnp.zeros(cfgs.cache_shape(num_pages), dtype=jnp.uint8)

    ref_wkv_proj_and_save_state_jit = jax.jit(
        proj_and_save_state_ref.ref_wkv_proj_and_save_state,
        static_argnums=(6, 7, 8, 9),
    )
    populated_cache = ref_wkv_proj_and_save_state_jit(
        hidden_states=hidden_states,
        wkv_wgate=wkv_wgate,
        ape=ape,
        positions=run_1_positions,
        slot_mapping=run_1_slot_mapping,
        cache=init_cache,
        state_block_size=state_block_size,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
    )


    # 3. Setup Kernel 2 inputs
    if token_to_req_indices is None:
      token_to_req_indices_filtered = np.zeros((num_boundary,), dtype=np.int32)
    else:
      token_to_req_indices_np = np.array(token_to_req_indices)
      token_to_req_indices_filtered = token_to_req_indices_np[boundary_mask]

    if kv_slot_mapping is None:
      kv_slot_mapping_filtered = benchmark_util.generate_kv_slot_mapping(
          num_boundary,
          num_pages,
          cfgs.kv_block_size,
          cfgs.kv_stride,
      )
    else:
      kv_slot_mapping_np = np.array(kv_slot_mapping)
      kv_slot_mapping_filtered = kv_slot_mapping_np[boundary_mask]

    # Pad to num_tokens
    positions_padded = np.pad(
        positions_filtered,
        (0, num_tokens - num_boundary),
        constant_values=0,
    )
    positions = jnp.array(positions_padded)

    token_to_req_indices_padded = np.pad(
        token_to_req_indices_filtered,
        (0, num_tokens - num_boundary),
        constant_values=0,
    )
    token_to_req_indices = jnp.array(token_to_req_indices_padded)

    kv_slot_mapping_padded = np.pad(
        kv_slot_mapping_filtered,
        (0, num_tokens - num_boundary),
        constant_values=-1,
    )
    kv_slot_mapping = jnp.array(kv_slot_mapping_padded)


    if block_table is None:
      block_table = jnp.array([[i for i in range(num_pages)]], dtype=jnp.int32)
    else:
      block_table = jnp.array(block_table)

    rms_weight = jax.random.normal(k4, (head_dim,))

    max_pos = int(jnp.max(positions)) + 1 if len(positions) > 0 else 0
    cos_sin_cache_len = max(max_pos, num_tokens)
    cos_sin_cache = jax.random.normal(k5, (cos_sin_cache_len, rope_head_dim))



    if cfgs.dims.has_rope_cache:
      init_rope_cache = jnp.zeros(cfgs.rope_cache_shape(num_pages), dtype=jnp.uint8)
    else:
      init_rope_cache = jnp.zeros((num_pages, 1, 1, 128), dtype=jnp.uint8)
    slot_mapping = jnp.where(positions >= 0, positions * slots_per_token, -1)

    is_quantized = overlap
    has_rope = rope_head_dim > 0

    # 4. Run Reference
    ref_compress_norm_rope_store_jit = jax.jit(
        compress_store_ref.ref_compress_norm_rope_store,
        static_argnums=(9, 10, 11, 12, 13, 14, 15, 16, 17, 18),
    )
    ref_out = ref_compress_norm_rope_store_jit(
        cache=populated_cache,
        rope_cache=init_rope_cache,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        state_block_size=state_block_size,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rms_eps=rms_eps,
        quant_block=quant_block,
        is_quantized=is_quantized,
        has_rope=has_rope,
        has_rope_cache=cfgs.dims.has_rope_cache,
    )
    ref_cache_output, ref_rope_output = ref_out

    # 5. Run Pallas


    pallas_out = compress_store.compress_norm_rope_store(
        jnp.copy(populated_cache),
        positions,
        block_table,
        token_to_req_indices,
        kv_slot_mapping,
        rms_weight,
        rope_cache=jnp.copy(init_rope_cache),
        cos_sin_cache=cos_sin_cache,
        compress_ratio=cfgs.dims.compress_ratio,
        overlap=cfgs.dims.overlap,
        quant_block=cfgs.dims.quant_block,
        rms_eps=cfgs.dims.rms_eps,
        interpret=interpret,
    )
    pallas_cache_output, pallas_rope_output = pallas_out

    if is_quantized:
      pallas_cache_normalized = normalize_fp8_zero_sign(pallas_cache_output)
      ref_cache_normalized = normalize_fp8_zero_sign(ref_cache_output)
    else:
      pallas_cache_normalized = normalize_bf16_zero_sign(pallas_cache_output)
      ref_cache_normalized = normalize_bf16_zero_sign(ref_cache_output)

    if cfgs.dims.has_rope_cache:
      pallas_rope_normalized = normalize_bf16_zero_sign(pallas_rope_output)
      ref_rope_normalized = normalize_bf16_zero_sign(ref_rope_output)
      self.assertArraysEqual(pallas_rope_normalized, ref_rope_normalized)

    self.assertArraysEqual(pallas_cache_normalized, ref_cache_normalized)

  @parameterized.named_parameters(
      (
          "csa_prefill",
          dict(
              num_tokens=128,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=4,
              overlap=True,
              physical_page_size=256,
          ),
      ),
      (
          "hca_prefill",
          dict(
              num_tokens=256,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=128,
              overlap=False,
              physical_page_size=128,
          ),
      ),
      (
          "hca_prefill_small",
          dict(
              num_tokens=128,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=128,
              overlap=False,
              physical_page_size=128,
          ),
      ),
      (
          "csa_decode_batch_large",
          dict(
              num_tokens=1024,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=4,
              overlap=True,
              physical_page_size=256,
              positions=(np.arange(1024, dtype=np.int32) * 3) % 1024,
              prefill_len=1024,
          ),
      ),
      (
          "csa_decode_batch_seq",
          dict(
              num_tokens=4,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=4,
              overlap=True,
              physical_page_size=256,
              positions=np.array([3, 7, 11, 15], dtype=np.int32),
              prefill_len=16,
          ),
      ),
      (
          "hca_decode_batch_seq",
          dict(
              num_tokens=4,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=128,
              overlap=False,
              physical_page_size=128,
              positions=np.array([127, 255, 383, 511], dtype=np.int32),
              prefill_len=512,
          ),
      ),
      (
          "csa_decode_batch_random",
          dict(
              num_tokens=4,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=4,
              overlap=True,
              physical_page_size=256,
              positions=np.array([11, 3, 19, 7], dtype=np.int32),
              prefill_len=32,
          ),
      ),
      (
          "hca_decode_batch_random",
          dict(
              num_tokens=4,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=128,
              overlap=False,
              physical_page_size=128,
              positions=np.array([383, 127, 511, 255], dtype=np.int32),
              prefill_len=512,
          ),
      ),
      (
          "csa_decode_batch_mixed",
          dict(
              num_tokens=6,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=4,
              overlap=True,
              physical_page_size=256,
              positions=np.array([3, 2, 7, 6, 11, 10], dtype=np.int32),
              prefill_len=32,
          ),
      ),
      (
          "hca_decode_batch_mixed",
          dict(
              num_tokens=6,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=128,
              overlap=False,
              physical_page_size=128,
              positions=np.array(
                  [127, 126, 255, 254, 383, 382], dtype=np.int32
              ),
              prefill_len=384,
          ),
      ),
      (
          "csa_indexer_prefill",
          dict(
              num_tokens=128,
              head_dim=128,
              rope_head_dim=64,
              compress_ratio=4,
              overlap=True,
              physical_page_size=32,
              quant_block=128,
          ),
      ),
      (
          "csa_indexer_decode",
          dict(
              num_tokens=4,
              head_dim=128,
              rope_head_dim=64,
              compress_ratio=4,
              overlap=True,
              physical_page_size=32,
              quant_block=128,
              positions=np.array([3, 7, 11, 15], dtype=np.int32),
              prefill_len=16,
          ),
      ),
  )
  def test_compress_store(self, cfg):
    self.run_compress_store_correctness(**cfg)

  def test_derive_aliases(self):
    # HCA: has_rope=True, has_rope_cache=False, num_scalar_prefetch=5
    self.assertEqual(
        compress_store.derive_aliases(
            has_rope=True, has_rope_cache=False, num_scalar_prefetch=5
        ),
        {7: 0},
    )

    # CSA: has_rope=True, has_rope_cache=True, num_scalar_prefetch=5
    self.assertEqual(
        compress_store.derive_aliases(
            has_rope=True, has_rope_cache=True, num_scalar_prefetch=5
        ),
        {7: 0, 8: 1},
    )

    # CSA_INDEXER: has_rope=True, has_rope_cache=False, num_scalar_prefetch=5
    self.assertEqual(
        compress_store.derive_aliases(
            has_rope=True, has_rope_cache=False, num_scalar_prefetch=5
        ),
        {7: 0},
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())