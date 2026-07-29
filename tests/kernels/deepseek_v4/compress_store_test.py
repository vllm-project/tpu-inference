# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Correctness test for Kernel 2 (compress_norm_rope_store)."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.experimental.deepseek_v4.compress_and_store import compress_store_ref
from tpu_inference.kernels.experimental.deepseek_v4.compress_and_store import config
from tpu_inference.kernels.experimental.deepseek_v4.compress_and_store import kernel as compress_store
from tpu_inference.kernels.experimental.deepseek_v4.compress_and_store import proj_and_save_state as proj_and_save_state_kernel
from tpu_inference.kernels.experimental.deepseek_v4.compress_and_store import project_and_save_state_ref as proj_and_save_state_ref


def generate_kv_slot_mapping(
    num_boundary_tokens: int,
    num_pages: int,
    page_size: int,
    slots_per_part_out: int,
) -> jax.Array:
  """Generates kv_slot_mapping for boundary tokens."""
  kv_slot_mapping_np = np.full((num_boundary_tokens,), -1, dtype=np.int32)
  boundary_count = 0
  total_slots_needed = num_boundary_tokens * slots_per_part_out
  pages_needed = (total_slots_needed + page_size - 1) // page_size
  start_page = max(0, num_pages - pages_needed)

  for t in range(num_boundary_tokens):
    kv_slot_mapping_np[t] = start_page * page_size + boundary_count
    boundary_count += slots_per_part_out
  return jnp.array(kv_slot_mapping_np, dtype=jnp.int32)


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
      hidden_size=None,
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
    # Real models do NOT have hidden_size == state_dim (DSv4-Flash: 4096 vs
    # 2048). Keeping them equal hides any k-tiling bug in the projection
    # kernel, because it makes `tile_k` divide `size_k` for free.
    if hidden_size is None:
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
    # `ref_wkv_proj_and_save_state` computes `hidden_states @ wkv_wgate`, so
    # the weight is [hidden_size, state_dim] -- indistinguishable from
    # [state_dim, hidden_size] only while the two sizes are equal.
    wkv_wgate = jax.random.normal(k2, (hidden_size, state_dim))
    ape = jax.random.normal(k3, (compress_ratio, state_width))
    run_1_positions = jnp.arange(run_1_tokens, dtype=jnp.int32)

    # slot_mapping is in physical cache-row units: both the projection kernel
    # (p = slot // cache.shape[1]) and its reference (slot // (page_size //
    # state_block_size)) read it that way, so one token advances by the number
    # of rows its state occupies -- not by the number of 128B sub-slots.
    slots_per_token = physical_page_size // state_block_size
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
      kv_slot_mapping_filtered = generate_kv_slot_mapping(
          num_boundary,
          num_pages,
          # Slots per page: kv_block_size records/page x kv_stride slots/record.
          cfgs.kv_block_size * cfgs.kv_stride,
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
          # DSv4-Flash's real geometry: hidden_size 4096 vs state_dim 2048.
          # `tile_k` must divide hidden_size or the last k block of the
          # projection reads out of bounds and every state value comes back
          # NaN. Every other case here has hidden_size == state_dim, which
          # makes that divisibility automatic.
          "csa_prod_hidden_4096",
          dict(
              num_tokens=8,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=4,
              overlap=True,
              physical_page_size=256,
              hidden_size=4096,
          ),
      ),
      (
          # hidden_size that is 128-aligned but shares no large divisor with
          # the 3584 cap, exercising the fallback in `_select_tile_k`.
          "csa_prod_hidden_5120",
          dict(
              num_tokens=8,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=4,
              overlap=True,
              physical_page_size=256,
              hidden_size=5120,
          ),
      ),
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

  @parameterized.named_parameters(
      # `hidden_size` is the k dimension of the projection. The kernel tiles it
      # with `_select_tile_k` and `emit_pipeline` runs a
      # `cdiv(size_k, tile_k)` grid; a `tile_k` that does not divide
      # `hidden_size` leaves the final partial k block of the bf16
      # `wkv_wgate` (whose k is the sublane dim) partly unwritten, and since
      # the matmul contracts over k that lands in *every* output element.
      #
      # Both axes matter: bf16 is required to trigger it (f32 handles the
      # remainder correctly) and bf16 is what production runs, so do not
      # "simplify" these to f32.
      ("bf16_hidden_4096_dsv4_flash", 4096, jnp.bfloat16),  # production shape
      ("bf16_hidden_5120_no_large_divisor", 5120, jnp.bfloat16),
      ("bf16_hidden_2048_eq_state", 2048, jnp.bfloat16),
      ("bf16_hidden_3584_at_cap", 3584, jnp.bfloat16),
      ("bf16_hidden_7168_two_full_tiles", 7168, jnp.bfloat16),
      ("f32_hidden_4096", 4096, jnp.float32),
  )
  def test_proj_and_save_state_matches_reference(self, hidden_size, dtype):
    """The Pallas projection kernel vs its JAX reference.

    ``run_compress_store_correctness`` populates the state cache with the
    *reference*, so it never exercises this kernel at all; without this test
    the projection has no coverage.
    """
    head_dim, compress_ratio, overlap = 512, 4, True
    physical_page_size, num_tokens = 256, 8
    coff = 1 + int(overlap)
    state_width = coff * head_dim
    state_dim = 2 * state_width
    state_block_size = 16  # physical_page_size // state_rows_per_token
    slots_per_token = physical_page_size // state_block_size

    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    hidden_states = jax.random.normal(k1, (num_tokens, hidden_size)).astype(
        dtype)
    wkv_wgate = jax.random.normal(k2, (hidden_size, state_dim)).astype(dtype)
    ape = jax.random.normal(k3, (compress_ratio, state_width))
    positions = jnp.arange(num_tokens, dtype=jnp.int32)
    slot_mapping = jnp.arange(num_tokens, dtype=jnp.int32) * slots_per_token
    cache = jnp.zeros((8, physical_page_size, 4, 128), dtype=jnp.uint8)

    actual = proj_and_save_state_kernel.proj_and_save_state(
        hidden_states=hidden_states,
        wkv_wgate=wkv_wgate,
        ape=ape,
        positions=positions,
        slot_mapping=slot_mapping,
        cache=cache,
        compress_ratio=compress_ratio,
    )
    # The kernel multiplies the bf16 operands but accumulates in f32, so the
    # reference is run in f32 on the same bit-exact values -- otherwise
    # `jnp.matmul` would return bf16 and the comparison would be measuring
    # output rounding rather than the kernel.
    expected = proj_and_save_state_ref.ref_wkv_proj_and_save_state(
        hidden_states=hidden_states.astype(jnp.float32),
        wkv_wgate=wkv_wgate.astype(jnp.float32),
        ape=ape,
        positions=positions,
        slot_mapping=slot_mapping,
        cache=cache,
        state_block_size=state_block_size,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
    )
    # Decode to f32: NaN is the failure mode this guards, and a raw uint8 diff
    # would report it as an opaque byte mismatch. The state is stored
    # byte-transposed -- u8[..., s, l] is byte s of the f32 in lane l.
    def _decode(cache_u8):
      b = jnp.asarray(cache_u8).astype(jnp.uint32)
      word = (b[..., 0, :] | (b[..., 1, :] << 8) | (b[..., 2, :] << 16)
              | (b[..., 3, :] << 24))
      return jax.lax.bitcast_convert_type(word, jnp.float32)

    actual_f32 = _decode(actual)
    expected_f32 = _decode(expected)
    self.assertFalse(
        bool(jnp.isnan(actual_f32).any()),
        f"projection produced NaN at hidden_size={hidden_size}")
    # Not bit-exact: whenever `hidden_size` needs more than one k tile the
    # kernel accumulates partial sums in a different order than the
    # reference's single matmul, so the low bits of the f32 differ.
    self.assertAllClose(actual_f32, expected_f32, rtol=2e-5, atol=1e-4)

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