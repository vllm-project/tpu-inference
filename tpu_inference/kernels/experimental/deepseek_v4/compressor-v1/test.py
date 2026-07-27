"""Correctness test for Kernel 2 (compress_norm_rope_store)."""

import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
  sys.path.insert(0, _DIR)

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
import jax.numpy as jnp
import numpy as np

import benchmark_util
import compressor
import config
import kernel as compress_store
import proj_and_save_state_ref
import ref as compress_store_ref

# Everything from `state_block_size` on is static, except `distribution`.
_REF_JIT = jax.jit(
    compress_store_ref.ref_compress_norm_rope_store,
    static_argnums=(9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20),
)


def _random_rope_cache(key, cfgs, num_pages):
  """Noise-filled rope cache, so preserved bytes are distinguishable."""
  return jax.random.randint(
      key,
      (num_pages, cfgs.rope_page_size, cfgs.rope_pack, 128),
      0,
      256,
      dtype=jnp.int32,
  ).astype(jnp.uint8)


def _changed_sub_slots(rope_cache, init_rope_cache):
  """Per-sub-slot mask of what a store actually touched."""
  return jnp.any(
      rope_cache.reshape(-1, 128) != init_rope_cache.reshape(-1, 128), axis=-1
  )


def _changed_rows(rope_cache, init_rope_cache):
  """How many packed rows a store touched at all."""
  rope_pack = rope_cache.shape[-2]
  changed = _changed_sub_slots(rope_cache, init_rope_cache)
  return int(jnp.sum(jnp.any(changed.reshape(-1, rope_pack), axis=-1)))


class CompressStoreTest(jtu.JaxTestCase):

  def run_compress_store_correctness(
      self,
      num_tokens,
      head_dim,
      rope_head_dim,
      compress_ratio,
      overlap,
      state_block_size,
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
        block_size=state_block_size,
        rms_eps=rms_eps,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        quant_block=quant_block,
    )

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

    total_slots_needed = num_boundary * cfgs.record_slots
    pages_for_kv_cache = (
        total_slots_needed + cfgs.page_size - 1
    ) // cfgs.page_size
    num_pages = pages_for_state + pages_for_kv_cache

    # 1. Initialize keys
    k = jax.random.key(0)
    k1, k2, k3, k4, k5, k6 = jax.random.split(k, 6)

    # 2. Populate cache
    hidden_states = jax.random.normal(k1, (run_1_tokens, hidden_size))
    wkv_wgate = jax.random.normal(k2, (state_dim, hidden_size))
    ape = jax.random.normal(k3, (compress_ratio, state_width))
    run_1_positions = jnp.arange(run_1_tokens, dtype=jnp.int32)

    slots_per_token = cfgs.slots_per_token
    run_1_slot_mapping = np.arange(run_1_tokens) * slots_per_token
    run_1_slot_mapping = jnp.array(run_1_slot_mapping, dtype=jnp.int32)

    init_cache = jnp.zeros(
        (num_pages, cfgs.page_size, cfgs.hbm_pack, 128), dtype=jnp.uint8
    )

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
          num_boundary, num_pages, cfgs.page_size, cfgs.record_slots
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

    init_rope_cache = _random_rope_cache(k6, cfgs, num_pages)
    slot_mapping = jnp.where(positions >= 0, positions * slots_per_token, -1)

    # A single chunked-prefill request, whose boundary tokens take consecutive
    # kv slots from a page boundary: tile i is exactly rope row i.
    distribution = jnp.array([0, 1, 1], dtype=jnp.int32)

    ref_rope_output = self._compare(
        cfgs=cfgs,
        populated_cache=populated_cache,
        init_rope_cache=init_rope_cache,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        distribution=distribution,
        rope_store_mode="prefill",
        interpret=interpret,
    )

    if cfgs.dims.has_rope_cache:
      # Records are dense, so they fill whole rows.
      self.assertEqual(
          _changed_rows(ref_rope_output, init_rope_cache),
          -(-num_boundary // cfgs.rope_pack),
      )

  def _run_ref(
      self,
      *,
      cfgs,
      cache,
      rope_cache,
      positions,
      slot_mapping,
      block_table,
      token_to_req_indices,
      kv_slot_mapping,
      rms_weight,
      cos_sin_cache,
      distribution,
      rope_store_mode,
  ):
    """Golden reference over one batch; `rope_store_mode=None` stores all."""
    dims = cfgs.dims
    return _REF_JIT(
        cache=cache,
        rope_cache=rope_cache,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        state_block_size=dims.block_size,
        head_dim=dims.head_dim,
        rope_head_dim=dims.rope_head_dim,
        compress_ratio=dims.compress_ratio,
        overlap=dims.overlap,
        rms_eps=dims.rms_eps,
        quant_block=dims.quant_block,
        is_quantized=dims.is_quantized,
        has_rope=dims.has_rope,
        has_rope_cache=dims.has_rope_cache,
        distribution=distribution,
        rope_store_mode=rope_store_mode,
    )

  def _run_kernel(
      self,
      *,
      cfgs,
      cache,
      rope_cache,
      positions,
      block_table,
      token_to_req_indices,
      kv_slot_mapping,
      rms_weight,
      cos_sin_cache,
      distribution,
      rope_store_mode,
      interpret=False,
  ):
    """One Pallas invocation, i.e. one class of token."""
    dims = cfgs.dims
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
        compress_ratio=dims.compress_ratio,
        overlap=dims.overlap,
        state_block_size=dims.block_size,
        quant_block=dims.quant_block,
        rms_eps=dims.rms_eps,
        rope_store_mode=rope_store_mode,
        interpret=interpret,
    )

  def _compare(
      self,
      *,
      cfgs,
      populated_cache,
      init_rope_cache,
      positions,
      slot_mapping,
      block_table,
      token_to_req_indices,
      kv_slot_mapping,
      rms_weight,
      cos_sin_cache,
      distribution,
      rope_store_mode,
      interpret=False,
  ):
    """Runs reference and Pallas on identical inputs and compares both caches."""
    ref_cache_output, ref_rope_output = self._run_ref(
        cfgs=cfgs,
        cache=populated_cache,
        rope_cache=init_rope_cache,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        distribution=distribution,
        rope_store_mode=rope_store_mode,
    )

    pallas_cache_output, pallas_rope_output = self._run_kernel(
        cfgs=cfgs,
        cache=jnp.copy(populated_cache),
        rope_cache=jnp.copy(init_rope_cache),
        positions=positions,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        distribution=distribution,
        rope_store_mode=rope_store_mode,
        interpret=interpret,
    )

    if cfgs.dims.has_rope_cache:
      self.assertArraysEqual(pallas_rope_output, ref_rope_output)

    self.assertArraysEqual(pallas_cache_output, ref_cache_output)
    return ref_rope_output

  def run_decode_correctness(
      self,
      num_reqs,
      history_len,
      head_dim=512,
      rope_head_dim=64,
      compress_ratio=4,
      state_block_size=16,
      quant_block=64,
      rms_eps=1e-6,
      interpret=False,
  ):
    """One decode token per request, each landing in its own packed rope row.

    Every request already holds ``history_len`` tokens of state and emits a
    single new token at position ``history_len - 1``, which is a compression
    boundary. Their rope records are scattered across distinct pages and
    distinct sub-slots so the read-modify-write has to preserve the rest of
    each row.
    """
    assert history_len % compress_ratio == 0
    assert history_len <= state_block_size, "one state page per request"

    cfgs = config.Configs.make(
        config.Mode.CSA,
        size_n=num_reqs,
        block_size=state_block_size,
        rms_eps=rms_eps,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        quant_block=quant_block,
    )
    state_width = cfgs.state_width
    state_dim = 2 * state_width
    hidden_size = state_dim
    slots_per_token = cfgs.slots_per_token
    page_size = cfgs.page_size

    # Pages [0, num_reqs) hold state, pages [num_reqs, 2 * num_reqs) hold the
    # compressed KV / rope records -- one page per request either way.
    num_pages = 2 * num_reqs
    block_table = jnp.arange(num_reqs, dtype=jnp.int32)[:, None]

    k1, k2, k3, k4, k5, k6 = jax.random.split(jax.random.key(0), 6)

    # Populate every request's full history in one scatter.
    run_1_tokens = num_reqs * history_len
    hidden_states = jax.random.normal(k1, (run_1_tokens, hidden_size))
    wkv_wgate = jax.random.normal(k2, (state_dim, hidden_size))
    ape = jax.random.normal(k3, (compress_ratio, state_width))

    run_1_positions = np.tile(np.arange(history_len), num_reqs).astype(np.int32)
    run_1_slot_mapping = (
        np.repeat(np.arange(num_reqs), history_len) * page_size
        + run_1_positions * slots_per_token
    ).astype(np.int32)

    populated_cache = jax.jit(
        proj_and_save_state_ref.ref_wkv_proj_and_save_state,
        static_argnums=(6, 7, 8, 9),
    )(
        hidden_states=hidden_states,
        wkv_wgate=wkv_wgate,
        ape=ape,
        positions=jnp.array(run_1_positions),
        slot_mapping=jnp.array(run_1_slot_mapping),
        cache=jnp.zeros(
            (num_pages, page_size, cfgs.hbm_pack, 128), dtype=jnp.uint8
        ),
        state_block_size=state_block_size,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        overlap=cfgs.dims.overlap,
    )

    # One boundary decode token per request, at the end of its history.
    decode_pos = history_len - 1
    positions = jnp.full((num_reqs,), decode_pos, dtype=jnp.int32)
    token_to_req_indices = jnp.arange(num_reqs, dtype=jnp.int32)

    # Spread the records over sub-slots 0..rope_pack-1 of different rows, so a
    # row-granular write that ignored the sub-slot index would be caught.
    kv_offsets = (np.arange(num_reqs) * 5) % page_size
    kv_slot_mapping = jnp.array(
        (num_reqs + np.arange(num_reqs)) * page_size + kv_offsets,
        dtype=jnp.int32,
    )
    slot_mapping = jnp.array(
        np.arange(num_reqs) * page_size + decode_pos * slots_per_token,
        dtype=jnp.int32,
    )

    rms_weight = jax.random.normal(k4, (head_dim,))
    cos_sin_cache = jax.random.normal(k5, (history_len, rope_head_dim))
    init_rope_cache = _random_rope_cache(k6, cfgs, num_pages)

    # All requests are decode-only.
    distribution = jnp.array([num_reqs] * 3, dtype=jnp.int32)

    ref_rope_output = self._compare(
        cfgs=cfgs,
        populated_cache=populated_cache,
        init_rope_cache=init_rope_cache,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        distribution=distribution,
        rope_store_mode="decode",
        interpret=interpret,
    )

    # Guard against a vacuous pass: the reference must really have touched
    # exactly one sub-slot per request and left the rest of each row alone.
    changed = _changed_sub_slots(ref_rope_output, init_rope_cache)
    self.assertEqual(int(jnp.sum(changed)), num_reqs)

  def run_mixed_correctness(
      self,
      num_decode_reqs,
      history_len=16,
      head_dim=512,
      rope_head_dim=64,
      compress_ratio=4,
      state_block_size=16,
      quant_block=64,
      rms_eps=1e-6,
      pad_to_tile=True,
      interpret=False,
  ):
    """Decode and chunked-prefill requests boundary-compacted into one batch.

    Mirrors what ``compressor_forward`` does: split the boundary tokens by
    class, hand each class the layout its rope store wants, and run the kernel
    once per class. The prefill request's records land in *consecutive* slots
    of a single page -- the dense case where up to rope_pack of them share a
    packed row -- so this pins down that the shared row survives all of them.
    """
    assert history_len % compress_ratio == 0
    assert history_len <= state_block_size, "one state page per request"

    # Requests [0, num_decode_reqs) decode, then one chunked-prefill request.
    prefill_req = num_decode_reqs
    num_reqs = num_decode_reqs + 1
    boundaries_per_prefill = history_len // compress_ratio
    num_boundary = num_decode_reqs + boundaries_per_prefill

    # The grid runs cdiv(num_valid, tile_n) whole tiles. `pad_to_tile=False`
    # hands the kernel a ragged batch to check it pads internally rather than
    # reading past the end of the arrays.
    tile_n = 4
    num_tokens = (
        -(-num_boundary // tile_n) * tile_n if pad_to_tile else num_boundary
    )
    num_pad = num_tokens - num_boundary

    cfgs = config.Configs.make(
        config.Mode.CSA,
        size_n=num_tokens,
        block_size=state_block_size,
        rms_eps=rms_eps,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        quant_block=quant_block,
    )
    state_dim = 2 * cfgs.state_width
    slots_per_token = cfgs.slots_per_token
    page_size = cfgs.page_size

    num_pages = 2 * num_reqs
    block_table = jnp.arange(num_reqs, dtype=jnp.int32)[:, None]

    k1, k2, k3, k4, k5, k6 = jax.random.split(jax.random.key(1), 6)

    run_1_tokens = num_reqs * history_len
    populated_cache = jax.jit(
        proj_and_save_state_ref.ref_wkv_proj_and_save_state,
        static_argnums=(6, 7, 8, 9),
    )(
        hidden_states=jax.random.normal(k1, (run_1_tokens, state_dim)),
        wkv_wgate=jax.random.normal(k2, (state_dim, state_dim)),
        ape=jax.random.normal(k3, (compress_ratio, cfgs.state_width)),
        positions=jnp.array(
            np.tile(np.arange(history_len), num_reqs).astype(np.int32)
        ),
        slot_mapping=jnp.array(
            (
                np.repeat(np.arange(num_reqs), history_len) * page_size
                + np.tile(np.arange(history_len), num_reqs) * slots_per_token
            ).astype(np.int32)
        ),
        cache=jnp.zeros(
            (num_pages, page_size, cfgs.hbm_pack, 128), dtype=jnp.uint8
        ),
        state_block_size=state_block_size,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        overlap=cfgs.dims.overlap,
    )

    # Boundary-compacted batch: decode tokens first (one per request, at the
    # end of its history), then the prefill request's boundary tokens.
    decode_pos = history_len - 1
    prefill_pos = (np.arange(boundaries_per_prefill) + 1) * compress_ratio - 1
    positions_np = np.concatenate([
        np.full(num_decode_reqs, decode_pos),
        prefill_pos,
        np.zeros(num_pad),
    ]).astype(np.int32)
    positions = jnp.array(positions_np)
    req_of_token = np.concatenate([
        np.arange(num_decode_reqs),
        np.full(boundaries_per_prefill, prefill_req),
        np.zeros(num_pad),
    ]).astype(np.int32)
    token_to_req_indices = jnp.array(req_of_token)

    # Decode records go to scattered sub-slots of their own page; the prefill
    # request's go to consecutive slots of its page, so they share rows.
    decode_slots = (num_reqs + np.arange(num_decode_reqs)) * page_size + (
        np.arange(num_decode_reqs) * 5
    ) % page_size
    prefill_slots = (num_reqs + prefill_req) * page_size + np.arange(
        boundaries_per_prefill
    )
    kv_slot_mapping = jnp.array(
        np.concatenate(
            [decode_slots, prefill_slots, np.full(num_pad, -1)]
        ).astype(np.int32)
    )
    slot_mapping_np = (
        req_of_token * page_size + positions_np * slots_per_token
    ).astype(np.int32)
    slot_mapping_np[num_boundary:] = -1
    slot_mapping = jnp.array(slot_mapping_np)

    distribution = jnp.array(
        [num_decode_reqs, num_reqs, num_reqs], dtype=jnp.int32
    )
    init_rope_cache = _random_rope_cache(k6, cfgs, num_pages)
    rms_weight = jax.random.normal(k4, (head_dim,))
    cos_sin_cache = jax.random.normal(k5, (history_len, rope_head_dim))

    # The reference sees the whole batch at once: what the pair of invocations
    # is expected to add up to.
    ref_cache_output, ref_rope_output = self._run_ref(
        cfgs=cfgs,
        cache=populated_cache,
        rope_cache=init_rope_cache,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        distribution=distribution,
        rope_store_mode=None,
    )

    is_boundary = kv_slot_mapping >= 0
    is_decode = token_to_req_indices < num_decode_reqs
    batches = (
        (
            "decode",
            compressor.pack_boundary_batch(
                positions,
                token_to_req_indices,
                kv_slot_mapping,
                keep=is_boundary & is_decode,
                size=num_tokens,
            ),
        ),
        (
            "prefill",
            compressor.row_tiled_boundary_batch(
                positions,
                token_to_req_indices,
                kv_slot_mapping,
                keep=is_boundary & ~is_decode,
                max_boundary=num_boundary,
                rope_pack=cfgs.rope_pack,
            ),
        ),
    )

    cache_out = jnp.copy(populated_cache)
    rope_out = jnp.copy(init_rope_cache)
    for rope_store_mode, (b_positions, b_req, b_kv_slot) in batches:
      cache_out, rope_out = self._run_kernel(
          cfgs=cfgs,
          cache=cache_out,
          rope_cache=rope_out,
          positions=b_positions,
          block_table=block_table,
          token_to_req_indices=b_req,
          kv_slot_mapping=b_kv_slot,
          rms_weight=rms_weight,
          cos_sin_cache=cos_sin_cache,
          distribution=distribution,
          rope_store_mode=rope_store_mode,
          interpret=interpret,
      )

    self.assertArraysEqual(rope_out, ref_rope_output)
    self.assertArraysEqual(cache_out, ref_cache_output)

    # One sub-slot per decode token, plus the prefill request's dense run --
    # which has to have landed in the same rows without clobbering each other.
    changed = _changed_sub_slots(ref_rope_output, init_rope_cache)
    self.assertEqual(int(jnp.sum(changed)), num_boundary)
    per_page = changed.reshape(num_pages, -1)
    self.assertEqual(
        int(jnp.sum(per_page[num_reqs + prefill_req])), boundaries_per_prefill
    )

  def run_prefill_rows_correctness(
      self,
      kv_offsets_per_req,
      history_len=16,
      head_dim=512,
      rope_head_dim=64,
      compress_ratio=4,
      state_block_size=16,
      quant_block=64,
      rms_eps=1e-6,
      interpret=False,
  ):
    """Chunked-prefill requests whose records straddle packed rope rows.

    ``kv_offsets_per_req`` gives each request's kv slot offsets within its own
    page, so a case can start mid-row (a chunk that resumed inside a row) and
    run over several rows. The kernel is handed the row-tiled layout, which
    leaves holes in the tiles those partial rows only half fill.
    """
    assert history_len % compress_ratio == 0
    assert history_len <= state_block_size, "one state page per request"

    num_reqs = len(kv_offsets_per_req)
    boundaries_per_req = history_len // compress_ratio
    assert all(len(o) == boundaries_per_req for o in kv_offsets_per_req)
    num_boundary = num_reqs * boundaries_per_req

    cfgs = config.Configs.make(
        config.Mode.CSA,
        size_n=num_boundary,
        block_size=state_block_size,
        rms_eps=rms_eps,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        quant_block=quant_block,
    )
    state_dim = 2 * cfgs.state_width
    slots_per_token = cfgs.slots_per_token
    page_size = cfgs.page_size

    num_pages = 2 * num_reqs
    block_table = jnp.arange(num_reqs, dtype=jnp.int32)[:, None]

    k1, k2, k3, k4, k5, k6 = jax.random.split(jax.random.key(2), 6)

    run_1_positions = np.tile(np.arange(history_len), num_reqs).astype(np.int32)
    populated_cache = jax.jit(
        proj_and_save_state_ref.ref_wkv_proj_and_save_state,
        static_argnums=(6, 7, 8, 9),
    )(
        hidden_states=jax.random.normal(
            k1, (num_reqs * history_len, state_dim)
        ),
        wkv_wgate=jax.random.normal(k2, (state_dim, state_dim)),
        ape=jax.random.normal(k3, (compress_ratio, cfgs.state_width)),
        positions=jnp.array(run_1_positions),
        slot_mapping=jnp.array(
            (
                np.repeat(np.arange(num_reqs), history_len) * page_size
                + run_1_positions * slots_per_token
            ).astype(np.int32)
        ),
        cache=jnp.zeros(
            (num_pages, page_size, cfgs.hbm_pack, 128), dtype=jnp.uint8
        ),
        state_block_size=state_block_size,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        overlap=cfgs.dims.overlap,
    )

    positions_np = np.tile(
        (np.arange(boundaries_per_req) + 1) * compress_ratio - 1, num_reqs
    ).astype(np.int32)
    req_of_token = np.repeat(np.arange(num_reqs), boundaries_per_req).astype(
        np.int32
    )
    kv_slot_mapping = jnp.array(
        (
            (num_reqs + req_of_token) * page_size
            + np.concatenate(kv_offsets_per_req)
        ).astype(np.int32)
    )
    positions = jnp.array(positions_np)
    token_to_req_indices = jnp.array(req_of_token)
    slot_mapping = jnp.array(
        (req_of_token * page_size + positions_np * slots_per_token).astype(
            np.int32
        )
    )

    # Every request is chunked-prefill.
    distribution = jnp.array([0, num_reqs, num_reqs], dtype=jnp.int32)
    init_rope_cache = _random_rope_cache(k6, cfgs, num_pages)
    rms_weight = jax.random.normal(k4, (head_dim,))
    cos_sin_cache = jax.random.normal(k5, (history_len, rope_head_dim))

    ref_cache_output, ref_rope_output = self._run_ref(
        cfgs=cfgs,
        cache=populated_cache,
        rope_cache=init_rope_cache,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        distribution=distribution,
        rope_store_mode="prefill",
    )

    b_positions, b_req, b_kv_slot = compressor.row_tiled_boundary_batch(
        positions,
        token_to_req_indices,
        kv_slot_mapping,
        keep=kv_slot_mapping >= 0,
        max_boundary=num_boundary,
        rope_pack=cfgs.rope_pack,
    )
    cache_out, rope_out = self._run_kernel(
        cfgs=cfgs,
        cache=jnp.copy(populated_cache),
        rope_cache=jnp.copy(init_rope_cache),
        positions=b_positions,
        block_table=block_table,
        token_to_req_indices=b_req,
        kv_slot_mapping=b_kv_slot,
        rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache,
        distribution=distribution,
        rope_store_mode="prefill",
        interpret=interpret,
    )

    self.assertArraysEqual(rope_out, ref_rope_output)
    self.assertArraysEqual(cache_out, ref_cache_output)

    # Every record landed, and only in the rows the offsets name -- a tile that
    # dropped its partner's sub-slot would show up as a missing change.
    self.assertEqual(
        int(jnp.sum(_changed_sub_slots(ref_rope_output, init_rope_cache))),
        num_boundary,
    )
    expected_rows = sum(
        len({o // cfgs.rope_pack for o in offsets})
        for offsets in kv_offsets_per_req
    )
    self.assertEqual(
        _changed_rows(ref_rope_output, init_rope_cache), expected_rows
    )

  def test_compressor_forward_ignores_padded_tail(self):
    """The batch past `query_start_loc[distribution[2]]` must not be stored.

    The caller's token buffers are persistent and only the scheduled prefix is
    rewritten each step, so the tail holds stale values -- and a stale position
    with ``(pos + 1) % compress_ratio == 0`` looks exactly like a boundary
    token. Running the same real batch under two different tails has to give
    the same caches.
    """
    head_dim, rope_head_dim, compress_ratio = 512, 64, 4
    state_block_size, quant_block, rms_eps = 16, 64, 1e-6
    num_tokens = 16  # 9 real (1 decode + 8 prefill), 7 padding.

    cfgs = config.Configs.make(
        config.Mode.CSA,
        size_n=num_tokens,
        block_size=state_block_size,
        rms_eps=rms_eps,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        quant_block=quant_block,
    )
    state_dim = 2 * cfgs.state_width
    num_pages = 4  # 0, 1 hold state; 2, 3 hold the compressed records.

    # Request 0 decodes one token, request 1 is a chunked prefill; slots 2 and 3
    # are padding. Their page-table rows point at *real* pages, as vLLM's do, so
    # a token that leaks out of the tail corrupts something observable.
    query_start_loc = jnp.array([0, 1, 9, 9, 9], dtype=jnp.int32)
    distribution = jnp.array([1, 2, 2], dtype=jnp.int32)
    block_table = jnp.array([[0, 0], [1, 1], [0, 0], [0, 0]], dtype=jnp.int32)
    kv_block_table = jnp.array([[2, 2], [3, 3], [2, 2], [2, 2]], dtype=jnp.int32)

    real_positions = np.concatenate([[7], np.arange(8)]).astype(np.int32)
    # Every stale position is a compression boundary, i.e. the worst case.
    stale_tail = np.array([3, 7, 3, 7, 3, 7, 3], dtype=np.int32)
    num_pad = num_tokens - real_positions.shape[0]
    assert stale_tail.shape[0] == num_pad

    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(jax.random.key(3), 7)
    shared = dict(
        wkv_wgate=jax.random.normal(k2, (state_dim, state_dim)),
        ape=jax.random.normal(k3, (compress_ratio, cfgs.state_width)),
        norm_weight=jax.random.normal(k4, (head_dim,)),
        cos_sin_cache=jax.random.normal(k5, (state_block_size, rope_head_dim)),
        block_table=block_table,
        query_start_loc=query_start_loc,
        kv_block_table=kv_block_table,
        distribution=distribution,
        state_block_size=state_block_size,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        overlap=cfgs.dims.overlap,
        rms_eps=rms_eps,
        quant_block=quant_block,
    )
    init_cache = jnp.zeros(
        (num_pages, cfgs.page_size, cfgs.hbm_pack, 128), dtype=jnp.uint8
    )
    init_rope_cache = _random_rope_cache(k6, cfgs, num_pages)
    real_hidden = jax.random.normal(k1, (num_tokens, state_dim))
    stale_hidden = real_hidden.at[-num_pad:].set(
        jax.random.normal(k7, (num_pad, state_dim))
    )

    def run(positions_tail, hidden_states):
      # Both caches are donated, so each run needs its own copy.
      cache, rope_cache = compressor.compressor_forward(
          hidden_states=hidden_states,
          positions=jnp.array(np.concatenate([real_positions, positions_tail])),
          cache=jnp.copy(init_cache),
          rope_cache=jnp.copy(init_rope_cache),
          **shared,
      )
      return jax.block_until_ready((cache, rope_cache))

    zeroed_cache, zeroed_rope = run(np.zeros(num_pad, np.int32), real_hidden)
    stale_cache, stale_rope = run(stale_tail, stale_hidden)

    self.assertArraysEqual(stale_cache, zeroed_cache)
    self.assertArraysEqual(stale_rope, zeroed_rope)
    # ... and the real tokens were stored, so the comparison isn't vacuous:
    # one record per boundary token (positions 7 / 3 / 7).
    self.assertEqual(
        int(jnp.sum(_changed_sub_slots(zeroed_rope, init_rope_cache))), 3
    )
    self.assertGreater(int(jnp.sum(zeroed_cache != 0)), 0)

  @parameterized.named_parameters(
      (
          "csa_prefill",
          dict(
              num_tokens=128,
              head_dim=512,
              rope_head_dim=64,
              compress_ratio=4,
              overlap=True,
              state_block_size=16,
          ),
      ),
  )
  def test_compress_store(self, cfg):
    self.run_compress_store_correctness(**cfg)

  @parameterized.named_parameters(
      ("csa_decode_8", dict(num_reqs=8, history_len=8)),
      ("csa_decode_16", dict(num_reqs=16, history_len=16)),
  )
  def test_compress_store_decode(self, cfg):
    self.run_decode_correctness(**cfg)

  @parameterized.named_parameters(
      ("csa_mixed_4", dict(num_decode_reqs=4)),
      ("csa_mixed_7", dict(num_decode_reqs=7)),
      # Ragged batch (11 tokens, tile_n = 4): the kernel must pad internally.
      ("csa_mixed_7_ragged", dict(num_decode_reqs=7, pad_to_tile=False)),
  )
  def test_compress_store_mixed(self, cfg):
    self.run_mixed_correctness(**cfg)

  @parameterized.named_parameters(
      # Row-aligned: one request, one full row.
      ("csa_rows_aligned", dict(kv_offsets_per_req=[[0, 1, 2, 3]])),
      # Resumed mid-row: spills into the next row, both partial.
      ("csa_rows_unaligned", dict(kv_offsets_per_req=[[2, 3, 4, 5]])),
      # A partial row per request, on separate pages, plus a full one.
      (
          "csa_rows_multi_req",
          dict(kv_offsets_per_req=[[2, 3, 4, 5], [0, 1, 2, 3], [7, 8, 9, 10]]),
      ),
      # One record per row: every tile is 3/4 holes.
      ("csa_rows_sparse", dict(kv_offsets_per_req=[[0, 4, 8, 12]])),
  )
  def test_compress_store_prefill_rows(self, cfg):
    self.run_prefill_rows_correctness(**cfg)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
