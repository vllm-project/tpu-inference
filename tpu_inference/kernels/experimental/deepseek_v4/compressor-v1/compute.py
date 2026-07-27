import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def interleaved_rope_vector(x, cos_val_32, sin_val_32):
  """Applies interleaved Rotary Position Embedding (RoPE) to a vector."""
  # x: (tile_n, 1, 128)
  # cos_val_32: (tile_n, 1, 32)
  # sin_val_32: (tile_n, 1, 32)
  tile_n = x.shape[0]

  # We work with 2D tensors for gather to keep dimensions simple.
  x_2d = jnp.squeeze(x, axis=1)  # (tile_n, 128)

  # swap adjacent pairs: [1, 0, 3, 2, ...]
  iota = jnp.arange(128)
  swap_indices = jnp.bitwise_xor(iota, 1)  # (128,)
  swap_coords = jnp.broadcast_to(swap_indices, (tile_n, 128))[
      :, :, None
  ]  # (tile_n, 128, 1)

  gather_dn = jax.lax.GatherDimensionNumbers(
      offset_dims=(),
      collapsed_slice_dims=(1,),
      start_index_map=(1,),
      operand_batching_dims=(0,),
      start_indices_batching_dims=(0,),
  )

  x_swapped_2d = jax.lax.gather(
      x_2d,
      swap_coords,
      dimension_numbers=gather_dn,
      slice_sizes=(1, 1),
      unique_indices=True,
      mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
  )

  # cos_val_32/sin_val_32 are (tile_n, 1, 32). Squeeze to (tile_n, 32)
  cos_32 = jnp.squeeze(cos_val_32, axis=1).astype(x.dtype)
  sin_32 = jnp.squeeze(sin_val_32, axis=1).astype(x.dtype)

  ones_32 = jnp.ones((tile_n, 32), dtype=x.dtype)
  zeros_32 = jnp.zeros((tile_n, 32), dtype=x.dtype)

  # pad to pairs: (tile_n, 64)
  cos_pairs = jnp.concatenate([ones_32, cos_32], axis=-1)
  sin_pairs = jnp.concatenate([zeros_32, sin_32], axis=-1)

  # duplicate indices: [0, 0, 1, 1, 2, 2, ..., 63, 63]
  dup_indices = iota // 2
  dup_coords = jnp.broadcast_to(dup_indices, (tile_n, 128))[:, :, None]

  cos_dup_2d = jax.lax.gather(
      cos_pairs,
      dup_coords,
      dimension_numbers=gather_dn,
      slice_sizes=(1, 1),
      unique_indices=False,
      mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
  )

  sin_dup_2d = jax.lax.gather(
      sin_pairs,
      dup_coords,
      dimension_numbers=gather_dn,
      slice_sizes=(1, 1),
      unique_indices=False,
      mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
  )

  # alternate sin: [-1, 1, -1, 1, ...]
  alt_mask = ((iota % 2) * 2 - 1).astype(x.dtype)[None, :]  # (1, 128)
  sin_alt_2d = sin_dup_2d * alt_mask

  out_2d = x_2d * cos_dup_2d + x_swapped_2d * sin_alt_2d

  return out_2d[:, None, :]


def quantize_fp8_tiled(x, block_size):
  # x: (tile_n, S, 128) f32
  fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
  tile_n, slots, width = x.shape  # (tile_n, S, 128) f32
  num_blocks = width // block_size

  qs = []
  scales = []
  for b in range(num_blocks):
    start = b * block_size
    end = start + block_size
    x_block = x[:, :, start:end]  # (tile_n, S, block_size) f32

    amax = jnp.clip(
        jnp.max(jnp.abs(x_block), axis=-1, keepdims=True), 1e-4, None
    )  # (tile_n, S, 1) f32

    log2_val = jnp.log2(amax / fp8_max)  # f32
    scale = jnp.exp2(jnp.ceil(log2_val))  # (tile_n, S, 1) f32

    q_block = (x_block * (1.0 / scale)).astype(
        jnp.float8_e4m3fn
    )  # (tile_n, S, block_size) fp8

    qs.append(q_block)
    scales.append(scale)

  q = jnp.concatenate(qs, axis=-1)  # (tile_n, S, 128) fp8
  scale = jnp.concatenate(scales, axis=-1)  # (tile_n, S, num_blocks) f32

  # Bitcast workaround directly on f32 to extract exponent
  # f32 exponent is at bits 23-30. Shift right by 23 to align it.
  scale_u32 = pltpu.bitcast(scale, jnp.uint32)
  scale_exp = scale_u32 >> 23
  scale_u8 = scale_exp.astype(jnp.uint8)
  scale_f8 = pltpu.bitcast(scale_u8, jnp.float8_e8m0fnu)

  return q, scale_f8


def pack_nope_tiled(q, scale, nope_dim, block_size, nope_width_bytes=512):
  # q: (tile_n, S, 128) fp8
  # scale: (tile_n, S, num_blocks) e8m0 (uint8 bitcasted)
  tile_n, _, _ = q.shape

  # Bitcast to uint8
  q_bytes = pltpu.bitcast(q, jnp.uint8)
  scale_bytes = pltpu.bitcast(scale, jnp.uint8)

  # Flat representations
  q_flat = q_bytes.reshape(tile_n, -1)  # (tile_n, S * 128)
  scale_flat = scale_bytes.reshape(tile_n, -1)  # (tile_n, S * num_blocks)

  # Select NOPE parts
  if nope_dim < q_flat.shape[1]:
    q_nope = q_flat[:, :nope_dim]
  else:
    q_nope = q_flat

  nope_blocks = (nope_dim + block_size - 1) // block_size
  if nope_blocks < scale_flat.shape[1]:
    scale_nope = scale_flat[:, :nope_blocks]
  else:
    scale_nope = scale_flat

  # Pad with zeros
  pad_size = nope_width_bytes - (nope_dim + nope_blocks)
  zeros = jnp.zeros((tile_n, pad_size), dtype=jnp.uint8)

  nope_record_padded = jnp.concatenate(
      [q_nope, scale_nope, zeros], axis=1
  )  # (tile_n, 512)
  return nope_record_padded.reshape(tile_n, nope_width_bytes // 128, 128)


def pack_rope_tiled(rope_slot_ropped, rope_head_dim_actual, rope_width=128):
  # rope_slot_ropped: (tile_n, 1, 128)
  # TODO: make this a config value rather than inferred
  tile_n = rope_slot_ropped.shape[0]
  start = 128 - rope_head_dim_actual
  rope_val = rope_slot_ropped[:, :, start:]  # (tile_n, 1, rope_head_dim)

  rope_bf16 = rope_val.astype(jnp.bfloat16)
  rope_width_bf16 = rope_width // 2

  if rope_head_dim_actual < rope_width_bf16:
    rope_padded_bf16 = jnp.pad(
        rope_bf16,
        ((0, 0), (0, 0), (0, rope_width_bf16 - rope_head_dim_actual)),
    )
  else:
    rope_padded_bf16 = rope_bf16

  rope_f32 = rope_padded_bf16.astype(jnp.float32)  # (tile_n, 1, 64)
  rope_u32 = pltpu.bitcast(rope_f32, jnp.uint32)  # (tile_n, 1, 64)

  rope_u32_2d = jnp.squeeze(rope_u32, axis=1)  # (tile_n, 64)

  iota = jnp.arange(128)
  dup_indices = iota // 2  # (128,)
  dup_coords = jnp.broadcast_to(dup_indices, (tile_n, 128))[:, :, None]

  gather_dn = jax.lax.GatherDimensionNumbers(
      offset_dims=(),
      collapsed_slice_dims=(1,),
      start_index_map=(1,),
      operand_batching_dims=(0,),
      start_indices_batching_dims=(0,),
  )

  dup_u32_2d = jax.lax.gather(
      rope_u32_2d,
      dup_coords,
      dimension_numbers=gather_dn,
      slice_sizes=(1, 1),
      unique_indices=False,
      mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
  )

  shifts = jnp.where(iota % 2 == 0, 16, 24)  # (128,)
  shifted = dup_u32_2d >> shifts[None, :]
  rope_uint8_2d = (shifted & 0xFF).astype(jnp.uint8)  # (tile_n, 128)

  return rope_uint8_2d[:, None, :]


def gather_from_page_buffer(
    page_buffer,
    positions_ref,
    kv_window_u8,
    score_window_u8,
    *,
    global_idx: int,
    num_tokens: int,
    window: int,
    block_size: int,
    pages_to_buffer_per_token: int,
    slots_per_part: int,
    slots_per_token: int,
    overlap: bool,
):
  """Extracts kv_window, score_window from page buffer."""
  tile_n = page_buffer.shape[0]
  slots_per_part_hbm = page_buffer.shape[3]
  slots_per_part_head = kv_window_u8.shape[2]
  slots_per_part_gather = slots_per_part
  total_blocks = slots_per_part_head * 4

  for n in range(tile_n):
    safe_idx = jnp.minimum(global_idx + n, num_tokens - 1)
    pos = positions_ref[safe_idx]

    block_idx_curr = pos // block_size
    pos_start = pos - window + 1

    @pl.loop(0, window)
    def body_w(w):
      pos_w = pos_start + w

      block_idx_w = pos_w // block_size
      p = block_idx_w - block_idx_curr + pages_to_buffer_per_token - 1

      if overlap:
        is_prev = w < (window // 2)
        kv_slot_start = jax.lax.select(is_prev, 0, slots_per_part_gather)
        score_slot_start = jax.lax.select(
            is_prev, 2 * slots_per_part_gather, 3 * slots_per_part_gather
        )
      else:
        kv_slot_start = 0
        score_slot_start = slots_per_part_gather

      offset_in_block = pos_w % block_size

      for k in range(total_blocks):
        s_idx = k // slots_per_part_hbm
        h_idx = k % slots_per_part_hbm
        d_idx = k // 4
        v_idx = k % 4

        kv_src_slot = offset_in_block * slots_per_token + kv_slot_start + s_idx
        score_src_slot = (
            offset_in_block * slots_per_token + score_slot_start + s_idx
        )

        val_kv = page_buffer[n, p, kv_src_slot, h_idx, :]
        val_score = page_buffer[n, p, score_src_slot, h_idx, :]

        kv_window_u8[n, w, d_idx, v_idx, :] = val_kv
        score_window_u8[n, w, d_idx, v_idx, :] = val_score