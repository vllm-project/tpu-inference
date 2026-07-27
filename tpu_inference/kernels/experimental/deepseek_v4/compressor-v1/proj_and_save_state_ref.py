import jax
import jax.numpy as jnp


def ref_wkv_proj_and_save_state(
    hidden_states: jax.Array,  # [num_tokens, hidden_size]
    wkv_wgate: jax.Array,  # [hidden_size, 2 * state_width]
    ape: jax.Array,  # [compress_ratio, state_width]
    positions: jax.Array,  # [num_tokens]
    slot_mapping: jax.Array,  # [num_tokens]
    cache: jax.Array,  # [num_pages, page_size, 4, 128] uint8
    state_block_size: int,
    head_dim: int,
    compress_ratio: int,
    overlap: bool,
) -> jax.Array:
  """Golden JAX reference for Kernel 1 (proj + save state)."""
  coff = 1 + int(overlap)
  state_width = coff * head_dim
  state_dim = 2 * state_width

  # 1. Project & APE
  kv_score = jnp.matmul(hidden_states, wkv_wgate)
  kv = kv_score[:, :state_width]
  score = kv_score[:, state_width:]

  ape_rows = jnp.mod(positions, compress_ratio)
  score_state = score + ape[ape_rows]
  packed = jnp.concatenate([kv, score_state], axis=-1)

  # 2. Geometry setup
  num_pages, page_size, d1, d2 = cache.shape
  assert d1 in (2, 4) and d2 == 128
  slots_per_token = page_size // state_block_size  # 16
  f32_per_slot = state_dim // slots_per_token  # 128
  bytes_per_slot = f32_per_slot * 4  # 512
  assert bytes_per_slot == d1 * d2  # 512

  # 3. Unpack inline
  cache_reshaped = cache.reshape(
      num_pages, state_block_size, slots_per_token, d1, d2
  )
  cache_t = cache_reshaped.transpose(0, 1, 2, 4, 3)
  cache_bitcast_shape = cache_t.reshape(
      num_pages, state_block_size, slots_per_token, (d2 * d1) // 4, 4
  )
  f32_flat = jax.lax.bitcast_convert_type(cache_bitcast_shape, jnp.float32)
  flat = f32_flat.reshape(num_pages * state_block_size, state_dim)

  # 4. Scatter
  num_tokens_flat = num_pages * state_block_size
  valid = slot_mapping >= 0
  slots = jnp.where(valid, slot_mapping // slots_per_token, num_tokens_flat)
  flat_padded = jnp.concatenate([flat, jnp.zeros((1, state_dim))], axis=0)
  flat_padded = flat_padded.at[slots].set(packed)
  flat = flat_padded[:-1]

  # 5. Pack inline back
  chunk = flat.reshape(
      num_pages, state_block_size, slots_per_token, f32_per_slot, 1
  )
  chunk_bytes = jax.lax.bitcast_convert_type(chunk, jnp.uint8)
  chunk_bytes_reshaped = chunk_bytes.reshape(
      num_pages, state_block_size, slots_per_token, d2, d1
  )
  chunk_bytes_t = chunk_bytes_reshaped.transpose(0, 1, 2, 4, 3)

  cache_view = cache.reshape(
      num_pages, state_block_size, slots_per_token, d1, d2
  )
  cache_view = cache_view.at[:].set(chunk_bytes_t)
  return cache_view.reshape(num_pages, page_size, d1, d2)