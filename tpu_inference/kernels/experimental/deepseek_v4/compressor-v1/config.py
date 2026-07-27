"""CSA Configs.

Modes:
  HCA:
    - State Bytes: 1024 (dim) x 4 (bytes per fp32) = 4096 bytes
    - Head Dim:    512 (bf16, no quantization)
    - Cache:       [num_pages, _, 4, 128] uint8 (where _ = state_block_size * 8
    or kv_block_size * 2)

  CSA:
    - State Bytes: 2048 (dim) x 4 (bytes per fp32) = 8192 bytes
    - Head Dim:    448 fp8 + 7 scale = 462 bytes -> 512 bytes
    - Cache:       [num_pages, _, 4, 128] uint8 (where _ = state_block_size * 16
    or kv_block_size)

  CSA-Indexer:
    - State Bytes: 1024 (dim) x 4 (bytes per fp32) = 4096 bytes
    - Head Dim:    128 fp8 + 1 scale = 129 bytes -> 256 bytes
    - Cache:       [num_pages, _, 2, 128] uint8 (where _ = state_block_size * 16
    or kv_block_size)
"""

import dataclasses
import enum

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp


# --- physical layout constants ------------------------------------------------
LANE = 128  # bytes per sub-slot / TPU lane width
SLOT_PACK = 4  # sub-slots packed into one physical HBM slot row
N_FIELDS = 2  # values stored per token: kv + score
FP32_BYTES = 4


class Mode(enum.Enum):
  HCA = "hca"
  CSA = "csa"
  CSA_INDEXER = "csa_indexer"


_MODE_DEFAULTS = {
    Mode.HCA: dict(
        head_dim=512,
        rope_head_dim=64,
        compress_ratio=128,
        quant_block=0,
        overlap=False,
        has_rope_cache=False,
    ),
    Mode.CSA: dict(
        head_dim=512,
        rope_head_dim=64,
        compress_ratio=4,
        quant_block=64,
        overlap=True,
        has_rope_cache=True,
    ),
    Mode.CSA_INDEXER: dict(
        head_dim=128,
        rope_head_dim=64,
        compress_ratio=4,
        quant_block=128,
        overlap=True,
        has_rope_cache=False,
    ),
}


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TileSizes:
  """Tile sizes for the kernel."""
  tile_n: int


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Dimensions:
  """True inputs only. Anything derivable is a property below."""

  mode: Mode = dataclasses.field(metadata=dict(static=True))
  size_n: int  # num_tokens
  head_dim: int
  rope_head_dim: int
  compress_ratio: int
  block_size: int  # state_block_size
  quant_block: int
  overlap: bool
  has_rope_cache: bool
  rms_eps: float = 1e-6

  cos_sin_dtype: jax.typing.DTypeLike = jnp.float32
  rope_width: int = 128

  @property
  def is_quantized(self) -> bool:
    return self.quant_block > 0

  @property
  def has_rope(self) -> bool:
    return self.rope_head_dim > 0

  @property
  def nope_dtype(self) -> jax.typing.DTypeLike:
    return jnp.uint8 if self.is_quantized else jnp.bfloat16

  @property
  def rope_dtype(self) -> jax.typing.DTypeLike:
    return self.nope_dtype


ROPE_STORE_MODES = ("decode", "prefill")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Configs:
  """Configuration for the kernel."""
  tile_sizes: TileSizes
  dims: Dimensions
  # How the rope records of this invocation reach the packed rope cache:
  #   "decode":  sparse tokens, each owning a different row -> one
  #              read-modify-write per token.
  #   "prefill": dense tokens, several per row -> one grid step per row, the
  #              tile's records merged into a single staged row.
  # See kernel.compress_norm_rope_store for the layout each mode expects.
  rope_store_mode: str = dataclasses.field(
      default="decode", metadata=dict(static=True)
  )

  # --- factory ---------------------------------------------------------------
  @classmethod
  def make(
      cls,
      mode: Mode,
      *,
      size_n,
      block_size,
      rms_eps=1e-6,
      tile_n=4,
      rope_store_mode="decode",
      **overrides,
  ) -> "Configs":
    """Build a config for `mode`; `overrides` replace per-mode defaults."""
    actual_overrides = {**_MODE_DEFAULTS[mode], **overrides}
    if mode == Mode.HCA:
      actual_overrides["quant_block"] = 0

    if rope_store_mode not in ROPE_STORE_MODES:
      raise ValueError(
          f"rope_store_mode must be one of {ROPE_STORE_MODES}; got"
          f" {rope_store_mode!r}"
      )

    dims = Dimensions(
        mode=mode,
        size_n=size_n,
        block_size=block_size,
        rms_eps=rms_eps,
        **actual_overrides,
    )
    return cls(
        tile_sizes=TileSizes(tile_n=tile_n),
        dims=dims,
        rope_store_mode=rope_store_mode,
    )

  # --- compute tiling (logical, in LANE-wide tiles) --------------------------
  @property
  def overlap_factor(self) -> int:
    """State copies kept per step: 1, or 2 when windows overlap (prev+curr)."""
    return 1 + int(self.dims.overlap)

  @property
  def state_width(self) -> int:
    """Width of one stored field (head_dim, doubled when overlapping)."""
    return self.overlap_factor * self.dims.head_dim

  @property
  def head_tiles(self) -> int:
    """head_dim split into LANE-wide compute tiles (old: slots_per_part)."""
    return self.dims.head_dim // LANE

  @property
  def window(self) -> int:
    """Timesteps compressed together (x overlap_factor when overlapping)."""
    return self.dims.compress_ratio * self.overlap_factor

  # --- rope sub-layout -------------------------------------------------------
  @property
  def nope_dim(self) -> int:
    """Width of the non-rope (nope) part of a head."""
    return self.dims.head_dim - self.dims.rope_head_dim

  @property
  def nope_store_dim(self) -> int:
    """Dimension of the nope storage (contains rope if no separate rope cache)."""
    return (
        self.dims.head_dim - self.dims.rope_head_dim
        if self.dims.has_rope_cache
        else self.dims.head_dim
    )

  @property
  def half_rope(self) -> int:
    """cos/sin split point (half the rope dim)."""
    return self.dims.rope_head_dim // 2

  @property
  def rope_slot(self) -> int:
    """Head-tile holding the rope channels (the last one)."""
    return self.head_tiles - 1

  # --- output record size ----------------------------------------------------
  @property
  def record_bytes(self) -> int:
    """Bytes in one packed output record (old: total_bytes_out)."""
    if not self.dims.is_quantized:
      return self.dims.head_dim * 2  # bf16
    # fp8 payload + scale + padding, capped to a fixed record width.
    return 256 if self.dims.head_dim == LANE else 512

  @property
  def record_subslots(self) -> int:
    """record_bytes counted in 128-byte sub-slots."""
    return self.record_bytes // LANE

  @property
  def record_slots(self) -> int:
    """Packed slot rows per output record."""
    return pl.cdiv(self.record_subslots, SLOT_PACK)

  # --- HBM storage packing ---------------------------------------------------
  @property
  def hbm_pack(self) -> int:
    """Sub-slots physically packed per slot row, <= SLOT_PACK."""
    return min(self.record_subslots, SLOT_PACK)

  def _slots(self, width_bytes: int) -> int:
    """Convert a byte width to a count of packed HBM slots. The one formula."""
    return width_bytes // (self.hbm_pack * LANE)

  @property
  def slots_per_token(self) -> int:
    """HBM slots occupied by one token's full state (kv + score)."""
    return self._slots(N_FIELDS * self.state_width * FP32_BYTES)

  @property
  def gather_slots(self) -> int:
    """HBM slots spanned by one field during gather."""
    return self._slots(self.dims.head_dim * FP32_BYTES)

  @property
  def page_size(self) -> int:
    """Slots per HBM page."""
    return self.dims.block_size * self.slots_per_token

  # --- rope cache packing ----------------------------------------------------
  @property
  def rope_pack(self) -> int:
    """128B rope records physically packed into one rope-cache row."""
    return SLOT_PACK

  @property
  def rope_page_size(self) -> int:
    """Rows per rope-cache page.

    The rope cache holds one `LANE`-byte record per slot, but packs `rope_pack`
    of them per row, so a page of `page_size` slots is stored as
    [rope_page_size, rope_pack, LANE].

    Returns:
      The number of packed rows in one rope-cache page.
    """
    return self.page_size // self.rope_pack

  @property
  def pages_to_buffer_per_token(self) -> int:
    """Pages that must be resident to cover one token's window (+1 guard)."""
    return pl.cdiv(self.window, self.dims.block_size) + 1

  # --- shapes (single source of truth for every reshape / BlockSpec) ---------
  @property
  def _tile_n(self) -> int:
    return self.tile_sizes.tile_n

  def window_shape(self) -> tuple[int, ...]:
    """f32 window scratch: (fields, tile, window, head_tiles, lane)."""
    return (N_FIELDS, self._tile_n, self.window, self.head_tiles, LANE)

  def window_bytes_shape(self) -> tuple[int, ...]:
    """uint8 view of the window scratch; each f32 lane -> FP32_BYTES rows.

    Note: that trailing FP32_BYTES (4) is bytes-per-f32, NOT the HBM SLOT_PACK
    (also 4) -- they're numerically equal but mean different things.

    Returns:
      The shape of the window bytes scratch.
    """
    return (
        N_FIELDS,
        self._tile_n,
        self.window,
        self.head_tiles,
        FP32_BYTES,
        LANE,
    )

  def output_shape(self) -> tuple[int, ...]:
    """Packed output tile / VMEM block: (tile, record_slots, hbm_pack, lane)."""
    return (self._tile_n, self.record_slots, self.hbm_pack, LANE)

  def page_buffer_shape(self) -> tuple[int, ...]:
    """Page-buffer VMEM block: (tile, pages, page_size, hbm_pack, lane)."""
    return (
        self._tile_n,
        self.pages_to_buffer_per_token,
        self.page_size,
        self.hbm_pack,
        LANE,
    )

  def rope_output_shape(self) -> tuple[int, ...]:
    """RoPE read-modify-write VMEM block.

    One token's record is a single LANE-byte sub-slot, but the smallest thing
    the packed cache can exchange with HBM is a whole (rope_pack, LANE) row, so
    the staging buffer holds full rows: one per token in "decode" mode, one for
    the whole tile in "prefill" mode (where the tile *is* a row).

    Returns:
      The shape of the RoPE VMEM block.
    """
    rows = self._tile_n if self.rope_store_mode == "decode" else 1
    return (rows, self.rope_pack, LANE)

  def cos_sin_shape(self) -> tuple[int, ...]:
    """cos/sin VMEM block: (tile, rope_head_dim)."""
    return (self._tile_n, self.dims.rope_head_dim)