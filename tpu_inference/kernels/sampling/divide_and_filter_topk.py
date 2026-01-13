"""Divide-and-filter top-k algorithm implementation."""

import functools
import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec as P

from tpu_inference.kernels.sampling.bitonic_topk import bitonic_topk_arrays as _bitonic_topk_arrays

from tpu_inference.kernels.sampling.topk_convergence_theory import (
  calculate_depth_thresholds,
)
from tpu_inference.kernels.sampling.utils import (
  unrolled_fori_loop,
  NUM_LANES,
  NUM_SUBLANES,
  pad,
  get_dtype_info,
  iota_tile,
  to_32bit_dtype,
  ceil_multiple,
  pack_bf16_u16_to_i32,
  unpack_bf16_u16_from_i32,
)


def _extract_remainder_slice(ref, slice_size):
  """Extract and pad the remainder slice when input doesn't divide evenly.

  Args:
    ref: Input reference array.
    slice_size: Size of each slice.

  Returns:
    Padded remainder slice or None if no remainder exists.
  """
  full_size = ref.shape[1]
  num_full_slices = full_size // slice_size
  remainder = full_size % slice_size
  if remainder > 0:
    # Load the final boundary slice
    remainder_vals = ref[
      ..., pl.dslice(num_full_slices * slice_size, remainder)
    ]
    # Pad with min value
    return pad(remainder_vals, (1, slice_size), val="min")
  return None


def nan_to_min(x):
  """Replace NaNs in an array with the dtype's minimum value.

  Args:
    x: Input array.

  Returns:
    Array with NaNs replaced by minimum value.
  """
  return jnp.where(jnp.isnan(x), get_dtype_info(x).min, x)


def to_comparison_dtype(x):
  """Convert array to 32-bit dtype suitable for comparisons.

  Args:
    x: Input array.

  Returns:
    Array converted to 32-bit dtype.
  """
  return x.astype(to_32bit_dtype(x.dtype))


def bitonic_topk_arrays(operands, k, val_dtype=None, max_index=None):
  """Top-k of arrays, packing bf16 and indices into single 32-bit dtype if possible.

  Args:
    operands: List of arrays to find top-k from.
    k: Number of top elements to find.
    val_dtype: Optional dtype for values (enables packing optimization).
    max_index: Optional maximum index value (enables packing optimization).

  Returns:
    List of top-k arrays.
  """
  if val_dtype is None or max_index is None:
    return _bitonic_topk_arrays(operands, k=k)
  assert len(operands) == 2 and type(operands) == list
  dtypes = [x.dtype for x in operands]
  pack = val_dtype == jnp.bfloat16 and max_index <= 2**16
  if pack:
    operands = [pack_bf16_u16_to_i32(*operands, stable=False)]
  operands = _bitonic_topk_arrays(operands, k=k)
  if pack:
    assert len(operands) == 1
    operands = list(unpack_bf16_u16_from_i32(operands[0], stable=False))
    assert len(operands) == 2
  # convert back dtypes (incase theyve changed)
  return [x.astype(dtype) for x, dtype in zip(operands, dtypes, strict=True)]


def binned_topk(
  logits,
  k: int,
  bins_topk_vals,
  bins_topk_idxs,
  completed_k: int = 0,
  num_bins: int = NUM_LANES,
  unroll: int = 32,
):
  """
  Compute binned top-k using a sinking sort approach.

  Processes the vocabulary in num_bins-sized chunks, maintaining the top-k elements
  across all processed bins using a sinking sort algorithm. Values "sink" through
  the maintained top-k list if they are smaller than existing elements.

  Args:
      logits: Input logits of shape [num_tokens, vocab_size].
      k: Number of top elements to find per bin.
      bins_topk_vals: List of k arrays, each of shape [num_tokens, num_bins],
          containing current top-k values per bin.
      bins_topk_idxs: List of k arrays, each of shape [num_tokens, num_bins],
          containing current top-k indices per bin.
      completed_k: Number of top-k positions already finalized (default: 0).
      num_bins: Number of bins/lanes to process simultaneously (default: 128).
      unroll: Loop unroll factor for the vocabulary scan (default: 32).

  Returns:
      Tuple of (bins_topk_vals, bins_topk_idxs) with updated top-k values and indices.
  """
  num_tokens, vocab_size = logits.shape

  def update_bins_topk(
    bubble_vals, bubble_idxs, bins_topk_vals, bins_topk_idxs
  ):
    """
    Update bins topk with bubble vals/idxs using sinking sort.

    Compares new values against existing top-k, swapping when new values are larger.
    Already-completed positions are invalidated to prevent re-selection.
    """
    bubble_vals = to_comparison_dtype(bubble_vals)
    # Sinking sort: compare and swap
    for i in range(completed_k):
      # Invalidate already-found elements
      # We use the idxs list to check identity
      bubble_vals = jnp.where(
        bubble_idxs == bins_topk_idxs[i],
        get_dtype_info(bubble_vals).min,
        bubble_vals,
      )
    for i in range(completed_k, k):
      # Exchange with stored top-k
      # Only perform the swap if the value is larger
      mask = bubble_vals > bins_topk_vals[i]
      bins_topk_vals[i], bubble_vals = (
        jnp.where(m, bubble_vals, bins_topk_vals[i]) for m in (mask, ~mask)
      )
      bins_topk_idxs[i], bubble_idxs = (
        jnp.where(m, bubble_idxs, bins_topk_idxs[i]) for m in (mask, ~mask)
      )
    return (bins_topk_vals, bins_topk_idxs)

  def compute_idxs(i):
    """Compute global vocabulary indices for bin slice i."""
    shape = (num_tokens, num_bins)
    return jnp.full(shape, i * num_bins, jnp.int32) + jax.lax.broadcasted_iota(
      jnp.int32, shape, 1
    )

  def loop_body(i, bins_topk_outs):
    vals = logits[..., pl.dslice(num_bins * i, num_bins)]
    idxs = compute_idxs(i)
    return update_bins_topk(vals, idxs, *bins_topk_outs)

  num_full_slices = vocab_size // num_bins
  bins_topk_outs = unrolled_fori_loop(
    num_full_slices,
    loop_body,
    (bins_topk_vals, bins_topk_idxs),
    unroll=unroll,
  )

  # Handle remaining elements if vocab_size doesn't divide num_bins
  rem_vals = _extract_remainder_slice(logits, num_bins)
  if rem_vals is not None:
    # Create idxs for the final segment
    rem_idxs = compute_idxs(num_full_slices)
    # Update bins topk with the overspill
    bins_topk_outs = update_bins_topk(rem_vals, rem_idxs, *bins_topk_outs)
  return bins_topk_outs


def _merge_unconverged_bins_topk(
  logits_ref,
  bins_topm_vals_ref,
  bins_topm_idxs_ref,
  *,
  num_bins: int,
  m: int,
  max_k: int,
):
  """Compute top-k from most active bins and merge with unconverged bins."""

  # Derive block_token from logits_ref shape
  block_token = logits_ref.shape[0]

  # The ⌈k/m⌉'th largest value across the m'th largest value in each partition is a lower bound for the top-k threshold, as in ⌈k/m⌉ bins there are at least m values larger or equal to it (⌈k/m⌉ is the ceiling division of k by m). All partitions where the m'th largest value is less than the threshold will not contribute any further values to top-k so only ⌈k/m⌉-1 partitions could possibly contribute to top-k beyond their top-m.
  # Derive num_packed_bins from max_k and m
  num_packed_bins = pl.cdiv(max_k, m) - 1
  if num_packed_bins > NUM_LANES or num_packed_bins > num_bins:
    raise NotImplementedError
  bin_vals = bins_topm_vals_ref[:, pl.dslice((m - 1) * num_bins, num_bins)]
  # Use bitonic_topk_arrays descending to get bin indices ordered by contribution count
  bin_indices = jax.lax.broadcasted_iota(jnp.int32, (block_token, num_bins), 1)
  # Sort descending by num_gt_k to get top NUM_LANES bin indices
  _, sorted_bin_indices = bitonic_topk_arrays(
    [bin_vals, bin_indices],
    k=num_packed_bins,
  )
  sorted_bin_indices = pad(sorted_bin_indices, (NUM_SUBLANES, NUM_LANES))

  # Repeat first num_packed_bins values across NUM_LANES positions to create packing permutation
  packing_perm = jnp.take_along_axis(
    sorted_bin_indices, iota_tile(1) % num_packed_bins, axis=1
  )

  # produce the (block_token, num_bins) mask
  # index[t, b] = b (the bin index in the second dimension)
  index = jax.lax.broadcasted_iota(jnp.int32, (block_token, num_bins), 1)
  indicator = jnp.zeros((block_token, num_bins), dtype=jnp.bool_)
  for i in range(num_packed_bins):
    # Mark positions where bin index matches the i-th active bin
    indicator |= index == packing_perm[:, i : i + 1]

  # invalidate active bins to avoid double inclusion
  bins_topm_vals_ref[:, : m * num_bins] = jnp.concat(
    [
      jnp.where(
        indicator,
        get_dtype_info(bins_topm_vals_ref).min,
        bins_topm_vals_ref[:, i * num_bins : (i + 1) * num_bins],
      )
      for i in range(m)
    ],
    axis=1,
  )

  # Loop over blocks and pack data from active bins
  vocab_size = logits_ref.shape[1]
  packed_vals = [
    jnp.full(
      (block_token, NUM_LANES),
      get_dtype_info(logits_ref).min,
      dtype=logits_ref.dtype,
    )
    for _ in range(
      pl.cdiv(vocab_size, (NUM_LANES // num_packed_bins) * num_bins)
    )
  ]

  assert num_bins % NUM_LANES == 0
  for offset in range(0, num_bins, NUM_LANES):
    local_perm = (packing_perm - offset) % NUM_LANES
    in_range_mask = (packing_perm >= offset) & (
      packing_perm < (offset + NUM_LANES)
    )

    # Extract values from all full bins at this offset
    num_full_slices = vocab_size // num_bins
    vals = [
      logits_ref[:, pl.dslice(i * num_bins + offset, NUM_LANES)]
      for i in range(num_full_slices)
    ]
    # deal with remainder if exists
    if num_full_slices * num_bins + offset < vocab_size:
      # if start is not out of array, take the full final num_bins slice then pull out this offset portion
      vals.append(
        _extract_remainder_slice(logits_ref, slice_size=num_bins)[
          :, offset : offset + NUM_LANES
        ]
      )
    # apply permutation
    vals = [
      jnp.take_along_axis(
        tile.astype(to_32bit_dtype(tile.dtype)), local_perm, axis=1
      )
      for tile in vals
    ]
    # Pack into positions based on active bin index
    index = iota_tile(1)
    for i in range(NUM_LANES // num_packed_bins):
      pack_mask = (
        (index >= i * num_packed_bins)
        & (index < (i + 1) * num_packed_bins)
        & in_range_mask
      )
      # Pack every num_packed_bins-th chunk starting from i
      for j, v in enumerate(vals[i :: NUM_LANES // num_packed_bins]):
        packed_vals[j] = jnp.where(pack_mask, v, packed_vals[j])

  packed_vals = jnp.concat(packed_vals, axis=1)
  n = packed_vals.shape[1]

  local_idxs = jax.lax.broadcasted_iota(jnp.int32, packed_vals.shape, 1)
  padding = NUM_LANES % num_packed_bins
  packed_idxs = (
    # num_packed_bins may not evenly divide NUM_LANES
    # so we have to offset the junk remainder data
    (local_idxs - (local_idxs // NUM_LANES) * padding) // num_packed_bins
  ) * num_bins + pltpu.repeat(packing_perm, n // NUM_LANES, axis=1)

  # we calculate the top k vals from the packed bins and a piece of bins_topm_(val/idx)s we overwrite
  # Build input arrays by concatenating packed vals and the top NUM_LANES values
  # avoid any nans ever entering into bitonic_topk. bins_topm_vals will have no nans, as it uses > comparison for filling and nan > x resolves to False in all cases
  packed_vals = nan_to_min(packed_vals)
  val_input = jnp.concat([packed_vals, bins_topm_vals_ref[:, :max_k]], axis=1)
  idx_input = jnp.concat([packed_idxs, bins_topm_idxs_ref[:, :max_k]], axis=1)
  (bins_topm_vals_ref[:, :max_k], bins_topm_idxs_ref[:, :max_k]) = (
    bitonic_topk_arrays(
      [val_input, idx_input],
      k=max_k,
      val_dtype=logits_ref.dtype,
      max_index=logits_ref.shape[1],
    )
  )


def dynamic_topk_refs(
  logits_ref,
  k_smem_ref,
  k_vmem_ref,
  topk_vals_ref,
  topk_idxs_ref,
  valid_ref,
  max_depth_ref,
  cutoff_vals_ref,
  # scratch
  bins_topm_vals_ref,
  bins_topm_idxs_ref,
  termination_flag_ref,
  *,
  max_k: int,
  num_bins: int,
  bins_topm_unroll: int,
  bins_topm_schedule: tuple[int, ...],
  guarantee_convergence: bool,
  replace_val: float | int | None,
):
  """
  Pallas kernel for computing binned top-k supersets until global top-k is guaranteed.

  Incrementally computes top-m supersets (m increasing per schedule) until the top-k
  is provably contained within the top-(m-1) bins. Supports dynamic k per token while
  using static max_k for compilation and scheduling.

  The termination criterion checks if the top-(m-1) bins collectively contain at least
  k values larger than the largest m-th largest value across all bins.
  """
  # Initialize buffers
  block_token = logits_ref.shape[0]
  shape = (block_token, bins_topm_vals_ref.shape[1])
  block_topk = bins_topm_vals_ref.shape[0]
  assert block_topk % block_token == 0, (
    "block_topk must be a multiple of block_token"
  )

  pid = pl.program_id(0)

  token_slice = pl.dslice(
    pl.multiple_of((pid * block_token) % block_topk, block_token), block_token
  )

  bins_topm_vals_ref[token_slice] = jnp.full(
    shape, get_dtype_info(logits_ref).min, dtype=bins_topm_vals_ref.dtype
  )

  for i in range(block_token):
    max_depth_ref[pid * block_token + i] = max_k
  termination_flag_ref[0] = 0

  # Incremental binned top-k computation
  for completed_m, m in zip(bins_topm_schedule, bins_topm_schedule[1:]):

    @pl.when(termination_flag_ref[0] < block_token)
    def _():
      # Compute binned top-m
      bins_topm_vals, bins_topm_idxs = binned_topk(
        logits_ref,
        k=m,
        bins_topk_vals=[
          bins_topm_vals_ref[
            token_slice, pl.dslice(i * num_bins, num_bins)
          ].astype(to_32bit_dtype(logits_ref.dtype))
          for i in range(m)
        ],
        bins_topk_idxs=[
          bins_topm_idxs_ref[token_slice, pl.dslice(i * num_bins, num_bins)]
          for i in range(m)
        ],
        num_bins=num_bins,
        completed_k=completed_m,
        unroll=bins_topm_unroll,
      )

      # Store results
      for i in range(completed_m, m):
        bins_topm_vals_ref[token_slice, pl.dslice(i * num_bins, num_bins)] = (
          bins_topm_vals[i].astype(bins_topm_vals_ref.dtype)
        )
        bins_topm_idxs_ref[token_slice, pl.dslice(i * num_bins, num_bins)] = (
          bins_topm_idxs[i].astype(bins_topm_idxs_ref.dtype)
        )
      if m >= max_k:
        # it's converged so no need for check
        return
      if m == 1:
        # Check not possible
        return
      # Termination criterion:
      # If top-(m-1) bins contain >= k vals larger than
      # the largest m-th largest value, then top-k is guaranteed to be in bins
      # top-(m-1) collated
      pivot = bins_topm_vals[m - 1].max(-1, keepdims=True)
      num_larger = (
        sum((v >= pivot) for v in bins_topm_vals[: m - 1])
        .astype(jnp.float32)
        .sum(-1)
      )

      termination_flag_ref[0] = 0
      for i in range(block_token):
        token_idx = pid * block_token + i
        # Dynamic check against k
        contains_topk = num_larger[i] >= k_smem_ref[token_idx]
        termination_flag_ref[0] += contains_topk

        # Record depth when criterion was met
        current_max = max_depth_ref[token_idx]
        max_depth_ref[token_idx] = jnp.where(
          contains_topk & (current_max == max_k), m - 1, current_max
        )
        # Record largest m-th largest value
        # Useful for bounds checking if running sharded topk
        cutoff_vals_ref[token_idx] = pivot.squeeze(1)[i]

  # Bin packing optimization for non-convergence cases
  m_final = bins_topm_schedule[-1]
  if guarantee_convergence and (m_final < max_k):

    @pl.when(termination_flag_ref[0] < block_token)
    def _():
      # This optimization applies when guarantee_convergence is enabled but
      # we haven't fully converged (m_final != max_k) and termination criterion not met.
      # Packs the most active bins to help converge.
      _merge_unconverged_bins_topk(
        logits_ref,
        bins_topm_vals_ref.at[token_slice],
        bins_topm_idxs_ref.at[token_slice],
        num_bins=num_bins,
        m=m_final,
        max_k=max_k,
      )

  # early on bins_topm_schedule are convergence checks so we go to bins-top-(m-1). For final bins-top-(m_max) for convergence guaranteed we only need to consider top-(m_max-1), if not must cover bins-top-(m_max)
  global_topk_schedule = [max(x - 1, 0) for x in bins_topm_schedule[:-1]] + [
    bins_topm_schedule[-1] - (1 if guarantee_convergence else 0)
  ]
  global_topk_schedule = tuple(sorted(set(bins_topm_schedule)))

  # Final top-k extraction (done on aggregated blocks as implementation gets more efficient)
  grid_i = pl.program_id(0)
  grid_size = pl.num_programs(0)
  topk_unroll = block_topk // block_token
  completed_i = grid_i + 1

  @pl.when(
    # final iter, or buffer filled
    (completed_i == grid_size) | ((completed_i % topk_unroll) == 0)
  )
  def _():
    # Find maximum depth across all tokens
    global_max_depth = jnp.array(0, dtype=jnp.int32)
    token_start = (grid_i // topk_unroll) * block_topk
    for i in range(block_topk):
      token_idx = i + token_start
      global_max_depth = jnp.maximum(
        global_max_depth,
        # the * deals with OOB access values
        max_depth_ref[token_idx] * (token_idx < max_depth_ref.shape[0]),
      )

    valid_ref[0] = (
      (
        (global_max_depth < bins_topm_schedule[-1])
        | (bins_topm_schedule[-1] >= max_k)
      )
      & valid_ref[0].astype(bool)
    ).astype(jnp.int32)

    # Use appropriate sorting depth based on global_max_depth
    for depth_lower, depth_upper in zip(
      global_topk_schedule, global_topk_schedule[1:]
    ):

      @pl.when(
        ((global_max_depth > depth_lower) & (global_max_depth <= depth_upper))
        | (
          # Sort to give approx topk if not fully converged
          (depth_upper == global_topk_schedule[-1])
          & (global_max_depth > depth_upper)
        )
      )
      def _():
        # Sort the binned superset
        vals_input = bins_topm_vals_ref[:, : depth_upper * num_bins]
        idxs_input = bins_topm_idxs_ref[:, : depth_upper * num_bins]
        vals, idxs = bitonic_topk_arrays(
          [vals_input, idxs_input],
          k=max_k,
          val_dtype=logits_ref.dtype,
          max_index=logits_ref.shape[1],
        )
        topk_vals_ref[...], topk_idxs_ref[...] = (
          vals.astype(topk_vals_ref.dtype),
          idxs,
        )
        if replace_val is not None:
          idx = jax.lax.broadcasted_iota(jnp.int32, vals.shape, 1)
          topk_vals_ref[...] = jnp.where(
            idx < k_vmem_ref[...], topk_vals_ref[...], replace_val
          )


@functools.partial(
  jit,
  static_argnames=(
    "max_k",
    "block_token",
    "block_topk",
    "num_bins",
    "bins_topm_unroll",
    "bins_topm_schedule",
    "guarantee_convergence",
    "replace_val",
    "interpret",
  ),
)
def _top_bounded_k(
  logits,
  k,
  max_k: int,
  block_token: int | None = None,
  block_topk: int | None = None,
  num_bins: int | None = None,
  bins_topm_unroll: int = 64,
  bins_topm_schedule: tuple[int, ...] | None = None,
  guarantee_convergence: bool = True,
  replace_val: float | int | None = None,
  interpret: bool = False,
):
  """
  High-level interface for adaptive binned top-k computation on TPU.

  This is a vmap'd implementation to jax.lax.top_k for traced k, where k is bounded
  by max_k. It's faster to compute top-(bounded-k) than computing top-(max-k) due to
  early convergence checks using the possibly lower k values.

  Supports dynamic k per token (each token can have a different k value) while
  maintaining efficient TPU execution through static compilation based on max_k.
  Automatically computes optimal search schedules if not provided.

  Behavior differences to jax.lax.top_k:
      - Handling of NaNs is different to jax.lax.top_k, here NaNs are never part of top-k.
      - Any output where k values are larger than or equal to the k'th largest value is considered valid, unlike jax.lax.top_k which in case of ties considers lower-index elements larger.
  If you wish exactly the same behavior, use `tallax.tax.bitonic_top_k(x, k=k, is_stable=True)`

  Sharding is supported in either/both dimensions if `guarantee_convergence=True`

  Args:
      logits: Input logits of shape [num_tokens, vocab_size].
      k: Per-token k values. Can be scalar (broadcast to all tokens) or array
          of shape [num_tokens].
      max_k: Static maximum k across all tokens. Used for buffer sizing and
          compilation. Must be >= all values in k.
      block_token: Number of tokens processed per program block.
      block_topk: Number of tokens processed to filtered subsets before subset top-k. Must be a multiple of block_token.
      num_bins: Number of bins for parallel binned operations.
      bins_topm_unroll: Loop unroll factor for binned top-m inner loop.
      bins_topm_schedule: Increasing sequence of m values for incremental top-m search.
          If None, automatically computed based on convergence probability thresholds.
      guarantee_convergence: If True, adds max_k to schedule to ensure full convergence
          and enables bin packing optimization for rare non-convergence cases (default: False).
      interpret: If True, run in CPU interpret mode instead of TPU compilation (default: False).

  Returns:
      When guarantee_convergence=False:
          Tuple of (topk_vals, topk_idxs, valid, depths, cutoff_vals):
          - topk_vals: Top-k values of shape [num_tokens, max_k].
          - topk_idxs: Top-k indices of shape [num_tokens, max_k].
          - valid: Boolean indicating if algorithm fully converged.
          - depths: Per-token convergence depth of shape [num_tokens].
          - cutoff_vals: Per-token pivot values of shape [num_tokens].
      When guarantee_convergence=True:
          Tuple of (topk_vals, topk_idxs):
          - topk_vals: Top-k values of shape [num_tokens, max_k].
          - topk_idxs: Top-k indices of shape [num_tokens, max_k].
  """

  # pad in first dimension for block spec
  # padding will count as immediately converged as it pads with minimum finite value (not -inf where comparison is difficult to define when checking for convergence)
  num_tokens, vocab_size = logits.shape
  if block_token is None:
    block_token = NUM_SUBLANES
  num_tokens_padded = ceil_multiple(num_tokens, block_token)

  if (
    num_bins is None
    and bins_topm_schedule is None
    and max_k <= 8
    and (max_k * vocab_size) < (8 * 2**17)
  ):
    # heuristic for very small k, up until large vocab size just do a bins top-k then aggregate. Provides constant runtime.
    num_bins = NUM_LANES
    bins_topm_schedule = (max_k,)

  if num_bins is None:
    # larger the input, the more costly bins top-m computing is, so relative to it bins top-m gets cheaper and higher num_bins requires less m for convergence (and filters more strongly for unconverged bins)
    num_bins = next(
      (
        n
        for limit, n in [(2**13, 128), (2**15, 256), (2**18, 512)]
        # we shift the limit slightly by 128 to be inclusive of LLM
        # sampling cases where additional vocab is added to power of 2 size vocabs
        # eg Gemma3 has 2**18 + 64 vocab size in practice
        if vocab_size <= (limit + 256)
      ),
      1024,
    )

  if block_topk is None:
    block_topk = ceil_multiple(min(num_tokens_padded, NUM_LANES), block_token)
  if block_topk % block_token != 0:
    raise ValueError("block_topk must be divisible by block_token")
  topk_unroll = block_topk // block_token

  if jnp.ndim(k) == 0:
    k = jnp.broadcast_to(k, (num_tokens,))
  k = pad(k, (num_tokens_padded,), val=0)

  # Auto-compute schedules if not provided
  if bins_topm_schedule is None:
    thresholds = calculate_depth_thresholds(
      max_k,
      num_bins,
      block_token,
      target_yields=(
        # if input is small bins top-m is cheap and we try avoid the lane permutes required for dealing with unconverged bins
        (0.8, 0.999) if vocab_size <= 2**13 else (0.8,)
      ),
    )
    bins_topm_schedule = tuple(sorted({min(t + 1, max_k) for t in thresholds}))
    print(
      f"Auto-computed bins top-m schedule for max_k={max_k}, num_bins={num_bins}: {bins_topm_schedule}"
    )
  bins_topm_schedule = tuple(sorted(set(bins_topm_schedule)))
  bins_topm_schedule = (0,) + bins_topm_schedule

  # binned topk / sort pad len
  max_m = bins_topm_schedule[-1]
  buffer_size = max_m * num_bins

  output_shapes = (
    jax.ShapeDtypeStruct((num_tokens, max_k), logits.dtype),
    jax.ShapeDtypeStruct((num_tokens, max_k), jnp.int32),
    jax.ShapeDtypeStruct((1,), jnp.int32),
    jax.ShapeDtypeStruct((num_tokens_padded,), jnp.int32),
    jax.ShapeDtypeStruct((num_tokens_padded,), to_32bit_dtype(logits.dtype)),
  )

  output_specs = (
    pl.BlockSpec((block_topk, max_k), lambda i: (i // topk_unroll, 0)),
    pl.BlockSpec((block_topk, max_k), lambda i: (i // topk_unroll, 0)),
    pl.BlockSpec(memory_space=pltpu.SMEM),
    pl.BlockSpec(memory_space=pltpu.SMEM),
    pl.BlockSpec(memory_space=pltpu.SMEM),
  )

  # Add scratch shapes

  scratch_shapes = [
    pltpu.VMEM((block_topk, buffer_size), to_32bit_dtype(logits.dtype)),
    pltpu.VMEM((block_topk, buffer_size), jnp.int32),
    pltpu.SMEM((1,), jnp.int32),
  ]

  outputs = pl.pallas_call(
    functools.partial(
      dynamic_topk_refs,
      max_k=max_k,
      num_bins=num_bins,
      bins_topm_unroll=bins_topm_unroll,
      bins_topm_schedule=bins_topm_schedule,
      guarantee_convergence=guarantee_convergence,
      replace_val=replace_val,
    ),
    in_specs=(
      pl.BlockSpec((block_token, vocab_size), lambda i: (i, 0)),
      # for TPU Pallas lowering reasons it's convenient to have both SMEM and VMEM k
      pl.BlockSpec(memory_space=pltpu.SMEM),
      pl.BlockSpec((block_topk, 1), lambda i: (i // topk_unroll, 0)),
    ),
    out_shape=output_shapes,
    scratch_shapes=tuple(scratch_shapes),
    grid=(pl.cdiv(num_tokens, block_token),),
    out_specs=output_specs,
    compiler_params=pltpu.CompilerParams(vmem_limit_bytes=int(0.9 * 2**27)),
    interpret=interpret,
  )(logits, k, k[:, None])
  topk_vals, topk_idxs, valid, depths, cutoff_vals = outputs

  topk_vals, topk_idxs = (
    x[:num_tokens, :max_k] for x in (topk_vals, topk_idxs)
  )
  valid = valid.squeeze().astype(bool)

  if guarantee_convergence:
    return topk_vals, topk_idxs
  return (
    topk_vals,
    topk_idxs,
    valid,
    depths[:num_tokens],
    cutoff_vals[:num_tokens],
  )


@functools.partial(
  jit,
  static_argnames=(
    "max_k",
    "block_token",
    "block_topk",
    "num_bins",
    "bins_topm_unroll",
    "bins_topm_schedule",
    "guarantee_convergence",
    "replace_val",
    "interpret",
  ),
)
@functools.wraps(_top_bounded_k)
def top_bounded_k(
  logits,
  k,
  max_k: int,
  block_token: int | None = None,
  block_topk: int | None = None,
  num_bins: int | None = None,
  bins_topm_unroll: int = 64,
  bins_topm_schedule: tuple[int, ...] | None = None,
  guarantee_convergence: bool = False,
  replace_val: float | int | None = None,
  interpret: bool = False,
):
  def _closed_topk(logits: jax.Array, k: jax.Array):
    return _top_bounded_k(
      logits,
      k=k,
      max_k=max_k,
      block_token=block_token,
      block_topk=block_topk,
      num_bins=num_bins,
      bins_topm_unroll=bins_topm_unroll,
      bins_topm_schedule=bins_topm_schedule,
      guarantee_convergence=guarantee_convergence,
      replace_val=replace_val,
      interpret=interpret,
    )

  @custom_partitioning
  def _sharded_topk(logits, k):
    return _closed_topk(logits, k)

  def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
    logits_spec = arg_shapes[0].sharding.spec
    return (NamedSharding(mesh, P(logits_spec[0], None)),) * 2

  def partition(mesh, arg_shapes, out_shapes):
    if not guarantee_convergence:
      raise NotImplementedError
    arg_shardings, out_shardings = jax.tree.map(
      lambda s: s.sharding, (arg_shapes, out_shapes)
    )
    axis_name = arg_shardings[0].spec[1]

    def shmap_fn(logits, k):
      topk_logits, topk_idxs = _closed_topk(logits, k)
      if axis_name is None:
        return topk_logits, topk_idxs
      # convert idxs to global frame
      i = jax.lax.axis_index(axis_name)
      topk_idxs += i * logits.shape[1]
      # all-gather and top-k
      operands = [
        jax.lax.collapse(
          jax.lax.all_gather(x, axis_name, axis=1),
        1)
        for x in (topk_logits, topk_idxs)
      ]
      topk_logits, topk_idxs = _bitonic_topk_arrays(operands, k=max_k)
      topk_logits = jnp.where(
        jax.lax.broadcasted_iota(jnp.int32, topk_logits.shape, 1) < k[:, None],
        topk_logits,
        replace_val,
      )
      return topk_logits, topk_idxs

    return mesh, shmap_fn, out_shardings, arg_shardings

  _sharded_topk.def_partition(
    infer_sharding_from_operands=infer_sharding_from_operands,
    partition=partition,
    sharding_rule="b v, b -> b k, b k",
  )
  return _sharded_topk(logits, k)


@functools.partial(
  jit,
  static_argnames=(
    "k",
    "block_token",
    "block_topk",
    "num_bins",
    "bins_topm_unroll",
    "bins_topm_schedule",
    "interpret",
  ),
)
def topk(
  logits,
  k: int,
  block_token: int | None = None,
  block_topk: int | None = None,
  num_bins: int | None = None,
  bins_topm_unroll: int = 64,
  bins_topm_schedule: tuple[int, ...] | None = None,
  interpret: bool = False,
):
  """
  Compute top-k element.

  Behavior differences to jax.lax.top_k:
      - Handling of NaNs is different to jax.lax.top_k, here NaNs are never part of top-k.
      - Any output where k values are larger than or equal to the k'th largest value is considered valid, unlike jax.lax.top_k which in case of ties in value considers lower-index elements larger.
  If you wish exactly the same behavior, use `tallax.tax.bitonic_top_k(x, k=k, is_stable=True)` instead.

  Sharding is supported in either/both dimensions

  Args:
      logits: Input logits of shape [num_tokens, vocab_size].
      k: Number of top elements to find (uniform across all tokens).
      block_token: Number of tokens processed per program block.
      block_topk: Number of tokens processed to filtered subsets before subset top-k. Must be a multiple of block_token.
      num_bins: Number of bins for parallel operations if not set a heuristic is used.
      bins_topm_unroll: Loop unroll factor for inner loop.
      bins_topm_schedule: Optional custom search schedule. If None, automatically
          computed.
      interpret: If True, run in CPU interpret mode (default: False).

  Returns:
      Tuple of (topk_vals, topk_idxs):
          - topk_vals: Top-k values of shape [num_tokens, k].
          - topk_idxs: Top-k indices of shape [num_tokens, k].
  """
  return top_bounded_k(
    logits,
    k=k,
    max_k=k,
    block_token=block_token,
    block_topk=block_topk,
    num_bins=num_bins,
    bins_topm_unroll=bins_topm_unroll,
    bins_topm_schedule=bins_topm_schedule,
    guarantee_convergence=True,
    interpret=interpret,
  )
