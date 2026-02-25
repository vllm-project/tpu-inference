"""Run this and get the HLO and LLO for the gather operation.
"""
from typing import Sequence
import os

import functools
import jax
import jaxtyping as jt
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc



@jax.jit
def gather_direct(indices, x):
  """Gather using direct indexing."""
  return x[indices]


@functools.partial(jax.jit, static_argnames=["window_bounds"])
def gather(
    a: jt.Real[jt.Array, "B D"],
    indices: jt.Int32[jt.Array, "I"],
    *,
    window_bounds: tuple[int, int] = (16, 128),
) -> jt.Real[jt.Array, "I D"]:
  """Gathers values from `a` given a set of indices.

  Args:
    a: The array to gather from.
    indices: The indices to gather.
    window_bounds: The window bounds in the `I` and `D` dimensions respectively.
      These should be set as big as `Spmem` allows. Some example good values are
      - Viperfish: (256, 128), Ghostlite: (128, 128).

  Returns:
    The gathered values.
  """
  _, inner_dim_size = a.shape
  output_bsz = indices.shape[0]

  # Pad to TensorCore tiling.
  a = _suffix_pad_to_multiple(a, (8, 128))
  indices = _suffix_pad_to_multiple(indices, (8,))

  # Pad to window bounds.
  a = _suffix_pad_to_multiple(a, window_bounds)
  indices = _suffix_pad_to_multiple(indices, window_bounds[:1])

  result_padded = _gather_2d_to_2d_impl(a, indices, window_bounds=window_bounds)
  return result_padded[:output_bsz, :inner_dim_size]


def _suffix_pad_to_multiple(
    a: jt.Real[jt.Array, "..."],
    divisors: Sequence[int],
) -> jt.Real[jt.Array, "..."]:
  padding = (
      lambda size, divisor: (size + divisor - 1) // divisor * divisor - size
  )
  pad_width = [(0, padding(s, m)) for s, m in zip(a.shape, divisors)]
  return jnp.pad(a, pad_width, mode="empty")



@functools.partial(jax.jit, static_argnames=["window_bounds"])
def _gather_2d_to_2d_impl(
    a: jt.Real[jt.Array, "B D"],
    indices: jt.Int32[jt.Array, "I"],
    *,
    window_bounds: tuple[int, int] = (16, 128),
) -> jt.Real[jt.Array, "I D"]:
  """Gathers values from `a` given a set of indices."""
  sc_chunk_size = 16
  # tpu_target = target.Target.from_device_kind(jax.devices()[0].device_kind)
  # if tpu_target.num_sc_cores == 0:
  #   raise ValueError("Need to have SparseCores to run this kernel.")
  # if tpu_target.generation not in (
  #     target.TpuGeneration.VIPERFISH,
  #     target.TpuGeneration.GHOSTLITE,
  #     target.TpuGeneration.GHOSTFISH,
  # ):
  #   # Other generations might work, but they haven't been tested.
  #   raise ValueError(f"Unsupported TPU generation: {tpu_target.generation}")
  print(f'{os.getenv("LIBTPU_INIT_ARGS")=}')
  if "--xla_tpu_use_tc_device_shape_on_sc=True" not in os.getenv("LIBTPU_INIT_ARGS", default=""):
    raise RuntimeError(
        "You must set the `--xla_tpu_use_tc_device_shape_on_sc=True` flag to"
        " use this kernel."
    )

  _, inner_dim_size = a.shape
  output_bsz = indices.shape[0]

  bitwidth = a.dtype.itemsize * 8
  packing = 32 // bitwidth

  # Window size in the `I` and `D` dimensions.
  ti, td = window_bounds
  assert inner_dim_size % td == 0

  @functools.partial(
      pl.pallas_call,
      out_shape=jax.ShapeDtypeStruct([output_bsz, inner_dim_size], a.dtype),
      grid=(output_bsz // ti, inner_dim_size // td),
      in_specs=[
          pl.BlockSpec((ti,), lambda i, j: i),
          # xw32: What do indexed_by and indexed_dim do?
          # Per the tutorial (lines 257-259), indexed_by and indexed_dim tell the SparseCore hardware to perform an indexed lookup (gather) as part of the DMA pipeline:
          #  - indexed_by=0 — Use input 0 (i.e., indices, the first argument to pallas_call) as the index source for this input's gather.
          #  - indexed_dim=0 — Apply the indexed lookup on dimension 0 (the row/batch dimension) of a.
          #  In other words, instead of reading contiguous blocks from a, the SparseCore will use the values in indices to fetch rows from a — effectively computing a[indices] in hardware. This is what makes it a gather operation.
          plsc.BlockSpec(
              (packing * ti, td),
              lambda i, j: (0, j),
              indexed_by=0,
              indexed_dim=0,
          ),
      ],
      out_specs=pl.BlockSpec((ti, td), lambda *ij: ij),
      # xw32: why is the second dimension arbitrary?
      # Looking at line 119:

      # dimension_semantics=["parallel", "arbitrary"],

      # The grid is (output_bsz // ti, inner_dim_size // td):
      # - Dim 0 (i): iterates over chunks of output rows
      # - Dim 1 (j): iterates over chunks of columns (the inner/hidden dimension D)

      # Dim 0 is parallel because each row chunk gathers from independent indices — they can
      #  be freely distributed across subcores.

      # Dim 1 is arbitrary because different j iterations process different column slices of
      #  the same gathered rows. Look at the BlockSpecs:

      # # indices: only depends on i, not j
      # pl.BlockSpec((ti,), lambda i, j: i),
      # # data: the gather (indexed_by=0) fetches rows based on indices,
      # #        then j selects which column slice
      # plsc.BlockSpec((packing * ti, td), lambda i, j: (0, j), indexed_by=0,
      # indexed_dim=0),

      # All j iterations for a given i share the same set of indices/gathered rows. If dim 1
      #  were parallel, those column iterations would be distributed across subcores, and
      # each subcore would need to redundantly perform the same gather for the same indices
      # — wasteful and potentially incorrect with the indexed_by pipeline. Marking it
      # arbitrary ensures column iterations happen sequentially within the same subcore,
      # reusing the already-gathered rows.
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=["parallel", "arbitrary"],
          kernel_type=pltpu.CoreType.SC_VECTOR_SUBCORE,
      ),
  )
  def kernel(indices_ref, x_ref, o_ref):
    # xw32: if bitwidth==16, then (1<<bitwidth)-1 will be (65535)10 in decimal, 0xFFFF, or (1111111111111111)2 (16 1s).
    mask = jax.lax.broadcast(
        (1 << bitwidth) - 1 if bitwidth < 32 else -1, [sc_chunk_size]
    )

    @pl.loop(0, ti, step=sc_chunk_size)
    def _(ti_offset):
      indices_chunk = indices_ref[pl.ds(ti_offset, sc_chunk_size)]

      @pl.loop(0, td, step=sc_chunk_size)
      def _(td_offset):
        # 1. Start by loading `packing * tpu_target.sc_chunk_size` worth of
        # rows from `x_ref` into registers.
        # xw32: Inner loop over column chunks. For each of 16 output rows, load a row from x_ref. 
        # Because of packing, x_ref has packing times more rows than actual output rows — each 32-bit word contains packing sub-word values interleaved. Each loaded vector is packing * sc_chunk_size elements wide.
        x_chunks = []
        for ti_offset_addend in range(sc_chunk_size):
          row = ti_offset * packing + ti_offset_addend * packing
          col = td_offset
          x_chunks.append(
              x_ref[row, pl.ds(col, packing * sc_chunk_size)]
          )

        # 2. We need to bitcast the loaded chunk to i32 to be able to do bit
        # twiddling on it.
        # xw32q: what does the plsc.bitcast do?
        x_chunks = [plsc.bitcast(x_chunk, jnp.int32) for x_chunk in x_chunks]

        # 3. We need to mask out all but the `packing` most significant bits
        # of each chunk. We have `packing` number of rows packed in 32 bits.
        # E.g. for bf16, there are 2 rows packed in 32 bits. The indirect
        # stream has copied both rows, although we only want to index one of
        # them. The way we do this is by bitshifting the row we want to the
        # most significant bits, and then masking out the rest. Back to the
        # bf16 example, row 0 will occupy bits 16-31, and row 1 will occupy
        # bits 0-15. By the end of this operation, one of these rows will
        # occupy bits 16-31 and the rest will be 0s.
        # Calculate `indices_chunk % packing` so we know how much to bitshift
        # each `mask`.
        # xw32: For bf16, each 32-bit word holds 2 bf16 values. index % packing determines which bf16 within the 32-bit word we actually want. For example, if index % 2 == 0, we want bits 0-15, so the mask stays at 0x0000FFFF. If index % 2 == 1, we want bits 16-31, so the mask becomes 0xFFFF0000. The AND zeroes out the unwanted half.
        # xw32q: For bf16 example, why does row 0 occuppy bits 16-31 instead of 0-15?
        shift_amounts = (indices_chunk % packing) * bitwidth
        shift_amounts = [
            shift_amounts[i] for i in range(sc_chunk_size)
        ]
        chunk_masks = [
            jax.lax.shift_left(mask, shift_amount)
            for shift_amount in shift_amounts
        ]
        x_chunks = [
            jax.lax.bitwise_and(x_chunk, chunk_mask)
            for x_chunk, chunk_mask in zip(x_chunks, chunk_masks)
        ]

        # 4. We need to shift values within a pack in a block diagonal so that
        # we can OR all the values later when we pack them back together.
        # Let's again consider the bf16 example. From the previous step, we'll
        # have two rows that we want to pack in bits 16-31 of two 32 bit
        # words. If we bitshift the second word by 16 bits to the right, the
        # data will now occupy bits 0-15.
        # xw32: Two shifts:
        # 1. Shift right to move the extracted value down to bit position 0.
        # 2. Shift left by bitwidth * (i % packing) to place each chunk's value into its correct slot for the output packing.
        # For bf16 with packing=2: chunk 0's value goes to bits 0-15, chunk 1's value goes to bits 16-31. This creates a "block diagonal" layout so they can be OR'd together without overlap.
        x_chunks = [
            jax.lax.shift_right_logical(x_chunk, shift_amount)
            for x_chunk, shift_amount in zip(x_chunks, shift_amounts)
        ]
        x_chunks = [
            jax.lax.shift_left(x_chunk, bitwidth * ((i % packing)))
            for i, x_chunk in enumerate(x_chunks)
        ]

        # 5. We need to pack the chunks back together. At the end we'll have
        # `tpu_target.sc_chunk_size / packing` o_chunks.
        # xw32: OR together every packing consecutive chunks. For bf16: x_chunks[0] | x_chunks[1] produces one 32-bit word with two correctly-gathered bf16 values. This reduces 16 chunks down to 8 output chunks.
        o_chunks = []
        for i in range(0, len(x_chunks), packing):
          o_chunk = x_chunks[i]
          for j in range(1, packing):
            o_chunk = jax.lax.bitwise_or(o_chunk, x_chunks[i + j])
          o_chunks.append(o_chunk)

        # 6. We need to bitcast the packed chunks back to the original type
        # and finally store them in the output reference.
        # xw32q: why do we need to do another bitcast?
        o_chunks = [plsc.bitcast(o_chunk, o_ref.dtype) for o_chunk in o_chunks]
        for ti_offset_addend, o_chunk in enumerate(o_chunks):
          row = ti_offset + ti_offset_addend * packing
          col = td_offset
          o_ref[row, pl.ds(col, packing * sc_chunk_size)] = o_chunk

  return kernel(indices, a)

# SparseCore Pallas kernel doesn't work in OSS as of 20260217.
# The same code works in G3.
def test_sparse_core_pallas_gather():
  num_local_tokens = 8192
  hidden_size = 6144
  topk = 8

  k1, k2 = jax.random.split(jax.random.key(0))
  indices = jax.random.randint(k1, (num_local_tokens * topk,), 0, num_local_tokens)
  hidden_states = jax.random.uniform(k2, (num_local_tokens, hidden_size))

  out = gather(hidden_states, indices)
  expected = gather_direct(indices, hidden_states)
  jnp.allclose(out, expected)

def profile():
  num_tokens = 8192
  hidden_size = 6144
  num_indices = 65536

  key = jax.random.PRNGKey(0)
  k1, k2 = jax.random.split(key)
  hidden_states = jax.random.normal(k1, (num_tokens, hidden_size), dtype=jnp.bfloat16)
  indices = jax.random.randint(k2, (num_indices,), 0, num_tokens)

  gather_direct(indices, hidden_states).block_until_ready()

  # profile_path = "/tmp/sort2_tokens_profile"
  # jax.profiler.start_trace(profile_path)
  for _ in range(3):
    gather_direct(indices, hidden_states).block_until_ready()
  # jax.profiler.stop_trace()

if __name__ == "__main__":
  test_sparse_core_pallas_gather()
