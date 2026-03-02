import functools
from typing import Sequence

from absl import flags
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

@functools.partial(jax.jit, static_argnames=["window_bounds"])
@jt.jaxtyped(typechecker=typeguard.typechecked)
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
@jt.jaxtyped(typechecker=typeguard.typechecked)
def _gather_2d_to_2d_impl(
    a: jt.Real[jt.Array, "B D"],
    indices: jt.Int32[jt.Array, "I"],
    *,
    window_bounds: tuple[int, int] = (16, 128),
) -> jt.Real[jt.Array, "I D"]:
  """Gathers values from `a` given a set of indices."""
  tpu_target = target.Target.from_device_kind(jax.devices()[0].device_kind)
  if tpu_target.num_sc_cores == 0:
    raise ValueError("Need to have SparseCores to run this kernel.")
  if tpu_target.generation not in (
      target.TpuGeneration.VIPERFISH,
      target.TpuGeneration.GHOSTLITE,
      target.TpuGeneration.GHOSTFISH,
  ):
    # Other generations might work, but they haven't been tested.
    raise ValueError(f"Unsupported TPU generation: {tpu_target.generation}")
  if not getattr(flags.FLAGS, "xla_tpu_use_tc_device_shape_on_sc", False):
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
          plsc.BlockSpec(
              (packing * ti, td),
              lambda i, j: (0, j),
              indexed_by=0,
              indexed_dim=0,
          ),
      ],
      out_specs=pl.BlockSpec((ti, td), lambda *ij: ij),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=["parallel", "arbitrary"],
          kernel_type=pltpu.CoreType.SC_VECTOR_SUBCORE,
      ),
  )
  def kernel(indices_ref, x_ref, o_ref):
    mask = jax.lax.broadcast(
        (1 << bitwidth) - 1 if bitwidth < 32 else -1, [tpu_target.sc_chunk_size]
    )

    @pl.loop(0, ti, step=tpu_target.sc_chunk_size)
    def _(ti_offset):
      indices_chunk = indices_ref[pl.ds(ti_offset, tpu_target.sc_chunk_size)]

      @pl.loop(0, td, step=tpu_target.sc_chunk_size)
      def _(td_offset):
        # 1. Start by loading `packing * tpu_target.sc_chunk_size` worth of
        # rows from `x_ref` into registers.
        x_chunks = []
        for ti_offset_addend in range(tpu_target.sc_chunk_size):
          row = ti_offset * packing + ti_offset_addend * packing
          col = td_offset
          x_chunks.append(
              x_ref[row, pl.ds(col, packing * tpu_target.sc_chunk_size)]
          )

        # 2. We need to bitcast the loaded chunk to i32 to be able to do bit
        # twiddling on it.
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
        shift_amounts = (indices_chunk % packing) * bitwidth
        shift_amounts = [
            shift_amounts[i] for i in range(tpu_target.sc_chunk_size)
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
        o_chunks = []
        for i in range(0, len(x_chunks), packing):
          o_chunk = x_chunks[i]
          for j in range(1, packing):
            o_chunk = jax.lax.bitwise_or(o_chunk, x_chunks[i + j])
          o_chunks.append(o_chunk)

        # 6. We need to bitcast the packed chunks back to the original type
        # and finally store them in the output reference.
        o_chunks = [plsc.bitcast(o_chunk, o_ref.dtype) for o_chunk in o_chunks]
        for ti_offset_addend, o_chunk in enumerate(o_chunks):
          row = ti_offset + ti_offset_addend * packing
          col = td_offset
          o_ref[row, pl.ds(col, packing * tpu_target.sc_chunk_size)] = o_chunk

  return kernel(indices, a)
