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

@functools.partial(jax.jit, static_argnames=["window_bounds", "unroll_factors"])
@jt.jaxtyped(typechecker=typeguard.typechecked)
def gather_3d_to_2d(
    a: jt.Real[jt.Array, "B R Dr"],
    indices: jt.Int32[jt.Array, "I"],
    *,
    window_bounds: tuple[int] = (16,),  # pylint: disable=[g-one-element-tuple]
    unroll_factors: tuple[int, int, int] = (0, 0, 0),
) -> jt.Real[jt.Array, "I R*Dr"]:
  """Gathers values from `a` given a set of indices, then ravels the last two dimensions.

  Args:
    a: The array to gather from.
    indices: The indices to gather.
    window_bounds: The window bounds in the `I` dimension. These should be set
      as big as `Spmem` allows. Some example good values are - Viperfish -
      (16,), Ghostlite - (8,).
    unroll_factors: The unroll factors for the window_bounds, and `R` and `D`
      dimensions. 0 = no unrolling, -1 = unroll by the dimension size, > 0 =
      unroll by this factor.

  Returns:
    The gathered values.
  """
  output_bsz = indices.shape[0]

  # Pad to window bounds.
  a = _suffix_pad_to_multiple(a, window_bounds + (1, 1))
  indices = _suffix_pad_to_multiple(indices, window_bounds[:1])

  result_padded = _gather_3d_to_2d_impl(
      a, indices, window_bounds=window_bounds, unroll_factors=unroll_factors
  )
  return result_padded[:output_bsz, :]

def _suffix_pad_to_multiple(
    a: jt.Real[jt.Array, "..."],
    divisors: Sequence[int],
) -> jt.Real[jt.Array, "..."]:
  padding = (
      lambda size, divisor: (size + divisor - 1) // divisor * divisor - size
  )
  pad_width = [(0, padding(s, m)) for s, m in zip(a.shape, divisors)]
  return jnp.pad(a, pad_width, mode="empty")

@functools.partial(jax.jit, static_argnames=["window_bounds", "unroll_factors"])
@jt.jaxtyped(typechecker=typeguard.typechecked)
def _gather_3d_to_2d_impl(
    a: jt.Real[jt.Array, "B R Dr"],
    indices: jt.Int32[jt.Array, "I"],
    *,
    window_bounds: tuple[int],  # pylint: disable=[g-one-element-tuple]
    unroll_factors: tuple[int, int, int],
) -> jt.Real[jt.Array, "I R*Dr"]:
  """Internal implementation of `gather_3d_to_2d`."""
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
  input_bsz, sublane_dim, lane_dim = a.shape
  if sublane_dim % 8 != 0 or lane_dim % 128 != 0:
    raise ValueError(
        "The last two operand dimensions must be divisible by 8 and 128"
        f" respectively. Got {a.shape=}."
    )
  itemsize = a.dtype.itemsize
  if itemsize != 2:
    raise ValueError(f"Only 16-bit types are supported. Got {a.dtype}.")

  inner_dim_size = sublane_dim * lane_dim
  output_bsz = indices.shape[0]

  bitwidth = itemsize * 8
  packing = 32 // bitwidth

  # Window size in the `I` dimension.
  ti, *_ = window_bounds
  tr, td = a.shape[1:]

  maybe_parallel_loop = (
      lambda *args, unroll, **kwargs: pl.loop(*args, **kwargs)
      if unroll == 0
      else plsc.parallel_loop(*args, unroll=unroll, **kwargs)
  )

  @functools.partial(
      pl.pallas_call,
      out_shape=jax.ShapeDtypeStruct([output_bsz, inner_dim_size], a.dtype),
      grid=(output_bsz // ti, sublane_dim // tr, lane_dim // td),
      in_specs=[
          pl.BlockSpec((ti,), lambda i, j, k: i),
          pl.BlockSpec(
              (input_bsz, sublane_dim, lane_dim),
              lambda i, j, k: (0, 0, 0),
              memory_space=pltpu.HBM,
          ),
      ],
      out_specs=pl.BlockSpec((ti, tr * td), lambda i, j, k: (i, j)),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=["parallel", "arbitrary", "arbitrary"],
          kernel_type=pltpu.CoreType.SC_VECTOR_SUBCORE,
      ),
  )
  def kernel(indices_ref, x_ref, o_ref):
    return pl.run_scoped(
        functools.partial(scoped_kernel, indices_ref, x_ref, o_ref),
        scratch_ref=plsc.MemoryRef(
            (ti, tr, td), a.dtype, memory_space=pltpu.VMEM,
        ),
        sem_ref=pltpu.SemaphoreType.DMA,
    )

  def scoped_kernel(indices_ref, x_ref, o_ref, scratch_ref, sem_ref):
    pltpu.async_copy(x_ref.at[indices_ref], scratch_ref, sem_ref).wait()

    chunk_size = tpu_target.sc_chunk_size
    ti_unroll_factor, tr_unroll_factor, td_unroll_factor = unroll_factors

    @maybe_parallel_loop(0, ti, step=packing, unroll=ti_unroll_factor)
    def _(ti_offset):
      @maybe_parallel_loop(0, tr, step=packing, unroll=tr_unroll_factor)
      def _(tr_offset):
        @maybe_parallel_loop(0, td, step=chunk_size, unroll=td_unroll_factor)
        def _(td_offset):
          chunks = []
          for row in range(packing):
            input_indices = (
                ti_offset + row,
                pl.ds(tr_offset, 2),
                pl.ds(td_offset, chunk_size),
            )
            chunks.append(
                jax.lax.bitcast_convert_type(
                    scratch_ref[input_indices], jnp.int16
                )
            )

          pack_format = plsc.PackFormat.COMPRESSED
          unpacked = zip(*(
              plsc.unpack(c.reshape(chunk_size, 2), format=pack_format)
              for c in chunks
          ))
          for i, (chunk0, chunk1) in enumerate(unpacked):
            result = plsc.pack(chunk1, chunk0, format=pack_format)
            result = result.reshape(2, chunk_size)
            output_indices = (
                pl.ds(ti_offset, 2),
                pl.ds(
                    tr_offset * td + (packing - 1 - i) * td + td_offset,
                    chunk_size,
                ),
            )
            o_ref[output_indices] = jax.lax.bitcast_convert_type(
                result, o_ref.dtype
            )

  return kernel(indices, a)

