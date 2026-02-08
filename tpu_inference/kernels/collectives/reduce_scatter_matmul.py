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

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.collectives import util

LEFT = 0
RIGHT = 1


def mod(x, n):
  return lax.rem(x + n, n)


def signal(left_or_right, semaphore, num_devices):
  my_id = lax.axis_index('x')
  if left_or_right == LEFT:
    neighbor = mod(my_id - 1, num_devices)
  else:
    neighbor = mod(my_id + 1, num_devices)
  pltpu.semaphore_signal(
      semaphore,
      inc=1,
      device_id=(neighbor,),
      device_id_type=pltpu.DeviceIdType.MESH,
  )


def reduce_scatter_matmul_kernel(
    x_ref,  # [m, k]
    y_ref,  # [n, k]
    # Output
    o_ref,  # [m_per_dev, n]
    left_hbm_scratch,  # [2, m_per_dev, n//2]
    right_hbm_scratch,  # [2, m_per_dev, n//2]
    # Scratches
    left_recv_sem,
    left_send_sem,
    left_copy_sem,
    right_copy_sem,
    right_recv_sem,
    right_send_sem,
    left_capacity_sem,
    right_capacity_sem,
    *,
    num_devices: int,
    debug_mode: bool = False,
    bm: int,
    bk: int,
    bn: int,
):
  my_id = lax.axis_index('x')

  def debug_print(msg, *args):
    if debug_mode:

      @pl.when(my_id == 0)
      def _debug_print():
        pl.debug_print(msg, *args)

  m, k_per_dev = x_ref.shape
  n, _ = y_ref.shape
  # TODO(xw32): temp assertion. Remove before merging.
  # assert m == 768
  # assert n == 512
  # assert k_per_dev == 128

  m_shard_sz = m//num_devices
  n_shard_sz = n//2
  assert left_hbm_scratch.shape == (2, m_shard_sz, n_shard_sz)
  assert right_hbm_scratch.shape == (2, m_shard_sz, n_shard_sz)
  print(f'xw32 reduce_scatter_matmul_kernel begins: {x_ref.shape=}, {y_ref.shape=}, {o_ref.shape=}, {left_hbm_scratch.shape=}, {right_hbm_scratch.shape=}, {left_hbm_scratch.shape=}, {right_hbm_scratch.shape=}')

  outer_idx = pl.program_id(0)
  phase_idx = pl.program_id(1)
  num_outer_steps = pl.num_programs(0)
  num_phases = pl.num_programs(1)
  global_idx = outer_idx*num_phases + phase_idx
  num_global_steps = num_outer_steps * num_phases
  debug_print(
      '===== starting a grid, outer_step={}, phase={}, num_devices={}, m={}, n={}, k_per_dev={} =====',
      outer_idx,
      phase_idx,
      num_devices,
      m,
      n,
      k_per_dev,
  )
  is_start = jnp.logical_and(outer_idx == 0, phase_idx == 0)
  is_end = (global_idx == num_global_steps - 1)

  left_working_slot = lax.rem(outer_idx, 2)
  left_receiving_slot = 1 - left_working_slot
  right_working_slot = lax.rem(pl.cdiv(global_idx, 2), 2)
  right_receiving_slot = 1 - right_working_slot
  right_neighbor = mod(my_id + 1, num_devices)
  left_neighbor = mod(my_id - 1, num_devices)

  # the device from where we receive a left copy.
  left_copy_device = mod(my_id + outer_idx + 1, num_devices)
  right_copy_device = mod(my_id - outer_idx - 1, num_devices)
  left_copy_slice = pl.ds(0, n_shard_sz)
  right_copy_slice = pl.ds(n_shard_sz, n_shard_sz)

  # xw32: cannot debug_print shapes.
  left_copy = pltpu.make_async_remote_copy(
      src_ref=left_hbm_scratch.at[left_working_slot],
      dst_ref=left_hbm_scratch.at[left_receiving_slot],
      send_sem=left_send_sem,
      recv_sem=left_recv_sem,
      device_id=(left_neighbor,),
      device_id_type=pltpu.DeviceIdType.MESH,
  )
  right_copy = pltpu.make_async_remote_copy(
      src_ref=right_hbm_scratch.at[right_working_slot],
      dst_ref=right_hbm_scratch.at[right_receiving_slot],
      send_sem=right_send_sem,
      recv_sem=right_recv_sem,
      device_id=(right_neighbor,),
      device_id_type=pltpu.DeviceIdType.MESH,
  )
  debug_print(
      '===== outer_idx={}, phase_idx={}, num_outer_steps={}, num_phases={}, global_idx={}, num_global_steps={} before Prologue left_working_slot={}, left_receiving_slot={}, right_working_slot={}, right_receiving_slot={}, left_neighbor={}, right_neighbor={} =====',
      outer_idx,
      phase_idx,
      num_outer_steps,
      num_phases,
      global_idx,
      num_global_steps,
      left_working_slot,
      left_receiving_slot,
      right_working_slot,
      right_receiving_slot,
      left_neighbor,
      right_neighbor,
  )

  # --- Prologue ---
  @pl.when(is_start)
  def _():
    # Barrier with both neighbors at the start, since we will be
    # communicating with both.
    util.local_barrier(left_neighbor, right_neighbor)
    # We tell our left neighbor that it is allowed to send to the right.
    # (and vice versa for right neighbor)
    signal(LEFT, right_capacity_sem, num_devices)
    signal(RIGHT, left_capacity_sem, num_devices)

  debug_print(
      '===== outer_step={}, phase={}, line161 =====',
      outer_idx,
      phase_idx,
  )
  @pl.when(~jnp.logical_or(is_start, is_end))
  def _():
    @pl.when(phase_idx == LEFT)
    def _():
      # We block here until our right neighbor tells use we can send to
      # the right.
      pltpu.semaphore_wait(right_capacity_sem, 1)
      right_copy.start()

    @pl.when(phase_idx == RIGHT)
    def _():
      # We block here until our left neighbor tells use we can send to
      # the left.
      pltpu.semaphore_wait(left_capacity_sem, 1)
      left_copy.start()

  # --- Body ---
  def inner_matmul_kernel(x_ref, y_ref, prev_z_ref, z_ref):
    # With should_accumulate_out=False, emit_pipeline does NOT load the
    # existing HBM output into VMEM. We pass the scratch as an explicit input
    # (prev_z_ref) so the received DMA data is loaded from HBM into VMEM.
    @pl.when(jnp.logical_and(outer_idx == 0, pl.program_id(2) == 0))
    def _():
      z_ref[...] = jnp.zeros_like(z_ref, dtype=jnp.float32)
      
    # Why having prev_z_ref?: emit_pipeline with should_accumulate_out=False does not load the existing HBM output into VMEM. The VMEM output buffer retains stale data from the previous pipeline call. Since LEFT and RIGHT matmul share the same pipeline, the LEFT result leaked into the RIGHT's VMEM accumulator at outer_idx=1, adding an extra 128.
    # Fix: Added prev_z_ref as a third input to the pipeline (same HBM ref as the output). The kernel explicitly initializes z_ref from prev_z_ref when outer_idx != 0, ensuring the DMA-received data is properly loaded from HBM rather than relying on stale VMEM state.
    
    @pl.when(jnp.logical_and(outer_idx != 0, pl.program_id(2) == 0))
    def _():
      z_ref[...] = prev_z_ref[...].astype(jnp.float32)
    z_ref[...] += jax.lax.dot_general(
        x_ref[...],
        y_ref[...],
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
  matmul_pipeline = pltpu.emit_pipeline(
      inner_matmul_kernel,
      in_specs=[pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bn, bk), lambda i, j, k: (j, k)),
                pl.BlockSpec((bm, bn), lambda i, j, k: (i, j))],
      out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
      should_accumulate_out=False,
      grid=(m_shard_sz//bm, n_shard_sz//bn, k_per_dev//bk),
      dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL, pltpu.ARBITRARY),
  )

  @pl.when(global_idx < num_global_steps - 2)
  def matmul():
    # x_start = outer_step*(m//num_devices)
    # NB: x_ref,  # [m, k]
    # NB: y_ref,  # [n, k]
    # NB: left_hbm_scratch,  # [2, m//num_devices, n//2]
    @pl.when(phase_idx == LEFT)
    def _():
      x_start = ((outer_idx+my_id)%num_devices)*m_shard_sz
      x_left_ref = x_ref.at[pl.ds(x_start, m_shard_sz)]
      y_left_ref = y_ref.at[left_copy_slice, :]

      # TODO(xw32): remove the special case assert
      # assert x_left_ref.shape == (384, 128)
      # assert y_left_ref.shape == (256, 128)

      left_hbm_scratch_ref = left_hbm_scratch.at[left_working_slot]
      matmul_pipeline(
          x_left_ref,
          y_left_ref,
          left_hbm_scratch_ref,
          left_hbm_scratch_ref,
      )

    @pl.when(phase_idx == RIGHT)
    def _():
      # m_shard_sz=384 # xw32: remove the temp comments later.
      # The LEFT and RIGHT halves of the reduce-scatter use bidirectional rings flowing in opposite directions. The LEFT ring sends data left (device d → d-1), so the x-block index should advance forward: (outer_idx + my_id) % N. The RIGHT ring sends data right (device d → d+1), so the x-block index should advance backward: (my_id - outer_idx) % N. Both phases were incorrectly using the forward formula. This was masked with 2 devices because (a+b) % 2 ≡ (a-b) % 2 for all integers, but breaks for 4+ devices
      # Here is the reason why we need "my_id-outer_idx" instead of "my_id+outer_idx" https://gist.github.com/vanbasten23/edfb673c0352c49efa8158c8338ec307
      x_start = mod(my_id - outer_idx, num_devices)*m_shard_sz
      x_right_ref = x_ref.at[pl.ds(x_start, m_shard_sz)]
      y_right_ref = y_ref.at[right_copy_slice, :]
      right_hbm_scratch_ref = right_hbm_scratch.at[right_working_slot]

      # TODO(xw32): remove the special case assert
      # assert x_right_ref.shape == (384, 128)
      # assert y_right_ref.shape == (256, 128)

      matmul_pipeline(
          x_right_ref,
          y_right_ref,
          right_hbm_scratch_ref,
          right_hbm_scratch_ref,
      )

  def inner_save_to_output_kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...]

  save_to_output_pipeline = pltpu.emit_pipeline(
      inner_save_to_output_kernel,
      in_specs=[pl.BlockSpec((bm, bn), lambda i, j: (i, j)),],
      out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
      should_accumulate_out=False,
      grid=(m_shard_sz//bm, n_shard_sz//bn),
      dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
  )

  @pl.when(global_idx >= num_global_steps - 2)
  def save_to_output():
    # NB:
    # o_ref,  # [m_per_dev, n]
    # left_hbm_scratch,  # [2, m_per_dev, n//2]
    # right_hbm_scratch,  # [2, m_per_dev, n//2]
    @pl.when(phase_idx == LEFT)
    def _():
      save_to_output_pipeline(
          left_hbm_scratch.at[left_working_slot],
          o_ref.at[:, pl.ds(0, n_shard_sz)],
      )

    @pl.when(phase_idx == RIGHT)
    def _():
      save_to_output_pipeline(
          right_hbm_scratch.at[right_working_slot],
          o_ref.at[:, pl.ds(n_shard_sz, n_shard_sz)],
      )

  # --- Epilogue ---
  @pl.when(~jnp.logical_or(is_start, is_end))
  def _():
    @pl.when(phase_idx == LEFT)
    def _():
      right_copy.wait()
      signal(LEFT, right_capacity_sem, num_devices)

    @pl.when(phase_idx == RIGHT)
    def _():
      left_copy.wait()
      signal(RIGHT, left_capacity_sem, num_devices)

  # Clean up semaphores so that they exit with a value of 0.
  @pl.when(outer_idx == num_outer_steps - 1)
  def _():
    @pl.when(phase_idx == LEFT)
    def _():
      pltpu.semaphore_wait(right_capacity_sem, 1)

    @pl.when(phase_idx == RIGHT)
    def _():
      pltpu.semaphore_wait(left_capacity_sem, 1)

  @pl.when(is_end)
  def _():
    debug_print(
        'End of kernel left_recv_sem={}, left_send_sem={}, left_copy_sem={},'
        ' right_copy_sem={}, right_recv_sem={}, right_send_sem={},'
        ' left_capacity_sem={}, right_capacity_sem={}',
        pltpu.semaphore_read(left_recv_sem),
        pltpu.semaphore_read(left_send_sem),
        pltpu.semaphore_read(left_copy_sem),
        pltpu.semaphore_read(right_copy_sem),
        pltpu.semaphore_read(right_recv_sem),
        pltpu.semaphore_read(right_send_sem),
        pltpu.semaphore_read(left_capacity_sem),
        pltpu.semaphore_read(right_capacity_sem),
    )


def reduce_scatter_matmul(
    x: jax.Array,
    y: jax.Array,
    mesh: jax.sharding.AbstractMesh,  # xw32q: do I need it?
    axis_name: str,
    collective_id: int | None = 0,
    scatter_dim: int = 1,
    rhs_transpose: bool = False,
    debug: bool = False,
    interpret: bool = False,
    debug_mode: bool = False,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
):
  """Reduce-scatter matmul kernel.
  
  Args:
    x: The left hand side of the matmul [m, k]
    y: The right hand side of the matmul [n, k]
    mesh: The mesh for the collective.
    axis_name: The axis name for the collective.
    collective_id: for the barrier semaphore.
    scatter_dim: Scatter dimension.
    rhs_transpose: RHS transpose.
    debug: Debug mode.
    interpret: Interpret mode.

  Returns:
    The result of the reduce-scatter matmul.
  """
  # Naive implementation.
  # return jax.lax.psum_scatter(
  #     jnp.dot(x, y.T, preferred_element_type=jnp.float32),
  #     axis_name=axis_name,
  #     tiled=True,
  #     scatter_dimension=scatter_dim,
  # )
  assert x.shape[1] == y.shape[1]
  # TODO(xw32): rename m, n to m_per_dev, n_per_dev.
  m, k = x.shape
  n, _ = y.shape
  # TODO(xw32): temp assertion. Remove before merging.
  # assert m == 768
  # assert n == 512
  # assert k == 128

  num_devices = jax.lax.psum(1, axis_name)
  out_shape = (
      # final result.
      jax.ShapeDtypeStruct(
          (m//num_devices, n), jnp.float32
      ),
      # The rs kernel put the hbm scratch space in out_shape.
      # out_left_hbm_scratch: [working/recv, m//num_devices//2, n//2]
      jax.ShapeDtypeStruct(
          (2, m//num_devices, n//2), jnp.float32
      ),
      # out_right_hbm_scratch: [working/recv, m//num_devices//2, n//2]
      jax.ShapeDtypeStruct(
          (2, m//num_devices, n//2), jnp.float32
      ),
  )
  grid_spec = pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
          pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
      ],
      out_specs=[
          pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
          pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
          pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
      ],
      grid=(num_devices+1, 2),
      scratch_shapes=(
          [pltpu.SemaphoreType.DMA] * 6
          + [pltpu.SemaphoreType.REGULAR] * 2  # Capacity semaphores
      ),
  )
  return pl.pallas_call(
      functools.partial(
          reduce_scatter_matmul_kernel,
          num_devices=num_devices,
          debug_mode=debug_mode,
          bm=bm,
          bk=bk,
          bn=bn,
      ),
      out_shape=out_shape,
      grid_spec=grid_spec,
      compiler_params=pltpu.CompilerParams(collective_id=collective_id),
  )(x, y)[0]


def reduce_scatter_matmul_ref_impl(
    x: jax.Array,
    y: jax.Array,
    axis_name: str,
    scatter_dim: int = 1,
    rhs_transpose: bool = False,
):
  print(
      f'xw32 reduce_scatter_matmul_ref_impl begins: local shapes are {x.shape=}, {y.shape=}')
  m, k = x.shape
  n, _ = y.shape
  assert x.shape[1] == y.shape[1]

  axis_size = jax.lax.psum(1, axis_name)
  if scatter_dim == 0:
    assert m % axis_size == 0
    m_shard_sz = m // axis_size
    n_shard_sz = n // 2
  else:
    # TODO(xw32)
    assert False, "NYI(xw32)"
  assert (m, k) == (768, 256 // axis_size)
  assert (n, k) == (512, 256 // axis_size)

  accum_dtype = jnp.float32  # Need to change to jnp.int32 for int8.
  out_left = jnp.zeros((m_shard_sz, n_shard_sz), accum_dtype)
  out_right = jnp.zeros((m_shard_sz, n_shard_sz), accum_dtype)
  print(
      f'xw32 reduce_scatter_matmul_ref_impl before fori_loop {m_shard_sz=}, {n_shard_sz=}, {out_left.shape=}, {out_right.shape=}'
  )

  # (int, a) -> a
  def body(i, carry):
    my_id = jax.lax.axis_index(axis_name)
    out_left, out_right = carry
    left_index = jax.lax.rem(my_id + i, axis_size)
    right_index = jax.lax.rem(my_id - i + axis_size, axis_size)
    if scatter_dim == 0:
      # what are the below variables for?
      x_left_chunk_index = left_index * m_shard_sz
      x_right_chunk_index = right_index * m_shard_sz
      y_left_chunk_index = 0
      y_right_chunk_index = n_shard_sz
    else:
      assert False, "NYI(xw32)"

    # dynamic_slice(operand, start_indices, slice_sizes)
    x_left = jax.lax.dynamic_slice(x, (x_left_chunk_index, 0),
                                   (m_shard_sz, k))
    x_right = jax.lax.dynamic_slice(x, (x_right_chunk_index, 0),
                                    (m_shard_sz, k))
    y_left = jax.lax.dynamic_slice(y, (y_left_chunk_index, 0),
                                   (n_shard_sz, k))
    y_right = jax.lax.dynamic_slice(y, (y_right_chunk_index, 0),
                                    (n_shard_sz, k))
    jax.debug.print('xw32 ref impl, my_id={}, i={}, x_left=x[{}:{}, {}:{}], x_right=x[{}:{},{}:{}], y_left=y[{}:{},{}:{}], y_right=y[{}:{},{}:{}]', my_id, i, x_left_chunk_index, x_left_chunk_index+m_shard_sz, 0, k, x_right_chunk_index, x_right_chunk_index+m_shard_sz, 0, k, y_left_chunk_index, y_left_chunk_index+n_shard_sz, 0, k, y_right_chunk_index, y_right_chunk_index+n_shard_sz, 0, k)

    assert x_left.shape == (m // axis_size, k)
    assert x_right.shape == (m // axis_size, k)
    assert y_left.shape == (n // 2, k)
    assert y_right.shape == (n // 2, k)
    print(
        f'xw32 reduce_scatter_matmul_ref_impl, inside fori_loop, {x_left.shape=}, {y_left.shape=}'
    )
    out_left_local = jnp.dot(x_left, y_left.T)
    out_right_local = jnp.dot(x_right, y_right.T)
    assert out_left_local.shape == (m // axis_size, n // 2)
    assert out_right_local.shape == (m // axis_size, n // 2)
    out_left += out_left_local
    out_right += out_right_local
    # jax.debug.print('xw32 line1050 my_id={}, i={}, out_left[0]={}, out_right[0]={}, x_left[0]={}, y_left[0]={}', my_id, i, out_left[0], out_right[0], x_left[0], y_left[0])
    assert out_left.shape == (m // axis_size, n // 2)
    assert out_right.shape == (m // axis_size, n // 2)

    out_left = jax.lax.ppermute(
        out_left,
        axis_name,
        [(i, (i - 1 % axis_size)) for i in range(axis_size)],
    )
    out_right = jax.lax.ppermute(
        out_right,
        axis_name,
        [(i, (i + 1) % axis_size) for i in range(axis_size)],
    )
    return out_left, out_right  # End of body

  # def fori_loop(lower, upper, body_fun, init_val):
  out_left, out_right = jax.lax.fori_loop(0, axis_size, body,
                                          (out_left, out_right))
  out_left = out_left.astype(x.dtype)
  out_right = out_right.astype(x.dtype)  # xw32q: what's the shape?
  res = jnp.concatenate([out_left, out_right],
                        axis=-1)  # xw32q: what's the shape?
  print(f'xw32 reduce_scatter_matmul_ref_impl ends: {res.shape=}')
  return res
