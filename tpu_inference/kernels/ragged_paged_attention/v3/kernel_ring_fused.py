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
"""Fused ring attention for prefill context parallelism (PCP).

Unlike ``kernel_ring_extern.py`` -- which processes one ``(query chunk, KV
shard)`` pair per kernel launch and merges the ``cp_group_size`` partials with
an external LSE reduction -- this kernel performs the whole ring **inside a
single launch**. It runs under ``jax.shard_map`` over the PCP mesh axis; each
device holds its local query tokens and its local KV shard, and the kernel:

  1. seeds a double-buffered HBM scratch with the local KV shard,
  2. loops ``P = cp_group_size`` times, and at each step
       - starts an async remote copy that rotates the current KV shard one hop
         around the ring (send to the left neighbor, receive from the right),
       - runs online-softmax flash attention of the local queries against the
         shard currently in hand, accumulating ``(m, l, acc)`` in VMEM,
       - waits for the rotated shard to land before the next step,
  3. finalizes ``out = acc / l`` (and optionally ``lse = m + log(l)``).

Correctness across the ring is carried entirely by **per-token global
positions**: each KV token's global position travels with the shard
(``kv_pos``) and each query token has its own global position (``q_pos``), so
the causal test is simply ``q_pos[i] >= kv_pos[j]``. This makes the
load-balanced head-tail layout fall out for free (a token's chunk only affects
its position) and lets padding be masked by giving padded KV tokens a sentinel
position. Shards that lie entirely in a query's future contribute nothing (all
masked); no launch is skipped, but the masked matmul is cheap and the ring stays
in lock-step across devices.

This kernel is intentionally a self-contained *dense* flash-attention ring (the
current-step KV is not yet in the paged cache). Attention against an existing
paged KV cache (the previous context) is a separate, non-ring launch whose
output is merged with this one via the returned LSE.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.collectives import util as collectives_util

# Sentinel global position for padded KV tokens: larger than any real position
# so the causal test ``q_pos >= kv_pos`` always excludes them.
PADDING_POSITION = jnp.iinfo(jnp.int32).max


def _ring_attention_kernel(
    # Inputs (HBM).
    q_hbm_ref,  # [t_q, num_q_heads, head_dim]
    kv_hbm_ref,  # [t_kv, num_kv_heads, 2 * head_dim] (k concat v on last dim)
    q_pos_hbm_ref,  # i32[t_q, 1]
    kv_pos_hbm_ref,  # i32[1, t_kv]
    # Outputs (HBM).
    o_hbm_ref,  # [t_q, num_q_heads, head_dim]
    lse_hbm_ref,  # f32[t_q, num_q_heads] or None
    # HBM scratch: double-buffered rotating KV shard + its positions.
    kv_scratch_ref,  # [2, t_kv, num_kv_heads, 2 * head_dim]
    kv_pos_scratch_ref,  # i32[2, 1, t_kv]
    # VMEM scratch.
    q_vmem_ref,  # [t_q, num_q_heads, head_dim]
    q_pos_vmem_ref,  # i32[t_q, 1]
    kv_vmem_ref,  # [t_kv, num_kv_heads, 2 * head_dim]
    kv_pos_vmem_ref,  # i32[1, t_kv]
    m_ref,  # f32[t_q, num_q_heads]
    l_ref,  # f32[t_q, num_q_heads]
    acc_ref,  # f32[t_q, num_q_heads, head_dim]
    o_vmem_ref,  # [t_q, num_q_heads, head_dim] (out dtype)
    lse_vmem_ref,  # f32[t_q, num_q_heads]
    # Semaphores.
    local_copy_sem,  # DMA
    send_sems,  # DMA[2, cp_group_size]
    recv_sems,  # DMA[2, cp_group_size]
    *,
    axis_name: str,
    cp_group_size: int,
    sm_scale: float,
    soft_cap: float | None,
):
    num_q_heads = q_hbm_ref.shape[1]
    num_kv_heads = kv_hbm_ref.shape[1]
    head_dim = q_hbm_ref.shape[2]
    q_per_kv = num_q_heads // num_kv_heads
    P = cp_group_size

    my_id = lax.axis_index(axis_name)
    left = lax.rem(my_id + P - 1, jnp.int32(P))
    right = lax.rem(my_id + 1, jnp.int32(P))

    # --- Seed: local shard -> working slot 0, load queries into VMEM. ---
    seed_kv = pltpu.make_async_copy(kv_hbm_ref, kv_scratch_ref.at[0],
                                    local_copy_sem)
    seed_kv.start()
    seed_kv.wait()
    seed_pos = pltpu.make_async_copy(kv_pos_hbm_ref, kv_pos_scratch_ref.at[0],
                                     local_copy_sem)
    seed_pos.start()
    seed_pos.wait()
    load_q = pltpu.make_async_copy(q_hbm_ref, q_vmem_ref, local_copy_sem)
    load_q.start()
    load_q.wait()
    load_qpos = pltpu.make_async_copy(q_pos_hbm_ref, q_pos_vmem_ref,
                                      local_copy_sem)
    load_qpos.start()
    load_qpos.wait()

    m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
    l_ref[...] = jnp.zeros_like(l_ref)
    acc_ref[...] = jnp.zeros_like(acc_ref)

    # Barrier so every device has seeded its working slot before any remote
    # write targets a neighbor's scratch.
    collectives_util.local_barrier(left, right)

    def _remote_rotate(step, wait):
        """Send working slot to the left, receive right's shard into recv slot."""
        work = lax.rem(step, 2)
        recv = lax.rem(step + 1, 2)
        kv_op = pltpu.make_async_remote_copy(
            src_ref=kv_scratch_ref.at[work],
            dst_ref=kv_scratch_ref.at[recv],
            send_sem=send_sems.at[0, step],
            recv_sem=recv_sems.at[0, step],
            device_id=(left, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        pos_op = pltpu.make_async_remote_copy(
            src_ref=kv_pos_scratch_ref.at[work],
            dst_ref=kv_pos_scratch_ref.at[recv],
            send_sem=send_sems.at[1, step],
            recv_sem=recv_sems.at[1, step],
            device_id=(left, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        if wait:
            kv_op.wait()
            pos_op.wait()
        else:
            kv_op.start()
            pos_op.start()

    def _attend(step):
        """Online-softmax update of (m, l, acc) with the shard in working slot."""
        work = lax.rem(step, 2)
        stage_kv = pltpu.make_async_copy(kv_scratch_ref.at[work], kv_vmem_ref,
                                         local_copy_sem)
        stage_kv.start()
        stage_kv.wait()
        stage_pos = pltpu.make_async_copy(kv_pos_scratch_ref.at[work],
                                          kv_pos_vmem_ref, local_copy_sem)
        stage_pos.start()
        stage_pos.wait()

        kv = kv_vmem_ref[...]  # [t_kv, num_kv_heads, 2 * head_dim]
        k = kv[..., :head_dim]
        v = kv[..., head_dim:]
        kv_pos = kv_pos_vmem_ref[...]  # [1, t_kv]
        q_pos = q_pos_vmem_ref[...]  # [t_q, 1]
        # [t_q, t_kv]: True where a query attends this KV token (also masks
        # padded KV via its sentinel position).
        causal = q_pos >= kv_pos

        for h in range(num_q_heads):
            kvh = h // q_per_kv
            q_h = q_vmem_ref[:, h, :]  # [t_q, head_dim]
            k_h = k[:, kvh, :]  # [t_kv, head_dim]
            v_h = v[:, kvh, :]  # [t_kv, head_dim]

            s = jnp.dot(q_h, k_h.T,
                        preferred_element_type=jnp.float32) * sm_scale
            if soft_cap is not None:
                s = soft_cap * jnp.tanh(s / soft_cap)
            s = jnp.where(causal, s, -jnp.inf)

            m_prev = m_ref[:, h:h + 1]  # [t_q, 1]
            s_max = jnp.max(s, axis=1, keepdims=True)  # [t_q, 1]
            m_cur = jnp.maximum(m_prev, s_max)
            # Guard the all-masked row (m_cur == -inf) so exp stays finite.
            m_safe = jnp.where(jnp.isinf(m_cur), 0.0, m_cur)
            p = jnp.exp(s - m_safe)  # [t_q, t_kv]
            correction = jnp.exp(m_prev - m_safe)  # [t_q, 1]

            l_ref[:, h:h + 1] = correction * l_ref[:, h:h + 1] + jnp.sum(
                p, axis=1, keepdims=True)
            m_ref[:, h:h + 1] = m_cur
            pv = jnp.dot(p.astype(v_h.dtype),
                         v_h,
                         preferred_element_type=jnp.float32)  # [t_q, head_dim]
            acc_ref[:, h, :] = correction * acc_ref[:, h, :] + pv

    # --- Ring: rotate + attend, P steps. ---
    for step in range(P):
        if step < P - 1:
            _remote_rotate(step, wait=False)
        _attend(step)
        if step < P - 1:
            _remote_rotate(step, wait=True)

    # --- Finalize (compute in VMEM, DMA to HBM outputs). ---
    l = l_ref[...]  # [t_q, num_q_heads]
    l_safe = jnp.where(l == 0.0, 1.0, l)
    o_vmem_ref[...] = (acc_ref[...] / l_safe[:, :, None]).astype(
        o_vmem_ref.dtype)
    store_o = pltpu.make_async_copy(o_vmem_ref, o_hbm_ref, local_copy_sem)
    store_o.start()
    store_o.wait()
    # Always emit LSE (cheap); the wrapper drops it unless requested.
    lse_vmem_ref[...] = jnp.where(l == 0.0, -jnp.inf,
                                  m_ref[...] + jnp.log(l_safe))
    store_lse = pltpu.make_async_copy(lse_vmem_ref, lse_hbm_ref,
                                      local_copy_sem)
    store_lse.start()
    store_lse.wait()


def _ring_attention_shard(
    q,  # [t_q, num_q_heads, head_dim]
    kv,  # [t_kv, num_kv_heads, 2 * head_dim]
    q_pos,  # i32[t_q, 1]
    kv_pos,  # i32[1, t_kv]
    *,
    axis_name: str,
    cp_group_size: int,
    sm_scale: float,
    soft_cap: float | None,
    return_lse: bool,
    collective_id: int,
    out_dtype,
):
    t_q, num_q_heads, head_dim = q.shape
    t_kv, num_kv_heads, _ = kv.shape

    # o and lse are real outputs; the two HBM buffers are kernel-managed HBM
    # scratch expressed as outputs (Mosaic only allocates VMEM/SMEM/sem scratch).
    out_shapes = [
        jax.ShapeDtypeStruct((t_q, num_q_heads, head_dim), out_dtype),  # o
        jax.ShapeDtypeStruct((t_q, num_q_heads), jnp.float32),  # lse
        jax.ShapeDtypeStruct((2, t_kv, num_kv_heads, 2 * head_dim),
                             kv.dtype),  # kv scratch
        jax.ShapeDtypeStruct((2, 1, t_kv), jnp.int32),  # kv_pos scratch
    ]
    scratch_shapes = [
        pltpu.VMEM((t_q, num_q_heads, head_dim), q.dtype),  # q
        pltpu.VMEM((t_q, 1), jnp.int32),  # q_pos
        pltpu.VMEM((t_kv, num_kv_heads, 2 * head_dim), kv.dtype),  # kv staged
        pltpu.VMEM((1, t_kv), jnp.int32),  # kv_pos staged
        pltpu.VMEM((t_q, num_q_heads), jnp.float32),  # m
        pltpu.VMEM((t_q, num_q_heads), jnp.float32),  # l
        pltpu.VMEM((t_q, num_q_heads, head_dim), jnp.float32),  # acc
        pltpu.VMEM((t_q, num_q_heads, head_dim), out_dtype),  # o (staged)
        pltpu.VMEM((t_q, num_q_heads), jnp.float32),  # lse (staged)
        pltpu.SemaphoreType.DMA,  # local copy sem
        pltpu.SemaphoreType.DMA((2, cp_group_size)),  # send sems
        pltpu.SemaphoreType.DMA((2, cp_group_size)),  # recv sems
    ]

    out = pl.pallas_call(
        functools.partial(
            _ring_attention_kernel,
            axis_name=axis_name,
            cp_group_size=cp_group_size,
            sm_scale=sm_scale,
            soft_cap=soft_cap,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 4,
            out_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 4,
            scratch_shapes=scratch_shapes,
            grid=(1, ),
        ),
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        out_shape=out_shapes,
        name=f"ring_attention_p{cp_group_size}",
    )(q, kv, q_pos, kv_pos)
    o, lse = out[0], out[1]  # out[2:] are the HBM scratch buffers (discarded)
    return (o, lse) if return_lse else o


def ring_attention(
    mesh: jax.sharding.Mesh,
    axis_name: str,
    q: jax.Array,  # [t_q, num_q_heads, head_dim]  (per-device local queries)
    k: jax.Array,  # [t_kv, num_kv_heads, head_dim] (per-device local KV shard)
    v: jax.Array,  # [t_kv, num_kv_heads, head_dim]
    q_positions: jax.Array,  # i32[t_q]   global position of each local query
    kv_positions: jax.
    Array,  # i32[t_kv]  global position of each local KV token
    *,
    sm_scale: float,
    soft_cap: float | None = None,
    return_lse: bool = False,
    out_dtype: Any = None,
    collective_id: int = 0,
):
    """Fused ring attention across the ``axis_name`` mesh axis.

    Every device passes only its *local* query tokens and its *local* KV shard;
    the kernel streams the shards around the ring internally. Global positions
    (``q_positions`` / ``kv_positions``) drive the causal mask and let padded KV
    tokens be excluded (set their position to ``PADDING_POSITION``).

    Args:
      mesh: device mesh; ``axis_name`` is the ring / PCP axis of size ``P``.
      axis_name: the ring axis name.
      q, k, v: per-device local tensors. ``num_q_heads`` must be a multiple of
        ``num_kv_heads`` (GQA/MQA).
      q_positions, kv_positions: per-token global positions.
      sm_scale: softmax scale applied to ``q @ k^T``.
      soft_cap: optional logit soft cap.
      return_lse: also return ``lse = m + log(l)`` per (token, head).
      out_dtype: output dtype (defaults to ``q.dtype``).
      collective_id: barrier-semaphore id (unique per concurrent kernel).

    Returns:
      ``out`` (and ``lse`` if ``return_lse``), sharded like ``q`` on the query
      axis.
    """
    if out_dtype is None:
        out_dtype = q.dtype
    cp_group_size = mesh.shape[axis_name]
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(f"num_q_heads={q.shape[1]} must be divisible by "
                         f"num_kv_heads={k.shape[1]}.")

    kv = jnp.concatenate([k, v], axis=-1)
    q_pos = q_positions.astype(jnp.int32).reshape(-1, 1)
    kv_pos = kv_positions.astype(jnp.int32).reshape(1, -1)

    P = jax.sharding.PartitionSpec
    q_spec = P(axis_name, None, None)
    kv_spec = P(axis_name, None, None)
    # Match _ring_attention_shard's return: bare `o`, or `(o, lse)`.
    out_specs = (q_spec, P(axis_name, None)) if return_lse else q_spec

    fn = jax.shard_map(
        functools.partial(
            _ring_attention_shard,
            axis_name=axis_name,
            cp_group_size=cp_group_size,
            sm_scale=sm_scale,
            soft_cap=soft_cap,
            return_lse=return_lse,
            collective_id=collective_id,
            out_dtype=out_dtype,
        ),
        mesh=mesh,
        in_specs=(q_spec, kv_spec, P(axis_name, None), P(None, axis_name)),
        out_specs=out_specs,
        check_vma=False,
    )
    return fn(q, kv, q_pos, kv_pos)
