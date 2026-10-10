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
"""Fused dense/shared expert MLP Pallas kernel for MoE layers on TPU.

Computes gate/up projections, SiLU gating, and down projection for the shared
expert branch with VMEM-streamed weight tiles.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

_MAX_EXPERT_BUFFERS = 3


def _select_expert_buffers(m, h, i, experts, vmem_capacity_bytes):
    """Choose up to three slots, reserving 20% for padding and temporaries.

    Count explicit VMEM inputs, output and scratch, including quantization
    scales expanded over 128 lanes. As in RPA, this is a sizing heuristic;
    the compiler still decides resource feasibility, even for one slot.
    """
    expert_bytes = 3 * h * i + 4 * (2 * i + h)
    # Hidden input/output, quantized hidden, and FP32 output accumulator.
    fixed_bytes = m * h * (2 + 2 + 1 + 4)
    # BF16 activation, FP8 activation, and the two BF16 quantization scales.
    fixed_bytes += m * i * (2 + 1) + (h // 512 + i // 512) * m * 128 * 2
    fixed_bytes += m * pl.cdiv(experts, 128) * 128 * 4
    budget = vmem_capacity_bytes * 4 // 5
    return max(
        1, min(_MAX_EXPERT_BUFFERS, (budget - fixed_bytes) // expert_bytes))


def can_use_dense_expert(hidden,
                         w1,
                         w2,
                         s1,
                         s2,
                         b1,
                         b2,
                         *,
                         activation,
                         rhs_quant_dtype,
                         skip_padded_tokens=False):
    """Check arithmetic/layout support; the compiler owns resource limits.

    Zero routing coefficients already suppress padded tokens, so either
    skip_padded_tokens setting is supported. There is no prefill/decode
    inference here: the MoE entry applies its configurable token threshold.
    """
    if activation != "silu":
        return False
    if rhs_quant_dtype not in (None, jnp.float8_e4m3fn):
        return False
    if b1 is not None or b2 is not None or s1 is None or s2 is None:
        return False
    if hidden.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        return False
    m, h = hidden.shape
    e, i, out = w2.shape
    if min(m, h, i, e) <= 0:
        return False
    if m % 8 or h % 512 or i % 512:
        return False
    if out != h or w1.shape != (e, h, 2 * i):
        return False
    if (hidden.dtype != jnp.bfloat16 or w1.dtype != jnp.float8_e4m3fn
            or w2.dtype != jnp.float8_e4m3fn):
        return False
    return (s1.shape == (e, 1, 1, 2 * i) and s2.shape == (e, 1, 1, h)
            and s1.dtype == jnp.float32 and s2.dtype == jnp.float32)


def _quantize(x, q_ref, scale_ref):
    """Same BF16 scale/inverse and 512-wide FP8 quantization as gmm_v2."""
    # Keep K blocks first so each scale read is a contiguous [tokens, 128]
    # plane; placing K between tokens and lanes causes expensive relayouts.
    for k in range(0, x.shape[1], 512):
        block = x[:, k:k + 512]
        scale = jnp.max(jnp.abs(block), axis=1, keepdims=True) / 448.0
        inv = jnp.where(scale == 0, 0, 1 / scale)
        q_ref[:, k:k + 512] = (block * inv).astype(jnp.float8_e4m3fn)
        scale_ref[k // 512, :, :] = jnp.broadcast_to(scale, (x.shape[0], 128))


def _kernel(
    active_ref,
    x_ref,
    coeff_ref,
    w1_hbm,
    w2_hbm,
    s1_hbm,
    s2_hbm,
    out_ref,
    w1_ref,
    w2_ref,
    s1_ref,
    s2_ref,
    xq_ref,
    xs_ref,
    a_ref,
    aq_ref,
    asc_ref,
    owner_ref,
    active_ids_ref,
    sem_ref,
    *,
    m,
    h,
    i,
    experts,
    num_buffers,
):
    owner_ref[...] = jnp.zeros((m, h), jnp.float32)
    _quantize(x_ref[...], xq_ref, xs_ref)
    expert_columns = lax.broadcasted_iota(jnp.int32, coeff_ref.shape, 1)

    def compute(e, w1_ref, w2_ref, s1_ref, s2_ref):
        coeff = jnp.sum(jnp.where(expert_columns == e, coeff_ref[...], 0),
                        axis=1,
                        keepdims=True)
        # Gate/up projections in interleaved 128-column pairs, matching
        # the original fused GMM's arithmetic and BF16 rounding points.
        for n in range(0, i, 128):
            acc = jnp.zeros((m, 256), jnp.bfloat16)
            scale = jnp.concatenate(
                (s1_ref[:, n:n + 128], s1_ref[:, i + n:i + n + 128]), axis=1)
            for k in range(0, h, 512):
                rhs = jnp.concatenate(
                    (
                        w1_ref[k:k + 512, n:n + 128],
                        w1_ref[k:k + 512, i + n:i + n + 128],
                    ),
                    axis=1,
                )
                part = jnp.matmul(xq_ref[:, k:k + 512],
                                  rhs,
                                  preferred_element_type=jnp.float32).astype(
                                      jnp.bfloat16)
                xs = xs_ref[k // 512, :, :]
                part *= jnp.concatenate((xs, xs), axis=1)
                part *= scale.astype(jnp.bfloat16)
                acc += part
            a_ref[:, n:n + 128] = jax.nn.silu(acc[:, :128]) * acc[:, 128:]
        _quantize(a_ref[...], aq_ref, asc_ref)
        for n in range(0, h, 256):
            acc = jnp.zeros((m, 256), jnp.bfloat16)
            for k in range(0, i, 512):
                part = jnp.matmul(
                    aq_ref[:, k:k + 512],
                    w2_ref[k:k + 512, n:n + 256],
                    preferred_element_type=jnp.float32,
                ).astype(jnp.bfloat16)
                asc = asc_ref[k // 512, :, :]
                part *= jnp.concatenate((asc, asc), axis=1)
                part *= s2_ref[:, n:n + 256].astype(jnp.bfloat16)
                acc += part
            # Mask before accumulation; zero routing must suppress even
            # non-finite unused expert outputs, rather than multiply NaN*0.
            weighted = jnp.where(coeff != 0,
                                 acc.astype(jnp.float32) * coeff, 0)
            owner_ref[:, n:n + 256] += weighted

    def compact(e, count):

        @pl.when(active_ref[e] != 0)
        def record():
            active_ids_ref[count] = e

        return count + (active_ref[e] != 0).astype(jnp.int32)

    count = lax.fori_loop(0, experts, compact, jnp.int32(0))

    def transfers(step):
        e, slot = active_ids_ref[step], step % num_buffers
        return [
            pltpu.make_async_copy(w1_hbm.at[e], w1_ref.at[slot],
                                  sem_ref.at[slot, 0]),
            pltpu.make_async_copy(w2_hbm.at[e], w2_ref.at[slot],
                                  sem_ref.at[slot, 1]),
            pltpu.make_async_copy(s1_hbm.at[e, 0], s1_ref.at[slot],
                                  sem_ref.at[slot, 2]),
            pltpu.make_async_copy(s2_hbm.at[e, 0], s2_ref.at[slot],
                                  sem_ref.at[slot, 3]),
        ]

    @pl.when(count > 0)
    def run_experts():
        for initial in range(num_buffers - 1):

            @pl.when(count > initial)
            def prime(*, initial=initial):
                for copy in transfers(jnp.int32(initial)):
                    copy.start()

        def expert_body(step, _):
            if num_buffers == 1:
                for copy in transfers(step):
                    copy.start()
            for copy in transfers(step):
                copy.wait()

            if num_buffers > 1:

                @pl.when(step + num_buffers - 1 < count)
                def prefetch():
                    for copy in transfers(step + num_buffers - 1):
                        copy.start()

            slot = step % num_buffers
            compute(
                active_ids_ref[step],
                w1_ref.at[slot],
                w2_ref.at[slot],
                s1_ref.at[slot],
                s2_ref.at[slot],
            )
            return None

        # Prefetch up to num_buffers - 1 experts ahead. With one slot, each
        # expert finishes computing before the next DMA overwrites its inputs.
        # Compacting the local-expert active bitmap uses only scalar SMEM ops.
        lax.fori_loop(0, count, expert_body, None)

    out_ref[...] = owner_ref[...].astype(jnp.bfloat16)


@functools.partial(jax.jit, static_argnames=("num_primary_experts", ))
def dense_expert_moe(
    hidden,
    w1,
    w2,
    s1,
    s2,
    topk_weights,
    local_ids,
    *,
    num_primary_experts: int | None = None,
):
    """Local routed contribution, same [tokens, hidden]/BF16 output as GMM.

    IDs outside this rank must be -1. Scales use [E,1,1,N]. Callers must
    first check can_use_dense_expert. Shared experts and collectives stay
    outside this kernel, exactly as for fused_moe_func's standard path.
    """
    m_orig, h = hidden.shape
    pad_m = (-m_orig) % 16
    if pad_m:
        hidden = jnp.pad(hidden, ((0, pad_m), (0, 0)))
        topk_weights = jnp.pad(topk_weights, ((0, pad_m), (0, 0)))
        local_ids = jnp.pad(local_ids, ((0, pad_m), (0, 0)),
                            constant_values=-1)
    m = m_orig + pad_m
    experts, i, _ = w2.shape
    if num_primary_experts is not None:
        experts = min(experts, num_primary_experts)
    vmem_capacity_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
    num_buffers = _select_expert_buffers(m, h, i, experts, vmem_capacity_bytes)
    # Dense route coefficients are only [tokens, local experts], not expert
    # output tensors. Accumulation handles repeated IDs without reloading an
    # expert. Unroll the static top-k dimension to avoid the [M,K,E] reduction
    # whose layout copies can be offloaded to SparseCore and delay all-gather.
    expert_ids = jnp.arange(experts, dtype=jnp.int32)[None, :]
    weights = topk_weights.astype(jnp.float32)
    coeff = jnp.zeros((m, experts), jnp.float32)
    for route in range(local_ids.shape[1]):
        selected = local_ids[:, route:route + 1] == expert_ids
        coeff = coeff + selected.astype(jnp.float32) * weights[:,
                                                               route:route + 1]
    active = jnp.any(coeff != 0, axis=0).astype(jnp.int32)
    coeff = jnp.pad(coeff, ((0, 0), (0, (-experts) % 128)))
    out = pl.pallas_call(
        functools.partial(_kernel,
                          m=m,
                          h=h,
                          i=i,
                          experts=experts,
                          num_buffers=num_buffers),
        out_shape=jax.ShapeDtypeStruct((m, h), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
                *[pl.BlockSpec(memory_space=pltpu.HBM)] * 4,
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                pltpu.VMEM((num_buffers, h, 2 * i), jnp.float8_e4m3fn),
                pltpu.VMEM((num_buffers, i, h), jnp.float8_e4m3fn),
                pltpu.VMEM((num_buffers, 1, 2 * i), jnp.float32),
                pltpu.VMEM((num_buffers, 1, h), jnp.float32),
                pltpu.VMEM((m, h), jnp.float8_e4m3fn),
                pltpu.VMEM((h // 512, m, 128), jnp.bfloat16),
                pltpu.VMEM((m, i), jnp.bfloat16),
                pltpu.VMEM((m, i), jnp.float8_e4m3fn),
                pltpu.VMEM((i // 512, m, 128), jnp.bfloat16),
                pltpu.VMEM((m, h), jnp.float32),
                pltpu.SMEM((experts, ), jnp.int32),
                pltpu.SemaphoreType.DMA((num_buffers, 4)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_capacity_bytes),
        name=
        (f"dense_expert_moe-e_{experts}-m_{m}-h_{h}-i_{i}-buffers_{num_buffers}"
         ),
    )(active, hidden, coeff, w1, w2, s1, s2)
    return out[:m_orig]
