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
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.fused_conv1d_gdn import (configs, conv1d, gdn,
                                                    ref_classes)


def inner_kernel(
    p_id: jax.Array,
    act_buffered_ref: ref_classes.ActivationBufferedRefs,
    state_buffered_ref: ref_classes.StateBufferedRefs,
    sem_ref: jax.Array,
    metadata_ref: ref_classes.MetadataRef,
    weights_ref: ref_classes.WeightRefs,
    prev_qkv_scratch_ref: jax.Array,
    prev_recurrent_state_scratch_ref: jax.Array,
    cfgs: configs.GDNConfigs,
):
    b_start = p_id * cfgs.tile_size
    prev_b_start = b_start - cfgs.tile_size
    next_b_start = b_start + cfgs.tile_size

    recv_sem = sem_ref.at[0]
    send_sem = sem_ref.at[1]

    slot = p_id % 2
    other_slot = (slot + 1) % 2

    qkv_slot_ref = act_buffered_ref.qkv.get_slot_vmem(slot)
    b_slot_ref = act_buffered_ref.b.get_slot_vmem(slot)
    a_slot_ref = act_buffered_ref.a.get_slot_vmem(slot)
    out_slot_ref = act_buffered_ref.out.get_slot_vmem(slot)

    conv_state_slot_ref = state_buffered_ref.conv.get_slot_vmem(slot)
    recurrent_slot_ref = state_buffered_ref.recurrent.get_slot_vmem(slot)

    # Wait DMA read for current tile.
    act_buffered_ref.wait_in(b_start, slot, recv_sem)
    state_buffered_ref.wait_in(b_start, slot, recv_sem)

    # Wait DMA write for previous tile.
    act_buffered_ref.wait_out(prev_b_start, other_slot, send_sem)
    state_buffered_ref.wait_out(prev_b_start, other_slot, send_sem)

    # Start DMA read for next tile.
    act_buffered_ref.copy_in(next_b_start, other_slot, recv_sem)
    state_buffered_ref.copy_in(next_b_start, other_slot, recv_sem)

    # Step 1: Conv1D.
    qkv_in_compact = qkv_slot_ref[...].astype(jnp.float32)
    qkv_in_compact = qkv_in_compact.reshape(cfgs.tile_size, 1, cfgs.dim_size)

    prev_qkv_scratch = prev_qkv_scratch_ref[...]
    qkv_in_compact = jnp.concat([prev_qkv_scratch, qkv_in_compact], axis=0)
    prev_qkv_scratch_ref[...] = qkv_in_compact[-cfgs.prev_kernel_size:]

    conv_state = conv_state_slot_ref[...]
    conv_state = conv_state.astype(jnp.float32)
    conv_state = conv_state.reshape(cfgs.tile_size, cfgs.prev_kernel_size, 1,
                                    -1)

    qkv_out_compact, conv_state_out = conv1d.causal_conv1d(
        metadata_ref=metadata_ref,
        b_start=b_start,
        lhs=qkv_in_compact,
        states=conv_state,
        conv_weights_ref=weights_ref.conv,
        cfgs=cfgs,
    )

    # Store conv state output to vmem.
    conv_state_slot_ref[...] = conv_state_out

    # Apply activation function.
    qkv_out_compact = jax.nn.silu(qkv_out_compact)

    # Step 2: GDN.
    out, new_recurrent_state = gdn.recurrent_gdn(
        metadata_ref=metadata_ref,
        b_start=b_start,
        qkv=qkv_out_compact.astype(cfgs.dtypes.compute),
        b=b_slot_ref[...].reshape(cfgs.tile_size, 1, 1, -1),
        a=a_slot_ref[...].reshape(cfgs.tile_size, 1, 1, -1),
        recurrent_states=recurrent_slot_ref[...],
        prev_recurrent_state_ref=prev_recurrent_state_scratch_ref,
        gdn_weights_ref=weights_ref.gdn,
        cfgs=cfgs,
    )

    # Store output and recurrent to vmem.
    out_slot_ref[...] = out.astype(cfgs.act_dtype)
    recurrent_slot_ref[...] = new_recurrent_state.astype(
        cfgs.dtypes.recurrent_state)

    # Start DMA write for current tile.
    act_buffered_ref.copy_out(b_start, slot, send_sem)
    state_buffered_ref.copy_out(b_start, slot, send_sem)


def create_buffered_refs(
    metadata_ref: ref_classes.MetadataRef,
    qkv_ref: jax.Array,
    qkv_scratch_ref: jax.Array,
    b_ref: jax.Array,
    b_scratch_ref: jax.Array,
    a_ref: jax.Array,
    a_scratch_ref: jax.Array,
    out_ref: jax.Array,
    out_scratch_ref: jax.Array,
    conv_state_ref: jax.Array,
    conv_state_scratch_ref: jax.Array,
    recurrent_state_ref: jax.Array,
    recurrent_state_scratch_ref: jax.Array,
    cfgs: configs.GDNConfigs,
) -> tuple[ref_classes.ActivationBufferedRefs, ref_classes.StateBufferedRefs]:
    qkv_buffered_ref = ref_classes.QKVBufferedRef(
        hbm_ref=qkv_ref,
        vmem_ref=qkv_scratch_ref,
        metadata_ref=metadata_ref,
        cfgs=cfgs,
    )
    b_buffered_ref = ref_classes.BBufferedRef(
        hbm_ref=b_ref,
        vmem_ref=b_scratch_ref,
        metadata_ref=metadata_ref,
        cfgs=cfgs,
    )
    a_buffered_ref = ref_classes.ABufferedRef(
        hbm_ref=a_ref,
        vmem_ref=a_scratch_ref,
        metadata_ref=metadata_ref,
        cfgs=cfgs,
    )
    out_buffered_ref = ref_classes.OutBufferedRef(
        hbm_ref=out_ref,
        vmem_ref=out_scratch_ref,
        metadata_ref=metadata_ref,
        cfgs=cfgs,
    )
    act_buffered_ref = ref_classes.ActivationBufferedRefs(
        qkv=qkv_buffered_ref,
        b=b_buffered_ref,
        a=a_buffered_ref,
        out=out_buffered_ref,
    )

    conv_state_buffered_ref = ref_classes.ConvStateBufferedRef(
        hbm_ref=conv_state_ref,
        vmem_ref=conv_state_scratch_ref,
        metadata_ref=metadata_ref,
        cfgs=cfgs,
    )
    recurrent_state_buffered_ref = ref_classes.RecurrentStateBufferedRef(
        hbm_ref=recurrent_state_ref,
        vmem_ref=recurrent_state_scratch_ref,
        metadata_ref=metadata_ref,
        cfgs=cfgs,
    )
    state_buffered_ref = ref_classes.StateBufferedRefs(
        conv=conv_state_buffered_ref,
        recurrent=recurrent_state_buffered_ref,
    )

    return act_buffered_ref, state_buffered_ref


def main_kernel(
    # Inputs.
    metadata_ref: ref_classes.MetadataRef,
    qkv_ref: jax.Array,
    b_ref: jax.Array,
    a_ref: jax.Array,
    conv_state_ref: jax.Array,
    recurrent_state_ref: jax.Array,
    weights_ref: ref_classes.WeightRefs,
    # Outputs.
    out_ref: jax.Array,
    conv_state_out_ref: jax.Array,
    recurrent_state_out_ref: jax.Array,
    # Scratch
    qkv_scratch_ref: jax.Array,
    b_scratch_ref: jax.Array,
    a_scratch_ref: jax.Array,
    out_scratch_ref: jax.Array,
    conv_state_scratch_ref: jax.Array,
    recurrent_state_scratch_ref: jax.Array,
    prev_qkv_scratch_ref: jax.Array,
    prev_recurrent_state_scratch_ref: jax.Array,
    sem_ref: jax.Array,
    *,
    cfgs: configs.GDNConfigs,
):
    del conv_state_out_ref, recurrent_state_out_ref

    act_buffered_ref, state_buffered_ref = create_buffered_refs(
        metadata_ref=metadata_ref,
        qkv_ref=qkv_ref,
        qkv_scratch_ref=qkv_scratch_ref,
        b_ref=b_ref,
        b_scratch_ref=b_scratch_ref,
        a_ref=a_ref,
        a_scratch_ref=a_scratch_ref,
        out_ref=out_ref,
        out_scratch_ref=out_scratch_ref,
        conv_state_ref=conv_state_ref,
        conv_state_scratch_ref=conv_state_scratch_ref,
        recurrent_state_ref=recurrent_state_ref,
        recurrent_state_scratch_ref=recurrent_state_scratch_ref,
        cfgs=cfgs,
    )

    recv_sem = sem_ref.at[0]
    send_sem = sem_ref.at[1]

    # Prologue: Start DMA read ofr the first tile.
    act_buffered_ref.copy_in(0, 0, recv_sem)
    state_buffered_ref.copy_in(0, 0, recv_sem)

    num_tiles = metadata_ref.num_tiles[...]
    last_b_start = (num_tiles - 1) * cfgs.tile_size

    @pl.loop(0, num_tiles)
    def loop_wrapper(p_id):
        inner_kernel(
            p_id=p_id,
            act_buffered_ref=act_buffered_ref,
            state_buffered_ref=state_buffered_ref,
            weights_ref=weights_ref,
            sem_ref=sem_ref,
            metadata_ref=metadata_ref,
            prev_qkv_scratch_ref=prev_qkv_scratch_ref,
            prev_recurrent_state_scratch_ref=prev_recurrent_state_scratch_ref,
            cfgs=cfgs,
        )

    # Epilogue: Wait DMA write of the last tile.
    act_buffered_ref.wait_out(last_b_start, 0, send_sem)
    state_buffered_ref.wait_out(last_b_start, 0, send_sem)


def preprocess_metadata(
    cfgs: configs.GDNConfigs,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    max_seqs: jax.Array,
) -> ref_classes.MetadataRef:
    """Preprocesses metadata for the convolution kernel."""
    num_seqs = state_indices.size

    # Mask out padded locations.
    max_token = query_start_loc[max_seqs]
    all_seqs = jnp.arange(num_seqs + 1)
    query_start_loc = jnp.where(all_seqs <= max_seqs, query_start_loc,
                                max_token)

    # Map batch index to sequence index.
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    has_initial_state = (seq_lens - query_lens) > 0
    seqs = jnp.arange(num_seqs)
    b_idx_to_s_idx = jnp.repeat(seqs,
                                query_lens,
                                total_repeat_length=cfgs.batch_size)
    b_idx_query_start_loc = query_start_loc[b_idx_to_s_idx]
    all_b_idx = jnp.arange(cfgs.batch_size)
    b_idx_query_len = 1 + all_b_idx - b_idx_query_start_loc

    b_idx_to_sz_from_new = jnp.minimum(b_idx_query_len, cfgs.kernel_size)
    b_idx_to_sz_from_old = cfgs.kernel_size - b_idx_to_sz_from_new
    b_idx_to_sz_from_old = jnp.minimum(b_idx_to_sz_from_old,
                                       cfgs.kernel_size - 1)

    b_idx_should_write = all_b_idx == (query_start_loc[b_idx_to_s_idx + 1] - 1)
    b_idx_should_write = b_idx_should_write.astype(jnp.int32)

    num_tiles = pl.cdiv(max_token, cfgs.tile_size)

    return ref_classes.MetadataRef(
        num_tiles=num_tiles,
        b_idx_to_s_idx=b_idx_to_s_idx,
        b_idx_to_sz_from_old=b_idx_to_sz_from_old,
        b_idx_should_write=b_idx_should_write,
        s_idx_to_state_idx=state_indices,
        s_idx_has_initial_state=has_initial_state,
    )


@jax.jit(
    donate_argnames=("conv_state", "recurrent_state"),
    static_argnames=("n_kq", "n_v", "d_k", "d_v", "kernel_size"),
)
def fused_conv1d_gdn(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    a_log: jax.Array,
    dt_bias: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:

    # Step 1: Validate inputs.
    num_seqs = state_indices.size
    batch_size, dim = qkv.shape
    assert conv_weight.shape == (dim, 1, kernel_size)
    if conv_bias is not None:
        assert conv_bias.shape == (dim, )
    assert query_start_loc.shape == (num_seqs + 1, )
    assert state_indices.shape == (num_seqs, )
    assert distribution.shape == (3, )
    act_dtype = qkv.dtype
    assert a.dtype == b.dtype == qkv.dtype == act_dtype

    tile_size = 8
    tile_size = min(tile_size, batch_size)
    cfgs = configs.GDNConfigs(
        batch_size=batch_size,
        kernel_size=kernel_size,
        tile_size=tile_size,
        dim_size=dim,
        num_kq_heads=n_kq,
        num_v_heads=n_v,
        kq_head_dim=d_k,
        v_head_dim=d_v,
        dtypes=configs.Dtypes(
            act=act_dtype,
            compute=act_dtype,
            recurrent_state=recurrent_state.dtype,
            conv_state=conv_state.dtype,
        ),
    )

    padded_batch_size = cfgs.padded_batch_size
    batch_padding_size = padded_batch_size - batch_size
    num_v_padding_size = cfgs.aligned_num_v_heads - n_v
    packing = cfgs.act_packing
    qkv = jnp.pad(qkv, ((0, batch_padding_size), (0, 0)))
    b = jnp.pad(b, ((0, batch_padding_size), (0, num_v_padding_size)))
    a = jnp.pad(a, ((0, batch_padding_size), (0, num_v_padding_size)))
    a_log = jnp.pad(a_log, ((0, num_v_padding_size)))
    dt_bias = jnp.pad(dt_bias, ((0, num_v_padding_size)))

    qkv = qkv.reshape(padded_batch_size // packing, packing, -1)
    b = b.reshape(padded_batch_size // packing, packing, -1)
    a = a.reshape(padded_batch_size // packing, packing, -1)

    # Step 3: States and weights pre-processing.
    # TODO(kyuyeunk): To eliminate runtime cost, move this logic into model
    # loading stage.
    conv_state_shape = conv_state.shape
    conv_state_dtype = conv_state.dtype
    conv_state_in = conv_state.astype(jnp.float32)
    conv_state_in = conv_state_in.reshape(-1, kernel_size - 1, 1, dim)
    conv_weight = conv_weight.swapaxes(0, 2).astype(jnp.float32)
    conv_bias = conv_bias.astype(
        jnp.float32) if conv_bias is not None else None

    # Step 4: Metadata preprocessing. Will be executed multiple times per-layer
    # but will be CSEed by compiler.
    metadata = preprocess_metadata(
        cfgs=cfgs,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        max_seqs=distribution[-1],
    )

    # Step 5: Wrap inputs for the kernel.
    conv_weights = ref_classes.ConvWeightsRef(weight=conv_weight,
                                              bias=conv_bias)
    gdn_weights = ref_classes.GDNWeightsRef(a_log=a_log, dt_bias=dt_bias)
    weights = ref_classes.WeightRefs(conv=conv_weights, gdn=gdn_weights)

    # Step 6: Create specs.
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    metadata_spec = metadata.get_spec()
    conv_spec = ref_classes.ConvWeightsRef(
        weight=vmem_spec,
        bias=None if conv_bias is None else vmem_spec,
    )
    gdn_spec = ref_classes.GDNWeightsRef(
        a_log=vmem_spec,
        dt_bias=vmem_spec,
    )
    weights_spec = ref_classes.WeightRefs(
        conv=conv_spec,
        gdn=gdn_spec,
    )

    out, new_conv_state, new_recurrent_state = pl.pallas_call(
        functools.partial(main_kernel, cfgs=cfgs),
        out_shape=(
            jax.ShapeDtypeStruct(cfgs.get_out_shape(), cfgs.act_dtype),
            conv_state_in,
            recurrent_state,
        ),
        in_specs=(
            metadata_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            weights_spec,
        ),
        out_specs=(hbm_spec, hbm_spec, hbm_spec),
        scratch_shapes=cfgs.get_scratch_shape_dict(),
        input_output_aliases={
            len(metadata) + 3: 1,
            len(metadata) + 4: 2
        },
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
        ),
        name="fused_conv1d_gdn_kernel",
    )(metadata, qkv, b, a, conv_state_in, recurrent_state, weights)

    out = out.reshape(padded_batch_size, cfgs.v_dim_size)[:batch_size]

    new_conv_state = new_conv_state.astype(conv_state_dtype)
    new_conv_state = new_conv_state.reshape(conv_state_shape)

    return (new_conv_state, new_recurrent_state), out
