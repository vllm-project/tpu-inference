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
                                                    ldst_helper, ref_classes)


@jax.named_scope("inner_kernel")
def inner_kernel(
    st_idx: jax.Array,
    act_buffered_ref: ref_classes.ActivationBufferedRefs,
    state_buffered_ref: ref_classes.StateBufferedRefs,
    sem_ref: jax.Array,
    metadata_ref: ref_classes.MetadataRef,
    weights_ref: ref_classes.WeightRefs,
    prev_qkv_scratch_ref: jax.Array | None,
    prev_recurrent_state_scratch_ref: jax.Array | None,
    cfgs: configs.GDNConfigs,
):
    prev_st_idx = st_idx - 1
    next_st_idx = st_idx + 1

    recv_sem = sem_ref.at[0]
    send_sem = sem_ref.at[1]

    slot = st_idx % 2
    other_slot = (slot + 1) % 2

    qkv_slot_ref = act_buffered_ref.qkv.get_slot_vmem(slot)
    b_slot_ref = act_buffered_ref.b.get_slot_vmem(slot)
    a_slot_ref = act_buffered_ref.a.get_slot_vmem(slot)
    out_slot_ref = act_buffered_ref.out.get_slot_vmem(slot)

    conv_state_slot_ref = state_buffered_ref.conv.get_slot_vmem(slot)
    recurrent_slot_ref = state_buffered_ref.recurrent.get_slot_vmem(slot)

    # Wait DMA read for current tile.
    act_buffered_ref.wait_in(st_idx, slot, recv_sem)
    state_buffered_ref.wait_in(st_idx, slot, recv_sem)

    # Wait DMA write for previous tile.
    act_buffered_ref.wait_out(prev_st_idx, other_slot, send_sem)
    state_buffered_ref.wait_out(prev_st_idx, other_slot, send_sem)

    # Start DMA read for next tile.
    act_buffered_ref.copy_in(next_st_idx, other_slot, recv_sem)
    state_buffered_ref.copy_in(next_st_idx, other_slot, recv_sem)

    # Prepare states.
    real_size, prev_conv, prev_recurrent = ldst_helper.load_and_mask_states(
        metadata_ref=metadata_ref,
        st_idx=st_idx,
        conv_state_slot_ref=conv_state_slot_ref,
        recurrent_slot_ref=recurrent_slot_ref,
        prev_qkv_scratch_ref=prev_qkv_scratch_ref,
        prev_recurrent_state_scratch_ref=prev_recurrent_state_scratch_ref,
        cfgs=cfgs,
    )

    # Step 1: Conv1D.
    # NOTE: Conv1D requires performing sliding window where inputs are slided
    # across rows. If typical 2D layout was used, multiple rows are stored in a
    # single register which necessitate costly shuffling for every sliding.
    # Therefore, it is extremely important to leverage compact layout that
    # ensures 1 register only stores data from 1 row.
    qkv_in_compact = qkv_slot_ref[...].astype(jnp.float32)
    qkv_in_compact = jnp.concat([prev_conv, qkv_in_compact], axis=1)

    target_val_list = []
    for s_idx in range(cfgs.seq_tile_size):
        target_s = qkv_in_compact[s_idx, 1:cfgs.kernel_size]
        for row_start in range(2, cfgs.tok_tile_size + 1):
            row_end = row_start + cfgs.prev_kernel_size
            target_s = jnp.where(
                row_start == real_size[s_idx],
                qkv_in_compact[s_idx, row_start:row_end],
                target_s,
            )
        target_val_list.append(target_s)
    target_val = jnp.stack(target_val_list, axis=0)
    conv_state_slot_ref[...] = target_val
    if prev_qkv_scratch_ref is not None:
        prev_qkv_scratch_ref[...] = target_val

    qkv_out_compact = conv1d.causal_conv1d(
        lhs=qkv_in_compact,
        conv_weights_ref=weights_ref.conv,
        cfgs=cfgs,
    )

    # Apply activation function.
    qkv_out_compact = jax.nn.silu(qkv_out_compact)

    # Step 2: GDN.
    if cfgs.tok_tile_size == 1:
        q_compact, k_compact, v_compact, b_compact, a_compact = (
            ldst_helper.load_activation_as_compact(
                real_size=real_size,
                qkv_vreg=qkv_out_compact,
                qkv_vmem_ref=qkv_slot_ref,
                b_vmem_ref=b_slot_ref,
                a_vmem_ref=a_slot_ref,
                cfgs=cfgs,
            ))

        out, new_recurrent_state = gdn.recurrent_gdn(
            q_compact=q_compact,
            k_compact=k_compact,
            v_compact=v_compact,
            b_compact=b_compact,
            a_compact=a_compact,
            state_prev=prev_recurrent,
            gdn_weights_ref=weights_ref.gdn,
            cfgs=cfgs,
        )

    else:
        q_large, k_large, v_large, b_large, a_large = (
            ldst_helper.load_activation_as_large(
                real_size=real_size,
                qkv_vreg=qkv_out_compact,
                qkv_vmem_ref=qkv_slot_ref,
                b_vmem_ref=b_slot_ref,
                a_vmem_ref=a_slot_ref,
                cfgs=cfgs,
            ))

        out, new_recurrent_state = gdn.chunked_gdn(
            q_large=q_large,
            k_large=k_large,
            v_large=v_large,
            b_large=b_large,
            a_large=a_large,
            state_prev=prev_recurrent,
            gdn_weights_ref=weights_ref.gdn,
            cfgs=cfgs,
        )

    # Store output and recurrent to vmem.
    out_slot_ref[...] = out.astype(out_slot_ref.dtype)
    recurrent_slot_ref[...] = new_recurrent_state.astype(
        recurrent_slot_ref.dtype)

    if prev_recurrent_state_scratch_ref is not None:
        prev_recurrent_state_scratch_ref[...] = new_recurrent_state

    # Start DMA write for current tile.
    act_buffered_ref.copy_out(st_idx, slot, send_sem)
    state_buffered_ref.copy_out(st_idx, slot, send_sem)


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
    qkv_buffered_ref = ref_classes.InBufferedRef(
        hbm_ref=qkv_ref,
        vmem_ref=qkv_scratch_ref,
        metadata_ref=metadata_ref,
        cfgs=cfgs,
    )
    b_buffered_ref = ref_classes.InBufferedRef(
        hbm_ref=b_ref,
        vmem_ref=b_scratch_ref,
        metadata_ref=metadata_ref,
        cfgs=cfgs,
    )
    a_buffered_ref = ref_classes.InBufferedRef(
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

    conv_state_buffered_ref = ref_classes.SharedStateBufferedRef(
        hbm_ref=conv_state_ref,
        vmem_ref=conv_state_scratch_ref,
        metadata_ref=metadata_ref,
        cfgs=cfgs,
    )
    recurrent_state_buffered_ref = ref_classes.SharedStateBufferedRef(
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
    _: jax.Array,
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
    prev_qkv_scratch_ref: jax.Array | None,
    prev_recurrent_state_scratch_ref: jax.Array | None,
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
    start_st_idx = metadata_ref.start_st_idx[...]
    act_buffered_ref.copy_in(start_st_idx, 0, recv_sem)
    state_buffered_ref.copy_in(start_st_idx, 0, recv_sem)

    num_tiles = metadata_ref.num_tiles[...]
    last_st_idx = num_tiles - 1

    @pl.loop(start_st_idx, num_tiles)
    def loop_wrapper(p_id):
        inner_kernel(
            st_idx=p_id,
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
    act_buffered_ref.wait_out(last_st_idx, 0, send_sem)
    state_buffered_ref.wait_out(last_st_idx, 0, send_sem)


def preprocess_decode_metadata(
    cfgs: configs.GDNConfigs,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    last_seqs: jax.Array,
) -> ref_classes.MetadataRef:
    """Preprocesses metadata for the convolution kernel."""
    max_seqs = state_indices.size

    # Mask out padded locations.
    last_token = query_start_loc[last_seqs]
    query_start_loc = jnp.where(
        jnp.arange(max_seqs + 1) <= last_seqs, query_start_loc, last_token)
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    has_initial_state = (seq_lens - query_lens) > 0
    all_valid_st_idx = query_lens > 0

    return ref_classes.MetadataRef.create(
        start_st_idx=lambda _: 0,
        num_tiles=pl.cdiv(last_seqs, cfgs.tile_size),
        st_idx_to_t_idx=lambda _: 0,
        st_idx_to_s_idx=lambda indices: indices,
        st_idx_to_b_idx=lambda indices: indices,
        st_idx_to_b_size=lambda _: 1,
        st_idx_is_first_tile=all_valid_st_idx,
        st_idx_is_last_tile=all_valid_st_idx,
        s_idx_to_num_tiles=query_lens,
        s_idx_has_initial_state=has_initial_state,
        s_idx_to_state_indices=state_indices,
    )


def preprocess_mixed_metadata(
    cfgs: configs.GDNConfigs,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    start_seqs: jax.Array,
    last_seqs: jax.Array,
) -> ref_classes.MetadataRef:
    """Preprocesses metadata for the convolution kernel."""
    max_seqs = state_indices.size

    # Mask out padded locations.
    max_tokens = cfgs.batch_size
    last_token = query_start_loc[last_seqs]
    query_start_loc = jnp.where(
        jnp.arange(max_seqs + 1) <= last_seqs, query_start_loc, last_token)

    # Map batch index to sequence index.
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    has_initial_state = (seq_lens - query_lens) > 0
    s_idx_to_num_tiles = pl.cdiv(query_lens, cfgs.tok_tile_size)
    s_idx_to_start_st_idx = jnp.cumulative_sum(s_idx_to_num_tiles,
                                               include_initial=True)
    start_st_idx = s_idx_to_start_st_idx[start_seqs]
    st_idx_to_s_idx = jnp.repeat(jnp.arange(max_seqs),
                                 s_idx_to_num_tiles,
                                 total_repeat_length=max_tokens)
    all_st = jnp.arange(max_tokens)
    st_idx_to_t_idx = all_st - s_idx_to_start_st_idx[st_idx_to_s_idx]

    st_idx_to_b_idx = (query_start_loc[st_idx_to_s_idx] +
                       st_idx_to_t_idx * cfgs.tok_tile_size)
    st_idx_to_b_size = jnp.minimum(
        cfgs.tile_size,
        query_lens[st_idx_to_s_idx] - st_idx_to_t_idx * cfgs.tok_tile_size,
    )

    st_idx_is_first_tile = st_idx_to_t_idx == 0
    st_idx_is_last_tile = st_idx_to_t_idx == (
        s_idx_to_num_tiles[st_idx_to_s_idx] - 1)

    return ref_classes.MetadataRef.create(
        start_st_idx=start_st_idx,
        num_tiles=s_idx_to_num_tiles.sum(),
        st_idx_to_t_idx=st_idx_to_t_idx,
        st_idx_to_s_idx=st_idx_to_s_idx,
        st_idx_to_b_idx=st_idx_to_b_idx,
        st_idx_to_b_size=st_idx_to_b_size,
        st_idx_is_first_tile=st_idx_is_first_tile,
        st_idx_is_last_tile=st_idx_is_last_tile,
        s_idx_to_num_tiles=s_idx_to_num_tiles,
        s_idx_has_initial_state=has_initial_state,
        s_idx_to_state_indices=state_indices,
    )


@jax.jit(
    donate_argnames=("conv_state", "recurrent_state"),
    static_argnames=(
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "kernel_size",
        "decode_tile_size",
        "mixed_tile_size",
    ),
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
    *,
    decode_tile_size: int = 8,
    mixed_tile_size: int = 64,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    # TODO(kyuyeunk): Support bf16
    orig_act_dtype = qkv.dtype
    orig_conv_state_dtype = conv_state.dtype
    orig_recurrent_state_dtype = recurrent_state.dtype
    qkv = qkv.astype(jnp.float32)
    b = b.astype(jnp.float32)
    a = a.astype(jnp.float32)
    conv_state = conv_state.astype(jnp.float32)

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

    num_lanes = pltpu.get_tpu_info().num_lanes
    packing = 4 // act_dtype.itemsize
    padded_batch_size = pl.cdiv(batch_size, packing) * packing
    decode_tile_size = min(decode_tile_size, padded_batch_size)
    mixed_tile_size = min(mixed_tile_size, padded_batch_size)
    aligned_num_v_heads = pl.cdiv(n_v, num_lanes) * num_lanes

    batch_padding_size = padded_batch_size - batch_size
    num_v_padding_size = aligned_num_v_heads - n_v
    qkv = jnp.pad(qkv, ((0, batch_padding_size), (0, 0)))
    b = jnp.pad(b, ((0, batch_padding_size), (0, num_v_padding_size)))
    a = jnp.pad(a, ((0, batch_padding_size), (0, num_v_padding_size)))
    a_log = jnp.pad(a_log, ((0, num_v_padding_size)))
    dt_bias = jnp.pad(dt_bias, ((0, num_v_padding_size)))

    qkv = qkv.reshape(padded_batch_size, 1, -1)
    b = b.reshape(padded_batch_size, 1, -1)
    a = a.reshape(padded_batch_size, 1, -1)

    # Step 3: States and weights pre-processing.
    # TODO(kyuyeunk): To eliminate runtime cost, move this logic into model
    # loading stage.
    conv_state_shape = conv_state.shape
    conv_state = conv_state.reshape(-1, kernel_size - 1, 1, dim)
    conv_weight = conv_weight.swapaxes(0, 2).astype(jnp.float32)
    conv_bias = conv_bias.astype(
        jnp.float32) if conv_bias is not None else None

    # Step 4: Wrap inputs for the kernel.
    conv_weights = ref_classes.ConvWeightsRef(weight=conv_weight,
                                              bias=conv_bias)
    gdn_weights = ref_classes.GDNWeightsRef(a_log=a_log, dt_bias=dt_bias)
    weights = ref_classes.WeightRefs(conv=conv_weights, gdn=gdn_weights)

    # Step 5: Create specs.
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    weights_spec = jax.tree.map(lambda _: vmem_spec, weights)

    def call_kernel(
        in_conv_state: jax.Array,
        in_recurrent_state: jax.Array,
        in_act: jax.Array | None,
        mode: configs.GDNMode,
    ):
        if mode == configs.GDNMode.DECODE:
            tile_size = decode_tile_size
        else:
            tile_size = mixed_tile_size

        cfgs = configs.GDNConfigs(
            mode=mode,
            batch_size=padded_batch_size,
            kernel_size=kernel_size,
            tile_size=tile_size,
            dim_size=dim,
            num_kq_heads=n_kq,
            num_v_heads=n_v,
            kq_head_dim=d_k,
            v_head_dim=d_v,
            dtypes=configs.Dtypes(
                act=jnp.dtype(act_dtype),
                compute=jnp.dtype(jnp.bfloat16),
                recurrent_state=jnp.dtype(in_recurrent_state.dtype),
                conv_state=jnp.dtype(in_conv_state.dtype),
            ),
        )

        # Step 6: Metadata preprocessing. Will be executed multiple times per-layer
        # but will be CSEed by compiler.
        if mode == configs.GDNMode.DECODE:
            metadata = preprocess_decode_metadata(
                cfgs=cfgs,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                state_indices=state_indices,
                last_seqs=distribution[0],
            )
        else:
            metadata = preprocess_mixed_metadata(
                cfgs=cfgs,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                state_indices=state_indices,
                start_seqs=distribution[0],
                last_seqs=distribution[-1],
            )

        metadata_spec = jax.tree.map(lambda _: smem_spec, metadata)

        # Step 7: Handle case where write needs to be done in existing out.
        in_out_spec = None
        input_output_aliases = {len(metadata) + 3: 1, len(metadata) + 4: 2}
        if in_act is not None:
            in_out_spec = hbm_spec
            input_output_aliases[len(metadata) + 5] = 0

        return pl.pallas_call(
            functools.partial(main_kernel, cfgs=cfgs),
            out_shape=(
                cfgs.get_out_shape(),
                in_conv_state,
                in_recurrent_state,
            ),
            in_specs=(
                metadata_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                in_out_spec,
                weights_spec,
            ),
            out_specs=(hbm_spec, hbm_spec, hbm_spec),
            scratch_shapes=cfgs.get_scratch_shape_dict(),
            input_output_aliases=input_output_aliases,
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
            ),
            name=cfgs.get_kernel_name(),
            metadata=cfgs.get_metadata(),
        )(metadata, qkv, b, a, in_conv_state, in_recurrent_state, in_act,
          weights)

    out_act, out_conv_state, out_recurrent_state = call_kernel(
        conv_state, recurrent_state, None, configs.GDNMode.DECODE)
    out_act, out_conv_state, out_recurrent_state = call_kernel(
        out_conv_state, out_recurrent_state, out_act, configs.GDNMode.MIXED)

    out_act = out_act.reshape(padded_batch_size, -1)[:batch_size]
    out_act = out_act.astype(orig_act_dtype)
    out_conv_state = out_conv_state.astype(orig_conv_state_dtype)
    out_conv_state = out_conv_state.reshape(conv_state_shape)
    out_recurrent_state = out_recurrent_state.astype(
        orig_recurrent_state_dtype)

    return (out_conv_state, out_recurrent_state), out_act
