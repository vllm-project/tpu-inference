"""TPU-Friendly Fused Mixture of Experts (MoE) kernel."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import tpu as pltpu

P = jax.sharding.PartitionSpec

cdiv = pl.cdiv


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_packing(dtype):
    bits = dtypes.bit_width(dtype)
    return 32 // bits


def broadcast_minor(src, shape):
    if src.shape == shape:
        return src
    assert src.shape[:-1] == shape[:-1]
    assert src.shape[-1] % 128 == 0
    target_minor = align_to(shape[-1], src.shape[-1])
    # no-op concatenation.
    return jnp.concatenate([src for _ in range(target_minor // src.shape[-1])],
                           axis=-1)[..., :shape[-1]]


def ref_moe(
    tokens: jax.Array,  # (num_tokens, hidden_size)
    w1: jax.Array,  # (num_experts, 2, hidden_size, intermediate_size)
    w2: jax.Array,  # (num_experts, intermediate_size, hidden_size)
    gating_output: jax.Array,  # (num_tokens, num_experts)
    top_k: int,
    activation="silu",
):
    n_tokens = tokens.shape[0]  # num_tokens

    # Compute gating scores for all experts
    gating_logits = jax.nn.softmax(gating_output,
                                   axis=-1)  # [num_tokens, n_experts]

    # Select top-k experts per token
    top_k_logits, top_k_indices = lax.top_k(
        gating_logits, top_k)  # [num_tokens, top_k], [num_tokens, top_k]

    t_outputs = []

    # Process each token individually
    for i in range(n_tokens):
        curr_token = jnp.expand_dims(tokens[i], axis=0)  # [1, d_model]
        assigned_expert_ids = top_k_indices[
            i]  # [top_k] - indices of selected experts for token i
        tok_expert_act = []

        # Process each selected expert for the current token
        for expert_id in assigned_expert_ids:
            # Get expert weights
            expert_weight_1 = jnp.concat(
                [w1[expert_id, 0], w1[expert_id, 1]],
                axis=-1)  # [d_model, 2 * intermediate_size]
            expert_weight_2 = w2[expert_id]  # [intermediate_size, d_model]

            # First linear layer with SwiGLU activation
            gmm_1_out = curr_token @ expert_weight_1  # [1, 2 * intermediate_size]

            # Split into gate and up projections for SwiGLU
            gmm1_w1_proj, gmm1_w3_proj = jnp.split(
                gmm_1_out, 2,
                axis=-1)  # [1, intermediate_size], [1, intermediate_size]

            # Apply gated activation: activation(gate) * up
            if activation == "silu":
                act = jax.nn.silu(
                    gmm1_w1_proj) * gmm1_w3_proj  # [1, intermediate_size]
            elif activation == "gelu":
                act = jax.nn.gelu(
                    gmm1_w1_proj) * gmm1_w3_proj  # [1, intermediate_size]
            else:
                raise ValueError(
                    f"Unsupported activation: {activation}. Use 'silu' or 'gelu'."
                )

            # Second linear layer (down projection)
            gmm_2_out = act @ expert_weight_2  # [1, d_model]
            tok_expert_act.append(gmm_2_out)

        # Combine outputs from all selected experts
        experts_act = jnp.concatenate(tok_expert_act,
                                      axis=0)  # [top_k, d_model]

        # Weighted sum using top-k gating weights
        top_k_weights = top_k_logits[i]  # [top_k]
        top_k_weights = jnp.expand_dims(top_k_weights, axis=1)  # [top_k, 1]
        weighted_output = jnp.sum(experts_act * top_k_weights,
                                  axis=0,
                                  keepdims=True)  # [1, d_model]

        t_outputs.append(weighted_output)

    return jnp.concatenate(t_outputs, axis=0)  # [num_tokens, d_model]


def _fused_ep_moe_kernel(
        # Input
        tokens_hbm,  # (local_num_tokens, t_packing, hidden_size // t_packing)
        w1_hbm,  # (local_num_experts, 2, hidden_size, intermediate_size)
        w2_hbm,  # (local_num_experts, intermediate_size, hidden_size)
        gating_hbm,  # (local_num_tokens, padded_num_experts)
        a2a_g_hbm,  # (num_experts, bt, t_packing, hidden_size // t_packing)
        # Output
    output_hbm,  # (local_num_tokens, hidden_size)
        # Scratch
    t2e_routing_x2_smem,  # <bt_sem_id> (2, bt, padded_num_experts)
        d2e_count_x2_smem,  # <bt_sem_id> (2, num_devices, 1, padded_num_experts)
        expert_offsets_x2_smem,  # <bt_sem_id> (2, 2, padded_num_experts): for a2a_s and a2a_g
        expert_starts_x2_smem,  # <bt_sem_id> (2, 1, padded_num_experts)
        expert_sizes_x2_smem,  # <bt_sem_id> (2, 1, padded_num_experts)
        a2a_s_sends_x2_smem,  # <e_sem_id> (2,)
        a2a_s_x2_vmem,  # <e_sem_id> (2, bt * num_devices, t_packing, hidden_size // t_packing)
        a2a_s_acc_x2_vmem,  # <e_sem_id> (2, bt * num_devices, t_packing, hidden_size // t_packing)
        ### Accumulation for gathered tokens:
    a2a_g_acc_vmem,  # (top_k, bt, t_packing, hidden_size // t_packing)
        ### Expert weight double buffering:
    b_gating_x2_vmem,  # <bt_sem_id> (2, bt, padded_num_experts)
        b_output_x2_vmem,  # <bt_sem_id> (2, bt, hidden_size)
        b_w1_x2_vmem,  # <bw_sem_id> (2, t_packing, bd1 // t_packing, bf)
        b_w3_x2_vmem,  # <bw_sem_id> (2, t_packing, bd1 // t_packing, bf)
        b_w2_x2_vmem,  # <bw_sem_id> (2, t_packing, bf, bd2 // t_packing)
        b_acc_vmem,  # F32(bt * num_devices, 1, bf * 2)
        ### Semaphores:
    local_sems,  # (2, 5): 2 x [b_gating_sem, b_w1_sem, b_w2_sem, b_w3_sem, b_output_sem]
        send_sems,  # <e_sem_id> (2,)
        recv_sems,  # <e_sem_id> (2,)
        a2a_gather_sem,
        a2a_acc_sem,
        *,
        top_k: int,
        ep_axis_name: str,
        # Kernel tuning params.
        bt: int,  # Block size of local_num_tokens.
        bf: int,  # Block size of intermediate_size.
        bd1: int,  # Block size of hidden_size in w1.
        bd2: int,  # Block size of hidden_size in w2.
        btc: int,  # Compute size of block tokens for active expert.
        bfc: int,  # Compute size of block intermediate_size.
        bd1c: int,  # Compute size of block hidden_size.
        bd2c: int,  # Compute size of block hidden_size.
):
    my_id = lax.axis_index(ep_axis_name)
    num_devices = lax.axis_size(ep_axis_name)
    local_num_tokens = tokens_hbm.shape[0]
    local_num_experts, intermediate_size, hidden_size = w2_hbm.shape
    # num_experts = local_num_experts * num_devices
    # padded_num_experts = expert_starts_x2_smem.shape[-1]
    right_id = (my_id + 1) % num_devices

    t_dtype = tokens_hbm.dtype
    t_packing = get_dtype_packing(t_dtype)
    t_bitwidth = 32 // t_packing
    assert a2a_g_hbm.dtype == t_dtype
    assert w1_hbm.dtype == t_dtype
    assert w2_hbm.dtype == t_dtype

    h_per_packing = hidden_size // t_packing
    assert tokens_hbm.shape[-1] == h_per_packing
    bd1_per_packing = bd1 // t_packing
    bd2_per_packing = bd2 // t_packing
    bd1c_per_packing = bd1c // t_packing
    bd2c_per_packing = bd2c // t_packing

    num_bt = cdiv(local_num_tokens, bt)
    num_bf = cdiv(intermediate_size, bf)
    num_bd1 = cdiv(hidden_size, bd1)
    num_bd2 = cdiv(hidden_size, bd2)

    def sync_barrier():
        barrier_sem = pltpu.get_barrier_semaphore()
        pltpu.semaphore_signal(
            barrier_sem,
            device_id=(0, right_id),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        pltpu.semaphore_wait(barrier_sem, 1)

    def start_fetch_b_gating(bt_id, priority=0):
        is_valid = jnp.logical_and(0 <= bt_id, bt_id < num_bt)
        sz = pl.multiple_of(lax.select(is_valid, bt, 0), bt)
        bt_sem_id = (bt_id + 2) % 2
        b_gating_sem = local_sems.at[bt_sem_id, 0]
        pltpu.make_async_copy(
            src_ref=gating_hbm.at[pl.ds(bt_id * bt, sz)],
            dst_ref=b_gating_x2_vmem.at[bt_sem_id, pl.ds(0, sz)],
            sem=b_gating_sem,
        ).start(priority=priority)

    def wait_fetch_b_gating(bt_id):
        bt_sem_id = bt_id % 2
        b_gating_sem = local_sems.at[bt_sem_id, 0]
        pltpu.make_async_copy(
            src_ref=b_gating_x2_vmem.at[bt_sem_id],
            dst_ref=b_gating_x2_vmem.at[bt_sem_id],
            sem=b_gating_sem,
        ).wait()

    def get_top_k(input, top_k):
        assert len(input.shape) == 2, input.shape
        input = input.astype(jnp.float32)
        top_k_logits_lst = []
        top_k_indices_lst = []
        t2e = jnp.zeros(input.shape, dtype=jnp.int32)
        t2e_routing = jnp.zeros(input.shape, dtype=jnp.int32)
        iota = jax.lax.broadcasted_iota(jnp.int32, input.shape, 1)
        for k_id in range(top_k):
            # TODO(jevinjiang): return both top_k values and indices in op in Mosaic
            top_k_logits = jnp.broadcast_to(
                jnp.max(input, axis=1, keepdims=True),
                (input.shape[0], 128)).astype(input.dtype)
            top_k_logits_lst.append(top_k_logits)
            # TODO(jevinjiang): support bf16 argmax in Mosaic
            top_k_indices = jnp.broadcast_to(
                jnp.argmax(input, axis=1, keepdims=True), input.shape)
            top_k_indices_lst.append(top_k_indices)
            t2e_routing = jnp.where(iota == k_id, top_k_indices, t2e_routing)
            mask = iota == top_k_indices
            t2e += mask.astype(jnp.int32)
            if k_id != top_k - 1:
                input = jnp.where(mask, -jnp.inf, input)

        expert_sizes = jnp.sum(t2e, axis=0, keepdims=True)
        expert_starts = jnp.zeros_like(expert_sizes)
        return top_k_logits_lst, t2e_routing, expert_sizes, expert_starts

    def all_reduce_metadata(bt_sem_id, t2e_routing, starts, sizes):
        send_sem = send_sems.at[0]
        recv_sem = recv_sems.at[0]

        # All-reduce to accumulate starts and sizes and transfer to SMEM.
        def _all_reduce_metadata(
            t2e_routing_vmem,
            d2e_count_vmem,
            offsets_vmem,
            starts_vmem,
            sizes_vmem,
        ):
            offsets_vmem[...] = jnp.zeros_like(offsets_vmem)
            # TODO(jevinjiang): check how slow is VMEM -> SMEM.
            offsets_copy = pltpu.async_copy(
                src_ref=offsets_vmem,
                dst_ref=expert_offsets_x2_smem.at[bt_sem_id],
                sem=send_sem,
            )
            t2e_routing_vmem[...] = t2e_routing
            t2e_routing_copy = pltpu.async_copy(
                src_ref=t2e_routing_vmem,
                dst_ref=t2e_routing_x2_smem.at[bt_sem_id],
                sem=send_sem,
            )
            reduced_sizes = sizes
            reduced_starts = starts
            row_id = my_id
            d2e_count_vmem[row_id] = sizes
            for i in range(num_devices - 1):
                sync_barrier()
                # TODO(jevinjiang): we can use double buffering to improve AR if needed.
                pltpu.async_remote_copy(
                    src_ref=d2e_count_vmem.at[row_id],
                    dst_ref=d2e_count_vmem.at[row_id],
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=(0, right_id),
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).wait()
                row_id = (row_id + num_devices - 1) % num_devices
                new_sizes = d2e_count_vmem[row_id]
                reduced_sizes += new_sizes
                reduced_starts += lax.select(my_id > i, new_sizes,
                                             jnp.zeros_like(new_sizes))
            starts_vmem[...] = reduced_starts
            sizes_vmem[...] = reduced_sizes

            starts_copy = pltpu.async_copy(
                src_ref=starts_vmem,
                dst_ref=expert_starts_x2_smem.at[bt_sem_id],
                sem=send_sem,
            )
            sizes_copy = pltpu.async_copy(
                src_ref=sizes_vmem,
                dst_ref=expert_sizes_x2_smem.at[bt_sem_id],
                sem=send_sem,
            )

            # TODO(jevinjiang): if d2e_count is too big, we can store in HBM and fetch
            # to SMEM partially.
            d2e_count_copy = pltpu.async_copy(
                src_ref=d2e_count_vmem,
                dst_ref=d2e_count_x2_smem.at[bt_sem_id],
                sem=send_sem,
            )

            t2e_routing_copy.wait()
            d2e_count_copy.wait()
            offsets_copy.wait()
            starts_copy.wait()
            sizes_copy.wait()

        pl.run_scoped(
            _all_reduce_metadata,
            pltpu.VMEM(t2e_routing_x2_smem.shape[1:],
                       t2e_routing_x2_smem.dtype),
            pltpu.VMEM(d2e_count_x2_smem.shape[1:], d2e_count_x2_smem.dtype),
            pltpu.VMEM(expert_offsets_x2_smem.shape[1:],
                       expert_offsets_x2_smem.dtype),
            pltpu.VMEM(expert_starts_x2_smem.shape[1:],
                       expert_starts_x2_smem.dtype),
            pltpu.VMEM(expert_sizes_x2_smem.shape[1:],
                       expert_sizes_x2_smem.dtype),
        )

    def start_a2a_scatter(bt_id, e_sem_id, local_e_id):
        bt_sem_id = bt_id % 2

        # Counting the number of remote sends from the current device.
        send_sz = 0
        for bt_t_id in range(bt):
            for k_id in range(top_k):
                e_id = t2e_routing_x2_smem[bt_sem_id, bt_t_id, k_id]
                is_active_expert = e_id % local_num_experts == local_e_id
                recv_id = e_id // local_num_experts
                offset = expert_offsets_x2_smem[bt_sem_id, 0, e_id]
                sz = lax.select(is_active_expert, 1, 0)
                is_local = recv_id == my_id
                local_sz = lax.select(is_local, sz, 0)
                remote_sz = lax.select(is_local, 0, sz)
                send_sz += remote_sz
                expert_offsets_x2_smem[bt_sem_id, 0,
                                       e_id] = (offset + local_sz + remote_sz)
                start = expert_starts_x2_smem[bt_sem_id, 0, e_id] + offset
                t_id = bt * bt_id + bt_t_id
                # TODO(jevinjiang): compare the perf when using branches.
                pltpu.make_async_copy(
                    src_ref=tokens_hbm.at[pl.ds(t_id, local_sz)],
                    dst_ref=a2a_s_x2_vmem.at[e_sem_id,
                                             pl.ds(start, local_sz)],
                    sem=recv_sems.at[e_sem_id],
                ).start()
                pltpu.make_async_remote_copy(
                    src_ref=tokens_hbm.at[pl.ds(t_id, remote_sz)],
                    dst_ref=a2a_s_x2_vmem.at[e_sem_id,
                                             pl.ds(start, remote_sz)],
                    send_sem=send_sems.at[e_sem_id],
                    recv_sem=recv_sems.at[e_sem_id],
                    device_id=(
                        0,
                        recv_id,
                    ),
                ).start()
        a2a_s_sends_x2_smem[e_sem_id] = send_sz

    def wait_a2a_scatter_recv(bt_id, e_sem_id, local_e_id):
        bt_sem_id = bt_id % 2
        e_id = my_id * local_num_experts + local_e_id
        sz = expert_sizes_x2_smem[bt_sem_id, 0, e_id]
        pltpu.make_async_copy(
            src_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(0, sz)],
            dst_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(0, sz)],
            sem=recv_sems.at[e_sem_id],
        ).wait()

    def wait_a2a_scatter_send(bt_id, e_sem_id, local_e_id):
        del bt_id, local_e_id
        sz = a2a_s_sends_x2_smem[e_sem_id]
        pltpu.make_async_copy(
            src_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(0, sz)],
            dst_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(0, sz)],
            sem=send_sems.at[e_sem_id],
        ).wait()

    def start_a2a_gather(bt_id, e_sem_id, local_e_id):
        my_e_id = my_id * local_num_experts + local_e_id
        bt_sem_id = bt_id % 2
        start = 0
        for recv_id in range(num_devices):
            sz = d2e_count_x2_smem[bt_sem_id, recv_id, 0, my_e_id]
            is_local = recv_id == my_id
            local_sz = lax.select(is_local, sz, 0)
            remote_sz = lax.select(is_local, 0, sz)
            pltpu.make_async_copy(
                src_ref=a2a_s_acc_x2_vmem.at[e_sem_id,
                                             pl.ds(start, local_sz)],
                dst_ref=a2a_g_hbm.at[my_e_id, pl.ds(0, local_sz)],
                sem=a2a_gather_sem,
            ).start()
            pltpu.make_async_remote_copy(
                src_ref=a2a_s_acc_x2_vmem.at[e_sem_id,
                                             pl.ds(start, remote_sz)],
                dst_ref=a2a_g_hbm.at[my_e_id, pl.ds(0, remote_sz)],
                send_sem=send_sems.at[e_sem_id],
                recv_sem=a2a_gather_sem,
                device_id=(0, recv_id),
            ).start()
            start += sz

    def wait_a2a_gather_send(bt_id, e_sem_id, local_e_id):
        my_e_id = my_id * local_num_experts + local_e_id
        bt_sem_id = bt_id % 2
        sz = expert_sizes_x2_smem[bt_sem_id, 0, my_e_id]
        local_sz = d2e_count_x2_smem[bt_sem_id, my_id, 0, my_e_id]
        remote_sz = sz - local_sz
        is_valid = jnp.logical_and(0 <= local_e_id, local_e_id
                                   < local_num_experts)
        remote_sz = lax.select(is_valid, remote_sz, 0)
        pltpu.make_async_copy(
            src_ref=a2a_g_hbm.at[0, pl.ds(0, remote_sz)],
            dst_ref=a2a_g_hbm.at[0, pl.ds(0, remote_sz)],
            sem=send_sems.at[e_sem_id],
        ).wait()

    def wait_a2a_gather_recv_all():
        sz = top_k * bt
        pltpu.make_async_copy(
            src_ref=a2a_g_hbm.at[0, pl.ds(0, sz)],
            dst_ref=a2a_g_hbm.at[0, pl.ds(0, sz)],
            sem=a2a_gather_sem,
        ).wait()

    def start_fetch_bw1(local_e_id, bw1_sem_id, bf_id, bd1_id):
        for p in range(t_packing):
            offset = p * h_per_packing + bd1_id * bd1_per_packing
            pltpu.make_async_copy(
                src_ref=w1_hbm.at[
                    local_e_id,
                    0,
                    pl.ds(offset, bd1_per_packing),
                    pl.ds(bf_id * bf, bf),
                ],
                dst_ref=b_w1_x2_vmem.at[bw1_sem_id, p],
                sem=local_sems.at[bw1_sem_id, 1],
            ).start()

    def start_fetch_bw2(local_e_id, bw2_sem_id, bf_id, bd2_id):
        for p in range(t_packing):
            offset = p * h_per_packing + bd2_id * bd2_per_packing
            pltpu.make_async_copy(
                src_ref=w2_hbm.at[
                    local_e_id,
                    pl.ds(bf_id * bf, bf),
                    pl.ds(offset, bd2_per_packing),
                ],
                dst_ref=b_w2_x2_vmem.at[bw2_sem_id, p],
                sem=local_sems.at[bw2_sem_id, 2],
            ).start()

    def start_fetch_bw3(local_e_id, bw3_sem_id, bf_id, bd3_id):
        for p in range(t_packing):
            offset = p * h_per_packing + bd3_id * bd1_per_packing
            pltpu.make_async_copy(
                src_ref=w1_hbm.at[
                    local_e_id,
                    1,
                    pl.ds(offset, bd1_per_packing),
                    pl.ds(bf_id * bf, bf),
                ],
                dst_ref=b_w3_x2_vmem.at[bw3_sem_id, p],
                sem=local_sems.at[bw3_sem_id, 3],
            ).start()

    def wait_fetch_bw1(local_e_id, bw1_sem_id, bf_id, bd1_id):
        del local_e_id, bf_id, bd1_id
        pltpu.make_async_copy(
            src_ref=b_w1_x2_vmem.at[bw1_sem_id],
            dst_ref=b_w1_x2_vmem.at[bw1_sem_id],
            sem=local_sems.at[bw1_sem_id, 1],
        ).wait()

    def wait_fetch_bw2(local_e_id, bw2_sem_id, bf_id, bd2_id):
        del local_e_id, bf_id, bd2_id
        pltpu.make_async_copy(
            src_ref=b_w2_x2_vmem.at[bw2_sem_id],
            dst_ref=b_w2_x2_vmem.at[bw2_sem_id],
            sem=local_sems.at[bw2_sem_id, 2],
        ).wait()

    def wait_fetch_bw3(local_e_id, bw3_sem_id, bf_id, bd3_id):
        del local_e_id, bf_id, bd3_id
        pltpu.make_async_copy(
            src_ref=b_w3_x2_vmem.at[bw3_sem_id],
            dst_ref=b_w3_x2_vmem.at[bw3_sem_id],
            sem=local_sems.at[bw3_sem_id, 3],
        ).wait()

    def start_fetch_next_bw(local_e_id, bw_sem_id, bf_id, bd1_id, bd2_id):
        next_bd1_id = bd1_id + 1
        next_bd2_id = bd2_id + 1
        next_sem_id = (bw_sem_id + 1) % 2

        if bf_id >= num_bf:
            return
        if next_bd1_id < num_bd1:
            start_fetch_bw1(local_e_id, next_sem_id, bf_id, next_bd1_id)
            start_fetch_bw3(local_e_id, next_sem_id, bf_id, next_bd1_id)
        elif next_bd1_id == num_bd1:
            start_fetch_bw2(local_e_id, next_sem_id, bf_id, 0)
        elif next_bd2_id < num_bd2:
            start_fetch_bw2(local_e_id, next_sem_id, bf_id, next_bd2_id)
        elif next_bd2_id == num_bd2:
            start_fetch_next_bw(local_e_id, bw_sem_id, bf_id + 1, -1, -1)
        else:
            raise RuntimeError("Unreachable")

    def dynamic_ffn1(
        t_b32_vmem,
        w1_vmem,
        w3_vmem,
        acc1_vmem,
        acc3_vmem,
        dyn_sz,
        should_init,
    ):
        assert t_b32_vmem.shape == (bt * num_devices, bd1 // t_packing)
        assert w1_vmem.shape == w3_vmem.shape == (t_packing, bd1_per_packing,
                                                  bf)
        assert acc1_vmem.shape == acc3_vmem.shape == (bt * num_devices, bf)
        assert bd1 % (t_packing * 128) == 0, (bd1, t_packing)
        assert bd1c % (t_packing * 128) == 0, (bd1c, t_packing)

        num_loops = cdiv(dyn_sz, btc)
        repack_ty = jnp.dtype(f"int{t_bitwidth}")

        def body(btc_id, _):
            for bd1c_id in range(cdiv(bd1, bd1c)):
                t_b32 = t_b32_vmem[
                    pl.ds(btc_id * btc, btc),
                    pl.ds(bd1c_id * bd1c_per_packing, bd1c_per_packing),
                ]
                for p_id in range(t_packing):
                    t = pltpu.bitcast(t_b32.astype(repack_ty), t_dtype)
                    t_b32 = t_b32 >> t_bitwidth
                    for bfc_id in range(cdiv(bf, bfc)):
                        w_slices = (
                            p_id,
                            pl.ds(bd1c_id * bd1c_per_packing,
                                  bd1c_per_packing),
                            pl.ds(bfc_id * bfc, bfc),
                        )
                        w1 = w1_vmem[*w_slices]
                        acc1 = jnp.dot(t,
                                       w1,
                                       preferred_element_type=jnp.float32)
                        w3 = w3_vmem[*w_slices]
                        acc3 = jnp.dot(t,
                                       w3,
                                       preferred_element_type=jnp.float32)
                        acc_slices = (pl.ds(btc_id * btc,
                                            btc), pl.ds(bfc_id * bfc, bfc))
                        if should_init and p_id == bd1c_id == 0:
                            acc1_vmem[*acc_slices] = acc1
                            acc3_vmem[*acc_slices] = acc3
                        else:
                            acc1_vmem[*acc_slices] += acc1
                            acc3_vmem[*acc_slices] += acc3

        lax.fori_loop(0, num_loops, body, None)

    def dynamic_ffn2(
        acc1_vmem,
        acc3_vmem,
        w2_vmem,
        res_b32_vmem,
        dyn_sz,
        should_init,
    ):
        assert res_b32_vmem.shape == (bt * num_devices, bd2_per_packing)
        assert w2_vmem.shape == (t_packing, bf, bd2_per_packing), (
            w2_vmem.shape,
            t_packing,
            bf,
            bd2_per_packing,
        )
        assert acc1_vmem.shape == acc3_vmem.shape == (bt * num_devices, bf)
        assert bd2 % (t_packing * 128) == 0, (bd2, t_packing)
        assert bd2c % (t_packing * 128) == 0, (bd2c, t_packing)
        assert t_dtype in (jnp.float32, jnp.bfloat16)

        num_loops = cdiv(dyn_sz, btc)
        assert bd2c % (t_packing * 128) == 0, (bd2c, t_packing)

        def body(btc_id, _):
            for bd2c_id in range(cdiv(bd2, bd2c)):
                res_lst = []
                for p_id in range(t_packing):
                    res = jnp.zeros((btc, bd2c_per_packing), dtype=jnp.float32)
                    for bfc_id in range(cdiv(bf, bfc)):
                        acc_slices = (pl.ds(btc_id * btc,
                                            btc), pl.ds(bfc_id * bfc, bfc))
                        acc1 = acc1_vmem[*acc_slices]
                        acc3 = acc3_vmem[*acc_slices]
                        act = jax.nn.silu(acc1) * acc3
                        w2 = w2_vmem[
                            p_id,
                            pl.ds(bfc_id * bfc, bfc),
                            pl.ds(bd2c_id *
                                  bd2c_per_packing, bd2c_per_packing),
                        ]
                        res += jnp.dot(act,
                                       w2,
                                       preferred_element_type=jnp.float32)
                    res = pltpu.bitcast(res, jnp.uint32)
                    if t_packing == 2:
                        res = res >> 16 << (16 * p_id)
                    else:
                        assert t_packing == 1
                    res_lst.append(res)
                res = res_lst[0]
                # TODO(jevinjiang): use interleaved packing when it is exposed to Pallas
                for i in range(1, t_packing):
                    res |= res_lst[i]
                sliced_res_vmem = res_b32_vmem.at[
                    pl.ds(btc_id * btc, btc),
                    pl.ds(bd2c_id * bd2c_per_packing, bd2c_per_packing),
                ]
                if should_init:
                    sliced_res_vmem[...] = res
                else:
                    sliced_res_vmem[...] = pltpu.bitcast(
                        sliced_res_vmem.bitcast(t_dtype)[...] +
                        pltpu.bitcast(res, t_dtype),
                        sliced_res_vmem.dtype,
                    )

        lax.fori_loop(0, num_loops, body, None)

    def expert_ffn(bt_id, e_sem_id, local_e_id):
        bt_sem_id = bt_id % 2
        bw_sem_id = 0
        # start_fetch_bw1(local_e_id, bw_sem_id, 0, 0)
        # start_fetch_bw3(local_e_id, bw_sem_id, 0, 0)
        a2a_s_b32_vmem = (a2a_s_x2_vmem.bitcast(jnp.uint32).reshape(
            2, bt * num_devices, hidden_size // t_packing).at[e_sem_id])
        a2a_s_acc_b32_vmem = (a2a_s_acc_x2_vmem.bitcast(jnp.uint32).reshape(
            2, bt * num_devices, hidden_size // t_packing).at[e_sem_id])
        b_acc_vmem_2d = b_acc_vmem.reshape(bt * num_devices, bf * 2)
        b_acc1_vmem = b_acc_vmem_2d.at[:, :bf]
        b_acc3_vmem = b_acc_vmem_2d.at[:, bf:]

        e_id = my_id * local_num_experts + local_e_id
        dyn_sz = expert_sizes_x2_smem[bt_sem_id, 0, e_id]

        bd1_per_packing = bd1 // t_packing
        bd2_per_packing = bd2 // t_packing

        for bf_id in range(num_bf):
            for bd1_id in range(num_bd1):
                start_fetch_next_bw(local_e_id, bw_sem_id, bf_id, bd1_id, 0)
                wait_fetch_bw1(local_e_id, bw_sem_id, bf_id, bd1_id)
                wait_fetch_bw3(local_e_id, bw_sem_id, bf_id, bd1_id)

                dynamic_ffn1(
                    t_b32_vmem=a2a_s_b32_vmem.at[
                        ...,
                        pl.ds(bd1_id * bd1_per_packing, bd1_per_packing)],
                    w1_vmem=b_w1_x2_vmem.at[bw_sem_id],
                    w3_vmem=b_w3_x2_vmem.at[bw_sem_id],
                    acc1_vmem=b_acc1_vmem,
                    acc3_vmem=b_acc3_vmem,
                    dyn_sz=dyn_sz,
                    should_init=(bd1_id == 0),
                )
                bw_sem_id = (bw_sem_id + 1) % 2

            for bd2_id in range(num_bd2):
                start_fetch_next_bw(local_e_id, bw_sem_id, bf_id, num_bd1,
                                    bd2_id)
                wait_fetch_bw2(local_e_id, bw_sem_id, bf_id, bd2_id)
                if bf_id == bd2_id == 0:
                    wait_a2a_gather_send(bt_id, e_sem_id, local_e_id - 2)

                dynamic_ffn2(
                    acc1_vmem=b_acc1_vmem,
                    acc3_vmem=b_acc3_vmem,
                    w2_vmem=b_w2_x2_vmem.at[bw_sem_id],
                    res_b32_vmem=a2a_s_acc_b32_vmem.at[
                        ...,
                        pl.ds(bd2_id * bd2_per_packing, bd2_per_packing)],
                    dyn_sz=dyn_sz,
                    should_init=(bf_id == 0),
                )
                bw_sem_id = (bw_sem_id + 1) % 2

    def bt_acc(bt_id, top_k_logits_lst):
        bt_sem_id = bt_id % 2
        for bt_t_id in range(bt):
            for k_id in range(top_k):
                e_id = t2e_routing_x2_smem[bt_sem_id, bt_t_id, k_id]
                offset = expert_offsets_x2_smem[bt_sem_id, 1, e_id]
                expert_offsets_x2_smem[bt_sem_id, 1, e_id] = offset + 1
                pltpu.make_async_copy(
                    src_ref=a2a_g_hbm.at[e_id, pl.ds(offset, 1)],
                    dst_ref=a2a_g_acc_vmem.at[k_id, pl.ds(bt_t_id, 1)],
                    sem=a2a_acc_sem,
                ).start()
        pltpu.make_async_copy(
            src_ref=a2a_g_acc_vmem,
            dst_ref=a2a_g_acc_vmem,
            sem=a2a_acc_sem,
        ).wait()
        output = None
        for k_id in range(top_k):
            acc = a2a_g_acc_vmem[k_id].reshape(bt, hidden_size)
            logits = broadcast_minor(top_k_logits_lst[k_id], acc.shape)
            acc *= logits
            if output is None:
                output = acc
            else:
                output += acc
        assert output is not None
        return output.astype(output_hbm.dtype)

    def start_send_bo(bt_id, priority=0):
        bt_sem_id = bt_id % 2
        b_output_sem = local_sems.at[bt_sem_id, 4]
        pltpu.make_async_copy(
            src_ref=b_output_x2_vmem.at[bt_sem_id],
            dst_ref=output_hbm.at[pl.ds(bt_id * bt, bt)],
            sem=b_output_sem,
        ).start(priority=priority)

    def wait_send_bo(bt_id):
        is_valid = jnp.logical_and(0 <= bt_id, bt_id < num_bt)
        sz = pl.multiple_of(lax.select(is_valid, bt, 0), bt)
        bt_sem_id = (bt_id + 2) % 2
        b_output_sem = local_sems.at[bt_sem_id, 4]
        pltpu.make_async_copy(
            src_ref=output_hbm.at[pl.ds(0, sz)],
            dst_ref=output_hbm.at[pl.ds(0, sz)],
            sem=b_output_sem,
        ).wait()

    ### ------- Kernel start ------- ###
    start_fetch_b_gating(bt_id=0)

    def run_per_bt(bt_id, e_sem_id):
        bt_sem_id = bt_id % 2
        next_bt_id = bt_id + 1
        start_fetch_b_gating(next_bt_id)
        wait_fetch_b_gating(bt_id)

        b_gating = b_gating_x2_vmem[bt_sem_id]
        b_gating_score = jax.nn.softmax(b_gating, axis=-1)
        top_k_logits_lst, t2e_routing, expert_sizes, expert_starts = get_top_k(
            b_gating_score, top_k)

        all_reduce_metadata(bt_sem_id, t2e_routing, expert_starts,
                            expert_sizes)

        start_a2a_scatter(bt_id=bt_id, e_sem_id=e_sem_id, local_e_id=0)

        def run_per_expert(local_e_id, e_sem_id):
            sync_barrier()
            next_e_sem_id = lax.select(e_sem_id == 0, 1, 0)
            next_local_e_id = local_e_id + 1

            @pl.when(next_local_e_id < local_num_experts)
            def _():
                start_a2a_scatter(bt_id, next_e_sem_id, next_local_e_id)

            # Prefetch weights for active expert.
            start_fetch_bw1(local_e_id, bw1_sem_id=0, bf_id=0, bd1_id=0)
            start_fetch_bw3(local_e_id, bw3_sem_id=0, bf_id=0, bd3_id=0)

            # Wait for a2a scatter and perform FFN for active expert.
            wait_a2a_scatter_recv(bt_id, e_sem_id, local_e_id)
            expert_ffn(bt_id, e_sem_id, local_e_id)

            # Wait for a2a gather to send back tokens for active expert.
            start_a2a_gather(bt_id, e_sem_id, local_e_id)

            # A must-wait before next sync_barrier.
            wait_a2a_scatter_send(bt_id, e_sem_id, local_e_id)
            return next_e_sem_id

        e_sem_id = lax.fori_loop(0,
                                 local_num_experts,
                                 run_per_expert,
                                 e_sem_id,
                                 unroll=False)

        wait_a2a_gather_recv_all()
        output = bt_acc(bt_id, top_k_logits_lst)

        # Make sure it is safe to overwrite output buffer.
        wait_send_bo(bt_id=bt_id - 2)
        b_output_x2_vmem[bt_sem_id] = output

        start_send_bo(bt_id)

        wait_a2a_gather_send(
            bt_id,
            e_sem_id=e_sem_id,
            local_e_id=local_num_experts - 2,
        )
        wait_a2a_gather_send(
            bt_id,
            e_sem_id=lax.select(e_sem_id == 0, 1, 0),
            local_e_id=local_num_experts - 1,
        )
        return e_sem_id

    lax.fori_loop(0, num_bt, run_per_bt, 0, unroll=False)
    wait_send_bo(bt_id=num_bt - 2)
    wait_send_bo(bt_id=num_bt - 1)

    ### ------- Kernel end ------- ###


@functools.partial(
    jax.jit,
    static_argnames=[
        "mesh",
        "top_k",
        "bt",
        "bf",
        "bd1",
        "bd2",
        "btc",
        "bfc",
        "bd1c",
        "bd2c",
        "ep_axis_name",
    ],
)
def fused_ep_moe(
    mesh: jax.sharding.Mesh,
    tokens: jax.Array,  # (num_tokens, hidden_size)
    w1: jax.Array,  # (num_experts, 2, hidden_size, intermediate_size)
    w2: jax.Array,  # (num_experts, intermediate_size, hidden_size)
    gating_output: jax.Array,  # (num_tokens, num_experts)
    top_k: int,
    *,
    # Kernel tuning parameters.
    bt: int,
    bf: int,
    bd1: int,
    bd2: int,
    btc: int,
    bfc: int,
    bd1c: int,
    bd2c: int,
    ep_axis_name: str = 'model',
):
    # Assert all other axes have length of 1
    assert len(mesh.shape) == 2, "Expect 2D mesh in tpu-inference"
    assert 'data' in mesh.shape and mesh.shape['data'] == 1, \
        "Expect data axis size of 1 in tpu-inference"

    ep_size = mesh.shape[ep_axis_name]
    num_devices = ep_size

    num_tokens, actual_hidden_size = tokens.shape
    num_experts, intermediate_size, _ = w2.shape

    assert num_tokens % ep_size == 0
    assert num_experts % ep_size == 0

    local_num_tokens = num_tokens // ep_size
    # local_num_experts = num_experts // ep_size
    padded_num_experts = align_to(num_experts, 128)

    t_dtype = tokens.dtype
    t_packing = get_dtype_packing(t_dtype)
    hidden_size = align_to(actual_hidden_size, 128 * t_packing)
    if hidden_size != actual_hidden_size:
        tokens = jnp.pad(
            tokens,
            ((0, 0), (0, hidden_size - actual_hidden_size)),
            constant_values=0,
        )
    tokens = tokens.reshape(-1, t_packing, hidden_size // t_packing)
    bt = min(bt, local_num_tokens)
    bf = min(bf, intermediate_size)
    bd1 = min(bd1, hidden_size)
    bd2 = min(bd2, hidden_size)

    btc = min(btc, bt * num_devices)
    bfc = min(bfc, bf)
    bd1c = min(bd1c, bd1)
    bd2c = min(bd2c, bd2)
    assert bfc % 128 == 0
    assert bd1c % (t_packing * 128) == 0
    assert bd2c % (t_packing * 128) == 0
    assert bf % bfc == 0
    assert bd1 % bd1c == 0
    assert bd2 % bd2c == 0

    if padded_num_experts != gating_output.shape[-1]:
        gating_output = jnp.pad(
            gating_output,
            ((0, 0), (0, padded_num_experts - gating_output.shape[-1])),
            constant_values=-jnp.inf,
        )

    scope_name = f"fused_moe_k-{top_k}_bt-{bt}-{btc}_bf-{bf}-{bfc}_bd1-{bd1}-{bd1c}_bd2-{bd2}-{bd2c}"
    fused_moe = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                _fused_ep_moe_kernel,
                top_k=top_k,
                ep_axis_name=ep_axis_name,
                bt=bt,
                bf=bf,
                bd1=bd1,
                bd2=bd2,
                btc=btc,
                bfc=bfc,
                bd1c=bd1c,
                bd2c=bd2c,
            ),
            out_shape=jax.ShapeDtypeStruct((local_num_tokens, hidden_size),
                                           t_dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                scratch_shapes=([
                    # t2e_routing_x2_smem
                    pltpu.SMEM((2, bt, padded_num_experts), jnp.int32),
                    # d2e_count_x2_smem
                    pltpu.SMEM((2, num_devices, 1, padded_num_experts),
                               jnp.int32),
                    # expert_offsets_x2_smem
                    pltpu.SMEM((2, 2, padded_num_experts), jnp.int32),
                    # expert_starts_x2_smem
                    pltpu.SMEM((2, 1, padded_num_experts), jnp.int32),
                    # expert_sizes_x2_smem
                    pltpu.SMEM((2, 1, padded_num_experts), jnp.int32),
                    # a2a_s_sends_x2_smem
                    pltpu.SMEM((2, ), jnp.int32),
                    # a2a_s_x2_vmem
                    pltpu.VMEM(
                        (
                            2,
                            bt * num_devices,
                            t_packing,
                            hidden_size // t_packing,
                        ),
                        t_dtype,
                    ),
                    # a2a_s_acc_x2_vmem
                    pltpu.VMEM(
                        (
                            2,
                            bt * num_devices,
                            t_packing,
                            hidden_size // t_packing,
                        ),
                        t_dtype,
                    ),
                    # a2a_g_acc_vmem
                    pltpu.VMEM(
                        (top_k, bt, t_packing, hidden_size // t_packing),
                        t_dtype),
                    # b_gating_x2_vmem
                    pltpu.VMEM((2, bt, padded_num_experts), t_dtype),
                    # b_output_x2_vmem
                    pltpu.VMEM((2, bt, hidden_size), t_dtype),
                    # b_w1_x2_vmem
                    pltpu.VMEM((2, t_packing, bd1 // t_packing, bf), w1.dtype),
                    # b_w3_x2_vmem
                    pltpu.VMEM((2, t_packing, bd1 // t_packing, bf), w1.dtype),
                    # b_w2_x2_vmem
                    pltpu.VMEM((2, t_packing, bf, bd2 // t_packing), w2.dtype),
                    # b_acc_vmem
                    pltpu.VMEM((bt * num_devices, 1, bf * 2), jnp.float32),
                    # local_sems
                    pltpu.SemaphoreType.DMA((2, 5)),
                    # send_sems
                    pltpu.SemaphoreType.DMA((2, )),
                    # recv_sems
                    pltpu.SemaphoreType.DMA((2, )),
                    # a2a_gather_sem
                    pltpu.SemaphoreType.DMA,
                    # a2a_acc_sem
                    pltpu.SemaphoreType.DMA,
                ]),
            ),
            compiler_params=pltpu.CompilerParams(
                collective_id=0,
                vmem_limit_bytes=100 * 1024 * 1024,
            ),
            name=scope_name,
        ))

    @jax.jit
    @functools.partial(
        shard_map.shard_map,
        mesh=mesh,
        in_specs=(P(ep_axis_name), P(ep_axis_name), P(ep_axis_name),
                  P(ep_axis_name), P()),
        out_specs=P(ep_axis_name),
        check_rep=False,
    )
    def kernel(tokens, w1, w2, gating_output, a2a_g_hbm_scratch):
        return fused_moe(
            pltpu.with_memory_space_constraint(tokens, pltpu.HBM),
            pltpu.with_memory_space_constraint(w1, pltpu.HBM),
            pltpu.with_memory_space_constraint(w2, pltpu.HBM),
            pltpu.with_memory_space_constraint(gating_output, pltpu.HBM),
            pltpu.with_memory_space_constraint(a2a_g_hbm_scratch, pltpu.HBM),
        )

    a2a_g_hbm_scratch = pl.empty(
        (num_experts, bt, t_packing, hidden_size // t_packing), t_dtype)
    results = kernel(
        tokens,
        w1,
        w2,
        gating_output,
        a2a_g_hbm_scratch,
    )
    return results[:, :actual_hidden_size]
