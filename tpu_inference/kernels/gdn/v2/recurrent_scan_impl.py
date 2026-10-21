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
"""Helper classes and processor implementation for Recurrent Scan Pallas kernel."""

# pylint: disable=invalid-name

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def l2_normalize(x, eps=1e-6):
    rnorm = jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)
    return x * rnorm


# 1. Dataclasses for holding references to inputs/outputs and shared data.
# These are passed as arguments to the processor classes.


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BranchRefs:
    """Inputs/Outputs specific to a single execution branch (prefill or decode)."""

    qkv: Any
    a_raw: Any
    b_raw: Any
    output: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SharedRefs:
    """Inputs/Outputs refs shared between both branches(prefill and decode)."""

    a_log: Any
    dt_bias: Any
    recurrent_state_in: Any
    recurrent_state_out: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PrefillScratchRefs:
    """Scratch VMEM and semaphores allocated for prefill."""

    scratch: Any
    semaphore: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DecodeScratchRefs:
    """Scratch VMEM and semaphores allocated for decode."""

    state: Any
    load: Any
    store: Any
    output: Any
    read_semaphores: Any
    write_semaphore: Any


# 2.  Configuration Dataclasses (static)


@dataclasses.dataclass(frozen=True)
class ModelDims:
    """Dimensions of the Recurrent Scan model configuration."""

    n_kq: int
    n_v: int
    d_k: int
    d_v: int

    @property
    def key_dim(self) -> int:
        return self.n_kq * self.d_k

    @property
    def repeat_factor(self) -> int:
        return self.n_v // self.n_kq


@dataclasses.dataclass(frozen=True)
class TilingConfig:
    """Tiling dimensions for memory copy blocks."""

    C: int
    BT: int
    sublanesize: int


@dataclasses.dataclass(frozen=True)
class ScanConfig:
    """Configuration holding model dimensions and tiling options."""

    model: ModelDims
    tiling: TilingConfig
    use_qk_norm_in_gdn: bool
    decode_tokens: int


# 3. DMA Helper


class DMAHelper:
    """Manages asynchronous state copies and double-buffering semaphores."""

    def __init__(self, state_in, state_out, commit_scratch, semaphore):
        self.state_in = state_in
        self.state_out = state_out
        self.commit_scratch = commit_scratch
        self.sem = semaphore
        # double or n buffering
        self.has_multiple_slots = commit_scratch.shape[0] > 1

    def build_copy_in(self, slot: int, state_idx: int):
        target_slot = slot if self.has_multiple_slots else 0
        return pltpu.make_async_copy(
            src_ref=self.state_in.at[pl.ds(state_idx, 1)],
            dst_ref=self.commit_scratch.at[pl.ds(target_slot, 1)],
            sem=self.sem.at[slot],
        )

    def commit_in(self, copy_op, slot: int, dst_ref, dst_slot: int):
        target_slot = slot if self.has_multiple_slots else 0
        copy_op.wait()
        dst_ref[dst_slot] = self.commit_scratch[target_slot].astype(
            dst_ref.dtype)

    def copy_out(self, slot: int, state_idx: int, src_scratch):
        target_slot = slot if self.has_multiple_slots else 0
        self.commit_scratch[target_slot] = src_scratch.astype(
            self.commit_scratch.dtype)
        copy_op = pltpu.make_async_copy(
            src_ref=self.commit_scratch.at[pl.ds(target_slot, 1)],
            dst_ref=self.state_out.at[pl.ds(state_idx, 1)],
            sem=self.sem.at[slot],
        )
        copy_op.start()
        return copy_op

    def wait_out(self, slot: int, state_idx: int):
        target_slot = slot if self.has_multiple_slots else 0
        copy_op = pltpu.make_async_copy(
            src_ref=self.commit_scratch.at[pl.ds(target_slot, 1)],
            dst_ref=self.state_out.at[pl.ds(state_idx, 1)],
            sem=self.sem.at[slot],
        )
        copy_op.wait()


# 4. Schedule step helper
COL_PREFILL_VALID = 0
COL_PREFILL_OFFSET = 1
COL_PREFILL_REQ_ID = 2
COL_PREFILL_COUNT = 3
COL_DECODE_VALID = 4
COL_DECODE_OFFSET = 5
COL_DECODE_REQ_ID = 6
COL_DECODE_COUNT = 7
COL_IS_LAST_CHUNK = 8
COL_IS_FIRST_CHUNK = 9
COL_IS_TRANSITION = 10
COL_SUBLANE_REQ_IDS = 11


class ScheduleStep:
    """Unpacks and holds the scheduling metadata for the current step."""

    def __init__(self, schedule_table, step):
        self.step = step
        self.schedule_table = schedule_table
        self.prefill_valid = schedule_table[step, COL_PREFILL_VALID][...]
        self.prefill_offset = schedule_table[step, COL_PREFILL_OFFSET][...]
        self.prefill_req_id = schedule_table[step, COL_PREFILL_REQ_ID][...]
        self.prefill_count = schedule_table[step, COL_PREFILL_COUNT][...]

        self.decode_valid = schedule_table[step, COL_DECODE_VALID][...]
        self.decode_offset = schedule_table[step, COL_DECODE_OFFSET][...]
        self.decode_req_id = schedule_table[step, COL_DECODE_REQ_ID][...]
        self.decode_count = schedule_table[step, COL_DECODE_COUNT][...]

        self.is_last_chunk = schedule_table[step, COL_IS_LAST_CHUNK][...]
        self.is_first_chunk = schedule_table[step, COL_IS_FIRST_CHUNK][...]
        self.is_transition = schedule_table[step, COL_IS_TRANSITION][...]


# 4. Base Processor Class


class ScanProcessor:
    """Base class for executing step calculations."""

    def __init__(
        self,
        config: ScanConfig,
        schedule: ScheduleStep,
        state_indices,
        has_initial_state,
    ):
        self.cfg = config
        self.schedule = schedule
        self.state_indices = state_indices
        self.has_initial_state = has_initial_state


def invert_triangular_matrix(A, block_size=16):
    """Inverts a unit lower triangular matrix A block-wise.

  Args:
    A: Unit lower triangular matrix of shape (B, N, N).
    block_size: Size of the blocks for Gaussian elimination.

  Returns:
    Inverse of A, of shape (B, N, N).
  """
    B, N, _ = A.shape
    num_blocks = N // block_size

    def local_forward_sub(A_mat, b_mat):
        x_list = []
        for i in range(block_size):
            b_i = b_mat[:, i, :]
            if i == 0:
                x_i = b_i
            else:
                stacked_x = jnp.stack(x_list, axis=1)
                all_prev_A = A_mat[:, i, :i]
                prev_sum = jnp.sum(all_prev_A[..., None] * stacked_x, axis=1)
                x_i = b_i - prev_sum
            x_list.append(x_i)
        return jnp.stack(x_list, axis=1)

    x_blocks = []
    for i in range(num_blocks):
        start, end = i * block_size, (i + 1) * block_size
        e_block = jnp.eye(N, dtype=A.dtype)[start:end, :]
        e_block = jnp.broadcast_to(e_block, (B, block_size, N))

        if i == 0:
            target_b = e_block
        else:
            interaction_A = A[:, start:end, :start]
            solved_x = jnp.concatenate(x_blocks, axis=1)
            prev_sum = jnp.matmul(interaction_A,
                                  solved_x,
                                  precision=jax.lax.Precision.HIGHEST)
            target_b = e_block - prev_sum

        local_A = A[:, start:end, start:end]
        x_block = local_forward_sub(local_A, target_b)
        x_blocks.append(x_block)

    return jnp.concatenate(x_blocks, axis=1)


class PrefillProcessor(ScanProcessor):
    """Handles prefill step processing."""

    def __init__(
        self,
        config: ScanConfig,
        schedule: ScheduleStep,
        state_indices,
        has_initial_state,
        refs: BranchRefs,
        shared: SharedRefs,
        scratch: PrefillScratchRefs,
        dma: DMAHelper,
    ):
        super().__init__(config, schedule, state_indices, has_initial_state)
        self.refs = refs
        self.shared = shared
        self.scratch = scratch
        self.dma = dma

    def process(self):
        is_trans = self.schedule.is_transition > 0
        jax.lax.cond(
            is_trans,
            lambda _: self._process_transition_prefill(),
            lambda _: self._process_regular_prefill(),
            operand=None,
        )

    def _process_regular_prefill(self):
        """Processes a regular prefill step without transition boundary overlaps."""
        prefill_req_id = self.schedule.prefill_req_id
        prefill_slot = prefill_req_id % 2
        init_has_init = self.has_initial_state[prefill_req_id][...]
        init_state_idx = self.state_indices[prefill_req_id][...]
        should_load_init = (self.schedule.is_first_chunk > 0) & (init_has_init
                                                                 > 0)
        should_zero_init = (self.schedule.is_first_chunk > 0) & (init_has_init
                                                                 == 0)

        init_copy_op = self.dma.build_copy_in(prefill_slot, init_state_idx)

        @pl.when(should_load_init)
        def _start_init_load():
            init_copy_op.start()

        @pl.when(should_zero_init)
        def _zero_init_state():
            self.scratch.scratch[prefill_slot] = jnp.zeros(
                (self.cfg.model.n_v, self.cfg.model.d_k, self.cfg.model.d_v),
                dtype=self.scratch.scratch.dtype,
            )

        key_dim = self.cfg.model.key_dim
        n_v = self.cfg.model.n_v
        d_k = self.cfg.model.d_k
        d_v = self.cfg.model.d_v
        n_kq = self.cfg.model.n_kq
        C = self.cfg.tiling.C

        qkv_chunk = self.refs.qkv[...]
        qkv_chunk = jax.nn.silu(qkv_chunk)
        q = qkv_chunk[:, :key_dim]
        k = qkv_chunk[:, key_dim:2 * key_dim]
        v = qkv_chunk[:, 2 * key_dim:]

        a_raw_chunk = self.refs.a_raw[...]
        b_raw_chunk = self.refs.b_raw[...]

        a_raw_processed_T = a_raw_chunk[:, :n_v].T
        b_raw_processed_T = b_raw_chunk[:, :n_v].T

        beta_T = jax.nn.sigmoid(b_raw_processed_T)
        g_T = -jnp.exp(
            # in jax 10.0.1 we can avoid the cast to float32,
            # jax.errors.JaxRuntimeError: INTERNAL: Mosaic failed to compile TPU kernel: failed to legalize operation
            # 'math.log1p': %7302 = "math.log1p"(%7295) <{fastmath =
            # #arith.fastmath<none>}> : (vector<8x128x2xbf16>) -> vector<8x128x2xbf16>
            self.shared.a_log[...])[:, None] * jax.nn.softplus(
                # same issue with the cast here
                a_raw_processed_T + self.shared.dt_bias[...][:, None])
        g_T = jnp.maximum(g_T, -100.0)

        prefill_count = self.schedule.prefill_count
        mask_float = (jnp.arange(C) < prefill_count).astype(q.dtype)
        q = jnp.where(mask_float[:, None] > 0, q, 0.0)
        k = jnp.where(mask_float[:, None] > 0, k, 0.0)
        g_T = jnp.where(mask_float[None, :] > 0, g_T, 0.0)
        v = jnp.where(mask_float[:, None] > 0, v, 0.0)
        beta_T = jnp.where(mask_float[None, :] > 0, beta_T, 0.0)

        q = q.reshape(C, n_kq, d_k)
        k = k.reshape(C, n_kq, d_k)
        v = v.reshape(C, n_v, d_v)

        if self.cfg.use_qk_norm_in_gdn:
            q = l2_normalize(q)
            k = l2_normalize(k)

        # Note: fusing transpose with (vmatpush.xpose) made it slower,
        # This has better instruction pipelining
        q_T = q.transpose(1, 0, 2)  # (n_kq, C, d_k)
        k_T = k.transpose(1, 0, 2)  # (n_kq, C, d_k)
        v_T = v.transpose(1, 0, 2)  # (n_v, C, d_v)

        repeat_factor = self.cfg.model.repeat_factor
        if repeat_factor > 1:
            q_T = jnp.repeat(q_T, repeat_factor, axis=0)
            k_T = jnp.repeat(k_T, repeat_factor, axis=0)

        scale = d_k**-0.5
        q_T = q_T * scale

        g_cumsum_list = []
        current_sum = jnp.zeros((n_v, ), dtype=jnp.float32)
        for i in range(C):
            current_sum = current_sum + g_T[:, i]
            g_cumsum_list.append(current_sum)
        g_cumsum_T = jnp.stack(g_cumsum_list, axis=1)  # shape (n_v, C)
        exp_g = jnp.exp(g_cumsum_T)[..., None]  # Precomputed for reuse
        k_beta = k_T * beta_T[..., None]

        # Concatenate along sequence dimension: (n_v, 2 * C, d_k)
        kbeta_q = jnp.concatenate([k_beta, q_T], axis=1)
        # Batch is n_v (axis 0), contract is d_k (axis 2).
        # Output shape: (n_v, 2 * C, C)
        S_both = jax.lax.dot_general(
            kbeta_q,
            k_T,
            (((2, ), (2, )), ((0, ), (0, ))),
            preferred_element_type=jnp.float32,
        )
        S = S_both[:, :C, :]
        S_q = S_both[:, C:, :]

        g_diff = g_cumsum_T[..., :, None] - g_cumsum_T[..., None, :]
        i_idx = jnp.arange(C)[:, None]
        j_idx = jnp.arange(C)[None, :]
        mask_float = (i_idx > j_idx).astype(jnp.float32)

        g_diff_safe = jnp.minimum(g_diff, 0.0)
        S = jnp.where(mask_float[None, :, :] > 0, S * jnp.exp(g_diff_safe),
                      0.0)

        mask_float_q = (i_idx >= j_idx).astype(jnp.float32)
        g_diff_Sq = g_diff_safe * mask_float_q[None, ...] + (
            1.0 - mask_float_q[None, ...]) * (-1e30)
        S_q = S_q * jnp.exp(g_diff_Sq)
        S_q = S_q * mask_float_q[None, ...]

        I_plus_S = jnp.eye(C, dtype=jnp.float32)[None, ...] + S
        A_inv = invert_triangular_matrix(I_plus_S, block_size=16)

        v_beta = v_T * beta_T[..., None]
        k_beta_g = k_beta * exp_g
        vk_in = jnp.concatenate(
            [
                v_beta,
                k_beta_g,
            ],
            axis=2,
        )  # (n_v, C, d_v + d_k)
        uw = jax.lax.dot_general(
            A_inv,
            vk_in,
            (((2, ), (1, )), ((0, ), (0, ))),
            precision=jax.lax.Precision.HIGHEST,
        )  # Output shape: (n_v, C, d_v + d_k)
        u = uw[..., :d_v]
        w = uw[..., d_v:]

        q_g = q_T * exp_g  # (n_v, C, d_k)

        @pl.when(should_load_init)
        def _finish_init_load():
            self.dma.commit_in(init_copy_op, prefill_slot,
                               self.scratch.scratch, prefill_slot)

        current_state = self.scratch.scratch[prefill_slot]  # (n_v, d_k, d_v)

        qw = jnp.concatenate([q_g, w], axis=1)  # (n_v, 2 * C, d_k)
        comb = jax.lax.dot_general(
            qw,
            current_state.astype(jnp.float32),
            (((2, ), (1, )), ((0, ), (0, ))),
            precision=jax.lax.Precision.DEFAULT,
        )  # Output shape: (n_v, 2 * C, d_v)
        attn_inter, v_prime = jnp.split(comb, 2, axis=1)

        v_new = u - v_prime
        term2 = jnp.matmul(S_q, v_new, precision=jax.lax.Precision.HIGHEST)
        o_c = attn_inter + term2  # (n_v, C, d_v)

        g_i_last_exp = exp_g[:, -1, None]
        g_diff_exp_state = jnp.exp(g_cumsum_T[..., -1, None] -
                                   g_cumsum_T)[..., None]
        k_i_g_diff = k_T * g_diff_exp_state
        update_term = jax.lax.dot_general(
            k_i_g_diff,
            v_new,
            (((1, ), (1, )), ((0, ), (0, ))),
            precision=jax.lax.Precision.DEFAULT,
        )  # Output shape: (n_v, d_k, d_v)
        h_new = current_state * g_i_last_exp + update_term

        self.scratch.scratch[prefill_slot] = h_new.astype(
            self.scratch.scratch.dtype)

        store_state_idx = self.state_indices[prefill_req_id][...]

        @pl.when(self.schedule.is_last_chunk > 0)
        def store_state():
            copy_op = self.dma.copy_out(prefill_slot, store_state_idx,
                                        self.scratch.scratch[prefill_slot])
            copy_op.wait()

        o_c_tr = o_c.transpose(1, 0, 2)
        o_c_flat = o_c_tr.reshape(C, n_v * d_v)

        mask_float = (jnp.arange(C) < prefill_count).astype(o_c_flat.dtype)
        o_c_flat_masked = o_c_flat * mask_float[:, None]
        self.refs.output[...] = o_c_flat_masked.astype(self.refs.output.dtype)

    def _process_transition_prefill(self):
        """Processes a transition prefill step with sublane stitching."""
        C_trans = self.cfg.tiling.sublanesize
        key_dim = self.cfg.model.key_dim
        n_v = self.cfg.model.n_v
        d_k = self.cfg.model.d_k
        d_v = self.cfg.model.d_v
        n_kq = self.cfg.model.n_kq

        first_req_id = self.schedule.schedule_table[self.schedule.step,
                                                    COL_SUBLANE_REQ_IDS][...]
        first_is_first = self.schedule.schedule_table[self.schedule.step,
                                                      COL_SUBLANE_REQ_IDS +
                                                      C_trans][...]
        first_slot = first_req_id % 2
        first_has_init = self.has_initial_state[first_req_id][...]
        should_load_first = (first_is_first > 0) & (first_has_init > 0)
        first_state_idx = self.state_indices[first_req_id][...]

        first_copy_op = self.dma.build_copy_in(first_slot, first_state_idx)

        @pl.when(should_load_first)
        def _start_first_load():
            first_copy_op.start()

        qkv_chunk = self.refs.qkv[:C_trans, :]
        qkv_chunk = jax.nn.silu(qkv_chunk)
        q = qkv_chunk[:, :key_dim]
        k = qkv_chunk[:, key_dim:2 * key_dim]
        v = qkv_chunk[:, 2 * key_dim:]

        a_raw_chunk = self.refs.a_raw[...]
        b_raw_chunk = self.refs.b_raw[...]

        a_raw_processed_T = a_raw_chunk[:C_trans, :n_v].T
        b_raw_processed_T = b_raw_chunk[:C_trans, :n_v].T

        beta_chunk_T = jax.nn.sigmoid(b_raw_processed_T)
        g_chunk_T = -jnp.exp(
            self.shared.a_log[...])[:, None] * jax.nn.softplus(
                a_raw_processed_T + self.shared.dt_bias[...][:, None])
        g_chunk_T = jnp.maximum(g_chunk_T, -100.0)

        q = q.reshape(C_trans, n_kq, d_k)
        k = k.reshape(C_trans, n_kq, d_k)
        v = v.reshape(C_trans, n_v, d_v)

        if self.cfg.use_qk_norm_in_gdn:
            q = l2_normalize(q)
            k = l2_normalize(k)

        repeat_factor = self.cfg.model.repeat_factor
        if repeat_factor > 1:
            q = jnp.repeat(q, repeat_factor, axis=1)
            k = jnp.repeat(k, repeat_factor, axis=1)

        scale = d_k**-0.5
        q = q * scale

        @pl.when((first_is_first > 0) & (first_has_init == 0))
        def _zero_first_slot():
            self.scratch.scratch[first_slot] = jnp.zeros(
                (n_v, d_k, d_v), dtype=self.scratch.scratch.dtype)

        @pl.when(should_load_first)
        def _finish_first_load():
            self.dma.commit_in(first_copy_op, first_slot, self.scratch.scratch,
                               first_slot)

        h = self.scratch.scratch[first_slot]
        current_r = first_req_id
        sequence_valid = True
        exp_g_chunk_T = jnp.exp(g_chunk_T)

        for i in range(C_trans):
            t_req = self.schedule.schedule_table[self.schedule.step,
                                                 11 + i][...]
            t_is_first = self.schedule.schedule_table[self.schedule.step,
                                                      11 + C_trans + i][...]
            t_is_last = self.schedule.schedule_table[self.schedule.step,
                                                     11 + 2 * C_trans + i][...]

            is_new_seq = t_req != current_r
            sequence_valid = jnp.where(is_new_seq, True, sequence_valid)

            is_decode_token = t_req < self.cfg.decode_tokens
            sequence_valid = jnp.where(is_decode_token, False, sequence_valid)

            c_slot = current_r % 2
            self.scratch.scratch[c_slot] = h

            def do_write(c_slot=c_slot, current_r=current_r, h=h):
                state_idx = self.state_indices[current_r][...]
                copy_op = self.dma.copy_out(c_slot, state_idx, h)
                copy_op.wait()
                return None

            is_current_r_prefill = current_r >= self.cfg.decode_tokens
            should_write = is_current_r_prefill & is_new_seq
            jax.lax.cond(should_write, do_write, lambda: None)

            t_slot = t_req % 2
            t_has_init = self.has_initial_state[t_req][...]

            def load_t_state(t_slot=t_slot, t_req=t_req):
                state_idx = self.state_indices[t_req][...]
                copy_op = self.dma.build_copy_in(t_slot, state_idx)
                copy_op.start()
                self.dma.commit_in(copy_op, t_slot, self.scratch.scratch,
                                   t_slot)

            should_load_t = (t_is_first > 0) & (t_has_init > 0)
            jax.lax.cond(should_load_t, load_t_state, lambda: None)

            should_zero = (t_is_first > 0) & (t_has_init == 0)

            def zero_t_slot(t_slot=t_slot, n_v=n_v, d_k=d_k, d_v=d_v):
                self.scratch.scratch[t_slot] = jnp.zeros(
                    (n_v, d_k, d_v), dtype=self.scratch.scratch.dtype)

            jax.lax.cond(should_zero, zero_t_slot, lambda: None)

            h = self.scratch.scratch[t_slot]
            current_r = t_req

            k_i = k[i, :, :]
            v_i = v[i, :, :]
            beta_i = beta_chunk_T[:, i]
            q_i = q[i, :, :]

            decay = exp_g_chunk_T[:, i][..., None]

            k_state = jnp.sum(k_i[..., None] * h, axis=1)
            v_diff = v_i - decay * k_state
            v_new = beta_i[:, None] * v_diff

            q_state = jnp.sum(q_i[..., None] * h, axis=1)
            q_k = jnp.sum(q_i * k_i, axis=-1, keepdims=True)

            out_i = decay * q_state + q_k * v_new

            k_v_new = k_i[..., None] * v_new[:, None, :]
            h_new = h * decay[..., None] + k_v_new

            h = jnp.where(sequence_valid, h_new, h)
            out_i = jnp.where(sequence_valid, out_i, 0.0)

            sequence_valid = jnp.where(t_is_last > 0, False, sequence_valid)

            self.refs.output[i, :] = out_i.reshape(n_v * d_v).astype(
                self.refs.output.dtype)

        final_slot = current_r % 2
        self.scratch.scratch[final_slot] = h

        is_current_r_prefill = current_r >= self.cfg.decode_tokens

        @pl.when(is_current_r_prefill)
        def do_final_write():
            state_idx = self.state_indices[current_r][...]
            copy_op = self.dma.copy_out(final_slot, state_idx, h)
            copy_op.wait()
            return None


class DecodeProcessor(ScanProcessor):
    """Handles batch decode step processing using double-buffering logic."""

    def __init__(
        self,
        config: ScanConfig,
        schedule: ScheduleStep,
        state_indices,
        has_initial_state,
        refs: BranchRefs,
        shared: SharedRefs,
        scratch: DecodeScratchRefs,
        dma: DMAHelper,
    ):
        super().__init__(config, schedule, state_indices, has_initial_state)
        self.refs = refs
        self.shared = shared
        self.scratch = scratch
        self.dma = dma

    def get_target_idx(self, b):
        safe_req_id = jnp.minimum(self.schedule.decode_req_id + b,
                                  self.state_indices.shape[0] - 1)
        return self.state_indices[safe_req_id][...]

    def process(self):
        """Processes decode steps in blocks."""
        decode_count = self.schedule.decode_count
        BT = self.cfg.tiling.BT
        n_v = self.cfg.model.n_v
        d_k = self.cfg.model.d_k
        d_v = self.cfg.model.d_v
        n_kq = self.cfg.model.n_kq
        key_dim = self.cfg.model.key_dim
        repeat_factor = self.cfg.model.repeat_factor
        use_qk_norm_in_gdn = self.cfg.use_qk_norm_in_gdn
        exp_a_log = jnp.exp(self.shared.a_log[...].astype(jnp.float32))
        dt_bias_f32 = self.shared.dt_bias[...].astype(jnp.float32)

        # Pre-loop: kick off async loads for iters 0 and 1.
        @pl.when(decode_count >= 1)
        def _preload_slot_0():
            tgt = self.get_target_idx(0)
            op = pltpu.make_async_copy(
                src_ref=self.shared.recurrent_state_in.at[pl.ds(tgt, 1)],
                dst_ref=self.scratch.load.at[pl.ds(0, 1)],
                sem=self.scratch.read_semaphores.at[0],
            )
            op.start()

        @pl.when(decode_count >= 2)
        def _preload_slot_1():
            tgt = self.get_target_idx(1)
            op = pltpu.make_async_copy(
                src_ref=self.shared.recurrent_state_in.at[pl.ds(tgt, 1)],
                dst_ref=self.scratch.load.at[pl.ds(1, 1)],
                sem=self.scratch.read_semaphores.at[1],
            )
            op.start()

        def process_decode_step(b, store_inflight):
            s0_inflight, s1_inflight = store_inflight
            is_valid = b < decode_count
            slot = b % 2
            using_slot_0 = slot == 0
            cur_slot_inflight = jax.lax.select(using_slot_0, s0_inflight,
                                               s1_inflight)

            @pl.when(is_valid)
            def do_work():
                # Wait for THIS iter's load.
                wait_load = pltpu.make_async_copy(
                    src_ref=self.shared.recurrent_state_in.at[pl.ds(0, 1)],
                    dst_ref=self.scratch.load.at[pl.ds(slot, 1)],
                    sem=self.scratch.read_semaphores.at[slot],
                )
                wait_load.wait()

                self.scratch.state[pl.ds(0, 1)] = self.scratch.load[pl.ds(
                    slot, 1)][...]

                # Prefetch load for iter b+2.
                next_b = b + 2

                @pl.when(next_b < decode_count)
                def _prefetch_next_load():
                    next_tgt = self.get_target_idx(next_b)
                    op = pltpu.make_async_copy(
                        src_ref=self.shared.recurrent_state_in.at[pl.ds(
                            next_tgt, 1)],
                        dst_ref=self.scratch.load.at[pl.ds(slot, 1)],
                        sem=self.scratch.read_semaphores.at[slot],
                    )
                    op.start()

                target_idx = self.get_target_idx(b)

                sublanesize = self.cfg.tiling.sublanesize
                b_aligned = (b // sublanesize) * sublanesize

                qkv_block_data = self.refs.qkv[
                    pl.ds(b_aligned, sublanesize), :].astype(jnp.float32)
                mask = (jnp.arange(sublanesize) == (b % sublanesize)).astype(
                    qkv_block_data.dtype)[:, None]
                qkv_row = jnp.sum(qkv_block_data * mask, axis=0, keepdims=True)

                # Fused SiLU
                qkv_row = jax.nn.silu(qkv_row)
                q = qkv_row[:, :key_dim].reshape(n_kq, d_k)
                k = qkv_row[:, key_dim:2 * key_dim].reshape(n_kq, d_k)
                v = qkv_row[:, 2 * key_dim:].reshape(n_v, d_v)

                if use_qk_norm_in_gdn:
                    q = l2_normalize(q)
                    k = l2_normalize(k)

                # Head repetition
                if repeat_factor > 1:
                    q = jnp.repeat(q, repeat_factor, axis=0)
                    k = jnp.repeat(k, repeat_factor, axis=0)

                scale = d_k**-0.5
                q = q * scale

                g_block_new = self.refs.a_raw[pl.ds(b_aligned, sublanesize), :]
                beta_block_new = self.refs.b_raw[
                    pl.ds(b_aligned, sublanesize), :]

                mask_new = (jnp.arange(sublanesize) == (
                    b % sublanesize)).astype(g_block_new.dtype)[:, None]

                a_raw_new = jnp.sum(g_block_new * mask_new,
                                    axis=0,
                                    keepdims=True)[0, :n_v].astype(jnp.float32)
                b_raw_new = jnp.sum(beta_block_new * mask_new,
                                    axis=0,
                                    keepdims=True)[0, :n_v].astype(jnp.float32)

                # Compute gate
                curr_beta = jax.nn.sigmoid(b_raw_new)
                curr_g = -exp_a_log * jax.nn.softplus(a_raw_new + dt_bias_f32)
                curr_g = jnp.maximum(curr_g, -100.0)
                decay = jnp.exp(curr_g)

                current_state = self.scratch.state[0]

                # 1. Batched dot product: k @ state -> (n_v, d_v)
                k_state = jax.lax.dot_general(
                    k.reshape(n_v, 1, d_k),
                    current_state,
                    (((2, ), (1, )), ((0, ), (0, ))),
                    preferred_element_type=jnp.float32,
                ).reshape(n_v, d_v)

                decay_k_state = jnp.where(
                    jnp.isinf(k_state),
                    0.0,
                    decay[:, None] * k_state,
                )
                v_diff = v - decay_k_state
                v_new = curr_beta[:, None] * v_diff

                # 2. Batched dot product: q @ state -> (n_v, d_v)
                q_state = jax.lax.dot_general(
                    q.reshape(n_v, 1, d_k),
                    current_state,
                    (((2, ), (1, )), ((0, ), (0, ))),
                    preferred_element_type=jnp.float32,
                ).reshape(n_v, d_v)

                q_k = jnp.sum(
                    q * k,
                    axis=-1,
                    keepdims=True,
                )

                decay_q_state = jnp.where(
                    jnp.isinf(q_state),
                    0.0,
                    decay[:, None] * q_state,
                )
                out_step = decay_q_state + q_k * v_new

                # 3. Outer product and decay update for state -> (n_v, d_k, d_v)
                decay_state = jnp.where(
                    jnp.isinf(current_state),
                    0.0,
                    current_state * decay[:, None, None],
                )
                k_v_new = k[:, :, None] * v_new[:, None, :]
                new_state = decay_state + k_v_new
                self.scratch.store[slot] = new_state.astype(
                    self.scratch.store.dtype)

                # Accumulate output in scratchpad
                current_output = self.scratch.output[...]
                mask = (jnp.arange(BT) == b).astype(current_output.dtype)[:,
                                                                          None]
                new_output = jnp.where(
                    mask,
                    out_step.reshape(1,
                                     n_v * d_v).astype(current_output.dtype),
                    current_output,
                )
                self.scratch.output[...] = new_output

                # Async store. Before writing to decode_store_scratch[slot],
                # wait for the previous same-slot store DMA (from iter b-2)
                @pl.when(cur_slot_inflight > 0)
                def _wait_same_slot_store():
                    copy_op = pltpu.make_async_copy(
                        src_ref=self.scratch.store.at[pl.ds(slot, 1)],
                        dst_ref=self.shared.recurrent_state_out.at[pl.ds(0,
                                                                         1)],
                        sem=self.scratch.write_semaphore.at[slot],
                    )
                    copy_op.wait()

                copy_op = pltpu.make_async_copy(
                    src_ref=self.scratch.store.at[pl.ds(slot, 1)],
                    dst_ref=self.shared.recurrent_state_out.at[pl.ds(
                        target_idx, 1)],
                    sem=self.scratch.write_semaphore.at[slot],
                )
                copy_op.start()

            next_s0_inflight = jax.lax.select(
                is_valid & using_slot_0,
                jnp.int32(1),
                s0_inflight,
            )
            next_s1_inflight = jax.lax.select(
                is_valid & (~using_slot_0),
                jnp.int32(1),
                s1_inflight,
            )
            return (next_s0_inflight, next_s1_inflight)

        final_s0_inflight, final_s1_inflight = jax.lax.fori_loop(
            0,
            BT,
            process_decode_step,
            (jnp.int32(0), jnp.int32(0)),
        )

        # Drain any remaining async store DMAs
        @pl.when(final_s0_inflight > 0)
        def _drain_slot_0():
            temp_desc = pltpu.make_async_copy(
                src_ref=self.scratch.store.at[pl.ds(0, 1)],
                dst_ref=self.shared.recurrent_state_out.at[pl.ds(0, 1)],
                sem=self.scratch.write_semaphore.at[0],
            )
            temp_desc.wait()

        @pl.when(final_s1_inflight > 0)
        def _drain_slot_1():
            temp_desc = pltpu.make_async_copy(
                src_ref=self.scratch.store.at[pl.ds(1, 1)],
                dst_ref=self.shared.recurrent_state_out.at[pl.ds(0, 1)],
                sem=self.scratch.write_semaphore.at[1],
            )
            temp_desc.wait()

        mask = (jnp.arange(BT)
                < decode_count).astype(self.scratch.output.dtype)[:, None]
        decode_output_scratch_masked = self.scratch.output[...] * mask
        self.refs.output[...] = decode_output_scratch_masked
