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
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from vllm.v1.outputs import LogprobsTensors

from tpu_inference import envs
from tpu_inference.layers.common.attention_metadata import (
    AttentionMetadata, GroupedAttentionMetadata, SharedAttentionMetadata)
from tpu_inference.models.common.compiler_options import (
    get_sampling_compiler_options, get_step_fn_compiler_options)

# ---- Dead-row KV clamp (env CONTINUE_DECODE_DEAD_KV_CLAMP=1, default off) ----
# In the fused continue-decode loop a row that sampled EOS stays in the batch
# until the window ends with its seq_lens frozen, so attention keeps reading its
# whole KV every step for a result that is discarded (post-EOS tokens are
# padded). ON: each step's attention metadata gets seq_lens=1 for the rows that
# were already finished when the step starts (and seq_lens > 0), laid out like
# _update_loop_state's `increment`. The carried seq_lens/positions are
# untouched, so active rows and the host bookkeeping are unchanged. A finished
# row's pad KV goes to slot 0 of its own first page (stock: slot seq_len-1 of
# the same request); with prefix caching off neither is read again, which the
# runner config gate enforces.
_DEADKV_ON = bool(envs.CONTINUE_DECODE_DEAD_KV_CLAMP)
_DEADKV = {
    "gate": None,
    "traces": 0,
    "chunks": 0,
    "row_steps": 0,
    "dead": 0,
    "errors": 0
}


def _dk_log(msg):
    print("[dead-kv-clamp] " + msg, file=sys.stderr, flush=True)


def _deadkv_config_gate(runner):
    """End of TPUModelRunner.__init__. ON: raise unless the clamp is exact for this config."""
    if not _DEADKV_ON:
        return
    vc = runner.vllm_config
    pc = getattr(getattr(vc, "cache_config", None), "enable_prefix_caching",
                 None)
    spec = getattr(vc, "speculative_config", None)
    pcp = int(
        getattr(getattr(vc, "sharding_config", None), "prefill_cp_size", 1)
        or 1)
    shape = dict(getattr(getattr(runner, "mesh", None), "shape", None) or {})
    mpcp, dcp = int(shape.get("pcp", 1)), int(shape.get("dcp", 1))
    hybrid = bool(
        getattr(getattr(vc, "model_config", None), "is_hybrid", False))
    bad = []
    if pc is not False:
        bad.append(
            "enable_prefix_caching=%r (the pad KV of a finished row goes to "
            "slot 0 of its first page, which prefix caching shares between "
            "requests)" % (pc, ))
    if spec is not None:
        bad.append("speculative_config=%s" % type(spec).__name__)
    if (pcp, mpcp, dcp) != (1, 1, 1):
        bad.append("prefill_cp_size=%d mesh pcp=%d dcp=%d (need 1)" %
                   (pcp, mpcp, dcp))
    if envs.USE_BATCHED_RPA_KERNEL:
        bad.append("USE_BATCHED_RPA_KERNEL=1 (audited for the RPA v3 "
                   "per-sequence kernel only)")
    if hybrid:
        bad.append("hybrid (mamba/linear-attention) model: the clamp is "
                   "audited for attention only")
    if bad:
        _DEADKV["gate"] = "refused"
        msg = ("[dead-kv-clamp] FATAL: refusing the dead-row KV clamp: " +
               "; ".join(bad))
        print(msg, file=sys.stderr, flush=True)
        raise RuntimeError(msg)
    _DEADKV["gate"] = {"prefix_caching": pc, "dp": int(runner.dp_size)}
    _dk_log("config gate PASS: prefix_caching=False spec_decode=None pcp=1 "
            "dcp=1 dp=%d eos=%r pad=%r" %
            (runner.dp_size, runner.eos_token_id, runner.pad_token_id))


def _deadkv_seq_lens(am, sl, dp_size, pad_len, mamba_state_indices):
    """Trace time, in _run_one_step. OFF: `sl` itself (the stock program).

    ON: seq_lens for the attention metadata only: 1 for every row finished
    before this step (active_mask False) with sl > 0, laid out exactly as
    _update_loop_state lays out `increment` (per-DP 2D; pad_len False columns,
    else the slice).
    """
    if not _DEADKV_ON:
        return sl
    st = _DEADKV
    if not isinstance(st["gate"], dict):
        raise RuntimeError(
            "[dead-kv-clamp] FATAL: decode loop traced with the clamp ON but the "
            "runner config gate has not passed (gate=%r)" % (st["gate"], ))
    if mamba_state_indices is not None:
        raise RuntimeError(
            "[dead-kv-clamp] FATAL: mamba_state_indices is set; "
            "the clamp is audited for attention only")
    dead_2d = jnp.logical_not(am).reshape(dp_size, -1)
    sl_2d = sl.reshape(dp_size, -1)
    if pad_len > 0:
        dead_2d = jnp.pad(dead_2d, ((0, 0), (0, pad_len)))
    else:
        dead_2d = dead_2d[:, :sl_2d.shape[1]]
    dead = dead_2d.ravel()
    if dead.shape != sl.shape:
        raise RuntimeError(
            "[dead-kv-clamp] FATAL: mask layout %s != seq_lens %s" %
            (dead.shape, sl.shape))
    st["traces"] += 1
    k = st["traces"]
    if k <= 3 or k % 50 == 0:
        _dk_log("ON trace#%d bucket=tokens%s seq_lens%s dp=%d pad_len=%d "
                "prefix_caching=%r" % (k, tuple(am.shape), tuple(
                    sl.shape), dp_size, pad_len, st["gate"]["prefix_caching"]))
    return jnp.where(jnp.logical_and(dead, sl > 0), jnp.ones_like(sl), sl)


def _deadkv_note(valid_lens, has_eos, num_reqs, actual_steps):
    """Host, per continue-decode window, from arrays the runner already holds.

    A row whose EOS is its valid_len-th token was clamped on each of the
    actual_steps - valid_len steps after it.
    """
    if not _DEADKV_ON:
        return
    st = _DEADKV
    try:
        n, a = int(num_reqs), int(actual_steps)
        fin = has_eos[:n]
        dead = int(((a - valid_lens[:n]) * fin).sum())
        st["chunks"] += 1
        st["row_steps"] += n * a
        st["dead"] += dead
        c = st["chunks"]
        if c == 1 and st["traces"] == 0:
            _dk_log("INERT: a continue-decode chunk resolved but no decode "
                    "trace carried the clamp")
        if c <= 3 or c % 200 == 0:
            _dk_log(
                "chunk#%d rows=%d steps=%d dead_row_steps=%d finished=%d "
                "| cum chunks=%d row_steps=%d dead_row_steps=%d (%.1f%%) "
                "traces=%d" %
                (c, n, a, dead, int(fin.sum()), c, st["row_steps"], st["dead"],
                 100.0 * st["dead"] / max(st["row_steps"], 1), st["traces"]))
    except Exception as exc:  # logging must never break decoding
        st["errors"] += 1
        if st["errors"] <= 3:
            _dk_log("log error (ignored): %r" % (exc, ))


# ---- end dead-row KV clamp ----


def _unpack_attn_metadata(
    attn: AttentionMetadata | GroupedAttentionMetadata,
) -> tuple[
        AttentionMetadata,
        jax.Array | tuple[jax.Array | None, ...] | None,
        tuple[tuple[str, ...], ...] | None,
]:
    if isinstance(attn, GroupedAttentionMetadata):
        first_attn = attn.groups[0]
        block_tables = tuple(g.block_tables for g in attn.groups)
        return first_attn, block_tables, attn.layer_names_per_group
    return attn, attn.block_tables, None


def _build_attn_metadata(
    *,
    input_positions: jax.Array,
    block_tables: jax.Array | tuple[jax.Array | None, ...] | None,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    mamba_state_indices: jax.Array | None,
    padded_num_reqs: int,
    layer_names_per_group: tuple[tuple[str, ...], ...] | None,
) -> AttentionMetadata | GroupedAttentionMetadata:
    if layer_names_per_group is None:
        return AttentionMetadata(
            input_positions=input_positions,
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
            mamba_state_indices=mamba_state_indices,
            padded_num_reqs=padded_num_reqs,
        )
    return GroupedAttentionMetadata(
        groups=tuple(
            AttentionMetadata(
                input_positions=input_positions,
                block_tables=bt,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
                mamba_state_indices=mamba_state_indices,
                padded_num_reqs=padded_num_reqs,
            ) for bt in block_tables),
        layer_names_per_group=layer_names_per_group,
    )


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "current_tokens",
        "active_mask",
        "attn_metadata",
        "step_counter",
    ],
    meta_fields=[],
)
@dataclass
class TpuSamplingState:
    # (batch_size,)
    current_tokens: jax.Array
    # (batch_size,)
    active_mask: jax.Array
    # AttentionMetadata or GroupedAttentionMetadata PyTree
    attn_metadata: AttentionMetadata | GroupedAttentionMetadata
    # scalar
    step_counter: jax.Array


@functools.partial(
    jax.jit,
    static_argnames=["eos_token_id", "padding_token_id", "dp_size", "pad_len"])
def _update_loop_state(
    next_tokens: jax.Array,
    active_mask: jax.Array,
    input_positions: jax.Array,
    seq_lens: jax.Array,
    eos_token_id: tuple[int, ...],
    padding_token_id: int,
    dp_size: int,
    pad_len: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    eos_arr = jnp.atleast_1d(jnp.array(eos_token_id, dtype=jnp.int32))
    is_eos = jnp.any(next_tokens[:, None] == eos_arr[None, :], axis=-1)
    new_active_mask = jnp.logical_and(active_mask, jnp.logical_not(is_eos))
    next_input_ids = jnp.where(new_active_mask, next_tokens, padding_token_id)
    increment = new_active_mask.astype(jnp.int32)
    new_positions = input_positions + increment

    # Fix DP padding bug by exposing DP dimension as 2D before padding
    increment_2d = increment.reshape(dp_size, -1)
    seq_lens_2d = seq_lens.reshape(dp_size, -1)
    if pad_len > 0:
        padded_increment_2d = jnp.pad(increment_2d, ((0, 0), (0, pad_len)))
    else:
        padded_increment_2d = increment_2d[:, :seq_lens_2d.shape[1]]
    new_seq_lens = (seq_lens_2d + padded_increment_2d).ravel()

    # Compute the token to record in generated_tokens
    step_record_tokens = jnp.where(active_mask, next_tokens, padding_token_id)

    any_hit_eos = jnp.any(jnp.logical_and(active_mask, is_eos))

    return new_active_mask, next_input_ids, new_positions, new_seq_lens, step_record_tokens, any_hit_eos


@functools.partial(jax.jit, static_argnums=(1, ))
def _split_rngs(rng, static_size, dynamic_size):
    all_rngs = jax.random.split(rng, static_size + 1)
    # Keep the per-step keys as an array (not a Python tuple): the decode loop
    # is a lax.while_loop, so step keys are indexed by a *traced* step counter,
    # which requires array indexing.
    return all_rngs[:static_size], all_rngs[dynamic_size]


def _decode_core_impl(
    *,
    state,
    kv_caches,
    step_rngs,
    sampling_metadata,
    inputs_embeds,
    lora_metadata,
    intermediate_tensors,
    block_tables,
    query_start_loc,
    request_distribution,
    mamba_state_indices,
    current_tokens,
    active_mask,
    input_positions,
    seq_lens,
    model_fn,
    compute_logits_fn,
    sample_fn,
    mesh,
    max_decode_steps,
    static_max_decode_steps,
    eos_token_id,
    padding_token_id,
    dp_size,
    pad_len,
    layer_name_to_kvcache_index,
    is_first_rank,
    is_last_rank,
    max_logprobs,
    logprobs_mode,
    collect_expert_indices: bool = False,
    continue_decode_eos_check_interval: int = 1,
    padded_num_reqs: int = -1,
    layer_names_per_group: tuple[tuple[str, ...], ...] | None = None,
):
    has_logprobs = False if sampling_metadata is None else sampling_metadata.logprobs
    from tpu_inference.layers.jax.sample.sampling import \
        distributed_sampling_allowed
    allow_distributed_sampling = distributed_sampling_allowed(
        has_logprobs, logprobs_mode)

    # Discover the per-step expert-indices shape at JIT trace time without
    # executing a step. Gated by collect_expert_indices so the abstract trace is
    # skipped for the common non-MoE path. Because _decode_core_impl is wrapped
    # in @jax.jit, this block only executes during compilation/tracing and has
    # zero runtime overhead once compiled.
    has_experts = False
    expert_shape = None
    expert_dtype = None
    if collect_expert_indices:

        def _model_experts_only(
            state_arg,
            kv_caches_arg,
            current_tokens_arg,
            input_positions_arg,
            seq_lens_arg,
            block_tables_arg,
            query_start_loc_arg,
            request_distribution_arg,
            mamba_state_indices_arg,
            inputs_embeds_arg,
            lora_metadata_arg,
            intermediate_tensors_arg,
        ):
            am = _build_attn_metadata(
                input_positions=input_positions_arg,
                block_tables=block_tables_arg,
                seq_lens=seq_lens_arg,
                query_start_loc=query_start_loc_arg,
                request_distribution=request_distribution_arg,
                mamba_state_indices=mamba_state_indices_arg,
                padded_num_reqs=padded_num_reqs,
                layer_names_per_group=layer_names_per_group,
            )
            shared_am = SharedAttentionMetadata(
                input_positions=input_positions_arg,
                seq_lens=seq_lens_arg,
                query_start_loc=query_start_loc_arg,
                request_distribution=request_distribution_arg,
                mamba_state_indices=mamba_state_indices_arg,
                padded_num_reqs=padded_num_reqs,
            )
            _, _, _, experts = model_fn(
                state_arg,
                kv_caches_arg,
                current_tokens_arg,
                am,
                inputs_embeds_arg,
                input_positions_arg,
                layer_name_to_kvcache_index,
                lora_metadata_arg,
                intermediate_tensors_arg,
                is_first_rank,
                is_last_rank,
                shared_attention_metadata=shared_am,
            )
            return experts

        expert_struct = jax.eval_shape(
            _model_experts_only,
            state,
            kv_caches,
            current_tokens,
            input_positions,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
            mamba_state_indices,
            inputs_embeds,
            lora_metadata,
            intermediate_tensors,
        )
        if expert_struct is not None:
            has_experts = True
            expert_shape = tuple(expert_struct.shape)
            expert_dtype = expert_struct.dtype

    def _run_one_step(step_idx, ct, am, pos, sl, kvc):
        step_rng = step_rngs[step_idx]
        # Dead-row KV clamp: attention-only seq_lens (off: `sl` itself).
        attn_sl = _deadkv_seq_lens(am, sl, dp_size, pad_len,
                                   mamba_state_indices)
        attn_metadata = _build_attn_metadata(
            input_positions=pos,
            block_tables=block_tables,
            seq_lens=attn_sl,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
            mamba_state_indices=mamba_state_indices,
            padded_num_reqs=padded_num_reqs,
            layer_names_per_group=layer_names_per_group,
        )
        shared_attn_metadata = SharedAttentionMetadata(
            input_positions=pos,
            seq_lens=attn_sl,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
            mamba_state_indices=mamba_state_indices,
            padded_num_reqs=padded_num_reqs,
        )
        kvc, hidden_states, _, expert_indices_step = model_fn(
            state,
            kvc,
            ct,
            attn_metadata,
            inputs_embeds,
            pos,
            layer_name_to_kvcache_index,
            lora_metadata,
            intermediate_tensors,
            is_first_rank,
            is_last_rank,
            shared_attention_metadata=shared_attn_metadata,
        )
        logits = compute_logits_fn(state, hidden_states, None)
        logits = logits.astype(jnp.float32)
        next_tokens, processed_logits = sample_fn(
            step_rng,
            mesh,
            logits,
            sampling_metadata,
            allow_distributed_sampling=allow_distributed_sampling)
        (new_active_mask, next_input_ids, new_positions, new_seq_lens,
         step_record_tokens, any_hit_eos) = _update_loop_state(
             next_tokens,
             am,
             pos,
             sl,
             eos_token_id,
             padding_token_id,
             dp_size,
             pad_len,
         )

        lp_ids_step = None
        lp_val_step = None
        lp_ranks_step = None
        if has_logprobs:
            from tpu_inference.layers.jax.sample.sampling import (
                compute_and_gather_logprobs, logprobs_use_processed_logits)
            logprobs_logits = (processed_logits
                               if logprobs_use_processed_logits(logprobs_mode)
                               else logits)
            step_logprobs = compute_and_gather_logprobs(
                logprobs_logits, next_tokens, max_logprobs)
            lp_ids_step = step_logprobs.logprob_token_ids
            lp_val_step = step_logprobs.logprobs
            lp_ranks_step = step_logprobs.selected_token_ranks

        return (next_input_ids, new_active_mask, new_positions, new_seq_lens,
                kvc, step_record_tokens, expert_indices_step, lp_ids_step,
                lp_val_step, lp_ranks_step, any_hit_eos)

    batch_size = current_tokens.shape[0]
    token_buffer = jnp.full((static_max_decode_steps, batch_size),
                            padding_token_id,
                            dtype=current_tokens.dtype)
    expert_buffer = None
    if has_experts:
        expert_buffer = jnp.zeros((static_max_decode_steps, ) + expert_shape,
                                  dtype=expert_dtype)

    logprob_token_ids_buffer = None
    logprobs_buffer = None
    selected_token_ranks_buffer = None
    if has_logprobs:
        logprob_token_ids_buffer = jnp.zeros(
            (static_max_decode_steps, batch_size, max_logprobs + 1),
            dtype=jnp.int32)
        logprobs_buffer = jnp.zeros(
            (static_max_decode_steps, batch_size, max_logprobs + 1),
            dtype=jnp.float32)
        selected_token_ranks_buffer = jnp.zeros(
            (static_max_decode_steps, batch_size), dtype=jnp.int32)

    def _pack(i, ct, am, pos, sl, kvc, tb, eb, lp_ids, lp_val, lp_ranks, eos):
        base = [i, ct, am, pos, sl, kvc, tb]
        if has_experts:
            base.append(eb)
        if has_logprobs:
            base.extend([lp_ids, lp_val, lp_ranks])
        base.append(eos)
        return tuple(base)

    def _unpack(carry):
        i, ct, am, pos, sl, kvc, tb = carry[:7]
        idx = 7
        eb = None
        if has_experts:
            eb = carry[idx]
            idx += 1

        lp_ids = None
        lp_val = None
        lp_ranks = None
        if has_logprobs:
            lp_ids = carry[idx]
            lp_val = carry[idx + 1]
            lp_ranks = carry[idx + 2]
            idx += 3

        eos = carry[idx]
        return i, ct, am, pos, sl, kvc, tb, eb, lp_ids, lp_val, lp_ranks, eos

    def cond_fn(carry):
        i = carry[0]
        eos_flag = carry[-1]
        not_done = i < max_decode_steps
        if continue_decode_eos_check_interval <= 0:
            return not_done
        should_check_eos = (i % continue_decode_eos_check_interval == 0)
        return jnp.logical_and(
            not_done,
            jnp.logical_not(jnp.logical_and(eos_flag, should_check_eos)),
        )

    def body_fn(carry):
        (i, ct, am, pos, sl, kvc, tb, eb, lp_ids_buf, lp_val_buf, lp_ranks_buf,
         eos_flag) = _unpack(carry)
        (next_ct, new_mask, new_pos, new_sl, kvc, rec_tokens, experts,
         lp_ids_step, lp_val_step, lp_ranks_step,
         hit) = _run_one_step(i, ct, am, pos, sl, kvc)
        tb = tb.at[i].set(rec_tokens)
        if has_experts:
            eb = eb.at[i].set(experts)
        if has_logprobs:
            lp_ids_buf = lp_ids_buf.at[i].set(lp_ids_step)
            lp_val_buf = lp_val_buf.at[i].set(lp_val_step)
            lp_ranks_buf = lp_ranks_buf.at[i].set(lp_ranks_step)
        if continue_decode_eos_check_interval <= 0:
            # cond_fn never reads the EOS flag in this mode. Carrying it
            # unchanged lets the compiler drop the per-step `any_hit_eos`
            # reduction, which is a cross-DP all-reduce over every device
            # on each decode iteration.
            new_eos_flag = eos_flag
        else:
            new_eos_flag = jnp.logical_or(eos_flag, hit)
        return _pack(i + 1, next_ct, new_mask, new_pos, new_sl, kvc, tb, eb,
                     lp_ids_buf, lp_val_buf, lp_ranks_buf, new_eos_flag)

    init_carry = _pack(
        jnp.array(0, dtype=jnp.int32),
        current_tokens,
        active_mask,
        input_positions,
        seq_lens,
        kv_caches,
        token_buffer,
        expert_buffer if has_experts else None,
        logprob_token_ids_buffer,
        logprobs_buffer,
        selected_token_ranks_buffer,
        jnp.array(False),
    )
    final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    (step_idx_final, current_tokens, active_mask, positions, seq_lens,
     kv_caches, token_buffer, expert_buffer, lp_ids_buffer, lp_val_buffer,
     lp_ranks_buffer, _) = _unpack(final_carry)

    return (step_idx_final, current_tokens, active_mask, positions, seq_lens,
            kv_caches, token_buffer, expert_buffer, lp_ids_buffer,
            lp_val_buffer, lp_ranks_buffer)


@functools.lru_cache(maxsize=1)
def _get_decode_core():
    """Build the fused-decode loop jit on first use.

    The compiler options are hoisted here from the model's step fn: JAX forbids
    `compiler_options` on a nested jit, and the step fn is called from inside
    this loop's `while_loop` body. They must come from
    `get_step_fn_compiler_options()` rather than a literal, so the fused program
    is compiled with exactly the same options as the non-fused decode path --
    that function also returns the SparseCore offload thresholds whenever
    `SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES` is non-zero (it defaults to
    "auto"), which a hardcoded literal silently dropped.

    Built lazily, not at import: the "auto" path queries the TPU VMEM size and
    needs initialized devices. Cached so the jit is constructed once rather than
    per call, which would rebuild the dispatch wrapper on every decode step.
    """
    return functools.partial(
        jax.jit,
        static_argnames=(
            "model_fn",
            "compute_logits_fn",
            "sample_fn",
            "mesh",
            "static_max_decode_steps",
            "eos_token_id",
            "padding_token_id",
            "dp_size",
            "pad_len",
            "collect_expert_indices",
            "layer_name_to_kvcache_index",
            "is_first_rank",
            "is_last_rank",
            "max_logprobs",
            "logprobs_mode",
            "continue_decode_eos_check_interval",
            "padded_num_reqs",
            "layer_names_per_group",
        ),
        donate_argnames=("kv_caches", ),
        compiler_options={
            **get_step_fn_compiler_options(),
            **get_sampling_compiler_options(),
        },
    )(_decode_core_impl)


def continue_decode(
    state: dict,
    model_fn: Callable,
    compute_logits_fn: Callable,
    sample_fn: Callable,
    init_state: TpuSamplingState,
    kv_caches: Any,
    max_decode_steps: int,
    static_max_decode_steps: int,
    eos_token_id: tuple[int, ...],
    padding_token_id: int,
    rng: jax.Array,
    *,
    mesh: Any,
    sampling_metadata: Any,
    inputs_embeds: jax.Array | None = None,
    layer_name_to_kvcache_index: tuple[tuple[str, int], ...] = (),
    lora_metadata: dict | None = None,
    intermediate_tensors: dict[str, jax.Array] | None = None,
    is_first_rank: bool = True,
    is_last_rank: bool = True,
    dp_size: int = 1,
    collect_expert_indices: bool = False,
    max_logprobs: int = 0,
    logprobs_mode: str = "raw",
    continue_decode_eos_check_interval: int = 1,
) -> tuple[jax.Array, Any, TpuSamplingState, jax.Array, jax.Array | None,
           Optional["LogprobsTensors"]]:
    """Run the TPU decode loop as one fused, kv-cache-donating program.

    Args:
      state: Model state dict (weights; passed through, not donated).
      model_fn: Stable model forward callable.
      compute_logits_fn: Stable logits callable.
      sample_fn: Stable sampling callable with signature
        (rng, mesh, logits, sampling_metadata, allow_distributed_sampling=...)
        -> (next_tokens, _). Must be a stable object (not a per-call closure)
        so the jit cache persists; per-call sampling data is threaded via
        `sampling_metadata`.
      init_state: Initial TpuSamplingState.
      kv_caches: KV caches. Donated into the fused loop and returned updated.
      max_decode_steps: Max steps to run (static loop bound).
      static_max_decode_steps: Static maximum steps for RNG splitting.
      eos_token_id: EOS token ID(s).
      padding_token_id: Padding token ID.
      rng: RNG key.
      mesh: Device mesh (static; stable runner object).
      sampling_metadata: Per-call sampling metadata pytree (traced).
      inputs_embeds: Optional input embeddings.
      layer_name_to_kvcache_index: Mapping from layer name to KV cache index.
      lora_metadata: Optional LoRA metadata.
      intermediate_tensors: Optional intermediate tensors.
      is_first_rank: Whether this is the first PP rank.
      is_last_rank: Whether this is the last PP rank.
      dp_size: Data parallel size.
      collect_expert_indices: Whether model_fn returns routed-expert indices
        (caller derives this from
        vllm_config.aux_output_config.enable_return_routed_experts). When True the
        expert-indices shape is discovered via jax.eval_shape inside the JIT
        at trace time (no runtime overhead) to presize the accumulation buffer.
      max_logprobs: Minimum number of logprobs to retain per token.
      logprobs_mode: Logprobs mode from model config. The processed modes
        (see PROCESSED_LOGPROBS_MODES) read sample()'s transformed logits;
        the raw modes read compute_logits' output.

    Returns:
      Tuple of (generated_tokens, final_kv_caches, final_state, final_rng,
      all_expert_indices, logprobs_tensors). generated_tokens is a fixed-size
      (max_decode_steps, batch_size) array and all_expert_indices, when not
      None, is (max_decode_steps, ...); rows beyond final_state.step_counter
      are padding (early EOS exit may stop before max_decode_steps), so the
      caller must trim with final_state.step_counter.
    """
    from tpu_inference.layers.jax.sample.sampling import sample
    if sample_fn is sample:
        sample_fn = sample.__wrapped__

    first_attn, block_tables, layer_names_per_group = _unpack_attn_metadata(
        init_state.attn_metadata)
    batch_size = init_state.current_tokens.shape[0]
    seq_lens_size = first_attn.seq_lens.shape[0]
    pad_len = (seq_lens_size - batch_size) // dp_size

    step_rngs, current_rng = _split_rngs(rng, static_max_decode_steps,
                                         max_decode_steps)

    (step_counter, current_tokens, active_mask, positions, seq_lens, kv_caches,
     token_buffer, expert_buffer, lp_ids_buffer, lp_val_buffer,
     lp_ranks_buffer) = _get_decode_core()(
         state=state,
         kv_caches=kv_caches,
         step_rngs=step_rngs,
         sampling_metadata=sampling_metadata,
         inputs_embeds=inputs_embeds,
         lora_metadata=lora_metadata,
         intermediate_tensors=intermediate_tensors,
         block_tables=block_tables,
         query_start_loc=first_attn.query_start_loc,
         request_distribution=first_attn.request_distribution,
         mamba_state_indices=first_attn.mamba_state_indices,
         current_tokens=init_state.current_tokens,
         active_mask=init_state.active_mask,
         input_positions=first_attn.input_positions,
         seq_lens=first_attn.seq_lens,
         model_fn=model_fn,
         compute_logits_fn=compute_logits_fn,
         sample_fn=sample_fn,
         mesh=mesh,
         max_decode_steps=max_decode_steps,
         static_max_decode_steps=static_max_decode_steps,
         eos_token_id=eos_token_id,
         padding_token_id=padding_token_id,
         dp_size=dp_size,
         pad_len=pad_len,
         collect_expert_indices=collect_expert_indices,
         layer_name_to_kvcache_index=layer_name_to_kvcache_index,
         is_first_rank=is_first_rank,
         is_last_rank=is_last_rank,
         max_logprobs=max_logprobs,
         logprobs_mode=logprobs_mode,
         continue_decode_eos_check_interval=continue_decode_eos_check_interval,
         padded_num_reqs=first_attn.padded_num_reqs,
         layer_names_per_group=layer_names_per_group,
     )

    final_state = TpuSamplingState(
        current_tokens=current_tokens,
        active_mask=active_mask,
        attn_metadata=_build_attn_metadata(
            input_positions=positions,
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_start_loc=first_attn.query_start_loc,
            request_distribution=first_attn.request_distribution,
            mamba_state_indices=first_attn.mamba_state_indices,
            padded_num_reqs=first_attn.padded_num_reqs,
            layer_names_per_group=layer_names_per_group,
        ),
        step_counter=step_counter.astype(jnp.int32),
    )

    all_expert_indices = expert_buffer

    logprobs_tensors = None
    if sampling_metadata is not None and sampling_metadata.logprobs:
        logprobs_tensors = LogprobsTensors(
            logprob_token_ids=lp_ids_buffer,
            logprobs=lp_val_buffer,
            selected_token_ranks=lp_ranks_buffer,
        )

    return (token_buffer, kv_caches, final_state, current_rng,
            all_expert_indices, logprobs_tensors)
