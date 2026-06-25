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
"""Tests for the DeepSeek-V4 KV compressor """

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

# isort: off
from tpu_inference.kernels.experimental.deepseek_v4.compress_norm_rope import (
    quantize_fp8_ue8m0,
    shared_sparse_cache_shape,
    unpack_sparse_kv_cache,
)
from tpu_inference.kernels.experimental.deepseek_v4.compressor import compressor_forward
# isort: on

requires_tpu = pytest.mark.skipif(
    jax.devices()[0].platform != "tpu",
    reason="requires a TPU backend",
)

# KV row-slots per shared-cache page (the MLA storage block size).
PAGE_SIZE = 64


def _interleaved_rope_ref(x, cos_sin, rope_head_dim):
    """NumPy interleaved RoPE on the trailing ``rope_head_dim`` elements."""
    head_dim = x.shape[-1]
    half_rope = rope_head_dim // 2
    num_pairs = head_dim // 2
    nope_pairs = num_pairs - half_rope

    pairs = x.reshape(*x.shape[:-1], num_pairs, 2)
    even = pairs[..., 0].copy()
    odd = pairs[..., 1].copy()

    cos = cos_sin[..., :half_rope]
    sin = cos_sin[..., half_rope:rope_head_dim]
    cos_full = np.concatenate(
        [np.ones((*cos.shape[:-1], nope_pairs), x.dtype), cos], axis=-1)
    sin_full = np.concatenate(
        [np.zeros((*sin.shape[:-1], nope_pairs), x.dtype), sin], axis=-1)

    new_even = even * cos_full - odd * sin_full
    new_odd = odd * cos_full + even * sin_full
    out = np.stack([new_even, new_odd], axis=-1)
    return out.reshape(x.shape)


def _boundary_store_ref(
    state_cache,
    positions,
    slot_mapping,
    block_table,
    token_to_req_indices,
    kv_slot_mapping,
    rms_weight,
    cos_sin_cache,
    state_block_size,
    head_dim,
    rope_head_dim,
    compress_ratio,
    overlap,
    rms_eps,
    quant_block,
):
    """Naive NumPy boundary store; returns nope/rope/scale per flat KV slot."""
    coff = 1 + int(overlap)
    state_width = coff * head_dim
    window = coff * compress_ratio
    num_pages = state_cache.shape[0]
    num_slots = num_pages * PAGE_SIZE
    nope_dim = head_dim - rope_head_dim
    n_qb = nope_dim // quant_block
    nope_out = np.zeros((num_slots, nope_dim), dtype=ml_dtypes.float8_e4m3fn)
    rope_out = np.zeros((num_slots, rope_head_dim), dtype=ml_dtypes.bfloat16)
    scale_out = np.zeros((num_slots, n_qb), dtype=np.float32)

    for t in range(positions.shape[0]):
        if slot_mapping[t] < 0 or kv_slot_mapping[t] < 0:
            continue
        if (int(positions[t]) + 1) % compress_ratio != 0:
            continue

        req = int(token_to_req_indices[t])
        start = int(positions[t]) - window + 1

        kv_win = np.zeros((window, head_dim), np.float32)
        score_win = np.full((window, head_dim), -np.inf, np.float32)
        for w in range(window):
            p = start + w
            if p < 0:
                continue
            bn = int(block_table[req, p // state_block_size])
            bo = p % state_block_size
            head_off = head_dim if w >= compress_ratio else 0
            kv_win[w] = state_cache[bn, bo, head_off:head_off + head_dim]
            score_win[w] = state_cache[bn, bo,
                                       state_width + head_off:state_width +
                                       head_off + head_dim]

        m = np.max(score_win, axis=0, keepdims=True)
        e = np.exp(score_win - m)
        weights = e / np.sum(e, axis=0, keepdims=True)
        compressed_kv = np.sum(weights * kv_win, axis=0)

        variance = np.mean(compressed_kv**2)
        normed = compressed_kv / np.sqrt(variance + rms_eps) * rms_weight

        compressed_pos = (int(positions[t]) // compress_ratio) * compress_ratio
        cos_sin = cos_sin_cache[compressed_pos]
        rotated = _interleaved_rope_ref(normed, cos_sin, rope_head_dim)

        nope = rotated[:nope_dim]
        rope = rotated[nope_dim:]

        q, scale = quantize_fp8_ue8m0(jnp.asarray(nope[None]), quant_block)
        q = np.asarray(q[0]).astype(ml_dtypes.float8_e4m3fn)
        scale = np.asarray(scale[0]).astype(np.float32)
        rope_q = rope.astype(ml_dtypes.bfloat16)

        slot = int(kv_slot_mapping[t])
        nope_out[slot] = q
        rope_out[slot] = rope_q
        scale_out[slot] = scale
    return nope_out, rope_out, scale_out


def _make_inputs(
    compress_ratio,
    overlap,
    head_dim=512,
    rope_head_dim=64,
    quant_block=64,
    hidden_size=256,
    num_reqs=2,
    seq_len=None,
    num_pad=0,
    seed=0,
):
    """Build a consistent batch: compressor slots follow the block table."""
    rng = np.random.default_rng(seed)
    coff = 1 + int(overlap)
    state_width = coff * head_dim
    state_block_size = 4 if compress_ratio == 4 else 8

    if seq_len is None:
        seq_len = 2 * compress_ratio
    num_tokens = num_reqs * seq_len

    max_blocks = (seq_len + state_block_size - 1) // state_block_size
    num_state_pages = num_reqs * max_blocks + 1  # +1 spare (page 0)

    # Compressed-KV output pages follow the state pages in the same buffer.
    num_kv_pages = (num_tokens // PAGE_SIZE) + 4
    num_pages = num_state_pages + num_kv_pages

    # Per-request contiguous physical state pages (page 0 left as spare).
    block_table = np.zeros((num_reqs, max_blocks), np.int32)
    nxt = 1
    for r in range(num_reqs):
        for b in range(max_blocks):
            block_table[r, b] = nxt
            nxt += 1

    positions = np.concatenate(
        [np.arange(seq_len, dtype=np.int32) for _ in range(num_reqs)])
    token_to_req_indices = np.repeat(np.arange(num_reqs, dtype=np.int32),
                                     seq_len)

    # Compressor (state) slot = physical slot of this token's logical position,
    # so the stage-2 window gather reads back exactly what stage 1 wrote.
    slot_mapping = np.empty(num_tokens, np.int32)
    for t in range(num_tokens):
        r = int(token_to_req_indices[t])
        p = int(positions[t])
        slot_mapping[t] = block_table[r, p // state_block_size] \
            * state_block_size + p % state_block_size

    # KV slots live in the KV page range so boundary writes never clobber the
    # state bytes the gather still needs.
    kv_base = num_state_pages * PAGE_SIZE
    kv_capacity = num_kv_pages * PAGE_SIZE
    kv_slot_mapping = (kv_base +
                       rng.permutation(kv_capacity)[:num_tokens]).astype(
                           np.int32)

    if num_pad > 0:
        pad_idx = rng.permutation(num_tokens)[:num_pad]
        slot_mapping[pad_idx] = -1
        kv_slot_mapping[pad_idx] = -1

    hidden_states = rng.standard_normal((num_tokens, hidden_size),
                                        dtype=np.float32)
    # Small projection weights keep the post-GEMM magnitudes reasonable.
    wkv_wgate = (rng.standard_normal(
        (2 * state_width, hidden_size), dtype=np.float32) * 0.05)
    # The projection now happens outside the kernel; feed it the result.
    kv_score = (hidden_states @ wkv_wgate.T).astype(np.float32)
    ape = rng.standard_normal((compress_ratio, state_width), dtype=np.float32)
    norm_weight = rng.standard_normal(head_dim, dtype=np.float32)

    max_pos = seq_len + compress_ratio
    cos_sin_cache = rng.standard_normal((max_pos, rope_head_dim),
                                        dtype=np.float32)

    cache_shape = shared_sparse_cache_shape(num_pages, PAGE_SIZE,
                                            head_dim - rope_head_dim,
                                            rope_head_dim, quant_block)
    cache = np.zeros(cache_shape, dtype=np.uint8)

    return dict(kv_score=kv_score,
                ape=ape,
                norm_weight=norm_weight,
                cos_sin_cache=cos_sin_cache,
                positions=positions,
                slot_mapping=slot_mapping,
                block_table=block_table,
                token_to_req_indices=token_to_req_indices,
                kv_slot_mapping=kv_slot_mapping,
                cache=cache,
                state_block_size=state_block_size,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
                compress_ratio=compress_ratio,
                overlap=overlap,
                rms_eps=1e-6,
                quant_block=quant_block)


def _to_jax(kw):
    return {
        k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
        for k, v in kw.items()
    }


def _save_state_ref(kv, score, kw):
    """Plain-NumPy ``save_partial_states``: build the f32 state view."""
    coff = 1 + int(kw["overlap"])
    state_dim = 2 * coff * kw["head_dim"]
    sb = kw["state_block_size"]
    num_pages = kw["cache"].shape[0]
    ape = kw["ape"].astype(np.float32)
    positions = kw["positions"]
    slot_mapping = kw["slot_mapping"]

    flat = np.zeros((num_pages * sb, state_dim), np.float32)
    for t in range(positions.shape[0]):
        slot = int(slot_mapping[t])
        if slot < 0:
            continue
        score_state = score[t] + ape[int(positions[t]) % kw["compress_ratio"]]
        flat[slot] = np.concatenate([kv[t], score_state])
    return flat.reshape(num_pages, sb, state_dim)


def _dequant(nope, rope, scale, kw):
    """Reconstruct fp32 compressed-KV ``[num_slots, head_dim]`` from a record."""
    nope_dim = kw["head_dim"] - kw["rope_head_dim"]
    n_qb = nope_dim // kw["quant_block"]
    nope = np.asarray(nope).reshape(-1, nope_dim).astype(np.float32)
    rope = np.asarray(rope).reshape(-1, kw["rope_head_dim"]).astype(np.float32)
    scale = np.asarray(scale).reshape(-1, n_qb).astype(np.float32)
    nope_deq = (nope.reshape(-1, n_qb, kw["quant_block"]) *
                scale[:, :, None]).reshape(-1, nope_dim)
    return np.concatenate([nope_deq, rope], axis=-1)


def _naive_reference(kw):
    """Naive end-to-end np ground truth for the whole compressor."""
    coff = 1 + int(kw["overlap"])
    state_width = coff * kw["head_dim"]

    kv_score = kw["kv_score"].astype(np.float32)
    kv = kv_score[:, :state_width]
    score = kv_score[:, state_width:2 * state_width]

    state_cache = _save_state_ref(kv, score, kw)

    nope, rope, scale = _boundary_store_ref(
        state_cache=state_cache,
        positions=kw["positions"],
        slot_mapping=kw["slot_mapping"],
        block_table=kw["block_table"],
        token_to_req_indices=kw["token_to_req_indices"],
        kv_slot_mapping=kw["kv_slot_mapping"],
        rms_weight=kw["norm_weight"],
        cos_sin_cache=kw["cos_sin_cache"],
        state_block_size=kw["state_block_size"],
        head_dim=kw["head_dim"],
        rope_head_dim=kw["rope_head_dim"],
        compress_ratio=kw["compress_ratio"],
        overlap=kw["overlap"],
        rms_eps=kw["rms_eps"],
        quant_block=kw["quant_block"])
    return _dequant(nope, rope, scale, kw)


def _dequant_written(act_cache, kw):
    """Unpack the kernel's shared buffer and dequantize every KV row-slot."""
    nope_dim = kw["head_dim"] - kw["rope_head_dim"]
    nope, rope, scale = unpack_sparse_kv_cache(act_cache, nope_dim,
                                               kw["rope_head_dim"],
                                               kw["quant_block"])
    return _dequant(nope, rope, scale, kw)


def _written_slots(kw):
    """Flat KV slots actually written (valid boundary tokens)."""
    slots = []
    for t in range(kw["positions"].shape[0]):
        if kw["slot_mapping"][t] < 0 or kw["kv_slot_mapping"][t] < 0:
            continue
        if (int(kw["positions"][t]) + 1) % kw["compress_ratio"] != 0:
            continue
        slots.append(int(kw["kv_slot_mapping"][t]))
    return np.array(sorted(set(slots)), dtype=np.int64)


# compress_ratio, overlap, seq_len, num_pad, seed
_CASES = [
    (4, True, 8, 0, 1),
    (4, True, 12, 3, 2),
    (128, False, 256, 0, 3),
    (128, False, 200, 4, 4),
]


@pytest.mark.parametrize("compress_ratio,overlap,seq_len,num_pad,seed", _CASES)
def test_compressor_forward_matches_reference(compress_ratio, overlap, seq_len,
                                              num_pad, seed):
    kw = _make_inputs(compress_ratio,
                      overlap,
                      seq_len=seq_len,
                      num_pad=num_pad,
                      seed=seed)
    ref_deq = _naive_reference(kw)

    act_cache = np.asarray(compressor_forward(**_to_jax(kw)))
    act_deq = _dequant_written(act_cache, kw)

    # Compare the dequantized compressed-KV at the boundary slots actually
    # written.
    slots = _written_slots(kw)
    assert slots.size > 0
    np.testing.assert_allclose(act_deq[slots],
                               ref_deq[slots],
                               rtol=2e-2,
                               atol=2e-2)


def test_compressor_forward_eval_shape():
    """Lowering-only check: traces without executing (no device required)."""
    kw = _make_inputs(4, True, seq_len=8)
    jkw = _to_jax(kw)

    def fn(kv_score, ape, norm_weight, cos_sin_cache, positions, slot_mapping,
           block_table, token_to_req_indices, kv_slot_mapping, cache):
        return compressor_forward(kv_score=kv_score,
                                  ape=ape,
                                  norm_weight=norm_weight,
                                  cos_sin_cache=cos_sin_cache,
                                  positions=positions,
                                  slot_mapping=slot_mapping,
                                  block_table=block_table,
                                  token_to_req_indices=token_to_req_indices,
                                  kv_slot_mapping=kv_slot_mapping,
                                  cache=cache,
                                  state_block_size=kw["state_block_size"],
                                  head_dim=kw["head_dim"],
                                  rope_head_dim=kw["rope_head_dim"],
                                  compress_ratio=kw["compress_ratio"],
                                  overlap=kw["overlap"],
                                  rms_eps=kw["rms_eps"],
                                  quant_block=kw["quant_block"])

    out_cache = jax.eval_shape(fn, jkw["kv_score"], jkw["ape"],
                               jkw["norm_weight"], jkw["cos_sin_cache"],
                               jkw["positions"], jkw["slot_mapping"],
                               jkw["block_table"], jkw["token_to_req_indices"],
                               jkw["kv_slot_mapping"], jkw["cache"])
    assert out_cache.shape == kw["cache"].shape
    assert out_cache.dtype == jnp.uint8


@requires_tpu
def test_compressor_forward_runs_on_tpu():
    """Executes on TPU and confirms the backend really is TPU."""
    kw = _make_inputs(128, False, seq_len=256, num_pad=4, seed=7)
    ref_deq = _naive_reference(kw)

    act = compressor_forward(**_to_jax(kw))
    act.block_until_ready()
    assert act.devices().pop().platform == "tpu"

    act_deq = _dequant_written(np.asarray(act), kw)
    slots = _written_slots(kw)
    assert slots.size > 0
    np.testing.assert_allclose(act_deq[slots],
                               ref_deq[slots],
                               rtol=2e-2,
                               atol=2e-2)
