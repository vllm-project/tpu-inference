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
"""The flywheel backend on CPU: its host-side scheduling, and attention()
dispatch with the kernel stubbed."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheGroupSpec,
                                        MambaSpec)

from tpu_inference.layers.common import attention_interface
from tpu_inference.layers.common import flywheel_attention as fa
from tpu_inference.layers.common.attention_backend import HostStepInputs
from tpu_inference.layers.common.attention_metadata import (
    AttentionMetadata, SharedAttentionMetadata)
from tpu_inference.runner import kv_cache as kv_cache_mod
from tpu_inference.utils import DeviceBuffer

TOTAL_TOKENS, NUM_SEQS, MAX_NUM_SEQS = 10, 2, 4
NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 24, 4, 256
NUM_BLOCKS, BLOCK_SIZE, MAX_BLOCKS_PER_SEQ = 32, 128, 8
KV_WRITE_BOUND = fa.kv_write_slice_bound(TOTAL_TOKENS, MAX_NUM_SEQS,
                                         BLOCK_SIZE)


@pytest.fixture
def mesh() -> Mesh:
    """One CPU device over the axes the dispatch path reads."""
    devices = np.array([jax.devices("cpu")[0]])
    return Mesh(devices.reshape((-1, 1, 1)), ("data", "attn_dp", "model"))


def build_metadata() -> tuple[AttentionMetadata, SharedAttentionMetadata]:
    """Metadata for a two-request prefill step of TOTAL_TOKENS tokens."""
    seq_lens = jnp.array([5, 5, 0, 0], jnp.int32)
    query_start_loc = jnp.array([0, 5, 10, 10, 10], jnp.int32)
    distribution = jnp.array([0, 0, NUM_SEQS], jnp.int32)
    md = AttentionMetadata(input_positions=jnp.arange(TOTAL_TOKENS,
                                                      dtype=jnp.int32),
                           block_tables=jnp.zeros(
                               (MAX_NUM_SEQS * MAX_BLOCKS_PER_SEQ, ),
                               jnp.int32),
                           seq_lens=seq_lens,
                           query_start_loc=query_start_loc,
                           request_distribution=distribution,
                           backend_metadata=fa.FlywheelMetadata(
                               kv_write_slices=jnp.zeros(
                                   (3 * KV_WRITE_BOUND, ), jnp.int32),
                               kv_write_num_slices=jnp.zeros((1, ), jnp.int32),
                               kernel_index=jnp.zeros((1, ), jnp.int32)))
    shared = SharedAttentionMetadata(input_positions=jnp.arange(
        TOTAL_TOKENS, dtype=jnp.int32),
                                     seq_lens=seq_lens,
                                     query_start_loc=query_start_loc,
                                     request_distribution=distribution)
    return md, shared


class TestDispatch:

    def run_attention(
            self, mesh: Mesh, monkeypatch: pytest.MonkeyPatch,
            **overrides: object
    ) -> tuple[MagicMock, tuple[jax.Array, jax.Array]]:
        """attention() over a stubbed sharded wrapper; returns the stub and
        what attention() returned."""
        q = jnp.ones((TOTAL_TOKENS, NUM_HEADS, HEAD_DIM), jnp.bfloat16)
        k = jnp.ones((TOTAL_TOKENS, NUM_KV_HEADS, HEAD_DIM), jnp.bfloat16)
        v = jnp.ones((TOTAL_TOKENS, NUM_KV_HEADS, HEAD_DIM), jnp.bfloat16)
        backend = fa.FlywheelAttentionBackend()
        kv_cache = jnp.zeros(
            kv_cache_mod.get_kv_cache_shape_with_mesh(
                mesh,
                NUM_BLOCKS,
                BLOCK_SIZE,
                NUM_KV_HEADS,
                HEAD_DIM,
                jnp.bfloat16,
                attention_backend=backend), jnp.bfloat16)
        wrapper = MagicMock(return_value=(jnp.ones(q.shape, q.dtype),
                                          kv_cache))
        monkeypatch.setattr(fa, "sharded_flywheel_tpu_attention", wrapper)
        md, shared = build_metadata()
        result = attention_interface.attention(
            kv_cache=kv_cache,
            q=q,
            k=k,
            v=v,
            attention_metadata=md,
            mesh=mesh,
            shared_attention_metadata=shared,
            attention_backend=backend,
            **overrides)
        return wrapper, result

    def test_routes_to_flywheel_wrapper(self, mesh, monkeypatch):
        wrapper, (kv_cache, output) = self.run_attention(mesh, monkeypatch)
        wrapper.assert_called_once()
        args, kwargs = wrapper.call_args
        *_, kv_write_slices, kv_write_num_slices, distribution, kernel_index = args
        assert args[0] is mesh
        assert len(args) == 12
        assert kernel_index.shape == (1, )
        assert kv_write_slices.shape == (3 * KV_WRITE_BOUND, )
        assert kv_write_num_slices.shape == (1, )
        np.testing.assert_array_equal(distribution, [0, 0, NUM_SEQS])
        assert kwargs == {"sm_scale": HEAD_DIM**-0.5}
        assert output.shape == (TOTAL_TOKENS, NUM_HEADS, HEAD_DIM)
        assert kv_cache.shape == (NUM_BLOCKS, BLOCK_SIZE, 2 * NUM_KV_HEADS,
                                  HEAD_DIM)

    @pytest.mark.parametrize(("option", "value", "expected"), [
        ("sinks", jnp.ones((NUM_HEADS, ), jnp.float32), "sinks"),
        ("attention_chunk_size", 512, "attention_chunk_size"),
        ("q_scale", 0.5, "q_scale"),
    ])
    def test_rejects_unsupported_features(self, mesh, monkeypatch, option,
                                          value, expected):
        with pytest.raises(NotImplementedError) as excinfo:
            self.run_attention(mesh, monkeypatch, **{option: value})
        assert expected in str(excinfo.value)


def make_vllm_config(*,
                     block_size: int = 16,
                     user_specified_block_size: bool = False,
                     dtype: torch.dtype = torch.bfloat16,
                     head_size: int = HEAD_DIM) -> SimpleNamespace:
    """The vllm_config fields FlywheelAttentionBackend.check_config reads."""
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            user_specified_block_size=user_specified_block_size,
            cache_dtype="auto"),
        model_config=SimpleNamespace(use_mla=False,
                                     dtype=dtype,
                                     get_head_size=lambda: head_size),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1,
                                        prefill_context_parallel_size=1),
        speculative_config=None,
        kv_transfer_config=None,
        additional_config={})


class TestCheckConfig:

    def test_rounds_the_block_size_up_to_whole_pages(self):
        vllm_config = make_vllm_config(block_size=16)

        fa.FlywheelAttentionBackend().check_config(vllm_config)

        assert vllm_config.cache_config.block_size == fa.PAGE_SIZE_MULTIPLE

    @pytest.mark.parametrize(("overrides", "expected"), [
        ({
            "block_size": 16,
            "user_specified_block_size": True
        }, "block size 16"),
        ({
            "dtype": torch.float16
        }, "not bfloat16"),
        ({
            "head_size": 64
        }, "head_dim 64"),
    ])
    def test_rejects_what_the_kernels_cannot_serve(self, overrides, expected):
        with pytest.raises(ValueError, match=expected):
            fa.FlywheelAttentionBackend().check_config(
                make_vllm_config(**overrides))


def naive_kv_write_slices(cu_q_lens: list[int], kv_lens: list[int],
                          block_table: np.ndarray,
                          page_size: int) -> list[list[int]]:
    """build_kv_write_slices token by token: (cache_row, new_row, length)
    runs of new tokens, a new run at every page start."""
    pieces: list[list[int]] = []
    for req, kv_len in enumerate(kv_lens):
        q_len = cu_q_lens[req + 1] - cu_q_lens[req]
        for pos in range(kv_len - q_len, kv_len):
            page_slot, offset = divmod(pos, page_size)
            cache_row = int(block_table[req, page_slot]) * page_size + offset
            new_row = cu_q_lens[req] + pos - (kv_len - q_len)
            if (pieces and offset != 0
                    and pieces[-1][0] + pieces[-1][2] == cache_row
                    and pieces[-1][1] + pieces[-1][2] == new_row):
                pieces[-1][2] += 1
            else:
                pieces.append([cache_row, new_row, 1])
    return pieces


@pytest.mark.parametrize("page_size", [128, 256])
@pytest.mark.parametrize("seed", range(5))
def test_kv_write_slices_match_a_token_by_token_reference(page_size, seed):
    rng = np.random.default_rng(seed)
    num_reqs, pages_per_seq = 6, 8
    q_lens = rng.integers(0, 3 * page_size, num_reqs)
    q_lens[:2] = 1
    kv_lens = q_lens + rng.integers(0, 4 * page_size, num_reqs)
    cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    block_table = rng.permutation(num_reqs * pages_per_seq).reshape(
        num_reqs, pages_per_seq).astype(np.int32)
    bound = fa.kv_write_slice_bound(int(cu_q_lens[-1]), num_reqs, page_size)
    out = np.full((3, bound), -1, np.int32)

    count = fa.build_kv_write_slices(cu_q_lens,
                                     kv_lens.astype(np.int32),
                                     block_table,
                                     page_size=page_size,
                                     out=out)

    expected = naive_kv_write_slices(cu_q_lens.tolist(), kv_lens.tolist(),
                                     block_table, page_size)
    assert count == len(expected) <= bound
    np.testing.assert_array_equal(out[:, :count].T, np.array(expected))
    np.testing.assert_array_equal(out[:, count:], 0)


@pytest.mark.parametrize(("num_decode", "seq_lens", "phase"), [
    (2, [(1, 300), (1, 129)], fa._DECODE_PHASE),
    (0, [(300, 300), (129, 129)], fa._PREFILL_PHASE),
    (1, [(1, 300), (129, 129)], fa._EXTEND_PHASE),
    (0, [(200, 457)], fa._EXTEND_PHASE),
])
def test_attention_phase(num_decode, seq_lens, phase):
    q_lens = [q_len for q_len, _ in seq_lens]
    cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)])
    kv_lens = np.array([kv_len for _, kv_len in seq_lens])

    assert fa.select_attention_phase(num_decode, cu_q_lens, kv_lens) == phase


def test_attention_phase_needs_decode_rows_first():
    with pytest.raises(ValueError, match="decode requests first"):
        fa.select_attention_phase(1, np.array([0, 5, 6]), np.array([5, 9]))


@pytest.mark.parametrize(
    ("scheduled_tokens_per_dp_rank", "bucket", "dense"),
    [
        ([[1000]], 1024, True),
        ([[100]], 1024, True),
        ([[1000], [700]], 1024, True),
        ([[500, 500]], 1024, False),
        ([[1000], []], 1024, False),
        # The dense kernel cannot tile these buckets.
        ([[64]], 64, False),
        ([[3000]], 3072, False),
    ])
def test_dense_prefill_for_one_request_per_rank(scheduled_tokens_per_dp_rank,
                                                bucket, dense):
    span = fa.varlen_block_span(scheduled_tokens_per_dp_rank, bucket)

    assert fa.use_dense_prefill(scheduled_tokens_per_dp_rank, bucket,
                                span) is dense


@pytest.mark.parametrize(
    ("scheduled_tokens_per_dp_rank", "bucket", "span"),
    [
        ([[128] * 16], 2048, 256),
        ([[2048]], 2048, 1024),
        # The DP ranks run concurrently and the step costs the slowest
        # rank, so together they pick a span neither picks alone.
        ([[128] * 16, [2048]], 2048, 512),
        ([[100]], 128, 128),
    ])
def test_varlen_block_span(scheduled_tokens_per_dp_rank, bucket, span):
    assert fa.varlen_block_span(scheduled_tokens_per_dp_rank, bucket) == span
    assert span in fa.varlen_block_spans(bucket)


def test_varlen_block_span_tie_goes_to_the_wider_span(monkeypatch):
    monkeypatch.setattr(fa, "ragged_step_cost", lambda *args: 1)

    assert fa.varlen_block_span([[1000]], 2048) == 2048


def test_block_sizes_come_from_flywheel():
    # The cost model was fitted with these, so a flywheel-tpu change to
    # FWD_BLOCKS should be a deliberate change here as well.
    assert fa._VARLEN_BLOCK_SPANS == (256, 512, 1024, 2048)
    assert (fa._DENSE_PAD_MULTIPLE, fa._DENSE_MAX_BLOCK) == (128, 2048)


PAGE_SIZES = (128, 256)
MAX_REQS_PER_RANK, PAGES_PER_REQ = 4, 8


def kv_cache_groups() -> list[KVCacheGroupSpec]:
    """Two attention groups of different page sizes, and a mamba group."""
    attention_groups = [
        KVCacheGroupSpec(layer_names=[f"attn_{gid}"],
                         kv_cache_spec=FullAttentionSpec(
                             block_size=page_size,
                             num_kv_heads=NUM_KV_HEADS,
                             head_size=HEAD_DIM,
                             dtype=torch.bfloat16))
        for gid, page_size in enumerate(PAGE_SIZES)
    ]
    mamba_group = KVCacheGroupSpec(layer_names=["mamba"],
                                   kv_cache_spec=MambaSpec(
                                       block_size=1,
                                       shapes=((1, ), ),
                                       dtypes=(torch.float32, )))
    return [*attention_groups, mamba_group]


def host_step(seq_lens_per_rank: list[list[tuple[int, int]]],
              bucket: int) -> HostStepInputs:
    """The runner's host arrays for a step of (q_len, kv_len) requests per DP
    rank, decode rows first, laid out per rank as _prepare_inputs does."""
    dp_size = len(seq_lens_per_rank)
    query_start_loc = np.zeros(dp_size * (MAX_REQS_PER_RANK + 1), np.int32)
    seq_lens = np.zeros(dp_size * MAX_REQS_PER_RANK, np.int32)
    request_distribution = np.zeros(dp_size * 3, np.int32)
    for rank, requests in enumerate(seq_lens_per_rank):
        q_lens = [q_len for q_len, _ in requests]
        cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)])
        query_offset = rank * (MAX_REQS_PER_RANK + 1)
        query_start_loc[query_offset:query_offset + MAX_REQS_PER_RANK +
                        1] = cu_q_lens[-1]
        query_start_loc[query_offset:query_offset + len(cu_q_lens)] = cu_q_lens
        req_offset = rank * MAX_REQS_PER_RANK
        seq_lens[req_offset:req_offset +
                 len(requests)] = [kv_len for _, kv_len in requests]
        num_decode = sum(q_len == 1 and kv_len > 1
                         for q_len, kv_len in requests)
        request_distribution[3 * rank:3 * rank + 3] = (num_decode, num_decode,
                                                       len(requests))
    rng = np.random.default_rng(0)
    num_pages = dp_size * MAX_REQS_PER_RANK * PAGES_PER_REQ
    return HostStepInputs(
        kv_cache_groups=kv_cache_groups(),
        block_tables={
            gid: rng.permutation(num_pages).reshape(-1, PAGES_PER_REQ)
            for gid in range(len(PAGE_SIZES))
        },
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        request_distribution=request_distribution,
        scheduled_tokens_per_dp_rank={
            rank: [q_len for q_len, _ in requests]
            for rank, requests in enumerate(seq_lens_per_rank)
        },
        dp_size=dp_size,
        max_num_reqs_per_dp_rank=MAX_REQS_PER_RANK,
        padded_num_scheduled_tokens_per_dp_rank=bucket)


def prepare(step: HostStepInputs) -> dict[str, jax.Array]:
    """prepare_step, and the device buffer unpacked as the runner does."""
    device_buffer = DeviceBuffer()
    fa.FlywheelAttentionBackend().prepare_step(step, device_buffer)
    blob, layout = device_buffer.build()
    return DeviceBuffer.unpack_arrays(jnp.asarray(blob), layout)


def step_kernel(arrays: dict[str, jax.Array], bucket: int) -> str | int:
    """The kernel prepare_step picked, the same on every DP rank."""
    (index, ) = set(np.asarray(arrays[fa._KERNEL_INDEX_KEY]).tolist())
    return fa.step_kernels(bucket)[index]


class TestPrepareStep:

    def test_dp_ranks_that_disagree_run_extend(self):
        step = host_step([[(1, 300), (1, 129)], [(256, 256)]], bucket=256)

        arrays = prepare(step)

        assert step_kernel(arrays, 256) == fa._EXTEND_PHASE

    def test_each_attention_group_gets_its_ranks_slice_tables(self):
        seq_lens_per_rank = [[(1, 300), (1, 129)], [(40, 700), (200, 457)]]
        step = host_step(seq_lens_per_rank, bucket=256)

        arrays = prepare(step)

        backend = fa.FlywheelAttentionBackend()
        assert backend.backend_metadata(arrays, len(PAGE_SIZES)) is None
        for gid, page_size in enumerate(PAGE_SIZES):
            md = backend.backend_metadata(arrays, gid)
            assert isinstance(md, fa.FlywheelMetadata)
            bound = fa.kv_write_slice_bound(256, MAX_REQS_PER_RANK, page_size)
            slices = np.asarray(md.kv_write_slices).reshape(2, 3, bound)
            for rank, requests in enumerate(seq_lens_per_rank):
                q_lens = [q_len for q_len, _ in requests]
                # The decode kernel appends the decode rows itself.
                num_decode = int(step.request_distribution[3 * rank])
                req_offset = rank * MAX_REQS_PER_RANK + num_decode
                expected = naive_kv_write_slices(
                    [0, *np.cumsum(q_lens).tolist()][num_decode:],
                    [kv_len for _, kv_len in requests][num_decode:],
                    step.block_tables[gid][req_offset:], page_size)
                assert md.kv_write_num_slices[rank] == len(expected)
                np.testing.assert_array_equal(
                    slices[rank][:, :len(expected)].T,
                    np.array(expected, np.int32).reshape(-1, 3))

    @pytest.mark.parametrize(("seq_lens_per_rank", "kernel"), [
        ([[(1000, 1000)], [(700, 700)]], fa._DENSE_KERNEL),
        ([[(500, 500), (500, 500)], [(700, 700)]], 512),
        ([[(1, 300), (1, 129)], [(1, 50)]], fa._DECODE_PHASE),
    ])
    def test_picks_its_kernel_on_the_host(self, seq_lens_per_rank, kernel):
        arrays = prepare(host_step(seq_lens_per_rank, bucket=1024))

        assert step_kernel(arrays, 1024) == kernel

    # A decode step's bucket is the smallest one that fits a full
    # request batch, 2 * MAX_REQS_PER_RANK tokens over both ranks.
    @pytest.mark.parametrize(("seq_lens_per_rank", "bucket"), [
        ([[(1, 300), (1, 129)], [(1, 50)]], 8),
        ([[(1, 300)], [(256, 256)]], 1024),
        ([[(1000, 1000)], [(700, 700)]], 1024),
        ([[(500, 500), (500, 500)], [(700, 700)]], 1024),
        ([[(128, 128)] * 4, [(1000, 1000)]], 1024),
    ])
    def test_the_warmup_primes_the_step(self, mesh, seq_lens_per_rank, bucket):
        arrays = prepare(host_step(seq_lens_per_rank, bucket))

        backend = fa.FlywheelAttentionBackend()
        warmup_arrays = backend.warmup_arrays(
            kv_cache_groups(),
            num_tokens=2 * bucket,
            max_num_reqs=2 * MAX_REQS_PER_RANK,
            dp_size=2,
            mesh=mesh,
            sharding=NamedSharding(mesh, P()))
        # Every group's metadata has the shapes serving passes, so the
        # bucket's one warmed up backbone serves the step.
        for gid in range(len(PAGE_SIZES) + 1):
            assert (jax.tree.map(
                jnp.shape,
                backend.backend_metadata(arrays, gid)) == jax.tree.map(
                    jnp.shape, backend.backend_metadata(warmup_arrays, gid)))
        assert step_kernel(arrays, bucket) in fa.step_kernels(bucket)
