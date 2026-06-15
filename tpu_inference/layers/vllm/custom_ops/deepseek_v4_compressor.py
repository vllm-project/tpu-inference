# pytype: skip-file
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPU-compatible DeepSeek-V4 KV/score compressor."""

import functools
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import PartitionSpec as P
from torchax.interop import jax_view
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4 import compressor as dsv4_compressor
from vllm.models.deepseek_v4.compressor import (CompressorStateCache,
                                                DeepseekCompressor)
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

from tpu_inference.kernels.experimental.deepseek_v4.compressor import (
    compressor_forward, compressor_forward_indexer)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)


def _raise_on_block_table_overlap(state_pages: jax.Array, other_pages: tuple,
                                  *, state_prefix: str,
                                  other_prefixes: tuple) -> None:
    """Host-side guard: crash if overlaid caches share live physical pages.

    The state cache overlays the *same* underlying buffer as one or more other
    caches (see the ``id(state_cache) == id(cache)`` assert in
    ``VllmDeepseekCompressor.forward``). Every cache draws its own block table
    from its attention metadata, and those tables must map to disjoint physical
    pages; if a page id is live in both the state cache and any cache sharing
    the buffer, one is about to clobber the other and silently corrupt the KV
    cache. Raising here (via ``jax.debug.callback``) propagates out of the
    dispatch and crashes the server, which is the intended fail-fast behavior.

    ``state_pages`` / ``other_pages`` are per-(ATTN_DATA)-shard live page-id
    arrays already masked to valid requests, with ``0`` (vLLM's null block)
    marking entries to ignore. They are computed inside the ``shard_map`` so the
    valid-request prefix is this shard's local count — a forward-level read of
    ``request_distribution[2]`` would only see shard 0's count under DP.
    """
    state_live = np.asarray(state_pages)
    state_live = np.unique(state_live[state_live > 0])
    # TODO(gxd): check any 2 bt not overlap
    for prefix, pages in zip(other_prefixes, other_pages):
        other_live = np.asarray(pages)
        other_live = np.unique(other_live[other_live > 0])
        overlap = np.intersect1d(state_live, other_live)
        if overlap.size:
            raise RuntimeError(
                f"DSV4 compressor: the state cache ({state_prefix!r}) and the "
                f"cache {prefix!r} overlay the same buffer but their block "
                f"tables share live page ids: {overlap.tolist()}. This "
                "corrupts the KV cache; aborting.\n"
                f"  state_live ({state_prefix!r}): {state_live.tolist()}\n"
                f"  other_live ({prefix!r}): {other_live.tolist()}")
    # print("gxd _raise_on_block_table_overlap ok ", str(len(other_pages)))


def _print_state_block_table(block_table: jax.Array,
                             num_valid_reqs: jax.Array) -> None:
    """Host-side print of the first 10 page ids of each valid request's row.

    ``block_table`` is this shard's local ``[max_num_seqs, max_blocks]`` state
    block table; only the first ``num_valid_reqs`` rows are real requests.
    """
    n = int(num_valid_reqs)
    bt = np.asarray(block_table)[:n, :10]
    # if n > 0:
    #     print(f"gxd state_cache block_table valid seqs [:10] (num_valid={n}):\n{bt}")



class VllmCompressorStateCache(CompressorStateCache):
    """TPU-compatible compressor state cache.

  The base ``get_kv_cache_spec`` returns a ``SlidingWindowMLASpec`` describing
  the CUDA/FlashMLA paged layout, which does not apply on TPU. The TPU KV-cache
  spec is handled elsewhere, so here ``get_kv_cache_spec`` is a no-op.
  """
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ):
        super().__init__(
                state_dim,
                dtype,
                compress_ratio,
                prefix,
            )
        coff = 1 + (compress_ratio == 4)
        self.head_dim = state_dim // 2 // coff
        compressed_kv_cache_bz = get_current_vllm_config().cache_config.block_size // compress_ratio
        assert compressed_kv_cache_bz > 0
        print("gxd VllmCompressorStateCache ", compressed_kv_cache_bz)
        self.block_size = compressed_kv_cache_bz // 4 // 2 // coff
        print(prefix, "gxd block_size ", self.block_size, " head_dim: ", self.head_dim, " sliding-window: ", self.sliding_window)

    # pylint: disable=unused-argument
    def get_kv_cache_spec(self, vllm_config):
        #TODO: gxd check if this need changes
        print("gxd VllmCompressorStateCache get_kv_cache_spec: " + str(self.head_dim))
        return SlidingWindowMLASpec(  # only has one vector instead of K + V
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=None,
        )


class VllmDeepseekCompressor(DeepseekCompressor):
    """TPU-compatible DeepSeek-V4 compressor.

  The base ``DeepseekCompressor`` forward dispatches to CUDA/triton kernels
  (``save_partial_states`` / ``compress_norm_rope_store_*``) that are not
  available on TPU. The TPU-native compress/store path is handled elsewhere,
  so here ``forward`` is a passthrough no-op.
  """

    def __init__(self, *args, **kwargs):
        # The base ctor builds ``self.state_cache`` from the stock
        # ``CompressorStateCache``, whose ``get_kv_cache_spec`` returns a
        # CUDA/FlashMLA spec that does not apply on TPU. Patch the module-level
        # symbol the base ctor references so it constructs the TPU subclass
        # directly, instead of rebuilding/re-registering the state cache after
        # the fact.
        with mock.patch.object(dsv4_compressor, "CompressorStateCache",
                               VllmCompressorStateCache):
            super().__init__(*args, **kwargs)

    # pylint: disable=unused-argument
    def forward(
        self,
        # [num_tokens, 2 * self.coff * self.head_dim]
        kv_score: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:

        # head_dim == 512: sparse CSA/HCA main path -> compressor_forward.
        # head_dim == 128: lightning-indexer path  -> compressor_forward_indexer.
        # Both share the same call shape; only the inner store kernel differs.
        assert self.head_dim in (512, 128)
        forward_fn = (compressor_forward
                      if self.head_dim == 512 else compressor_forward_indexer)
        kv_score = kv_score.to(torch.float32)

        attn_metadata = get_forward_context().attn_metadata
        # State-cache metadata carries the per-token scatter indices; the k_cache
        # layer owns the physical buffer the state and compressed-KV share.
        state_metadata = attn_metadata[self.state_cache.prefix]
        k_cache_metadata = attn_metadata[self.k_cache_prefix]

        wrapper_ctx = get_vllm_model_wrapper_context()
        mesh = wrapper_ctx.mesh
        cache_index = wrapper_ctx.layer_name_to_kvcache_index[
            self.k_cache_prefix]
        # [num_pages, page_size//4, 4, width] uint8 — shared state + KV buffer.
        cache = wrapper_ctx.kv_caches[cache_index]
        
        state_cache_index = wrapper_ctx.layer_name_to_kvcache_index[self.state_cache.prefix]
        state_cache = wrapper_ctx.kv_caches[state_cache_index]
        # state cache and the compressed kv cache overlay on the same buffer
        assert id(state_cache) == id(cache)
        # TODO: we should let HCA's state cache overlay on CSA's main kv cache, which is
        # bigger


        # multiple caches mapped to the same buffer, find their indices.
        all_caches_indices = []
        # TODO: all_caches_indices size must be 1, don't need the loop
        for i, c in enumerate(wrapper_ctx.kv_caches):
            if id(c) == id(cache):
                all_caches_indices.append(i)

        # Guard: every other cache overlaying the same physical buffer as the
        # state cache must use disjoint pages, else they clobber each other.
        # Collect one representative layer prefix per *other* sharing cache
        # (kvcache index) that has attention metadata. Their block tables are
        # threaded into the shard_map below so the overlap check runs per
        # ATTN_DATA shard with a consistent (shard-local) valid-request count;
        # a forward-level check would only see shard 0's request_distribution.
        index_to_prefix: dict[int, list] = {}
        for name, idx in wrapper_ctx.layer_name_to_kvcache_index.items():
            assert name in attn_metadata
            if idx not in index_to_prefix:
                index_to_prefix[idx] = []
                index_to_prefix[idx].append(name)
            else:
                index_to_prefix[idx].append(name)

        other_prefixes = []
        other_block_tables = []
        # for idx, prefixes in index_to_prefix.items():
        #     if idx != state_cache_index:
        #         continue
        #     for p in prefixes:
        #         if p == self.state_cache.prefix:
        #             continue
        #         other_prefixes.append(p)
        #         other_block_tables.append(attn_metadata[p].block_tables)
        # print("gxd other_block_tables, ", len(other_block_tables))
        # print("gxd colocate, ", self.state_cache.prefix, " ", other_prefixes)


        # Array inputs are sharded on their leading axis along ATTN_DATA; the
        # shared cache is sharded along BATCH; weights/tables are replicated.
        # The per-token scatter indices are derived inside the shard_map so each
        # ATTN_DATA shard computes its slot mappings and masking from its local
        # metadata slice (cf. deepseek_v4_indexer.py).
        data_spec = P(ShardingAxisName.ATTN_DATA)
        cache_spec = P(ShardingAxisName.BATCH)
        in_specs = (
            data_spec,  # kv_score  [num_tokens, 2 * self.coff * self.head_dim]
            P(),  # ape                 [compress_ratio, coff*head_dim]
            P(),  # norm_weight         [head_dim]
            P(),  # cos_sin_cache       [max_pos, rope_head_dim]
            data_spec,  # positions             [num_tokens]
            data_spec,  # request_distribution
            data_spec,  # query_start_loc       [num_reqs + 1]
            data_spec,  # seq_lens              [num_reqs]
            data_spec,  # state_input_positions [num_tokens]
            data_spec,  # state_block_tables    [num_reqs * max_blocks]
            data_spec,  # k_input_positions     [num_tokens]
            data_spec,  # k_block_tables        [num_reqs * max_blocks]
            cache_spec,  # cache
            tuple(data_spec for _ in other_block_tables),  # other_block_tables
        )
        out_specs = cache_spec

        def _compress(kv_score, ape, norm_weight, cos_sin_cache, positions,
                      request_distribution, query_start_loc, seq_lens,
                      state_input_positions, state_block_tables,
                      k_input_positions, k_block_tables, cache,
                      other_block_tables_local):
            num_valid_reqs = request_distribution[2]
            num_tokens = kv_score.shape[0]
            num_valid_tokens = query_start_loc[num_valid_reqs]

            q_per_req = query_start_loc[1:] - query_start_loc[:-1]
            max_num_seqs = seq_lens.shape[0]

            # # DEBUG (per ATTN_DATA shard): first 10 page ids of each valid
            # # request's row of the state cache's block table. request_distribution,
            # # seq_lens, and state_block_tables are all this shard's local slice,
            # # so num_valid_reqs and the reshape are self-consistent. Slicing to
            # # valid rows is done on the host (dynamic shape), hence the callback.
            # jax.debug.callback(_print_state_block_table,
            #                    state_block_tables.reshape(max_num_seqs, -1),
            #                    num_valid_reqs)

            # # Per-shard overlap guard: the state cache and every other cache
            # # overlaying the same physical buffer must use disjoint live pages.
            # # Computing live pages here (not at the forward level) makes the
            # # valid-request prefix this shard's local count, so the check is
            # # correct under ATTN_DATA data parallelism. Live page = a block-table
            # # entry of a valid request that is ``> 0`` (block 0 is the null
            # # block); non-live entries are zeroed and ignored on the host.
            # if other_prefixes:

            #     def _live_page_ids(block_tables):
            #         bt = block_tables.reshape(max_num_seqs, -1)
            #         valid = (jnp.arange(max_num_seqs)[:, None] < num_valid_reqs
            #                  ) & (bt > 0)
            #         return jnp.where(valid, bt, 0).reshape(-1)

            #     jax.debug.callback(
            #         functools.partial(_raise_on_block_table_overlap,
            #                           state_prefix=self.state_cache.prefix,
            #                           other_prefixes=tuple(other_prefixes)),
            #         _live_page_ids(state_block_tables),
            #         tuple(
            #             _live_page_ids(bt)
            #             for bt in other_block_tables_local))

            token_to_req_indices = jnp.repeat(
                jnp.arange(max_num_seqs), q_per_req,
                total_repeat_length=num_tokens)
            token_to_req_indices = jnp.where(
                jnp.arange(num_tokens) < num_valid_tokens,
                token_to_req_indices, 0)

            state_block_size = self.state_cache.block_size
            assert state_block_tables.shape[0] % max_num_seqs == 0
            max_num_blocks_per_req = state_block_tables.shape[0] // max_num_seqs
            state_cache_block_num = state_input_positions // state_block_size + token_to_req_indices * max_num_blocks_per_req
            state_cache_block_offset = state_input_positions % state_block_size
            slot_mapping = (state_block_tables[state_cache_block_num] *
                            state_block_size + state_cache_block_offset)
            slot_mapping = jnp.where(
                jnp.arange(num_tokens) < num_valid_tokens, slot_mapping, -1)

            k_block_size = cache.shape[1] * cache.shape[2]
            assert k_block_tables.shape[0] % max_num_seqs == 0
            max_num_blocks_per_req = k_block_tables.shape[0] // max_num_seqs
            k_cache_block_num = (k_input_positions //
                                 self.compress_ratio) // k_block_size + token_to_req_indices * max_num_blocks_per_req
            k_cache_block_offset = (k_input_positions //
                                    self.compress_ratio) % k_block_size
            kv_slot_mapping = (k_block_tables[k_cache_block_num] * k_block_size +
                               k_cache_block_offset)
            kv_slot_mapping = jnp.where(
                jnp.arange(num_tokens) < num_valid_tokens, kv_slot_mapping, -1)

            block_table = state_block_tables.reshape(max_num_seqs, -1)

            new_cache = forward_fn(
                kv_score=kv_score,
                ape=ape,
                norm_weight=norm_weight,
                cos_sin_cache=cos_sin_cache,
                positions=positions,
                slot_mapping=slot_mapping,
                block_table=block_table,
                token_to_req_indices=token_to_req_indices,
                kv_slot_mapping=kv_slot_mapping,
                cache=cache,
                state_block_size=self.state_cache.block_size,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                compress_ratio=self.compress_ratio,
                overlap=self.overlap,
                rms_eps=self.rms_norm_eps,
                quant_block=self._quant_block,
            )

            return new_cache

        new_cache = jax.shard_map(
            _compress,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )(
            jax_view(kv_score),
            jax_view(self.ape),
            jax_view(self.norm.weight),
            jax_view(rotary_emb.cos_sin_cache),
            jax_view(positions),
            state_metadata.request_distribution,
            state_metadata.query_start_loc,
            state_metadata.seq_lens,
            state_metadata.input_positions,
            state_metadata.block_tables,
            k_cache_metadata.input_positions,
            k_cache_metadata.block_tables,
            cache,
            tuple(other_block_tables),
        )
        
        for i in all_caches_indices:
            wrapper_ctx.kv_caches[i] = new_cache
