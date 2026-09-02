# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dedicated block pools for mamba KV cache groups, driven through vLLM's
`KVCacheManager` exactly as the scheduler drives it (CPU only)."""
from math import lcm

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (get_request_block_hasher,
                                         init_none_hash)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, MambaSpec)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.request import Request

from tpu_inference.core.mamba_block_pool import (TPUMambaManager,
                                                 TPUMambaSpec,
                                                 register_tpu_mamba_spec)

BLOCK_SIZE = 16


@pytest.fixture(autouse=True)
def _register():
    init_none_hash(sha256)
    register_tpu_mamba_spec()


def make_request(request_id: str, prompt_token_ids: list[int]) -> Request:
    sampling_params = SamplingParams(max_tokens=17)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        lora_request=None,
        cache_salt=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def make_config(num_blocks: int,
                mamba_num_blocks: int | None,
                num_mamba_groups: int = 2) -> KVCacheConfig:
    """1 full-attention group on the shared pool + mamba "align" groups."""
    mamba_spec = TPUMambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((1, ), ),
        dtypes=(torch.float32, ),
        mamba_cache_mode="align",
        num_blocks=mamba_num_blocks,
    )
    groups = [
        KVCacheGroupSpec(
            ["attn"],
            FullAttentionSpec(block_size=BLOCK_SIZE,
                              num_kv_heads=1,
                              head_size=1,
                              dtype=torch.float32))
    ] + [
        KVCacheGroupSpec([f"mamba{i}"], mamba_spec)
        for i in range(num_mamba_groups)
    ]
    return KVCacheConfig(num_blocks=num_blocks,
                         kv_cache_tensors=[],
                         kv_cache_groups=groups)


def make_manager(kv_cache_config: KVCacheConfig) -> KVCacheManager:
    return KVCacheManager(
        kv_cache_config,
        max_model_len=8192,
        scheduler_block_size=lcm(*(g.kv_cache_spec.block_size
                                   for g in kv_cache_config.kv_cache_groups)),
        hash_block_size=BLOCK_SIZE,
        enable_caching=True,
    )


def private_pools(manager: KVCacheManager):
    return [m.block_pool for m in manager.coordinator.single_type_managers[1:]]


def test_registry_maps_spec_to_tpu_manager():
    spec = make_config(10, 4).kv_cache_groups[1].kv_cache_spec
    assert KVCacheSpecRegistry.get_manager_class(spec) is TPUMambaManager
    assert isinstance(spec, MambaSpec)
    # Idempotent (vLLM re-runs registration per process).
    register_tpu_mamba_spec()


def test_without_num_blocks_shares_the_pool():
    manager = make_manager(make_config(30, None))
    for mgr in manager.coordinator.single_type_managers[1:]:
        assert isinstance(mgr, TPUMambaManager)
        assert not mgr.has_private_pool
        assert mgr.block_pool is manager.block_pool


def test_dedicated_pools_allocate_hit_and_free():
    """Attention draws from the shared pool, each mamba group from its own
    pool, hits reconcile across the batched group lookup, and blocks popped
    for deferred free return to their own pool."""
    num_blocks, mamba_num_blocks = 30, 8
    manager = make_manager(make_config(num_blocks, mamba_num_blocks))
    pools = private_pools(manager)
    assert [p.num_gpu_blocks for p in pools] == [mamba_num_blocks] * 2
    assert pools[0] is not pools[1]
    assert all(p is not manager.block_pool for p in pools)
    all_free = [num_blocks - 1] + [mamba_num_blocks - 1] * 2

    def free_counts():
        return [manager.block_pool.get_num_free_blocks()
                ] + [p.get_num_free_blocks() for p in pools]

    assert free_counts() == all_free

    # Exactly 3 full blocks: the align-mode checkpoint lands on the last
    # full block and is cacheable.
    tokens = [i for i in range(3) for _ in range(BLOCK_SIZE)]
    req0 = make_request("0", tokens)
    computed, num_computed, _ = manager.get_computed_blocks(req0)
    assert num_computed == 0
    blocks = manager.allocate_slots(req0, 3 * BLOCK_SIZE, 0, computed)
    assert blocks is not None
    block_ids = blocks.get_block_ids()
    assert manager.block_pool.get_num_free_blocks() == num_blocks - 1 - 3
    for gid in (1, 2):
        assert block_ids[gid]
        assert all(0 <= b < mamba_num_blocks for b in block_ids[gid])
        assert pools[gid - 1].get_num_free_blocks() < mamba_num_blocks - 1
    manager.free(req0)
    assert free_counts() == all_free
    manager.new_step_starts()

    # Same prefix: a hit in every group, each from its own pool (the two
    # mamba groups share a spec, so the coordinator batches their lookup).
    req1 = make_request("1", tokens + [9] * 5)
    computed, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 3 * BLOCK_SIZE
    assert all(len(g) > 0 for g in computed.blocks)
    assert manager.allocate_slots(req1, 5, num_computed, computed) is not None

    # Deferred-free path: the scheduler frees popped blocks into the shared
    # pool, so mamba blocks must not be handed back to it.
    popped = manager.pop_blocks_for_free(req1)
    assert popped and all(
        manager.block_pool.blocks[b.block_id] is b for b in popped)
    manager.block_pool.free_blocks(reversed(popped))
    assert manager.block_pool.get_num_free_blocks() == num_blocks - 1
    # The mamba blocks are released after the in-flight window.
    assert all(p.get_num_free_blocks() < mamba_num_blocks - 1 for p in pools)
    delay = manager.coordinator.single_type_managers[1]._free_delay_steps
    for _ in range(delay):
        manager.new_step_starts()
    assert free_counts() == all_free


def test_dedicated_pool_gates_admission():
    """An exhausted mamba pool refuses admission even though the shared pool
    has plenty of room; freeing a request unblocks it."""
    num_blocks, mamba_num_blocks = 100, 3  # null + 2 usable per group
    manager = make_manager(make_config(num_blocks, mamba_num_blocks))
    num_tokens = 3 * BLOCK_SIZE + 7

    def new_request(idx: int) -> Request:
        return make_request(str(idx),
                            [1000 * idx + j for j in range(num_tokens)])

    requests = []
    for idx in range(2):
        req = new_request(idx)
        computed, num_computed, _ = manager.get_computed_blocks(req)
        assert manager.allocate_slots(req, num_tokens, num_computed,
                                      computed) is not None
        requests.append(req)
    assert [p.get_num_free_blocks() for p in private_pools(manager)] == [0, 0]
    assert manager.block_pool.get_num_free_blocks() > num_blocks // 2

    req2 = new_request(2)
    computed, num_computed, _ = manager.get_computed_blocks(req2)
    assert manager.allocate_slots(req2, num_tokens, num_computed,
                                  computed) is None

    manager.free(requests[0])
    computed, num_computed, _ = manager.get_computed_blocks(req2)
    assert manager.allocate_slots(req2, num_tokens, num_computed,
                                  computed) is not None


def test_shared_pool_reset_resets_dedicated_pools():
    manager = make_manager(make_config(30, 8))
    tokens = [i for i in range(3) for _ in range(BLOCK_SIZE)]
    req0 = make_request("0", tokens)
    computed, num_computed, _ = manager.get_computed_blocks(req0)
    manager.allocate_slots(req0, 3 * BLOCK_SIZE, 0, computed)
    manager.free(req0)
    assert all(len(p.cached_block_hash_to_block) > 0
               for p in private_pools(manager))

    assert manager.reset_prefix_cache()
    manager.new_step_starts()
    assert all(len(p.cached_block_hash_to_block) == 0
               for p in private_pools(manager))
    req1 = make_request("1", tokens + [9] * 5)
    _, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed == 0
