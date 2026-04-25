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
"""Unit tests for HMA-aware tracking + per-group transfer primitives.

These tests cover the new code added to support hybrid attention+mamba
models like Qwen3.5:
  * `gather_kv_blocks_per_group` / `scatter_kv_blocks_per_group` round-trip
    correctly when kv_caches contain mixed jax.Array (attention) and
    tuple[jax.Array, ...] (Mamba state) entries.
  * `_pick_attention_group_idx` returns the right index for hybrid
    KVCacheConfig.
  * The `_init_hma_tracking` discovery path produces consistent
    layer_to_group_id, group_is_mamba, and per-array group mapping for a
    synthetic hybrid kv_caches list.

These tests do not boot vLLM — they construct synthetic kv_caches lists
directly so the contract (handles tuples, multi-group block IDs) is
verifiable without paying for model load.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from jax._src import compilation_cache as cc
from jax._src import test_util as jtu
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, MambaSpec)

from tpu_inference.offload.tpu_offload_connector import \
    _pick_attention_group_idx
from tpu_inference.offload.utils import (gather_kv_blocks_per_group,
                                         scatter_kv_blocks_per_group)

P = jax.sharding.PartitionSpec


def _alloc_random(shape, dtype, sharding) -> jax.Array:

    @jax.jit
    def _fn(seed):
        return jax.random.uniform(jax.random.key(seed),
                                  shape=shape,
                                  dtype=jnp.float32).astype(dtype)

    return jax.device_put(_fn(0), sharding)


class PickAttentionGroupIdxTest(absltest.TestCase):
    """Verify the connector picks the correct group from a hybrid config."""

    def _make_config(self, group_specs: list[Any]) -> KVCacheConfig:
        groups = [
            KVCacheGroupSpec(layer_names=[f"L{i}"], kv_cache_spec=spec)
            for i, spec in enumerate(group_specs)
        ]
        return KVCacheConfig(num_blocks=128,
                             kv_cache_tensors=[],
                             kv_cache_groups=groups)

    def _attn_spec(self) -> FullAttentionSpec:
        return FullAttentionSpec(block_size=16,
                                 num_kv_heads=1,
                                 head_size=8,
                                 dtype=torch.bfloat16)

    def _mamba_spec(self) -> MambaSpec:
        return MambaSpec(block_size=16,
                         shapes=((4, 8), ),
                         dtypes=(torch.bfloat16, ))

    def test_single_attention_group(self) -> None:
        cfg = self._make_config([self._attn_spec()])
        self.assertEqual(_pick_attention_group_idx(cfg), 0)

    def test_hybrid_attn_first(self) -> None:
        cfg = self._make_config(
            [self._attn_spec(),
             self._mamba_spec(),
             self._mamba_spec()])
        self.assertEqual(_pick_attention_group_idx(cfg), 0)

    def test_hybrid_attn_last_qwen35_layout(self) -> None:
        # Qwen3.5: 3 mamba groups then 1 attn group → attn_group_idx = 3
        cfg = self._make_config([
            self._mamba_spec(),
            self._mamba_spec(),
            self._mamba_spec(),
            self._attn_spec()
        ])
        self.assertEqual(_pick_attention_group_idx(cfg), 3)

    def test_no_kv_cache_config(self) -> None:
        self.assertEqual(_pick_attention_group_idx(None), 0)


class HMAGatherScatterRoundTripTest(jtu.JaxTestCase):
    """End-to-end round-trip on a synthetic hybrid kv_caches list."""

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()
        cc.reset_cache()
        jax.clear_caches()
        gc.collect()

    def test_attention_only_round_trip(self) -> None:
        """Non-hybrid: 4 attention layers, all same shape."""
        mesh = jax.sharding.Mesh(jax.devices(), ("x", ))
        spec = P()
        sharding = jax.sharding.NamedSharding(mesh, spec)
        page = (128, 128)
        dtype = jnp.bfloat16
        num_blocks = 16
        kv_caches = [
            _alloc_random((num_blocks, ) + page, dtype, sharding)
            for _ in range(4)
        ]
        original_np = [np.asarray(jax.device_get(c)) for c in kv_caches]

        layer_to_group_id = [0, 0, 0, 0]
        block_ids_per_group = [[3, 5, 7]]

        # Gather
        flat = gather_kv_blocks_per_group(kv_caches, block_ids_per_group,
                                          layer_to_group_id)
        # 4 attn layers -> 4 entries
        self.assertEqual(len(flat), 4)
        for arr in flat:
            self.assertEqual(arr.shape, (3, ) + page)

        # Scatter into a fresh zeroed cache
        @jax.jit
        def _zeros():
            return jnp.zeros((num_blocks, ) + page, dtype=dtype)

        target = [jax.device_put(_zeros(), sharding) for _ in range(4)]
        treedef = jax.tree_util.tree_structure(kv_caches)
        new_kv = scatter_kv_blocks_per_group(target, flat, [[10, 11, 12]],
                                             layer_to_group_id, treedef)
        for c in new_kv:
            c.block_until_ready()

        for L in range(4):
            t = np.asarray(jax.device_get(new_kv[L]))
            for src, dst in zip([3, 5, 7], [10, 11, 12]):
                np.testing.assert_array_equal(t[dst], original_np[L][src],
                                              f"L={L} src={src} dst={dst}")

    def test_hybrid_attn_plus_mamba_round_trip(self) -> None:
        """Hybrid: 2 attention layers + 2 mamba layers (each a tuple)."""
        mesh = jax.sharding.Mesh(jax.devices(), ("x", ))
        spec = P()
        sharding = jax.sharding.NamedSharding(mesh, spec)
        dtype = jnp.bfloat16
        num_blocks = 8
        attn_page = (128, 128)
        # Mamba state: tuple of (conv_state, ssm_state) with DIFFERENT shapes
        conv_shape = (128, 128)  # mock conv (same shape as attn for mesh fit)
        ssm_shape = (128, 128)  # mock ssm

        # kv_caches layout: [attn0, mamba0_tuple, attn1, mamba1_tuple]
        attn0 = _alloc_random((num_blocks, ) + attn_page, dtype, sharding)
        mamba0_conv = _alloc_random((num_blocks, ) + conv_shape, dtype,
                                    sharding)
        mamba0_ssm = _alloc_random((num_blocks, ) + ssm_shape, dtype, sharding)
        attn1 = _alloc_random((num_blocks, ) + attn_page, dtype, sharding)
        mamba1_conv = _alloc_random((num_blocks, ) + conv_shape, dtype,
                                    sharding)
        mamba1_ssm = _alloc_random((num_blocks, ) + ssm_shape, dtype, sharding)

        kv_caches = [
            attn0, (mamba0_conv, mamba0_ssm), attn1, (mamba1_conv, mamba1_ssm)
        ]
        # Attention is group 0, mamba is group 1.
        layer_to_group_id = [0, 1, 0, 1]
        block_ids_per_group = [
            [3, 5],  # attn group: 2 blocks
            [4],  # mamba group: 1 block
        ]

        flat = gather_kv_blocks_per_group(kv_caches, block_ids_per_group,
                                          layer_to_group_id)
        # 2 attn + 4 mamba arrays (2 mamba layers * 2 states) = 6 arrays
        self.assertEqual(len(flat), 6)
        # Verify shapes per array
        self.assertEqual(flat[0].shape, (2, ) + attn_page)  # attn0
        self.assertEqual(flat[1].shape, (1, ) + conv_shape)  # mamba0 conv
        self.assertEqual(flat[2].shape, (1, ) + ssm_shape)  # mamba0 ssm
        self.assertEqual(flat[3].shape, (2, ) + attn_page)  # attn1
        self.assertEqual(flat[4].shape, (1, ) + conv_shape)  # mamba1 conv
        self.assertEqual(flat[5].shape, (1, ) + ssm_shape)  # mamba1 ssm

        # Scatter back into fresh zeroed caches
        @jax.jit
        def _zero_attn():
            return jnp.zeros((num_blocks, ) + attn_page, dtype=dtype)

        @jax.jit
        def _zero_conv():
            return jnp.zeros((num_blocks, ) + conv_shape, dtype=dtype)

        @jax.jit
        def _zero_ssm():
            return jnp.zeros((num_blocks, ) + ssm_shape, dtype=dtype)

        target = [
            jax.device_put(_zero_attn(), sharding),
            (jax.device_put(_zero_conv(),
                            sharding), jax.device_put(_zero_ssm(), sharding)),
            jax.device_put(_zero_attn(), sharding),
            (jax.device_put(_zero_conv(),
                            sharding), jax.device_put(_zero_ssm(), sharding)),
        ]
        treedef = jax.tree_util.tree_structure(kv_caches)

        new_kv = scatter_kv_blocks_per_group(target, flat, [[6, 7], [2]],
                                             layer_to_group_id, treedef)
        # Materialize all
        for entry in new_kv:
            if isinstance(entry, tuple):
                for s in entry:
                    s.block_until_ready()
            else:
                entry.block_until_ready()

        # Verify attention layers
        attn0_orig = np.asarray(jax.device_get(attn0))
        attn0_new = np.asarray(jax.device_get(new_kv[0]))
        for src, dst in zip([3, 5], [6, 7]):
            np.testing.assert_array_equal(attn0_new[dst], attn0_orig[src],
                                          f"attn0 src={src} dst={dst}")
        attn1_orig = np.asarray(jax.device_get(attn1))
        attn1_new = np.asarray(jax.device_get(new_kv[2]))
        for src, dst in zip([3, 5], [6, 7]):
            np.testing.assert_array_equal(attn1_new[dst], attn1_orig[src],
                                          f"attn1 src={src} dst={dst}")

        # Verify mamba layers (tuple structure preserved)
        for L_idx, (orig_conv,
                    orig_ssm) in enumerate([(mamba0_conv, mamba0_ssm),
                                            (mamba1_conv, mamba1_ssm)]):
            kv_layer_idx = 1 if L_idx == 0 else 3
            self.assertIsInstance(new_kv[kv_layer_idx], tuple)
            new_conv, new_ssm = new_kv[kv_layer_idx]
            new_conv_np = np.asarray(jax.device_get(new_conv))
            new_ssm_np = np.asarray(jax.device_get(new_ssm))
            orig_conv_np = np.asarray(jax.device_get(orig_conv))
            orig_ssm_np = np.asarray(jax.device_get(orig_ssm))
            np.testing.assert_array_equal(new_conv_np[2], orig_conv_np[4],
                                          f"mamba{L_idx} conv mismatch")
            np.testing.assert_array_equal(new_ssm_np[2], orig_ssm_np[4],
                                          f"mamba{L_idx} ssm mismatch")

    def test_empty_block_ids_for_group_skips(self) -> None:
        """A group with empty block_ids should be a no-op for that group."""
        mesh = jax.sharding.Mesh(jax.devices(), ("x", ))
        spec = P()
        sharding = jax.sharding.NamedSharding(mesh, spec)
        dtype = jnp.bfloat16
        attn = _alloc_random((4, 128, 128), dtype, sharding)
        mamba_conv = _alloc_random((4, 128, 128), dtype, sharding)
        mamba_ssm = _alloc_random((4, 128, 128), dtype, sharding)

        kv_caches = [attn, (mamba_conv, mamba_ssm)]
        layer_to_group_id = [0, 1]
        # mamba group has empty block IDs (e.g. didn't get scheduled)
        block_ids_per_group = [[1, 2], []]

        flat = gather_kv_blocks_per_group(kv_caches, block_ids_per_group,
                                          layer_to_group_id)
        self.assertEqual(len(flat), 3)  # 1 attn + 2 mamba states
        # mamba arrays will be (0, *page) since block_ids is empty
        self.assertEqual(flat[1].shape[0], 0)
        self.assertEqual(flat[2].shape[0], 0)

        # Scatter back: mamba should be untouched.
        target = [
            attn,  # reuse same array as target
            (mamba_conv, mamba_ssm),
        ]
        treedef = jax.tree_util.tree_structure(kv_caches)
        new_kv = scatter_kv_blocks_per_group(target, flat, [[2, 3], []],
                                             layer_to_group_id, treedef)
        for entry in new_kv:
            if isinstance(entry, tuple):
                for s in entry:
                    s.block_until_ready()
            else:
                entry.block_until_ready()
        # Mamba layer should be unchanged.
        new_conv = np.asarray(jax.device_get(new_kv[1][0]))
        orig_conv = np.asarray(jax.device_get(mamba_conv))
        np.testing.assert_array_equal(new_conv, orig_conv,
                                      "mamba conv changed despite empty IDs")


class WorkerInitHMATrackingTest(jtu.JaxTestCase):
    """Verify _init_hma_tracking populates per-array fields correctly.

    Uses a stub runner (no real TPU runner / vllm engine) — only exercises
    the discovery path so it's fast.
    """

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()
        cc.reset_cache()
        jax.clear_caches()
        gc.collect()

    def _make_worker_with_stubs(self, kv_caches: list,
                                kv_cache_groups: list[KVCacheGroupSpec],
                                attn_group_idx: int) -> Any:
        """Construct a TPUOffloadConnectorWorker bypassing __init__ side
        effects, then call register_runner with the stub runner."""
        from dataclasses import field
        from typing import Any as A

        from tpu_inference.offload.tpu_offload_connector import \
            TPUOffloadConnectorWorker

        @dataclass
        class _StubKVTransfer:
            kv_connector_extra_config: dict = field(default_factory=dict)

        @dataclass
        class _StubCacheConfig:
            block_size: int = 16

        @dataclass
        class _StubModelConfig:
            max_model_len: int = 1024
            model: str = "stub-model"

        @dataclass
        class _StubVllmConfig:
            kv_transfer_config: _StubKVTransfer
            cache_config: A
            model_config: A

        @dataclass
        class _StubKVCacheConfig:
            kv_cache_groups: list

        @dataclass
        class _StubRunner:
            kv_caches: list
            mesh: A
            devices: list
            layer_name_to_kvcache_index: dict
            kv_cache_config: A

            def get_kv_cache_layout(self):
                return "NHD"

        vllm_config = _StubVllmConfig(
            kv_transfer_config=_StubKVTransfer(),
            cache_config=_StubCacheConfig(),
            model_config=_StubModelConfig(),
        )

        # Build a minimal connector that holds attn_group_idx; the worker
        # references it.
        @dataclass
        class _StubConnector:
            attn_group_idx: int

        connector = _StubConnector(attn_group_idx=attn_group_idx)
        worker = TPUOffloadConnectorWorker(vllm_config, connector)

        # Build layer_name -> index map. Conventionally one layer per group entry.
        layer_name_to_idx = {}
        for cache_idx in range(len(kv_caches)):
            # Find which group this layer belongs to
            cum = 0
            for g_idx, g in enumerate(kv_cache_groups):
                if cache_idx < cum + len(g.layer_names):
                    layer_name_to_idx[g.layer_names[cache_idx -
                                                    cum]] = cache_idx
                    break
                cum += len(g.layer_names)

        runner = _StubRunner(
            kv_caches=kv_caches,
            mesh=jax.sharding.Mesh(jax.devices(), ("x", )),
            devices=jax.devices(),
            layer_name_to_kvcache_index=layer_name_to_idx,
            kv_cache_config=_StubKVCacheConfig(
                kv_cache_groups=kv_cache_groups),
        )
        return worker, runner

    def test_non_hybrid_single_attn_group(self) -> None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x", ))
        sharding = jax.sharding.NamedSharding(mesh, P())
        page = (128, 128)
        dtype = jnp.bfloat16
        # 4 attention layers, all the same shape.
        kv_caches = [
            jax.device_put(jnp.zeros((8, ) + page, dtype=dtype), sharding)
            for _ in range(4)
        ]
        groups = [
            KVCacheGroupSpec(
                layer_names=[f"L{i}" for i in range(4)],
                kv_cache_spec=FullAttentionSpec(block_size=16,
                                                num_kv_heads=1,
                                                head_size=8,
                                                dtype=torch.bfloat16),
            )
        ]
        worker, runner = self._make_worker_with_stubs(kv_caches, groups, 0)
        worker._init_hma_tracking(runner, runner.kv_cache_config)
        self.assertFalse(worker.is_hybrid)
        self.assertEqual(worker.num_groups, 1)
        self.assertEqual(worker.group_is_mamba, [False])
        self.assertEqual(worker.layer_to_group_id, [0, 0, 0, 0])
        self.assertEqual(len(worker._kv_array_to_group_id), 4)
        self.assertEqual(worker._kv_array_to_group_id, [0, 0, 0, 0])
        # max blocks per group: attn = max_model_len / block_size = 1024/16 = 64
        self.assertEqual(worker._max_blocks_per_group, [64])

    def test_hybrid_mamba_then_attn_qwen35_layout(self) -> None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x", ))
        sharding = jax.sharding.NamedSharding(mesh, P())
        page = (128, 128)
        dtype = jnp.bfloat16
        # Layout matching Qwen3.5: 3 mamba layers (each a tuple) + 1 attention.
        attn = jax.device_put(jnp.zeros((8, ) + page, dtype=dtype), sharding)
        mamba_layers = []
        for _ in range(3):
            conv = jax.device_put(jnp.zeros((8, ) + page, dtype=dtype),
                                  sharding)
            ssm = jax.device_put(jnp.zeros((8, ) + page, dtype=dtype),
                                 sharding)
            mamba_layers.append((conv, ssm))
        kv_caches = mamba_layers + [attn]  # m, m, m, attn
        groups = [
            # group 0..2: mamba (one layer each)
            KVCacheGroupSpec(
                layer_names=[f"M{g}"],
                kv_cache_spec=MambaSpec(
                    block_size=16,
                    shapes=((128, 128), (128, 128)),
                    dtypes=(torch.bfloat16, torch.bfloat16)),
            ) for g in range(3)
        ] + [
            KVCacheGroupSpec(
                layer_names=["A0"],
                kv_cache_spec=FullAttentionSpec(block_size=16,
                                                num_kv_heads=1,
                                                head_size=8,
                                                dtype=torch.bfloat16),
            )
        ]
        worker, runner = self._make_worker_with_stubs(kv_caches,
                                                      groups,
                                                      attn_group_idx=3)
        worker._init_hma_tracking(runner, runner.kv_cache_config)
        self.assertTrue(worker.is_hybrid)
        self.assertEqual(worker.num_groups, 4)
        self.assertEqual(worker.group_is_mamba, [True, True, True, False])
        # 4 layers in runner.kv_caches, layer_to_group_id should map them
        # to groups [0, 1, 2, 3] respectively.
        self.assertEqual(worker.layer_to_group_id, [0, 1, 2, 3])
        # tree_flatten flattens each mamba tuple into 2 entries; attn is 1
        # entry. So 3*2 + 1 = 7 flat arrays.
        self.assertEqual(len(worker._kv_array_to_group_id), 7)
        # Mamba layers (groups 0, 1, 2) each contribute 2 flat arrays;
        # then attn layer (group 3) contributes 1. Verify mapping.
        self.assertEqual(worker._kv_array_to_group_id, [0, 0, 1, 1, 2, 2, 3])
        # Mamba groups: 1 block per request; attn: max_model_len/block_size = 64
        self.assertEqual(worker._max_blocks_per_group, [1, 1, 1, 64])
        # Attention sharding spec gathered for group 3.
        self.assertIsNotNone(worker._group_attn_sharding_spec[3])
        # Mamba groups have None sharding spec (we don't track it).
        for g in [0, 1, 2]:
            self.assertIsNone(worker._group_attn_sharding_spec[g])


class WorkerMambaGatherScatterRoundTripTest(WorkerInitHMATrackingTest):
    """Verify the worker's per-request mamba gather + scatter is identity.

    Saves the mamba state for a request (singleton block per mamba group),
    zeroes out the runner's mamba slots, then scatters the saved state into a
    different destination block. Checks that the destination matches the
    original, the source is now zero, and other blocks are untouched.
    """

    def test_hybrid_mamba_round_trip(self) -> None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x", ))
        sharding = jax.sharding.NamedSharding(mesh, P())
        page = (128, 128)
        dtype = jnp.bfloat16
        # 1 mamba layer (tuple of 2 states) + 1 attn layer.
        # Use 8 mamba blocks to test selecting from a non-zero index.
        conv = _alloc_random((8, ) + page, dtype, sharding)
        ssm = _alloc_random((8, ) + page, dtype, sharding)
        attn = _alloc_random((8, ) + page, dtype, sharding)
        kv_caches = [(conv, ssm), attn]

        groups = [
            KVCacheGroupSpec(layer_names=["M0"],
                             kv_cache_spec=MambaSpec(block_size=16,
                                                     shapes=((128, 128),
                                                             (128, 128)),
                                                     dtypes=(torch.bfloat16,
                                                             torch.bfloat16))),
            KVCacheGroupSpec(layer_names=["A0"],
                             kv_cache_spec=FullAttentionSpec(
                                 block_size=16,
                                 num_kv_heads=1,
                                 head_size=8,
                                 dtype=torch.bfloat16)),
        ]
        worker, runner = self._make_worker_with_stubs(kv_caches,
                                                      groups,
                                                      attn_group_idx=1)
        worker._init_hma_tracking(runner, runner.kv_cache_config)
        worker.runner = runner
        # Sanity: hybrid + 1 mamba group + 1 attn group.
        self.assertTrue(worker.is_hybrid)
        self.assertEqual(worker.group_is_mamba, [True, False])

        # Snapshot block 5 of the mamba layer's (conv, ssm) before gather.
        src_block_id = 5
        dst_block_id = 2
        original_conv_b5 = np.asarray(jax.device_get(conv))[src_block_id]
        original_ssm_b5 = np.asarray(jax.device_get(ssm))[src_block_id]
        original_conv_b1 = np.asarray(jax.device_get(conv))[1]  # untouched

        # Gather: src is mamba block 5; attn group is empty.
        local_per_group = [[src_block_id], []]
        gathered = worker._gather_mamba_state_for_request(local_per_group)
        # 1 mamba layer with tuple of 2 → 2 flat arrays.
        self.assertEqual(len(gathered), 2)
        for arr in gathered:
            self.assertEqual(arr.shape, (1, ) + page)

        # Zero out the source block via scatter to verify scatter overwrites.
        zero_slice = jnp.zeros((1, ) + page, dtype=dtype)
        worker._scatter_mamba_state_for_request([zero_slice, zero_slice],
                                                [[src_block_id], []])
        post_zero_conv = np.asarray(
            jax.device_get(worker.runner.kv_caches[0][0]))
        np.testing.assert_array_equal(
            post_zero_conv[src_block_id],
            np.zeros(page, dtype=np.float32).astype(jax.dtypes.bfloat16))

        # Scatter the originally-gathered state into a DIFFERENT dst block.
        worker._scatter_mamba_state_for_request(gathered, [[dst_block_id], []])
        post_conv = np.asarray(jax.device_get(worker.runner.kv_caches[0][0]))
        post_ssm = np.asarray(jax.device_get(worker.runner.kv_caches[0][1]))

        # dst block now matches original src content.
        np.testing.assert_array_equal(post_conv[dst_block_id],
                                      original_conv_b5)
        np.testing.assert_array_equal(post_ssm[dst_block_id], original_ssm_b5)
        # untouched block 1 is still original.
        np.testing.assert_array_equal(post_conv[1], original_conv_b1)


if __name__ == "__main__":
    absltest.main()
