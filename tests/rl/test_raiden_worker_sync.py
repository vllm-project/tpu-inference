# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Unit tests for tpu_inference.rl.raiden_worker_sync."""

import unittest
import unittest.mock
from types import SimpleNamespace

from tpu_inference.rl import raiden_worker_sync as rws


class MaxTextForCausalLM:
    """Named to match the real class by `__name__`; see `is_maxtext_model`."""


class TestIsMaxtextModel(unittest.TestCase):

    def test_matches_by_class_name(self):
        self.assertTrue(rws.is_maxtext_model(MaxTextForCausalLM()))

    def test_none_model(self):
        self.assertFalse(rws.is_maxtext_model(None))

    def test_other_model(self):
        self.assertFalse(rws.is_maxtext_model(object()))


class TestExtractWeightState(unittest.TestCase):

    def test_maxtext_unwraps_model_key(self):
        state = {"model": {"params": 1}}
        result = rws.extract_weight_state(state, MaxTextForCausalLM())
        self.assertEqual(result, {"base": {"params": 1}})

    def test_maxtext_missing_model_key_falls_back_to_state(self):
        state = {"other": {"params": 1}}
        result = rws.extract_weight_state(state, MaxTextForCausalLM())
        self.assertIs(result, state)

    def test_non_maxtext_returns_state_as_is(self):
        state = {"params": 1}
        result = rws.extract_weight_state(state, object())
        self.assertIs(result, state)

    def test_no_state_no_model_returns_none(self):
        self.assertIsNone(rws.extract_weight_state(None, None))

    def test_maxtext_model_without_state_and_no_inner_model_returns_none(self):
        model = MaxTextForCausalLM()  # no `.model` attribute
        self.assertIsNone(rws.extract_weight_state(None, model))


class TestFlattenWeights(unittest.TestCase):

    def test_flattens_leaves_with_shape_and_dtype(self):
        leaf = SimpleNamespace(shape=(2, 2), dtype="float32")
        names, arrays = rws.flatten_weights({"w": leaf})
        self.assertEqual(len(names), 1)
        self.assertIs(arrays[0], leaf)

    def test_skips_leaves_without_shape_or_dtype(self):
        names, arrays = rws.flatten_weights({"w": 3})
        self.assertEqual(names, [])
        self.assertEqual(arrays, [])


class TestAxisName(unittest.TestCase):

    def test_none(self):
        self.assertEqual(rws._axis_name(None), "")

    def test_str(self):
        self.assertEqual(rws._axis_name("fsdp"), "fsdp")

    def test_tuple(self):
        self.assertEqual(rws._axis_name(("fsdp", "tp")), "fsdp,tp")


class TestIsWeight(unittest.TestCase):
    """`_filter_bindable` drops KV-cache leaves; the trainer has no counterpart
    for them and the controller pairs by name."""

    def test_keeps_ordinary_weights(self):
        self.assertTrue(
            rws._is_weight("['base']['decoder']['layers_0']['mlp']['kernel']"))

    def test_drops_cache_leaves(self):
        self.assertFalse(
            rws._is_weight(
                "['base']['decoder']['layers_0']['attention']['cache']"
                "['cached_prefill_key']"))

    def test_filter_bindable_drops_cache_without_touching_weights(self):
        leaf = SimpleNamespace(shape=(2, ), dtype="float32")
        names = ["['w']", "['attention']['cache']['cached_prefill_key']"]
        kept_names, kept_arrays = rws._filter_bindable(names, [leaf, leaf])
        # The cache leaf is dropped before `_bindable` is consulted, so it goes
        # even though it is an ordinary float array.
        self.assertEqual(kept_names, [])
        self.assertEqual(kept_arrays, [])


class TestWaitUntilSettled(unittest.TestCase):

    def _sync_with_digests(self, digests):
        sync = rws.RaidenWorkerSync("rollout")
        sync.arrays = [object()]
        seq = iter(digests)
        last = [digests[-1]]

        def fake_l1_norm(_arrays):
            try:
                last[0] = next(seq)
            except StopIteration:
                pass
            return last[0]

        return sync, fake_l1_norm

    def test_returns_once_the_digest_stops_changing(self):
        sync, fake = self._sync_with_digests([1.0, 2.0, 3.0, 3.0, 3.0, 3.0])
        with unittest.mock.patch.object(rws, "_l1_norm", fake), \
             unittest.mock.patch.object(rws.time, "sleep", lambda _s: None):
            sync._wait_until_settled(timeout_s=5.0, interval_s=0.0)

    def test_returns_early_when_the_transfer_has_not_started(self):
        """Known limitation: a digest that has not moved yet reads as settled.

        `stable` counts *unchanged* reads and there is no "saw it change at
        least once" precondition, so if h2d() returns before Raiden has DMA'd
        any bytes this declares success on the pre-sync weights.
        """
        sync, fake = self._sync_with_digests([7.0] * 6)
        with unittest.mock.patch.object(rws, "_l1_norm", fake), \
             unittest.mock.patch.object(rws.time, "sleep", lambda _s: None):
            sync._wait_until_settled(timeout_s=5.0, interval_s=0.0)


class TestRaidenWorkerSyncMetadataDict(unittest.TestCase):

    def test_raises_without_a_sharded_array(self):
        sync = rws.RaidenWorkerSync("rollout")
        sync.names = ["w"]
        sync.arrays = [
            SimpleNamespace(shape=(2, ),
                            dtype=SimpleNamespace(itemsize=4),
                            sharding=None,
                            ndim=1)
        ]
        with self.assertRaises(RuntimeError):
            sync.metadata_dict()

    def test_bound_is_false_until_bind(self):
        sync = rws.RaidenWorkerSync("rollout")
        self.assertFalse(sync.bound)
        sync.names = ["w"]
        self.assertTrue(sync.bound)

    def test_metadata_dict_publishes_one_global_shard_index_per_shard(self):
        """Index j describes shards[j], i.e. addressable_shards[j]'s slice."""
        sync = rws.RaidenWorkerSync("rollout", bind_ip="1.2.3.4")
        sync.names = ["w"]
        dev0, dev1 = object(), object()
        # (8, 8) sharded P("x", "y") on a 2x2 mesh; this host owns the x=1 row:
        # (1,0) -> rows 4:8 / cols 0:4 and (1,1) -> rows 4:8 / cols 4:8.
        indices_map = {
            object(): (slice(0, 4), slice(0, 4)),
            object(): (slice(0, 4), slice(4, 8)),
            dev0: (slice(4, 8), slice(0, 4)),
            dev1: (slice(4, 8), slice(4, 8)),
        }
        mock_sharding = SimpleNamespace(
            mesh=SimpleNamespace(axis_names=("x", "y"), shape={
                "x": 2,
                "y": 2
            }),
            spec=("x", "y"),
            shard_shape=lambda shape: (4, 4),
            devices_indices_map=lambda shape: indices_map,
        )
        sync.arrays = [
            SimpleNamespace(
                shape=(8, 8),
                dtype=SimpleNamespace(itemsize=4),
                sharding=mock_sharding,
                ndim=2,
                addressable_shards=[
                    SimpleNamespace(device=dev0),
                    SimpleNamespace(device=dev1),
                ],
            )
        ]
        sync._sync = SimpleNamespace(num_shards=2,
                                     local_port=12345,
                                     listener_port=23456)
        meta = sync.metadata_dict()
        self.assertEqual(meta["shards"], ["1.2.3.4:12345"] * 2)
        self.assertEqual(meta["mesh_shape"], [2, 2])
        self.assertEqual(meta["variables"][0]["mesh_shape"], [2, 2])
        # Row-major over the (2, 2) shard grid: (1,0) -> 2, (1,1) -> 3.
        self.assertEqual(meta["variables"][0]["global_shard_indices"], [2, 3])
        self.assertNotIn("host_subgrid", meta)

    def test_metadata_dict_indices_follow_addressable_shard_order(self):
        """The index list follows addressable_shards, not the mesh order."""
        sync = rws.RaidenWorkerSync("rollout", bind_ip="1.2.3.4")
        sync.names = ["w"]
        dev0, dev1 = object(), object()
        indices_map = {
            dev0: (slice(4, 8), slice(0, 4)),
            dev1: (slice(4, 8), slice(4, 8)),
        }
        mock_sharding = SimpleNamespace(
            mesh=SimpleNamespace(axis_names=("x", "y"), shape={
                "x": 2,
                "y": 2
            }),
            spec=("x", "y"),
            shard_shape=lambda shape: (4, 4),
            devices_indices_map=lambda shape: indices_map,
        )
        sync.arrays = [
            SimpleNamespace(
                shape=(8, 8),
                dtype=SimpleNamespace(itemsize=4),
                sharding=mock_sharding,
                ndim=2,
                addressable_shards=[
                    SimpleNamespace(device=dev1),
                    SimpleNamespace(device=dev0),
                ],
            )
        ]
        sync._sync = SimpleNamespace(num_shards=2,
                                     local_port=12345,
                                     listener_port=23456)
        meta = sync.metadata_dict()
        self.assertEqual(meta["variables"][0]["global_shard_indices"], [3, 2])

    def test_metadata_dict_replicated_variable_repeats_index_zero(self):
        """Every local shard of a fully replicated variable holds slice 0."""
        sync = rws.RaidenWorkerSync("rollout", bind_ip="1.2.3.4")
        sync.names = ["w"]
        dev0, dev1 = object(), object()
        indices_map = {
            dev0: (slice(None), slice(None)),
            dev1: (slice(None), slice(None)),
        }
        mock_sharding = SimpleNamespace(
            mesh=SimpleNamespace(axis_names=("x", "y"), shape={
                "x": 1,
                "y": 2
            }),
            spec=(),
            shard_shape=lambda shape: shape,
            devices_indices_map=lambda shape: indices_map,
        )
        sync.arrays = [
            SimpleNamespace(
                shape=(8, 8),
                dtype=SimpleNamespace(itemsize=4),
                sharding=mock_sharding,
                ndim=2,
                addressable_shards=[
                    SimpleNamespace(device=dev0),
                    SimpleNamespace(device=dev1),
                ],
            )
        ]
        sync._sync = SimpleNamespace(num_shards=2,
                                     local_port=12345,
                                     listener_port=23456)
        meta = sync.metadata_dict()
        self.assertEqual(meta["variables"][0]["mesh_shape"], [1, 1])
        self.assertEqual(meta["variables"][0]["global_shard_indices"], [0, 0])

    def test_metadata_dict_rejects_a_num_shards_mismatch(self):
        sync = rws.RaidenWorkerSync("rollout", bind_ip="1.2.3.4")
        sync.names = ["w"]
        dev0 = object()
        mock_sharding = SimpleNamespace(
            mesh=SimpleNamespace(axis_names=("x", ), shape={"x": 1}),
            spec=(),
            shard_shape=lambda shape: shape,
            devices_indices_map=lambda shape: {dev0: (slice(None), )},
        )
        sync.arrays = [
            SimpleNamespace(
                shape=(8, ),
                dtype=SimpleNamespace(itemsize=4),
                sharding=mock_sharding,
                ndim=1,
                addressable_shards=[SimpleNamespace(device=dev0)],
            )
        ]
        sync._sync = SimpleNamespace(num_shards=2,
                                     local_port=12345,
                                     listener_port=23456)
        with self.assertRaisesRegex(RuntimeError,
                                    "1 addressable shard.*num_shards=2"):
            sync.metadata_dict()

    def test_metadata_dict_rejects_an_index_count_mismatch(self):
        sync = rws.RaidenWorkerSync("rollout", bind_ip="1.2.3.4")
        sync.names = ["w"]
        mock_sharding = SimpleNamespace(
            mesh=SimpleNamespace(axis_names=("x", ), shape={"x": 1}),
            spec=(),
            shard_shape=lambda shape: shape,
            devices_indices_map=lambda shape: {},
        )
        sync.arrays = [
            SimpleNamespace(
                shape=(8, ),
                dtype=SimpleNamespace(itemsize=4),
                sharding=mock_sharding,
                ndim=1,
                addressable_shards=[SimpleNamespace(device=object())],
            )
        ]
        sync._sync = SimpleNamespace(num_shards=1,
                                     local_port=12345,
                                     listener_port=23456)
        with unittest.mock.patch.object(sync,
                                        "_shard_devices",
                                        return_value=[]):
            with self.assertRaisesRegex(
                    RuntimeError,
                    "'w' publishes 0 global_shard_indices for 1 shards"):
                sync.metadata_dict()


class TestGlobalShardIndex(unittest.TestCase):

    def test_1d(self):
        index = rws._global_shard_index
        self.assertEqual(index("w", (slice(0, 2), ), (8, ), (4, )), 0)
        self.assertEqual(index("w", (slice(6, 8), ), (8, ), (4, )), 3)

    def test_2d_is_row_major(self):
        self.assertEqual(
            rws._global_shard_index("w", (slice(4, 8), slice(2, 4)), (8, 8),
                                    (2, 4)), 1 * 4 + 1)

    def test_replicated_dimension_is_coordinate_zero(self):
        index = rws._global_shard_index
        self.assertEqual(
            index("w", (slice(None), slice(4, 8)), (8, 8), (1, 2)), 1)
        self.assertEqual(
            index("w", (slice(4, 8), slice(None)), (8, 8), (2, 1)), 1)

    def test_rejects_misaligned_start(self):
        with self.assertRaisesRegex(ValueError, "'w'.*multiple of tile 2"):
            rws._global_shard_index("w", (slice(1, 3), ), (8, ), (4, ))

    def test_rejects_start_beyond_the_grid(self):
        with self.assertRaisesRegex(ValueError, "'w'.*below 2 tiles"):
            rws._global_shard_index("w", (slice(8, 12), ), (8, ), (2, ))

    def test_rejects_rank_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same rank"):
            rws._global_shard_index("w", (slice(0, 4), ), (8, 8), (2, 1))


class TestTensorMetadataDict(unittest.TestCase):

    def test_without_real_sharding_leaves_indices_empty(self):
        arr = SimpleNamespace(shape=(8, ),
                              dtype=SimpleNamespace(itemsize=4),
                              sharding=None,
                              ndim=1)
        md = rws._tensor_metadata_dict("w", arr, 0, [object()])
        self.assertEqual(md["mesh_shape"], [1])
        self.assertEqual(md["global_shard_indices"], [])


class TestRaidenWorkerSyncH2D(unittest.TestCase):

    def test_h2d_calls_wait_for_transfer_completion_when_available(self):
        sync = rws.RaidenWorkerSync("rollout")
        mock_ws = unittest.mock.MagicMock()
        sync._sync = mock_ws
        sync.arrays = [SimpleNamespace()]

        with unittest.mock.patch("jax.block_until_ready"
                                 ) as mock_block, unittest.mock.patch.object(
                                     sync,
                                     "_wait_until_settled") as mock_settle:
            sync.h2d(uuid=42)
            mock_ws.wait_for_transfer_completion.assert_called_once_with(42)
            mock_ws.h2d.assert_not_called()
            mock_block.assert_called_once_with(sync.arrays)
            mock_settle.assert_not_called()

    def test_h2d_fallback_when_wait_for_transfer_completion_missing(self):
        sync = rws.RaidenWorkerSync("rollout")
        mock_ws = unittest.mock.MagicMock(
            spec=["h2d"])  # lacks wait_for_transfer_completion
        sync._sync = mock_ws
        sync.arrays = [SimpleNamespace()]

        with unittest.mock.patch("jax.block_until_ready"
                                 ) as mock_block, unittest.mock.patch.object(
                                     sync,
                                     "_wait_until_settled") as mock_settle:
            sync.h2d()
            mock_ws.h2d.assert_called_once()
            mock_block.assert_called_once_with(sync.arrays)
            mock_settle.assert_called_once()

    def test_h2d_fallback_does_not_settle_when_env_disabled(self):
        sync = rws.RaidenWorkerSync("rollout")
        mock_ws = unittest.mock.MagicMock(
            spec=["h2d"])  # lacks wait_for_transfer_completion
        sync._sync = mock_ws
        sync.arrays = [SimpleNamespace()]

        with unittest.mock.patch("jax.block_until_ready"
                                 ) as mock_block, unittest.mock.patch.object(
                                     rws.envs, "RAIDEN_H2D_SETTLE",
                                     False), unittest.mock.patch.object(
                                         sync,
                                         "_wait_until_settled") as mock_settle:
            sync.h2d()
            mock_ws.h2d.assert_called_once()
            mock_block.assert_called_once_with(sync.arrays)
            mock_settle.assert_not_called()


if __name__ == "__main__":
    unittest.main()
