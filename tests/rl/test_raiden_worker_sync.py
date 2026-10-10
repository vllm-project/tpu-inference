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

    def test_metadata_dict_extracts_host_subgrid_from_local_mesh(self):
        sync = rws.RaidenWorkerSync("rollout")
        sync.names = ["w"]
        mock_devices = unittest.mock.MagicMock()
        mock_devices.shape = (1, 4)
        mock_local_mesh = SimpleNamespace(devices=mock_devices)
        mock_mesh = SimpleNamespace(
            axis_names=("x", "y"),
            shape={
                "x": 1,
                "y": 4
            },
            local_mesh=mock_local_mesh,
        )
        mock_sharding = SimpleNamespace(
            mesh=mock_mesh,
            spec=(),
            shard_shape=lambda shape: shape,
        )
        sync.arrays = [
            SimpleNamespace(
                shape=(2, 4),
                dtype=SimpleNamespace(itemsize=4),
                sharding=mock_sharding,
                ndim=2,
            )
        ]
        meta = sync.metadata_dict()
        self.assertEqual(meta["host_subgrid"], [1, 4])

    def test_metadata_dict_dual_nic_partitions_shards_across_nics(self):
        sync = rws.RaidenWorkerSync("rollout", bind_ip="10.11.0.5")
        sync.names = ["w"]
        mock_mesh = SimpleNamespace(
            axis_names=("x", "y"),
            shape={
                "x": 2,
                "y": 8
            },
            local_mesh=None,
        )
        mock_sharding = SimpleNamespace(
            mesh=mock_mesh,
            spec=(),
            shard_shape=lambda shape: shape,
        )
        sync.arrays = [
            SimpleNamespace(
                shape=(2, 4),
                dtype=SimpleNamespace(itemsize=4),
                sharding=mock_sharding,
                ndim=2,
            )
        ]
        sync._sync = SimpleNamespace(
            num_shards=8,
            local_port=12345,
            listener_port=23456,
        )
        with unittest.mock.patch.object(
                rws,
                "_resolve_data_nic_ips",
                return_value=["10.11.0.5", "10.12.0.5"],
        ):
            meta = sync.metadata_dict()
        self.assertEqual(
            meta["shards"],
            ["10.11.0.5:12345"] * 4 + ["10.12.0.5:12345"] * 4,
        )
        self.assertEqual(meta["control_plane_rpc_address"], "10.11.0.5:23456")

    def test_metadata_dict_dual_nic_stripes_single_manager_get_local_endpoints(
            self):
        sync = rws.RaidenWorkerSync("rollout", bind_ip="10.11.0.5")
        sync.names = ["w"]
        mock_mesh = SimpleNamespace(
            axis_names=("x", "y"),
            shape={
                "x": 2,
                "y": 8
            },
            local_mesh=None,
        )
        mock_sharding = SimpleNamespace(
            mesh=mock_mesh,
            spec=(),
            shard_shape=lambda shape: shape,
        )
        sync.arrays = [
            SimpleNamespace(
                shape=(2, 4),
                dtype=SimpleNamespace(itemsize=4),
                sharding=mock_sharding,
                ndim=2,
            )
        ]
        sync._sync = SimpleNamespace(
            num_shards=8,
            local_port=12345,
            listener_port=23456,
            get_local_endpoints=lambda: [{
                "endpoint": "10.11.0.5:12345",
                "shards": list(range(8)),
            }],
        )
        with unittest.mock.patch.object(
                rws,
                "_resolve_data_nic_ips",
                return_value=["10.11.0.5", "10.12.0.5"],
        ):
            meta = sync.metadata_dict()
        self.assertEqual(
            meta["shards"],
            ["10.11.0.5:12345"] * 4 + ["10.12.0.5:12345"] * 4,
        )
        self.assertEqual(meta["control_plane_rpc_address"], "10.11.0.5:23456")


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

    def test_deferred_h2d_waits_for_transfer_completion_and_calls_h2d(self):
        sync = rws.RaidenWorkerSync("rollout", auto_h2d=False)
        mock_ws = unittest.mock.MagicMock()
        sync._sync = mock_ws
        sync.arrays = [SimpleNamespace()]

        with unittest.mock.patch("jax.block_until_ready"
                                 ) as mock_block, unittest.mock.patch.object(
                                     rws.envs, "RAIDEN_H2D_SETTLE",
                                     False), unittest.mock.patch.object(
                                         sync,
                                         "_wait_until_settled") as mock_settle:
            sync.h2d(uuid=42)
            mock_ws.wait_for_transfer_completion.assert_called_once_with(42)
            mock_ws.h2d.assert_called_once()
            mock_block.assert_called_once_with(sync.arrays)
            mock_settle.assert_not_called()

    def test_deferred_h2d_rejects_missing_or_non_positive_uuid(self):
        sync = rws.RaidenWorkerSync("rollout", auto_h2d=False)
        sync._sync = unittest.mock.MagicMock()
        sync.arrays = [SimpleNamespace()]

        for bad_uuid in (None, 0, -5):
            with self.assertRaises(ValueError):
                sync.h2d(uuid=bad_uuid)

    def test_parallel_h2h_env_defaults_auto_h2d_to_false(self):
        with unittest.mock.patch.dict("os.environ",
                                      {"WEIGHT_SYNC_PARALLEL_H2H": "1"}):
            sync = rws.RaidenWorkerSync("rollout")
            self.assertFalse(sync._auto_h2d)
            self.assertFalse(sync._effective_auto_h2d)


class TestRaidenWorkerSyncBindIp(unittest.TestCase):
    """bind() hands the native layer an explicit source-bind policy.

    With several data NICs and ENABLE_MULTI_NUMA off, the native default
    binds every outbound socket to NIC 0's IP; packets then leave NIC 1 with
    NIC 0's source address and VPC anti-spoofing drops them. "0.0.0.0" tells
    the native layer to skip the bind and let the kernel route per peer.
    """

    def _bind_and_capture_bind_ip(self, env: dict, nic_ips: list) -> object:
        mock_ws_cls = unittest.mock.MagicMock()
        mock_ws_lib = SimpleNamespace(WeightSynchronizer=mock_ws_cls)
        with unittest.mock.patch.dict("os.environ", env, clear=False), \
             unittest.mock.patch.object(rws, "_ws_lib", mock_ws_lib), \
             unittest.mock.patch.object(rws, "_resolve_data_nic_ips",
                                        return_value=nic_ips), \
             unittest.mock.patch.object(rws, "flatten_weights",
                                        return_value=(["w"], [object()])), \
             unittest.mock.patch.object(rws, "_filter_bindable",
                                        side_effect=lambda n, a: (n, a)):
            sync = rws.RaidenWorkerSync("rollout", bind_ip="10.11.0.5")
            sync.bind({"w": object()})
        mock_ws_cls.assert_called_once()
        return mock_ws_cls.call_args.kwargs["bind_ip"]

    def test_multi_nic_without_multi_numa_lets_kernel_route(self):
        bind_ip = self._bind_and_capture_bind_ip(
            {
                "ENABLE_MULTI_NUMA": "0",
                "TPU_RAIDEN_DATA_NICS": "eth0,eth1"
            }, ["10.11.0.5", "10.10.0.5"])
        self.assertEqual(bind_ip, "0.0.0.0")

    def test_multi_nic_with_multi_numa_keeps_native_binding(self):
        bind_ip = self._bind_and_capture_bind_ip(
            {
                "ENABLE_MULTI_NUMA": "1",
                "TPU_RAIDEN_DATA_NICS": "eth0,eth1"
            }, ["10.11.0.5", "10.10.0.5"])
        self.assertIsNone(bind_ip)

    def test_single_nic_keeps_native_binding(self):
        bind_ip = self._bind_and_capture_bind_ip(
            {
                "ENABLE_MULTI_NUMA": "0",
                "TPU_RAIDEN_DATA_NICS": ""
            }, ["10.11.0.5"])
        self.assertIsNone(bind_ip)


if __name__ == "__main__":
    unittest.main()
