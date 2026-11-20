# Copyright 2025 Google LLC
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

import dataclasses
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinatorNoPrefixCache
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_planning import get_kv_cache_groups
from vllm.v1.core.single_type_kv_cache_manager import (FullAttentionManager,
                                                       MambaManager)
from vllm.v1.kv_cache_interface import (AttentionSpec, KVCacheConfig,
                                        KVCacheGroup, KVCacheLayerConfig,
                                        MambaSpec)
from vllm.v1.request import Request

from tpu_inference.core import runner_slot_mamba
from tpu_inference.core.runner_slot_mamba import (
    RunnerSlotMambaManager, maybe_runner_slot_mamba_layer_config)
from tpu_inference.platforms.tpu_platform import TpuPlatform

# Qwen3.5-style layout: one attention group plus three GDN groups. With prefix
# caching off vLLM sets the GDN block size to max_model_len, one block per
# request per group.
MAX_MODEL_LEN = 1024
BLOCK = 16
TOKENS = 160  # 10 attention blocks per request
NUM_GDN_GROUPS = 3
ATTN_PAGE_BYTES = 2 * BLOCK * 8 * 128 * 2  # K and V, 8 heads x 128, bf16


def _attn_layer_config() -> KVCacheLayerConfig:
    return FullAttentionManager.layer_config(
        AttentionSpec(block_size=BLOCK,
                      num_kv_heads=8,
                      head_size=128,
                      dtype=torch.bfloat16))


def _gdn_layer_config(
        manager_cls=MambaManager,
        spec_cls=MambaSpec,
        mamba_cache_mode: str = "none",
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN) -> KVCacheLayerConfig:
    spec = spec_cls(shapes=((3, 64), (8, 64, 16)),
                    dtypes=(torch.bfloat16, torch.float32),
                    block_size=MAX_MODEL_LEN,
                    page_size_padded=ATTN_PAGE_BYTES,
                    mamba_type=mamba_type)
    return manager_cls.layer_config(spec, mamba_cache_mode=mamba_cache_mode)


def _vllm_config(mamba_cache_mode: str = "none",
                 speculative: bool = False,
                 kv_connector: bool = False) -> MagicMock:
    vllm_config = MagicMock()
    vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    vllm_config.speculative_config = MagicMock() if speculative else None
    vllm_config.kv_transfer_config = MagicMock() if kv_connector else None
    return vllm_config


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("SKIP_MAMBA_SCHEDULER_BLOCKS", "1")


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.delenv("SKIP_MAMBA_SCHEDULER_BLOCKS", raising=False)


class TestGate:

    def test_flag_off_keeps_layer_config(self, flag_off):
        cfg = _gdn_layer_config()
        assert maybe_runner_slot_mamba_layer_config(cfg, _vllm_config()) is cfg

    def test_flag_on_swaps_gdn_manager(self, flag_on):
        cfg = _gdn_layer_config()
        out = maybe_runner_slot_mamba_layer_config(cfg, _vllm_config())
        assert out.manager_cls is RunnerSlotMambaManager
        assert out.manager_config.params == cfg.manager_config.params
        assert out.spec == cfg.spec
        assert out.spec.page_size_bytes == cfg.spec.page_size_bytes

    @pytest.mark.parametrize(
        "config_kwargs, reason",
        [
            (dict(speculative=True), "speculative decoding"),
            (dict(mamba_cache_mode="align"), "mamba cache mode"),
            (dict(kv_connector=True), "KV connector"),
        ],
        ids=["mtp", "prefix_caching", "kv_connector"],
    )
    def test_flag_ignored_keeps_stock_manager(self, flag_on, config_kwargs,
                                              reason):
        cfg = _gdn_layer_config(
            mamba_cache_mode=config_kwargs.get("mamba_cache_mode", "none"))
        with patch.object(runner_slot_mamba.logger, "warning_once") as warn:
            out = maybe_runner_slot_mamba_layer_config(
                cfg, _vllm_config(**config_kwargs))
        assert out is cfg
        warn.assert_called_once()
        assert reason in warn.call_args.args[1]

    def test_non_gdn_mamba_keeps_stock_manager(self, flag_on):
        cfg = _gdn_layer_config(mamba_type=MambaAttentionBackendEnum.MAMBA2)
        with patch.object(runner_slot_mamba.logger, "warning_once") as warn:
            out = maybe_runner_slot_mamba_layer_config(cfg, _vllm_config())
        assert out is cfg
        assert "GDN layers only" in warn.call_args.args[1]

    def test_mamba_subclass_keeps_its_config(self, flag_on):

        @dataclasses.dataclass(frozen=True)
        class OtherMambaSpec(MambaSpec):
            extra: int = 0

        cfg = _gdn_layer_config(spec_cls=OtherMambaSpec)
        with patch.object(runner_slot_mamba.logger, "warning_once") as warn:
            out = maybe_runner_slot_mamba_layer_config(cfg, _vllm_config())
        assert out is cfg
        assert "OtherMambaSpec" in warn.call_args.args[1]

    def test_attention_config_untouched(self, flag_on):
        cfg = _attn_layer_config()
        with patch.object(runner_slot_mamba.logger, "warning_once") as warn:
            assert maybe_runner_slot_mamba_layer_config(cfg,
                                                        _vllm_config()) is cfg
        warn.assert_not_called()


class TestGrouping:

    def test_flag_reaches_ray_workers(self):
        # The flag is read by the runner on every host; Ray copies only VLLM_*
        # and the platform's additional_env_vars to remote workers.
        assert "SKIP_MAMBA_SCHEDULER_BLOCKS" in TpuPlatform.additional_env_vars

    def test_grouping_keeps_manager_class(self):
        # A config that lost its manager_cls while vLLM groups the layers would
        # silently map back to the stock MambaManager.
        vllm_config = MagicMock()
        vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
        vllm_config.cache_config.block_size = BLOCK
        layer_configs = {"attn_0": _attn_layer_config()}
        layer_configs.update({
            f"gdn_{i}":
            _gdn_layer_config(manager_cls=RunnerSlotMambaManager)
            for i in range(NUM_GDN_GROUPS)
        })
        groups = get_kv_cache_groups(vllm_config, layer_configs)
        gdn_groups = [
            g for g in groups if issubclass(g.manager_cls, MambaManager)
        ]
        assert gdn_groups
        assert all(g.manager_cls is RunnerSlotMambaManager for g in gdn_groups)


class TestSchedulerAccounting:
    """vLLM's own KVCacheManager with prefix caching off, as the scheduler
    builds it, with stock MambaManager and with RunnerSlotMambaManager."""

    def _manager(self,
                 gdn_manager_cls,
                 num_blocks: int = 131) -> KVCacheManager:
        groups = [KVCacheGroup.from_layers({"attn_0": _attn_layer_config()})]
        groups += [
            KVCacheGroup.from_layers(
                {f"gdn_{i}": _gdn_layer_config(manager_cls=gdn_manager_cls)})
            for i in range(NUM_GDN_GROUPS)
        ]
        config = KVCacheConfig(num_blocks=num_blocks,
                               kv_cache_tensors=[],
                               kv_cache_groups=groups)
        return KVCacheManager(
            kv_cache_config=config,
            max_model_len=MAX_MODEL_LEN,
            # What vLLM's resolve_kv_cache_block_sizes passes with prefix
            # caching off: both are the lcm of the group block sizes.
            scheduler_block_size=MAX_MODEL_LEN,
            hash_block_size=MAX_MODEL_LEN,
            enable_caching=False,
        )

    @staticmethod
    def _request(i: int) -> Request:
        return Request(request_id=f"r{i}",
                       prompt_token_ids=list(range(TOKENS)),
                       sampling_params=MagicMock(),
                       pooling_params=None)

    def _fill(self, manager: KVCacheManager, n: int) -> list[Request]:
        admitted = []
        for i in range(n):
            req = self._request(i)
            if manager.allocate_slots(req, TOKENS) is None:
                break
            admitted.append(req)
        return admitted

    def test_stock_manager_charges_gdn_blocks(self):
        # 130 usable blocks; each request takes 10 attention + 3 GDN.
        manager = self._manager(MambaManager)
        assert not any(
            isinstance(m, RunnerSlotMambaManager)
            for m in manager.coordinator.single_type_managers)
        assert len(self._fill(manager, 13)) == 10

    def test_runner_slot_manager_charges_attention_only(self):
        manager = self._manager(RunnerSlotMambaManager)
        assert type(manager.coordinator) is KVCacheCoordinatorNoPrefixCache
        managers = manager.coordinator.single_type_managers
        assert all(type(m) is RunnerSlotMambaManager for m in managers[1:])

        admitted = self._fill(manager, 14)
        assert len(admitted) == 13
        assert manager.block_pool.get_num_free_blocks() == 0
        for req in admitted:
            attn_ids, *gdn_ids = manager.get_block_ids(req.request_id)
            assert len(attn_ids) == 10
            assert gdn_ids == [[]] * NUM_GDN_GROUPS

    def test_usage_is_attention_usage(self):
        manager = self._manager(RunnerSlotMambaManager)
        assert len(self._fill(manager, 1)) == 1
        assert manager.usage == pytest.approx(10 / 130)

    def test_decode_steps_free_and_refill(self):
        manager = self._manager(RunnerSlotMambaManager)
        req = self._fill(manager, 1)[0]
        req.num_computed_tokens = TOKENS
        # Decode to 192 tokens: the attention row grows to 12 blocks, and
        # remove_skipped_blocks runs on GDN groups that hold nothing.
        for _ in range(32):
            assert manager.allocate_slots(req, 1) is not None
            req.num_computed_tokens += 1
        attn_ids, *gdn_ids = manager.get_block_ids(req.request_id)
        assert len(attn_ids) == 12
        assert gdn_ids == [[]] * NUM_GDN_GROUPS

        # Freeing returns everything, as preemption does before a request
        # is re-admitted from scratch.
        manager.free(req)
        assert manager.block_pool.get_num_free_blocks() == 130
        assert len(self._fill(manager, 14)) == 13
