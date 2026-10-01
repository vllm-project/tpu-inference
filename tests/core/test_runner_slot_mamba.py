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
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, MambaSpec)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.request import Request

from tpu_inference.core import runner_slot_mamba
from tpu_inference.core.runner_slot_mamba import (RunnerSlotMambaManager,
                                                  RunnerSlotMambaSpec,
                                                  maybe_runner_slot_mamba_spec)
from tpu_inference.platforms.tpu_platform import TpuPlatform

# Qwen3.5-style layout: one attention group plus three GDN groups. With prefix
# caching off vLLM sets the GDN block size to max_model_len, one block per
# request per group.
MAX_MODEL_LEN = 1024
BLOCK = 16
TOKENS = 160  # 10 attention blocks per request
NUM_GDN_GROUPS = 3
ATTN_PAGE_BYTES = 2 * BLOCK * 8 * 128 * 2  # K and V, 8 heads x 128, bf16


def _attn_spec() -> FullAttentionSpec:
    return FullAttentionSpec(block_size=BLOCK,
                             num_kv_heads=8,
                             head_size=128,
                             dtype=torch.bfloat16)


def _gdn_spec(spec_cls=MambaSpec,
              mamba_cache_mode: str = "none",
              mamba_type=MambaAttentionBackendEnum.GDN_ATTN) -> MambaSpec:
    return spec_cls(shapes=((3, 64), (8, 64, 16)),
                    dtypes=(torch.bfloat16, torch.float32),
                    block_size=MAX_MODEL_LEN,
                    page_size_padded=ATTN_PAGE_BYTES,
                    mamba_type=mamba_type,
                    mamba_cache_mode=mamba_cache_mode)


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

    def test_flag_off_keeps_spec(self, flag_off):
        spec = _gdn_spec()
        assert maybe_runner_slot_mamba_spec(spec, _vllm_config()) is spec

    def test_flag_on_swaps_gdn_spec(self, flag_on):
        spec = _gdn_spec()
        out = maybe_runner_slot_mamba_spec(spec, _vllm_config())
        assert type(out) is RunnerSlotMambaSpec
        assert isinstance(out, MambaSpec)
        for f in dataclasses.fields(spec):
            assert getattr(out, f.name) == getattr(spec, f.name), f.name
        assert out.page_size_bytes == spec.page_size_bytes

    @pytest.mark.parametrize(
        "config_kwargs, reason",
        [
            (dict(speculative=True), "speculative decoding"),
            (dict(mamba_cache_mode="align"), "mamba cache mode"),
            (dict(kv_connector=True), "KV connector"),
        ],
        ids=["mtp", "prefix_caching", "kv_connector"],
    )
    def test_flag_ignored_keeps_stock_spec(self, flag_on, config_kwargs,
                                           reason):
        spec = _gdn_spec(
            mamba_cache_mode=config_kwargs.get("mamba_cache_mode", "none"))
        with patch.object(runner_slot_mamba.logger, "warning_once") as warn:
            out = maybe_runner_slot_mamba_spec(spec,
                                               _vllm_config(**config_kwargs))
        assert out is spec
        warn.assert_called_once()
        assert reason in warn.call_args.args[1]

    def test_non_gdn_mamba_keeps_stock_spec(self, flag_on):
        spec = _gdn_spec(mamba_type=MambaAttentionBackendEnum.MAMBA2)
        with patch.object(runner_slot_mamba.logger, "warning_once") as warn:
            out = maybe_runner_slot_mamba_spec(spec, _vllm_config())
        assert out is spec
        assert "GDN layers only" in warn.call_args.args[1]

    def test_attention_spec_untouched(self, flag_on):
        spec = _attn_spec()
        with patch.object(runner_slot_mamba.logger, "warning_once") as warn:
            assert maybe_runner_slot_mamba_spec(spec, _vllm_config()) is spec
        warn.assert_not_called()


class TestRegistration:

    def test_platform_hook_maps_spec_to_manager(self):
        # Looking up a built-in spec loads vLLM's registrations, which end by
        # calling the platform hook.
        assert KVCacheSpecRegistry.get_manager_class(
            _gdn_spec()) is MambaManager
        # The hook may run again (another process, another config).
        TpuPlatform.register_custom_kv_cache_specs(MagicMock())
        TpuPlatform.register_custom_kv_cache_specs(MagicMock())

        spec = _gdn_spec(RunnerSlotMambaSpec)
        assert KVCacheSpecRegistry.get_manager_class(
            spec) is RunnerSlotMambaManager
        assert KVCacheSpecRegistry.get_uniform_type_base_spec(
            spec) is MambaSpec
        assert KVCacheSpecRegistry.get_manager_class(
            _gdn_spec()) is MambaManager

    def test_grouping_keeps_spec_class(self):
        # A spec that lost its subclass while vLLM groups the layers would
        # silently map back to the stock MambaManager.
        vllm_config = MagicMock()
        vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
        specs = {"attn_0": _attn_spec()}
        specs.update({
            f"gdn_{i}": _gdn_spec(RunnerSlotMambaSpec)
            for i in range(NUM_GDN_GROUPS)
        })
        groups = get_kv_cache_groups(vllm_config, specs)
        gdn_groups = [
            g for g in groups if isinstance(g.kv_cache_spec, MambaSpec)
        ]
        assert gdn_groups
        assert all(
            type(g.kv_cache_spec) is RunnerSlotMambaSpec for g in gdn_groups)


class TestSchedulerAccounting:
    """vLLM's own KVCacheManager with prefix caching off, as the scheduler
    builds it, with stock GDN specs and with RunnerSlotMambaSpec."""

    def _manager(self, gdn_spec_cls, num_blocks: int = 131) -> KVCacheManager:
        groups = [KVCacheGroupSpec(["attn_0"], _attn_spec())]
        groups += [
            KVCacheGroupSpec([f"gdn_{i}"], _gdn_spec(gdn_spec_cls))
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

    def test_stock_spec_charges_gdn_blocks(self):
        # 130 usable blocks; each request takes 10 attention + 3 GDN.
        manager = self._manager(MambaSpec)
        assert not any(
            isinstance(m, RunnerSlotMambaManager)
            for m in manager.coordinator.single_type_managers)
        assert len(self._fill(manager, 13)) == 10

    def test_runner_slot_spec_charges_attention_only(self):
        manager = self._manager(RunnerSlotMambaSpec)
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
        manager = self._manager(RunnerSlotMambaSpec)
        assert len(self._fill(manager, 1)) == 1
        assert manager.usage == pytest.approx(10 / 130)

    def test_decode_steps_free_and_refill(self):
        manager = self._manager(RunnerSlotMambaSpec)
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
