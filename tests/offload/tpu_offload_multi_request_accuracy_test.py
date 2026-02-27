# SPDX-License-Identifier: Apache-2.0

import pytest
from vllm import SamplingParams
from vllm.config import KVTransferConfig

from tests.offload.tpu_offload_accuracy_test import (
    _test_kv_cache_cpu_offloading_accuracy, read_prompt_from_file)


@pytest.fixture
def sampling_config():
    """deterministic sampling config"""
    return SamplingParams(temperature=0.0,
                          max_tokens=10,
                          seed=42,
                          ignore_eos=True)


@pytest.fixture
def kv_transfer_config():
    """use TPUOffloadConnector"""
    return KVTransferConfig(
        kv_connector="TPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_module_path="tpu_inference.offload.tpu_offload_connector",
    )


def _run_multi_request_test_logic(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    kv_transfer_config: KVTransferConfig,
    group_indices: list,
    num_req_per_group: int,
    label: str,
):
    """
    Common logic for multi-request tests with unique group prefixes.

    This function implements a robust isolation strategy to prevent cache
    contamination across test runs.

    Isolation Strategy:
    1. Unique Prompt Content: Each test function (1G_1R, 1G_2R, 2G_1R) uses
       a completely different set of 2000-word prompt groups (Indices 1-7).
       This ensures the model never sees the same base text across scenarios.
    2. Label-based Salting: Each request group is prefixed with a unique label.
       Even if the base content were reused, the token sequence remains
       unique. This forces the TPU Offload Connector to generate
       fresh cache keys (hashes), preventing the loading of stale KV blocks
       from previous process-resident sessions.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        sampling_config: vLLM sampling configuration.
        kv_transfer_config: KV transfer configuration for TPU offloading.
        group_indices: List of integers specifying which prompt groups to use.
        num_req_per_group: Number of identical requests to create for each group.
        label: Descriptive string used for request prefixing and logging.
    """
    num_requests = len(group_indices) * num_req_per_group
    prompts = []
    for i in group_indices:
        base_content = read_prompt_from_file(f"prompt_group_{i}.txt")
        # Ensure absolute isolation with label while maintaining group sharing
        group_prompt = f"request: {label}_{i:04d} - {base_content}"
        prompts.extend([group_prompt] * num_req_per_group)

    # Logic for cpu_chunks:
    # 1. Each prompt_group_*.txt is approximately 2000 words/tokens.
    # 2. With a standard vLLM block_size of 16 tokens, each unique prompt
    #    requires ~125 blocks (2000 / 16 = 125).
    # 3. The sampling_config.max_tokens adds another ~1-2 blocks during generation.
    # 4. Total blocks per request is ~130. We use a multiplier of 140 to provide
    #    sufficient headroom. This ensures that all requests in the batch fit
    #    entirely within the CPU host RAM without triggering LRU evictions,
    #    allowing for a 100% cache hit on the second pass.
    _test_kv_cache_cpu_offloading_accuracy(
        monkeypatch,
        sampling_config,
        kv_transfer_config,
        swap_op_type="jax",
        skip_precompile="0",
        decode_save="0",
        batched_save="0",
        cpu_chunks=str(num_requests * 140),
        prompts=prompts,
    )


def test_kv_cache_cpu_offloading_accuracy_multi_request_1G_1R(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    kv_transfer_config: KVTransferConfig,
):
    """1 Group, 1 Request per group: Uses prompt group 1"""
    _run_multi_request_test_logic(monkeypatch, sampling_config,
                                  kv_transfer_config, [1], 1, "1G1R")


def test_kv_cache_cpu_offloading_accuracy_multi_request_1G_2R(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    kv_transfer_config: KVTransferConfig,
):
    """1 Group, 2 Requests per group: Uses prompt group 2"""
    _run_multi_request_test_logic(monkeypatch, sampling_config,
                                  kv_transfer_config, [2], 2, "1G2R")


def test_kv_cache_cpu_offloading_accuracy_multi_request_2G_1R(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    kv_transfer_config: KVTransferConfig,
):
    """2 Groups, 1 Request per group: Uses prompt groups 3, 4"""
    _run_multi_request_test_logic(monkeypatch, sampling_config,
                                  kv_transfer_config, [3, 4], 1, "2G1R")
