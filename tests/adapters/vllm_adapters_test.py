# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from tpu_inference.adapters.vllm_adapters import (VllmKVCacheConfigAdapter,
                                                  VllmKVCacheSpecAdapter,
                                                  VllmLoRARequestAdapter,
                                                  VllmModelRunnerOutputAdapter,
                                                  VllmSchedulerOutputAdapter)


def test_model_runner_output_adapter():
    mock_vllm_output = Mock()
    adapter = VllmModelRunnerOutputAdapter(mock_vllm_output)
    assert adapter.vllm_output is mock_vllm_output


def test_scheduler_output_adapter():
    mock_vllm_scheduler_output = Mock()
    adapter = VllmSchedulerOutputAdapter(mock_vllm_scheduler_output)
    assert adapter.vllm_scheduler_output is mock_vllm_scheduler_output


def test_lora_request_adapter():
    mock_vllm_lora_request = Mock()
    adapter = VllmLoRARequestAdapter(mock_vllm_lora_request)
    assert adapter.vllm_lora_request is mock_vllm_lora_request


def test_kv_cache_config_adapter():
    mock_vllm_kv_cache_config = Mock()
    adapter = VllmKVCacheConfigAdapter(mock_vllm_kv_cache_config)
    assert adapter.vllm_kv_cache_config is mock_vllm_kv_cache_config


def test_kv_cache_spec_adapter():
    mock_vllm_kv_cache_spec = Mock()
    adapter = VllmKVCacheSpecAdapter(mock_vllm_kv_cache_spec)
    assert adapter.vllm_kv_cache_spec is mock_vllm_kv_cache_spec
