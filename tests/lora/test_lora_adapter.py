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

import gc
import os

import pytest
import vllm
from vllm.lora.request import LoRARequest

# -------------------------------------------------------------------------
# Multi-Host CI Detection
# -------------------------------------------------------------------------
# The model Qwen2.5-3B-Instruct has only 2 KV heads, so it cannot support
# a tensor parallel size > 2. If we run TP=1 or TP=2 on a 16-chip multi-host
# cluster, JAX will deadlock waiting for the other hosts to join the mesh.
is_large_topology = False

# Heuristic 1: Check Buildkite CI step name (e.g., "tpu6e_test_16")
buildkite_step = os.environ.get("BUILDKITE_STEP_KEY", "").lower()
if "16" in buildkite_step:
    is_large_topology = True

# Heuristic 2: Check standard TPU multi-host environment variables
worker_hostnames = os.environ.get("TPU_WORKER_HOSTNAMES", "")
if worker_hostnames and "," in worker_hostnames:
    is_large_topology = True

skip_reason = (
    "Skipping large topology: Base model Qwen2.5-3B-Instruct has 2 KV heads "
    "and cannot run on >2 chips without causing a multi-host JAX deadlock.")
skip_multihost = pytest.mark.skipif(is_large_topology, reason=skip_reason)

# For multi-chip test, we only use TP=2 because the base model Qwen2.5-3B-Instruct has 2 kv heads
TP = [2] if os.environ.get("TEST_LORA_TP", False) else [1]


def setup_vllm(num_loras: int, tp: int = 1) -> vllm.LLM:
    return vllm.LLM(
        model="Qwen/Qwen2.5-3B-Instruct",
        max_model_len=256,
        max_num_batched_tokens=64,
        max_num_seqs=8,
        tensor_parallel_size=tp,
        enable_lora=True,
        max_loras=num_loras,
        async_scheduling=0,
        max_lora_rank=8,
    )


def test_dummy_pass():
    """
    Dummy test to ensure Pytest returns Exit Code 0 (Success) instead of 
    Exit Code 5 ("No tests collected") when the real tests are skipped.
    """
    assert True


@skip_multihost
@pytest.mark.parametrize("tp", TP)
def test_dynamic_lora_loading_api(tp):
    """This test verifies we can load, list, pin, and unload adapters dynamically
    using the LLMEngine dynamic adapter-management APIs.
    """
    llm = setup_vllm(4, tp)

    lora_name_template = "Username6568/Qwen2.5-3B-Instruct-1_plus_1_equals_{}_adapter"
    lora_request = LoRARequest("lora_adapter_2", 2,
                               lora_name_template.format(2))

    # 1. Dynamically add the adapter
    success = llm.llm_engine.add_lora(lora_request)
    assert success is True

    # 2. Verify it is listed in the registered adapters
    registered_loras = llm.llm_engine.list_loras()
    assert 2 in registered_loras

    # 3. Pin the adapter to prevent eviction
    assert llm.llm_engine.pin_lora(2) is True

    # 4. Dynamically remove the adapter
    success_remove = llm.llm_engine.remove_lora(2)
    assert success_remove is True

    # 5. Verify it is no longer listed
    assert 2 not in llm.llm_engine.list_loras()

    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()


@skip_multihost
@pytest.mark.parametrize("tp", TP)
def test_dynamic_lora_loading_multiple(tp):
    """Loads multiple adapters dynamically and verifies pinning/listing."""
    llm = setup_vllm(4, tp)

    lora_name_template = "Username6568/Qwen2.5-3B-Instruct-1_plus_1_equals_{}_adapter"
    req2 = LoRARequest("lora_adapter_2", 2, lora_name_template.format(2))
    req3 = LoRARequest("lora_adapter_3", 3, lora_name_template.format(3))

    # Load both adapters
    assert llm.llm_engine.add_lora(req2) is True
    assert llm.llm_engine.add_lora(req3) is True

    # Verify listings
    registered = llm.llm_engine.list_loras()
    assert 2 in registered
    assert 3 in registered

    # Pin adapter 3
    assert llm.llm_engine.pin_lora(3) is True

    # Unload both
    assert llm.llm_engine.remove_lora(2) is True
    assert llm.llm_engine.remove_lora(3) is True
    assert len(llm.llm_engine.list_loras()) == 0

    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()


@skip_multihost
@pytest.mark.parametrize("tp", TP)
def test_dynamic_lora_lru_eviction(tp):
    """Tests LRU caching behaviour by loading more adapters than max_loras."""
    # Set max_loras to 1
    llm = setup_vllm(1, tp)

    lora_name_template = "Username6568/Qwen2.5-3B-Instruct-1_plus_1_equals_{}_adapter"
    req2 = LoRARequest("lora_adapter_2", 2, lora_name_template.format(2))
    req3 = LoRARequest("lora_adapter_3", 3, lora_name_template.format(3))

    # 1. Load adapter 2
    assert llm.llm_engine.add_lora(req2) is True
    assert 2 in llm.llm_engine.list_loras()

    # 2. Load adapter 3 (since max_loras=1, this should evict adapter 2)
    assert llm.llm_engine.add_lora(req3) is True
    registered = llm.llm_engine.list_loras()
    assert 3 in registered
    assert 2 not in registered  # evicted

    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
