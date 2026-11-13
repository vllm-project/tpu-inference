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

from unittest.mock import MagicMock, patch

import pytest

# Adjust the import below to match the actual location of your function
# For example: from my_module.moe_selector import select_moe_backend_from_fused_moe_config
from tpu_inference.layers.common.moe import MoEBackend
# We assume the function is in a file that can be imported.
# If testing locally without the full repo, you might need to adjust imports.
from tpu_inference.layers.vllm.interface.moe import \
    select_moe_backend_from_fused_moe_config


# Mock the FusedMoEConfig so we don't need vllm installed to test the logic
@pytest.fixture
def mock_moe_config():
    """Creates a mock object mimicking FusedMoEConfig."""
    config = MagicMock()
    return config


@pytest.mark.parametrize(
    "env_use_ep_kernel, config_use_ep, expected_backend",
    [
        # Case 1: EP Kernel Env=True, Config EP=True -> FUSED_MOE
        (True, True, MoEBackend.FUSED_MOE),

        # Case 2: EP Kernel Env=False, Config EP=True -> GMM_EP
        (False, True, MoEBackend.GMM_EP),

        # Case 3: EP Kernel Env=False, Config EP=False -> GMM_TP
        (False, False, MoEBackend.GMM_TP),

        # Case 4: EP Kernel Env=True, Config EP=False -> Fallback to GMM_TP
        (True, False, MoEBackend.GMM_TP),
    ])
def test_select_moe_backend_logic(monkeypatch, mock_moe_config,
                                  env_use_ep_kernel, config_use_ep,
                                  expected_backend):
    """
    Tests the main logic paths for backend selection based on environment variables
    and the MoE configuration.
    """
    # Use patch as a context manager to limit the scope of the change to this block.
    # This avoids using monkeypatch and ensures the mock is cleaned up immediately.
    with (patch("tpu_inference.envs.USE_MOE_EP_KERNEL", env_use_ep_kernel),
          patch("tpu_inference.envs.USE_FUSED_EP_MOE_V2",
                False), patch("tpu_inference.envs.USE_ROUTING_MOE", False)):
        mock_moe_config.use_ep = config_use_ep

        result = select_moe_backend_from_fused_moe_config(mock_moe_config)

    assert result == expected_backend


@pytest.mark.parametrize(
    "env_use_v2, env_use_ep_kernel, config_use_ep, expected_backend",
    [
        # USE_FUSED_EP_MOE_V2 with EP -> FUSED_EP_MOE_V2
        (True, False, True, MoEBackend.FUSED_EP_MOE_V2),
        # Without EP it falls back to GMM_TP
        (True, False, False, MoEBackend.GMM_TP),
        # USE_MOE_EP_KERNEL keeps precedence when both are set
        (True, True, True, MoEBackend.FUSED_MOE),
    ])
def test_select_fused_ep_moe_v2(mock_moe_config, env_use_v2, env_use_ep_kernel,
                                config_use_ep, expected_backend):
    with (patch("tpu_inference.envs.USE_FUSED_EP_MOE_V2", env_use_v2),
          patch("tpu_inference.envs.USE_MOE_EP_KERNEL", env_use_ep_kernel)):
        mock_moe_config.use_ep = config_use_ep
        result = select_moe_backend_from_fused_moe_config(mock_moe_config)
    assert result == expected_backend


def test_select_warns_when_both_kernels_set(mock_moe_config):
    """With both flags set, the v1 fused kernel runs and a warning says so."""
    mock_moe_config.use_ep = True
    with (patch("tpu_inference.envs.USE_FUSED_EP_MOE_V2",
                True), patch("tpu_inference.envs.USE_MOE_EP_KERNEL", True),
          patch("tpu_inference.envs.USE_ROUTING_MOE",
                False), patch("tpu_inference.layers.vllm.interface.moe.logger")
          as logger):
        result = select_moe_backend_from_fused_moe_config(mock_moe_config)
    assert result == MoEBackend.FUSED_MOE
    logger.warning_once.assert_called_once()
    assert "USE_FUSED_EP_MOE_V2" in logger.warning_once.call_args.args[0]


@pytest.mark.parametrize(
    "env_use_routing, env_use_v2, env_use_ep_kernel, config_use_ep, "
    "expected_backend",
    [
        # USE_ROUTING_MOE with EP -> ROUTING_MOE
        (True, False, False, True, MoEBackend.ROUTING_MOE),
        # Without EP it falls back to GMM_TP
        (True, False, False, False, MoEBackend.GMM_TP),
        # USE_FUSED_EP_MOE_V2 and USE_MOE_EP_KERNEL take precedence
        (True, True, False, True, MoEBackend.FUSED_EP_MOE_V2),
        (True, False, True, True, MoEBackend.FUSED_MOE),
    ])
def test_select_routing_moe(mock_moe_config, env_use_routing, env_use_v2,
                            env_use_ep_kernel, config_use_ep,
                            expected_backend):
    with (patch("tpu_inference.envs.USE_ROUTING_MOE", env_use_routing),
          patch("tpu_inference.envs.USE_FUSED_EP_MOE_V2", env_use_v2),
          patch("tpu_inference.envs.USE_MOE_EP_KERNEL", env_use_ep_kernel)):
        mock_moe_config.use_ep = config_use_ep
        result = select_moe_backend_from_fused_moe_config(mock_moe_config)
    assert result == expected_backend


@pytest.mark.parametrize("env_use_v2, env_use_ep_kernel, expected_backend", [
    (True, False, MoEBackend.FUSED_EP_MOE_V2),
    (False, True, MoEBackend.FUSED_MOE),
])
def test_select_warns_when_routing_moe_is_overridden(mock_moe_config,
                                                     env_use_v2,
                                                     env_use_ep_kernel,
                                                     expected_backend):
    """With USE_ROUTING_MOE and a higher-priority kernel flag set, the other
    kernel runs and a warning names USE_ROUTING_MOE."""
    mock_moe_config.use_ep = True
    with (patch("tpu_inference.envs.USE_ROUTING_MOE", True),
          patch("tpu_inference.envs.USE_FUSED_EP_MOE_V2", env_use_v2),
          patch("tpu_inference.envs.USE_MOE_EP_KERNEL", env_use_ep_kernel),
          patch("tpu_inference.layers.vllm.interface.moe.logger") as logger):
        result = select_moe_backend_from_fused_moe_config(mock_moe_config)
    assert result == expected_backend
    logger.warning_once.assert_called_once()
    assert "USE_ROUTING_MOE" in logger.warning_once.call_args.args[0]
