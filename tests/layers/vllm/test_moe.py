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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe import (FusedMoEMethodBase,
                                                  RoutedExperts)

# Adjust the import below to match the actual location of your function
# For example: from my_module.moe_selector import select_moe_backend_from_fused_moe_config
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.moe_weights import \
    FusedMoEWeights
# We assume the function is in a file that can be imported.
# If testing locally without the full repo, you might need to adjust imports.
from tpu_inference.layers.vllm.interface.moe import (
    select_moe_backend_from_fused_moe_config, vllm_moe_apply)


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


# ---------------------------------------------------------------------------
# vllm_moe_apply: hash-MoE routing table lookup
# ---------------------------------------------------------------------------

_MOE_MODULE = "tpu_inference.layers.vllm.interface.moe"

# Distinct tables so each test can tell which source was used.
_GATE_TABLE = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.int32)
_LAYER_TABLE = _GATE_TABLE + 100
_INPUT_IDS = torch.tensor([3, 0, 2])


def _runner(gate):
    return SimpleNamespace(gate=gate, _fused_output_is_reduced=True)


def _run_vllm_moe_apply(*,
                        layer_table,
                        runner=None,
                        forward_context=True,
                        input_ids=_INPUT_IDS):
    """Calls vllm_moe_apply with the kernel and TPU interop stubbed out.

    Returns the `extra_backend_kwargs` passed to `moe_apply` and the
    `get_layer_from_name` mock.
    """
    layer = MagicMock(spec=RoutedExperts)
    layer.layer_name = "model.layers.0.ffn.experts"
    layer.hash_indices_table = layer_table
    quant_method = MagicMock(spec=FusedMoEMethodBase)
    quant_method.extra_backend_kwargs = {}
    quant_method.moe_backend = MoEBackend.GMM_TP
    quant_method.mesh = MagicMock()

    with (patch(f"{_MOE_MODULE}.is_forward_context_available",
                return_value=forward_context),
          patch(f"{_MOE_MODULE}.get_layer_from_name", return_value=runner) as
          get_layer_from_name,
          patch(f"{_MOE_MODULE}.is_attn_dp", return_value=False),
          patch(f"{_MOE_MODULE}.jax_view", new=lambda t: t),
          patch(f"{_MOE_MODULE}.torch_view", new=lambda t: t),
          patch(f"{_MOE_MODULE}.moe_apply", return_value="out") as moe_apply,
          patch("tpu_inference.envs.MOE_ROUTE_PADDING_TO_EXPERT0", False),
          patch(
              "tpu_inference.models.vllm.vllm_model_wrapper_context."
              "get_vllm_model_wrapper_context",
              side_effect=AssertionError)):
        out = vllm_moe_apply(layer=layer,
                             weights=MagicMock(spec=FusedMoEWeights),
                             quant_method_instance=quant_method,
                             x=torch.zeros(3, 8),
                             router_logits=torch.zeros(3, 4),
                             input_ids=input_ids)

    assert out == "out"
    return moe_apply.call_args.kwargs[
        "extra_backend_kwargs"], get_layer_from_name


def test_vllm_moe_apply_reads_hash_table_from_runner_gate():
    """The gate-owned `tid2eid` (the param bound by functional_call) is used,
    not the plain `RoutedExperts.hash_indices_table` reference."""
    extra, get_layer_from_name = _run_vllm_moe_apply(
        layer_table=_LAYER_TABLE,
        runner=_runner(SimpleNamespace(tid2eid=_GATE_TABLE)))

    get_layer_from_name.assert_called_once_with("model.layers.0.ffn.experts")
    assert torch.equal(extra["hash_based_topk_indices"],
                       _GATE_TABLE[_INPUT_IDS])


@pytest.mark.parametrize("runner, forward_context", [
    pytest.param(_runner(None), True, id="runner_without_gate"),
    pytest.param(_runner(SimpleNamespace()), True, id="gate_without_tid2eid"),
    pytest.param(
        _runner(SimpleNamespace(tid2eid=None)), True, id="gate_tid2eid_none"),
    pytest.param(None, False, id="no_forward_context"),
])
def test_vllm_moe_apply_raises_when_gate_hash_table_missing(
        runner, forward_context):
    """A hash-routed layer never falls back to the plain
    `RoutedExperts.hash_indices_table` reference (the original CPU tensor)."""
    with pytest.raises(RuntimeError, match="runner.gate.tid2eid"):
        _run_vllm_moe_apply(layer_table=_LAYER_TABLE,
                            runner=runner,
                            forward_context=forward_context)


@pytest.mark.parametrize("runner, forward_context", [
    pytest.param(_runner(None), True, id="runner_without_gate"),
    pytest.param(_runner(SimpleNamespace(tid2eid=None)),
                 True,
                 id="non_hash_dsv4_layer"),
    pytest.param(None, False, id="no_forward_context"),
])
def test_vllm_moe_apply_without_hash_table(runner, forward_context):
    extra, _ = _run_vllm_moe_apply(layer_table=None,
                                   runner=runner,
                                   forward_context=forward_context)

    assert "hash_based_topk_indices" not in extra


def test_vllm_moe_apply_hash_table_requires_input_ids():
    with pytest.raises(AssertionError, match="input_ids must be provided"):
        _run_vllm_moe_apply(layer_table=_LAYER_TABLE,
                            runner=_runner(
                                SimpleNamespace(tid2eid=_GATE_TABLE)),
                            input_ids=None)
