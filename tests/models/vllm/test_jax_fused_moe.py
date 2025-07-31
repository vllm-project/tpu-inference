import jax
import pytest
import torch
import torchax
import utils as test_utils
from jax.sharding import NamedSharding, PartitionSpec
from torchax.interop import torch_view
from torchax.ops.mappings import j2t, t2j
from vllm.config import ParallelConfig, set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.moe_torch_iterative import \
    fused_moe as torch_moe

from tpu_commons.models.vllm.jax_fused_moe import JaxFusedMoE

P = PartitionSpec


@pytest.fixture(scope="module", autouse=True)
def setup_torchax():
    """Enable torchax globally before all tests, disable after all tests."""
    torchax.enable_globally()
    yield
    torchax.disable_globally()


@pytest.mark.parametrize("use_ep", [True, False])
@pytest.mark.parametrize("mesh", [test_utils.get_spmd_mesh()])
@pytest.mark.parametrize("num_tokens", [8])
@pytest.mark.parametrize("intermediate_size", [1024, 2048])
@pytest.mark.parametrize("hidden_size", [128, 512])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("topk", [2])
def test_jax_fused_moe(use_ep, mesh, num_tokens, intermediate_size,
                       hidden_size, num_experts, topk):
    dtype = torch.bfloat16

    a = torch.randn((num_tokens, hidden_size), dtype=dtype) / 10
    w1 = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size), dtype=dtype) / 10
    w2 = torch.randn(
        (num_experts, hidden_size, intermediate_size), dtype=dtype) / 10
    score = torch.randn((num_tokens, num_experts), dtype=dtype)

    torch_output = torch_moe(
        hidden_states=a,
        w1=w1,
        w2=w2,
        gating_output=score,
        topk=topk,
        global_num_experts=num_experts,
        expert_map=None,
        renormalize=False,
    )

    engine_args = EngineArgs(
        model="Qwen/Qwen2-1.5B-Instruct",
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = dtype
    with set_current_vllm_config(vllm_config):
        vllm_fused_moe = FusedMoE(num_experts=num_experts,
                                  top_k=topk,
                                  hidden_size=hidden_size,
                                  intermediate_size=intermediate_size,
                                  reduce_results=False,
                                  renormalize=False,
                                  tp_size=1,
                                  dp_size=1)
    vllm_fused_moe.w13_weight.data = w1
    vllm_fused_moe.w2_weight.data = w2

    vllm_parallel_config = ParallelConfig()
    vllm_parallel_config.enable_expert_parallel = use_ep

    a = torch_view(t2j(a))
    a.apply_jax_(jax.device_put, NamedSharding(mesh, P(None, None)))
    score = torch_view(t2j(score))
    score.apply_jax_(jax.device_put, NamedSharding(mesh, P(None, None)))

    jax_fused_moe = JaxFusedMoE(vllm_fused_moe, mesh, vllm_parallel_config)
    jax_output = jax_fused_moe(hidden_states=a, router_logits=score)
    # j2t() doens't support bfloat16, so we cast it into float32 as an intermedate step.
    jax_output = j2t(jax_output.to(torch.float32)).to(dtype)

    # The error margins are adapted from vllm tests/tpu/test_moe_pallas.py
    torch.testing.assert_close(
        torch_output,
        jax_output,
        atol=2e-2,
        rtol=0,
    )
