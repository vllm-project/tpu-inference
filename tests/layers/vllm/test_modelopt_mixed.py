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
"""Tests for ModelOpt MIXED_PRECISION (modelopt_mixed) quantization on TPU.

The quantization config mirrors nvidia/Qwen3.8-2.4T-A95B-NVFP4: per-tensor FP8
attention / linear-attention projections, NVFP4 routed experts, everything
else unquantized and the MTP block excluded. Weights are fed through the real
vLLM weight loaders in the checkpoint's layout (one tensor per projection and
per expert).
"""
import tempfile

import jax
import pytest
import torch
import torchax
from jax._src import test_util as jtu
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import j2t, t2j
from vllm.config import ParallelConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForCausalLM

from tests.layers.common import utils as test_utils
from tests.layers.vllm.nvfp4_utils import (NVFP4_GROUP_SIZE, quantize_to_nvfp4,
                                           ref_dequant_nvfp4)
from tpu_inference.layers.common.quant_methods import MODELOPT_MIXED
from tpu_inference.layers.vllm.interface.moe import FusedMoEFactory
from tpu_inference.layers.vllm.process_weights.cleanup_sharding import \
    _tensor_is_in_cpu
from tpu_inference.layers.vllm.quantization.fp8 import VllmFp8LinearMethod
from tpu_inference.layers.vllm.quantization.modelopt_mixed import \
    VllmModelOptMixedConfig
from tpu_inference.layers.vllm.quantization.nvfp4 import VllmNvfp4MoEMethod
from tpu_inference.layers.vllm.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod, VllmUnquantizedLinearMethod)
from tpu_inference.models.vllm.vllm_model_loader import \
    attach_incremental_weight_loader
from tpu_inference.platforms.tpu_platform import TpuPlatform

BASE_MODEL = "Qwen/Qwen3-0.6B"  # only used to build a VllmConfig
FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max


def qwen38_style_quant_config(num_layers: int = 4,
                              full_attention_interval: int = 4,
                              overrides: dict | None = None) -> dict:
    """Shaped like `quantization_config` in the NVFP4 checkpoint's config.json."""
    layers = {}
    for i in range(num_layers):
        if (i + 1) % full_attention_interval == 0:
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                layers[f"model.layers.{i}.self_attn.{proj}"] = {
                    "quant_algo": "FP8"
                }
        else:
            for proj in ("conv1d", "in_proj_qkv", "in_proj_z", "in_proj_b",
                         "in_proj_a", "out_proj"):
                layers[f"model.layers.{i}.linear_attn.{proj}"] = {
                    "quant_algo": "FP8"
                }
        layers[f"model.layers.{i}.mlp.experts"] = {
            "quant_algo": "NVFP4",
            "group_size": NVFP4_GROUP_SIZE,
        }
    layers.update(overrides or {})
    return {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "producer": {
            "name": "modelopt",
            "version": "0.42.0"
        },
        "kv_cache_scheme": {
            "dynamic": False,
            "num_bits": 8,
            "type": "float"
        },
        "ignore": ["mtp*", "mtp.layers.0*"],
        "quantized_layers": layers,
    }


def make_vllm_config(num_devices: int = 1):
    engine_args = EngineArgs(model=BASE_MODEL,
                             max_model_len=64,
                             max_num_batched_tokens=64,
                             max_num_seqs=4)
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = torch.bfloat16
    vllm_config.parallel_config = ParallelConfig(
        tensor_parallel_size=num_devices)
    return vllm_config


def make_quant_config(vllm_config, mesh, overrides=None):
    VllmModelOptMixedConfig.set_configs(vllm_config, mesh)
    quant_config = VllmModelOptMixedConfig.from_config(
        qwen38_style_quant_config(overrides=overrides))
    # vLLM copies the model's packed_modules_mapping onto the quant config.
    quant_config.packed_modules_mapping = dict(
        Qwen3_5MoeForCausalLM.packed_modules_mapping)
    # Production sets this; VllmQKVParallelLinear reads the mesh from it.
    vllm_config.quant_config = quant_config
    return quant_config


@pytest.fixture(autouse=True)
def setup_environment():
    vllm_config = make_vllm_config()
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            1,
            0,
            local_rank=0,
            distributed_init_method=f"file://{tempfile.mkstemp()[1]}",
            backend="gloo")
        ensure_model_parallel_initialized(1, 1)


def test_registration_and_override():
    assert get_quantization_config(MODELOPT_MIXED) is VllmModelOptMixedConfig
    assert MODELOPT_MIXED in TpuPlatform.supported_quantization
    assert VllmModelOptMixedConfig.override_quantization_method(
        qwen38_style_quant_config(), user_quant=None) == MODELOPT_MIXED
    assert VllmModelOptMixedConfig.override_quantization_method(
        {
            "quant_method": "modelopt",
            "quant_algo": "FP8"
        }, user_quant=None) is None


# ---------------------------------------------------------------------------
# Layer construction helpers, shaped like Qwen3.8's vLLM layers (scaled down).
# ---------------------------------------------------------------------------


def build_linear(kind: str, prefix: str, quant_config, vllm_config):
    common = dict(bias=False,
                  params_dtype=torch.bfloat16,
                  quant_config=quant_config,
                  prefix=prefix)
    with set_current_vllm_config(vllm_config):
        if kind == "qkv":  # self_attn.qkv_proj <- q_proj, k_proj, v_proj
            layer = QKVParallelLinear(hidden_size=1024,
                                      head_size=128,
                                      total_num_heads=16,
                                      total_num_kv_heads=4,
                                      **common)
            shards = [("q", 2048), ("k", 512), ("v", 512)]
        elif kind == "qkvz":  # linear_attn.in_proj_qkvz <- in_proj_qkv/_z
            # Qwen3.5 layout: four output parts [q, k, v, z]. The checkpoint's
            # in_proj_qkv fills parts (0, 1, 2) with a single tensor and a
            # single per-tensor scale; in_proj_z fills part 3.
            layer = MergedColumnParallelLinear(
                input_size=1024,
                output_sizes=[256, 256, 2048, 2048],
                return_bias=False,
                **common)
            shards = [((0, 1, 2), 256 + 256 + 2048), (3, 2048)]
            check_sizes = [256, 256, 2048, 2048]
        elif kind == "ba":  # linear_attn.in_proj_ba <- in_proj_b/_a
            layer = MergedColumnParallelLinear(input_size=1024,
                                               output_sizes=[128, 128],
                                               return_bias=False,
                                               **common)
            shards = [(0, 128), (1, 128)]
        elif kind == "row":  # self_attn.o_proj / linear_attn.out_proj
            layer = RowParallelLinear(input_size=2048,
                                      output_size=1024,
                                      return_bias=False,
                                      **common)
            shards = [(None, 1024)]
        elif kind == "gate_up":  # mlp.shared_expert.gate_up_proj
            layer = MergedColumnParallelLinear(input_size=1024,
                                               output_sizes=[512, 512],
                                               return_bias=False,
                                               **common)
            shards = [(0, 512), (1, 512)]
        elif kind == "router":  # mlp.gate
            layer = ReplicatedLinear(input_size=1024,
                                     output_size=16,
                                     return_bias=False,
                                     **common)
            shards = [(None, 16)]
        else:
            raise ValueError(kind)
    if kind != "qkvz":
        check_sizes = [rows for _, rows in shards]
    return layer, shards, check_sizes


def build_moe(prefix,
              quant_config,
              vllm_config,
              num_experts=4,
              topk=2,
              hidden_size=256,
              intermediate_size=256):
    with set_current_vllm_config(vllm_config):
        return FusedMoEFactory(num_experts=num_experts,
                               top_k=topk,
                               hidden_size=hidden_size,
                               intermediate_size=intermediate_size,
                               renormalize=False,
                               tp_size=1,
                               dp_size=1,
                               quant_config=quant_config,
                               prefix=prefix)


def test_layer_dispatch():
    vllm_config = make_vllm_config()
    mesh = test_utils.get_spmd_mesh(1)
    qc = make_quant_config(vllm_config, mesh)

    fp8_layers = [
        ("qkv", "model.layers.3.self_attn.qkv_proj"),
        ("row", "model.layers.3.self_attn.o_proj"),
        ("qkvz", "model.layers.0.linear_attn.in_proj_qkvz"),
        ("ba", "model.layers.0.linear_attn.in_proj_ba"),
        ("row", "model.layers.0.linear_attn.out_proj"),
    ]
    for kind, prefix in fp8_layers:
        layer, _, _ = build_linear(kind, prefix, qc, vllm_config)
        assert isinstance(layer.quant_method, VllmFp8LinearMethod), prefix
        assert not layer.quant_method.block_quant, prefix
        assert hasattr(layer, "input_scale"), prefix

    unquantized_layers = [
        ("gate_up", "model.layers.0.mlp.shared_expert.gate_up_proj"),
        ("router", "model.layers.0.mlp.gate"),
        ("qkv", "mtp.layers.0.self_attn.qkv_proj"),  # excluded
    ]
    for kind, prefix in unquantized_layers:
        layer, _, _ = build_linear(kind, prefix, qc, vllm_config)
        assert isinstance(layer.quant_method,
                          VllmUnquantizedLinearMethod), prefix

    moe = build_moe("model.layers.0.mlp.experts", qc, vllm_config)
    assert isinstance(moe.routed_experts.quant_method, VllmNvfp4MoEMethod)
    mtp_moe = build_moe("mtp.layers.0.mlp.experts", qc, vllm_config)
    assert isinstance(mtp_moe.routed_experts.quant_method,
                      VllmUnquantizedFusedMoEMethod)


def test_unsupported_algos_raise():
    vllm_config = make_vllm_config()
    mesh = test_utils.get_spmd_mesh(1)
    qc = make_quant_config(vllm_config,
                           mesh,
                           overrides={
                               **{
                                   f"model.layers.3.self_attn.{p}": {
                                       "quant_algo": "MXFP8"
                                   }
                                   for p in ("q_proj", "k_proj", "v_proj")
                               },
                               "model.layers.1.mlp.experts": {
                                   "quant_algo": "FP8"
                               },
                           })
    with pytest.raises(NotImplementedError, match="MXFP8"):
        build_linear("qkv", "model.layers.3.self_attn.qkv_proj", qc,
                     vllm_config)
    with pytest.raises(NotImplementedError, match="FP8 MoE"):
        build_moe("model.layers.1.mlp.experts", qc, vllm_config)


# ---------------------------------------------------------------------------
# Numerics: per-tensor FP8 linear layers.
# ---------------------------------------------------------------------------


def quantize_per_tensor_fp8(w: torch.Tensor):
    scale = w.abs().max() / FP8_MAX
    return (w / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8), scale.reshape(1)


def load_fp8_shards(layer, shards):
    """One FP8 weight + F32 [1] weight_scale + input_scale per checkpoint
    projection, through the vLLM loaders. Returns the dequantized weight."""
    torch.manual_seed(0)
    deq_rows = []
    for i, (shard_id, rows) in enumerate(shards):
        # Different magnitude per shard so a misplaced scale shows up.
        w = torch.randn(rows, layer.input_size) * (0.02 * (i + 1))
        w_q, w_s = quantize_per_tensor_fp8(w)
        for name, tensor in (("weight", w_q), ("weight_scale", w_s),
                             ("input_scale", torch.tensor([0.05]))):
            param = getattr(layer, name)
            if shard_id is None:
                param.weight_loader(param, tensor)
            else:
                param.weight_loader(param, tensor, shard_id)
        deq_rows.append(w_q.to(torch.float32) * w_s)
    return torch.cat(deq_rows, dim=0)


FP8_CASES = {
    "qkv": "model.layers.3.self_attn.qkv_proj",
    "qkvz": "model.layers.0.linear_attn.in_proj_qkvz",
    "ba": "model.layers.0.linear_attn.in_proj_ba",
    "row": "model.layers.3.self_attn.o_proj",
}


@pytest.mark.parametrize("kind", list(FP8_CASES))
@pytest.mark.parametrize("num_devices", [1, jax.local_device_count()])
def test_fp8_per_tensor_linear(kind, num_devices):
    vllm_config = make_vllm_config(num_devices)
    mesh = test_utils.get_spmd_mesh(num_devices)
    qc = make_quant_config(vllm_config, mesh)
    layer, shards, check_sizes = build_linear(kind, FP8_CASES[kind], qc,
                                              vllm_config)
    qm = layer.quant_method
    assert isinstance(qm, VllmFp8LinearMethod)
    assert qm.linear_config.fuse_matmuls  # default for these layers
    qm.linear_config.requant_block_size = None
    qm.linear_config.requant_weight_dtype = jax.numpy.float8_e4m3fn

    w_deq = load_fp8_shards(layer, shards)
    torch.manual_seed(1)
    x = (torch.randn(16, layer.input_size) * 0.5).to(torch.bfloat16)
    ref = x.to(torch.float32) @ w_deq.T

    mesh_ctx = getattr(jax, "set_mesh", None) or jax.sharding.use_mesh
    with torchax.default_env(), mesh_ctx(mesh):
        qm.process_weights_after_loading(layer)

        # Run under jax.jit like the model runner: the KV-head replication
        # path of VllmQKVParallelLinear uses abstract-mesh shard_maps.
        def fwd(x_jax):
            y = layer(torch_view(x_jax))
            return jax_view(y[0] if isinstance(y, tuple) else y)

        out = jax.jit(fwd)(t2j(x, use_dlpack=False))
        out = j2t(out.astype(jax.numpy.float32))

    # Check every output part separately (e.g. q, k, v, z of in_proj_qkvz):
    # a scale that lands in the wrong slot shows up in that part only.
    start = 0
    for part, size in enumerate(check_sizes):
        o, r = out[:, start:start + size], ref[:, start:start + size]
        rel = ((o - r).norm() / r.norm()).item()
        # Dynamic FP8 activation quantization alone gives ~0.027; a misplaced
        # scale gives >= 0.25.
        assert rel < 0.05, f"{kind} part {part}: rel_err {rel:.3f}"
        start += size


# ---------------------------------------------------------------------------
# Numerics: NVFP4 routed experts loaded in the checkpoint layout.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not jtu.is_device_tpu_at_least(version=7),
                    reason="NVFP4 MoE needs TPUv7+")
@pytest.mark.parametrize("num_devices", [1, 2])
@pytest.mark.parametrize("disable_requant", [False, True])
@pytest.mark.parametrize("load_mode", ["eager", "staged", "incremental"])
def test_nvfp4_moe_checkpoint_layout(num_devices, disable_requant, load_mode,
                                     monkeypatch):
    """eager: process after loading. staged: MOE_STAGE_WEIGHTS_ON_HOST.
    incremental: VLLM_INCREMENTAL_FP8_LOADING + MOE_STAGE_WEIGHTS_ON_HOST
    through the incremental loader wrapper, like the multi-host runs."""
    if disable_requant:
        monkeypatch.setenv("DISABLE_WEIGHT_REQUANTIZATION", "1")
    if load_mode in ("staged", "incremental"):
        monkeypatch.setenv("MOE_STAGE_WEIGHTS_ON_HOST", "1")
    if load_mode == "incremental":
        monkeypatch.setenv("VLLM_INCREMENTAL_FP8_LOADING", "1")
    num_experts, topk, hidden, inter = 4, 2, 256, 256
    prefix = "model.layers.0.mlp.experts"
    vllm_config = make_vllm_config(num_devices)
    mesh = test_utils.get_spmd_mesh(num_devices)
    qc = make_quant_config(vllm_config, mesh)
    moe_layer = build_moe(prefix, qc, vllm_config, num_experts, topk, hidden,
                          inter)
    experts = moe_layer.routed_experts
    assert isinstance(experts.quant_method, VllmNvfp4MoEMethod)

    if load_mode == "incremental":
        attach_incremental_weight_loader(moe_layer)

    torch.manual_seed(42)
    w1_deq = torch.empty(num_experts, 2 * inter, hidden)
    w2_deq = torch.empty(num_experts, hidden, inter)
    # The model loader loads on the CPU device; incremental processing runs
    # inside the weight loader, so it happens in this context too.
    load_ctx = jax.default_device(jax.devices("cpu")[0])
    load_ctx.__enter__()
    for e in range(num_experts):
        # gate (w1), up (w3), down (w2) are separate tensors in the checkpoint,
        # each with its own weight_scale_2; gate/up input_scale match.
        for proj, shard_id, shape in (("gate_proj", "w1", (inter, hidden)),
                                      ("up_proj", "w3", (inter, hidden)),
                                      ("down_proj", "w2", (hidden, inter))):
            w = torch.randn(shape) / 10
            packed, scale, global_scale = quantize_to_nvfp4(
                w, NVFP4_GROUP_SIZE)
            deq = ref_dequant_nvfp4(packed, scale, global_scale,
                                    NVFP4_GROUP_SIZE)
            if shard_id == "w1":
                w1_deq[e, :inter] = deq
            elif shard_id == "w3":
                w1_deq[e, inter:] = deq
            else:
                w2_deq[e] = deq
            w_prefix = "w2" if shard_id == "w2" else "w13"
            for suffix, tensor in (("weight", packed), ("weight_scale", scale),
                                   ("weight_scale_2", global_scale.reshape(1)),
                                   ("input_scale", torch.tensor([0.01]))):
                name = f"{w_prefix}_{suffix}"
                if not hasattr(experts, name):
                    # Incremental: the layer was processed after its last
                    # required weight; the remaining activation scales are
                    # unused on TPU.
                    assert load_mode == "incremental" and suffix == "input_scale"
                    continue
                param = getattr(experts, name)
                if (load_mode != "incremental" and name == "w13_weight_scale_2"
                        and shard_id == "w3"):
                    # The separate gate/up global scales must both survive.
                    assert param.shape == (num_experts, 2)
                # param.weight_loader is the incremental wrapper when attached.
                param.weight_loader(param,
                                    tensor,
                                    f"{prefix}.{name}",
                                    shard_id=shard_id,
                                    expert_id=e)
    load_ctx.__exit__(None, None, None)

    if load_mode == "incremental":
        # Processed as soon as the last expert's weights arrived.
        assert not _tensor_is_in_cpu(experts.w13_weight)
    else:
        assert not torch.equal(experts.w13_weight_scale_2[:, 0],
                               experts.w13_weight_scale_2[:, 1])

    a = torch.randn((8, hidden), dtype=torch.bfloat16) / 10
    score = torch.randn((8, num_experts), dtype=torch.bfloat16)
    expected = test_utils.ref_moe(a, score, w1_deq.to(torch.bfloat16),
                                  w2_deq.to(torch.bfloat16), None, None, topk,
                                  experts.renormalize,
                                  moe_layer.activation.value)

    with torchax.default_env(), set_forward_context(None, vllm_config):
        # A no-op if incremental loading already processed the layer.
        experts.quant_method.process_weights_after_loading(experts)
        assert not _tensor_is_in_cpu(experts.w13_weight)
        actual = moe_layer(a.to("jax"), score.to("jax"))
        actual = j2t(jax_view(actual).astype(jax.numpy.float32))

    expected = expected.to(torch.float32)
    rel = ((actual - expected).norm() / expected.norm()).item()
    print(f"nvfp4 moe devices={num_devices} disable_requant={disable_requant}"
          f" load_mode={load_mode} rel_err={rel:.4f}")
    # Reference is the dequantized checkpoint, so only kernel / requant error
    # remains. Default requantization re-quantizes FP4 per row, which is
    # lossier than keeping the native 16-element blocks.
    assert rel < (0.05 if disable_requant else 0.25), rel


def test_nvfp4_moe_keeps_only_local_experts():
    """On a multi-host slice each host copies only the experts it holds.

    The single-host mask must cover every expert; a synthetic two-host mask
    must leave the other host's experts unwritten while still acknowledging
    their tensors to the model's weight loader.
    """
    num_experts, hidden, inter = 4, 256, 256
    vllm_config = make_vllm_config()
    mesh = test_utils.get_spmd_mesh(1)
    qc = make_quant_config(vllm_config, mesh)
    moe_layer = build_moe("model.layers.0.mlp.experts", qc, vllm_config,
                          num_experts, 2, hidden, inter)
    experts = moe_layer.routed_experts
    qm = experts.quant_method
    assert all(qm._local_expert_mask(num_experts))

    local = [True, False, True, False]
    qm._keep_local_experts_only(experts, local)
    for name in ("w13_weight", "w2_weight", "w13_weight_scale_2",
                 "w2_weight_scale_2"):
        getattr(experts, name).data.zero_()

    torch.manual_seed(0)
    for e in range(num_experts):
        for proj, shard_id, shape in (("gate_proj", "w1", (inter, hidden)),
                                      ("up_proj", "w3", (inter, hidden)),
                                      ("down_proj", "w2", (hidden, inter))):
            packed, scale, global_scale = quantize_to_nvfp4(
                torch.randn(shape) / 10, NVFP4_GROUP_SIZE)
            w_prefix = "w2" if shard_id == "w2" else "w13"
            for suffix, tensor in (("weight", packed), ("weight_scale", scale),
                                   ("weight_scale_2",
                                    global_scale.reshape(1))):
                name = f"{w_prefix}_{suffix}"
                param = getattr(experts, name)
                assert param.weight_loader(
                    param,
                    tensor,
                    f"model.layers.0.mlp.experts.{name}",
                    shard_id=shard_id,
                    expert_id=e,
                    return_success=True) is True

    for e in range(num_experts):
        for name in ("w13_weight", "w2_weight", "w13_weight_scale_2",
                     "w2_weight_scale_2"):
            written = bool(getattr(experts, name).data[e].any())
            assert written == local[e], (name, e)
    # With only local experts kept, processing waits for the full load.
    assert qm._defer_processing
