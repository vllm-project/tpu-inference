# SPDX-License-Identifier: Apache-2.0
import tempfile

import jax
import pytest
import torch
from jax.sharding import Mesh
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs

from tpu_commons.models.torchax.tpu import TPUModelLoader


def _setup_environment(model):
    engine_args = EngineArgs(model=model, )
    vllm_config = engine_args.create_engine_config()
    with set_current_vllm_config(vllm_config):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(
            1,
            0,
            local_rank=0,
            distributed_init_method=f"file://{temp_file}",
            backend="gloo")
        # Under single worker mode, full model is init first and then
        # partitioned using GSPMD.
        ensure_model_parallel_initialized(1, 1)
    return vllm_config


REPLACED_MODULE_NAMES = {
    "QKVParallelLinear",
    "MergedColumnParallelLinear",
}

SINGLE_CHIP_TEST_MODELS = {
    "Qwen/Qwen2-1.5B-Instruct",
}

MULTI_CHIP_TEST_MODELS = {
    "meta-llama/Llama-3.1-8B-Instruct",
}


@pytest.mark.parametrize("model", [
    "Qwen/Qwen2-1.5B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
])
def test_tpu_model_loader(model):
    vllm_config = _setup_environment(model)
    # Workaround since it's converted in platforms/tpu_jax.py
    vllm_config.model_config.dtype = torch.bfloat16
    if model in SINGLE_CHIP_TEST_MODELS:
        mesh = None
    else:
        mesh = Mesh(jax.devices(), axis_names=('x', ))
    loader = TPUModelLoader(load_config=vllm_config.load_config)
    model = loader.load_model(vllm_config=vllm_config,
                              model_config=vllm_config.model_config,
                              mesh=mesh)

    if mesh is not None:
        for name, module in model.named_modules():
            fqn = module.__class__.__qualname__
            assert fqn not in REPLACED_MODULE_NAMES, \
                f"Module {fqn} should be replaced by JAX version, " \
                "please check the TPU backend configuration."
