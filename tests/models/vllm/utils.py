from typing import Optional

import jax
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import \
    CompressedTensorsConfig


def get_spmd_mesh():
    axis_names = ("data", "model")
    devices = sorted(jax.devices(), key=lambda d: d.id)
    mesh_shape = (1, len(devices))
    return jax.make_mesh(mesh_shape, axis_names, devices=devices)


def quantized_matmul_ref(self,
                         layer: torch.nn.Module,
                         x: torch.Tensor,
                         bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    w_q = getattr(layer, "weight")
    w_s = getattr(layer, "weight_scale")
    output = F.linear(x, w_q.type(torch.bfloat16))
    output = output * w_s

    if bias is not None:
        output = output + bias
    return output


def gen_vllm_w8a8_int8_config():
    return CompressedTensorsConfig.from_config({
        "format": "int-quantized",
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "dynamic": True,
                    "num_bits": 8,
                    "strategy": "token",
                    "symmetric": True,
                    "type": "int"
                },
                "targets": ["Linear"],
                "weights": {
                    "dynamic": False,
                    "num_bits": 8,
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "int"
                }
            }
        }
    })
