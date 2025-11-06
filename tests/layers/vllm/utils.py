import jax
import torch


def get_spmd_mesh(num_devices: int = 1):
    axis_names = ("data", "model")
    devices = sorted(jax.devices(), key=lambda d: d.id)[0:num_devices]
    mesh_shape = (1, len(devices))
    return jax.make_mesh(mesh_shape, axis_names, devices=devices)


def find_all_layer_type(module: torch.nn.Module, layer_type: torch.nn.Module):
    ret = []
    for name, child in module.named_children():
        if isinstance(child, layer_type):
            ret.append(child)
        else:
            ret.extend(find_all_layer_type(child, layer_type))
    return ret
