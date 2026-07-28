# Copyright 2025 Google LLC
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
"""Eight host devices, and the scaffolding the MoE suites here share.

The MoE suites build a serving mesh, which needs more devices than a CPU
host reports. The flag is set from a conftest in the directories that need
it rather than at the root: it is process-global, and at the root it also
reached suites that gate themselves on the device count or size their
tensors from it, turning a clean skip into a failure in one and doubling
the run time of another.

It has to be set before jax initializes a backend. pytest imports a
directory's conftest before it imports any module in that directory, and
nothing in this tree opens a backend at import, so this is early enough.

Everything below the flag is the fake layer, the fake weight bundle, the
mesh builder and the base test case the suites in this directory used to
carry a copy of each.
"""

import os

_HOST_DEVICE_FLAG = "--xla_force_host_platform_device_count"
_xla_flags = os.environ.get("XLA_FLAGS", "")
if not any(f.split("=")[0] == _HOST_DEVICE_FLAG for f in _xla_flags.split()):
    os.environ["XLA_FLAGS"] = (f"{_xla_flags} {_HOST_DEVICE_FLAG}=8").strip()

import dataclasses  # noqa: E402
from typing import Any  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax._src import test_util as jtu  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from tpu_inference.layers.common.sharding import (  # noqa: E402
    MESH_AXIS_NAMES, ShardingAxisName, ShardingAxisNameBase)

# The served generation. The route asks the kernel how much VMEM it has,
# which a CPU cannot answer; nothing numeric is decided by it.
SERVED_CHIP = pltpu.ChipVersion.TPU_7X


class FakeMoELayer:
    """The attributes moe_apply and the fused-EP gate read off a layer."""

    def __init__(self, **attributes):
        self.use_ep = True
        self.top_k = 8
        self.renormalize = True
        self.scoring_func = "softmax"
        self.activation = "silu"
        self.__dict__.update(attributes)

    def _get_name(self):
        return "FakeMoELayer"


@dataclasses.dataclass
class FakeWeights:
    """The attributes the gate reads off a serving MoE weight bundle."""
    w13_weight: Any
    w2_weight: Any
    w13_weight_scale: Any
    w2_weight_scale: Any
    w13_bias: Any = None
    w2_bias: Any = None


# Registered so the bundle traces as one argument; absent biases are None.
jax.tree_util.register_dataclass(
    FakeWeights,
    data_fields=[f.name for f in dataclasses.fields(FakeWeights)],
    meta_fields=[])


def mesh_devices(count):
    """`count` device objects. The mesh is built with jax.eval_shape, so
    these need to be device objects rather than working accelerators: the
    CPU backend supplies them wherever the accelerator list is short."""
    devices = jax.devices()
    if len(devices) < count:
        devices = jax.devices("cpu")
    return devices[:count]


def serving_mesh(shape=None, names=MESH_AXIS_NAMES, devices=None):
    """A mesh in the serving axis set, over however many devices it needs.
    The default is the served one: attention pure data parallel over eight."""
    if shape is None:
        shape = tuple(8 if axis == "attn_dp" else 1 for axis in names)
    count = int(np.prod(shape))
    devices = mesh_devices(count) if devices is None else devices
    return Mesh(np.array(devices).reshape(shape), names)


def quantize_weight(w, dtype, block=None):
    """Quantize along the contraction axis the way serving does: block None
    is one scale per output channel, an integer one per contraction block."""
    experts, contract, out = w.shape
    span = contract if block is None else block
    peak = float(jnp.finfo(dtype).max)
    blocks = contract // span
    reshaped = w.reshape(experts, blocks, span, out)
    amax = jnp.max(jnp.abs(reshaped), axis=2, keepdims=True)
    scale = jnp.where(amax == 0, 1.0, amax / peak).astype(jnp.float32)
    quantized = (reshaped / scale).astype(dtype).reshape(
        experts, contract, out)
    return quantized, scale.reshape(experts, blocks, 1, out)


def moe_activations(key, tokens, hidden, experts):
    """A batch to send through a served-shaped expert layer."""
    x_key, gate_key = jax.random.split(key)
    x = (jax.random.normal(x_key, (tokens, hidden), jnp.float32) / 10).astype(
        jnp.bfloat16)
    return x, jax.random.normal(gate_key, (tokens, experts), jnp.float32)


def relative_l2(actual, want):
    a = np.asarray(actual, np.float64)
    w = np.asarray(want, np.float64)
    return float(np.linalg.norm(a - w) / np.linalg.norm(w))


class EightDeviceTestCase(jtu.JaxTestCase):
    """An eight-device CPU serving mesh, in the base sharding axis set: the
    2D set's ATTN_DATA is a single axis, which is not the shape the adapter
    reconciles."""

    def setUp(self):
        super().setUp()
        # The device list is read here rather than at module scope, where it
        # would start the backend during collection.
        self.cpu_devices = jax.devices("cpu")[:8]
        if len(self.cpu_devices) < 8:
            self.skipTest(
                f"this suite builds an eight-device serving mesh and the CPU "
                f"backend offers {len(jax.devices('cpu'))} device(s). This "
                "conftest asks for eight before jax starts; something removed "
                "that or started a backend first.")
        ShardingAxisName.override(ATTN_DATA=ShardingAxisNameBase.ATTN_DATA)
        self.addCleanup(ShardingAxisName.reset)

    def pin_served_chip(self):
        """Answer the kernel's VMEM questions with the served chip's record,
        for the duration of the test."""
        info = pltpu.get_tpu_info_for_chip(SERVED_CHIP, 1)
        original = pltpu.get_tpu_info
        pltpu.get_tpu_info = lambda: info
        self.addCleanup(setattr, pltpu, "get_tpu_info", original)
