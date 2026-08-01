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
"""Shard-then-decode MXFP4 experts produce the host decode's exact bytes.

Decoding each device's shard on that device instead of decoding a whole layer
on the host is only sound because the fp4 unpack and the E8M0 expansion are
elementwise along the packed axis, so every shard boundary falls on a whole
uint8 word. That is an argument about the sharding arithmetic, so these tests
compare the two paths *bit for bit* -- `.view(uint8)` / `.view(uint32)`, not
`==`, so a `-0.0` that should be `+0.0` cannot pass -- over the mesh shapes
Kimi-K3 actually uses, and check that a shard which would split a packed word
is refused rather than silently mis-decoded.

Needs at least 8 devices for the interesting mesh shapes. On a host without
them, run with
``XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu``.

`K3_TINY_MXFP4_CKPT` and `K3_REAL_SHARD` (see the sibling MXFP4 tests) add the
same comparison over real checkpoint bytes.
"""

import glob
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from safetensors import safe_open

from tpu_inference.layers.common.quantization import (e8m0_to_fp32,
                                                      u8_unpack_e2m1)
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.layers.jax.quantization.mxfp4 import (
    CompressedTensorsMxfp4MoEMethod, _decode_sharded)

TINY_MXFP4_CKPT = os.environ.get("K3_TINY_MXFP4_CKPT", "")
REAL_SHARD_DIR = os.environ.get("K3_REAL_SHARD", "")
REAL_SHARD_LAYER = int(os.environ.get("K3_REAL_SHARD_LAYER", "1"))

# `(expert axis, model axis)` device counts. K3 runs `E` over
# `ShardingAxisName.ATTN_DATA_EXPERT` and `F` over `ShardingAxisName.MODEL`;
# the pairs below are the composed mesh the sliced-checkpoint rig uses (2x4),
# plus each axis carrying the whole split on its own.
MESH_SHAPES = ((2, 4), (1, 8), (8, 1), (4, 2))

EXPERT_AXIS = "attn_dp_expert"
MODEL_AXIS = "model"


def _mesh(num_expert: int, num_model: int) -> Mesh:
    devices = jax.devices()
    if len(devices) < num_expert * num_model:
        pytest.skip(f"needs {num_expert * num_model} devices, "
                    f"have {len(devices)}")
    shape = tuple(num_expert if name == EXPERT_AXIS else num_model if name ==
                  MODEL_AXIS else 1 for name in MESH_AXIS_NAMES)
    grid = np.array(devices[:num_expert * num_model]).reshape(shape)
    return Mesh(grid, axis_names=MESH_AXIS_NAMES)


def _specs():
    """`(edf, efd)` in the spelling `routed_expert_shardings` produces."""
    return (P(EXPERT_AXIS, None, MODEL_AXIS), P(EXPERT_AXIS, MODEL_AXIS, None))


def _stage(rng, num_experts, out, packed_in, group):
    """Per-expert `[1, out, in]` staging tensors, as `load_weights` builds."""
    packed = [
        jnp.asarray(rng.integers(0, 256, (1, out, packed_in), dtype=np.uint8))
        for _ in range(num_experts)
    ]
    scale = [
        jnp.asarray(
            rng.integers(0,
                         256, (1, out, packed_in * 2 // group),
                         dtype=np.uint8)) for _ in range(num_experts)
    ]
    return packed, scale


def _assert_bit_identical(new, old, what):
    assert new.shape == old.shape, f"{what}: {new.shape} != {old.shape}"
    assert new.dtype == old.dtype, f"{what}: {new.dtype} != {old.dtype}"
    assert new.sharding == old.sharding, (
        f"{what}: {new.sharding} != {old.sharding}")
    raw = np.uint8 if old.dtype.itemsize == 1 else np.uint32
    got = np.asarray(new).view(raw)
    want = np.asarray(old).view(raw)
    mismatch = int(np.count_nonzero(got != want))
    assert mismatch == 0, (f"{what}: {mismatch} of {got.size} raw words "
                           "differ from the host decode")


def _compare_paths(packed, scale, spec, mesh):
    """Both decode paths over the same staged tensors, compared bitwise."""
    host_values, host_scale = CompressedTensorsMxfp4MoEMethod._decode_on_host(
        packed, scale, spec, mesh)
    sharded_values = _decode_sharded(packed, u8_unpack_e2m1, 2, spec, mesh)
    sharded_scale = _decode_sharded(scale, e8m0_to_fp32, 1, spec, mesh)
    _assert_bit_identical(sharded_values, host_values, f"values {spec}")
    _assert_bit_identical(sharded_scale, host_scale, f"scales {spec}")


@pytest.mark.parametrize("num_expert,num_model", MESH_SHAPES)
def test_shard_then_decode_is_bit_identical_to_the_host_decode(
        num_expert, num_model):
    """Every K3 mesh shape, both expert-kernel orientations."""
    mesh = _mesh(num_expert, num_model)
    rng = np.random.default_rng(0)
    edf, efd = _specs()
    # `[E, out, in//2]`: gate/up are `[F, D]`, down is `[D, F]`. Both widths
    # divide by 8 the way K3's do (D=3584, F=3072, 32-wide groups), including
    # the scales' `in//group` axis, which `efd` shards.
    for spec in (edf, efd):
        packed, scale = _stage(rng, 16, out=64, packed_in=128, group=32)
        _compare_paths(packed, scale, spec, mesh)


def test_decode_survives_the_loader_context():
    """As the loader calls it: inside a mesh context, caches cleared between
    layers.

    `_load_module` runs `jax.clear_caches()` after every module's
    `process_weights_after_loading`, which drops `jax.jit`'s trace and lowering
    caches -- so the per-shard decode is held as a compiled executable, and
    this checks that the second "layer" reuses it instead of compiling again
    (93 layers x 6 shard shapes of needless compiles otherwise) and still
    produces the host decode's bytes.
    """
    from tpu_inference.layers.jax.quantization import mxfp4

    mesh = _mesh(2, 4)
    rng = np.random.default_rng(2)
    spec = _specs()[1]
    packed, scale = _stage(rng, 16, out=64, packed_in=128, group=32)
    mxfp4._DECODE_EXECUTABLES.clear()
    with jax.set_mesh(mesh):
        _compare_paths(packed, scale, spec, mesh)
        compiled_after_first = len(mxfp4._DECODE_EXECUTABLES)
        jax.clear_caches()
        _compare_paths(packed, scale, spec, mesh)
    assert compiled_after_first > 0
    assert len(mxfp4._DECODE_EXECUTABLES) == compiled_after_first, (
        "a second layer of the same shape re-compiled its per-shard decode")


def test_a_shard_that_splits_a_packed_word_is_refused():
    """The nibble-alignment guard, not a silently wrong decode."""
    mesh = _mesh(1, 4)
    rng = np.random.default_rng(1)
    # `in//2 = 2` packed words -> 4 fp4 values over 4 model shards, so every
    # shard but the first starts on a half-word.
    packed, _ = _stage(rng, 1, 1, 2, group=4)
    with pytest.raises(ValueError, match="splits a packed uint8 word"):
        _decode_sharded(packed, u8_unpack_e2m1, 2,
                        P(EXPERT_AXIS, MODEL_AXIS, None), mesh)


def _checkpoint_experts(path, layer, prefix, projection, limit=None):
    """`(packed, scale)` staging lists for one layer's expert projection."""
    with safe_open(path, framework="numpy") as f:
        base = f"{prefix}.layers.{layer}.block_sparse_moe.experts"
        ids = sorted(
            int(n[len(base) + 1:].split(".")[0]) for n in f.keys()
            if n.startswith(base + ".")
            and n.endswith(f".{projection}.weight_packed"))
        if limit:
            ids = ids[:limit]
        if not ids:
            pytest.skip(f"no {projection} experts for layer {layer}")
        staged = []
        for kind in ("weight_packed", "weight_scale"):
            staged.append([
                jnp.asarray(
                    f.get_tensor(f"{base}.{i}.{projection}.{kind}")[None])
                for i in ids
            ])
    return staged[0], staged[1]


@pytest.mark.parametrize("projection", ("w1", "w2"))
def test_tiny_checkpoint_bytes_decode_identically(projection):
    """Real MXFP4 bytes, small enough to run everywhere."""
    if not TINY_MXFP4_CKPT or not os.path.isdir(TINY_MXFP4_CKPT):
        pytest.skip("set K3_TINY_MXFP4_CKPT")
    mesh = _mesh(2, 4)
    shard = os.path.join(TINY_MXFP4_CKPT, "model.safetensors")
    packed, scale = _checkpoint_experts(shard, 1, "model", projection)
    edf, efd = _specs()
    _compare_paths(packed, scale, efd if projection == "w2" else edf, mesh)


@pytest.mark.parametrize("projection", ("w1", "w2"))
def test_real_shard_bytes_decode_identically(projection):
    """The released checkpoint's own bytes: every expert of a real layer."""
    if not REAL_SHARD_DIR or not os.path.isdir(REAL_SHARD_DIR):
        pytest.skip("set K3_REAL_SHARD")
    shards = glob.glob(os.path.join(REAL_SHARD_DIR, "*.safetensors"))
    if len(shards) != 1:
        pytest.skip(f"expected one shard in {REAL_SHARD_DIR}, "
                    f"found {len(shards)}")
    mesh = _mesh(2, 4)
    limit = int(os.environ.get("K3_REAL_DECODE_EXPERTS", "0")) or None
    packed, scale = _checkpoint_experts(shards[0], REAL_SHARD_LAYER,
                                        "language_model.model", projection,
                                        limit)
    edf, efd = _specs()
    _compare_paths(packed, scale, efd if projection == "w2" else edf, mesh)
