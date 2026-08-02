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
"""Per-host sharded expert streaming: filter, gate, and decode guarantees.

Covers the four load-bearing pieces of ``K3_SHARDED_EXPERT_STREAMING=1``:

1. The needed-expert set derived from the kernels' sharding partitions the
   experts across simulated hosts and matches the per-device index map the
   shard decode itself uses.
2. The filtered streaming requests cover exactly the kept tensors' byte
   ranges (contiguous runs coalesced), and streaming them yields the kept
   tensors bit-identically to ``safetensors.safe_open``.
3. Decoding from a host's staged subset (non-local experts left ``None``)
   is bit-identical to the full decode, and never touches an unstaged slot.
4. The relaxed completeness gate waits on the local set only, and a needed
   expert that never arrives fails loudly with its ids, not as a
   half-loaded model.

Needs 8 devices for the mesh shapes. On a host without them, run with
``XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu``.
"""

import json
import struct
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from safetensors import safe_open
from safetensors.torch import save_file

from tpu_inference import envs
from tpu_inference.layers.common.quantization import (e8m0_to_fp32,
                                                      u8_unpack_e2m1)
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.layers.jax.moe.moe import MoEBackend
from tpu_inference.layers.jax.quantization.mxfp4 import (
    _SHARDED_STREAM_NEEDED_ATTR, CompressedTensorsMxfp4MoEMethod,
    _staged_expert_bytes, _staging_complete)
from tpu_inference.models.jax.utils.sharded_stream import (
    build_filtered_requests, expert_id_from_tensor_name,
    maybe_load_with_sharded_expert_streaming, needed_expert_ids,
    validate_sharded_streaming_complete)

EXPERT_AXIS = "attn_dp_expert"
MODEL_AXIS = "model"

# Staged checkpoint orientation is `[E, out, in]` with `in` packed 2 fp4 per
# byte; the decoded kernel is `[E, in, out]`. Sizes divide by 8 the way K3's
# do.
NUM_EXPERTS = 8
OUT = 64
PACKED_IN = 128
GROUP = 32

EDF_SPEC = P(EXPERT_AXIS, None, MODEL_AXIS)
EFD_SPEC = P(EXPERT_AXIS, MODEL_AXIS, None)


def _mesh(num_expert: int, num_model: int) -> Mesh:
    needed = num_expert * num_model
    devices = jax.devices()
    if len(devices) < needed:
        pytest.skip(f"needs {needed} devices, have {len(devices)}")
    sizes = {EXPERT_AXIS: num_expert, MODEL_AXIS: num_model}
    shape = tuple(sizes.get(name, 1) for name in MESH_AXIS_NAMES)
    grid = np.array(devices[:needed]).reshape(shape)
    return Mesh(grid, axis_names=MESH_AXIS_NAMES)


def _expert_groups(mesh: Mesh):
    """Device lists per expert-axis coordinate: one simulated host each."""
    axis = MESH_AXIS_NAMES.index(EXPERT_AXIS)
    return [
        list(np.take(mesh.devices, g, axis=axis).ravel())
        for g in range(mesh.devices.shape[axis])
    ]


# --- 1. Needed-set derivation -------------------------------------------


def test_expert_id_from_tensor_name():
    name = ("language_model.model.layers.3.block_sparse_moe.experts.17.w1."
            "weight_packed")
    assert expert_id_from_tensor_name(name) == 17
    assert expert_id_from_tensor_name("experts.0.w2.weight_scale") == 0
    assert expert_id_from_tensor_name("model.embed_tokens.weight") is None
    assert expert_id_from_tensor_name(
        "model.layers.0.mlp.experts.gate") is None
    assert expert_id_from_tensor_name(
        "model.layers.0.mlp.shared_experts.w1.weight") is None


@pytest.mark.parametrize("spec", (EDF_SPEC, EFD_SPEC))
def test_needed_expert_ids_partitions_across_expert_groups(spec):
    """Two simulated hosts' sets partition range(E) and follow the map."""
    mesh = _mesh(2, 4)
    groups = _expert_groups(mesh)
    sets = [
        needed_expert_ids(spec, mesh, NUM_EXPERTS, local_devices=group)
        for group in groups
    ]
    assert sets[0] == frozenset(range(0, 4))
    assert sets[1] == frozenset(range(4, 8))
    assert sets[0] | sets[1] == frozenset(range(NUM_EXPERTS))
    assert not sets[0] & sets[1]


def test_needed_expert_ids_default_covers_all_addressable():
    """Single process addresses the whole mesh, so the default is all E."""
    mesh = _mesh(2, 4)
    assert needed_expert_ids(EDF_SPEC, mesh,
                             NUM_EXPERTS) == frozenset(range(NUM_EXPERTS))


def test_needed_expert_ids_matches_devices_indices_map_ground_truth():
    """Per device, the real staged-shape index map's expert slice is inside
    the host set derived from the dummy-shape map."""
    mesh = _mesh(2, 4)
    named = NamedSharding(mesh, EDF_SPEC)
    shape = (NUM_EXPERTS, PACKED_IN * 2, OUT)
    index_map = named.devices_indices_map(shape)
    for group in _expert_groups(mesh):
        needed = needed_expert_ids(EDF_SPEC,
                                   mesh,
                                   NUM_EXPERTS,
                                   local_devices=group)
        for device in group:
            expert_slice = index_map[device][0]
            start = expert_slice.start or 0
            stop = (expert_slice.stop
                    if expert_slice.stop is not None else NUM_EXPERTS)
            assert set(range(start, stop)) <= needed


def test_needed_expert_ids_replicated_expert_axis_needs_everything():
    """A spec that does not shard the expert axis keeps all experts local."""
    mesh = _mesh(2, 4)
    for group in _expert_groups(mesh):
        needed = needed_expert_ids(P(None, None, MODEL_AXIS),
                                   mesh,
                                   NUM_EXPERTS,
                                   local_devices=group)
        assert needed == frozenset(range(NUM_EXPERTS))


# --- 2. Filtered request construction and streaming ----------------------


def _write_checkpoint(path):
    """A small safetensors file with expert and non-expert tensors."""
    g = torch.Generator().manual_seed(7)

    def u8(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)

    tensors = {}
    for i in range(4):
        tensors[f"model.layers.0.mlp.experts.{i}.w1.weight_packed"] = u8(
            16, 32)
    tensors["model.embed_tokens.weight"] = u8(8, 64)
    for i in range(4, 8):
        tensors[f"model.layers.0.mlp.experts.{i}.w1.weight_packed"] = u8(
            16, 32)
        tensors[f"model.layers.0.mlp.experts.{i}.w1.weight_scale"] = u8(16, 2)
    tensors["lm_head.weight"] = u8(8, 64)
    save_file(tensors, path)
    return set(tensors)


def _parse_metadata(path):
    """`(data_start, [SafetensorMetadata by offset], [sizes])` from the raw
    header, the same triple `prepare_request` builds."""
    from runai_model_streamer.safetensors_streamer.safetensors_pytorch import \
        SafetensorsMetadata

    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        blob = json.loads(f.read(header_size))
    meta = SafetensorsMetadata(blob, 8 + header_size)
    return meta.offset, meta.tensors_metadata, meta.read_sizes


KEEP_EXPERTS = frozenset({0, 1, 4, 5})


def _keep(name):
    expert_id = expert_id_from_tensor_name(name)
    return expert_id is None or expert_id in KEEP_EXPERTS


def test_build_filtered_requests_covers_exactly_the_kept_ranges(tmp_path):
    pytest.importorskip("runai_model_streamer")
    path = str(tmp_path / "model.safetensors")
    _write_checkpoint(path)
    data_start, tensors_metadata, sizes = _parse_metadata(path)

    requests, id_to_meta, stats = build_filtered_requests(
        [path], [(data_start, tensors_metadata, sizes)], _keep)

    kept = [m for m in tensors_metadata if _keep(m.name)]
    dropped = [m for m in tensors_metadata if not _keep(m.name)]
    assert stats.total_tensors == len(tensors_metadata)
    assert stats.kept_tensors == len(kept)
    assert stats.kept_bytes == sum(m.get_bytesize() for m in kept)
    assert stats.skipped_bytes == sum(m.get_bytesize() for m in dropped)
    assert dropped, "filter dropped nothing; the test is vacuous"

    # Every kept tensor appears exactly once, in a request whose byte range
    # is the run's contiguous span.
    covered = []
    assert sorted(id_to_meta) == [r.id for r in requests]
    for request in requests:
        metas = id_to_meta[request.id]
        assert request.chunks == [m.get_bytesize() for m in metas]
        assert request.offset == data_start + metas[0].offsets.start
        # Chunks within a request are contiguous in the file.
        cursor = metas[0].offsets.start
        for m in metas:
            assert m.offsets.start == cursor
            cursor = m.offsets.end
        covered.extend(m.name for m in metas)
    assert sorted(covered) == sorted(m.name for m in kept)

    # Runs are maximal: consecutive requests are separated by at least one
    # dropped tensor, so no two requests could have been merged.
    boundaries = {m.offsets.start for m in dropped}
    for first, second in zip(requests, requests[1:]):
        gap_start = id_to_meta[first.id][-1].offsets.end
        assert gap_start in boundaries


def test_filtered_stream_yields_kept_tensors_bit_identically(tmp_path):
    pytest.importorskip("runai_model_streamer")
    from runai_model_streamer import SafetensorsStreamer

    from tpu_inference.models.jax.utils.sharded_stream import _filtered_stream

    path = str(tmp_path / "model.safetensors")
    all_names = _write_checkpoint(path)
    expected = {name for name in all_names if _keep(name)}
    assert expected != all_names

    got = {}
    with SafetensorsStreamer() as streamer:
        stats = _filtered_stream(streamer, [path], _keep)
        for name, tensor in streamer.get_tensors():
            got[name] = tensor.clone()
    assert set(got) == expected
    assert stats.kept_tensors == len(expected)

    with safe_open(path, framework="pt") as f:
        for name in expected:
            assert torch.equal(got[name], f.get_tensor(name)), name


# --- 3. Subset decode == full decode -------------------------------------


def _stage(rng, out, packed_in, group):
    packed = [
        jnp.asarray(rng.integers(0, 256, (1, out, packed_in), dtype=np.uint8))
        for _ in range(NUM_EXPERTS)
    ]
    scale = [
        jnp.asarray(
            rng.integers(0,
                         256, (1, out, packed_in * 2 // group),
                         dtype=np.uint8)) for _ in range(NUM_EXPERTS)
    ]
    return packed, scale


def _slices(index, shape):
    out = []
    for s, size in zip(index, shape):
        out.append(slice(s.start or 0, size if s.stop is None else s.stop))
    return tuple(out)


@pytest.mark.parametrize("spec", (EDF_SPEC, EFD_SPEC))
def test_subset_decode_is_bit_identical_to_full_decode(spec):
    """Each simulated host decodes from only its staged subset; every device
    shard matches the full-staging host decode bit for bit."""
    mesh = _mesh(2, 4)
    rng = np.random.default_rng(11)
    packed, scale = _stage(rng, OUT, PACKED_IN, GROUP)

    full_values, full_scale = CompressedTensorsMxfp4MoEMethod._decode_on_host(
        packed, scale, spec, mesh)
    np_full_values = np.asarray(full_values).view(np.uint8)
    np_full_scale = np.asarray(full_scale).view(np.uint32)

    named = NamedSharding(mesh, spec)
    for staged, decode, expansion, np_full, raw in (
        (packed, u8_unpack_e2m1, 2, np_full_values, np.uint8),
        (scale, e8m0_to_fp32, 1, np_full_scale, np.uint32),
    ):
        packed_in = staged[0].shape[-1]
        shape = (NUM_EXPERTS, packed_in * expansion, OUT)
        index_map = named.devices_indices_map(shape)
        for group in _expert_groups(mesh):
            needed = needed_expert_ids(spec,
                                       mesh,
                                       NUM_EXPERTS,
                                       local_devices=group)
            subset = [w if i in needed else None for i, w in enumerate(staged)]
            cache = {}
            for device in group:
                expert_slice, in_slice, out_slice = _slices(
                    index_map[device], shape)
                assert in_slice.start % expansion == 0
                assert in_slice.stop % expansion == 0
                packed_slice = slice(in_slice.start // expansion,
                                     in_slice.stop // expansion)
                # The gather `_decode_sharded` performs, from the subset:
                # absent experts must never be touched (they would raise).
                local = np.concatenate([
                    _staged_expert_bytes(subset, cache, e)[:, out_slice,
                                                           packed_slice]
                    for e in range(expert_slice.start, expert_slice.stop)
                ],
                                       axis=0)
                shard = np.asarray(
                    jnp.swapaxes(decode(jnp.asarray(local)), 1, 2)).view(raw)
                want = np_full[expert_slice, in_slice, out_slice]
                assert shard.shape == want.shape
                assert np.array_equal(shard, want), (
                    f"{spec} {device}: subset-decoded shard differs from "
                    f"the full decode")


def test_gather_raises_loudly_on_an_unstaged_needed_expert():
    staged = [
        jnp.zeros((1, 4, 4), dtype=jnp.uint8), None, None,
        jnp.zeros((1, 4, 4), dtype=jnp.uint8)
    ]
    cache = {}
    _staged_expert_bytes(staged, cache, 0)  # staged: fine
    with pytest.raises(ValueError, match=r"\[mxfp4-ct\] expert 2"):
        _staged_expert_bytes(staged, cache, 2)


# --- 4. Completeness gate and the loud missing-expert failure -------------


def _param(weights):
    return SimpleNamespace(_weights_to_load=list(weights))


_STAGED_ATTRS = ("gate_packed", "gate_scale", "up_packed", "up_scale",
                 "down_packed", "down_scale")


def _fake_layer(staged_slots, needed=None, prefix="layers.0.mlp"):
    """A stand-in MoE layer: six staging attrs sharing one slot pattern."""
    layer = SimpleNamespace(prefix=prefix, moe_backend=MoEBackend.DENSE_MAT)
    for attr in _STAGED_ATTRS:
        setattr(layer, attr, _param(staged_slots))
    if needed is not None:
        setattr(layer, _SHARDED_STREAM_NEEDED_ATTR, frozenset(needed))
    layer.quant_method = CompressedTensorsMxfp4MoEMethod(layer)
    return layer


W = object()  # any staged (non-None) sentinel


def test_gate_without_needed_set_requires_every_expert():
    assert _staging_complete(_fake_layer([W, W, W, W]), list(_STAGED_ATTRS))
    assert not _staging_complete(_fake_layer([W, W, None, W]),
                                 list(_STAGED_ATTRS))


def test_gate_with_needed_set_waits_on_local_experts_only():
    # Non-local experts 2,3 never arrive; the relaxed gate must pass.
    assert _staging_complete(_fake_layer([W, W, None, None], needed={0, 1}),
                             list(_STAGED_ATTRS))
    # A missing LOCAL expert must hold the gate.
    assert not _staging_complete(
        _fake_layer([W, None, None, None], needed={0, 1}), list(_STAGED_ATTRS))


def _fake_model(*layers):
    return SimpleNamespace(named_modules=lambda: [(
        f"model.layers.{i}.mlp", layer) for i, layer in enumerate(layers)])


def test_missing_local_expert_fails_loudly_after_load():
    bad = _fake_layer([W, None, None, None], needed={0, 1})
    with pytest.raises(ValueError,
                       match=r"\[mxfp4\].*missing local expert ids \[1\]"):
        validate_sharded_streaming_complete(_fake_model(bad))


def test_decoded_layer_passes_validation():
    # Healthy end state: the decode ran and deleted the staging attributes.
    decoded = _fake_layer([W, W, None, None], needed={0, 1})
    for attr in _STAGED_ATTRS:
        delattr(decoded, attr)
    validate_sharded_streaming_complete(_fake_model(decoded))
    # Layers never marked for sharded streaming are not validated here; the
    # unrelaxed gate already guarantees all-experts staging for them.
    unmarked = _fake_layer([W, None, None, None])
    validate_sharded_streaming_complete(_fake_model(unmarked))


def test_undecoded_layer_with_complete_staging_still_fails():
    # Validation runs after the load: staging attributes still present means
    # the decode never fired, even when every local expert arrived.
    stuck = _fake_layer([W, W, None, None], needed={0, 1})
    with pytest.raises(ValueError, match="decode never ran"):
        validate_sharded_streaming_complete(_fake_model(stuck))


def test_host_decode_with_filtered_staging_is_refused(monkeypatch):
    layer = _fake_layer([W, W, None, None], needed={0, 1})
    monkeypatch.setattr(envs, "MXFP4_SHARD_THEN_DECODE", False, raising=False)
    with pytest.raises(ValueError, match="MXFP4_SHARD_THEN_DECODE"):
        layer.quant_method.process_weights_after_loading(layer)


def test_flag_off_is_a_noop(monkeypatch):
    monkeypatch.setattr(envs,
                        "K3_SHARDED_EXPERT_STREAMING",
                        False,
                        raising=False)
    # Nothing may be touched when the flag is off: passing junk proves it.
    assert maybe_load_with_sharded_expert_streaming(None, None, None) is False


def test_flag_on_with_non_runai_loader_falls_back(monkeypatch):
    monkeypatch.setattr(envs,
                        "K3_SHARDED_EXPERT_STREAMING",
                        True,
                        raising=False)
    assert maybe_load_with_sharded_expert_streaming(object(), None,
                                                    None) is False
