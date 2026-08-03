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
"""The Kimi-K3 checkpoint slicer.

Two properties matter and neither is visible from the slicer's output alone:

  * the kept bytes are the ORIGINAL bytes, for every dtype. A slice exists to
    reproduce load-path failures of the real checkpoint, so a repack that
    round-tripped uint8-packed 4-bit weights or fp32 recurrent-state tensors
    through some intermediate representation would make the rig lie.
  * the sliced config still describes a valid layer stack. The layer lists in
    ``linear_attn_config`` are 1-INDEXED, an off-by-one there silently turns a
    KDA layer into a full-attention one and the checkpoint would then fail to
    load for a reason that has nothing to do with the code under test.
"""

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).parents[2] / "tools" / "slice_kimi_k3_checkpoint.py"
_SPEC = importlib.util.spec_from_file_location("k3_slicer_under_test", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
slicer = importlib.util.module_from_spec(_SPEC)
sys.modules["k3_slicer_under_test"] = slicer
_SPEC.loader.exec_module(slicer)

# One tensor per dtype class the real checkpoint mixes inside a single shard:
# uint8 packed expert weights, their uint8 group scales, an fp32 KDA tensor,
# and a bf16 projection. Values are arbitrary but distinct, and are compared
# byte-for-byte, so nothing here depends on a tensor library.
_TENSORS = {
    "language_model.model.layers.0.self_attn.A_log": ("F32", [4],
                                                      bytes(range(16))),
    "language_model.model.layers.0.mlp.gate_proj.weight":
    ("BF16", [2, 3], bytes(range(100, 112))),
    "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed":
    ("U8", [3, 4], bytes(range(200, 212))),
    "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_scale":
    ("U8", [3, 2], bytes(range(50, 56))),
    "language_model.model.embed_tokens.weight": ("BF16", [2,
                                                          2], bytes(range(8))),
    "vision_tower.patch_embed.proj.weight": ("BF16", [2,
                                                      2], bytes(range(20,
                                                                      28))),
}


def _write_safetensors(path, tensors):
    """Write a safetensors file from {name: (dtype, shape, raw_bytes)}."""
    header = {}
    cursor = 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + len(raw)],
        }
        cursor += len(raw)
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for _, _, raw in tensors.values():
            fh.write(raw)


def _config(num_layers=8):
    """A config shaped like the released one: nested text_config, 1-indexed
    layer lists, a full-attention layer every fourth position."""
    return {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": num_layers,
            "hidden_act": "situ",
            "linear_attn_config": {
                "kda_layers":
                [i for i in range(1, num_layers + 1) if i % 4 != 0],
                "full_attn_layers":
                [i for i in range(1, num_layers + 1) if i % 4 == 0],
                "use_full_rank_gate":
                True,
            },
        },
    }


def test_kept_names_are_the_leading_layers_plus_everything_outside_them():
    weight_map = {name: "shard" for name in _TENSORS}
    kept = slicer.kept_tensor_names(weight_map, num_layers=1, keep_vision=True)
    assert "language_model.model.layers.0.self_attn.A_log" in kept
    assert not any(".layers.1." in name for name in kept)
    # Anything outside a decoder layer survives every slice depth.
    assert "language_model.model.embed_tokens.weight" in kept


def test_vision_tensors_are_kept_by_default_and_droppable():
    weight_map = {name: "shard" for name in _TENSORS}
    kept = slicer.kept_tensor_names(weight_map, 2, keep_vision=True)
    assert "vision_tower.patch_embed.proj.weight" in kept
    dropped = slicer.kept_tensor_names(weight_map, 2, keep_vision=False)
    assert "vision_tower.patch_embed.proj.weight" not in dropped


def test_slice_config_truncates_the_one_indexed_layer_lists():
    config = _config(num_layers=12)
    slicer.slice_config(config, 8)
    linear = config["text_config"]["linear_attn_config"]
    assert config["text_config"]["num_hidden_layers"] == 8
    assert linear["full_attn_layers"] == [4, 8]
    assert linear["kda_layers"] == [1, 2, 3, 5, 6, 7]
    # Every kept layer is classified exactly once: the union covers 1..8 and
    # the two lists do not overlap.
    assert set(linear["kda_layers"]) | set(linear["full_attn_layers"]) == set(
        range(1, 9))
    assert not set(linear["kda_layers"]) & set(linear["full_attn_layers"])
    # Fields that do not depend on depth are untouched.
    assert config["text_config"]["hidden_act"] == "situ"
    assert linear["use_full_rank_gate"] is True


@pytest.mark.parametrize("num_layers, reason", [
    (3, "full-attention"),
    (16, "declares only"),
])
def test_slice_config_refuses_a_slice_that_would_not_load(num_layers, reason):
    with pytest.raises(ValueError, match=reason):
        slicer.slice_config(_config(num_layers=12), num_layers)


def test_slice_config_refuses_a_layer_in_neither_list():
    config = _config(num_layers=12)
    config["text_config"]["linear_attn_config"]["kda_layers"].remove(2)
    with pytest.raises(ValueError, match="neither"):
        slicer.slice_config(config, 8)


def test_tensor_nbytes_rejects_a_header_whose_range_contradicts_its_shape():
    assert slicer.tensor_nbytes({
        "dtype": "F32",
        "shape": [2, 3],
        "data_offsets": [0, 24]
    }) == 24
    with pytest.raises(ValueError, match="data range"):
        slicer.tensor_nbytes({
            "dtype": "U8",
            "shape": [3, 4],
            "data_offsets": [0, 11]
        })


def test_repack_keeps_the_original_bytes_and_dtypes(tmp_path):
    src = tmp_path / "in.safetensors"
    dst = tmp_path / "out.safetensors"
    _write_safetensors(src, _TENSORS)
    header, data_start = slicer.read_safetensors_header(str(src))
    keep = {
        name
        for name in _TENSORS if ".layers.1." in name or "embed_tokens" in name
    }

    slicer.repack_shard(str(src), str(dst), header, data_start, keep)

    out_header, out_start = slicer.read_safetensors_header(str(dst))
    assert set(out_header) == keep
    raw = dst.read_bytes()
    for name in keep:
        dtype, shape, original = _TENSORS[name]
        entry = out_header[name]
        assert entry["dtype"] == dtype
        assert entry["shape"] == shape
        start, end = entry["data_offsets"]
        assert raw[out_start + start:out_start + end] == original
    # The dropped tensors' bytes are gone, not merely unreferenced.
    dropped_bytes = _TENSORS["language_model.model.layers.0.self_attn.A_log"][
        2]
    assert dropped_bytes not in raw


def _twelve_layer_snapshot():
    """A 12-layer checkpoint whose shards straddle the slice boundary.

    Shard 1 holds only layers that survive (whole-file copy), shard 2 straddles
    the boundary and shard 3 is mostly dropped but holds the embeddings, so one
    run covers both the copy and the repack path.
    """
    tensors = {}
    for layer in range(12):
        prefix = f"language_model.model.layers.{layer}"
        tensors[f"{prefix}.self_attn.A_log"] = ("F32", [
            2
        ], bytes([layer, 1, 2, 3, layer, 4, 5, 6]))
        tensors[f"{prefix}.block_sparse_moe.experts.0.w1.weight_packed"] = (
            "U8", [2, 3], bytes([layer + 60] * 6))
    tensors["language_model.model.embed_tokens.weight"] = ("BF16", [2, 2],
                                                           bytes(
                                                               range(240,
                                                                     248)))
    tensors["vision_tower.patch_embed.proj.weight"] = ("BF16", [2, 2],
                                                       bytes(range(230, 238)))
    shards = {
        "model-00001-of-00003.safetensors":
        lambda n: any(f".layers.{i}." in n for i in range(4)),
        "model-00002-of-00003.safetensors":
        lambda n: any(f".layers.{i}." in n for i in range(4, 9)),
        "model-00003-of-00003.safetensors":
        lambda n: True,
    }
    placed, weight_map = {}, {}
    for shard, belongs in shards.items():
        placed[shard] = {
            n: v
            for n, v in tensors.items() if n not in weight_map and belongs(n)
        }
        weight_map.update({n: shard for n in placed[shard]})
    return tensors, placed, weight_map


def test_end_to_end_slice_of_a_local_snapshot(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    tensors, placed, weight_map = _twelve_layer_snapshot()
    for shard, contents in placed.items():
        _write_safetensors(src / shard, contents)
    (src / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {
                "total_size": 0
            },
            "weight_map": weight_map
        }))
    (src / "config.json").write_text(json.dumps(_config(num_layers=12)))
    (src / "tokenizer_config.json").write_text('{"tokenizer_class": "K3"}')

    assert slicer.main(
        ["--src", str(src), "--dst",
         str(dst), "--num-layers", "8"]) == 0

    expected = {
        n
        for n in tensors if not any(f".layers.{i}." in n for i in range(8, 12))
    }
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    config = json.loads((dst / "config.json").read_text())
    assert config["text_config"]["num_hidden_layers"] == 8
    assert set(index["weight_map"]) == expected
    # total_size is recomputed over the kept tensors, not copied through.
    assert index["metadata"]["total_size"] == sum(
        len(tensors[n][2]) for n in expected)
    # The config family rides along so the slice is loadable on its own.
    assert (dst / "tokenizer_config.json").exists()
    # Both shard paths ran: one shard was copied whole, two were repacked and
    # so are smaller than their sources.
    assert (dst / "model-00001-of-00003.safetensors").read_bytes() == (
        src / "model-00001-of-00003.safetensors").read_bytes()
    assert (dst / "model-00003-of-00003.safetensors").stat().st_size < (
        src / "model-00003-of-00003.safetensors").stat().st_size
    # Every tensor the index promises is really in the shard it names, with
    # the original bytes, dtype and shape.
    for name, shard in index["weight_map"].items():
        header, start = slicer.read_safetensors_header(str(dst / shard))
        dtype, shape, original = tensors[name]
        assert header[name]["dtype"] == dtype
        assert header[name]["shape"] == shape
        lo, hi = header[name]["data_offsets"]
        assert (dst / shard).read_bytes()[start + lo:start + hi] == original
