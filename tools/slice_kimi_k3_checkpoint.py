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
"""Cut a depth-sliced snapshot out of a Kimi-K3 checkpoint.

The released Kimi-K3 checkpoint is 1.56 TB over 96 shards and only fits on a
4-host TPU pod, so every bug that lives in the *load* path -- checkpoint
layout, tensor shapes, quantization metadata -- costs a multi-host slot to
observe. Those bugs do not depend on depth: the first few decoder layers
already carry every tensor pattern the whole stack has. This tool keeps the
first ``--num-layers`` decoder layers plus the embedding / head / final-norm
block and rewrites the config and the safetensors index so the result is a
self-consistent checkpoint that a single host can serve.

The slice is deliberately NOT a distillation: the kept tensors are the
original bytes and the kept layers are the original layers, so the model's
output text is meaningless (a 93-layer stack truncated to 8 is not a language
model any more) while its *load* behaviour is bit-for-bit the real thing --
MXFP4-packed experts, fp32 KDA tensors, the multimodal ``language_model.``
wrapper, the vision tower that has to be dropped, and the compressed-tensors
quantization config. Serve failures reproduced on the slice are real failures
of the real checkpoint.

Layout fast path
----------------
When a shard's tensors are all kept, the shard is copied whole -- for a
``gs://`` source and destination that is a server-side copy, so slicing moves
no bytes to the caller at all. Shards that hold a mix of kept and dropped
tensors are repacked by copying byte ranges out of the safetensors data block,
which preserves every dtype exactly (uint8 packed weights and their uint8
scales stay uint8, fp32 recurrent-state tensors stay fp32) without needing a
tensor library that can represent them.

Usage::

    python3 tools/slice_kimi_k3_checkpoint.py \\
        --src gs://bucket/kimi/k3 --dst gs://bucket/kimi/k3-sliced \\
        --num-layers 8

Add ``--dry-run`` to print the plan (kept shards, byte totals, config edits)
without copying anything.
"""

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

# Tensor-name prefixes that belong to the vision stack. They are kept by
# default: the text-only serving path has to *drop* them, and a slice that
# omits them would not exercise that drop.
VISION_PREFIXES = ("vision_tower.", "mm_projector.")

# Matches the decoder-layer index in a tensor name, under any wrapper prefix
# (the released checkpoint nests the text stack under `language_model.`).
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")

# safetensors dtype -> bytes per element, for recomputing the index's
# total_size over the kept tensors only.
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _is_gcs(path):
    return path.startswith("gs://")


def _run(cmd):
    """Run a command, raising with its stderr attached on failure."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"[slice] command failed ({proc.returncode}): "
                           f"{' '.join(cmd)}\n{proc.stderr.decode()[-2000:]}")
    return proc.stdout


def read_bytes(path, start=None, length=None):
    """Read a whole object/file, or the byte range [start, start+length)."""
    if _is_gcs(path):
        if start is None:
            return _run(["gsutil", "cat", path])
        end = start + length - 1
        return _run(["gsutil", "cat", "-r", f"{start}-{end}", path])
    with open(path, "rb") as fh:
        if start is None:
            return fh.read()
        fh.seek(start)
        return fh.read(length)


def read_safetensors_header(path):
    """Return (header_dict, data_start) for a safetensors file.

    Only the header is read, so this costs one small range request even when
    the file is 17 GB in a bucket.
    """
    size_bytes = read_bytes(path, 0, 8)
    (header_len, ) = struct.unpack("<Q", size_bytes)
    header = read_bytes(path, 8, header_len)
    return json.loads(header), 8 + header_len


def tensor_nbytes(entry):
    """Byte size of one safetensors header entry.

    Raises when the entry's byte range disagrees with its declared dtype and
    shape -- the checkpoint would be silently truncated for that tensor, and
    this is the one place every kept tensor passes through.
    """
    start, end = entry["data_offsets"]
    size = end - start
    dtype = entry["dtype"]
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"[slice] unknown safetensors dtype {dtype!r}")
    expected = _DTYPE_BYTES[dtype]
    for dim in entry["shape"]:
        expected *= dim
    if expected != size:
        raise ValueError(
            f"[slice] header says dtype={dtype} shape={entry['shape']} "
            f"({expected} bytes) but the data range is {size} bytes")
    return size


def kept_tensor_names(weight_map, num_layers, keep_vision):
    """Names to keep: the first `num_layers` decoder layers plus everything
    that is not inside a decoder layer (embeddings, final norm, head, the
    model-level attention-residual site), minus the vision stack if dropped.
    """
    kept = []
    for name in weight_map:
        if not keep_vision and name.startswith(VISION_PREFIXES):
            continue
        match = _LAYER_RE.search(name)
        if match and int(match.group(1)) >= num_layers:
            continue
        kept.append(name)
    return sorted(kept)


def slice_config(config, num_layers):
    """Rewrite the depth-dependent fields of a Kimi-K3 config in place.

    ``linear_attn_config``'s layer lists are 1-INDEXED in the released config,
    so keeping decoder layers ``0..num_layers-1`` keeps list entries
    ``1..num_layers`` and needs no renumbering.
    """
    text = config["text_config"] if "text_config" in config else config
    old_layers = text["num_hidden_layers"]
    if num_layers > old_layers:
        raise ValueError(f"[slice] asked for {num_layers} layers but the "
                         f"config declares only {old_layers}")
    text["num_hidden_layers"] = num_layers
    edits = {"text_config.num_hidden_layers": f"{old_layers} -> {num_layers}"}

    linear = text.get("linear_attn_config")
    if linear:
        for key in ("kda_layers", "full_attn_layers"):
            if key not in linear:
                continue
            old = linear[key]
            new = [i for i in old if i <= num_layers]
            linear[key] = new
            edits[f"text_config.linear_attn_config.{key}"] = (
                f"{len(old)} entries -> {new}")
        covered = set(linear.get("kda_layers", [])) | set(
            linear.get("full_attn_layers", []))
        missing = sorted(set(range(1, num_layers + 1)) - covered)
        if missing:
            raise ValueError(
                f"[slice] 1-indexed layers {missing} are in neither "
                "kda_layers nor full_attn_layers after slicing; the layer "
                "pattern would be ambiguous")
        if not linear.get("full_attn_layers"):
            raise ValueError(
                f"[slice] no full-attention (MLA) layer survives a "
                f"{num_layers}-layer slice; raise --num-layers so the slice "
                "covers at least one")
        if len(linear.get("kda_layers", [])) < 2:
            raise ValueError(
                f"[slice] fewer than two KDA layers survive a {num_layers}-"
                "layer slice; raise --num-layers")
    return edits


def repack_shard(src_path, dst_path, header, data_start, keep_names):
    """Write a new safetensors file holding only `keep_names`.

    Byte ranges are copied straight out of the source data block, so dtypes
    and values are preserved exactly; nothing is decoded.
    """
    keep = [n for n in header if n != "__metadata__" and n in keep_names]
    keep.sort(key=lambda n: header[n]["data_offsets"][0])
    new_header = {}
    if "__metadata__" in header:
        new_header["__metadata__"] = header["__metadata__"]
    cursor = 0
    for name in keep:
        entry = header[name]
        size = tensor_nbytes(entry)
        new_header[name] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    blob = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    pad = (-len(blob)) % 8
    blob += b" " * pad
    with open(dst_path, "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        with open(src_path, "rb") as src:
            for name in keep:
                start, end = header[name]["data_offsets"]
                src.seek(data_start + start)
                remaining = end - start
                while remaining:
                    chunk = src.read(min(remaining, 32 << 20))
                    if not chunk:
                        raise RuntimeError(
                            f"[slice] {src_path} ended inside tensor {name}")
                    out.write(chunk)
                    remaining -= len(chunk)


def copy_object(src, dst):
    if _is_gcs(src) or _is_gcs(dst):
        _run(["gsutil", "-q", "cp", src, dst])
    else:
        shutil.copyfile(src, dst)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src",
                        required=True,
                        help="Source snapshot directory (local or gs://).")
    parser.add_argument(
        "--dst",
        required=True,
        help="Destination snapshot directory (local or gs://).")
    parser.add_argument("--num-layers",
                        type=int,
                        required=True,
                        help="Number of leading decoder layers to keep.")
    parser.add_argument("--index-file",
                        default=None,
                        help="Local copy of model.safetensors.index.json. "
                        "Read from --src when omitted (it is ~60 MB).")
    parser.add_argument("--drop-vision",
                        action="store_true",
                        help="Drop vision_tower/mm_projector tensors. They "
                        "are kept by default so the slice still exercises "
                        "the text-only loader's vision-drop path.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    src = args.src.rstrip("/")
    dst = args.dst.rstrip("/")
    keep_vision = not args.drop_vision

    if args.index_file:
        with open(args.index_file) as fh:
            index = json.load(fh)
    else:
        index = json.loads(read_bytes(f"{src}/model.safetensors.index.json"))
    weight_map = index["weight_map"]

    keep = set(kept_tensor_names(weight_map, args.num_layers, keep_vision))
    print(f"[slice] keeping {len(keep)} of {len(weight_map)} tensors "
          f"({args.num_layers} decoder layers, "
          f"vision={'kept' if keep_vision else 'dropped'})")

    shard_of = {}
    for name in keep:
        shard_of.setdefault(weight_map[name], set()).add(name)
    shards = sorted(shard_of)

    # Whole-shard copies vs shards that need a repack.
    whole, mixed, total_bytes = [], [], 0
    new_weight_map = {}
    for shard in shards:
        header, data_start = read_safetensors_header(f"{src}/{shard}")
        names = [n for n in header if n != "__metadata__"]
        dropped = [n for n in names if n not in keep]
        for name in shard_of[shard]:
            entry = header[name]
            total_bytes += tensor_nbytes(entry)
            new_weight_map[name] = shard
        if dropped:
            mixed.append((shard, header, data_start, len(dropped)))
        else:
            whole.append(shard)
        print(f"[slice]   {shard}: keep {len(shard_of[shard])}/{len(names)}"
              f"{' (repack)' if dropped else ' (whole-file copy)'}")

    print(f"[slice] {len(whole)} shards copied whole, {len(mixed)} repacked; "
          f"{total_bytes} tensor bytes ({total_bytes / 2**30:.1f} GiB)")

    config = json.loads(read_bytes(f"{src}/config.json"))
    edits = slice_config(config, args.num_layers)
    for key, value in edits.items():
        print(f"[slice] config: {key}: {value}")

    index["metadata"]["total_size"] = total_bytes
    index["weight_map"] = dict(sorted(new_weight_map.items()))

    if args.dry_run:
        print("[slice] dry run, nothing written")
        return 0

    workdir = tempfile.mkdtemp(prefix="k3slice-")
    try:
        with open(os.path.join(workdir, "config.json"), "w") as fh:
            json.dump(config, fh, indent=2, sort_keys=True)
        with open(os.path.join(workdir, "model.safetensors.index.json"),
                  "w") as fh:
            json.dump(index, fh)
        copy_object(os.path.join(workdir, "config.json"), f"{dst}/config.json")
        copy_object(os.path.join(workdir, "model.safetensors.index.json"),
                    f"{dst}/model.safetensors.index.json")

        # Everything that is neither a shard nor the config we just rewrote:
        # tokenizer, generation config, remote-code modules, licence.
        if _is_gcs(src):
            listing = _run(["gsutil", "ls", f"{src}/"]).decode().split()
            names = [
                os.path.basename(p) for p in listing if not p.endswith("/")
            ]
        else:
            names = os.listdir(src)
        for name in sorted(names):
            if name.endswith(".safetensors") or name in (
                    "config.json", "model.safetensors.index.json"):
                continue
            copy_object(f"{src}/{name}", f"{dst}/{name}")
            print(f"[slice] copied {name}")

        for shard in whole:
            copy_object(f"{src}/{shard}", f"{dst}/{shard}")
            print(f"[slice] copied whole {shard}")

        for shard, header, data_start, ndropped in mixed:
            local_src = os.path.join(workdir, f"in-{shard}")
            local_dst = os.path.join(workdir, shard)
            copy_object(f"{src}/{shard}", local_src)
            repack_shard(local_src, local_dst, header, data_start, keep)
            os.remove(local_src)
            copy_object(local_dst, f"{dst}/{shard}")
            os.remove(local_dst)
            print(f"[slice] repacked {shard} (dropped {ndropped} tensors)")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print(f"[slice] wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
