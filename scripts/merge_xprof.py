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
"""Combine multiple .xplane.pb captures into a single XSpace .xplane.pb.

Phased MPMD profiling writes one xplane.pb per DP rank into
`<phase>/plugins/profile/<ts>/dp{0,1,...}_*.xplane.pb`. Tools like the
TensorBoard profile plugin and `xprof` can already read that whole
directory as one distributed session, but some workflows want a single
artifact — e.g. attaching to a bug, feeding into a tool that takes
exactly one file, or sharing via a single URL. This script merges the
XSpaces from each input into one output proto.

For files sharing a host (the MPMD case — peer DP ranks on one worker),
plane names collide (every rank has `/device:TPU:0`) and plane ids
collide (JAX numbers devices per-process from 0). xprof's trace_viewer
dedups by id, so a naive concat silently drops peer-rank data. To
resolve:

  * Planes whose name ends in `<prefix>:<id>` (`/device:TPU:0`, or any
    future `/host:CPU:0`-style plane) are renumbered globally per host,
    independently per prefix. With DP=4 × TP=2, the 4 ranks'
    `/device:TPU:0,1` become `/device:TPU:0..7`. SparseCore indices are
    per-chip and stay verbatim (e.g. `SparseCore 0`).
  * Planes without a numeric suffix (`/host:CPU`,
    `/device:CUSTOM:Megascale Trace`, `Task Environment`) get the rank
    index appended: `/host:CPU 0`, `/host:CPU 1`, ...
  * Plane ids are reassigned to be globally unique.

Files from different hosts already have distinct hostnames, so no
cross-host renaming is applied — only the per-host TPU/ rank logic runs
independently within each host group.

Usage:
    # Merge specific files
    python scripts/merge_xprof.py a.xplane.pb b.xplane.pb -o merged.xplane.pb

    # Merge every .xplane.pb under a dir (typically plugins/profile/<ts>/)
    python scripts/merge_xprof.py path/to/plugins/profile/<ts> -o merged.xplane.pb
"""

import argparse
import os
import re
import sys
from collections import OrderedDict
from typing import List, Tuple

# Matches a plane name of the form `<prefix>:<id>` (e.g. `/device:TPU:0`,
# `/host:CPU:2`), optionally followed by any whitespace-led trailing
# segment (e.g. ` SparseCore 0`, ` Core:0`, ` Bar`). Group 1 is the
# prefix (rewritten verbatim), group 2 is the id that gets bumped, group
# 3 is the trailing segment which is preserved as-is — any indices
# inside it are sub-device (per-chip) labels and not part of the global
# renumbering scheme.
_PREFIX_ID_RE = re.compile(r"^(\S+):(\d+)(\s+.*)?$")


def _load_xplane_pb2():
    """Locate XSpace proto bindings.

    xprof / tensorboard-plugin-profile read xplane.pb via a C++ extension
    and don't ship Python bindings. TensorFlow does — but importing TF
    eagerly pulls in many heavy deps (gast, astunparse, etc.). We first
    try the regular import paths and, if those fail, fall back to loading
    the `xplane_pb2.py` file directly from site-packages, bypassing any
    parent `__init__.py` chain.
    """
    candidates_modules = (
        "tensorflow.tsl.profiler.protobuf.xplane_pb2",
        "tensorflow.core.profiler.protobuf.xplane_pb2",
        "xprof.protobuf.xplane_pb2",
        "tensorboard_plugin_profile.protobuf.xplane_pb2",
        "tensorboard.plugins.profile.protobuf.xplane_pb2",
    )
    for modname in candidates_modules:
        try:
            return __import__(modname, fromlist=["XSpace"])
        except ImportError:
            continue

    # Direct file load — searches site-packages for the generated pb2
    # without executing any package `__init__.py` along the way.
    import importlib.util
    import site
    candidate_rel_paths = (
        "tensorflow/tsl/profiler/protobuf/xplane_pb2.py",
        "tensorflow/core/profiler/protobuf/xplane_pb2.py",
        "xprof/protobuf/xplane_pb2.py",
        "tensorboard_plugin_profile/protobuf/xplane_pb2.py",
    )
    search_dirs = list(site.getsitepackages())
    user_sp = site.getusersitepackages()
    if user_sp:
        search_dirs.append(user_sp)
    for sp_dir in search_dirs:
        for rel in candidate_rel_paths:
            full = os.path.join(sp_dir, rel)
            if not os.path.exists(full):
                continue
            spec = importlib.util.spec_from_file_location(
                "_merge_xprof_xplane_pb2", full)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "XSpace"):
                return module

    raise ImportError(
        "Could not find XSpace proto bindings. Install one of:\n"
        "    pip install tensorflow  # ships xplane_pb2.py\n"
        "    pip install tensorflow-cpu  # same, no GPU/TPU runtime\n"
        "Note: `xprof` and `tensorboard-plugin-profile` do NOT ship the "
        "XSpace Python proto — they read xplane.pb via a C++ extension.")


def _collect_inputs(paths: List[str], output: str) -> List[str]:
    """Expand input list: directories → every .xplane.pb under them (recursive).

    Excludes `output` if it lives under one of the input dirs — the common
    workflow is to write `merged.xplane.pb` back into the source directory,
    so a re-run that walks that dir would otherwise pick up the previous
    merged file as a fresh input and double the output each time.
    """
    out_abs = os.path.abspath(output)
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, fnames in os.walk(p):
                for fn in sorted(fnames):
                    if fn.endswith(".xplane.pb"):
                        full = os.path.join(root, fn)
                        if os.path.abspath(full) == out_abs:
                            continue
                        files.append(full)
        elif os.path.isfile(p):
            if os.path.abspath(p) != out_abs:
                files.append(p)
        else:
            raise FileNotFoundError(p)
    return files


def _group_by_host(inputs: List[str],
                   xplane_pb2) -> "OrderedDict[str, List[Tuple[str, object]]]":
    """Read each input and group (path, XSpace) tuples by hostname.

    The hostname comes from `XSpace.hostnames[0]` (JAX writes the worker
    host there). Files missing a hostname are bucketed under the empty
    string; they share one group, which means their planes will be
    renumbered together — the safe default for unknown provenance.
    Within each group, files are sorted alphabetically by basename so
    the rank index is deterministic across runs.
    """
    groups: "OrderedDict[str, List[Tuple[str, object]]]" = OrderedDict()
    for path in inputs:
        xs = xplane_pb2.XSpace()
        with open(path, "rb") as fh:
            xs.ParseFromString(fh.read())
        host = xs.hostnames[0] if xs.hostnames else ""
        groups.setdefault(host, []).append((path, xs))
    for host in groups:
        groups[host].sort(key=lambda pair: os.path.basename(pair[0]))
    return groups


def _rename_planes_in_host_group(files: List[Tuple[str, object]]) -> None:
    """Rewrite plane names in place for one host's files.

    Any plane whose name starts with `<prefix>:<id>` (optionally
    followed by a whitespace-led trailing segment) is renumbered
    globally across this host: each distinct prefix (`/device:TPU`,
    `/host:CPU`, ...) has its own counter that increments as new local
    ids are encountered, file by file in sort order. Within one file,
    planes that share a prefix and local id (e.g. `/device:TPU:0` and
    `/device:TPU:0 SparseCore 0`) receive the same new id; the trailing
    segment is preserved verbatim, so any sub-device indices inside it
    are left untouched. Planes that don't match the `<prefix>:<id>`
    pattern get the file's rank index appended (`/host:CPU 0`,
    `/host:CPU 1`, ...).
    """
    next_id_per_prefix: dict = {}
    for rank_idx, (_, xs) in enumerate(files):
        # Map of (prefix, local_id) → globally-allocated id, scoped to
        # this file so SparseCore planes pick up the same new id as
        # their owning device plane regardless of plane order.
        local_to_global: dict = {}
        for plane in xs.planes:
            m = _PREFIX_ID_RE.match(plane.name)
            if m:
                prefix, local_id_str, sc_suffix = (m.group(1), m.group(2),
                                                   m.group(3) or "")
                key = (prefix, int(local_id_str))
                if key not in local_to_global:
                    local_to_global[key] = next_id_per_prefix.get(prefix, 0)
                    next_id_per_prefix[prefix] = local_to_global[key] + 1
                plane.name = f"{prefix}:{local_to_global[key]}{sc_suffix}"
            else:
                plane.name = f"{plane.name} {rank_idx}"


def merge(inputs: List[str], output: str) -> None:
    xplane_pb2 = _load_xplane_pb2()
    host_groups = _group_by_host(inputs, xplane_pb2)

    for host, files in host_groups.items():
        _rename_planes_in_host_group(files)

    # Stitch into one XSpace, assigning globally unique plane ids.
    # xprof's trace_viewer dedups by id (not name), so peer-rank XSpaces
    # that each numbered from the same id space (typical for JAX MPMD)
    # would lose data without a fresh allocation here.
    merged = xplane_pb2.XSpace()
    next_id = 1
    total_planes = 0
    for host, files in host_groups.items():
        if host:
            merged.hostnames.append(host)
        for path, xs in files:
            for plane in xs.planes:
                plane.id = next_id
                next_id += 1
            merged.planes.extend(xs.planes)
            merged.errors.extend(xs.errors)
            merged.warnings.extend(xs.warnings)
            print(f"  + {path}: {len(xs.planes)} plane(s)")
            total_planes += len(xs.planes)

    out_dir = os.path.dirname(os.path.abspath(output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # Write atomically via .tmp + rename so a mid-write failure (ENOSPC,
    # SIGKILL, etc.) doesn't leave a half-serialized file at `output`
    # that downstream tools will silently treat as valid.
    tmp_path = output + ".tmp"
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(merged.SerializeToString())
        os.replace(tmp_path, output)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    out_size_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"Merged {len(inputs)} file(s) → {output} "
          f"({total_planes} planes, {out_size_mb:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more .xplane.pb files, or a directory containing them "
        "(walked recursively).")
    parser.add_argument("-o",
                        "--output",
                        required=True,
                        help="Output path for the merged .xplane.pb.")
    args = parser.parse_args()

    try:
        inputs = _collect_inputs(args.inputs, args.output)
    except FileNotFoundError as e:
        print(f"error: input not found: {e}", file=sys.stderr)
        return 2
    if not inputs:
        print("error: no .xplane.pb files found in the given inputs",
              file=sys.stderr)
        return 2

    try:
        merge(inputs, args.output)
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
