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

Plane names are taken from each input as-is; xprof identifies devices/
hosts by plane name, so coinciding names from peer DP ranks (e.g.
`/device:TPU:0` on a single host) would conflict. Pass `--prefix-planes`
to namespace each input's planes with its filename stem (e.g.
`dp0_t1v-..-w-0: /device:TPU:0`) if you need to keep them distinct.

Usage:
    # Merge specific files
    python scripts/merge_xprof.py a.xplane.pb b.xplane.pb -o merged.xplane.pb

    # Merge every .xplane.pb under a dir (typically plugins/profile/<ts>/)
    python scripts/merge_xprof.py path/to/plugins/profile/<ts> -o merged.xplane.pb

    # Namespace per-input planes to avoid name collisions
    python scripts/merge_xprof.py <inputs> -o merged.xplane.pb --prefix-planes
"""

import argparse
import os
import sys
from typing import List


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


def _collect_inputs(paths: List[str]) -> List[str]:
    """Expand input list: directories → every .xplane.pb under them (recursive)."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, fnames in os.walk(p):
                for fn in sorted(fnames):
                    if fn.endswith(".xplane.pb"):
                        files.append(os.path.join(root, fn))
        elif os.path.isfile(p):
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return files


def _prefix_for(path: str) -> str:
    """Return a short prefix derived from the file's stem (drop .xplane.pb)."""
    base = os.path.basename(path)
    if base.endswith(".xplane.pb"):
        base = base[:-len(".xplane.pb")]
    return base


def merge(inputs: List[str], output: str, prefix_planes: bool) -> None:
    xplane_pb2 = _load_xplane_pb2()
    merged = xplane_pb2.XSpace()
    total_planes = 0

    for path in inputs:
        with open(path, "rb") as fh:
            xs = xplane_pb2.XSpace()
            xs.ParseFromString(fh.read())
        if prefix_planes:
            prefix = _prefix_for(path)
            for plane in xs.planes:
                plane.name = f"{prefix}: {plane.name}"
        merged.planes.extend(xs.planes)
        merged.errors.extend(xs.errors)
        merged.warnings.extend(xs.warnings)
        merged.hostnames.extend(xs.hostnames)
        print(f"  + {path}: {len(xs.planes)} plane(s)")
        total_planes += len(xs.planes)

    out_dir = os.path.dirname(os.path.abspath(output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output, "wb") as fh:
        fh.write(merged.SerializeToString())

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
    parser.add_argument(
        "--prefix-planes",
        action="store_true",
        help="Prefix each input's plane names with the input's filename "
        "stem so coinciding names across DP ranks stay distinct in the "
        "merged XSpace. Off by default to preserve original device names "
        "(some xprof tools key off them).")
    args = parser.parse_args()

    try:
        inputs = _collect_inputs(args.inputs)
    except FileNotFoundError as e:
        print(f"error: input not found: {e}", file=sys.stderr)
        return 2
    if not inputs:
        print("error: no .xplane.pb files found in the given inputs",
              file=sys.stderr)
        return 2

    try:
        merge(inputs, args.output, args.prefix_planes)
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
