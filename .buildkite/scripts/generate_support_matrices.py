#!/usr/bin/env python3
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

"""
Support Matrix Generator for TPU Inference CI.

Generates nightly and release support matrix CSVs (and tarball archive) across:
- Model Support Matrix (text-only, multimodal, embedding, diffusion)
- Feature Support Matrices (general features, parallelism, quantization, RL)
- Kernel Microbenchmarks Matrix (pivoted display format)
"""

import concurrent.futures
import csv
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
from typing import Dict, Iterable, List, Optional, Set, Tuple

# ==============================================================================
# Static Schema Definitions (Table Columns & Business Rules)
# ==============================================================================
MODEL_STAGES = ["Type", "UnitTest", "Accuracy/Correctness", "Benchmark"]
MODEL_TYPES = {"multimodal": "Multimodal", "embedding": "Embedding", "diffusion": "Diffusion"}
FEATURE_STAGES = ["CorrectnessTest", "PerformanceTest"]
CATEGORY_CONFIG: Dict[str, Tuple[str, List[str]]] = {
    "quantization support matrix": (
        "Quantization dtype,Quantization methods,Recommended TPU Generations,CorrectnessTest,PerformanceTest",
        ["QuantizationMethods", "RecommendedTPUGenerations", "CorrectnessTest", "PerformanceTest"],
    ),
    "kernel support matrix microbenchmarks": (
        "kernels,CorrectnessTest,PerformanceTest",
        FEATURE_STAGES,
    ),
    "parallelism support matrix": (
        "Feature,Single-Host CorrectnessTest,Single-Host PerformanceTest,Multi-Host CorrectnessTest,Multi-Host PerformanceTest",
        ["Single-Host CorrectnessTest", "Single-Host PerformanceTest", "Multi-Host CorrectnessTest", "Multi-Host PerformanceTest"],
    ),
}
DEFAULT_CATEGORY_CONFIG = ("Feature,CorrectnessTest,PerformanceTest", FEATURE_STAGES)

QUANTIZATION_SPECS = {
    "INT8 W8A8": ("compressed-tensor", '"v5, v6"'),
    "INT4 W4A16": ("awq", '"v5, v6"'),
    "FP8 W8A8": ("compressed-tensor", "v7"),
    "FP8 W8A16": ("compressed-tensor", "v7"),
    "FP4 W4A16": ("mxfp4", "v7"),
    "NVFP4 W4A16": ("modelopt_fp4", "v7"),
}

ROADMAP_STATUSES = {"beta": "⚠️ Beta", "experimental": "🧪 Experimental", "planned": "📝 Planned", "unplanned": "⛔️ Unplanned"}
BASE_PASSES = {"✅ Passing", "⚪ N/A", "❓ Untested"}
MODEL_PASSES = BASE_PASSES | {"not enough HBM"}
FEATURE_PASSES = BASE_PASSES | set(ROADMAP_STATUSES.values())

KERNEL_SUBSTITUTIONS = {
    "generic ragged paged attention v3": "generic ragged paged<br>attention v3*",
    "generic_ragged_paged_attention_v3": "generic ragged paged<br>attention v3*",
    "mla": "mla*",
    "ragged paged attention v3 head_dim 64": "ragged paged attention v3<br>head_dim 64*",
    "ragged_paged_attention_v3_head_dim_64": "generic ragged paged<br>attention v3 (head_dim=64)*",
}


def version_sort_key(s: str) -> List:
    """Case-sensitive ASCII version sort matching GNU `sort -V`."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


# Backward-compatible helpers for external imports/tests
def get_tpu_generation(key: str) -> str:
    return QUANTIZATION_SPECS.get(key, ("", "N/A"))[1]


def get_quantization_method(key: str) -> str:
    return QUANTIZATION_SPECS.get(key, ("N/A", ""))[0]


def format_feature_status(raw: str) -> str:
    return ROADMAP_STATUSES.get(raw.strip().lower(), raw)


natural_sort_key = version_sort_key


# ==============================================================================
# Buildkite Client (Fast Key-Filtering & Multi-Threaded Prefetch)
# ==============================================================================
class BuildkiteClient:
    """Buildkite Agent wrapper with key discovery and concurrent prefetching."""

    def __init__(self, agent_cmd: str = "buildkite-agent", max_workers: int = 16):
        self.agent_cmd = agent_cmd
        self.max_workers = max_workers
        self._cache: Dict[str, str] = {}
        self._existing_keys: Optional[Set[str]] = None

    def get_existing_keys(self) -> Optional[Set[str]]:
        if self._existing_keys is None:
            try:
                res = subprocess.run([self.agent_cmd, "meta-data", "keys"], capture_output=True, text=True, check=False)
                if res.returncode == 0:
                    self._existing_keys = {k.strip() for k in res.stdout.splitlines() if k.strip()}
            except FileNotFoundError:
                pass
        return self._existing_keys

    def _fetch_cli(self, key: str, default: str = "") -> str:
        try:
            res = subprocess.run([self.agent_cmd, "meta-data", "get", key, "--default", default], capture_output=True, text=True, check=False)
            return res.stdout.strip()
        except FileNotFoundError:
            return default

    def prefetch_metadata(self, keys: Iterable[str]) -> None:
        """Discovers existing keys in one CLI call and fetches them in parallel into _cache."""
        existing = self.get_existing_keys()
        needed = [k for k in set(keys) if k not in self._cache and (existing is None or k in existing)]
        if not needed:
            return

        def _fetch(k: str) -> Tuple[str, str]:
            return k, self._fetch_cli(k, default="")

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(needed))) as ex:
            for k, val in ex.map(_fetch, needed):
                self._cache[k] = val

    def get_metadata(self, key: str, default: str = "") -> str:
        if key in self._cache:
            return self._cache[key] or default
        existing = self.get_existing_keys()
        if existing is not None and key not in existing:
            return default
        val = self._fetch_cli(key, default)
        self._cache[key] = val
        return val

    def set_metadata(self, key: str, val: str) -> None:
        self._cache[key] = val
        try:
            subprocess.run([self.agent_cmd, "meta-data", "set", key, val], check=False, capture_output=True)
        except FileNotFoundError:
            pass

    def upload_artifact(self, file_path: str) -> None:
        try:
            subprocess.run([self.agent_cmd, "artifact", "upload", file_path], check=False, capture_output=True)
        except FileNotFoundError:
            pass

    # Aliases
    get = get_metadata
    set = set_metadata
    fetch_all = prefetch_metadata


# ==============================================================================
# Helper Functions
# ==============================================================================
def upload_csv(csv_file: Path, title: str, bk: BuildkiteClient) -> None:
    """Prints CSV content to stdout and uploads as a Buildkite artifact."""
    if not csv_file.is_file():
        return
    print(f"--- Uploading {title}: {csv_file} ---")
    print(csv_file.read_text(encoding="utf-8"))
    bk.upload_artifact(str(csv_file))


def pivot_microbenchmarks(tpu_dir: Path, bk: BuildkiteClient) -> Optional[Path]:
    """Pivots microbenchmarks CSV into multi-quantization display matrix."""
    input_csv = tpu_dir / "kernel_support_matrix_microbenchmarks.csv"
    output_file = tpu_dir / "kernel_support_matrix-microbenchmarks.csv"
    if not input_csv.is_file():
        print(f"Warning: Input CSV {input_csv} not found. Skipping pivot.")
        return None

    header = (
        "Kernel,W16 A16 (Corr),W16 A16 (Perf),W8 A8 (Corr),W8 A8 (Perf),"
        "W8 A16 (Corr),W8 A16 (Perf),W4 A4 (Corr),W4 A4 (Perf),"
        "W4 A8 (Corr),W4 A8 (Perf),W4 A16 (Corr),W4 A16 (Perf)"
    )
    quant_cols = ["w16a16", "w8a8", "w8a16", "w4a4", "w4a8", "w4a16"]
    quant_pattern = re.compile(r"-(w\d+a\d+)$")

    kernel_order: List[str] = []
    matrix: Dict[Tuple[str, str], Tuple[str, str]] = {}

    with input_csv.open(encoding="utf-8") as f:
        reader = list(csv.reader(f))

    for row in reader[1:]:
        if not row or not row[0].strip():
            continue
        col1 = row[0].strip().replace('"', "")
        m = quant_pattern.search(col1)
        base_kernel, quant = (col1[: m.start()], m.group(1)) if m else (col1, "w16a16")
        matrix[(base_kernel, quant)] = (row[1] if len(row) > 1 else "❓ Untested", row[2] if len(row) > 2 else "❓ Untested")
        if base_kernel not in kernel_order:
            kernel_order.append(base_kernel)

    lines = [header]
    for k in kernel_order:
        row_items = [f'"{KERNEL_SUBSTITUTIONS.get(k, k)}"']
        for q in quant_cols:
            corr, perf = matrix.get((k, q), ("❓ Untested", "❓ Untested"))
            row_items.append(f"{corr},{perf}" if corr or perf else "❓ Untested,❓ Untested")
        lines.append(",".join(row_items))

    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    upload_csv(output_file, "Pivoted Kernel Matrix", bk)
    return output_file


def package_tar(tpu_dir: Path, bk: BuildkiteClient) -> Optional[Path]:
    """Packages all support matrix CSVs into {tpu_dir}_{impl}.tar.gz archive."""
    model_impl = os.environ.get("MODEL_IMPL_TYPE", "auto")
    impl_type = model_impl if model_impl in ("vllm", "flax_nnx") else "default"
    archive_path = Path(f"{tpu_dir.name}_{impl_type}.tar.gz")

    csv_files = list(tpu_dir.glob("*_support_matrix.csv"))
    micro_csv = tpu_dir / "kernel_support_matrix-microbenchmarks.csv"
    if micro_csv.is_file() and micro_csv not in csv_files:
        csv_files.append(micro_csv)

    if not csv_files:
        print(f"No CSV matrices to package for {archive_path.name}")
        return None

    print(f"--- Creating Tar Archive: {archive_path.name} ({tpu_dir.name}/{impl_type}/) ---")
    with tarfile.open(archive_path, "w:gz") as tar:
        for cf in csv_files:
            tar.add(cf, arcname=f"{tpu_dir.name}/{impl_type}/{cf.name}")

    print(f"--- Uploading Tar Artifact: {archive_path.name} ---")
    bk.upload_artifact(str(archive_path))
    archive_path.unlink(missing_ok=True)
    return archive_path


# Backward-compatible aliases
package_support_matrices_tar = package_tar
process_kernel_matrix_to_pivot = pivot_microbenchmarks
upload_matrix_csv = upload_csv


# ==============================================================================
# Main Linear Execution Pipeline
# ==============================================================================
def run_pipeline(
    bk: BuildkiteClient,
    features_file: Path = Path(".buildkite/features/default_features.txt"),
    cleanup: bool = True,
) -> bool:
    """Runs support matrix pipeline in 4 clear, linear steps."""
    any_failed = False
    is_v7 = os.environ.get("TPU_VERSION", "tpu6e").startswith("v7")
    tpu_dir = Path("v7x" if is_v7 else "v6e")
    tpu_prefix = "v7" if is_v7 else "v6"
    tpu_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory set to: {tpu_dir} (Prefix: '{tpu_prefix}')")

    # --------------------------------------------------------------------------
    # 1. READ INPUTS & DISCOVER DEFAULT FEATURES
    # --------------------------------------------------------------------------
    models = [m.strip() for m in bk.get_metadata("model-list").splitlines() if m.strip()]
    metadata_features = [f.strip() for f in bk.get_metadata("feature-list").splitlines() if f.strip()]

    default_features: Dict[str, str] = {}
    if features_file.is_file():
        print("--- Loading Feature Categories from file ---")
        regex = re.compile(r"^(.+)\s+\((.+)\)$")
        for line in features_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            m = regex.match(line)
            name, cat = (m.group(1).strip(), m.group(2).strip()) if m else (line, "feature support matrix")
            default_features[name] = cat
            print(f"Setting category for '{name}': {cat}")
            bk.set_metadata(f"{tpu_prefix}{name}_category", cat)

    # --------------------------------------------------------------------------
    # 2. PREFETCH ALL METADATA IN PARALLEL
    # --------------------------------------------------------------------------
    all_feature_stages = {s for _, stages in CATEGORY_CONFIG.values() for s in stages} | set(FEATURE_STAGES)
    all_feature_stages -= {"QuantizationMethods", "RecommendedTPUGenerations"}

    keys_to_fetch = [f"{tpu_prefix}{m}_category" for m in models]
    keys_to_fetch.extend(f"{tpu_prefix}{m}:{s}" for m in models for s in MODEL_STAGES[1:])
    keys_to_fetch.extend(f"{tpu_prefix}{f}_category" for f in metadata_features)
    keys_to_fetch.extend(f"{tpu_prefix}{f}:{s}" for f in metadata_features for s in all_feature_stages)
    bk.prefetch_metadata(keys_to_fetch)

    # --------------------------------------------------------------------------
    # 3. BUILD MODEL MATRIX
    # --------------------------------------------------------------------------
    if models:
        model_rows = []
        for m in models:
            cat = bk.get_metadata(f"{tpu_prefix}{m}_category", default="text-only")
            row = [f'"{m}"', MODEL_TYPES.get(cat, "Text")]
            for s in MODEL_STAGES[1:]:
                res = bk.get_metadata(f"{tpu_prefix}{m}:{s}", default="❓ Untested")
                row.append(res)
                if res not in MODEL_PASSES:
                    any_failed = True
            model_rows.append(row)
        model_rows.sort(key=lambda r: (r[1], r[0]))

        model_csv = tpu_dir / "model_support_matrix.csv"
        model_csv.write_text(",".join(["Model"] + MODEL_STAGES) + "\n" + "".join(",".join(r) + "\n" for r in model_rows), encoding="utf-8")
        upload_csv(model_csv, "Model Matrix", bk)

    # --------------------------------------------------------------------------
    # 4. BUILD FEATURE MATRICES (GROUPED BY CATEGORY)
    # --------------------------------------------------------------------------
    if default_features or metadata_features:
        tables: Dict[str, Tuple[str, List[List[str]]]] = {}

        def add_feature_row(feat: str, cat: str, is_default: bool):
            nonlocal any_failed
            header, stages = CATEGORY_CONFIG.get(cat, DEFAULT_CATEGORY_CONFIG)
            if cat not in tables:
                existing = []
                csv_path = tpu_dir / f"{cat.replace(' ', '_')}.csv"
                if csv_path.is_file():
                    with csv_path.open(encoding="utf-8") as f_ex:
                        reader = csv.reader(f_ex)
                        next(reader, None)
                        existing = [[f'"{r[0]}"' if not r[0].startswith('"') else r[0]] + r[1:] for r in reader if r]
                tables[cat] = (header, existing)

            row = [f'"{feat}"']
            for s in stages:
                if cat == "quantization support matrix" and s == "RecommendedTPUGenerations":
                    val = QUANTIZATION_SPECS.get(feat, ("", "N/A"))[1]
                elif cat == "quantization support matrix" and s == "QuantizationMethods":
                    val = QUANTIZATION_SPECS.get(feat, ("N/A", ""))[0]
                elif is_default:
                    val = "✅ Passing"
                else:
                    raw = bk.get_metadata(f"{tpu_prefix}{feat}:{s}", default="❓ Untested")
                    val = ROADMAP_STATUSES.get(raw.strip().lower(), raw)
                row.append(val)
                if s not in ("QuantizationMethods", "RecommendedTPUGenerations") and val not in FEATURE_PASSES:
                    any_failed = True
            tables[cat][1].append(row)

        for f, cat in default_features.items():
            if f:
                add_feature_row(f, cat, is_default=True)
        for f in metadata_features:
            cat = bk.get_metadata(f"{tpu_prefix}{f}_category", default="feature support matrix")
            if cat:
                add_feature_row(f, cat, is_default=False)

        for cat, (hdr, rows) in tables.items():
            rows.sort(key=lambda r: version_sort_key(r[0]))
            csv_path = tpu_dir / f"{cat.replace(' ', '_')}.csv"
            csv_path.write_text(hdr + "\n" + "".join(",".join(r) + "\n" for r in rows), encoding="utf-8")
            if not csv_path.name.endswith("kernel_support_matrix_microbenchmarks.csv"):
                upload_csv(csv_path, "Feature Matrix", bk)
            else:
                print(f"Skipping direct upload for {csv_path} (will be pivoted later).")

    # --------------------------------------------------------------------------
    # 5. PIVOT KERNEL MICROBENCHMARKS & PACKAGE TAR ARCHIVE
    # --------------------------------------------------------------------------
    pivot_microbenchmarks(tpu_dir, bk)
    package_tar(tpu_dir, bk)

    # --------------------------------------------------------------------------
    # 6. SET CI RESULT METADATA FLAG
    # --------------------------------------------------------------------------
    bk.set_metadata(f"{tpu_prefix}_CI_TESTS_FAILED", str(any_failed).lower())
    print("Reports uploaded successfully.")

    if cleanup and tpu_dir.is_dir():
        shutil.rmtree(tpu_dir, ignore_errors=True)

    return any_failed


def main():
    bk = BuildkiteClient()
    run_pipeline(bk, cleanup=True)


if __name__ == "__main__":
    main()
