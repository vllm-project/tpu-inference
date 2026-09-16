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

# Configuration Constants
MODEL_LIST_KEY = "model-list"
FEATURE_LIST_KEY = "feature-list"
DEFAULT_FEATURES_FILE = Path(".buildkite/features/default_features.txt")

MODEL_STAGES = ["Type", "UnitTest", "Accuracy/Correctness", "Benchmark"]
FEATURE_STAGES = ["CorrectnessTest", "PerformanceTest"]
FEATURE_STAGES_QUANT = ["QuantizationMethods", "RecommendedTPUGenerations", "CorrectnessTest", "PerformanceTest"]
FEATURE_STAGES_MICRO = ["CorrectnessTest", "PerformanceTest"]
PARALLELISM_STAGES = [
    "Single-Host CorrectnessTest", "Single-Host PerformanceTest",
    "Multi-Host CorrectnessTest", "Multi-Host PerformanceTest"
]
QUANT_COLS = ["w16a16", "w8a8", "w8a16", "w4a4", "w4a8", "w4a16"]
QUANT_COLS_LIST = QUANT_COLS

MODEL_VALID_PASSES = {"✅ Passing", "⚪ N/A", "❓ Untested", "not enough HBM"}
FEATURE_VALID_PASSES = {
    "✅ Passing", "⚪ N/A", "❓ Untested",
    "⚠️ Beta", "🧪 Experimental", "📝 Planned", "⛔️ Unplanned"
}

MODEL_TYPE_MAP = {"multimodal": "Multimodal", "embedding": "Embedding", "diffusion": "Diffusion"}
ROADMAP_STATUS_MAP = {"beta": "⚠️ Beta", "experimental": "🧪 Experimental", "planned": "📝 Planned", "unplanned": "⛔️ Unplanned"}

TPU_GENERATIONS = {
    "INT8 W8A8": '"v5, v6"', "INT4 W4A16": '"v5, v6"',
    "FP8 W8A8": "v7", "FP8 W8A16": "v7", "FP4 W4A16": "v7", "NVFP4 W4A16": "v7"
}
QUANT_METHODS = {
    "INT8 W8A8": "compressed-tensor", "INT4 W4A16": "awq",
    "FP8 W8A8": "compressed-tensor", "FP8 W8A16": "compressed-tensor",
    "FP4 W4A16": "mxfp4", "NVFP4 W4A16": "modelopt_fp4"
}

CATEGORY_CONFIG: Dict[str, Tuple[str, List[str]]] = {
    "quantization support matrix": (
        "Quantization dtype,Quantization methods,Recommended TPU Generations,CorrectnessTest,PerformanceTest",
        FEATURE_STAGES_QUANT,
    ),
    "kernel support matrix microbenchmarks": (
        "kernels,CorrectnessTest,PerformanceTest",
        FEATURE_STAGES_MICRO,
    ),
    "parallelism support matrix": (
        "Feature,Single-Host CorrectnessTest,Single-Host PerformanceTest,Multi-Host CorrectnessTest,Multi-Host PerformanceTest",
        PARALLELISM_STAGES,
    ),
}
DEFAULT_CATEGORY_CONFIG = ("Feature,CorrectnessTest,PerformanceTest", FEATURE_STAGES)

KERNEL_NAME_SUBSTITUTIONS = {
    "generic ragged paged attention v3": "generic ragged paged<br>attention v3*",
    "generic_ragged_paged_attention_v3": "generic ragged paged<br>attention v3*",
    "mla": "mla*",
    "ragged paged attention v3 head_dim 64": "ragged paged attention v3<br>head_dim 64*",
    "ragged_paged_attention_v3_head_dim_64": "generic ragged paged<br>attention v3 (head_dim=64)*",
}


def get_tpu_generation(key: str) -> str:
    return TPU_GENERATIONS.get(key, "N/A")


def get_quantization_method(key: str) -> str:
    return QUANT_METHODS.get(key, "N/A")


def format_feature_status(raw: str) -> str:
    return ROADMAP_STATUS_MAP.get(raw.strip().lower(), raw)


def version_sort_key(s: str) -> List:
    """Provides case-sensitive ASCII version sort matching GNU sort -V."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


natural_sort_key = version_sort_key


class BuildkiteClient:
    """Buildkite Agent CLI wrapper with in-memory caching and parallel pre-fetching."""

    def __init__(self, agent_cmd: str = "buildkite-agent", max_workers: int = 16):
        self.agent_cmd = agent_cmd
        self.max_workers = max_workers
        self._cache: Dict[str, str] = {}
        self._existing_keys: Optional[Set[str]] = None

    def get_existing_keys(self) -> Optional[Set[str]]:
        """Discovers all existing metadata keys in a single CLI call."""
        if self._existing_keys is None:
            try:
                res = subprocess.run([self.agent_cmd, "meta-data", "keys"], capture_output=True, text=True, check=False)
                if res.returncode == 0:
                    self._existing_keys = {line.strip() for line in res.stdout.splitlines() if line.strip()}
            except FileNotFoundError:
                pass
        return self._existing_keys

    def prefetch_metadata(self, keys: Iterable[str]) -> None:
        """Prefetches metadata keys concurrently, skipping keys known not to exist."""
        existing = self.get_existing_keys()
        needed = [k for k in set(keys) if k not in self._cache and (existing is None or k in existing)]
        if not needed:
            return

        def _fetch(key: str) -> Tuple[str, str]:
            return key, self._fetch_cli(key, default="")

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(needed))) as executor:
            for key, val in executor.map(_fetch, needed):
                self._cache[key] = val

    def _fetch_cli(self, key: str, default: str = "") -> str:
        try:
            res = subprocess.run([self.agent_cmd, "meta-data", "get", key, "--default", default], capture_output=True, text=True, check=False)
            return res.stdout.strip()
        except FileNotFoundError:
            return default

    def get_metadata(self, key: str, default: str = "") -> str:
        if key in self._cache:
            return self._cache[key] or default
        existing = self.get_existing_keys()
        if existing is not None and key not in existing:
            return default
        val = self._fetch_cli(key, default)
        self._cache[key] = val
        return val

    def set_metadata(self, key: str, value: str) -> None:
        self._cache[key] = value
        try:
            subprocess.run([self.agent_cmd, "meta-data", "set", key, value], check=False, capture_output=True)
        except FileNotFoundError:
            pass

    def upload_artifact(self, file_path: str) -> None:
        try:
            subprocess.run([self.agent_cmd, "artifact", "upload", file_path], check=False, capture_output=True)
        except FileNotFoundError:
            pass


def upload_matrix_csv(csv_file: Path, title: str, bk: BuildkiteClient) -> None:
    if not csv_file.is_file():
        return
    print(f"--- Uploading {title}: {csv_file} ---")
    with open(csv_file, "r", encoding="utf-8") as f:
        print(f.read())
    bk.upload_artifact(str(csv_file))


def write_csv(path: Path, header: str, rows: List[List[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(header + "\n" + "".join(",".join(r) + "\n" for r in rows))


def load_default_features(default_file: Path, bk: BuildkiteClient, tpu_prefix: str) -> Dict[str, str]:
    """Reads default features from file, caches categories, and sets Buildkite metadata for external tools."""
    if not default_file.is_file():
        print(f"Warning: Default features file not found at {default_file}")
        return {}

    print("--- Loading Feature Categories from file ---")
    regex = re.compile(r"^(.+)\s+\((.+)\)$")
    feature_categories: Dict[str, str] = {}
    with open(default_file, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            m = regex.match(clean)
            name, cat = (m.group(1).strip(), m.group(2).strip()) if m else (clean, "feature support matrix")
            feature_categories[name] = cat
            print(f"Setting category for '{name}': {cat}")
            bk.set_metadata(f"{tpu_prefix}{name}_category", cat)
    return feature_categories


def collect_metadata_keys(models: List[str], metadata_features: List[str], tpu_prefix: str) -> List[str]:
    """Collects all metadata keys needed by models and features for parallel pre-fetching."""
    stages = (
        "CorrectnessTest", "PerformanceTest",
        "Single-Host CorrectnessTest", "Single-Host PerformanceTest",
        "Multi-Host CorrectnessTest", "Multi-Host PerformanceTest",
    )
    keys = [f"{tpu_prefix}{m}_category" for m in models] + [f"{tpu_prefix}{m}:{s}" for m in models for s in MODEL_STAGES[1:]]
    keys += [f"{tpu_prefix}{f}_category" for f in metadata_features] + [f"{tpu_prefix}{f}:{s}" for f in metadata_features for s in stages]
    return keys


def build_model_matrix(models: List[str], bk: BuildkiteClient, tpu_prefix: str) -> Tuple[List[List[str]], bool]:
    """Builds model support matrix rows and determines if any test failed."""
    rows: List[List[str]] = []
    any_failed = False

    for model in filter(None, models):
        category = bk.get_metadata(f"{tpu_prefix}{model}_category", default="text-only")
        row = [f'"{model}"', MODEL_TYPE_MAP.get(category, "Text")]
        for stage in MODEL_STAGES[1:]:
            res = bk.get_metadata(f"{tpu_prefix}{model}:{stage}", default="❓ Untested")
            row.append(res)
            if res not in MODEL_VALID_PASSES:
                any_failed = True
        rows.append(row)

    rows.sort(key=lambda r: (r[1], r[0]))
    return rows, any_failed


def resolve_feature_cell(
    feature: str, stage: str, is_quant: bool, is_default: bool, bk: BuildkiteClient, tpu_prefix: str
) -> str:
    """Resolves a single cell status for a feature stage."""
    if is_quant and stage == "RecommendedTPUGenerations":
        return get_tpu_generation(feature)
    if is_quant and stage == "QuantizationMethods":
        return get_quantization_method(feature)
    if is_default:
        return "✅ Passing"
    raw_res = bk.get_metadata(f"{tpu_prefix}{feature}:{stage}", default="❓ Untested")
    return format_feature_status(raw_res)


def build_feature_matrices(
    default_features: Dict[str, str],
    metadata_features: List[str],
    bk: BuildkiteClient,
    tpu_prefix: str,
    tpu_dir: Path,
) -> Tuple[Dict[str, Tuple[str, List[List[str]]]], bool]:
    """Builds feature support matrices grouped by category."""
    categorized: Dict[str, Tuple[str, List[List[str]]]] = {}
    any_failed = False

    def _add_feature(feature: str, category: str, is_default: bool):
        nonlocal any_failed
        header, stages = CATEGORY_CONFIG.get(category, DEFAULT_CATEGORY_CONFIG)
        is_quant = category == "quantization support matrix"

        if category not in categorized:
            existing = []
            csv_file = tpu_dir / f"{category.replace(' ', '_')}.csv"
            if csv_file.is_file():
                with open(csv_file, "r", encoding="utf-8") as f_ex:
                    reader = csv.reader(f_ex)
                    next(reader, None)
                    existing = [[f'"{r[0]}"' if not r[0].startswith('"') else r[0]] + r[1:] for r in reader if r]
            categorized[category] = (header, existing)

        row = [f'"{feature}"']
        for stage in stages:
            result = resolve_feature_cell(feature, stage, is_quant, is_default, bk, tpu_prefix)
            row.append(result)
            if stage not in ("QuantizationMethods", "RecommendedTPUGenerations") and result not in FEATURE_VALID_PASSES:
                any_failed = True
        categorized[category][1].append(row)

    for feature, category in default_features.items():
        if feature:
            _add_feature(feature, category, is_default=True)

    for feature in filter(None, metadata_features):
        category = bk.get_metadata(f"{tpu_prefix}{feature}_category", default="feature support matrix")
        if category:
            _add_feature(feature, category, is_default=False)

    for cat, (header, rows) in categorized.items():
        rows.sort(key=lambda r: version_sort_key(r[0]))

    return categorized, any_failed


def process_kernel_matrix_to_pivot(tpu_dir: Path, bk: BuildkiteClient) -> Optional[Path]:
    """Pivots microbenchmarks CSV into a multi-quantization display matrix."""
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
    kernel_order: List[str] = []
    matrix: Dict[Tuple[str, str], Tuple[str, str]] = {}
    quant_pattern = re.compile(r"-(w\d+a\d+)$")

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row or not row[0].strip():
                continue
            col1 = row[0].strip().replace('"', "")
            m = quant_pattern.search(col1)
            base_kernel, quant_type = (col1[: m.start()], m.group(1)) if m else (col1, "w16a16")
            matrix[(base_kernel, quant_type)] = (
                row[1] if len(row) > 1 else "❓ Untested",
                row[2] if len(row) > 2 else "❓ Untested",
            )
            if base_kernel not in kernel_order:
                kernel_order.append(base_kernel)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        f.write(header + "\n")
        for k in kernel_order:
            row_items = [f'"{KERNEL_NAME_SUBSTITUTIONS.get(k, k)}"']
            for q in QUANT_COLS:
                corr, perf = matrix.get((k, q), ("❓ Untested", "❓ Untested"))
                row_items.append(f"{corr},{perf}" if corr or perf else "❓ Untested,❓ Untested")
            f.write(",".join(row_items) + "\n")

    upload_matrix_csv(output_file, "Pivoted Kernel Matrix", bk)
    return output_file


def package_support_matrices_tar(tpu_dir: Path, bk: BuildkiteClient) -> Optional[Path]:
    """Packages all support matrix CSVs into a tar.gz archive."""
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


def run_pipeline(bk: BuildkiteClient, features_file: Path = DEFAULT_FEATURES_FILE, cleanup: bool = True) -> bool:
    """Main execution pipeline."""
    any_failed = False
    is_v7 = os.environ.get("TPU_VERSION", "tpu6e").startswith("v7")
    tpu_dir = Path("v7x" if is_v7 else "v6e")
    tpu_prefix = "v7" if is_v7 else "v6"
    tpu_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory set to: {tpu_dir} (Prefix: '{tpu_prefix}')")

    models = [m.strip() for m in bk.get_metadata(MODEL_LIST_KEY, default="").splitlines() if m.strip()]
    metadata_features = [f.strip() for f in bk.get_metadata(FEATURE_LIST_KEY, default="").splitlines() if f.strip()]
    default_features = load_default_features(features_file, bk, tpu_prefix)

    # Pre-fetch all metadata keys in parallel across worker threads
    keys_to_prefetch = collect_metadata_keys(models, metadata_features, tpu_prefix)
    bk.prefetch_metadata(keys_to_prefetch)

    # 1. Models
    if models:
        model_rows, models_failed = build_model_matrix(models, bk, tpu_prefix)
        if models_failed:
            any_failed = True
        model_csv = tpu_dir / "model_support_matrix.csv"
        write_csv(model_csv, ",".join(["Model"] + MODEL_STAGES), model_rows)
        upload_matrix_csv(model_csv, "Model Matrix", bk)

    # 2. Features
    if default_features or metadata_features:
        categorized, features_failed = build_feature_matrices(default_features, metadata_features, bk, tpu_prefix, tpu_dir)
        if features_failed:
            any_failed = True

        for category, (header, rows) in categorized.items():
            csv_file = tpu_dir / f"{category.replace(' ', '_')}.csv"
            write_csv(csv_file, header, rows)
            if not csv_file.name.endswith("kernel_support_matrix_microbenchmarks.csv"):
                upload_matrix_csv(csv_file, "Feature Matrix", bk)
            else:
                print(f"Skipping direct upload for {csv_file} (will be pivoted later).")

    # 3. Kernel Microbenchmarks Pivot & Tar Packaging
    process_kernel_matrix_to_pivot(tpu_dir, bk)
    package_support_matrices_tar(tpu_dir, bk)

    # 4. Status Notification Key
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
