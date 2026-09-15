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

import csv
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
from typing import Dict, List, Optional, Tuple

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

# Domain validation sets (kept separate for domain clarity)
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


def get_tpu_generation(key: str) -> str:
    return TPU_GENERATIONS.get(key, "N/A")


def get_quantization_method(key: str) -> str:
    return QUANT_METHODS.get(key, "N/A")


def format_feature_status(raw: str) -> str:
    return ROADMAP_STATUS_MAP.get(raw.strip().lower(), raw)


def version_sort_key(s: str) -> List:
    """Provides case-sensitive ASCII version sort matching GNU sort -V."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


# Alias for backward compatibility
natural_sort_key = version_sort_key


class BuildkiteClient:
    """Buildkite Agent CLI wrapper with support for mocking/testing."""

    def __init__(self, agent_cmd: str = "buildkite-agent"):
        self.agent_cmd = agent_cmd

    def get_metadata(self, key: str, default: str = "") -> str:
        try:
            res = subprocess.run(
                [self.agent_cmd, "meta-data", "get", key, "--default", default],
                capture_output=True, text=True, check=False
            )
            return res.stdout.strip()
        except FileNotFoundError:
            return default

    def set_metadata(self, key: str, value: str) -> None:
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
    rows.sort(key=lambda r: version_sort_key(r[0]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(header + "\n" + "".join(",".join(r) + "\n" for r in rows))


def parse_default_features(default_file: Path, bk: BuildkiteClient, tpu_prefix: str) -> List[str]:
    """Reads default features and sets category metadata."""
    if not default_file.is_file():
        print(f"Warning: Default features file not found at {default_file}")
        return []

    print("--- Loading Feature Categories from file ---")
    regex = re.compile(r"^(.+)\s+\((.+)\)$")
    feature_names: List[str] = []
    with open(default_file, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            m = regex.match(clean)
            name, cat = (m.group(1).strip(), m.group(2).strip()) if m else (clean, "feature support matrix")
            feature_names.append(name)
            print(f"Setting category for '{name}': {cat}")
            bk.set_metadata(f"{tpu_prefix}{name}_category", cat)
    return feature_names


def process_models(model_list: List[str], bk: BuildkiteClient, tpu_dir: Path, tpu_prefix: str) -> Tuple[List[Path], bool]:
    """Builds and writes model support matrix CSV."""
    rows: List[List[str]] = []
    any_failed = False

    for model in filter(None, model_list):
        category = bk.get_metadata(f"{tpu_prefix}{model}_category", default="text-only")
        row = [f'"{model}"', MODEL_TYPE_MAP.get(category, "Text")]
        for stage in MODEL_STAGES[1:]:
            res = bk.get_metadata(f"{tpu_prefix}{model}:{stage}", default="❓ Untested")
            row.append(res)
            if res not in MODEL_VALID_PASSES:
                any_failed = True
        rows.append(row)

    if not rows:
        return [], any_failed

    rows.sort(key=lambda r: (r[1], r[0]))
    csv_file = tpu_dir / "model_support_matrix.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        f.write(",".join(["Model"] + MODEL_STAGES) + "\n" + "".join(",".join(r) + "\n" for r in rows))
    return [csv_file], any_failed


def process_features(
    mode: str,
    feature_list: List[str],
    bk: BuildkiteClient,
    tpu_dir: Path,
    tpu_prefix: str,
    categorized_rows: Optional[Dict[Path, Tuple[str, List[List[str]]]]] = None,
    write_to_disk: bool = True,
) -> Tuple[Dict[Path, Tuple[str, List[List[str]]]], bool]:
    """Builds feature support matrices grouped by category."""
    if categorized_rows is None:
        categorized_rows = {}
    any_failed = False

    for feature in filter(None, feature_list):
        category = bk.get_metadata(f"{tpu_prefix}{feature}_category", default="feature support matrix")
        if not category:
            continue

        csv_file = tpu_dir / f"{category.replace(' ', '_')}.csv"
        header, stages = CATEGORY_CONFIG.get(category, DEFAULT_CATEGORY_CONFIG)
        is_quant = category == "quantization support matrix"

        if csv_file not in categorized_rows:
            existing = []
            if csv_file.is_file():
                with open(csv_file, "r", encoding="utf-8") as f_ex:
                    reader = csv.reader(f_ex)
                    next(reader, None)
                    existing = [[f'"{r[0]}"' if not r[0].startswith('"') else r[0]] + r[1:] for r in reader if r]
            categorized_rows[csv_file] = (header, existing)

        row = [f'"{feature}"']
        for stage in stages:
            if is_quant and stage == "RecommendedTPUGenerations":
                result = get_tpu_generation(feature)
            elif is_quant and stage == "QuantizationMethods":
                result = get_quantization_method(feature)
            elif mode == "DEFAULT":
                result = "✅ Passing"
            else:
                raw_res = bk.get_metadata(f"{tpu_prefix}{feature}:{stage}", default="❓ Untested")
                result = format_feature_status(raw_res)

            row.append(result)
            if stage not in ("QuantizationMethods", "RecommendedTPUGenerations") and result not in FEATURE_VALID_PASSES:
                any_failed = True

        categorized_rows[csv_file][1].append(row)

    if write_to_disk:
        for csv_file, (header, rows) in categorized_rows.items():
            write_csv(csv_file, header, rows)

    return categorized_rows, any_failed


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

    subs = {
        "generic ragged paged attention v3": "generic ragged paged<br>attention v3*",
        "generic_ragged_paged_attention_v3": "generic ragged paged<br>attention v3*",
        "mla": "mla*",
        "ragged paged attention v3 head_dim 64": "ragged paged attention v3<br>head_dim 64*",
        "ragged_paged_attention_v3_head_dim_64": "generic ragged paged<br>attention v3 (head_dim=64)*",
    }

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        f.write(header + "\n")
        for k in kernel_order:
            row_items = [f'"{subs.get(k, k)}"']
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
    default_features = parse_default_features(features_file, bk, tpu_prefix)

    model_csv_files, models_failed = process_models(models, bk, tpu_dir, tpu_prefix) if models else ([], False)
    if models_failed:
        any_failed = True

    all_feature_csvs: Dict[Path, Tuple[str, List[List[str]]]] = {}
    if default_features:
        _, def_failed = process_features("DEFAULT", default_features, bk, tpu_dir, tpu_prefix, all_feature_csvs, False)
        if def_failed:
            any_failed = True

    if metadata_features:
        _, meta_failed = process_features("METADATA", metadata_features, bk, tpu_dir, tpu_prefix, all_feature_csvs, False)
        if meta_failed:
            any_failed = True

    for csv_file, (header, rows) in all_feature_csvs.items():
        write_csv(csv_file, header, rows)

    bk.set_metadata(f"{tpu_prefix}_CI_TESTS_FAILED", str(any_failed).lower())

    for cf in model_csv_files:
        upload_matrix_csv(cf, "Model Matrix", bk)

    for cf in all_feature_csvs:
        if not cf.name.endswith("kernel_support_matrix_microbenchmarks.csv"):
            upload_matrix_csv(cf, "Feature Matrix", bk)
        else:
            print(f"Skipping direct upload for {cf} (will be pivoted later).")

    process_kernel_matrix_to_pivot(tpu_dir, bk)
    package_support_matrices_tar(tpu_dir, bk)
    print("Reports uploaded successfully.")

    if cleanup and tpu_dir.is_dir():
        shutil.rmtree(tpu_dir, ignore_errors=True)

    return any_failed


def main():
    bk = BuildkiteClient()
    run_pipeline(bk, cleanup=True)


if __name__ == "__main__":
    main()
