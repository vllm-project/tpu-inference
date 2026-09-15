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
FEATURE_STAGES_QUANTIZATION = [
    "QuantizationMethods",
    "RecommendedTPUGenerations",
    "CorrectnessTest",
    "PerformanceTest",
]
FEATURE_STAGES_MICROBENCHMARKS = ["CorrectnessTest", "PerformanceTest"]
PARALLELISM_STAGES = [
    "Single-Host CorrectnessTest",
    "Single-Host PerformanceTest",
    "Multi-Host CorrectnessTest",
    "Multi-Host PerformanceTest",
]

QUANT_COLS_LIST = ["w16a16", "w8a8", "w8a16", "w4a4", "w4a8", "w4a16"]

# Validation Sets (kept separate for domain clarity)
MODEL_VALID_PASSES = {"✅ Passing", "⚪ N/A", "❓ Untested", "not enough HBM"}
FEATURE_VALID_PASSES = {
    "✅ Passing",
    "⚪ N/A",
    "❓ Untested",
    "⚠️ Beta",
    "🧪 Experimental",
    "📝 Planned",
    "⛔️ Unplanned",
}

MODEL_TYPE_MAPPING = {
    "multimodal": "Multimodal",
    "embedding": "Embedding",
    "diffusion": "Diffusion",
}

# Category Configurations: (Header, Stages)
CATEGORY_CONFIG: Dict[str, Tuple[str, List[str]]] = {
    "quantization support matrix": (
        "Quantization dtype,Quantization methods,Recommended TPU Generations,CorrectnessTest,PerformanceTest",
        FEATURE_STAGES_QUANTIZATION,
    ),
    "kernel support matrix microbenchmarks": (
        "kernels,CorrectnessTest,PerformanceTest",
        FEATURE_STAGES_MICROBENCHMARKS,
    ),
    "parallelism support matrix": (
        "Feature,Single-Host CorrectnessTest,Single-Host PerformanceTest,Multi-Host CorrectnessTest,Multi-Host PerformanceTest",
        PARALLELISM_STAGES,
    ),
}
DEFAULT_CATEGORY_CONFIG = (
    "Feature,CorrectnessTest,PerformanceTest",
    FEATURE_STAGES,
)


def get_tpu_generation(key: str) -> str:
    """Maps quantization dtype to recommended TPU generations."""
    mapping = {
        "INT8 W8A8": '"v5, v6"',
        "INT4 W4A16": '"v5, v6"',
        "FP8 W8A8": "v7",
        "FP8 W8A16": "v7",
        "FP4 W4A16": "v7",
        "NVFP4 W4A16": "v7",
    }
    return mapping.get(key, "N/A")


def get_quantization_method(key: str) -> str:
    """Maps quantization dtype to quantization method."""
    mapping = {
        "INT8 W8A8": "compressed-tensor",
        "INT4 W4A16": "awq",
        "FP8 W8A8": "compressed-tensor",
        "FP8 W8A16": "compressed-tensor",
        "FP4 W4A16": "mxfp4",
        "NVFP4 W4A16": "modelopt_fp4",
    }
    return mapping.get(key, "N/A")


def format_feature_status(raw_status: str) -> str:
    """Formats custom roadmap status strings from feature configs to emoji labels."""
    mapping = {
        "beta": "⚠️ Beta",
        "experimental": "🧪 Experimental",
        "planned": "📝 Planned",
        "unplanned": "⛔️ Unplanned",
    }
    return mapping.get(raw_status.strip().lower(), raw_status)


def version_sort_key(s: str) -> List:
    """Provides version sort key matching GNU sort -V (case-sensitive ASCII order)."""
    return [
        int(text) if text.isdigit() else text
        for text in re.split(r"(\d+)", s)
    ]


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
                capture_output=True,
                text=True,
                check=False,
            )
            return res.stdout.strip()
        except FileNotFoundError:
            return default

    def set_metadata(self, key: str, value: str) -> None:
        try:
            subprocess.run(
                [self.agent_cmd, "meta-data", "set", key, value],
                check=False,
                capture_output=True,
            )
        except FileNotFoundError:
            pass

    def upload_artifact(self, file_path: str) -> None:
        try:
            subprocess.run(
                [self.agent_cmd, "artifact", "upload", file_path],
                check=False,
                capture_output=True,
            )
        except FileNotFoundError:
            pass


def upload_matrix_csv(csv_file: Path, title: str, bk: BuildkiteClient) -> None:
    """Prints and uploads a CSV matrix artifact."""
    if not csv_file.is_file():
        return
    print(f"--- Uploading {title}: {csv_file} ---")
    with open(csv_file, "r", encoding="utf-8") as f:
        print(f.read())
    bk.upload_artifact(str(csv_file))


def parse_default_features(
    default_features_file: Path, bk: BuildkiteClient, tpu_prefix: str
) -> List[str]:
    """Reads default features and sets category metadata."""
    default_feature_names: List[str] = []
    if not default_features_file.is_file():
        print(
            f"Warning: Default features file not found at {default_features_file}"
        )
        return default_feature_names

    print("--- Loading Feature Categories from file ---")
    regex = re.compile(r"^(.+)\s+\((.+)\)$")
    with open(default_features_file, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = line.strip()
            if not clean_line:
                continue
            m = regex.match(clean_line)
            if m:
                feature_name, category = m.group(1).strip(), m.group(2).strip()
                default_feature_names.append(feature_name)
                print(f"Setting category for '{feature_name}': {category}")
                bk.set_metadata(f"{tpu_prefix}{feature_name}_category", category)
            else:
                default_feature_names.append(clean_line)
                print(
                    f"Warning: No category found for '{clean_line}', defaulting to 'feature support matrix'"
                )
    return default_feature_names


def process_models(
    model_list: List[str],
    bk: BuildkiteClient,
    tpu_dir: Path,
    tpu_prefix: str,
) -> Tuple[List[Path], bool]:
    """Builds and writes model support matrix CSV."""
    model_rows: List[List[str]] = []
    any_failed = False

    for model in model_list:
        if not model:
            continue
        category = bk.get_metadata(
            f"{tpu_prefix}{model}_category", default="text-only"
        )
        type_val = MODEL_TYPE_MAPPING.get(category, "Text")

        row = [f'"{model}"', type_val]
        for stage in MODEL_STAGES[1:]:
            res = bk.get_metadata(
                f"{tpu_prefix}{model}:{stage}", default="❓ Untested"
            )
            row.append(res)
            if res not in MODEL_VALID_PASSES:
                any_failed = True

        model_rows.append(row)

    if not model_rows:
        return [], any_failed

    # Sort data rows based on the 'Type' column (matching sort -t',' -k2,2)
    model_rows.sort(key=lambda r: (r[1], r[0]))

    csv_file = tpu_dir / "model_support_matrix.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(["Model"] + MODEL_STAGES)
        for r in model_rows:
            f.write(",".join(r) + "\n")

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

    for feature in feature_list:
        if not feature:
            continue

        # In upstream bash, category lookup always defaults to "feature support matrix"
        category = bk.get_metadata(
            f"{tpu_prefix}{feature}_category", default="feature support matrix"
        )
        if not category:
            continue

        category_filename = category.replace(" ", "_")
        csv_file = tpu_dir / f"{category_filename}.csv"

        header, stages_to_use = CATEGORY_CONFIG.get(category, DEFAULT_CATEGORY_CONFIG)
        is_quant = category == "quantization support matrix"

        if csv_file not in categorized_rows:
            existing_rows: List[List[str]] = []
            if csv_file.is_file():
                with open(csv_file, "r", encoding="utf-8") as f_ex:
                    reader = csv.reader(f_ex)
                    try:
                        next(reader)  # skip header
                        for r in reader:
                            if r:
                                col0 = f'"{r[0]}"' if not r[0].startswith('"') else r[0]
                                existing_rows.append([col0] + r[1:])
                    except StopIteration:
                        pass
            categorized_rows[csv_file] = (header, existing_rows)

        row = [f'"{feature}"']
        for stage in stages_to_use:
            if is_quant and stage == "RecommendedTPUGenerations":
                result = get_tpu_generation(feature)
            elif is_quant and stage == "QuantizationMethods":
                result = get_quantization_method(feature)
            elif mode == "DEFAULT":
                result = "✅ Passing"
            else:
                raw_res = bk.get_metadata(
                    f"{tpu_prefix}{feature}:{stage}", default="❓ Untested"
                )
                result = format_feature_status(raw_res)

            row.append(result)

            if (
                stage not in ("QuantizationMethods", "RecommendedTPUGenerations")
                and result not in FEATURE_VALID_PASSES
            ):
                any_failed = True

        categorized_rows[csv_file][1].append(row)

    if write_to_disk:
        for csv_file, (header, rows) in categorized_rows.items():
            rows.sort(key=lambda r: version_sort_key(r[0]))
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                f.write(header + "\n")
                for r in rows:
                    f.write(",".join(r) + "\n")

    return categorized_rows, any_failed


def process_kernel_matrix_to_pivot(
    tpu_dir: Path, bk: BuildkiteClient
) -> Optional[Path]:
    """Replicates the AWK microbenchmarks pivot logic in pure Python."""
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

    quant_cols = QUANT_COLS_LIST
    kernel_order: List[str] = []
    seen_kernels = set()
    matrix: Dict[Tuple[str, str], Tuple[str, str]] = {}

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            next(reader)  # Skip header
        except StopIteration:
            pass

        quant_pattern = re.compile(r"-(w\d+a\d+)$")
        for row in reader:
            if not row or not row[0].strip():
                continue
            col1 = row[0].strip().replace('"', "")
            m = quant_pattern.search(col1)
            if m:
                quant_type = m.group(1)
                base_kernel = col1[: m.start()]
            else:
                quant_type = "w16a16"
                base_kernel = col1

            corr = row[1] if len(row) > 1 else "❓ Untested"
            perf = row[2] if len(row) > 2 else "❓ Untested"
            matrix[(base_kernel, quant_type)] = (corr, perf)

            if base_kernel not in seen_kernels:
                seen_kernels.add(base_kernel)
                kernel_order.append(base_kernel)

    # Name substitutions matching upstream
    substitutions = {
        "generic ragged paged attention v3": "generic ragged paged<br>attention v3*",
        "generic_ragged_paged_attention_v3": "generic ragged paged<br>attention v3*",
        "mla": "mla*",
        "ragged paged attention v3 head_dim 64": "ragged paged attention v3<br>head_dim 64*",
        "ragged_paged_attention_v3_head_dim_64": "generic ragged paged<br>attention v3 (head_dim=64)*",
    }

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        f.write(header + "\n")
        for k_orig in kernel_order:
            disp_name = substitutions.get(k_orig, k_orig)
            row_items = [f'"{disp_name}"']
            for q in quant_cols:
                if (k_orig, q) in matrix:
                    corr, perf = matrix[(k_orig, q)]
                    cell = f"{corr},{perf}" if corr or perf else "❓ Untested,❓ Untested"
                else:
                    cell = "❓ Untested,❓ Untested"
                row_items.append(cell)
            f.write(",".join(row_items) + "\n")

    upload_matrix_csv(output_file, "Pivoted Kernel Matrix", bk)
    return output_file


def package_support_matrices_tar(
    tpu_dir: Path, bk: BuildkiteClient
) -> Optional[Path]:
    """Packages all support matrix CSVs into a tar.gz matching upstream package_support_matrices_tar."""
    model_impl = os.environ.get("MODEL_IMPL_TYPE", "auto")
    impl_type = model_impl if model_impl in ("vllm", "flax_nnx") else "default"

    archive_name = f"{tpu_dir.name}_{impl_type}.tar.gz"
    archive_path = Path(archive_name)

    # Collect support matrix CSV files
    csv_files = list(tpu_dir.glob("*_support_matrix.csv"))
    microbench_csv = tpu_dir / "kernel_support_matrix-microbenchmarks.csv"
    if microbench_csv.is_file() and microbench_csv not in csv_files:
        csv_files.append(microbench_csv)

    if not csv_files:
        print(f"No CSV matrices to package for {archive_name}")
        return None

    print(
        f"--- Creating Tar Archive: {archive_name} ({tpu_dir.name}/{impl_type}/) ---"
    )
    with tarfile.open(archive_path, "w:gz") as tar:
        for csv_file in csv_files:
            arcname = f"{tpu_dir.name}/{impl_type}/{csv_file.name}"
            tar.add(csv_file, arcname=arcname)

    print(f"--- Uploading Tar Artifact: {archive_name} ---")
    bk.upload_artifact(str(archive_path))
    if archive_path.is_file():
        archive_path.unlink()

    return archive_path


def run_pipeline(
    bk: BuildkiteClient,
    features_file: Path = DEFAULT_FEATURES_FILE,
    cleanup: bool = True,
) -> bool:
    """Main execution pipeline."""
    any_failed = False

    tpu_version = os.environ.get("TPU_VERSION", "tpu6e")
    if tpu_version.startswith("v7"):
        tpu_dir = Path("v7x")
        tpu_prefix = "v7"
    else:
        tpu_dir = Path("v6e")
        tpu_prefix = "v6"

    tpu_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory set to: {tpu_dir} (Prefix: '{tpu_prefix}')")

    model_list_str = bk.get_metadata(MODEL_LIST_KEY, default="")
    model_list = [m.strip() for m in model_list_str.splitlines() if m.strip()]

    feature_list_str = bk.get_metadata(FEATURE_LIST_KEY, default="")
    metadata_feature_list = [
        f.strip() for f in feature_list_str.splitlines() if f.strip()
    ]

    default_feature_names = parse_default_features(
        features_file, bk, tpu_prefix
    )

    # Process Models
    model_csv_files: List[Path] = []
    if model_list:
        model_csv_files, models_failed = process_models(
            model_list, bk, tpu_dir, tpu_prefix
        )
        if models_failed:
            any_failed = True

    # Process Features
    all_feature_csvs: Dict[Path, Tuple[str, List[List[str]]]] = {}
    if default_feature_names:
        _, default_failed = process_features(
            "DEFAULT",
            default_feature_names,
            bk,
            tpu_dir,
            tpu_prefix,
            categorized_rows=all_feature_csvs,
            write_to_disk=False,
        )
        if default_failed:
            any_failed = True

    if metadata_feature_list:
        _, meta_failed = process_features(
            "METADATA",
            metadata_feature_list,
            bk,
            tpu_dir,
            tpu_prefix,
            categorized_rows=all_feature_csvs,
            write_to_disk=False,
        )
        if meta_failed:
            any_failed = True

    # Sort each category file rows with version_sort_key (matching sort -V) and write to disk
    for csv_file, (header, rows) in all_feature_csvs.items():
        rows.sort(key=lambda r: version_sort_key(r[0]))
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            f.write(header + "\n")
            for r in rows:
                f.write(",".join(r) + "\n")

    # Set overall test failure flag in metadata
    bk.set_metadata(f"{tpu_prefix}_CI_TESTS_FAILED", str(any_failed).lower())

    # Upload Model Matrices
    for csv_file in model_csv_files:
        upload_matrix_csv(csv_file, "Model Matrix", bk)

    # Upload Feature Matrices (skipping raw microbenchmarks)
    for csv_file in all_feature_csvs.keys():
        if not csv_file.name.endswith("kernel_support_matrix_microbenchmarks.csv"):
            upload_matrix_csv(csv_file, "Feature Matrix", bk)
        else:
            print(
                f"Skipping direct upload for {csv_file} (will be pivoted later)."
            )

    # Pivot Microbenchmark Matrix & Upload
    process_kernel_matrix_to_pivot(tpu_dir, bk)

    # Package all CSV matrices into a tar archive
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
