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

import csv
import datetime
import os
import re

# --- CONFIGURATION ---
# This dictionary maps the "markers" in your README to your CSV files.
CSV_MAP = {
    "release": {
        "model_support": {
            "v6_pytorch":
            "support_matrices/release/v6e/vllm/model_support_matrix.csv",
            "v7_pytorch":
            "support_matrices/release/v7x/vllm/model_support_matrix.csv"
        },
        "core_features": {
            "v6_flax":
            "support_matrices/release/v6e/flax_nnx/feature_support_matrix.csv",
            "v6_pytorch":
            "support_matrices/release/v6e/vllm/feature_support_matrix.csv",
            "v6_default":
            "support_matrices/release/v6e/default/feature_support_matrix.csv",
            "v7_flax":
            "support_matrices/release/v7x/flax_nnx/feature_support_matrix.csv",
            "v7_pytorch":
            "support_matrices/release/v7x/vllm/feature_support_matrix.csv",
            "v7_default":
            "support_matrices/release/v7x/default/feature_support_matrix.csv"
        },
        "parallelism": {
            "v6_flax":
            "support_matrices/release/v6e/flax_nnx/parallelism_support_matrix.csv",
            "v6_pytorch":
            "support_matrices/release/v6e/vllm/parallelism_support_matrix.csv",
            "v7_flax":
            "support_matrices/release/v7x/flax_nnx/parallelism_support_matrix.csv",
            "v7_pytorch":
            "support_matrices/release/v7x/vllm/parallelism_support_matrix.csv"
        },
        "quantization": {
            "static":
            "support_matrices/release/v6e/vllm/quantization_support_matrix.csv",
            "v6_flax":
            "support_matrices/release/v6e/flax_nnx/quantization_support_matrix.csv",
            "v6_pytorch":
            "support_matrices/release/v6e/vllm/quantization_support_matrix.csv",
            "v7_flax":
            "support_matrices/release/v7x/flax_nnx/quantization_support_matrix.csv",
            "v7_pytorch":
            "support_matrices/release/v7x/vllm/quantization_support_matrix.csv"
        },
        "microbenchmarks": {
            "v6":
            "support_matrices/release/v6e/default/kernel_support_matrix-microbenchmarks.csv",
            "v7":
            "support_matrices/release/v7x/default/kernel_support_matrix-microbenchmarks.csv"
        }
    },
    "nightly": {
        "model_support": {
            "v6_pytorch":
            "support_matrices/nightly/v6e/vllm/model_support_matrix.csv",
            "v7_pytorch":
            "support_matrices/nightly/v7x/vllm/model_support_matrix.csv"
        },
        "core_features": {
            "v6_flax":
            "support_matrices/nightly/v6e/flax_nnx/feature_support_matrix.csv",
            "v6_pytorch":
            "support_matrices/nightly/v6e/vllm/feature_support_matrix.csv",
            "v6_default":
            "support_matrices/nightly/v6e/default/feature_support_matrix.csv",
            "v7_flax":
            "support_matrices/nightly/v7x/flax_nnx/feature_support_matrix.csv",
            "v7_pytorch":
            "support_matrices/nightly/v7x/vllm/feature_support_matrix.csv",
            "v7_default":
            "support_matrices/nightly/v7x/default/feature_support_matrix.csv"
        },
        "parallelism": {
            "v6_flax":
            "support_matrices/nightly/v6e/flax_nnx/parallelism_support_matrix.csv",
            "v6_pytorch":
            "support_matrices/nightly/v6e/vllm/parallelism_support_matrix.csv",
            "v7_flax":
            "support_matrices/nightly/v7x/flax_nnx/parallelism_support_matrix.csv",
            "v7_pytorch":
            "support_matrices/nightly/v7x/vllm/parallelism_support_matrix.csv"
        },
        "quantization": {
            "static":
            "support_matrices/nightly/v6e/vllm/quantization_support_matrix.csv",
            "v6_flax":
            "support_matrices/nightly/v6e/flax_nnx/quantization_support_matrix.csv",
            "v6_pytorch":
            "support_matrices/nightly/v6e/vllm/quantization_support_matrix.csv",
            "v7_flax":
            "support_matrices/nightly/v7x/flax_nnx/quantization_support_matrix.csv",
            "v7_pytorch":
            "support_matrices/nightly/v7x/vllm/quantization_support_matrix.csv"
        },
        "microbenchmarks": {
            "v6":
            "support_matrices/nightly/v6e/default/kernel_support_matrix-microbenchmarks.csv",
            "v7":
            "support_matrices/nightly/v7x/default/kernel_support_matrix-microbenchmarks.csv"
        }
    }
}

README_PATH = "README.md"


def read_csv_data(file_path):
    """Reads a CSV file and returns headers and data rows."""
    if not os.path.exists(file_path):
        return None, []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        return (rows[0], rows[1:]) if rows else (None, [])


def generate_markdown_table(headers, data):
    """Generates a Markdown table string."""
    if not headers:
        return ""

    # helper to replace spaces with &nbsp; for consistent markdown sizing
    def _nbsp(text):
        if not text:
            return text
        return text.replace(" ", "&nbsp;")

    def _format_markdown_cell(text):
        if not text:
            return ""

        text_str = str(text)
        # If the cell is a status cell (contains our standard emojis), format it as an icon tooltip
        if any(emoji in text_str for emoji in ["✅", "❌", "❓", "⚠️"]):
            return _format_cell(text_str)

        return _nbsp(text_str)

    header_line = "| " + " | ".join([_nbsp(h) for h in headers]) + " |\n"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |\n"
    data_lines = ""
    for row in data:
        if len(row) < len(headers):
            row += [""] * (len(headers) - len(row))

        # Format links for the Sequence Parallelism vote explicitly
        formatted_row = []
        is_sp_row = (row and row[0].strip() == "SP")
        for c in row:
            formatted_cell = _format_markdown_cell(c)
            if is_sp_row and "Experimental" in c:
                formatted_row.append(
                    formatted_cell +
                    "&nbsp;([vote&nbsp;to&nbsp;prioritize](https://github.com/vllm-project/tpu-inference/issues/1749))"
                )
            else:
                formatted_row.append(formatted_cell)

        data_lines += "| " + " | ".join(formatted_row) + " |\n"
    return header_line + separator_line + data_lines


def _format_cell(status_string, hw_prefix=None):
    """Formats the cell with a tooltip containing the full status and optional hardware prefix."""
    status_string = str(status_string).strip()

    # Extract the pure status word for the tooltip (e.g., "Passing" from "✅&nbsp;Passing")
    # We must split on "&nbsp;" because merge_metrics uses non-breaking spaces
    clean_string = status_string.replace("&nbsp;", " ")
    parts = clean_string.split(" ", 1)
    icon = parts[0] if parts else ""
    # Unused text_status variable removed

    # Build the visual display text (just icon, or icon + prefix)
    display_text = icon
    if hw_prefix:
        display_text = f"{icon}&nbsp;{hw_prefix}"

    # The tooltip should be descriptive
    tooltip = status_string
    if hw_prefix:
        tooltip = f"{hw_prefix} {status_string}"

    return f'<span title="{tooltip}">{display_text}</span>'


def _merge_hw_status(status_v6, status_v7):
    """Globally condenses v6 and v7 component statuses into a single cell outcome."""
    s6 = str(status_v6).lower()
    s7 = str(status_v7).lower()

    v6_fail = "❌" in s6 or "fail" in s6
    v7_fail = "❌" in s7 or "fail" in s7
    v6_pass = "✅" in s6 or "pass" in s6
    v7_pass = "✅" in s7 or "pass" in s7

    if v6_fail or v7_fail:
        return _format_cell("❌&nbsp;Failing")

    if v6_pass and v7_pass:
        return _format_cell("✅&nbsp;Passing")

    # Handling N/A edge cases
    if "n/a" in s6 and "n/a" in s7:
        return _format_cell("N/A")

    return _format_cell("❓&nbsp;Untested")


def _merge_model_status_text(status_v6, status_v7):
    """Globally condenses v6 and v7 statuses into a single text outcome (no HTML)."""
    s6 = str(status_v6).lower()
    s7 = str(status_v7).lower()

    v6_fail = "❌" in s6 or "fail" in s6
    v7_fail = "❌" in s7 or "fail" in s7
    v6_pass = "✅" in s6 or "pass" in s6
    v7_pass = "✅" in s7 or "pass" in s7
    v6_oom = "not enough hbm" in s6 or "oom" in s6
    v7_oom = "not enough hbm" in s7 or "oom" in s7

    if v6_fail or v7_fail:
        return "❌ Failing"

    if v6_pass and v7_pass:
        return "✅ Passing"

    # Treat OOM as a Pass if the other version passes
    if v7_pass and v6_oom:
        return "✅ Passing"
    if v6_pass and v7_oom:
        return "✅ Passing"

    if "⚠️" in s6 or "beta" in s6 or "⚠️" in s7 or "beta" in s7:
        return "⚠️ Beta"
    if "📝" in s6 or "plan" in s6 or "📝" in s7 or "plan" in s7:
        return "📝 Planned"

    return "❓ Untested"


def generate_html_feature_table(headers, data):
    """Generates an HTML table specifically for the core feature matrix."""
    if not headers:
        return ""

    html = []
    html.append("<table>")
    html.append("  <thead>")
    html.append("    <tr>")
    html.append("      <th>Feature</th>")
    html.append("      <th>Flax</th>")
    html.append("      <th>Torchax</th>")
    html.append("      <th>Default</th>")
    html.append("    </tr>")
    html.append("  </thead>")
    html.append("  <tbody>")

    for row in data:
        html.append("    <tr>")
        html.append(f"      <td>{row[0]}</td>")
        html.append(f"      <td>{row[1]}</td>")
        html.append(f"      <td>{row[2]}</td>")
        html.append(f"      <td>{row[3]}</td>")
        html.append("    </tr>")

    html.append("  </tbody>")
    html.append("</table>")
    return "\n".join(html)


def generate_html_quantization_table(headers, data):
    """Generates an HTML table specifically for the quantization methods matrix."""
    if not headers:
        return ""

    html = []
    html.append("<table>")
    html.append("  <thead>")
    html.append("    <tr>")
    html.append("      <th>Checkpoint dtype</th>")
    html.append("      <th>Method</th>")
    html.append("      <th>Supported<br>Hardware Acceleration</th>")
    html.append("      <th>Flax</th>")
    html.append("      <th>Torchax</th>")
    html.append("    </tr>")
    html.append("  </thead>")
    html.append("  <tbody>")

    for row in data:
        html.append("    <tr>")
        html.append(f"      <td>{row[0]}</td>")
        html.append(f"      <td>{row[1]}</td>")
        html.append(f"      <td>{row[2]}</td>")
        html.append(f"      <td>{row[3]}</td>")
        html.append(f"      <td>{row[4]}</td>")
        html.append("    </tr>")

    html.append("  </tbody>")
    html.append("</table>")
    return "\n".join(html)


def merge_metrics(c, p):
    """Merges Correctness (c) and Performance (p) metrics."""
    c_clean = str(c).replace("&nbsp;", " ").strip()
    p_clean = str(p).replace("&nbsp;", " ").strip()

    if "❌" in c_clean or "❌" in p_clean or "fail" in c_clean.lower(
    ) or "fail" in p_clean.lower():
        return "❌&nbsp;Failing"

    if ("✅" in c_clean
            or "pass" in c_clean.lower()) and ("✅" in p_clean
                                               or "pass" in p_clean.lower()):
        return "✅&nbsp;Passing"

    if "❓" in c_clean or "❓" in p_clean:
        return "❓&nbsp;Untested"

    return "❓&nbsp;Untested"


def _score_status(cell):
    """Scores status cell: ✅=0, ❌=1, ❓=2, others=3."""
    s = str(cell)
    if '✅' in s:
        return 0
    if '❌' in s:
        return 1
    if '❓' in s:
        return 2
    return 3


def _get_model_status_rank(row):
    """Ranks model row status: Has ✅ -> 0, Has ❌ -> 1, Only ❓ -> 2."""
    cells = [str(c) for c in row[2:]]
    if any('✅' in c for c in cells):
        return 0
    if any('❌' in c for c in cells):
        return 1
    return 2


def _find_quantization_status(weight, method, nightly_rows):
    """Finds quantization status recursively inside nightly CSV arrays."""
    if not nightly_rows:
        return "❓ Untested"
    weight = weight.lower().replace(" ", "")
    method = method.lower().replace(" ", "").rstrip('s')

    matched = []
    for nr in nightly_rows:
        if not nr:
            continue
        nr_dtype = nr[0].lower().replace(" ", "")
        nr_method = nr[1].lower().replace(" ", "") if len(nr) > 1 else ""

        if weight in nr_dtype:
            if method in nr_method or method in nr_dtype:
                matched.append(nr)

    if not matched:
        return "❓ Untested"

    overall_corr, overall_perf = "✅", "✅"
    for nr in matched:
        c = nr[3] if len(nr) > 3 else ""
        p = nr[4] if len(nr) > 4 else ""
        if "fail" in c.lower() or "❌" in c:
            overall_corr = "❌"
        elif ("unverified" in c.lower() or "untested" in c.lower()
              or "❓" in c) and overall_corr != "❌":
            overall_corr = "❓"

        if "fail" in p.lower() or "❌" in p:
            overall_perf = "❌"
        elif ("unverified" in p.lower() or "untested" in p.lower()
              or "❓" in p) and overall_perf != "❌":
            overall_perf = "❓"

    return merge_metrics(overall_corr, overall_perf)


def format_kernel_name(name):
    """Formats kernel names to wrap cleanly in max 2-3 lines by using non-breaking spaces and hyphens."""
    name = str(name).replace("-", "&#8209;")
    name = re.sub(r'(?i)<br\s*/?>', ' ',
                  name)  # Clean up any existing manual tags, case-insensitive
    words = name.split(" ")
    lines = []
    current_line = []
    current_len = 0

    for w in words:
        if not w:
            continue
        # If adding this word exceeds ~20 chars and we already have words on this line, break it
        if current_len + len(w) > 20 and current_line:
            lines.append("&nbsp;".join(current_line))
            current_line = [w]
            current_len = len(w)
        else:
            current_line.append(w)
            current_len += len(w) + 1

    if current_line:
        lines.append("&nbsp;".join(current_line))

    return "<br>".join(lines)


def generate_html_microbenchmark_table(headers, data):
    """Generates an HTML table specifically for the microbenchmarks matrix."""
    categories = {"Moe": [], "Dense": [], "Attention": []}

    for row in data:
        k = row[0].lower()
        if "moe" in k or "gmm" in k:
            categories["Moe"].append(row)
        elif "attention" in k or "mla" in k:
            categories["Attention"].append(row)
        else:
            categories["Dense"].append(row)

    html = []
    html.append("<table>")
    html.append("  <thead>")
    html.append("    <tr>")
    html.append(
        "      <th width=\"150\" style=\"text-align:left\">Category</th>")
    html.append("      <th width=\"300\" style=\"text-align:left\">Test</th>")
    html.append("      <th>W16A16</th>")
    html.append("      <th>W8A8</th>")
    html.append("      <th>W8A16</th>")
    html.append("      <th>W4A4</th>")
    html.append("      <th>W4A8</th>")
    html.append("      <th>W4A16</th>")
    html.append("    </tr>")
    html.append("  </thead>")

    display_names = {
        "all-gather-matmul":
        "All&#8209;gather&nbsp;matmul",
        "fused moe":
        "Fused&nbsp;MoE",
        "gmm":
        "gmm",
        "mla*":
        "MLA",
        "generic ragged paged attention v3*":
        "Generic ragged paged attention v3*",
        "ragged paged attention v3 head_dim 64*":
        "Ragged paged attention v3 head_dim 64*"
    }

    for cat_name in ["Moe", "Dense", "Attention"]:
        rows = categories[cat_name]
        if not rows:
            continue

        html.append("  <tbody>")
        for idx, row in enumerate(rows):
            html.append("    <tr>")
            if idx == 0:
                html.append(
                    f"      <td rowspan=\"{len(rows)}\"><b>{cat_name}</b></td>"
                )

            kernel_key = row[0].lower()
            safe_name = display_names.get(kernel_key,
                                          format_kernel_name(row[0].title()))
            html.append(f"      <td>{safe_name}</td>")

            for i in range(1, 7):
                cell_data = row[i] if i < len(
                    row) else '<span title="❓&nbsp;Untested">❓</span>'
                html.append(f"      <td>{cell_data}</td>")
            html.append("    </tr>")
        html.append("  </tbody>")

    html.append("</table>")
    return "\n".join(html)


def generate_html_parallelism_table(headers, data):
    """Generates an HTML table specifically for the parallelism matrix, with Single-host and Multi-host split."""
    if not headers:
        return ""

    html = []
    html.append("<table>")
    html.append("  <thead>")
    html.append("    <tr>")
    html.append(
        "      <th rowspan=\"2\" width=\"150\" style=\"text-align:left\">Feature</th>"
    )
    html.append("      <th colspan=\"2\">Flax</th>")
    html.append("      <th colspan=\"2\">Torchax</th>")
    html.append("    </tr>")
    html.append("    <tr>")
    html.append("      <th>Single-host</th>")
    html.append("      <th>Multi-host</th>")
    html.append("      <th>Single-host</th>")
    html.append("      <th>Multi-host</th>")
    html.append("    </tr>")
    html.append("  </thead>")
    html.append("  <tbody>")

    for row in data:
        html.append("    <tr>")
        feature_name = row[0]
        if feature_name.strip() == "SP":
            feature_html = 'SP&nbsp;(<a href="https://github.com/vllm-project/tpu-inference/issues/1749">vote&nbsp;to&nbsp;prioritize</a>)'
        else:
            feature_html = f"{feature_name}"

        html.append(f"      <td>{feature_html}</td>")
        html.append(f"      <td>{row[1]}</td>")
        html.append(f"      <td>{row[2]}</td>")
        html.append(f"      <td>{row[3]}</td>")
        html.append(f"      <td>{row[4]}</td>")
        html.append("    </tr>")

    html.append("  </tbody>")
    html.append("</table>")
    return "\n".join(html)


def _process_model_support(file_sources):
    """Processes model support matrix data."""
    merged_models = {}
    for col_key, fpath in file_sources.items():
        _, data_rows = read_csv_data(fpath)
        if not data_rows:
            continue
        for row in data_rows:
            if not row:
                continue
            model_name = row[0].strip()
            m_type = row[1] if len(row) > 1 else ""
            unit = row[2] if len(row) > 2 else "❓ Untested"
            corr = row[3] if len(row) > 3 else "❓ Untested"
            bench = row[4] if len(row) > 4 else "❓ Untested"

            if model_name not in merged_models:
                merged_models[model_name] = {
                    "Type": m_type,
                    "v6": {
                        "u": "❓ Untested",
                        "c": "❓ Untested",
                        "b": "❓ Untested"
                    },
                    "v7": {
                        "u": "❓ Untested",
                        "c": "❓ Untested",
                        "b": "❓ Untested"
                    }
                }

            hw_key = "v6" if "v6" in col_key else "v7"
            merged_models[model_name][hw_key] = {
                "u": unit,
                "c": corr,
                "b": bench
            }

    headers = [
        "Model", "Type", "Unit Test", "Correctness Test", "Performance Test"
    ]
    all_data = []
    for model_name, metrics in sorted(merged_models.items(),
                                      key=lambda x:
                                      (x[1]["Type"].lower(), x[0].lower())):
        u_combined = _merge_model_status_text(metrics["v6"]["u"],
                                              metrics["v7"]["u"])
        c_combined = _merge_model_status_text(metrics["v6"]["c"],
                                              metrics["v7"]["c"])
        b_combined = _merge_model_status_text(metrics["v6"]["b"],
                                              metrics["v7"]["b"])

        all_data.append(
            [model_name, metrics["Type"], u_combined, c_combined, b_combined])

    all_data.sort(key=lambda row: (tuple(_score_status(c) for c in row[2:]),
                                   row[1].lower(), row[0].lower()))

    for row in all_data:
        if row and row[0]:
            raw_model_name = row[0].strip("'` ")
            if raw_model_name and not row[0].startswith("["):
                row[0] = f"[{row[0]}](https://huggingface.co/{raw_model_name})"
    return generate_markdown_table(headers, all_data)


def _process_core_features(file_sources):
    """Processes core features matrix data."""
    merged_features = {}
    for col_key, fpath in file_sources.items():
        _, data_rows = read_csv_data(fpath)
        if data_rows:
            for row in data_rows:
                if not row:
                    continue
                feature = row[0].strip()
                if feature not in merged_features:
                    merged_features[feature] = {
                        "v6_flax": "",
                        "v6_pytorch": "",
                        "v6_default": "",
                        "v7_flax": "",
                        "v7_pytorch": "",
                        "v7_default": ""
                    }
                c = row[1] if len(row) > 1 else ""
                p = row[2] if len(row) > 2 else ""
                merged_features[feature][col_key] = merge_metrics(c, p)

    all_data = []
    for feature in sorted(merged_features.keys(), key=lambda x: x.lower()):
        metrics = merged_features[feature]
        merged_flax = _merge_hw_status(metrics["v6_flax"], metrics["v7_flax"])
        merged_pytorch = _merge_hw_status(metrics["v6_pytorch"],
                                          metrics["v7_pytorch"])
        merged_default = _merge_hw_status(metrics["v6_default"],
                                          metrics["v7_default"])
        all_data.append([feature, merged_flax, merged_pytorch, merged_default])

    all_data.sort(key=lambda row:
                  (tuple(_score_status(c) for c in row[1:]), row[0].lower()))

    return generate_html_feature_table(["Feature"], all_data)


def _process_parallelism(file_sources):
    """Processes parallelism matrix data."""
    merged_features = {}
    for col_key, fpath in file_sources.items():
        _, data_rows = read_csv_data(fpath)
        if data_rows:
            for row in data_rows:
                if not row:
                    continue
                feature = row[0].strip()
                if feature not in merged_features:
                    merged_features[feature] = {
                        "v6_flax": {
                            "c": "❓",
                            "p": "❓"
                        },
                        "v6_pytorch": {
                            "c": "❓",
                            "p": "❓"
                        },
                        "v7_flax": {
                            "c": "❓",
                            "p": "❓"
                        },
                        "v7_pytorch": {
                            "c": "❓",
                            "p": "❓"
                        }
                    }
                single_merged = merge_metrics(row[1] if len(row) > 1 else "",
                                              row[2] if len(row) > 2 else "")
                multi_merged = merge_metrics(row[3] if len(row) > 3 else "",
                                             row[4] if len(row) > 4 else "")
                merged_features[feature][col_key] = {
                    "single": single_merged,
                    "multi": multi_merged
                }

    all_data = []
    for feature in sorted(merged_features.keys(), key=lambda x: x.lower()):
        metrics = merged_features[feature]

        def _get_status(hw_key, type_key):
            return metrics.get(hw_key, {}).get(type_key, "❓ Untested")

        flax_single = _merge_hw_status(_get_status("v6_flax", "single"),
                                       _get_status("v7_flax", "single"))
        flax_multi = _merge_hw_status(_get_status("v6_flax", "multi"),
                                      _get_status("v7_flax", "multi"))
        torch_single = _merge_hw_status(_get_status("v6_pytorch", "single"),
                                        _get_status("v7_pytorch", "single"))
        torch_multi = _merge_hw_status(_get_status("v6_pytorch", "multi"),
                                       _get_status("v7_pytorch", "multi"))

        all_data.append(
            [feature, flax_single, flax_multi, torch_single, torch_multi])

    all_data.sort(key=lambda row:
                  (tuple(_score_status(c) for c in row[1:]), row[0].lower()))

    return generate_html_parallelism_table(["Feature"], all_data)


def _process_microbenchmarks(file_sources):
    """Processes microbenchmarks matrix data."""
    _, v6_d = read_csv_data(file_sources["v6"])
    _, v7_d = read_csv_data(file_sources["v7"])

    merged_data = {}
    if v6_d:
        for row in v6_d:
            if not row or row[0].lower() == "kernels":
                continue
            merged_data[row[0]] = {"v6": row[1:]}

    if v7_d:
        for row in v7_d:
            if not row or row[0].lower() == "kernels":
                continue
            if row[0] not in merged_data:
                merged_data[row[0]] = {}
            merged_data[row[0]]["v7"] = row[1:]

    all_data = []
    for kernel in sorted(merged_data.keys()):
        v6_metrics = merged_data[kernel].get("v6", [""] * 18)
        v7_metrics = merged_data[kernel].get("v7", [""] * 18)
        v6_metrics = v6_metrics + [""] * (18 - len(v6_metrics))
        v7_metrics = v7_metrics + [""] * (18 - len(v7_metrics))

        merged_row = [kernel]
        for i in [0, 3, 6, 9, 12, 15]:
            stat_v6 = merge_metrics(v6_metrics[i], v6_metrics[i + 1])
            stat_v7 = merge_metrics(v7_metrics[i], v7_metrics[i + 1])
            merged_row.append(_merge_hw_status(stat_v6, stat_v7))

        all_data.append(merged_row)

    all_data.sort(key=lambda row:
                  (tuple(_score_status(c) for c in row[1:]), row[0].lower()))

    return generate_html_microbenchmark_table(["test"], all_data)


def _process_quantization(file_sources):
    """Processes quantization matrix data."""
    static_file = file_sources["static"]
    headers, static_d = read_csv_data(static_file)
    if not headers:
        return ""

    nightly_data = {}
    for k in ["v6_flax", "v6_pytorch", "v7_flax", "v7_pytorch"]:
        _, d = read_csv_data(file_sources[k])
        nightly_data[k] = d

    all_data = []
    for row in static_d:
        if not row or len(row) < 3:
            continue
        w = row[0]
        m = row[1]
        v6_f = _find_quantization_status(w, m, nightly_data["v6_flax"])
        v6_p = _find_quantization_status(w, m, nightly_data["v6_pytorch"])
        v7_f = _find_quantization_status(w, m, nightly_data["v7_flax"])
        v7_p = _find_quantization_status(w, m, nightly_data["v7_pytorch"])

        merged_flax = _merge_hw_status(v6_f, v7_f)
        merged_pytorch = _merge_hw_status(v6_p, v7_p)

        new_row = row[:3] + [merged_flax, merged_pytorch]
        all_data.append(new_row)

    all_data.sort(key=lambda row:
                  (tuple(_score_status(c) for c in row[3:]), row[0].lower()))

    return generate_html_quantization_table(headers, all_data)


def update_readme():
    """Finds markers in README_dual.md and replaces content with fresh tables."""
    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    for type_key in ["release", "nightly"]:
        type_map = CSV_MAP[type_key]
        for section_key, file_sources in type_map.items():
            if section_key == "model_support":
                new_table = _process_model_support(file_sources)
            elif section_key == "core_features":
                new_table = _process_core_features(file_sources)
            elif section_key == "parallelism":
                new_table = _process_parallelism(file_sources)
            elif section_key == "quantization":
                new_table = _process_quantization(file_sources)
            elif section_key == "microbenchmarks":
                new_table = _process_microbenchmarks(file_sources)
            else:
                continue

            if section_key == "microbenchmarks":
                new_table += "\n\n> **Note:**\n> - *For attention kernels, W[x]A[y] denotes KV cache as W, A as compute, and x, y as bit precision.*"
            elif section_key == "quantization":
                new_table += "\n\n> **Note:**\n> - *This table only tests checkpoint loading compatibility.*"

            os.makedirs("docs/includes", exist_ok=True)
            if type_key == "release":
                snippet_path = os.path.join("docs", "includes",
                                            f"{section_key}.md")
            else:
                snippet_path = os.path.join("docs", "includes",
                                            f"{type_key}_{section_key}.md")
            with open(snippet_path, "w", encoding="utf-8") as f:
                f.write(new_table.strip() + "\n")

            start_marker = f"<!-- START: {type_key}_{section_key} -->"
            end_marker = f"<!-- END: {type_key}_{section_key} -->"
            pattern = f"({re.escape(start_marker)})(.*?)({re.escape(end_marker)})"
            if start_marker in content:
                replacement = f"\\1\n{new_table}\n\\3"
                content = re.sub(pattern,
                                 replacement,
                                 content,
                                 flags=re.DOTALL)

    current_time = datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y-%m-%d %I:%M %p UTC")
    content = re.sub(r"\*Last Updated: .*\*?",
                     f"*Last Updated: {current_time}*", content)

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(
        f"✅ {README_PATH} and MkDocs snippets have been automatically updated."
    )


if __name__ == "__main__":
    update_readme()
