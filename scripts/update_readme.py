import csv
import os
import re

# --- CONFIGURATION ---
# This dictionary maps the "markers" in your README to your CSV files.
# This MUST match your file structure exactly.
CSV_MAP = {
    "model_support": [
        "support_matrices/combined_model_support_matrix.csv"
    ],
    "core_features": {
        "v6_flax": "support_matrices/nightly/v6e/flax_nnx/feature_support_matrix.csv",
        "v6_pytorch": "support_matrices/nightly/v6e/vllm/feature_support_matrix.csv",
        "v6_default": "support_matrices/nightly/v6e/default/feature_support_matrix.csv",
        "v7_flax": "support_matrices/nightly/v7x/flax_nnx/feature_support_matrix.csv",
        "v7_pytorch": "support_matrices/nightly/v7x/vllm/feature_support_matrix.csv",
        "v7_default": "support_matrices/nightly/v7x/default/feature_support_matrix.csv"
    },
    "parallelism": {
        "v6_flax": "support_matrices/nightly/v6e/flax_nnx/parallelism_support_matrix.csv",
        "v6_pytorch": "support_matrices/nightly/v6e/vllm/parallelism_support_matrix.csv",
        "v7_flax": "support_matrices/nightly/v7x/flax_nnx/parallelism_support_matrix.csv",
        "v7_pytorch": "support_matrices/nightly/v7x/vllm/parallelism_support_matrix.csv"
    },
    "quantization": {
        "static": "support_matrices/quantization_support_matrix.csv",
        "v6_flax": "support_matrices/nightly/v6e/flax_nnx/quantization_support_matrix.csv",
        "v6_pytorch": "support_matrices/nightly/v6e/vllm/quantization_support_matrix.csv",
        "v6_default": "support_matrices/nightly/v6e/default/quantization_support_matrix.csv",
        "v7_flax": "support_matrices/nightly/v7x/flax_nnx/quantization_support_matrix.csv",
        "v7_pytorch": "support_matrices/nightly/v7x/vllm/quantization_support_matrix.csv",
        "v7_default": "support_matrices/nightly/v7x/default/quantization_support_matrix.csv"
    },
    "kernel_support": "support_matrices/kernel_support_matrix.csv",
    "microbenchmarks": {
        "v6": "support_matrices/nightly/v6e/vllm/kernel_support_matrix-microbenchmarks.csv",
        "v7": "support_matrices/nightly/v7x/vllm/kernel_support_matrix-microbenchmarks.csv"
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
                formatted_row.append(formatted_cell + "&nbsp;([vote&nbsp;to&nbsp;prioritize](https://github.com/vllm-project/tpu-inference/issues/1749))")
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
    
    # Failing takes ultimate priority: if any fails, the whole box fails.
    if "❌" in s6 or "fail" in s6 or "❌" in s7 or "fail" in s7:
        return _format_cell("❌&nbsp;Failing")
        
    # If both pass natively
    if ("✅" in s6 or "pass" in s6) and ("✅" in s7 or "pass" in s7):
        return _format_cell("✅&nbsp;Passing")
        
    # Handling N/A edge cases
    s6_na = "n/a" in s6
    s7_na = "n/a" in s7
    s6_pass = "✅" in s6 or "pass" in s6
    s7_pass = "✅" in s7 or "pass" in s7
    
    if s6_pass and s7_na:
        return _format_cell("✅&nbsp;Passing")
    if s7_pass and s6_na:
        return _format_cell("✅&nbsp;Passing")
    if s6_na and s7_na:
        return _format_cell("N/A")
        
    # Any residual state (unverified, untested, missing) goes to untested.
    return _format_cell("❓&nbsp;Untested")

def generate_html_feature_table(headers, data):
    """Generates an HTML table specifically for the core feature matrix, merging v6e and v7x."""
    if not headers:
        return ""
    
    html = []
    html.append("<table>")
    html.append("  <thead>")
    html.append("    <tr>")
    html.append("      <th>Test / Feature</th>")
    html.append("      <th>flax</th>")
    html.append("      <th>torchax</th>")
    html.append("      <th>default</th>")
    html.append("    </tr>")
    html.append("  </thead>")
    html.append("  <tbody>")

    for row in data:
        html.append("    <tr>")
        html.append(f"      <td>{row[0]}</td>")  # Feature name, bolded for style
        
        merged_flax = _merge_hw_status(row[1], row[4])
        merged_pytorch = _merge_hw_status(row[2], row[5])
        merged_default = _merge_hw_status(row[3], row[6])
        
        html.append(f"      <td>{merged_flax}</td>")
        html.append(f"      <td>{merged_pytorch}</td>")
        html.append(f"      <td>{merged_default}</td>")
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
    html.append("      <th>flax</th>")
    html.append("      <th>torchax</th>")
    html.append("    </tr>")
    html.append("  </thead>")
    html.append("  <tbody>")
    
    for row in data:
        html.append("    <tr>")
        # Ensure we have 9 columns worth of data, then drop default columns (indices 5 and 8)
        padded_row = row + [""] * (9 - len(row))
        html.append(f"      <td>{padded_row[0]}</td>")
        html.append(f"      <td>{padded_row[1]}</td>")
        html.append(f"      <td>{padded_row[2]}</td>")
        
        merged_flax = _merge_hw_status(padded_row[3], padded_row[6])
        merged_pytorch = _merge_hw_status(padded_row[4], padded_row[7])
        
        html.append(f"      <td>{merged_flax}</td>")
        html.append(f"      <td>{merged_pytorch}</td>")
        html.append("    </tr>")
        
    html.append("  </tbody>")
    html.append("</table>")
    return "\n".join(html)



def merge_metrics(c, p):
    """Merges Correctness (c) and Performance (p) metrics."""
    c = str(c).strip()
    p = str(p).strip()
    
    if c == p:
        return c.replace(" ", "&nbsp;")
        
    c_clean = c.replace("&nbsp;", " ")
    p_clean = p.replace("&nbsp;", " ")
    
    if "❌" in c_clean or "❌" in p_clean:
        return "❌&nbsp;Failing"
        
    if "❓" in c_clean or "❓" in p_clean:
        return "❓&nbsp;Untested"
        
    if "✅" in c_clean and "✅" in p_clean:
        return "✅&nbsp;Passing"
        
    return "❓&nbsp;Untested"

import re

def format_kernel_name(name):
    """Formats kernel names to wrap cleanly in max 2-3 lines by using non-breaking spaces and hyphens."""
    name = str(name).replace("-", "&#8209;")
    name = re.sub(r'(?i)<br\s*/?>', ' ', name) # Clean up any existing manual tags, case-insensitive
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
    html.append("      <th width=\"150\" style=\"text-align:left\">Category</th>")
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
        "all-gather-matmul": "All&#8209;gather&nbsp;matmul",
        "fused moe": "Fused&nbsp;MoE",
        "gmm": "gmm",
        "mla*": "MLA",
        "generic ragged paged attention v3*": "Generic ragged paged attention v3*",
        "ragged paged attention v3 head_dim 64*": "Ragged paged attention v3 head_dim 64*"
    }
    
    for cat_name in ["Moe", "Dense", "Attention"]:
        rows = categories[cat_name]
        if not rows:
            continue
        
        html.append("  <tbody>")
        for idx, row in enumerate(rows):
            html.append("    <tr>")
            if idx == 0:
                html.append(f"      <td rowspan=\"{len(rows)}\"><b>{cat_name}</b></td>")
            
            kernel_key = row[0].lower()
            safe_name = display_names.get(kernel_key, format_kernel_name(row[0].title()))
            html.append(f"      <td>{safe_name}</td>")
            
            for i in range(1, 7):
                cell_data = row[i] if i < len(row) else '<span title="❓&nbsp;Untested">❓</span>'
                html.append(f"      <td>{cell_data}</td>")
            html.append("    </tr>")
        html.append("  </tbody>")
        
    html.append("</table>")
    return "\n".join(html)

def generate_html_parallelism_table(headers, data):
    """Generates an HTML table specifically for the parallelism matrix, with merged correctness and performance."""
    if not headers:
        return ""
    
    html = []
    html.append("<table>")
    html.append("  <thead>")
    html.append("    <tr>")
    html.append("      <th width=\"150\" style=\"text-align:left\">Feature</th>")
    html.append("      <th>Flax</th>")
    html.append("      <th>torchax</th>")
    html.append("    </tr>")
    html.append("  </thead>")
    html.append("  <tbody>")

    for row in data:
        html.append("    <tr>")
        feature_name = row[0]
        if feature_name.strip() == "SP":
            feature_html = '<span style="white-space: nowrap;">SP (<a href="https://github.com/vllm-project/tpu-inference/issues/1749">vote to prioritize</a>)</span>'
        else:
            feature_html = f"{feature_name}"
        
        html.append(f"      <td>{feature_html}</td>")
        
        flax_merged = _merge_hw_status(row[1], row[3])
        torchax_merged = _merge_hw_status(row[2], row[4])
        
        html.append(f"      <td>{flax_merged}</td>")
        html.append(f"      <td>{torchax_merged}</td>")
        html.append("    </tr>")
        
    html.append("  </tbody>")
    html.append("</table>")
    return "\n".join(html)

def update_readme():
    """Finds markers in README.md and replaces content with fresh tables."""
    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    for section_key, file_sources in CSV_MAP.items():
        headers, all_data = [], []
        
        if section_key in ("core_features", "parallelism"):
            merged_features = {}
            for col_key, fpath in file_sources.items():
                h, d = read_csv_data(fpath)
                if d:
                    for r in d:
                        if not r:
                            continue
                        feature = r[0].strip()
                        
                        if feature not in merged_features:
                            if section_key == "parallelism":
                                merged_features[feature] = {
                                    "v6_flax": "", 
                                    "v6_pytorch": "", 
                                    "v7_flax": "", 
                                    "v7_pytorch": ""
                                }
                            else:
                                merged_features[feature] = {"v6_flax": "", "v6_pytorch": "", "v6_default": "", "v7_flax": "", "v7_pytorch": "", "v7_default": ""}
                        
                        c = r[1] if len(r) > 1 else ""
                        p = r[2] if len(r) > 2 else ""
                        
                        merged_features[feature][col_key] = merge_metrics(c, p)

            for feature in sorted(merged_features.keys(), key=lambda x: x.lower()):
                metrics = merged_features[feature]
                if section_key == "core_features":
                    row = [
                        feature,
                        metrics["v6_flax"],
                        metrics["v6_pytorch"],
                        metrics["v6_default"],
                        metrics["v7_flax"],
                        metrics["v7_pytorch"],
                        metrics["v7_default"]
                    ]
                else:
                    row = [
                        feature,
                        metrics.get("v6_flax", "❓"),
                        metrics.get("v6_pytorch", "❓"),
                        metrics.get("v7_flax", "❓"),
                        metrics.get("v7_pytorch", "❓")
                    ]
                all_data.append(row)
                
            headers = ["Feature"]
            if section_key == "core_features":
                new_table = generate_html_feature_table(headers, all_data)
            else:
                new_table = generate_html_parallelism_table(headers, all_data)
            
        elif section_key == "microbenchmarks":
            # Custom merge logic for microbenchmarks (Horizontal Join of v6 and v7)
            v6_h, v6_d = read_csv_data(file_sources["v6"])
            v7_h, v7_d = read_csv_data(file_sources["v7"])
            
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
                    
            for kernel in sorted(merged_data.keys()):
                v6_metrics = merged_data[kernel].get("v6", [""] * 18)
                v7_metrics = merged_data[kernel].get("v7", [""] * 18)
                # Ensure they have exactly 18 columns to cover 6 quantizations (3 columns each)
                v6_metrics = v6_metrics + [""] * (18 - len(v6_metrics))
                v7_metrics = v7_metrics + [""] * (18 - len(v7_metrics))
                
                merged_row = [kernel]
                for i in [0, 3, 6, 9, 12, 15]: 
                    stat_v6 = merge_metrics(v6_metrics[i], v6_metrics[i+1])
                    stat_v7 = merge_metrics(v7_metrics[i], v7_metrics[i+1])
                    merged_row.append(_merge_hw_status(stat_v6, stat_v7))
                
                all_data.append(merged_row)
                
            headers = ["test"] # Dummy header, script handles rendering manually
            new_table = generate_html_microbenchmark_table(headers, all_data)
            
        elif section_key == "quantization":
            static_file = file_sources["static"]
            headers, static_d = read_csv_data(static_file)
            if not headers:
                continue
            
            nightly_data = {}
            for k in ["v6_flax", "v6_pytorch", "v6_default", "v7_flax", "v7_pytorch", "v7_default"]:
                _, d = read_csv_data(file_sources[k])
                nightly_data[k] = d

            def find_status(weight, method, nightly_rows):
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
                    if "unverified" in c.lower() or "untested" in c.lower() or "❓" in c:
                        overall_corr = "❓"
                    elif "fail" in c.lower() or "❌" in c:
                        overall_corr = "❌"
                        
                    if "unverified" in p.lower() or "untested" in p.lower() or "❓" in p:
                        overall_perf = "❓"
                    elif "fail" in p.lower() or "❌" in p:
                        overall_perf = "❌"
                        
                return merge_metrics(overall_corr, overall_perf)
                
            for row in static_d:
                if not row or len(row) < 3:
                    continue
                w = row[0]
                m = row[1]
                v6_f = find_status(w, m, nightly_data["v6_flax"])
                v6_p = find_status(w, m, nightly_data["v6_pytorch"])
                v6_d = find_status(w, m, nightly_data["v6_default"])
                v7_f = find_status(w, m, nightly_data["v7_flax"])
                v7_p = find_status(w, m, nightly_data["v7_pytorch"])
                v7_d = find_status(w, m, nightly_data["v7_default"])
                
                new_row = row[:3] + [v6_f, v6_p, v6_d, v7_f, v7_p, v7_d]
                all_data.append(new_row)
                
            new_table = generate_html_quantization_table(headers, all_data)
            
        else:
            sources = file_sources if isinstance(file_sources, list) else [file_sources]
            for i, file_path in enumerate(sources):
                h, d = read_csv_data(file_path)
                if h:
                    if not headers:
                        headers = h
                    all_data.extend(d)
            
            if section_key == "core_features":
                new_table = generate_html_feature_table(headers, all_data)
            else:
                if section_key == "model_support":
                    for row in all_data:
                        if row and row[0]:
                            raw_model_name = row[0].strip("'` ")
                            if raw_model_name and not row[0].startswith("["):
                                row[0] = f"[{row[0]}](https://huggingface.co/{raw_model_name})"
                new_table = generate_markdown_table(headers, all_data)
        
        # Special handling for microbenchmarks to append footer
        if section_key == "microbenchmarks":
            try:
                # dt and date_str removed
                pass
            except Exception:
                pass

            footer = (
                "\n\n> **Note:**\n"
                "> *   *For attention kernels, W[x]A[y] denotes KV cache as W, A as compute, and x, y as bit precision.*"
            )
            new_table += footer
        elif section_key == "quantization":
            footer = (
                "\n\n> **Note:**\n"
                "> &bull;&nbsp;&nbsp;&nbsp;*This table only tests checkpoint loading compatibility.*"
            )
            new_table += footer

        start_marker, end_marker = f"<!-- START: {section_key} -->", f"<!-- END: {section_key} -->"
        pattern = f"({re.escape(start_marker)})(.*?)({re.escape(end_marker)})"
        replacement = f"\\1\n{new_table}\n\\3"
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # Automatically update the Last Updated timestamp
    import datetime
    current_time = datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y-%m-%d %I:%M %p UTC")
    content = re.sub(r"\*Last Updated: .*\*?",
                     f"*Last Updated: {current_time}*", content)

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ README.md has been automatically updated.")


if __name__ == "__main__":
    update_readme()
