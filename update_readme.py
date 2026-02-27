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
        "v6_flax": "support_matrices/nightly/flax_nnx/v6/feature_support_matrix.csv",
        "v6_pytorch": "support_matrices/nightly/vllm/v6/feature_support_matrix.csv",
        "v6_default": "support_matrices/nightly/default/v6/feature_support_matrix.csv",
        "v7_flax": "support_matrices/nightly/flax_nnx/v7/feature_support_matrix.csv",
        "v7_pytorch": "support_matrices/nightly/vllm/v7/feature_support_matrix.csv",
        "v7_default": "support_matrices/nightly/default/v7/feature_support_matrix.csv"
    },
    "parallelism": "support_matrices/parallelism_support_matrix.csv",
    "quantization": "support_matrices/quantization_support_matrix.csv",
    "kernel_support": "support_matrices/kernel_support_matrix.csv",
    "microbenchmarks": {
        "v6": "support_matrices/nightly/vllm/v6/kernel_support_matrix-microbenchmarks.csv",
        "v7": "support_matrices/nightly/vllm/v7/kernel_support_matrix-microbenchmarks.csv"
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
    if not headers: return ""
    
    # helper to replace spaces with &nbsp; for consistent markdown sizing
    def _nbsp(text):
        if not text: return text
        return text.replace(" ", "&nbsp;")

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
            if is_sp_row and "Experimental" in c:
                formatted_row.append(_nbsp("🧪 Experimental") + " ([vote to prioritize](https://github.com/vllm-project/tpu-inference/issues/1749))")
            else:
                formatted_row.append(_nbsp(c))
                
        data_lines += "| " + " | ".join(formatted_row) + " |\n"
    return header_line + separator_line + data_lines

def generate_html_feature_table(headers, data):
    """Generates an HTML table specifically for the core feature matrix."""
    if not headers: return ""
    
    html = []
    html.append("<table>")
    html.append("  <thead>")
    html.append("    <tr>")
    html.append("      <th rowspan=\"2\">Test / Feature</th>")
    html.append("      <th colspan=\"3\">v6e</th>")
    html.append("      <th colspan=\"3\">v7x</th>")
    html.append("    </tr>")
    html.append("    <tr>")
    html.append("      <th>flax</th>")
    html.append("      <th>pytorch</th>")
    html.append("      <th>default</th>")
    html.append("      <th>flax</th>")
    html.append("      <th>pytorch</th>")
    html.append("      <th>default</th>")
    html.append("    </tr>")
    html.append("  </thead>")
    html.append("  <tbody>")
    
    for row in data:
        html.append("    <tr>")
        html.append(f"      <td>{row[0]}</td>")     # Feature name
        html.append(f"      <td>{row[1]}</td>")     # v6e flax
        html.append(f"      <td>{row[2]}</td>")     # v6e pytorch
        html.append(f"      <td>{row[3]}</td>")     # v6e default
        html.append(f"      <td>{row[4]}</td>")     # v7x flax
        html.append(f"      <td>{row[5]}</td>")     # v7x pytorch
        html.append(f"      <td>{row[6]}</td>")     # v7x default
        html.append("    </tr>")
        
    html.append("  </tbody>")
    html.append("</table>")
    return "\n".join(html)

def generate_html_quantization_table(headers, data):
    """Generates an HTML table specifically for the quantization methods matrix."""
    if not headers: return ""
    
    html = []
    html.append("<table>")
    html.append("  <thead>")
    html.append("    <tr>")
    html.append("      <th rowspan=\"2\">Format</th>")
    html.append("      <th rowspan=\"2\">Method</th>")
    html.append("      <th rowspan=\"2\">Recommended<br>TPU Generations</th>")
    html.append("      <th colspan=\"2\">v6e</th>")
    html.append("      <th colspan=\"2\">v7x</th>")
    html.append("    </tr>")
    html.append("    <tr>")
    html.append("      <th>flax</th>")
    html.append("      <th>pytorch</th>")
    html.append("      <th>flax</th>")
    html.append("      <th>pytorch</th>")
    html.append("    </tr>")
    html.append("  </thead>")
    html.append("  <tbody>")
    
    def _format_cell(text):
        text = str(text)
        for status in ["❓ Untested", "✅ Passing", "❌ Failed", "❌ Failing", "⚠️ Beta", "🧪 Experimental", "📝 Planned", "⚪ N/A"]:
            text = text.replace(status, status.replace(" ", "&nbsp;"))
        return text

    for row in data:
        html.append("    <tr>")
        # Ensure we have 9 columns worth of data, then drop default columns (indices 5 and 8)
        padded_row = row + [""] * (9 - len(row))
        indices_to_keep = [0, 1, 2, 3, 4, 6, 7]
        for idx in indices_to_keep:
            html.append(f"      <td>{_format_cell(padded_row[idx])}</td>")
        html.append("    </tr>")
        
    html.append("  </tbody>")
    html.append("</table>")
    return "\n".join(html)

def merge_metrics(c, p):
    """Merges Correctness (c) and Performance (p) metrics."""
    c = str(c).strip()
    p = str(p).strip()
    
    # Empty or hyphen in the CSV should be treated as Untested
    if not c or c == "-": c = "❓"
    if not p or p == "-": p = "❓"
    
    is_failed = "❌" in c or "❌" in p or "Failed" in c or "Failed" in p or "🔴" in c or "🔴" in p
    
    if is_failed:
        return "❓&nbsp;Untested" if "❓" in c or "❓" in p else "❌&nbsp;Failed" 
        
    if "✅" in c and "✅" in p:
        return "✅&nbsp;Passing"
        
    return "❓&nbsp;Untested"

def format_kernel_name(name):
    """Formats kernel names to wrap cleanly in max 2-3 lines by using non-breaking spaces and hyphens."""
    name = str(name).replace("-", "&#8209;")
    name = name.replace("<br>", " ") # Clean up any existing manual tags
    words = name.split(" ")
    lines = []
    current_line = []
    current_len = 0
    
    for w in words:
        if not w: continue
        # If adding this word exceeds ~15 chars and we already have words on this line, break it
        if current_len + len(w) > 15 and current_line:
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
    if not headers: return ""
    
    html = []
    html.append("<table>")
    html.append("  <thead>")
    html.append("    <tr>")
    html.append("      <th rowspan=\"2\" width=\"300\">test</th>")
    html.append("      <th colspan=\"6\">v6e</th>")
    html.append("      <th colspan=\"6\">V7X</th>")
    html.append("    </tr>")
    html.append("    <tr>")
    for _ in range(2):
        html.append("      <th>W16A16</th>")
        html.append("      <th>W8A16</th>")
        html.append("      <th>W8 A8</th>")
        html.append("      <th>W4A4</th>")
        html.append("      <th>W4A8</th>")
        html.append("      <th>W4A16</th>")
    html.append("    </tr>")
    html.append("  </thead>")
    html.append("  <tbody>")
    
    for row in data:
        html.append("    <tr>")
        padded_row = row + [""] * (25 - len(row))
        html.append(f"      <td>{format_kernel_name(padded_row[0])}</td>") # Kernel
        
        # v6e metrics
        html.append(f"      <td>{merge_metrics(padded_row[1], padded_row[2])}</td>")
        html.append(f"      <td>{merge_metrics(padded_row[3], padded_row[4])}</td>")
        html.append(f"      <td>{merge_metrics(padded_row[5], padded_row[6])}</td>")
        html.append(f"      <td>{merge_metrics(padded_row[7], padded_row[8])}</td>")
        html.append(f"      <td>{merge_metrics(padded_row[9], padded_row[10])}</td>")
        html.append(f"      <td>{merge_metrics(padded_row[11], padded_row[12])}</td>")
        
        # v7x metrics
        html.append(f"      <td>{merge_metrics(padded_row[13], padded_row[14])}</td>")
        html.append(f"      <td>{merge_metrics(padded_row[15], padded_row[16])}</td>")
        html.append(f"      <td>{merge_metrics(padded_row[17], padded_row[18])}</td>")
        html.append(f"      <td>{merge_metrics(padded_row[19], padded_row[20])}</td>")
        html.append(f"      <td>{merge_metrics(padded_row[21], padded_row[22])}</td>")
        html.append(f"      <td>{merge_metrics(padded_row[23], padded_row[24])}</td>")
        
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
        
        if section_key == "core_features":
            merged_features = {}
            for col_key, fpath in file_sources.items():
                h, d = read_csv_data(fpath)
                if d:
                    for r in d:
                        if not r: continue
                        feature = r[0].strip()
                        if feature not in merged_features:
                            merged_features[feature] = {"v6_flax": "", "v6_pytorch": "", "v6_default": "", "v7_flax": "", "v7_pytorch": "", "v7_default": ""}
                        c = r[1] if len(r) > 1 else ""
                        p = r[2] if len(r) > 2 else ""
                        merged_features[feature][col_key] = merge_metrics(c, p)

            for feature in sorted(merged_features.keys(), key=lambda x: x.lower()):
                metrics = merged_features[feature]
                row = [
                    feature,
                    metrics["v6_flax"],
                    metrics["v6_pytorch"],
                    metrics["v6_default"],
                    metrics["v7_flax"],
                    metrics["v7_pytorch"],
                    metrics["v7_default"]
                ]
                all_data.append(row)
                
            headers = ["Feature"]
            new_table = generate_html_feature_table(headers, all_data)
            
        elif section_key == "microbenchmarks":
            # Custom merge logic for microbenchmarks (Horizontal Join of v6 and v7)
            v6_h, v6_d = read_csv_data(file_sources["v6"])
            v7_h, v7_d = read_csv_data(file_sources["v7"])
            
            merged_data = {}
            if v6_d:
                for row in v6_d:
                    if not row: continue
                    merged_data[row[0]] = {"v6": row[1:]}
                    
            if v7_d:
                for row in v7_d:
                    if not row: continue
                    if row[0] not in merged_data:
                        merged_data[row[0]] = {}
                    merged_data[row[0]]["v7"] = row[1:]
                    
            for kernel in sorted(merged_data.keys()):
                v6_metrics = merged_data[kernel].get("v6", [""] * 12)
                v7_metrics = merged_data[kernel].get("v7", [""] * 12)
                # Ensure they have exactly 12 columns
                v6_metrics = v6_metrics + [""] * (12 - len(v6_metrics))
                v7_metrics = v7_metrics + [""] * (12 - len(v7_metrics))
                all_data.append([kernel] + v6_metrics[:12] + v7_metrics[:12])
                
            headers = ["test"] # Dummy header, script handles rendering manually
            new_table = generate_html_microbenchmark_table(headers, all_data)
            
        else:
            sources = file_sources if isinstance(file_sources, list) else [file_sources]
            for i, file_path in enumerate(sources):
                h, d = read_csv_data(file_path)
                if h:
                    if not headers: headers = h
                    all_data.extend(d)
            
            if section_key == "core_features":
                new_table = generate_html_feature_table(headers, all_data)
            elif section_key == "quantization":
                new_table = generate_html_quantization_table(headers, all_data)
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
                import datetime
                v7_path = file_sources["v7"]
                mtime = os.path.getmtime(v7_path)
                dt = datetime.datetime.fromtimestamp(mtime)
                date_str = dt.strftime("%Y%m%d")
            except Exception:
                date_str = "Unknown"
                
            footer = (
                "\n\n> **Note:**\n"
                f"> *   *Tested on TPU v7 (Nightly {date_str})*\n"
                "> *   *For attention kernels, W[x]A[y] denotes KV cache as W, A as compute, and x, y as bit precision.*"
            )
            new_table += footer

        start_marker, end_marker = f"<!-- START: {section_key} -->", f"<!-- END: {section_key} -->"
        pattern = f"({re.escape(start_marker)})(.*?)({re.escape(end_marker)})"
        replacement = f"\\1\n{new_table}\n\\3"
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # Automatically update the Last Updated timestamp
    import datetime
    current_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %I:%M %p UTC")
    content = re.sub(r"\*Last Updated: .*\*?", f"*Last Updated: {current_time}*", content)

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ README.md has been automatically updated.")

if __name__ == "__main__":
    update_readme()
