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
        
    def _format_markdown_cell(text):
        if not text: return ""
        text_str = _nbsp(str(text))
        
        # Look for known status strings to format as tooltips
        for status in ["❓ Untested", "✅ Passing", "❌ Failed", "❌ Failing", "🧪 Experimental", "📝 Planned", "⛔️ Unplanned"]:
            nbsp_status = _nbsp(status)
            if nbsp_status in text_str:
                parts = status.split(" ", 1)
                icon = parts[0] if parts else ""
                tooltip = nbsp_status
                wrapped = f'<span title="{tooltip}">{icon}</span>'
                text_str = text_str.replace(nbsp_status, wrapped)
                
        return text_str

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
                formatted_row.append(formatted_cell + " ([vote to prioritize](https://github.com/vllm-project/tpu-inference/issues/1749))")
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
    text_status = parts[1] if len(parts) > 1 else ""
    
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
    """Merges v6 and v7 statuses. If identical, returns one. If different, stacks them."""
    s6 = status_v6.strip()
    s7 = status_v7.strip()
    if s6 == s7:
        return _format_cell(s6)
    
    return _format_cell(s6, "v6e") + "<br>" + _format_cell(s7, "v7x")

def generate_html_feature_table(headers, data):
    """Generates an HTML table specifically for the core feature matrix, merging v6e and v7x."""
    if not headers: return ""
    
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
        html.append(f"      <td><strong>{row[0]}</strong></td>")  # Feature name, bolded for style
        
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
    if not headers: return ""
    
    html = []
    html.append("<table>")
    html.append("  <thead>")
    html.append("    <tr>")
    html.append("      <th>Format</th>")
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
    return """<table>
  <thead>
    <tr>
      <th width="150" style="text-align:left">Category</th>
      <th width="300" style="text-align:left">test</th>
      <th>W16A16</th>
      <th>W8A16</th>
      <th>W8A8</th>
      <th>W4A4</th>
      <th>W4A8</th>
      <th>W4A16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><b>Moe</b></td>
      <td>Fused&nbsp;MoE</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>gmm</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"><b>Dense</b></td>
      <td>All&#8209;gather&nbsp;matmul</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="2"><b>Attention</b></td>
      <td>MLA</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>Ragged&nbsp;paged&nbsp;attention</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
  </tbody>
</table>"""

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
