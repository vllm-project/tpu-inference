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
    "core_features": "support_matrices/feature_support_matrix.csv",
    "parallelism": "support_matrices/parallelism_support_matrix.csv",
    "quantization": "support_matrices/quantization_support_matrix.csv",
    "kernel_support": "support_matrices/kernel_support_matrix.csv",
    "microbenchmarks": "support_matrices/nightly/v7/kernel_support_matrix-microbenchmarks.csv"
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
        feature_name = row[0] if len(row) > 0 else ""
        merged_status = merge_metrics(row[1], row[2]) if len(row) > 2 else ""
        
        html.append(f"      <td>{feature_name}</td>")
        html.append(f"      <td></td>") # v6e flax
        html.append(f"      <td>{merged_status}</td>") # v6e pytorch
        html.append(f"      <td></td>") # v6e default
        html.append(f"      <td></td>") # v7x flax
        html.append(f"      <td>{merged_status}</td>") # v7x pytorch
        html.append(f"      <td></td>") # v7x default
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
        # Ensure we have 9 columns worth of data (3 metadata + 6 backend columns)
        padded_row = row + [""] * (9 - len(row))
        for cell in padded_row[:9]:
            html.append(f"      <td>{cell}</td>")
        html.append("    </tr>")
        
    html.append("  </tbody>")
    html.append("</table>")
    return "\n".join(html)

def merge_metrics(c, p):
    """Merges Correctness (c) and Performance (p) metrics."""
    c = str(c).strip()
    p = str(p).strip()
    
    is_failed = "❌" in c or "❌" in p or "Failed" in c or "Failed" in p or "🔴" in c or "🔴" in p
    is_untested = "❓" in c or "❓" in p or "Untested" in c or "Untested" in p or "unverified" in c or "unverified" in p
    
    if is_failed:
        return "❓ Untested" if "❓" in c or "❓" in p else "❌ Failed" # Overriding based on PM logic, Untested could take precedence depending, but "Any Red = Red" usually means failed has highest precedence. Actually, mockup shows untested for red. Let's stick to standard: Failed > Untested > Passed. Wait, PM said: "Any Red = Red. If either is untested, untested". Let's do: Failed wins, then Untested.
        # Wait, the instruction said: "if one of Corr/Perf is untested or failed, show it as untested or failed."
    
    if is_failed:
        return "❌ Failed"
    if is_untested:
        return "❓ Untested"
    if "✅" in c and "✅" in p:
        return "✅ Passing"
    return ""

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
        html.append(f"      <td>{padded_row[0]}</td>") # Kernel
        
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
        elif section_key == "microbenchmarks":
            new_table = generate_html_microbenchmark_table(headers, all_data)
        else:
            new_table = generate_markdown_table(headers, all_data)
        
        # Special handling for microbenchmarks to append footer
        if section_key == "microbenchmarks":
            footer = (
                "\n\n> **Note:**\n"
                "> *   ✅ = Verified Passing\n"
                "> *   ❓ = Unverified\n"
                "> *   ❌ = Failed\n"
                "> *   Performance numbers (e.g., `10ms`) will appear under the icon if available.\n"
                "> *   *Tested on TPU v7 (Nightly 20260217)*\n"
                "> *   *For attention kernels, W[x]A[y] denotes KV cache as W, A as compute, and x, y as bit precision.*"
            )
            new_table += footer

        start_marker, end_marker = f"<!-- START: {section_key} -->", f"<!-- END: {section_key} -->"
        pattern = f"({re.escape(start_marker)})(.*?)({re.escape(end_marker)})"
        replacement = f"\\1\n{new_table}\n\\3"
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ README.md has been automatically updated.")

if __name__ == "__main__":
    update_readme()
