import csv
import sys

def generate_markdown_table(csv_path):
    with open(csv_path, 'r') as f:
        reader = list(csv.reader(f))

    # Row 0: Precisions (skip empty cols) -> [w16a16, w8a8, w8a16, w4a4, w4a8, w4a16]
    # Row 1: Headers (kernels, correctness, performance, tpu versions...)
    # Row 2+: Data

    # Mapping based on CSV structure (1-based index for clarity in comments, 0-based in code)
    # Col 0: Kernel Name
    # Col 1,2,3: w16a16 (Correctness, Performance, TPU)
    # Col 4,5,6: w8a8
    # Col 7,8,9: w8a16
    # Col 10,11,12: w4a4
    # Col 13,14,15: w4a8
    # Col 16,17,18: w4a16

    precisions = ["W16 A16", "W8 A8", "W8 A16", "W4 A4", "W4 A8", "W4 A16"]
    
    # Start Markdown Table
    headers = ["Kernel"] + precisions
    md_lines = []
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| :--- | " + " | ".join([":---:"] * len(precisions)) + " |")

    # Parse Data Rows (Skip first 2 header rows)
    for row in reader[2:]:
        if not row: continue
        kernel_name = row[0]
        
        # Clean up kernel name (remove quotes if any, though csv reader handles it)
        
        row_cells = [kernel_name]
        
        # Iterating through precisions (striding by 3 columns: Correctness, Performance, TPU)
        # Data starts at index 1
        for i in range(6):
            base_idx = 1 + (i * 3)
            if base_idx >= len(row):
                row_cells.append("")
                continue
                
            corr = row[base_idx].strip().lower()
            perf = row[base_idx+1].strip()
            # tpu = row[base_idx+2] # Ignoring as requested

            # Format Cell
            icon = ""
            if corr == "verified" or corr == "passing" or corr == "ok": # adjusting for likely values
                icon = "✅"
            elif corr == "unverified":
                icon = "❓"
            elif corr == "failed":
                icon = "❌"
            else:
                icon = corr if corr else ""

            cell_content = icon
            if perf and perf != "unverified" and perf != "untested":
                 cell_content += f" <br> *{perf}*"
            
            row_cells.append(cell_content)

        md_lines.append("| " + " | ".join(row_cells) + " |")

    # Append Footer (Legend/Notes)
    footer_lines = [
        "",
        "> **Note:**",
        "> *   ✅ = Verified Passing",
        "> *   ❓ = Unverified",
        "> *   ❌ = Failed",
        "> *   Performance numbers (e.g., `10ms`) will appear under the icon if available.",
        "> *   *Tested on TPU v7 (Nightly 20260217)*", # TODO: Dynamically fetch this if possible
        "> *   *For attention kernels, W[x]A[y] denotes KV cache as W, A as compute, and x, y as bit precision.*"
    ]
    md_lines.extend(footer_lines)

    return "\n".join(md_lines)

if __name__ == "__main__":
    print(generate_markdown_table("support_matrices/nightly/flax_nnx/v7/kernel_support_matrix-microbenchmarks.csv"))
