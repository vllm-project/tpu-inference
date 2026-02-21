import csv
import os
import glob

def transform_csv(file_path):
    print(f"Processing {file_path}...")
    with open(file_path, 'r') as f:
        reader = list(csv.reader(f))

    # Expecting standard format:
    # Row 0: Precisions (skip empty cols) -> [w16a16, w8a8, w8a16, w4a4, w4a8, w4a16]
    # Row 1: Headers (kernels, correctness, performance, tpu versions...)
    # Row 2+: Data

    if len(reader) < 2:
        print(f"Skipping {file_path}: Too few rows")
        return

    # New Header: 1 + (6 * 2) = 13 columns
    # Kernel, W16&nbsp;A16<br>(Corr), W16&nbsp;A16<br>(Perf), ...
    precisions = ["W16&nbsp;A16", "W8&nbsp;A8", "W8&nbsp;A16", "W4&nbsp;A4", "W4&nbsp;A8", "W4&nbsp;A16"]
    # Pad Kernel to force width in Markdown
    new_header = ["Kernel" + "&nbsp;"*18]
    for p in precisions:
        new_header.append(f"{p}<br>(Corr)")
        new_header.append(f"{p}<br>(Perf)")
    
    transformed_rows = []
    transformed_rows.append(new_header)

    # Determine if this is a raw file or already updated
    # Raw file: Row 0 has ",w16a16", Row 1 has "kernels,correctness" -> Data starts at 2
    # Updated file: Row 0 has "Kernel" -> Data starts at 1
    
    start_row_idx = 0
    if len(reader) > 0 and "w16a16" in reader[0][1].lower():
        start_row_idx = 2
    elif len(reader) > 0 and "Kernel" in reader[0][0]:
        start_row_idx = 1
    else:
        print(f"Unknown format for {file_path}, defaulting to row 2")
        start_row_idx = 2

    # Process Data Rows
    for row in reader[start_row_idx:]:
        if not row: continue
        kernel_name = row[0].strip()
        
        # Skip description/legend/footer rows
        # Filter out rows starting with *, >, <, or empty
        if not kernel_name or kernel_name.startswith("*") or kernel_name.startswith(">") or kernel_name.startswith("<"):
            continue

        row_cells = [kernel_name]

        # Manually break long lines for better layout
        if kernel_name == "generic ragged paged attention v3*":
            kernel_name = "generic ragged paged<br>attention v3*"
        elif kernel_name == "ragged paged attention v3 head_dim 64*":
            kernel_name = "ragged paged attention v3<br>head_dim 64*"

        row_cells = [kernel_name]
        
        # Iterating through precisions (striding by 3 columns: Correctness, Performance, TPU)
        # Data starts at index 1
        for i in range(6):
            base_idx = 1 + (i * 3)
            if base_idx >= len(row):
                row_cells.append("")
                row_cells.append("")
                continue
                
            corr = row[base_idx].strip().lower()
            perf = row[base_idx+1].strip()
            # tpu = row[base_idx+2] # Ignoring as requested

            # Format Cell Content for CSV
            
            # Correctness Cell
            icon = ""
            if corr in ["verified", "passing", "ok"]:
                icon = "✅"
            elif corr == "unverified":
                icon = "❓"
            elif corr == "failed":
                icon = "❌"
            else:
                icon = corr if corr else "-"
            
            # Performance Cell
            perf_text = ""
            if perf in ["unverified", "untested"]:
                perf_text = "❓"
            elif perf:
                 perf_text = perf
            
            row_cells.append(icon)
            row_cells.append(perf_text)

        transformed_rows.append(row_cells)

    # Footer
    footer = [
        [],
        ["> **Note:**"],
        ["> *   ✅ = Verified Passing"],
        ["> *   ❓ = Unverified"],
        ["> *   ❌ = Failed"],
        ["> *   Performance numbers (e.g., `10ms`) will appear under the icon if available."],
        ["> *   *Tested on TPU v7 (Nightly 20260217)*"],
        ["> *   *For attention kernels, W[x]A[y] denotes KV cache as W, A as compute, and x, y as bit precision.*"]
    ]
    transformed_rows.extend(footer)

    # Write back to file
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(transformed_rows)
    print(f"Updated {file_path}")

def main():
    root_dir = "."
    # Find all matching files recursively
    pattern = "**/*kernel_support_matrix-microbenchmarks.csv"
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        print("No files found!")
        return

    for file_path in files:
        transform_csv(file_path)

if __name__ == "__main__":
    main()
