import csv
import os
import glob

def transform_microbenchmark_csv(file_path):
    print(f"Processing Microbenchmark {file_path}...")
    with open(file_path, 'r') as f:
        reader = list(csv.reader(f))

    if len(reader) < 2:
        print(f"Skipping {file_path}: Too few rows")
        return

    # New Header: 1 + (6 * 2) = 13 columns
    # Clean text only: Kernel, W16 A16 (Corr), W16 A16 (Perf), ...
    precisions = ["W16 A16", "W8 A8", "W8 A16", "W4 A4", "W4 A8", "W4 A16"]
    
    new_header = ["Kernel"]
    for p in precisions:
        new_header.append(f"{p} (Corr)")
        new_header.append(f"{p} (Perf)")
    
    transformed_rows = []
    transformed_rows.append(new_header)

    # Determine if this is a raw file or already updated
    # Raw file: Row 0 has ",w16a16", Row 1 has "kernels,correctness" -> Data starts at 2. Stride 3.
    # Updated file: Row 0 has "Kernel" -> Data starts at 1. Stride 2.
    
    start_row_idx = 0
    stride = 3
    if len(reader) > 0 and "w16a16" in reader[0][1].lower():
        start_row_idx = 2
        stride = 3
    elif len(reader) > 0 and "Kernel" in reader[0][0]:
        start_row_idx = 1
        stride = 2
    else:
        # Fallback
        start_row_idx = 2
        stride = 3
        
    for row in reader[start_row_idx:]:
        if not row: continue
        kernel_name = row[0].strip()
        if not kernel_name or kernel_name.startswith("*") or kernel_name.startswith(">") or kernel_name.startswith("<"):
            continue

        # Manually break long lines
        if kernel_name == "generic ragged paged attention v3*":
            kernel_name = "generic ragged paged<br>attention v3*"
        elif kernel_name == "ragged paged attention v3 head_dim 64*":
            kernel_name = "ragged paged attention v3<br>head_dim 64*"

        row_cells = [kernel_name]
        
        for i in range(6):
            base_idx = 1 + (i * stride)
            if base_idx >= len(row):
                row_cells.append("")
                row_cells.append("")
                continue
                
            corr = row[base_idx].strip().lower()
            perf = row[base_idx+1].strip()
            # If stride is 3, we skip the 3rd column (TPU version) automatically by just picking +0 and +1
            
            # Icon logic
            icon = ""
            if corr in ["verified", "passing", "ok", "✅"]:
                icon = "✅"
            elif corr in ["unverified", "untested", "❓"]:
                icon = "❓"
            elif corr in ["failed", "❌"]:
                icon = "❌"
            else:
                icon = corr if corr else "-"
            
            perf_text = ""
            if perf in ["unverified", "untested", "❓"]:
                perf_text = "❓"
            elif perf:
                 perf_text = perf
            
            row_cells.append(icon)
            row_cells.append(perf_text)

        transformed_rows.append(row_cells)

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(transformed_rows)
    print(f"Updated {file_path}")

def transform_feature_csv(file_path):
    print(f"Processing Feature Matrix {file_path}...")
    with open(file_path, 'r') as f:
        reader = list(csv.reader(f))
        
    if not reader: return

    header = reader[0]
    processed_rows = [header] # Keep header as is
    
    for row in reader[1:]:
        new_row = []
        for cell in row:
            val = cell.strip()
            # formatting logic
            if val == "✅" or val.lower() == "passing":
                 new_row.append("✅ Passing")
            elif val.lower() in ["unverified", "untested", "❓"]:
                 new_row.append("❓ Untested") 
            elif val == "❌" or val.lower() in ["failed", "failing"]:
                 new_row.append("❌ Failing")
            elif val.lower() == "n/a":
                 new_row.append("⚪ N/A") 
            elif val.lower() == "beta":
                 new_row.append("⚠️ Beta")
            elif val.lower() == "experimental":
                 new_row.append("🧪 Experimental")
            elif val.lower() == "planned":
                 new_row.append("📝 Planned")
            else:
                 new_row.append(val)
        processed_rows.append(new_row)
        
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(processed_rows)
    print(f"Updated {file_path}")

def main():
    root_dir = "."
    # 1. Process Microbenchmarks (Specific Logic)
    pattern_micro = "**/*kernel_support_matrix-microbenchmarks.csv"
    files_micro = glob.glob(pattern_micro, recursive=True)
    for f in files_micro:
        transform_microbenchmark_csv(f)

    # 2. Process Other Matrices (Generic Logic)
    # Exclude microbenchmarks to avoid double processing
    pattern_all = "support_matrices/**/*.csv"
    all_csvs = glob.glob(pattern_all, recursive=True)
    
    for f in all_csvs:
        if f in files_micro: continue
        # Also skip combined matrix if we are not regenerating it here (but usually we format it too)
        # For now, let's treat all other CSVs as feature matrices
        transform_feature_csv(f)

if __name__ == "__main__":
    main()
