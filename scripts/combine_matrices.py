import csv
import os

# --- Configuration ---
SUPPORT_MATRICES_DIR = "support_matrices"
MULTIMODAL_CSV = os.path.join(SUPPORT_MATRICES_DIR, "multimodal_model_support_matrix.csv")
TEXT_ONLY_CSV = os.path.join(SUPPORT_MATRICES_DIR, "text_only_model_support_matrix.csv")
OUTPUT_CSV = os.path.join(SUPPORT_MATRICES_DIR, "combined_model_support_matrix.csv")

# Column name mapping
COLUMN_MAPPING = {
    "UnitTest": "Load Test",
    "Accuracy/Correctness": "Correctness Test"
}

# Value formatting mapping
def format_value(val):
    val = val.strip()
    if val == "✅":
        return "✅&nbsp;Passing"
    elif val.lower() == "unverified":
        return "❓&nbsp;Untested"
    elif val == "❌":
        return "❌&nbsp;Failing"
    elif val.lower() == "n/a" or val == "":
        return "⚪&nbsp;N/A"
    return val

def read_and_format_csv(file_path, model_type):
    """Reads a CSV, appends the Type column, and formats the values."""
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: File not found: {file_path}")
        return [], []
        
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        
    if not rows:
        return [], []
        
    original_headers = rows[0]
    
    # Map headers if they exist in mapping, otherwise keep original
    new_headers = ["Model", "Type"]
    for header in original_headers[1:]:
        new_headers.append(COLUMN_MAPPING.get(header, header))
        
    data_rows = []
    for row in rows[1:]:
        if not row: continue
        
        model_name = row[0]
        # Format the model name as inline code (ticks) based on screenshot
        formatted_model = f"`{model_name}`"
        
        new_row = [formatted_model, model_type]
        
        # Format remaining columns
        for val in row[1:]:
            new_row.append(format_value(val))
            
        # Pad row if missing columns
        while len(new_row) < len(new_headers):
            new_row.append("⚪ N/A")
            
        data_rows.append(new_row)
        
    return new_headers, data_rows

def main():
    print("Combining model support matrices...")
    
    # Read and process both files
    mm_headers, mm_data = read_and_format_csv(MULTIMODAL_CSV, "Multimodal")
    text_headers, text_data = read_and_format_csv(TEXT_ONLY_CSV, "Text")
    
    # Determine unified headers (assume they are identical or text_headers is the superset)
    combined_headers = text_headers if text_headers else mm_headers
    
    if not combined_headers:
        print("❌ Error: Could not read headers from either source CSV.")
        return
        
    combined_data = mm_data + text_data
    
    # Write output
    os.makedirs(SUPPORT_MATRICES_DIR, exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(combined_headers)
        writer.writerows(combined_data)
        
    print(f"✅ Successfully wrote combined matrix to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
