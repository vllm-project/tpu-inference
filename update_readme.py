import csv
import os
import re

# --- CONFIGURATION ---
# We map the README markers (keys) to your EXISTING CSV filenames.
# Note: 'model_support' is a LIST because we need to merge two files.
CSV_MAP = {
    "model_support": [
        "support_matrices/text_only_model_support_matrix.csv",
        "support_matrices/multimodal_model_support_matrix.csv"
    ],
    "core_features": "support_matrices/feature_support_matrix.csv",
    "parallelism": "support_matrices/parallelism_support_matrix.csv",
    "quantization": "support_matrices/quantization_support_matrix.csv",
    "kernel_support": "support_matrices/kernel_support_matrix.csv"
}

README_PATH = "README.md"

def read_csv_data(file_path):
    """Reads a CSV file and returns headers and data rows."""
    if not os.path.exists(file_path):
        print(f"⚠️  Warning: File not found: {file_path}")
        return None, []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        if not rows:
            return None, []
        return rows[0], rows[1:]

def generate_markdown_table(headers, data):
    """Generates a Markdown table string from headers and data."""
    if not headers:
        return ""

    # 1. Header Row
    md_output = "| " + " | ".join(headers) + " |\n"
    # 2. Separator Row
    md_output += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    # 3. Data Rows
    for row in data:
        # Handle cases where row length doesn't match header length
        if len(row) < len(headers):
            row += [""] * (len(headers) - len(row))
        md_output += "| " + " | ".join(row) + " |\n"

    return md_output

def update_readme():
    """Finds markers in README.md and replaces content with CSV data."""
    if not os.path.exists(README_PATH):
        print(f"❌ Error: {README_PATH} not found.")
        return

    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"📖 Reading {README_PATH}...")

    for section_key, file_sources in CSV_MAP.items():
        print(f"   Processing section: {section_key}...")

        headers = []
        all_data = []

        # Handle merging multiple files (like for Model Support)
        if isinstance(file_sources, list):
            for i, file_path in enumerate(file_sources):
                h, d = read_csv_data(file_path)
                if h:
                    if not headers: 
                        headers = h # Use headers from the first valid file
                    all_data.extend(d)
        else:
            # Single file case
            headers, all_data = read_csv_data(file_sources)

        # Generate the table
        new_table = generate_markdown_table(headers, all_data)
        
        # Define markers
        start_marker = f"<!-- START: {section_key} -->"
        end_marker = f"<!-- END: {section_key} -->"
        
        # Regex to find block between markers
        pattern = f"({re.escape(start_marker)})(.*?)({re.escape(end_marker)})"
        
        if start_marker not in content:
            print(f"      ❌ Marker {start_marker} not found in README.")
            continue

        # Replace content
        # \1 keeps start marker, \n adds newlines, \3 keeps end marker
        replacement = f"\\1\n\n{new_table}\n\\3"
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✅ README.md updated successfully!")

if __name__ == "__main__":
    update_readme()
