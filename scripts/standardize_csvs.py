import os
import csv
import glob

SUPPORT_MATRICES_DIR = "support_matrices"
COMBINED_FILE = "combined_model_support_matrix.csv"

# Mapping from older representations to new Standardized Legend
STATUS_MAP = {
    # Passing variants
    "✅": "✅&nbsp;Passing",
    "passing": "✅&nbsp;Passing",
    
    # Failing variants
    "❌": "❌&nbsp;Failing",
    "x": "❌&nbsp;Failing",
    "failing": "❌&nbsp;Failing",
    
    # Untested variants
    "❓": "❓&nbsp;Untested",
    "unverified": "❓&nbsp;Untested",
    "untested": "❓&nbsp;Untested",
    
    # Beta variants
    "⚠️": "⚠️&nbsp;Beta",
    "beta": "⚠️&nbsp;Beta",
    
    # Planned variants
    "📝": "📝&nbsp;Planned",
    "planned": "📝&nbsp;Planned",
    
    # N/A variants
    "⚪": "⚪&nbsp;N/A",
    "n/a": "⚪&nbsp;N/A",
    "": "⚪&nbsp;N/A",
}

def standardize_cell(val):
    """Normalize and map the cell value, returning the original if no match is found."""
    clean_val = val.strip().lower()
    
    # Check if it already matches a standard exactly (ignore case)
    for std_val in STATUS_MAP.values():
        if clean_val == std_val.lower():
            return std_val # Already standard, return the properly cased one
            
    # Check exact matches against mapping keys
    if clean_val in STATUS_MAP:
        return STATUS_MAP[clean_val]
        
    return val # Return original if it's a model name or header

def process_csv(filepath):
    """Reads a CSV, rewrites its cells to standard formats, and saves it in-place."""
    if os.path.basename(filepath) == COMBINED_FILE:
        print(f"⏩ Skipping auto-generated file: {filepath}")
        return

    print(f"🔄 Processing: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        
    if not rows:
        return
        
    new_rows = []
    # Headers remain unchanged
    new_rows.append(rows[0])
    
    changes_made = 0
    for row in rows[1:]:
        new_row = []
        for i, cell in enumerate(row):
            # We assume column 0 is always the "Name" (Model/Feature) and shouldn't be touched by the status mapper
            # Although standardizing it won't hurt if the model name doesn't happen to be "unverified",
            # it's safer to explicitly skip the first column.
            if i == 0:
                new_row.append(cell)
            else:
                new_val = standardize_cell(cell)
                if new_val != cell:
                    changes_made += 1
                new_row.append(new_val)
        new_rows.append(new_row)
        
    if changes_made > 0:
        with open(filepath, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(new_rows)
        print(f"  ✅ Updated {changes_made} cells.")
    else:
        print(f"  ℹ️ No changes needed.")

def main():
    csv_files = glob.glob(os.path.join(SUPPORT_MATRICES_DIR, "*.csv"))
    if not csv_files:
        print("❌ No CSV files found.")
        return
        
    for filepath in csv_files:
        process_csv(filepath)

if __name__ == "__main__":
    main()
