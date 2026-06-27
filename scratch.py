import os
import csv
import glob
from collections import defaultdict

def calculate_coverage():
    table_metrics = defaultdict(lambda: {"total": 0, "passing": 0})
    csv_files = glob.glob('support_matrices/**/*.csv', recursive=True)
    
    for fpath in csv_files:
        table_type = os.path.basename(fpath).replace('.csv', '').replace('_matrix', '')
        
        with open(fpath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if not headers: continue
                
            for row in reader:
                for cell in row[1:]:
                    c = cell.strip()
                    if not c: continue
                    # Exclude N/A from total actionable metrics
                    if 'N/A' in c: continue
                        
                    table_metrics[table_type]["total"] += 1
                    if '✅' in c:
                        table_metrics[table_type]["passing"] += 1

    print("### Validated Breakdown per Table (Including Nightly)")
    print("| Table Type | Verified Passing | Actionable/Missing | Total | % Missing |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    total_metrics = 0
    total_passing = 0
    for table_type, metrics in sorted(table_metrics.items()):
        total = metrics["total"]
        passing = metrics["passing"]
        missing = total - passing
        total_metrics += total
        total_passing += passing
        
        pct = (missing / total * 100) if total > 0 else 0
        display_name = table_type.replace('_', ' ').title()
        print(f"| {display_name} | {passing} | {missing} | {total} | **{pct:.1f}%** |")

    print("\n--- Summary ---")
    print(f"Total Tracked Metrics/Tests Statuses: {total_metrics}")
    print(f"Total ✅ Verified Passing: {total_passing}")
    print(f"Total Actionable/Missing: {total_metrics - total_passing}")
    missing_pct = ((total_metrics - total_passing) / total_metrics * 100) if total_metrics > 0 else 0
    print(f"Current Support Matrix Missing %: {missing_pct:.1f}%")

calculate_coverage()
