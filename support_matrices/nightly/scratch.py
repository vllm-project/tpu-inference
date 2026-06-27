import os
import csv
import glob
from collections import defaultdict

def calculate_coverage():
    table_metrics = defaultdict(lambda: {"total": 0, "passing": 0})
    csv_files = glob.glob('**/*.csv', recursive=True)
    
    for fpath in csv_files:
        if 'nightly' in fpath:
            continue
            
        table_type = os.path.basename(fpath).replace('.csv', '').replace('_matrix', '')
        
        with open(fpath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if not headers: continue
                
            for row in reader:
                for cell in row[1:]:
                    c = cell.strip()
                    if not c: continue
                        
                    # Target explicit status cells containing standard emojis
                    if any(icon in c for icon in ('✅', '❌', '❓', '📝', '🧪')):
                        table_metrics[table_type]["total"] += 1
                        if '✅' in c:
                            table_metrics[table_type]["passing"] += 1

    total_metrics = 0
    total_passing = 0
    for table_type, metrics in sorted(table_metrics.items()):
        total_metrics += metrics["total"]
        total_passing += metrics["passing"]
        
    print(f"Total metrics: {total_metrics}")
    print(f"Total passing: {total_passing}")

calculate_coverage()
