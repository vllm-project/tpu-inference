import re
import os
import sys
import numpy as np

debug_dir = "/home/wyzhang_google_com/mnt/ullm/debug"
experiments = ["gather-gather", "onehot-onehot", "gather-fence"]
iters = 3

for exp in experiments:
    throughputs = []
    print(f"--- Experiment: {exp} ---")
    for i in range(1, iters + 1):
        log_file = os.path.join(debug_dir, exp, f"bench_log_{i}", f"bench_{i}.log")
        if not os.path.exists(log_file):
            print(f"Iter {i}: Log file not found at {log_file}")
            continue
        
        with open(log_file, 'r') as f:
            content = f.read()
            # Look for: Output token throughput (tok/s): 1234.56 or "Output token throughput (tok/s): 1234.56"
            # It might appear differently. We will match "Output token throughput (tok/s):" and then float.
            match = re.search(r"Output token throughput \(tok/s\):\s*([\d\.]+)", content)
            if match:
                val = float(match.group(1))
                throughputs.append(val)
                print(f"Iter {i}: {val:.2f} tok/s")
            else:
                print(f"Iter {i}: Metric not found in log.")
    
    if throughputs:
        avg = sum(throughputs) / len(throughputs)
        print(f"Average: {avg:.2f} tok/s\n")
    else:
        print("No valid data collected.\n")
