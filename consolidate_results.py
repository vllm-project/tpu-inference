import os
import sys
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="qwen2_5vl-7b")
args = parser.parse_args()

eval_dir = os.path.join("/drive/SpatialScore/eval_results", args.model_name)
modes = ['standard', 'blind', 'shuffled', 'blurred', 'option_shuffled', 'attention_ablated']

results = {}

print(f"=== Consolidating Ablation Results for {args.model_name} ===")
for mode in modes:
    path = os.path.join(eval_dir, mode, "summary_report.json")
    if not os.path.exists(path):
        print(f"  [!] summary_report.json for mode '{mode}' not found at {path}. Skipping.")
        continue
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    overall_acc = data.get('overall', {}).get('accuracy', 0.0) * 100
    total = data.get('overall', {}).get('count', 0)
    
    results[mode] = {
        "overall_acc": overall_acc,
        "total_samples": total
    }
    print(f"  - {mode}: {overall_acc:.2f}% (Total: {total})")

if not results:
    print(f"No results found to consolidate for model {args.model_name}.")
    exit(1)

# Generate comparative markdown report
md = []
md.append(f"# 📊 SpatialScore {args.model_name.upper()} Ablation & Robustness Scorecard\n")
md.append("| Ablation Mode | Overall Accuracy | Drop (Delta vs. Standard) | Visual Grounding Status |")
md.append("| :--- | :---: | :---: | :--- |")

std_acc = results.get('standard', {}).get('overall_acc', 0.0)
for mode in modes:
    if mode not in results:
        continue
    acc = results[mode]['overall_acc']
    if mode == 'standard':
        delta_str = "Baseline"
        status = "Baseline Reference"
    else:
        if std_acc > 0:
            delta = std_acc - acc
            delta_str = f"-{delta:.2f}%"
            
            # Interpret Status
            if mode == 'blind':
                status = "✅ Robust (High Drop)" if delta > 5 else "❌ Language Shortcut Detected (Low Drop)"
            elif mode == 'shuffled':
                status = "✅ Truly Spatial (High Drop)" if delta > 5 else "❌ Bag-of-Objects Shortcut (Low Drop)"
            elif mode == 'blurred':
                status = "✅ Visually Grounded (High Drop)" if delta > 5 else "❌ Context-Biased (Low Drop)"
            elif mode == 'option_shuffled':
                status = "✅ Option Independent (Low Drop)" if abs(delta) < 3 else "❌ Positional Bias Detected (High Drop)"
            elif mode == 'attention_ablated':
                status = "✅ Causally Grounded (High Drop)" if delta > 5 else "❌ Spurious Attention / Distractor (Low Drop)"
        else:
            delta_str = "N/A (No Standard Run)"
            status = "Awaiting Standard Run comparison"
            
    md.append(f"| **{mode.capitalize()}** | {acc:.2f}% | {delta_str} | {status} |")

# Write to a report file
report_path = os.path.join(eval_dir, "consolidation_report.md")
with open(report_path, 'w') as f:
    f.write("\n".join(md))

# Also output to stdout
print("\n" + "\n".join(md))
print(f"\nReport successfully saved to {report_path}!")
