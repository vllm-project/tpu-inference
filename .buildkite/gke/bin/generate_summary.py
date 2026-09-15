#!/usr/bin/env python3
"""
TPU vLLM Benchmark Run Summary Generator
Generates comprehensive report: topology, model, mode, storage, traffic,
benchmark latency/throughput results, and associated log files into log_summary.<number>.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import yaml

def get_job_metadata(job_name, log_num, log_dir):
    meta = {
        "job_name": job_name,
        "mode": "unknown",
        "model_name": "unknown",
        "max_model_len": "4096",
        "load_strategy": "prefetch",
        "image": "docker.io/vllm/vllm-tpu:v0.27.0",
        "storage_type": "unknown",
        "bucket_name": "",
        "cache_subpath": "",
        "ram_cache_limit": "",
        "request_rate": "10.0",
        "stages": [],
        "prefill_topology": "unknown",
        "prefill_vms": 1,
        "prefill_tp": 8,
        "prefill_port": "8400",
        "decode_topology": "unknown",
        "decode_vms": 1,
        "decode_replicas": 1,
        "decode_tp": 8,
        "decode_port": "9400",
        "decode_placement": "",
        "kv_connector": "TPUConnector",
        "proxy_port": "8000",
        "server_topology": "unknown",
        "server_vms": 1,
        "server_tp": 8,
        "server_placement": "",
        "server_port": "8000",
        "service_account": "vllm-sa",
        "status": "UNKNOWN"
    }

    # 1. Try helm get values
    try:
        res = subprocess.run(["helm", "get", "values", job_name], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            vals = yaml.safe_load(res.stdout) or {}
            meta["mode"] = vals.get("mode", meta["mode"])
            meta["image"] = vals.get("image", meta["image"])
            meta["service_account"] = vals.get("serviceAccount", meta["service_account"])

            if "model" in vals:
                meta["model_name"] = vals["model"].get("name", meta["model_name"])
                meta["max_model_len"] = str(vals["model"].get("maxModelLen", meta["max_model_len"]))
                meta["load_strategy"] = vals["model"].get("safetensorsLoadStrategy", meta["load_strategy"])

            if "storage" in vals:
                meta["storage_type"] = vals["storage"].get("type", meta["storage_type"])
                meta["bucket_name"] = vals["storage"].get("bucketName", "")
                meta["cache_subpath"] = vals["storage"].get("cacheSubpath", "")
                meta["ram_cache_limit"] = vals["storage"].get("ramCacheLimit", "")

            if "benchmark" in vals:
                meta["request_rate"] = str(vals["benchmark"].get("requestRate", meta["request_rate"]))
                meta["stages"] = vals["benchmark"].get("stages", [])

            if "disaggregated" in vals:
                dis = vals["disaggregated"]
                meta["kv_connector"] = dis.get("kvConnector") or meta["kv_connector"]
                if "prefill" in dis:
                    meta["prefill_topology"] = dis["prefill"].get("topology", meta["prefill_topology"])
                    meta["prefill_port"] = str(dis["prefill"].get("port", meta["prefill_port"]))
                if "decode" in dis:
                    meta["decode_topology"] = dis["decode"].get("topology", meta["decode_topology"])
                    meta["decode_replicas"] = int(dis["decode"].get("replicas", meta["decode_replicas"]))
                    meta["decode_port"] = str(dis["decode"].get("port", meta["decode_port"]))
                    meta["decode_placement"] = dis["decode"].get("placementPolicy", "")
                if "proxy" in dis:
                    meta["proxy_port"] = str(dis["proxy"].get("port", meta["proxy_port"]))

            if "tpu" in vals:
                meta["server_topology"] = vals["tpu"].get("topology", meta["server_topology"])
                meta["server_placement"] = vals["tpu"].get("placementPolicy", "")
    except Exception:
        pass

    # 2. Try kubectl get jobset
    try:
        res = subprocess.run(["kubectl", "get", "jobset", job_name, "-o", "json"], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            js = json.loads(res.stdout)
            
            # Status check
            conditions = js.get("status", {}).get("conditions", [])
            for cond in conditions:
                if cond.get("type") == "Completed" and cond.get("status") == "True":
                    meta["status"] = "COMPLETED"
                elif cond.get("type") == "Failed" and cond.get("status") == "True":
                    meta["status"] = "FAILED"
            if meta["status"] == "UNKNOWN":
                meta["status"] = "RUNNING"

            rep_jobs = js.get("spec", {}).get("replicatedJobs", [])
            rep_names = [r.get("name") for r in rep_jobs]
            if "p" in rep_names and "d" in rep_names:
                meta["mode"] = "disaggregated"
            elif "server" in rep_names:
                meta["mode"] = "aggregated"

            for r in rep_jobs:
                name = r.get("name")
                tmpl = r.get("template", {}).get("spec", {}).get("template", {})
                selectors = tmpl.get("spec", {}).get("nodeSelector", {})
                containers = tmpl.get("spec", {}).get("containers", [])
                
                if name == "p":
                    meta["prefill_topology"] = selectors.get("cloud.google.com/gke-tpu-topology", meta["prefill_topology"])
                elif name == "d":
                    meta["decode_topology"] = selectors.get("cloud.google.com/gke-tpu-topology", meta["decode_topology"])
                    meta["decode_replicas"] = int(r.get("replicas", meta["decode_replicas"]))
                    meta["decode_placement"] = selectors.get("cloud.google.com/placement-policy-name", meta["decode_placement"])
                elif name == "server":
                    meta["server_topology"] = selectors.get("cloud.google.com/gke-tpu-topology", meta["server_topology"])
                    meta["server_placement"] = selectors.get("cloud.google.com/placement-policy-name", meta["server_placement"])

                if containers and meta["image"] == "docker.io/vllm/vllm-tpu:v0.27.0":
                    meta["image"] = containers[0].get("image", meta["image"])
    except Exception:
        pass

    # 3. Fallback to scanning log files if cluster resources were uninstalled
    client_log = os.path.join(log_dir, f"client.log{log_num}")
    prefill_log = os.path.join(log_dir, f"prefill.log{log_num}")
    decode_log = os.path.join(log_dir, f"decode.log{log_num}")
    server_log = os.path.join(log_dir, f"server.log{log_num}")

    for log_path in [prefill_log, decode_log, server_log, client_log]:
        if os.path.isfile(log_path):
            try:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read(50000)
                    if meta["model_name"] == "unknown":
                        m_m = re.search(r"repo_id=['\"]([^'\"]+)['\"]", txt) or re.search(r"--model[ =]([^ \n]+)", txt)
                        if m_m:
                            meta["model_name"] = m_m.group(1)
                    if meta["mode"] == "unknown":
                        if "kv_producer" in txt or "kv_consumer" in txt or os.path.isfile(prefill_log):
                            meta["mode"] = "disaggregated"
                        elif os.path.isfile(server_log):
                            meta["mode"] = "aggregated"
                    if "kv_connector" in txt:
                        m_kv = re.search(r'"kv_connector":\s*"([^"]+)"', txt)
                        if m_kv:
                            meta["kv_connector"] = m_kv.group(1)
                    m_rr = re.search(r"Traffic request rate:\s+([\d\.]+)", txt) or re.search(r"--request-rate[ =]([\d\.]+)", txt)
                    if m_rr:
                        meta["request_rate"] = m_rr.group(1)
            except Exception:
                pass

    # Helper to calculate VMs and TP size
    def calc_topo(topo_str):
        if not topo_str or topo_str == "unknown":
            return 1, 8
        try:
            parts = [int(p) for p in topo_str.split("x")]
            if len(parts) == 3:
                chips = parts[0] * parts[1] * parts[2]
                return max(1, chips // 4), chips * 2
        except Exception:
            pass
        return 1, 8

    meta["prefill_vms"], meta["prefill_tp"] = calc_topo(meta["prefill_topology"])
    meta["decode_vms"], meta["decode_tp"] = calc_topo(meta["decode_topology"])
    meta["server_vms"], meta["server_tp"] = calc_topo(meta["server_topology"])

    return meta

def parse_client_log(client_log_path):
    if not os.path.isfile(client_log_path):
        return []

    try:
        with open(client_log_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return []

    stage_matches = list(re.finditer(r'\[Stage (\d+)/(\d+)\] Input=(\d+), Output=(\d+) \((\d+) prompts\)', text))
    result_matches = list(re.finditer(r'============ Serving Benchmark Result ============([\s\S]*?)==================================================', text))

    parsed_stages = []
    for stage_m, res_m in zip(stage_matches, result_matches):
        st_idx, total_st, in_len, out_len, num_p = stage_m.groups()
        block = res_m.group(1)

        def extract(pattern, default="N/A"):
            m = re.search(pattern, block)
            return m.group(1).strip() if m else default

        parsed_stages.append({
            "stage": int(st_idx),
            "total_stages": int(total_st),
            "in_len": in_len,
            "out_len": out_len,
            "num_prompts": num_p,
            "completed": extract(r'Successful requests:\s+(\d+)', "0"),
            "failed": extract(r'Failed requests:\s+(\d+)', "0"),
            "duration": extract(r'Benchmark duration \(s\):\s+([\d\.]+)'),
            "req_throughput": extract(r'Request throughput \(req/s\):\s+([\d\.]+)'),
            "tok_throughput": extract(r'Output token throughput \(tok/s\):\s+([\d\.]+)'),
            "peak_tok_throughput": extract(r'Peak output token throughput \(tok/s\):\s+([\d\.]+)'),
            "mean_ttft": extract(r'Mean TTFT \(ms\):\s+([\d\.]+)'),
            "median_ttft": extract(r'Median TTFT \(ms\):\s+([\d\.]+)'),
            "p99_ttft": extract(r'P99 TTFT \(ms\):\s+([\d\.]+)'),
            "mean_tpot": extract(r'Mean TPOT \(ms\):\s+([\d\.]+)'),
            "median_tpot": extract(r'Median TPOT \(ms\):\s+([\d\.]+)'),
            "p99_tpot": extract(r'P99 TPOT \(ms\):\s+([\d\.]+)'),
        })

    return parsed_stages

def format_file_info(filepath):
    if not os.path.isfile(filepath):
        return None
    size_bytes = os.path.getsize(filepath)
    if size_bytes >= 1024 * 1024:
        size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
    elif size_bytes >= 1024:
        size_str = f"{size_bytes / 1024:.1f} KB"
    else:
        size_str = f"{size_bytes} B"

    line_count = 0
    with open(filepath, "rb") as f:
        for _ in f:
            line_count += 1
    return size_str, line_count

def generate_summary(job_name, log_num, log_dir):
    meta = get_job_metadata(job_name, log_num, log_dir)
    client_log = os.path.join(log_dir, f"client.log{log_num}")
    prefill_log = os.path.join(log_dir, f"prefill.log{log_num}")
    decode_log = os.path.join(log_dir, f"decode.log{log_num}")
    proxy_log = os.path.join(log_dir, f"proxy.log{log_num}")
    server_log = os.path.join(log_dir, f"server.log{log_num}")
    summary_file = os.path.join(log_dir, f"log_summary.{log_num}")

    stages_results = parse_client_log(client_log)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("=" * 80)
    lines.append(" ⚡ TPU vLLM Benchmark & Inference Execution Summary")
    lines.append("=" * 80)
    lines.append(f" Release / JobSet : {meta['job_name']}")
    lines.append(f" Log Suffix Index : {log_num}")
    lines.append(f" Serving Mode     : {meta['mode'].upper()}")
    lines.append(f" Generated At     : {now_str}")
    lines.append(f" Job Status       : {meta['status']}")
    lines.append("-" * 80)

    # 1. Model & Storage Configuration
    lines.append(" [1] Model & Storage Configuration")
    lines.append(f"   - Model Target       : {meta['model_name']}")
    lines.append(f"   - Max Context Length : {meta['max_model_len']} tokens")
    lines.append(f"   - Container Image    : {meta['image']}")
    lines.append(f"   - Load Strategy      : {meta['load_strategy']}")
    lines.append(f"   - Storage Mode       : {meta['storage_type']}")
    if meta['bucket_name']:
        lines.append(f"   - GCS Bucket / Path  : gs://{meta['bucket_name']}/{meta['cache_subpath']}")
    if meta['ram_cache_limit']:
        lines.append(f"   - RAM Cache Limit    : {meta['ram_cache_limit']}")
    lines.append(f"   - Service Account    : {meta['service_account']}")
    lines.append("")

    # 2. Hardware Topology
    lines.append(" [2] Hardware Topology & Cluster Layout")
    if meta["mode"] == "disaggregated":
        lines.append(f"   - Prefill Slice (P)  : {meta['prefill_topology']} ({meta['prefill_vms']} VM(s), TP={meta['prefill_tp']}, Port {meta['prefill_port']})")
        if meta['decode_replicas'] > 1:
            lines.append(f"   - Decode Slice (D)   : {meta['decode_replicas']}x {meta['decode_topology']} ({meta['decode_replicas'] * meta['decode_vms']} VM(s), TP={meta['decode_tp']} each, Port {meta['decode_port']})")
        else:
            lines.append(f"   - Decode Slice (D)   : {meta['decode_topology']} ({meta['decode_vms']} VM(s), TP={meta['decode_tp']}, Port {meta['decode_port']})")
        if meta["decode_placement"]:
            lines.append(f"     Placement Policy   : {meta['decode_placement']}")
        lines.append(f"   - KV Connector (P2P) : {meta['kv_connector']} (Port 9100)")
        lines.append(f"   - Proxy Router (X)   : CPU Pod on Port {meta['proxy_port']}")
    else:
        lines.append(f"   - Server Slice       : {meta['server_topology']} ({meta['server_vms']} VM(s), TP={meta['server_tp']}, Port {meta['server_port']})")
        if meta["server_placement"]:
            lines.append(f"     Placement Policy   : {meta['server_placement']}")
    lines.append("")

    # 3. Traffic Benchmark Setup
    lines.append(" [3] Traffic Benchmark Configuration")
    lines.append(f"   - Request Rate (RPS) : {meta['request_rate']} req/s")
    if meta["stages"]:
        lines.append(f"   - Planned Stages     : {len(meta['stages'])} stages")
        for i, s in enumerate(meta["stages"]):
            lines.append(f"       * Stage {i+1}: Input {s.get('inputLen')}, Output {s.get('outputLen')} ({s.get('numPrompts', 100)} prompts)")
    else:
        lines.append("   - Planned Stages     : 5 progressive stages (128/128, 512/256, 1024/512, 2048/512, 3072/512)")
    lines.append("")

    # 4. Performance Metrics Results Table
    lines.append(" [4] Progressive Benchmark Results")
    if stages_results:
        table_header = f" {'Stage':<18} | {'Prompts':<9} | {'Throughput (req/s)':<20} | {'Output (tok/s)':<16} | {'Peak (tok/s)':<13} | {'Mean TTFT':<11} | {'Mean TPOT':<11}"
        lines.append(table_header)
        lines.append(" " + "-" * (len(table_header) - 1))
        for s in stages_results:
            st_label = f"Stage {s['stage']} ({s['in_len']}/{s['out_len']})"
            prompts_str = f"{s['completed']}/{s['num_prompts']}"
            req_tp_str = f"{s['req_throughput']} req/s"
            tok_tp_str = f"{s['tok_throughput']} tok/s"
            peak_tp_str = f"{s['peak_tok_throughput']} tok/s"
            ttft_str = f"{s['mean_ttft']} ms"
            tpot_str = f"{s['mean_tpot']} ms"
            lines.append(f" {st_label:<18} | {prompts_str:<9} | {req_tp_str:<20} | {tok_tp_str:<16} | {peak_tp_str:<13} | {ttft_str:<11} | {tpot_str:<11}")
        lines.append("")
        
        # Latency Percentiles Detailed Breakdown
        lines.append("   Latency Percentiles Breakdown (Median / P99):")
        for s in stages_results:
            lines.append(f"     - Stage {s['stage']} ({s['in_len']}/{s['out_len']}) : TTFT (Med: {s['median_ttft']}ms, P99: {s['p99_ttft']}ms) | TPOT (Med: {s['median_tpot']}ms, P99: {s['p99_tpot']}ms)")
    else:
        lines.append("   (Benchmark results still in progress or not available in client log)")
    lines.append("")

    # 5. Associated Log Files
    lines.append(" [5] Associated Log Files")
    log_candidates = [
        ("Client Benchmark Log", client_log),
        ("Prefill Server Log", prefill_log),
        ("Decode Server Log", decode_log),
        ("Proxy Router Log", proxy_log),
        ("Monolithic Server Log", server_log),
        ("Run Summary Report", summary_file)
    ]
    for desc, path in log_candidates:
        info = format_file_info(path)
        if info:
            size_str, lcnt = info
            lines.append(f"   - {desc:<23} : {path} ({size_str}, {lcnt} lines)")
        elif os.path.basename(path).startswith("log_summary"):
            lines.append(f"   - {desc:<23} : {path} (Generating...)")

    lines.append("=" * 80)
    summary_content = "\n".join(lines) + "\n"

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_content)

    return summary_content, summary_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark run summary")
    parser.add_argument("--job", default="", help="JobSet / Release name")
    parser.add_argument("--number", required=True, help="Log number index")
    parser.add_argument("--dir", default="log", help="Log output directory")
    args = parser.parse_args()

    job_name = args.job
    if not job_name:
        # Fallback to finding latest jobset (preferring current user)
        try:
            user = os.environ.get('USER', '')
            clean_user = ''.join(ch for ch in user.lower() if ch.isalnum())
            p = subprocess.run(['kubectl', 'get', 'jobset', '-o', 'jsonpath={.items[*].metadata.name}'], capture_output=True, text=True)
            jobs = p.stdout.split()
            user_jobs = [j for j in jobs if f'{clean_user}-test-' in j]
            if user_jobs:
                job_name = user_jobs[-1]
            elif jobs:
                job_name = jobs[-1]
            else:
                job_name = 'unknown'
        except Exception:
            job_name = 'unknown' 

    content, out_path = generate_summary(job_name, args.number, args.dir)
    print(content)
    print(f"✅ Run summary saved to: {out_path}")
