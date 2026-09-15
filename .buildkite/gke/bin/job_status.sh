#!/usr/bin/env bash
# ==============================================================================
#  ⚡ TPU vLLM Benchmark & Workload Diagnostic Utility
#  Author: Advanced Agentic Assistant for TPU vLLM Inference
# ==============================================================================
# Usage:
#   ./gke/bin/job_status.sh [OPTIONS] [JOB_NAME]
#
# Examples:
#   ./gke/bin/job_status.sh
#   ./gke/bin/job_status.sh my-benchmark-job
#   ./gke/bin/job_status.sh -w my-benchmark-job   # Watch mode (refreshes every 5s)
# ==============================================================================

set -eo pipefail

WATCH_MODE=false
WATCH_INTERVAL=5
JOB_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--watch)
            WATCH_MODE=true
            shift
            ;;
        -i|--interval)
            WATCH_INTERVAL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [JOB_NAME]"
            echo ""
            echo "Options:"
            echo "  -w, --watch         Watch live status with periodic refresh (default: 5s)"
            echo "  -i, --interval <N>  Set refresh interval in seconds (default: 5)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            if [ -z "$JOB_NAME" ]; then
                JOB_NAME="$1"
            fi
            shift
            ;;
    esac
done

CURRENT_USER=$(whoami 2>/dev/null || echo "user")
CLEAN_USER="${CURRENT_USER//_/-}"

# Auto-detect latest JobSet if not provided
if [ -z "$JOB_NAME" ]; then
    JOB_NAME=$(kubectl get jobset -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null | grep "${CLEAN_USER}-test-" | tail -n 1 || true)
    if [ -z "$JOB_NAME" ]; then
        JOB_NAME=$(kubectl get jobset -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null || true)
    fi
fi

if [ -z "$JOB_NAME" ]; then
    echo "❌ Error: No active JobSets found in namespace. Specify JobSet name explicitly."
    exit 1
fi

run_diagnostics() {
python3 - "${JOB_NAME}" << 'PYEOF'
import sys, json, subprocess, shutil, re

job_name = sys.argv[1]

# Terminal colors
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
RESET = "\033[0m"

terminal_width = shutil.get_terminal_size((120, 24)).columns
sep = "=" * min(terminal_width, 120)
subsep = "-" * min(terminal_width, 120)

def truncate_str(val, max_len, mode="end"):
    """Truncates string to max_len using '...' if it exceeds limits."""
    if not val:
        return ""
    val_str = str(val).strip()
    if len(val_str) <= max_len:
        return val_str
    if max_len <= 3:
        return val_str[:max_len]
    if mode == "middle":
        p_len = (max_len - 3) // 2
        s_len = max_len - 3 - p_len
        return f"{val_str[:p_len]}...{val_str[-s_len:]}"
    return val_str[:max_len - 3] + "..."

print(f"{BOLD}{CYAN}{sep}{RESET}")
print(f"{BOLD}{CYAN} ⚡ TPU vLLM Workload & Cluster Diagnostics: {YELLOW}{job_name}{RESET}")
print(f"{BOLD}{CYAN}{sep}{RESET}")

# 1. JobSet Overview
jobset_data = None
try:
    raw_js = subprocess.check_output(["kubectl", "get", "jobset", job_name, "-o", "json"], stderr=subprocess.DEVNULL).decode()
    jobset_data = json.loads(raw_js)
except Exception:
    pass

if not jobset_data:
    print(f"{YELLOW}⚠️ JobSet '{job_name}' not found in active JobSets. Querying pods directly...{RESET}")
else:
    status_obj = jobset_data.get("status", {})
    term_state = status_obj.get("terminalState")
    conditions = status_obj.get("conditions", [])
    cond_str = "Active"
    if term_state:
        cond_str = f"{RED}Failed{RESET}" if "fail" in term_state.lower() else f"{GREEN}{term_state}{RESET}"
    elif conditions:
        latest_c = conditions[-1]
        c_type = latest_c.get("type", "")
        c_status = latest_c.get("status", "")
        c_msg = latest_c.get("message", "")
        if c_type == "Failed" and c_status == "True":
            cond_str = f"{RED}Failed: {c_msg}{RESET}"
        elif c_type == "Completed" and c_status == "True":
            cond_str = f"{GREEN}Completed{RESET}"
        else:
            cond_str = f"{CYAN}{c_type} ({c_status}){RESET}"

    created_at = jobset_data.get("metadata", {}).get("creationTimestamp", "Unknown")
    replicated_jobs = jobset_data.get("spec", {}).get("replicatedJobs", [])
    rep_summary = ", ".join([f"{rj.get('name')}: {rj.get('replicas', 1)}x" for rj in replicated_jobs])
    
    print(f" {BOLD}Status{RESET}       : {cond_str}")
    print(f" {BOLD}Created At{RESET}   : {created_at}")
    print(f" {BOLD}Components{RESET}   : {rep_summary}")

# 2. Query Pods & Batch Events
raw_pods = ""
try:
    raw_pods = subprocess.check_output(["kubectl", "get", "pods", "-l", f"jobset.sigs.k8s.io/jobset-name={job_name}", "-o", "json"], stderr=subprocess.DEVNULL).decode()
except Exception as e:
    print(f"{RED}Error fetching pods for {job_name}: {e}{RESET}")
    sys.exit(1)

pods_data = json.loads(raw_pods).get("items", [])
if not pods_data:
    print(f"{RED}No pods found for JobSet '{job_name}'.{RESET}")
    sys.exit(0)

# Batch fetch pod events
pod_events_map = {}
try:
    raw_evs = subprocess.check_output(["kubectl", "get", "events", "--field-selector", "involvedObject.kind=Pod", "-o", "json"], stderr=subprocess.DEVNULL).decode()
    all_evs = json.loads(raw_evs).get("items", [])
    for ev in all_evs:
        p_name = ev.get("involvedObject", {}).get("name")
        if p_name and job_name in p_name:
            pod_events_map.setdefault(p_name, []).append(ev)
except Exception:
    pass

# Sort pods logically: client -> x -> p -> d
def pod_sort_key(pod):
    name = pod["metadata"]["name"]
    short = name[len(job_name)+1:]
    parts = short.split("-")
    prefix = parts[0]
    p_order = {"client": 0, "x": 1, "p": 2, "d": 3, "server": 4}.get(prefix, 5)
    r_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    p_idx = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    return (p_order, r_idx, p_idx, name)

pods_data.sort(key=pod_sort_key)

def get_role_info(name, job):
    short = name[len(job)+1:]
    parts = short.split("-")
    prefix = parts[0]
    if prefix == "client": return ("Client", "Benchmarking client pod")
    if prefix == "x": return ("Proxy Router", "Disaggregated proxy router")
    if prefix == "p":
        idx = parts[2] if len(parts) > 2 else "0"
        return ("Prefill Pod 0", "Primary writer & TPU prefill head") if idx == "0" else (f"Prefill Worker {idx}", f"TPU worker node {idx}")
    if prefix == "d":
        rep = parts[1] if len(parts) > 1 else "0"
        idx = parts[2] if len(parts) > 2 else "0"
        return (f"Decode R{rep} Head", f"Replica {rep} Ray head & vLLM consumer") if idx == "0" else (f"Decode R{rep} Worker {idx}", f"Replica {rep} TPU worker {idx}")
    if prefix == "server":
        idx = parts[2] if len(parts) > 2 else "0"
        return ("Server Head", "Monolithic Ray head & vLLM API server") if idx == "0" else (f"Server Worker {idx}", f"TPU multi-host worker {idx}")
    return (short, "Workload pod")

# Query pod top metrics
top_metrics = {}
try:
    raw_top = subprocess.check_output(["kubectl", "top", "pod", "-l", f"jobset.sigs.k8s.io/jobset-name={job_name}", "--containers"], stderr=subprocess.DEVNULL).decode()
    for line in raw_top.strip().splitlines()[1:]:
        t_parts = line.split()
        if len(t_parts) >= 4:
            p_n, c_n, cpu, mem = t_parts[0], t_parts[1], t_parts[2], t_parts[3]
            top_metrics.setdefault(p_n, {})[c_n] = (cpu, mem)
except Exception:
    pass

# Parallel fetch logs for running pods
from concurrent.futures import ThreadPoolExecutor

def fetch_pod_log(pname):
    target_container = "vllm-tpu"
    if "client" in pname: target_container = "benchmark-client"
    elif "-x-" in pname: target_container = "proxy-server"
    try:
        raw_log = subprocess.check_output(
            ["kubectl", "logs", pname, "-c", target_container, "--tail=5"],
            stderr=subprocess.STDOUT, timeout=2
        ).decode().strip()
        return (pname, raw_log)
    except Exception:
        return (pname, "")

running_pod_names = [p["metadata"]["name"] for p in pods_data if p["status"].get("phase") == "Running"]
pod_logs_map = {}
if running_pod_names:
    with ThreadPoolExecutor(max_workers=min(12, len(running_pod_names))) as executor:
        results = executor.map(fetch_pod_log, running_pod_names)
        pod_logs_map = dict(results)

# Helper to get pod diagnostics
def get_pod_diagnostic(pod):
    phase = pod["status"].get("phase", "Unknown")
    pname = pod["metadata"]["name"]
    
    # Check container statuses
    c_statuses = pod["status"].get("containerStatuses", []) or []
    init_statuses = pod["status"].get("initContainerStatuses", []) or []
    
    for cs in c_statuses + init_statuses:
        state = cs.get("state", {})
        if "waiting" in state:
            w_reason = state["waiting"].get("reason", "")
            w_msg = state["waiting"].get("message", "")
            if w_reason in ("CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull"):
                return f"{RED}[{w_reason}] {w_msg[:60]}{RESET}"
            if w_reason == "PodInitializing":
                return f"{CYAN}🔄 Pulling container image & initializing...{RESET}"
        if "terminated" in state and state["terminated"].get("exitCode", 0) != 0:
            t_reason = state["terminated"].get("reason", "Error")
            code = state["terminated"].get("exitCode")
            return f"{RED}[Exited {code}] {t_reason}{RESET}"

    if phase == "Running":
        raw_log = pod_logs_map.get(pname, "")
        lines = [l.strip() for l in raw_log.splitlines() if l.strip()]
        if lines:
            last_l = lines[-1]
            clean = re.sub(r'^[A-Z0-9:\.\- ]+\[[^\]]+\]\s*', '', last_l)
            if "Caching model weights" in clean:
                return f"{CYAN}⏳ Caching/verifying model weights...{RESET}"
            if "Waiting for Pod 0 cache barrier" in clean or "Waiting for Prefill Pod 0 cache barrier" in clean:
                return f"{YELLOW}⏸️  Waiting for Prefill Pod 0 cache barrier{RESET}"
            if "Cache barrier cleared" in clean:
                return f"{GREEN}⚡ Cache barrier cleared, initializing TPU{RESET}"
            if "Starting Ray Head" in clean or "Ray nodes connected" in clean:
                return f"{CYAN}🔄 Initializing Ray distributed mesh...{RESET}"
            if "Starting vLLM API Server" in clean:
                return f"{CYAN}🚀 Launching vLLM Engine on TPUs...{RESET}"
            if "Uvicorn running on" in clean or "Application startup complete" in clean:
                return f"{GREEN}✅ TPU Engine Serving & Healthy!{RESET}"
            if "Waiting for Serving endpoint" in clean:
                return f"{YELLOW}🔍 Probing backend endpoint health...{RESET}"
            if "Stage" in clean:
                return f"{GREEN}📊 {clean[:55]}{RESET}"
            return f"{DIM}{clean[:55]}{RESET}"
            
        return f"{GREEN}Running{RESET}"

    if phase == "Pending":
        evs = pod_events_map.get(pname, [])
        if evs:
            latest = sorted(evs, key=lambda e: e.get("lastTimestamp") or e.get("eventTime") or "")[-1]
            reason = latest.get("reason", "")
            msg = latest.get("message", "")
            if "exceeded quota" in msg.lower():
                return f"{RED}⚠️  Pending: TPU reservation quota exceeded{RESET}"
            if "TriggeredScaleUp" in reason:
                return f"{CYAN}🚀 Scale-up triggered: Provisioning new TPU node{RESET}"
            if "FailedScheduling" in reason:
                if "unschedulable" in msg or "didn't match" in msg:
                    return f"{YELLOW}⏳ Waiting for available TPU node matching topology{RESET}"
                return f"{YELLOW}⏳ Scheduling: {reason}{RESET}"
            return f"{DIM}[{reason}] {msg[:50]}{RESET}"
        return f"{YELLOW}⏳ Pending: Waiting for node allocation{RESET}"

    return f"{DIM}{phase}{RESET}"

print(f"\n{BOLD}[1] Pod Status & Cluster Placement ({len(pods_data)} Pods Total){RESET}")
header_fmt = f"{BOLD}{'Role':<22} | {'Pod Name':<34} | {'Status':<14} | {'Node':<28} | {'Live Progress / Diagnostic'}{RESET}"
print(header_fmt)
print(subsep)

pending_count = 0
running_count = 0

for p in pods_data:
    pname = p["metadata"]["name"]
    role, role_desc = get_role_info(pname, job_name)
    phase = p["status"].get("phase", "Unknown")
    node = p["spec"].get("nodeName") or "<none>"
    
    if phase == "Running": running_count += 1
    elif phase == "Pending": pending_count += 1
    
    status_color = GREEN if phase == "Running" else (YELLOW if phase == "Pending" else RED)
    status_str = f"{status_color}{phase:<9}{RESET}"
    
    diag = get_pod_diagnostic(p)
    role_fmt = truncate_str(role, 22)
    pname_fmt = truncate_str(pname, 34, mode="middle")
    node_fmt = truncate_str(node, 28)
    print(f"{role_fmt:<22} | {pname_fmt:<34} | {status_str:<23} | {node_fmt:<28} | {diag}")

print(subsep)
print(f"Summary: {GREEN}{running_count} Running{RESET}, {YELLOW}{pending_count} Pending{RESET} / {len(pods_data)} Total Pods")

# 3. Resource Usage Section
if top_metrics:
    print(f"\n{BOLD}[2] Real-Time Container Resource Metrics (kubectl top){RESET}")
    print(f"{BOLD}{'Pod Name':<34} | {'Container':<22} | {'CPU Cores':<14} | {'Memory (RAM)'}{RESET}")
    print(subsep)
    for p_name, c_dict in top_metrics.items():
        for c_name, (cpu, mem) in c_dict.items():
            c_color = CYAN if "sidecar" in c_name else (GREEN if "vllm" in c_name else RESET)
            p_name_fmt = truncate_str(p_name, 34, mode="middle")
            c_name_fmt = truncate_str(c_name, 22)
            cpu_fmt = truncate_str(cpu, 14)
            mem_fmt = truncate_str(mem, 16)
            print(f"{p_name_fmt:<34} | {c_color}{c_name_fmt:<22}{RESET} | {cpu_fmt:<14} | {mem_fmt}")
    print(subsep)

# 4. Cluster TPU Nodes & Autoscaler Analysis
print(f"\n{BOLD}[3] Cluster TPU v7x Hardware & Autoscaler Overview{RESET}")
try:
    raw_nodes = subprocess.check_output(["kubectl", "get", "nodes", "-l", "cloud.google.com/gke-tpu-accelerator=tpu7x", "-o", "json"], stderr=subprocess.DEVNULL).decode()
    nodes_list = json.loads(raw_nodes).get("items", [])
    
    topo_counts = {}
    for n in nodes_list:
        labels = n.get("metadata", {}).get("labels", {})
        topo = labels.get("cloud.google.com/gke-tpu-topology", "unknown")
        taints = n.get("spec", {}).get("taints", []) or []
        is_candidate = any("DeletionCandidate" in t.get("key", "") for t in taints)
        unsched = n.get("spec", {}).get("unschedulable", False)
        
        topo_counts.setdefault(topo, {"ready": 0, "draining": 0, "total": 0})
        topo_counts[topo]["total"] += 1
        if is_candidate or unsched:
            topo_counts[topo]["draining"] += 1
        else:
            topo_counts[topo]["ready"] += 1

    print(f"{BOLD}{'Topology':<16} | {'Total Nodes':<14} | {'Ready / Available':<20} | {'Draining / Scale-Down'}{RESET}")
    print(subsep)
    for topo, counts in sorted(topo_counts.items()):
        drain_str = f"{YELLOW}{counts['draining']} node(s){RESET}" if counts['draining'] > 0 else "0"
        print(f"{topo:<16} | {counts['total']:<14} | {GREEN}{counts['ready']} node(s){RESET}             | {drain_str}")
    print(subsep)
except Exception as e:
    print(f"{YELLOW}Unable to query cluster TPU nodes: {e}{RESET}")

# 5. Check Autoscaler Status
try:
    raw_as = subprocess.check_output(["kubectl", "get", "cm", "-n", "kube-system", "cluster-autoscaler-status", "-o", "json"], stderr=subprocess.DEVNULL).decode()
    as_status = json.loads(raw_as).get("data", {}).get("status", "")
    
    scale_down_candidates = re.findall(r'name:\s*(.*?nap-tpu7x.*?)\s*scaleDown:\s*candidates:\s*(\d+)', as_status, re.DOTALL)
    if scale_down_candidates:
        print(f"\n{BOLD}[4] Active Node Auto-Provisioning (NAP) Activity{RESET}")
        for mig, cand in scale_down_candidates:
            mig_short = mig.split("/")[-1]
            print(f" ⚠️  Scale-Down In Progress: {YELLOW}{mig_short}{RESET} has {RED}{cand} idle node(s){RESET} marked for deletion.")
except Exception:
    pass

print(f"{BOLD}{CYAN}{sep}{RESET}\n")
PYEOF
}

if [ "$WATCH_MODE" = true ]; then
    while true; do
        clear || true
        run_diagnostics
        echo -e "${DIM}Watching JobSet '${JOB_NAME}' (Refreshing every ${WATCH_INTERVAL}s... Press Ctrl+C to stop)${RESET}"
        sleep "${WATCH_INTERVAL}"
    done
else
    run_diagnostics
fi
