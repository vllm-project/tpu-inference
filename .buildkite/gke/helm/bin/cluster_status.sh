#!/usr/bin/env bash
# ==============================================================================
#  ⚡ GKE TPU Cluster Status & Queue Diagnostic Utility
#  Author: Advanced Agentic Assistant for TPU vLLM Inference
# ==============================================================================
# Usage:
#   ./gke/helm/bin/cluster_status.sh [OPTIONS]
#
# Examples:
#   ./gke/helm/bin/cluster_status.sh                         # Full cluster & queue status + Admission Guide
#   ./gke/helm/bin/cluster_status.sh -c 2x2x4                # Pre-flight check: Can I run a 2x2x4 job now?
#   ./gke/helm/bin/cluster_status.sh -c 32 -q default        # Pre-flight check for 32 chips in default queue
#   ./gke/helm/bin/cluster_status.sh -w                      # Watch mode (live refresh every 10s)
#   ./gke/helm/bin/cluster_status.sh -q default              # Filter by queue name
#   ./gke/helm/bin/cluster_status.sh -a                      # Anomalies and warnings only
# ==============================================================================

set -eo pipefail

WATCH_MODE=false
WATCH_INTERVAL=10
FILTER_QUEUE=""
ANOMALIES_ONLY=false
CHECK_TARGET=""

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
        -q|--queue)
            FILTER_QUEUE="$2"
            shift 2
            ;;
        -a|--anomalies-only)
            ANOMALIES_ONLY=true
            shift
            ;;
        -c|--check)
            CHECK_TARGET="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --check <TOPO|CHIPS>  Pre-flight check: Can a workload run now? (e.g. '2x2x4' or '32')"
            echo "  -q, --queue <NAME>        Target queue (default: 'default')"
            echo "  -w, --watch               Watch live status with periodic refresh (default: 10s)"
            echo "  -i, --interval <N>        Set refresh interval in seconds (default: 10)"
            echo "  -a, --anomalies-only      Show only cluster anomalies, failures, and warnings"
            echo "  -h, --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage instructions."
            exit 1
            ;;
    esac
done

run_cluster_diagnostics() {
python3 - "$FILTER_QUEUE" "$ANOMALIES_ONLY" "$CHECK_TARGET" << 'PYEOF'
import sys, json, subprocess, shutil, re, os, signal
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

# Ignore SIGPIPE for downstream pipes like `| head`
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

filter_queue = sys.argv[1].strip()
anomalies_only = sys.argv[2].strip().lower() == "true"
check_target = sys.argv[3].strip()

# Terminal colors
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

terminal_width = shutil.get_terminal_size((130, 24)).columns
sep = "=" * min(terminal_width, 130)
subsep = "-" * min(terminal_width, 130)

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

def parse_quantity(val_str):
    if not val_str:
        return 0.0
    val_str = str(val_str).strip()
    units = {
        "m": 1/1000.0, "k": 1000.0, "M": 10**6, "G": 10**9, "T": 10**12, "P": 10**15,
        "Ki": 1024.0, "Mi": 1024.0**2, "Gi": 1024.0**3, "Ti": 1024.0**4, "Pi": 1024.0**5
    }
    for u, mult in sorted(units.items(), key=lambda x: -len(x[0])):
        if val_str.endswith(u):
            try:
                return float(val_str[:-len(u)]) * mult
            except Exception:
                pass
    try:
        return float(val_str)
    except Exception:
        return 0.0

def format_quantity(val, is_bytes=False):
    if val <= 0:
        return "0"
    if is_bytes:
        for unit in ['B', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi']:
            if val < 1024.0:
                return f"{val:.1f}{unit}".rstrip('0').rstrip('.')
            val /= 1024.0
        return f"{val:.1f}Pi"
    if val >= 1000:
        if val >= 10**6:
            return f"{val/10**6:.1f}M".rstrip('0').rstrip('.')
        return f"{val/1000:.1f}k".rstrip('0').rstrip('.')
    if int(val) == val:
        return str(int(val))
    return f"{val:.2f}"

def run_cmd(cmd, timeout=15):
    """Executes a command and returns (json_obj, error_message)."""
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        if res.returncode != 0:
            return None, res.stderr.strip()
        return json.loads(res.stdout), None
    except subprocess.TimeoutExpired:
        return None, f"Command timed out after {timeout}s"
    except json.JSONDecodeError:
        return None, "Failed to parse JSON response"
    except Exception as e:
        return None, str(e)

# 0. Check Kubernetes Connectivity & Context
raw_ctx = ""
try:
    raw_ctx = subprocess.check_output(["kubectl", "config", "current-context"], stderr=subprocess.PIPE, text=True, timeout=5).strip()
except Exception as e:
    raw_ctx = "Unknown"

# Determine project from context
cluster_proj = "cloud-tpu-shared-capacity"
if "_" in raw_ctx:
    parts = raw_ctx.split("_")
    if len(parts) >= 2:
        cluster_proj = parts[1]

print(f"{BOLD}{CYAN}{sep}{RESET}")
print(f"{BOLD}{CYAN} ⚡ GKE TPU Cluster Status, Queue & Capacity Diagnostic Dashboard{RESET}")
print(f"{BOLD}{CYAN}{sep}{RESET}")
print(f" {BOLD}Context{RESET}       : {BLUE}{raw_ctx}{RESET}")
print(f" {BOLD}GCP Project{RESET}   : {CYAN}{cluster_proj}{RESET}")
print(f" {BOLD}Checked At{RESET}    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
target_queue_display = filter_queue if filter_queue else "default"
print(f" {BOLD}Target Queue{RESET}  : {YELLOW}{target_queue_display}{RESET}" + ("" if filter_queue else f" {DIM}(default){RESET}"))

# Pre-flight auth check
preflight_data, preflight_err = run_cmd(["kubectl", "get", "namespace", "default", "-o", "json"], timeout=8)
if preflight_err and ("gke-gcloud-auth-plugin" in preflight_err or "invalid_rapt" in preflight_err or "getting credentials" in preflight_err or "timed out" in preflight_err):
    print(f"\n{RED}❌ Kubernetes Authentication Required:{RESET}")
    print(f"{YELLOW}   gke-gcloud-auth-plugin could not authenticate with Google Cloud.{RESET}")
    print(f"{YELLOW}   Reason: Security key re-authentication (Titan/YubiKey) or session refresh required.{RESET}")
    print(f"\n{BOLD}   👉 Action required:{RESET} Please run:")
    print(f"      {CYAN}gcloud auth login{RESET}  (or touch your physical security key)")
    print(f"   Then re-run this script.\n")
    print(f"{BOLD}{CYAN}{sep}{RESET}\n")
    sys.exit(0)

# 1. Fetch Cluster & Cloud Resources in Parallel
query_tasks = {
    "cluster_queues": ["kubectl", "get", "clusterqueues", "-o", "json"],
    "local_queues": ["kubectl", "get", "localqueues", "-A", "-o", "json"],
    "workloads": ["kubectl", "get", "workloads", "-A", "-o", "json"],
    "nodes": ["kubectl", "get", "nodes", "-o", "json"],
    "pods": ["kubectl", "get", "pods", "-A", "-o", "json"],
    "events": ["kubectl", "get", "events", "-A", "--field-selector", "type=Warning", "-o", "json"],
    "gce_reservations": ["gcloud", "compute", "reservations", "list", f"--project={cluster_proj}", "--format=json"]
}

results = {}
with ThreadPoolExecutor(max_workers=7) as executor:
    future_to_key = {executor.submit(run_cmd, cmd): k for k, cmd in query_tasks.items()}
    for f in future_to_key:
        k = future_to_key[f]
        try:
            data, _ = f.result()
            results[k] = data
        except Exception:
            results[k] = None

cluster_queues_data = results.get("cluster_queues")
workloads_data = results.get("workloads")
nodes_data = results.get("nodes")
pods_data = results.get("pods")
events_data = results.get("events")
reservations_data = results.get("gce_reservations") or []

def parse_age(ts_str):
    if not ts_str:
        return "-"
    try:
        t = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        secs = int((now - t).total_seconds())
        if secs < 0:
            secs = 0
        if secs < 60:
            return f"{secs}s"
        elif secs < 3600:
            return f"{secs//60}m {secs%60}s"
        elif secs < 86400:
            return f"{secs//3600}h {(secs%3600)//60}m"
        else:
            return f"{secs//86400}d {(secs%86400)//3600}h"
    except Exception:
        return ts_str

# ==============================================================================
# SECTION 1: HARDWARE & CAPACITY OVERVIEW (Nodes, TPUs, Topologies)
# ==============================================================================
nodes = nodes_data.get("items", []) if nodes_data else []
pods = pods_data.get("items", []) if pods_data else []

total_nodes = len(nodes)
ready_nodes = 0
not_ready_nodes = []
cordoned_nodes = []
node_pressure_nodes = []

tpu_nodes_by_topo = {}
nodepool_vms = {}
node_reservations = {}

for n in nodes:
    n_name = n["metadata"]["name"]
    labels = n["metadata"].get("labels", {})
    spec = n.get("spec", {})
    status = n.get("status", {})
    
    conditions = {c["type"]: c["status"] for c in status.get("conditions", [])}
    if conditions.get("Ready") == "True":
        ready_nodes += 1
    else:
        not_ready_nodes.append((n_name, conditions.get("Ready", "Unknown")))
        
    for p in ["MemoryPressure", "DiskPressure", "PIDPressure"]:
        if conditions.get(p) == "True":
            node_pressure_nodes.append((n_name, p))
            
    if spec.get("unschedulable"):
        cordoned_nodes.append(n_name)
        
    tpu_topo = labels.get("cloud.google.com/gke-tpu-topology")
    nodepool = labels.get("cloud.google.com/gke-nodepool", "unknown-np")
    res_name = labels.get("cloud.google.com/reservation-name")
    
    if res_name:
        node_reservations[n_name] = res_name
        
    if tpu_topo:
        tpu_nodes_by_topo.setdefault(tpu_topo, []).append(n)
        nodepool_vms.setdefault(nodepool, {"topo": tpu_topo, "nodes": []})["nodes"].append(n_name)

# Map running pods to nodes
node_tpu_allocated = {}
for p in pods:
    p_status = p.get("status", {})
    p_spec = p.get("spec", {})
    node_name = p_spec.get("nodeName")
    phase = p_status.get("phase")
    if node_name and phase in ["Running", "Pending"]:
        for c in p_spec.get("containers", []):
            req_tpu = c.get("resources", {}).get("requests", {}).get("google.com/tpu") or c.get("resources", {}).get("limits", {}).get("google.com/tpu")
            if req_tpu:
                try:
                    node_tpu_allocated[node_name] = node_tpu_allocated.get(node_name, 0) + int(req_tpu)
                except Exception:
                    node_tpu_allocated[node_name] = node_tpu_allocated.get(node_name, 0) + 4

total_cluster_chips = 0
total_free_chips = 0
chips_per_vm = 4  # Standard TPU v7x VM has 4 chips
topo_stats = {}

for topo, t_nodes in sorted(tpu_nodes_by_topo.items()):
    total_vms = len(t_nodes)
    vms_allocated = sum(1 for n in t_nodes if node_tpu_allocated.get(n["metadata"]["name"], 0) > 0)
    vms_free = total_vms - vms_allocated
    
    dims = [int(x) for x in topo.split("x") if x.isdigit()]
    chips_per_slice = 1
    for d in dims:
        chips_per_slice *= d
    vms_per_slice = max(1, chips_per_slice // chips_per_vm)
    
    free_slices = vms_free // vms_per_slice
    topo_stats[topo] = {
        "total_vms": total_vms,
        "vms_free": vms_free,
        "chips_per_slice": chips_per_slice,
        "vms_per_slice": vms_per_slice,
        "free_slices": free_slices
    }

if not anomalies_only:
    print(f"\n{BOLD}🌐 1. Cluster Nodes & Physical Hardware Capacity{RESET}")
    print(subsep)
    print(f" Total Nodes: {BOLD}{total_nodes}{RESET} | Ready: {GREEN}{ready_nodes}{RESET} | NotReady: {RED if not_ready_nodes else GREEN}{len(not_ready_nodes)}{RESET} | Cordoned: {YELLOW if cordoned_nodes else GREEN}{len(cordoned_nodes)}{RESET}")
    
    print(f"\n {BOLD}{'TPU Topology':<15} {'Total VMs':<11} {'Physical Slices':<17} {'Allocated VMs':<15} {'Free VMs (Ready)':<18} {'Total Chips':<13} {'Free Chips'}{RESET}")
    print(f" {'-'*14:<15} {'-'*10:<11} {'-'*16:<17} {'-'*14:<15} {'-'*17:<18} {'-'*12:<13} {'-'*10}")
    
    for topo, t_nodes in sorted(tpu_nodes_by_topo.items()):
        st = topo_stats[topo]
        total_vms = st["total_vms"]
        vms_free = st["vms_free"]
        vms_allocated = total_vms - vms_free
        vms_per_slice = st["vms_per_slice"]
        
        num_slices = max(1, total_vms // vms_per_slice)
        slice_str = f"{num_slices} slice(s)" if vms_per_slice > 1 else f"{total_vms} host(s)"
        
        topo_total_chips = total_vms * chips_per_vm
        topo_free_chips = vms_free * chips_per_vm
        total_cluster_chips += topo_total_chips
        total_free_chips += topo_free_chips
        
        free_vm_color = GREEN if vms_free > 0 else DIM
        print(f" {CYAN}{topo:<15}{RESET} {total_vms:<11} {slice_str:<17} {vms_allocated:<15} {free_vm_color}{vms_free:<18}{RESET} {topo_total_chips:<13} {free_vm_color}{topo_free_chips}{RESET}")
        
    if not tpu_nodes_by_topo:
        print(f"  {DIM}(No active TPU nodes currently provisioned. NAP will scale up dynamically when jobs are admitted){RESET}")
    else:
        print(f" {'-'*14:<15} {'-'*10:<11} {'-'*16:<17} {'-'*14:<15} {'-'*17:<18} {'-'*12:<13} {'-'*10}")
        print(f" {BOLD}{'Total TPU':<15} {sum(len(v) for v in tpu_nodes_by_topo.values()):<11} {'-':<17} {'-':<15} {'-':<18} {total_cluster_chips:<13} {GREEN}{total_free_chips}{RESET}")

# ==============================================================================
# SECTION 2: GCE RESOURCE RESERVATIONS & HEADROOM
# ==============================================================================
total_reserved_chips = 0
total_inuse_chips = 0
total_left_chips = 0

for r in reservations_data:
    agg = r.get("aggregateReservation", {})
    spec_res = r.get("specificReservation", {})
    if agg:
        res_accels = agg.get("reservedResources", [{}])
        in_accels = agg.get("inUseResources", [{}])
        if res_accels and "accelerator" in res_accels[0]:
            total_reserved_chips += res_accels[0]["accelerator"].get("acceleratorCount", 0)
        if in_accels and "accelerator" in in_accels[0]:
            total_inuse_chips += in_accels[0]["accelerator"].get("acceleratorCount", 0)
    elif spec_res:
        count = int(spec_res.get("count", 0))
        in_use_cnt = int(spec_res.get("inUseCount", 0))
        g_accels = spec_res.get("instanceProperties", {}).get("guestAccelerators", [])
        acc_mult = g_accels[0].get("acceleratorCount", 1) if g_accels else 1
        total_reserved_chips += count * acc_mult
        total_inuse_chips += in_use_cnt * acc_mult

total_left_chips = max(0, total_reserved_chips - total_inuse_chips)

if not anomalies_only:
    print(f"\n{BOLD}🏷️  2. GCE TPU Resource Reservations & Headroom{RESET}")
    print(subsep)
    
    if not reservations_data:
        print(f" {DIM}No GCE compute reservations found in project {cluster_proj}.{RESET}")
    else:
        print(f" {BOLD}{'Reservation Name':<36} {'Zone':<14} {'Family / Accelerator':<24} {'Reserved':<10} {'In-Use':<10} {'Left':<10} {'Utilization'}{RESET}")
        print(f" {'-'*34:<36} {'-'*12:<14} {'-'*22:<24} {'-'*8:<10} {'-'*8:<10} {'-'*8:<10} {'-'*11}")
        
        for r in reservations_data:
            r_name = r.get("name", "Unknown")
            r_zone = r.get("zone", "").split("/")[-1]
            agg = r.get("aggregateReservation", {})
            spec_res = r.get("specificReservation", {})
            
            accel_desc = "Standard VM"
            in_use_val = 0
            reserved_val = 0
            
            if agg:
                vm_family = agg.get("vmFamily", "")
                short_family = "TPU7X" if "TPU7X" in vm_family else ("TPU" if "TPU" in vm_family else vm_family)
                res_accels = agg.get("reservedResources", [{}])
                in_accels = agg.get("inUseResources", [{}])
                accel_name = "tpu7x"
                if res_accels and "accelerator" in res_accels[0]:
                    accel_name = res_accels[0]["accelerator"].get("acceleratorType", "").split("/")[-1]
                    reserved_val = res_accels[0]["accelerator"].get("acceleratorCount", 0)
                if in_accels and "accelerator" in in_accels[0]:
                    in_use_val = in_accels[0]["accelerator"].get("acceleratorCount", 0)
                accel_desc = f"{accel_name} ({short_family})"
            elif spec_res:
                count = int(spec_res.get("count", 0))
                in_use_cnt = int(spec_res.get("inUseCount", 0))
                g_accels = spec_res.get("instanceProperties", {}).get("guestAccelerators", [])
                accel_type = g_accels[0].get("acceleratorType") if g_accels else spec_res.get("instanceProperties", {}).get("machineType", "VM")
                acc_mult = g_accels[0].get("acceleratorCount", 1) if g_accels else 1
                reserved_val = count * acc_mult
                in_use_val = in_use_cnt * acc_mult
                accel_desc = f"{accel_type}"
                
            left_val = max(0, reserved_val - in_use_val)
            util_pct = (in_use_val / reserved_val * 100) if reserved_val > 0 else 0.0
            
            r_name_fmt = truncate_str(r_name, 34, mode="middle")
            zone_fmt = truncate_str(r_zone, 13)
            desc_fmt = truncate_str(accel_desc, 23)
            
            util_color = GREEN if util_pct < 75 else (YELLOW if util_pct < 95 else RED)
            left_color = GREEN if left_val > 0 else RED
            
            print(f" {r_name_fmt:<36} {zone_fmt:<14} {desc_fmt:<24} {reserved_val:<10} {in_use_val:<10} {left_color}{left_val:<10}{RESET} {util_color}{util_pct:5.1f}%{RESET}")
            
        print(f" {'-'*34:<36} {'-'*12:<14} {'-'*22:<24} {'-'*8:<10} {'-'*8:<10} {'-'*8:<10} {'-'*11}")
        total_util = (total_inuse_chips / total_reserved_chips * 100) if total_reserved_chips > 0 else 0.0
        print(f" {BOLD}{'Total Reservations':<36} {'-':<14} {'-':<24} {total_reserved_chips:<10} {total_inuse_chips:<10} {GREEN}{total_left_chips:<10}{RESET} {total_util:5.1f}%{RESET}")
        
        diff_chips = total_inuse_chips - total_cluster_chips
        inflight_msg = f" ({abs(diff_chips)} chips booting/terminating in GCE)" if diff_chips != 0 else ""
        
        print(f"\n  {BOLD}💡 Capacity Insights:{RESET}")
        print(f"     • {BOLD}GCE Reservation In-Use{RESET} : {total_inuse_chips} chips allocated across GCE VMs.")
        print(f"     • {BOLD}K8s Nodes Registered{RESET}   : {sum(len(v) for v in tpu_nodes_by_topo.values())} VMs ({total_cluster_chips} chips){inflight_msg}.")
        print(f"     • {BOLD}Immediately Schedulable{RESET}: {GREEN}{total_free_chips} chips{RESET} on idle ready nodes (0s startup wait).")
        print(f"     • {BOLD}Reservation Headroom{RESET}   : {GREEN}{total_left_chips} chips{RESET} unallocated in GCE (available for NAP autoscaler).")

# ==============================================================================
# SECTION 3: KUEUE QUEUES, SHARED CAPACITY & REMAINING QUOTA
# ==============================================================================
cluster_queues = cluster_queues_data.get("items", []) if cluster_queues_data else []
queue_tpu_capacity = {}

if not anomalies_only:
    print(f"\n{BOLD}📊 3. Kueue Queues, Shared Capacity & Quota Limits{RESET}")
    print(subsep)
    
    if not cluster_queues:
        print(f" {YELLOW}⚠️ No Kueue ClusterQueues found in the cluster.{RESET}")
    else:
        for cq in cluster_queues:
            cq_name = cq["metadata"]["name"]
            cq_spec = cq.get("spec", {})
            cq_status = cq.get("status", {})
            cohort = cq_spec.get("cohortName") or cq_spec.get("cohort") or "None (No shared borrowing)"
            admitted_count = cq_status.get("admittedWorkloads", 0)
            pending_count = cq_status.get("pendingWorkloads", 0)
            
            resource_groups = cq_spec.get("resourceGroups", [])
            flavors_usage = {f.get("name"): f.get("resources", []) for f in cq_status.get("flavorsUsage", [])}
            
            q_tpu_left_nom = 0.0
            q_tpu_left_bor = 0.0
            
            # Print table if matched filter or no filter
            should_print = not filter_queue or filter_queue == cq_name
            if should_print:
                print(f" {BOLD}ClusterQueue{RESET}: {CYAN}{BOLD}{cq_name}{RESET} | Cohort: {MAGENTA}{cohort}{RESET} | Admitted: {GREEN}{admitted_count}{RESET} | Pending/Queued: {YELLOW if pending_count > 0 else GREEN}{pending_count}{RESET}")
                print(f"   {'Resource Flavor':<22} {'Resource':<18} {'Nominal':<12} {'Used':<12} {'Borrowed':<12} {'Left (Nominal)':<16} {'Left (w/ Borrow)'}{RESET}")
                print(f"   {'-'*20:<22} {'-'*16:<18} {'-'*10:<12} {'-'*10:<12} {'-'*10:<12} {'-'*14:<16} {'-'*16}")
            
            for rg in resource_groups:
                for flv in rg.get("flavors", []):
                    flv_name = flv.get("name")
                    flv_usage_list = flavors_usage.get(flv_name, [])
                    usage_map = {u.get("name"): u for u in flv_usage_list}
                    
                    for res in flv.get("resources", []):
                        r_name = res.get("name")
                        nominal_str = str(res.get("nominalQuota", "0"))
                        borrowing_limit_str = res.get("borrowingLimit")
                        
                        u_obj = usage_map.get(r_name, {})
                        used_str = str(u_obj.get("total", "0"))
                        borrowed_str = str(u_obj.get("borrowed", "0"))
                        
                        is_bytes = any(x in r_name.lower() for x in ["memory", "storage"])
                        nom_val = parse_quantity(nominal_str)
                        used_val = parse_quantity(used_str)
                        bor_val = parse_quantity(borrowed_str)
                        
                        left_nom_val = max(0.0, nom_val - used_val)
                        left_nom_fmt = format_quantity(left_nom_val, is_bytes)
                        
                        if borrowing_limit_str is not None:
                            bor_lim_val = parse_quantity(borrowing_limit_str)
                            left_bor_val = max(0.0, (nom_val + bor_lim_val) - used_val)
                            left_bor_fmt = format_quantity(left_bor_val, is_bytes)
                        else:
                            left_bor_val = left_nom_val
                            left_bor_fmt = "Unlimited" if cohort != "None (No shared borrowing)" else left_nom_fmt
                            if cohort != "None (No shared borrowing)":
                                left_bor_val = 9999.0
                                
                        if r_name == "google.com/tpu":
                            q_tpu_left_nom = left_nom_val
                            q_tpu_left_bor = left_bor_val
                            
                        if should_print:
                            used_color = YELLOW if used_val > 0 else DIM
                            left_color = GREEN if left_nom_val > 0 else (YELLOW if left_bor_fmt != "0" else RED)
                            flv_fmt = truncate_str(flv_name, 21)
                            res_fmt = truncate_str(r_name, 17)
                            nom_fmt = truncate_str(nominal_str, 11)
                            usd_fmt = truncate_str(used_str, 11)
                            bor_fmt = truncate_str(borrowed_str, 11)
                            print(f"   {flv_fmt:<22} {res_fmt:<18} {nom_fmt:<12} {used_color}{usd_fmt:<12}{RESET} {bor_fmt:<12} {left_color}{left_nom_fmt:<16}{RESET} {left_bor_fmt}")
                            
            queue_tpu_capacity[cq_name] = {
                "cohort": cohort,
                "left_nominal": int(q_tpu_left_nom),
                "left_borrow": int(q_tpu_left_bor)
            }
            if should_print:
                print("")

# ==============================================================================
# SECTION 4: WORKLOAD ADMISSION & SCHEDULING GUIDE (CAN I RUN NOW?)
# ==============================================================================
active_eval_queue = filter_queue if filter_queue in queue_tpu_capacity else "default"
q_info = queue_tpu_capacity.get(active_eval_queue, {"cohort": "None", "left_nominal": 0, "left_borrow": 0})
q_effective_tpu = q_info["left_borrow"] if q_info["cohort"] != "None (No shared borrowing)" else q_info["left_nominal"]

# If user invoked --check <TOPO|CHIPS|FILE>
if check_target:
    print(f"\n{BOLD}{CYAN}🔍 PRE-FLIGHT ADMISSION CHECK for '{check_target}' in queue '{active_eval_queue}':{RESET}")
    print(subsep)
    
    components = []
    
    # Check if target is a Helm values YAML file
    if os.path.isfile(check_target) and (check_target.endswith(".yaml") or check_target.endswith(".yml")):
        try:
            import yaml
            with open(check_target, "r") as f:
                v_data = yaml.safe_load(f)
            v_tpu = v_data.get("tpu", {}) or {}
            default_topo = v_tpu.get("topology", "2x2x1")

            def chips_of(topo):
                n = 1
                for d in topo.split("x"):
                    if d.isdigit():
                        n *= int(d)
                return n

            # A JobSet is a single Kueue workload: every step's podSet needs a
            # flavor assigned before any of them starts, even though the chart
            # runs them InOrder. So the figure that decides admission is the SUM
            # over all steps, not the largest one.
            for job in (v_data.get("scriptJobs") or []):
                j_topo = ((job.get("tpu") or {}).get("topology")) or default_topo
                components.append({"role": job.get("name", "step"),
                                   "topo": j_topo,
                                   "count": 1,
                                   "chips_per_slice": chips_of(j_topo)})
        except Exception as e:
            components = []
            
    if not components:
        # Parse multi-topology expression: "1x2x2x1, 2x2x2", "2x2x4 + 2x2x2x4", "96"
        parts = re.split(r'[,+;]\s*', check_target.strip())
        for idx, p in enumerate(parts):
            p = p.strip()
            if not p: continue
            m = re.match(r'^(?:(\d+)\s*[*x]\s*)?([1-9]\d*x[1-9]\d*x[1-9]\d*)$', p)
            if m:
                count = int(m.group(1)) if m.group(1) else 1
                topo = m.group(2)
                dims = [int(x) for x in topo.split("x")]
                chips = dims[0] * dims[1] * dims[2]
                role = f"Slice-{idx+1}" if len(parts) > 1 else "Workload"
                components.append({"role": role, "topo": topo, "count": count, "chips_per_slice": chips})
            else:
                try:
                    c = int(p)
                    components.append({"role": f"Slice-{idx+1}", "topo": "custom", "count": 1, "chips_per_slice": c})
                except Exception:
                    pass

    if not components:
        components.append({"role": "Workload", "topo": "2x2x1", "count": 1, "chips_per_slice": 4})

    total_req_chips = sum(c["count"] * c["chips_per_slice"] for c in components)
    total_scaleup_chips = 0
    all_instant_ready = True
    
    print(f" {BOLD}{'Component':<15} {'Topology':<12} {'Count':<8} {'Chips/Slice':<14} {'Total Chips':<14} {'Idle Slices in GKE':<20} {'Action Required'}{RESET}")
    print(f" {'-'*13:<15} {'-'*10:<12} {'-'*6:<8} {'-'*12:<14} {'-'*12:<14} {'-'*18:<20} {'-'*28}")
    
    for c in components:
        role = c["role"]
        topo = c["topo"]
        count = c["count"]
        chips_per_s = c["chips_per_slice"]
        tot_c = count * chips_per_s
        
        free_s = topo_stats.get(topo, {}).get("free_slices", 0) if topo in topo_stats else 0
        slices_ready = min(count, free_s)
        slices_needed = count - slices_ready
        scaleup_for_c = slices_needed * chips_per_s
        total_scaleup_chips += scaleup_for_c
        
        if slices_needed > 0:
            all_instant_ready = False
            action_desc = f"{YELLOW}🟡 {slices_ready} ready, {slices_needed} need NAP scale-up{RESET}" if slices_ready > 0 else f"{YELLOW}🟡 Need NAP scale-up ({slices_needed} slice){RESET}"
        else:
            action_desc = f"{GREEN}🟢 All {count} slice(s) ready now{RESET}"
            
        idle_str = f"{free_s} slice(s) free" if topo in topo_stats else f"{DIM}0 booted{RESET}"
        print(f" {CYAN}{role:<15}{RESET} {topo:<12} {count:<8} {chips_per_s:<14} {tot_c:<14} {idle_str:<20} {action_desc}")
        
    print(f" {'-'*13:<15} {'-'*10:<12} {'-'*6:<8} {'-'*12:<14} {'-'*12:<14} {'-'*18:<20} {'-'*28}")
    print(f" {BOLD}{'Total Workload':<15} {'-':<12} {sum(c['count'] for c in components):<8} {'-':<14} {BOLD}{total_req_chips:<14}{RESET} {'-':<20} (Scale-up needed: {total_scaleup_chips} chips)")

    print(f"\n {BOLD}Quota & Hardware Verification:{RESET}")
    kueue_pass = total_req_chips <= q_effective_tpu
    kueue_msg = f"{GREEN}✅ PASS{RESET} ({q_effective_tpu} chips available in queue quota >= {total_req_chips} chips requested)" if kueue_pass else f"{RED}❌ FAIL{RESET} (Requires {total_req_chips} chips, but queue only has {q_effective_tpu} left)"
    
    gce_pass = total_scaleup_chips <= total_left_chips
    gce_msg = f"{GREEN}✅ PASS{RESET} ({total_left_chips} chips headroom in GCE reservation >= {total_scaleup_chips} chips to scale-up)" if gce_pass else f"{RED}❌ NO{RESET} (Requires {total_scaleup_chips} chips to scale up, but GCE reservation only has {total_left_chips} left)"

    print(f" [1] Kueue Queue Quota   : {kueue_msg}")
    print(f" [2] GCE Cloud Headroom  : {gce_msg}")
    
    print(f"\n {BOLD}🎯 VERDICT:{RESET}")
    if not kueue_pass:
        print(f"   {RED}⏳ WILL BE QUEUED (Pending){RESET}: Queue '{active_eval_queue}' does not have enough quota.")
    elif all_instant_ready:
        print(f"   {GREEN}🟢 INSTANT START (0s Wait){RESET}: All requested slices exist as idle nodes in the cluster. Workload will run immediately!")
    elif gce_pass:
        print(f"   {YELLOW}🟡 SCALE-UP START (~2-3m Wait){RESET}: Some slices need provisioning, but GCE reservation has enough headroom ({total_left_chips} >= {total_scaleup_chips} chips). GKE NAP will create the node pools.")
    else:
        print(f"   {RED}⏳ WILL BE QUEUED / PARTIALLY BLOCKED{RESET}: GCE Reservation has only {total_left_chips} chips headroom, but you need {total_scaleup_chips} chips to provision missing slices.")
        print(f"   👉 {BOLD}Recommendation:{RESET} Wait for running jobs to finish and release at least {total_scaleup_chips - total_left_chips} more chips, or adjust slice count.")
    print("")

if not anomalies_only:
    print(f"\n{BOLD}🚦 4. Workload Admission & Scheduling Guide (Can I submit a job now?){RESET}")
    print(subsep)
    print(f" Target Queue: {BOLD}{CYAN}{active_eval_queue}{RESET} (Cohort: {MAGENTA}{q_info['cohort']}{RESET} | Queue Quota Left: {GREEN}{q_effective_tpu} chips{RESET} | GCE Headroom: {GREEN}{total_left_chips} chips{RESET})")
    
    instant_topos = []
    autoscale_topos = []
    blocked_topos = []
    
    standard_topos = ["2x2x1", "2x2x2", "2x2x4", "4x4x4", "4x4x8"]
    all_topos = sorted(set(list(topo_stats.keys()) + standard_topos))
    
    for t in all_topos:
        dims = [int(x) for x in t.split("x") if x.isdigit()]
        t_chips = 1
        for d in dims: t_chips *= d
        
        st = topo_stats.get(t, {"free_slices": 0})
        free_s = st["free_slices"]
        
        if free_s > 0 and t_chips <= q_effective_tpu:
            instant_topos.append((t, t_chips, free_s))
        elif t_chips <= min(q_effective_tpu, total_left_chips):
            autoscale_topos.append((t, t_chips))
        else:
            blocked_topos.append((t, t_chips))
            
    print(f"\n  {GREEN}🟢 INSTANT START (0s Wait - Ready Slices Available Now):{RESET}")
    if instant_topos:
        for t, c, s in instant_topos:
            print(f"     • {BOLD}{t:<8}{RESET} ({c:>3} chips) : {GREEN}{s} physical slice(s) ready{RESET}  -->  Admitted and scheduled immediately.")
    else:
        print(f"     {DIM}(No fully idle slices currently booted. Jobs will scale up or queue).{RESET}")
        
    print(f"\n  {YELLOW}🟡 SCALE-UP START (~2-3m Wait via GKE Node Auto-Provisioning):{RESET}")
    if autoscale_topos:
        for t, c in autoscale_topos:
            print(f"     • {BOLD}{t:<8}{RESET} ({c:>3} chips) : Within GCE reservation headroom ({total_left_chips} chips left) --> NAP will auto-create node pool.")
    else:
        print(f"     {DIM}(None - GCE Reservation has only {total_left_chips} chips headroom remaining).{RESET}")
        
    print(f"\n  {RED}🔴 WILL BE QUEUED (Pending - Must wait for running jobs to finish):{RESET}")
    if blocked_topos:
        for t, c in blocked_topos:
            reason = f"exceeds GCE headroom ({total_left_chips} chips left)" if t_chips > total_left_chips else "exceeds queue quota"
            print(f"     • {BOLD}{t:<8}{RESET} ({c:>3} chips) : ⏳ Queued ({reason}).")

# ==============================================================================
# SECTION 5 & 6: WORKLOADS BREAKDOWN (QUEUED & ADMITTED)
# ==============================================================================
workloads = workloads_data.get("items", []) if workloads_data else []

queued_workloads = []
admitted_workloads = []
failed_evicted_workloads = []

for wl in workloads:
    w_name = wl["metadata"]["name"]
    w_ns = wl["metadata"].get("namespace", "default")
    w_spec = wl.get("spec", {})
    w_status = wl.get("status", {})
    queue_name = w_spec.get("queueName", "default")
    
    if filter_queue and filter_queue != queue_name:
        continue
        
    created_at = wl["metadata"].get("creationTimestamp")
    age_str = parse_age(created_at)
    priority_class = w_spec.get("priorityClassName", "default")
    priority_score = w_spec.get("priority", 0)
    
    tpu_requested = 0
    for ps in w_spec.get("podSets", []):
        count = ps.get("count", 1)
        pod_spec = ps.get("template", {}).get("spec", {})
        for c in pod_spec.get("containers", []):
            res = c.get("resources", {}).get("requests", {}) or c.get("resources", {}).get("limits", {})
            t_req = res.get("google.com/tpu")
            if t_req:
                try:
                    tpu_requested += int(t_req) * count
                except Exception:
                    tpu_requested += 4 * count
                
    conditions = {c.get("type"): (c.get("status"), c.get("reason"), c.get("message")) for c in w_status.get("conditions", [])}
    is_admitted = conditions.get("Admitted", ("False", "", ""))[0] == "True"
    is_finished = conditions.get("Finished", ("False", "", ""))[0] == "True"
    is_evicted = conditions.get("Evicted", ("False", "", ""))[0] == "True"
    
    parent_job = "Unknown"
    owner_refs = wl["metadata"].get("ownerReferences", [])
    if owner_refs:
        parent_job = f"{owner_refs[0].get('kind', 'Job')}/{owner_refs[0].get('name')}"
    else:
        parent_job = w_name
        
    wl_info = {
        "name": w_name,
        "namespace": w_ns,
        "parent": parent_job,
        "queue": queue_name,
        "priority": f"{priority_class} ({priority_score})",
        "tpu_requested": tpu_requested,
        "age": age_str,
        "conditions": conditions
    }
    
    if is_finished:
        continue
    elif is_evicted:
        failed_evicted_workloads.append(wl_info)
    elif is_admitted:
        admitted_workloads.append(wl_info)
    else:
        queued_workloads.append(wl_info)

if not anomalies_only:
    print(f"\n{BOLD}⏳ 5. Queued & Pending Workloads (Waiting in Queues){RESET}")
    print(subsep)
    if not queued_workloads:
        print(f" {GREEN}✅ No queued workloads waiting in queue. All submitted workloads are admitted.{RESET}")
    else:
        print(f" {BOLD}{'Workload / Parent Job':<38} {'Queue':<15} {'Priority':<16} {'TPU Req':<10} {'Wait Age':<11} {'Pending Reason / Status'}{RESET}")
        print(f" {'-'*36:<38} {'-'*13:<15} {'-'*14:<16} {'-'*8:<10} {'-'*9:<11} {'-'*35}")
        for qw in queued_workloads:
            msg = "Waiting for quota"
            conds = qw["conditions"]
            if "QuotaReserved" in conds:
                msg = f"{conds['QuotaReserved'][1]}: {conds['QuotaReserved'][2]}"
            elif "Admitted" in conds:
                msg = f"{conds['Admitted'][1]}: {conds['Admitted'][2]}"
            
            p_display = truncate_str(qw['parent'], 36, mode="middle")
            q_display = truncate_str(qw['queue'], 14)
            pr_display = truncate_str(qw['priority'], 15)
            tpu_display = truncate_str(str(qw['tpu_requested']), 8)
            age_display = truncate_str(qw['age'], 10)
            msg_display = truncate_str(msg, 55)
            
            print(f" {YELLOW}{p_display:<38}{RESET} {q_display:<15} {pr_display:<16} {BOLD}{tpu_display:<10}{RESET} {age_display:<11} {DIM}{msg_display}{RESET}")

    print(f"\n{BOLD}🚀 6. Admitted & Running Workloads{RESET}")
    print(subsep)
    if not admitted_workloads:
        print(f" {DIM}No workloads currently active/admitted in queues.{RESET}")
    else:
        print(f" {BOLD}{'Workload / Parent Job':<38} {'Namespace':<12} {'Queue':<15} {'Priority':<16} {'TPUs':<8} {'Running Age'}{RESET}")
        print(f" {'-'*36:<38} {'-'*10:<12} {'-'*13:<15} {'-'*14:<16} {'-'*6:<8} {'-'*12}")
        for aw in admitted_workloads:
            p_display = truncate_str(aw['parent'], 36, mode="middle")
            ns_display = truncate_str(aw['namespace'], 11)
            q_display = truncate_str(aw['queue'], 14)
            pr_display = truncate_str(aw['priority'], 15)
            tpu_display = truncate_str(str(aw['tpu_requested']), 6)
            age_display = truncate_str(aw['age'], 11)
            
            print(f" {GREEN}{p_display:<38}{RESET} {ns_display:<12} {q_display:<15} {pr_display:<16} {BOLD}{tpu_display:<8}{RESET} {age_display}")

# ==============================================================================
# SECTION 7: CLUSTER ANOMALIES & HEALTH DIAGNOSTICS
# ==============================================================================
print(f"\n{BOLD}⚠️  7. Cluster Anomalies & Diagnostic Warnings{RESET}")
print(subsep)

anomalies_detected = 0

# 7a. Node Anomalies
if not_ready_nodes:
    anomalies_detected += len(not_ready_nodes)
    print(f"\n {RED}❌ Unhealthy Nodes (Not Ready):{RESET}")
    for n_name, r_state in not_ready_nodes:
        n_fmt = truncate_str(n_name, 35)
        print(f"   - {BOLD}{n_fmt}{RESET}: Ready={RED}{r_state}{RESET}")

if node_pressure_nodes:
    anomalies_detected += len(node_pressure_nodes)
    print(f"\n {YELLOW}⚠️ Nodes Under Pressure:{RESET}")
    for n_name, p_cond in node_pressure_nodes:
        n_fmt = truncate_str(n_name, 35)
        print(f"   - {BOLD}{n_fmt}{RESET}: Condition={YELLOW}{p_cond}{RESET}")

# 7b. Check Incomplete Multi-Host TPU Slices
for np_name, np_info in nodepool_vms.items():
    topo = np_info["topo"]
    vms = np_info["nodes"]
    dims = [int(x) for x in topo.split("x") if x.isdigit()]
    chips = 1
    for d in dims:
        chips *= d
    expected_vms = max(1, chips // chips_per_vm)
    if expected_vms > 1 and len(vms) != expected_vms:
        anomalies_detected += 1
        np_fmt = truncate_str(np_name, 35)
        print(f"\n {RED}❌ Broken / Incomplete TPU Slice Detected:{RESET}")
        print(f"   - Node Pool: {BOLD}{np_fmt}{RESET} (Topology: {CYAN}{topo}{RESET})")
        print(f"     Expected {expected_vms} VMs, but only found {len(vms)} VMs: {vms}")
        print(f"     {YELLOW}Impact: Multi-host TPU jobs scheduled across incomplete slices will crash with SLICE_FAILURE_CHIP_DRIVER_ERROR!{RESET}")

# 7c. Pod Anomalies
pod_anomalies = []
for p in pods:
    p_name = p["metadata"]["name"]
    p_ns = p["metadata"].get("namespace", "default")
    p_phase = p.get("status", {}).get("phase", "Unknown")
    c_statuses = p.get("status", {}).get("containerStatuses", [])
    init_c_statuses = p.get("status", {}).get("initContainerStatuses", [])
    
    is_anomaly = False
    reason_str = ""
    
    for cs in init_c_statuses + c_statuses:
        waiting = cs.get("state", {}).get("waiting", {})
        terminated = cs.get("state", {}).get("terminated", {})
        
        if waiting:
            w_reason = waiting.get("reason", "")
            if w_reason in ["CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull", "CreateContainerConfigError"]:
                is_anomaly = True
                reason_str = f"{w_reason} ({waiting.get('message', '')})"
                break
        if terminated:
            t_reason = terminated.get("reason", "")
            exit_code = terminated.get("exitCode", 0)
            if exit_code != 0 and t_reason not in ["Completed"]:
                is_anomaly = True
                reason_str = f"{t_reason} (ExitCode {exit_code})"
                break
                
    if p_phase == "Failed":
        is_anomaly = True
        if not reason_str:
            reason_str = p.get("status", {}).get("message", "Pod execution failed")
            
    if is_anomaly:
        pod_anomalies.append((p_ns, p_name, reason_str))

if pod_anomalies:
    anomalies_detected += len(pod_anomalies)
    print(f"\n {RED}❌ Abnormal Pods ({len(pod_anomalies)}):{RESET}")
    print(f"   {'Namespace':<15} {'Pod Name':<42} {'Failure Reason'}")
    print(f"   {'-'*13:<15} {'-'*40:<42} {'-'*40}")
    for p_ns, p_name, r_str in pod_anomalies[:12]:
        p_ns_fmt = truncate_str(p_ns, 14)
        p_name_fmt = truncate_str(p_name, 40, mode="middle")
        r_fmt = truncate_str(r_str, 55)
        print(f"   {p_ns_fmt:<15} {RED}{p_name_fmt:<42}{RESET} {r_fmt}")
    if len(pod_anomalies) > 12:
        print(f"   {DIM}... and {len(pod_anomalies) - 12} more abnormal pods.{RESET}")

# 7d. Evicted / Failed Workloads
if failed_evicted_workloads:
    anomalies_detected += len(failed_evicted_workloads)
    print(f"\n {RED}❌ Evicted or Terminated Workloads ({len(failed_evicted_workloads)}):{RESET}")
    for fw in failed_evicted_workloads:
        ev_msg = fw["conditions"].get("Evicted", ("", "", "Evicted by Kueue"))[2]
        fw_name_fmt = truncate_str(fw['name'], 32, mode="middle")
        fw_parent_fmt = truncate_str(fw['parent'], 35, mode="middle")
        ev_msg_fmt = truncate_str(ev_msg, 65)
        print(f"   - Workload: {BOLD}{fw_name_fmt}{RESET} ({fw_parent_fmt}) | Reason: {RED}{ev_msg}{RESET}")

# 7e. Recent Warning Events
warning_events = events_data.get("items", []) if events_data else []
if warning_events:
    grouped_warns = {}
    for ev in warning_events:
        reason = ev.get("reason", "Warning")
        obj_kind = ev.get("involvedObject", {}).get("kind", "")
        obj_name = ev.get("involvedObject", {}).get("name", "")
        msg = ev.get("message", "").strip()
        grouped_warns.setdefault((reason, obj_kind), []).append((obj_name, msg, ev.get("lastTimestamp")))
        
    print(f"\n {YELLOW}🔔 Recent Warning Events ({len(warning_events)} total):{RESET}")
    print(f"   {'Reason':<25} {'Target Kind':<14} {'Count':<8} {'Sample Target & Message'}")
    print(f"   {'-'*23:<25} {'-'*12:<14} {'-'*6:<8} {'-'*45}")
    for (r_name, o_kind), ev_list in sorted(grouped_warns.items(), key=lambda x: len(x[1]), reverse=True)[:8]:
        sample_name, sample_msg, ts = ev_list[0]
        r_name_fmt = truncate_str(r_name, 23)
        o_kind_fmt = truncate_str(o_kind, 12)
        sample_fmt = truncate_str(f"{sample_name}: {sample_msg}", 65)
        print(f"   {YELLOW}{r_name_fmt:<25}{RESET} {o_kind_fmt:<14} {len(ev_list):<8} {DIM}{sample_fmt}{RESET}")

if anomalies_detected == 0:
    print(f"\n {GREEN}🎉 No cluster anomalies detected! All nodes and workloads are healthy.{RESET}")
else:
    print(f"\n {YELLOW}⚠️ Total Anomaly Signals: {BOLD}{anomalies_detected}{RESET}")

print(f"\n{BOLD}{CYAN}{sep}{RESET}\n")

PYEOF
}

if [ "$WATCH_MODE" = true ]; then
    while true; do
        clear
        run_cluster_diagnostics
        echo -e "\033[2mWatching live cluster status (Refresh interval: ${WATCH_INTERVAL}s). Press Ctrl+C to exit...\033[0m"
        sleep "$WATCH_INTERVAL"
    done
else
    run_cluster_diagnostics
fi
