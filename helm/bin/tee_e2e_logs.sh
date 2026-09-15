#!/usr/bin/env bash
# ==============================================================================
# TPU vLLM E2E Pytest Log Streaming, Capture & Report Utility
# Streams & tees logs from e2e (e2e-pytest container)
# into log/e2e.log<number> and automatically retrieves /tmp/e2e_report.xml
# ==============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Resolve log directory (prefers repo root log/ if symlinked, or helm/bin/log)
if [ -d "${REPO_ROOT}/log" ]; then
    LOG_DIR="${REPO_ROOT}/log"
elif [ -d "${SCRIPT_DIR}/log" ]; then
    LOG_DIR="${SCRIPT_DIR}/log"
else
    LOG_DIR="${SCRIPT_DIR}/log"
    mkdir -p "${LOG_DIR}"
fi

CURRENT_USER="${USER:-$(whoami)}"
CLEAN_USER=$(echo "$CURRENT_USER" | tr '[:upper:]' '[:lower:]' | tr -dc 'a-z0-9')

JOB_NAME=""
LOG_NUM=""
MODE="stream"      # "stream" | "dump" | "summary"
FOLLOW=true

# Terminal colors
BOLD="\033[1m"
DIM="\033[2m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
CYAN="\033[36m"
MAGENTA="\033[35m"
RESET="\033[0m"

usage() {
    echo -e "${BOLD}${CYAN}==================================================================${RESET}"
    echo -e "${BOLD}${CYAN} ⚡ TPU vLLM E2E Pytest Log Streaming & Test Report Utility${RESET}"
    echo -e "${BOLD}${CYAN}==================================================================${RESET}"
    echo "Usage: $0 [options] [JOB_NAME] [LOG_NUMBER]"
    echo ""
    echo "Arguments (can be passed positionally in any intuitive order):"
    echo "  JOB_NAME            JobSet / Release name (e.g. ${CLEAN_USER}-tc-abcde)."
    echo "                      If omitted, automatically detects latest active E2E JobSet."
    echo "  LOG_NUMBER          Numeric suffix for log files (e.g. 1 -> e2e.log1)."
    echo "                      If omitted, auto-increments to next available number."
    echo ""
    echo "Options:"
    echo "  -j, --job <NAME>       Specify JobSet name explicitly"
    echo "  -n, --number <NUM>     Specify log number suffix explicitly"
    echo "  -s, --save, --dump     Snapshot/dump current logs from pods immediately without following (-f)"
    echo "  -S, --summary          Generate and display test execution summary report"
    echo "  -o, --dir <DIR>        Custom output directory (default: ${LOG_DIR})"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  1. Stream active E2E pytest job (auto-detect job & next log number):"
    echo "     $0"
    echo ""
    echo "  2. Stream into log/e2e.log2 and fetch log/e2e_report.log2.xml:"
    echo "     $0 2"
    echo ""
    echo "  3. Stream a specific E2E job into log/e2e.log1:"
    echo "     $0 my-tc-release 1"
    echo ""
    echo "  4. Snapshot current logs immediately from a finished or running job:"
    echo "     $0 --dump"
    echo ""
    echo "  5. Parse and display summary of existing test run #1:"
    echo "     $0 --summary 1"
    echo -e "${BOLD}${CYAN}==================================================================${RESET}"
    exit 0
}

# --- PARSE ARGUMENTS ---
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--job)
            JOB_NAME="$2"
            shift 2
            ;;
        -n|--number)
            LOG_NUM="$2"
            shift 2
            ;;
        -s|--save|--dump)
            MODE="dump"
            FOLLOW=false
            shift
            ;;
        -S|--summary)
            MODE="summary"
            FOLLOW=false
            shift
            ;;
        -o|--dir)
            LOG_DIR="$2"
            mkdir -p "${LOG_DIR}"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

# Process positional arguments intuitively
for arg in "${POSITIONAL[@]}"; do
    if [[ "$arg" =~ ^[0-9]+$ ]] && [ -z "$LOG_NUM" ]; then
        LOG_NUM="$arg"
    elif [ -z "$JOB_NAME" ]; then
        JOB_NAME="$arg"
    fi
done

# --- AUTO-DISCOVER ACTIVE JOBSET IF NOT PROVIDED ---
if [ -z "$JOB_NAME" ]; then
    echo -e "${CYAN}🔍 Detecting latest active E2E JobSet...${RESET}"
    # Prefer current user's tc/test jobs
    JOB_NAME=$(kubectl get jobset -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null | grep -E "^(${CLEAN_USER}-tc-|${CLEAN_USER}-test-)" | tail -n 1 || true)

    if [ -z "$JOB_NAME" ]; then
        # Check any jobset with an e2e replicatedJob
        JOB_NAME=$(kubectl get pods -l "jobset.sigs.k8s.io/replicatedjob-name=e2e" -o jsonpath='{range .items[*]}{.metadata.labels.jobset\.sigs\.k8s\.io/jobset-name}{"\n"}{end}' 2>/dev/null | sort -u | tail -n 1 || true)
    fi

    if [ -z "$JOB_NAME" ]; then
        # Fallback to any latest jobset
        JOB_NAME=$(kubectl get jobset -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null || true)
    fi

    if [ -z "$JOB_NAME" ]; then
        echo -e "${RED}❌ Error: No active JobSets found on cluster. Please specify JobSet name explicitly.${RESET}"
        exit 1
    fi
    echo -e "   Found JobSet: ${YELLOW}${JOB_NAME}${RESET}"
fi

# --- AUTO-DISCOVER NEXT LOG NUMBER IF NOT PROVIDED ---
if [ -z "$LOG_NUM" ]; then
    MAX_NUM=$(ls -1 "${LOG_DIR}"/e2e.log* "${LOG_DIR}"/e2e_report.log*.xml 2>/dev/null | grep -o '[0-9]\+' | sort -n | tail -n 1 || true)
    if [ -z "$MAX_NUM" ]; then
        LOG_NUM=1
    else
        LOG_NUM=$((MAX_NUM + 1))
    fi
fi

# Define target files
E2E_LOG="${LOG_DIR}/e2e.log${LOG_NUM}"
REPORT_XML="${LOG_DIR}/e2e_report.log${LOG_NUM}.xml"
SUMMARY_FILE="${LOG_DIR}/e2e_summary.log${LOG_NUM}"

# --- FUNCTION: GENERATE SUMMARY REPORT ---
generate_summary() {
    python3 - "${JOB_NAME}" "${LOG_NUM}" "${E2E_LOG}" "${REPORT_XML}" "${SUMMARY_FILE}" << 'PYEOF'
import sys, os, xml.etree.ElementTree as ET, re

job_name = sys.argv[1]
log_num = sys.argv[2]
e2e_log = sys.argv[3]
report_xml = sys.argv[4]
summary_file = sys.argv[5]

# Terminal styles
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"

sep = "=" * 80
subsep = "-" * 80

out = []
out.append(sep)
out.append(f" ⚡ TPU vLLM E2E Pytest Execution Summary")
out.append(sep)
out.append(f" JobSet Name     : {job_name}")
out.append(f" Log Suffix Index: {log_num}")
out.append(f" Log File        : {e2e_log}")
if os.path.isfile(report_xml):
    out.append(f" JUnit XML Report: {report_xml}")
out.append(subsep)

has_report = False
if os.path.isfile(report_xml) and os.path.getsize(report_xml) > 0:
    try:
        tree = ET.parse(report_xml)
        root = tree.getroot()
        
        # Testsuite attributes
        suites = root.findall(".//testsuite")
        if not suites and root.tag == "testsuite":
            suites = [root]

        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_skipped = 0
        total_time = 0.0

        testcases = []
        for suite in suites:
            total_tests += int(suite.attrib.get("tests", 0))
            total_failures += int(suite.attrib.get("failures", 0))
            total_errors += int(suite.attrib.get("errors", 0))
            total_skipped += int(suite.attrib.get("skipped", 0))
            total_time += float(suite.attrib.get("time", 0.0))

            for tc in suite.findall("testcase"):
                classname = tc.attrib.get("classname", "")
                name = tc.attrib.get("name", "")
                t_duration = float(tc.attrib.get("time", 0.0))
                
                status = "PASS"
                err_msg = ""
                fail_elem = tc.find("failure")
                err_elem = tc.find("error")
                skip_elem = tc.find("skipped")

                if fail_elem is not None:
                    status = "FAIL"
                    err_msg = fail_elem.attrib.get("message", "") or fail_elem.text or ""
                elif err_elem is not None:
                    status = "ERROR"
                    err_msg = err_elem.attrib.get("message", "") or err_elem.text or ""
                elif skip_elem is not None:
                    status = "SKIPPED"
                    err_msg = skip_elem.attrib.get("message", "")

                testcases.append((classname, name, status, t_duration, err_msg.strip()))

        passed = total_tests - total_failures - total_errors - total_skipped

        out.append(f" [1] Test Metrics Overview")
        out.append(f"   - Total Tests Executed : {total_tests}")
        out.append(f"   - Passed               : {passed}")
        out.append(f"   - Failed / Errors      : {total_failures + total_errors}")
        out.append(f"   - Skipped              : {total_skipped}")
        out.append(f"   - Total Duration       : {total_time:.2f}s ({total_time/60.0:.2f} mins)")
        out.append("")
        out.append(f" [2] Test Case Breakdown")
        for cls, name, st, dur, msg in testcases:
            st_str = f"[{st}]"
            short_cls = cls.split(".")[-1]
            out.append(f"   {st_str:9} {short_cls}::{name} ({dur:.2f}s)")
            if msg and st in ("FAIL", "ERROR"):
                first_line = msg.splitlines()[0][:100]
                out.append(f"             Reason: {first_line}")
        has_report = True
    except Exception as e:
        out.append(f" ⚠️ Could not parse JUnit XML ({e}). Falling back to raw log analysis...")

# If JUnit XML was unavailable, parse raw log for pytest summary
if not has_report and os.path.isfile(e2e_log):
    out.append(f" [1] Pytest Console Summary (parsed from {os.path.basename(e2e_log)})")
    with open(e2e_log, "r", errors="ignore") as f:
        lines = f.readlines()
    
    summary_lines = []
    capture = False
    for line in lines[-100:]:
        if "=== short test summary info ===" in line or "====" in line and ("passed" in line or "failed" in line or "error" in line):
            capture = True
        if capture:
            summary_lines.append(line.rstrip())
            if "=== " in line and (" in " in line or " seconds" in line):
                break
    
    if summary_lines:
        for s in summary_lines:
            out.append(f"   {s}")
    else:
        out.append("   (Pytest summary not found yet in log output)")

out.append(sep)

summary_text = "\n".join(out)
# Save plain text to file
try:
    with open(summary_file, "w") as f:
        f.write(summary_text + "\n")
except Exception:
    pass

# Print with colors to stdout
for line in out:
    if "PASS" in line or "passed" in line.lower():
        print(f"{GREEN}{line}{RESET}")
    elif "FAIL" in line or "ERROR" in line or "failed" in line.lower() or "error" in line.lower():
        print(f"{RED}{line}{RESET}")
    elif line.startswith("="):
        print(f"{BOLD}{CYAN}{line}{RESET}")
    elif line.startswith(" -") or line.startswith(" ["):
        print(f"{BOLD}{line}{RESET}")
    else:
        print(line)

PYEOF
}

if [ "$MODE" = "summary" ]; then
    generate_summary
    exit 0
fi

echo -e "${BOLD}${CYAN}============================================================${RESET}"
echo -e "${BOLD}${CYAN} ⚡ TPU vLLM E2E Pytest Log Streamer & Report Utility${RESET}"
echo -e "${BOLD}${CYAN}============================================================${RESET}"
echo -e " JobSet Name : ${YELLOW}${JOB_NAME}${RESET}"
echo -e " Log Index   : ${LOG_NUM}"
echo -e " Mode        : ${MODE}"
echo -e " Output Dir  : ${LOG_DIR}"
echo -e " Target Log  : ${E2E_LOG}"
echo -e " Report XML  : ${REPORT_XML}"
echo -e "${BOLD}${CYAN}============================================================${RESET}"

# Verify JobSet exists or check pods
if ! kubectl get jobset "${JOB_NAME}" >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️ Warning: JobSet '${JOB_NAME}' not found. Checking if pods exist...${RESET}"
    if [ -z "$(kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=${JOB_NAME}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)" ]; then
        echo -e "${RED}❌ Error: No pods found for '${JOB_NAME}'.${RESET}"
        exit 1
    fi
fi

LABEL="jobset.sigs.k8s.io/jobset-name=${JOB_NAME},jobset.sigs.k8s.io/replicatedjob-name=e2e"
CONTAINER="e2e-pytest"

fetch_junit_report() {
    local pod_name
    pod_name=$(kubectl get pod -l "$LABEL" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    if [ -n "$pod_name" ]; then
        echo -e "📥 Copying JUnit XML report from pod ${pod_name}:/tmp/e2e_report.xml -> $(basename "$REPORT_XML")..."
        if kubectl cp "${pod_name}:/tmp/e2e_report.xml" "${REPORT_XML}" 2>/dev/null; then
            echo -e "${GREEN}✅ JUnit report successfully retrieved: ${REPORT_XML}${RESET}"
        else
            echo -e "${DIM}ℹ️  /tmp/e2e_report.xml not found yet inside pod (test may still be starting or crashed).${RESET}"
        fi
    fi
}

CLEANED_UP=false
cleanup() {
    if [ "$CLEANED_UP" = true ]; then
        return
    fi
    CLEANED_UP=true

    echo ""
    fetch_junit_report
    echo ""
    generate_summary
}
trap cleanup EXIT INT TERM

# Wait for at least one pod to be present
echo -e "${CYAN}⏳ Waiting for E2E pod to appear...${RESET}"
while [ -z "$(kubectl get pods -l "$LABEL" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)" ]; do
    sleep 2
done

if [ "$MODE" = "dump" ]; then
    echo -e "📥 Dumping current E2E logs to $(basename "$E2E_LOG")..."
    kubectl logs -l "$LABEL" -c "$CONTAINER" --tail=-1 > "$E2E_LOG" 2>&1 || true
    echo -e "${GREEN}✅ Logs dumped to ${E2E_LOG}${RESET}"
    exit 0
fi

# STREAM (DEFAULT) MODE:
FOLLOW_FLAG=""
if [ "$FOLLOW" = true ]; then
    FOLLOW_FLAG="-f"
fi

echo -e "📊 Streaming & teeing E2E pytest execution live to terminal..."
while true; do
    kubectl logs -l "$LABEL" -c "$CONTAINER" --tail=-1 $FOLLOW_FLAG 2>&1 | tee "$E2E_LOG" || true
    
    # Check pod completion status
    PHASE=$(kubectl get pods -l "$LABEL" -o jsonpath='{.items[0].status.phase}' 2>/dev/null || true)
    if [ "$PHASE" = "Succeeded" ] || [ "$PHASE" = "Failed" ] || [ -z "$PHASE" ]; then
        break
    fi
    sleep 2
done

echo ""
echo -e "${GREEN}✅ E2E streaming finished.${RESET}"

