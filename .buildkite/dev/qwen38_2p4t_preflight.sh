#!/bin/bash
# Host-side preflight for the Qwen3.8-2.4T v7x-32 dev pipeline.
#
# Answers two questions about *every* host in the slice, before anything
# expensive starts, and prints the answers to stdout as `KEY=VALUE` lines the
# caller can `eval`:
#
#   CKPT_MOUNT=<abs path>   a locally-mounted copy of the checkpoint that every
#                           host can read, or empty if any host lacks it
#   CACHE_MIN_FREE_MB=<n>   free space on the JAX-compilation-cache filesystem,
#                           minimised over all hosts
#   CACHE_WRITABLE=0|1      1 only if the cache directory is writable everywhere
#
# Both are all-or-nothing on purpose. Every rank resolves the model path and
# the cache directory independently, so a mount or a writable cache that exists
# on three hosts out of four is worse than one that exists on none: the run
# gets three quarters of the way into a multi-hour load and then dies on the
# odd host. Falling back uniformly is slower but always correct.
#
# The mount is *discovered*, never hardcoded. Operators attach the disk
# wherever they like; this looks for a directory named after the checkpoint
# under the usual mount roots and validates it by content.
#
# All diagnostics go to stderr so `eval "$(preflight.sh)"` stays clean.
set -uo pipefail

CKPT_DIRNAME="${CKPT_DIRNAME:-Qwen3.8-2.4T-A95B-FP8}"
# 213 files in this checkpoint. A partially-synced disk is worse than no disk,
# so require nearly all of them rather than merely "some".
CKPT_MIN_SAFETENSORS="${CKPT_MIN_SAFETENSORS:-200}"
CACHE_HOST_DIR="${CACHE_HOST_DIR:-/tmp/jax_cache_tpu7x}"
# Space-separated, deliberately unquoted where used. First root with a hit wins.
CKPT_SEARCH_ROOTS="${CKPT_SEARCH_ROOTS:-/mnt/disks /mnt}"

log() { echo "[preflight] $*" >&2; }

# --- the per-host probe, run locally and shipped over ssh unchanged ----------
PROBE=$(mktemp /tmp/qwen38_probe.XXXXXX.sh)
trap 'rm -f "$PROBE"' EXIT
cat > "$PROBE" <<'PROBE_BODY'
# args: <ckpt-dirname> <min-safetensors> <cache-host-dir> <search-roots...>
set -u
dirname_want="$1"; min_st="$2"; cache_dir="$3"; shift 3
found=""
for root in "$@"; do
for cand in "$root"/*/"$dirname_want"; do
  [ -d "$cand" ] || continue
  [ -r "$cand/config.json" ] || { echo "  reject $cand: no readable config.json" >&2; continue; }
  n=$(ls -1 "$cand"/*.safetensors 2>/dev/null | wc -l)
  if [ "$n" -lt "$min_st" ]; then
    echo "  reject $cand: only $n safetensors (want >= $min_st)" >&2
    continue
  fi
  echo "  accept $cand: $n safetensors" >&2
  found="$cand"
  break
done
[ -n "$found" ] && break
done
mkdir -p "$cache_dir" 2>/dev/null
writable=0
if touch "$cache_dir/.preflight_wtest" 2>/dev/null; then
  rm -f "$cache_dir/.preflight_wtest"
  writable=1
fi
free_mb=$(df -Pm "$cache_dir" 2>/dev/null | awk 'NR==2{print $4}')
[ -n "${free_mb:-}" ] || free_mb=0
echo "RESULT ckpt=$found free_mb=$free_mb writable=$writable"
PROBE_BODY

run_probe() {  # $1 = "" for local, else worker ip
  if [ -z "$1" ]; then
    bash "$PROBE" "$CKPT_DIRNAME" "$CKPT_MIN_SAFETENSORS" "$CACHE_HOST_DIR" $CKPT_SEARCH_ROOTS
  else
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@$1" \
      "bash -s -- '$CKPT_DIRNAME' '$CKPT_MIN_SAFETENSORS' '$CACHE_HOST_DIR' $CKPT_SEARCH_ROOTS" < "$PROBE"
  fi
}

# --- discover the rest of the slice -----------------------------------------
# Same mechanism run_multihost.sh uses: we are on the head node, and gcloud
# lists every endpoint of the TPU slice with the head first.
SSH_USER="${SSH_USER:-$(whoami)}"
[ -f ~/.ssh/id_rsa ] || { mkdir -p ~/.ssh; ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa -q; }
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=15 -i ~/.ssh/id_rsa)

worker_ips=""
if [ -n "${WORKER_IPS:-}" ]; then
  worker_ips="${WORKER_IPS//,/ }"
else
  md="http://metadata.google.internal/computeMetadata/v1/instance"
  zone=$(curl -s -m 5 -H "Metadata-Flavor: Google" "$md/zone" | awk -F/ '{print $NF}')
  tpu=$(curl -s -m 5 -H "Metadata-Flavor: Google" "$md/description" 2>/dev/null)
  if [ -n "$zone" ] && [ -n "$tpu" ]; then
    all=$(gcloud compute tpus tpu-vm describe "$tpu" --zone "$zone" \
            --format="value(networkEndpoints[].ipAddress)" 2>/dev/null)
    all="${all//;/ }"; all="${all//,/ }"
    # shellcheck disable=SC2206
    arr=($all)
    worker_ips="${arr[*]:1}"
  fi
fi

if [ -z "$worker_ips" ]; then
  # Cannot prove the other hosts are equipped, so do not let the head act as
  # if they were. Report nothing usable and let the caller take the safe path.
  log "could not discover worker IPs -- assuming no shared mount, no cache"
  echo "CKPT_MOUNT="
  echo "CACHE_MIN_FREE_MB=0"
  echo "CACHE_WRITABLE=0"
  exit 0
fi

log "slice hosts: head(local) $worker_ips"

# --- collect ----------------------------------------------------------------
mounts=()
min_free=""
writable_all=1
for host in "" $worker_ips; do
  label="${host:-head}"
  out=$(run_probe "$host" 2>/tmp/qwen38_probe_${label//./_}.err)
  sed "s/^/[preflight] ${label}: /" "/tmp/qwen38_probe_${label//./_}.err" >&2 || true
  rm -f "/tmp/qwen38_probe_${label//./_}.err"
  line=$(printf '%s\n' "$out" | grep '^RESULT ' | tail -1)
  if [ -z "$line" ]; then
    # Unreachable host. The checkpoint verdict must go negative -- every rank
    # resolves MODEL_PATH itself, so an unverified host is a coin flip on a
    # three-hour load. The cache verdict need not: JAX's persistent cache is
    # per host, a host without one simply recompiles, and the serve script
    # re-checks free space inside every container anyway.
    log "$label: probe produced no result (ssh failure?) -- no local checkpoint for the slice"
    mounts+=("")
    continue
  fi
  ckpt=$(printf '%s' "$line" | sed -n 's/.* ckpt=\([^ ]*\).*/\1/p')
  free=$(printf '%s' "$line" | sed -n 's/.* free_mb=\([0-9]*\).*/\1/p')
  wr=$(printf '%s' "$line" | sed -n 's/.* writable=\([01]\).*/\1/p')
  log "$label: ckpt='${ckpt:-<none>}' cache_free=${free}MB writable=${wr}"
  mounts+=("$ckpt")
  [ "${wr:-0}" = "1" ] || writable_all=0
  if [ -z "$min_free" ] || [ "${free:-0}" -lt "$min_free" ]; then min_free="${free:-0}"; fi
done

# Every host must agree on the same path: the container bind-mounts it at the
# identical location on all hosts and each rank resolves MODEL_PATH itself.
mount_common="${mounts[0]}"
for m in "${mounts[@]}"; do
  if [ -z "$m" ] || [ "$m" != "$mount_common" ]; then
    if [ -n "$mount_common" ]; then
      log "hosts disagree on the checkpoint mount ('$mount_common' vs '${m:-<none>}') -- falling back to object storage"
    fi
    mount_common=""
    break
  fi
done

[ -n "$mount_common" ] \
  && log "USING LOCAL CHECKPOINT: $mount_common (all $(( 1 + $(wc -w <<< "$worker_ips") )) hosts)" \
  || log "USING OBJECT STORAGE for the checkpoint"

echo "CKPT_MOUNT=$mount_common"
echo "CACHE_MIN_FREE_MB=${min_free:-0}"
echo "CACHE_WRITABLE=$writable_all"
