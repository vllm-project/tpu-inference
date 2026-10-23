{{/*
Name every resource after the Helm release, so `helm uninstall <release>` and
the log/cleanup scripts can all key off the same string.
*/}}
{{- define "tpu-testcase.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{/*
Calculate TPU chips per VM from topology string (e.g. 2x1x1 -> 2, 2x2x1 -> 4, 2x2x2 -> 4 per VM)
*/}}
{{- define "tpu-testcase.chipsPerVm" -}}
{{- $parts := splitList "x" . -}}
{{- $x := index $parts 0 | int -}}
{{- $y := index $parts 1 | int -}}
{{- $z := index $parts 2 | int -}}
{{- $chips := mul $x $y $z -}}
{{- if lt $chips 4 -}}
{{- $chips -}}
{{- else -}}
4
{{- end -}}
{{- end }}

{{/*
================================================================================
Shared Storage & Caching Snippets
================================================================================
*/}}
{{- define "tpu-testcase.gcsFuseAnnotations" -}}
gke-gcsfuse/volumes: "true"
gke-gcsfuse/memory-limit: {{ .Values.storage.gcsFuse.memoryLimit | default "72Gi" | quote }}
gke-gcsfuse/cpu-limit: {{ .Values.storage.gcsFuse.cpuLimit | default "8" | quote }}
{{- end }}

{{- define "tpu-testcase.storageVolumeMounts" -}}
{{- if eq .Values.storage.type "gcs-cache" }}
- name: hf-gcs-cache
  mountPath: /root/.cache/huggingface
{{- else if eq .Values.storage.type "ramdisk" }}
- name: cache-ramdisk
  mountPath: /root/.cache
{{- else if eq .Values.storage.type "gcs-direct" }}
- name: gcs-fuse-models
  mountPath: /gcs-models
  readOnly: true
{{- end }}
{{- end }}

{{- define "tpu-testcase.storageVolumes" -}}
{{- if eq .Values.storage.type "gcs-cache" }}
- name: hf-gcs-cache
  csi:
    driver: gcsfuse.csi.storage.gke.io
    volumeAttributes:
      bucketName: {{ .Values.storage.bucketName }}
      mountOptions: "implicit-dirs,file-cache:max-size-mb:-1,file-cache:enable-parallel-downloads:true,file-cache:parallel-downloads-per-file:100{{ if .Values.storage.cacheSubpath }},only-dir={{ .Values.storage.cacheSubpath }}{{ end }}"
- name: gke-gcsfuse-cache
  emptyDir:
    medium: Memory
    sizeLimit: {{ .Values.storage.ramCacheLimit | default "32Gi" }}
{{- else if eq .Values.storage.type "ramdisk" }}
- name: cache-ramdisk
  emptyDir:
    medium: Memory
    sizeLimit: {{ .Values.storage.ramCacheLimit | default "32Gi" }}
{{- else if eq .Values.storage.type "gcs-direct" }}
- name: gcs-fuse-models
  csi:
    driver: gcsfuse.csi.storage.gke.io
    volumeAttributes:
      bucketName: {{ .Values.storage.bucketName }}
      mountOptions: "implicit-dirs,file-cache:max-size-mb:-1,file-cache:enable-parallel-downloads:true,file-cache:parallel-downloads-per-file:100"
- name: gke-gcsfuse-cache
  emptyDir:
    medium: Memory
    sizeLimit: {{ .Values.storage.ramCacheLimit | default "32Gi" }}
{{- end }}
{{- end }}

{{/*
================================================================================
Container Image Determination Helper
================================================================================
`image` is a map of registry / tpuInferenceCommit / vllmCommit, composed into
<registry>:<tpuInferenceCommit>-<vllmCommit>-<tpuVersion>. That is exactly the
tag the image builder produces and pushes, so the chart and the builder can
never disagree about what to run. No other tag format is ever used.
*/}}
{{- define "tpu-testcase.imageRef" -}}
  {{- $registry := "us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/tpu-inference-ci/vllm-tpu" -}}
  {{- $tpuCommit := "" -}}
  {{- $vllmCommit := "" -}}
  {{- if kindIs "map" .Values.image -}}
    {{- if .Values.image.registry -}}{{- $registry = .Values.image.registry -}}{{- end -}}
    {{- if .Values.image.tpuInferenceCommit -}}{{- $tpuCommit = .Values.image.tpuInferenceCommit -}}{{- end -}}
    {{- if .Values.image.vllmCommit -}}{{- $vllmCommit = .Values.image.vllmCommit -}}{{- end -}}
  {{- end -}}
  {{- if and .Values.builder -}}
    {{- if .Values.builder.registry -}}{{- $registry = .Values.builder.registry -}}{{- end -}}
    {{- if and (not $tpuCommit) .Values.builder.tpuInferenceCommit -}}{{- $tpuCommit = .Values.builder.tpuInferenceCommit -}}{{- end -}}
    {{- if and (not $vllmCommit) .Values.builder.vllmCommit -}}{{- $vllmCommit = .Values.builder.vllmCommit -}}{{- end -}}
  {{- end -}}
  {{- if not $tpuCommit -}}
    {{- fail "image.tpuInferenceCommit is required and cannot be empty. Please specify a commit hash in your values file or via --set image.tpuInferenceCommit=<hash>" -}}
  {{- end -}}
  {{- if not $vllmCommit -}}
    {{- fail "image.vllmCommit is required and cannot be empty. Please specify a commit hash in your values file or via --set image.vllmCommit=<hash>" -}}
  {{- end -}}
  {{- $tpuVer := .Values.tpu.accelerator | default "tpu7x" -}}
  {{- printf "%s:%s-%s-%s" $registry $tpuCommit $vllmCommit $tpuVer -}}
{{- end }}

{{/*
================================================================================
Image Builder Script
================================================================================
Runs Docker-in-Docker to check if the target image exists in Google Artifact Registry (GAR).
If missing, clones the repo, builds the image with setup_docker_env.sh flags, and pushes it.
Tag format is strictly: <registry>:<tpuInferenceCommit>-<vllmCommit>-<tpuVersion>

This emits ONLY the shell script body (no container spec, no indentation), so the
caller decides how to wrap it. It is consumed by templates/image-builder-job.yaml,
which runs it as a standalone CPU-only Helm pre-install hook Job.

Why a separate Job instead of an initContainer on the TPU pods:
a JobSet is admitted by Kueue as a single Workload, so every replicatedJob's podSet
is assigned a resource flavor and TAS nodes the moment the JobSet is created, even
though startupPolicy: InOrder runs them one at a time. Building inside the JobSet
therefore holds TPU quota for the entire build and exposes it to TPU node failures.
Running the build before the JobSet exists is the only way to decouple the two.
*/}}
{{- define "tpu-testcase.imageBuilderScript" -}}
{{- $registry := "us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/tpu-inference-ci/vllm-tpu" -}}
{{- $tpuCommit := "" -}}
{{- $vllmCommit := "" -}}
{{- if kindIs "map" .Values.image -}}
  {{- if .Values.image.registry -}}{{- $registry = .Values.image.registry -}}{{- end -}}
  {{- if .Values.image.tpuInferenceCommit -}}{{- $tpuCommit = .Values.image.tpuInferenceCommit -}}{{- end -}}
  {{- if .Values.image.vllmCommit -}}{{- $vllmCommit = .Values.image.vllmCommit -}}{{- end -}}
{{- end -}}
{{- if and .Values.builder -}}
  {{- if .Values.builder.registry -}}{{- $registry = .Values.builder.registry -}}{{- end -}}
  {{- if and (not $tpuCommit) .Values.builder.tpuInferenceCommit -}}{{- $tpuCommit = .Values.builder.tpuInferenceCommit -}}{{- end -}}
  {{- if and (not $vllmCommit) .Values.builder.vllmCommit -}}{{- $vllmCommit = .Values.builder.vllmCommit -}}{{- end -}}
{{- end -}}
{{- $tpuVer := .Values.tpu.accelerator | default "tpu7x" -}}
{{- $gitRepo := "https://github.com/vllm-project/tpu-inference.git" -}}
{{- if and .Values.builder .Values.builder.gitRepo -}}{{- $gitRepo = .Values.builder.gitRepo -}}{{- end -}}
{{- $cacheCfg := dict -}}
{{- if and .Values.builder .Values.builder.cache -}}{{- $cacheCfg = .Values.builder.cache -}}{{- end -}}
{{- $highPct := 70 -}}
{{- $lowPct := 50 -}}
{{- if hasKey $cacheCfg "highWatermarkPercent" -}}{{- $highPct = $cacheCfg.highWatermarkPercent -}}{{- end -}}
{{- if hasKey $cacheCfg "lowWatermarkPercent" -}}{{- $lowPct = $cacheCfg.lowWatermarkPercent -}}{{- end -}}
set -euo pipefail
TARGET_IMAGE="{{ include "tpu-testcase.imageRef" . }}"
REGISTRY_HOST="{{ (splitList "/" $registry) | first }}"
echo "============================================================"
echo " Image Builder"
echo " Target Image:       ${TARGET_IMAGE}"
echo " Registry Host:      ${REGISTRY_HOST}"
echo " TPU Inference Ref:  {{ $tpuCommit }}"
echo " vLLM Commit Ref:    {{ $vllmCommit }}"
echo "============================================================"

echo "Starting dockerd in background..."
dockerd > /tmp/dockerd.log 2>&1 &
DOCKERD_PID=$!
trap 'kill $DOCKERD_PID >/dev/null 2>&1 || true' EXIT

# /var/lib/docker is a persistent volume shared across runs, so the daemon may
# need to replay/verify existing layer metadata on startup. Allow more time than
# an empty data root would need, and surface the daemon log if it never comes up
# (otherwise every later docker command fails with an opaque socket error).
echo "Waiting for Docker daemon to become ready..."
DOCKER_READY=""
for i in $(seq 1 120); do
  if docker info > /dev/null 2>&1; then
    DOCKER_READY="yes"
    echo "Docker daemon ready after ${i}s."
    break
  fi
  sleep 1
done
if [ -z "$DOCKER_READY" ]; then
  echo "ERROR: Docker daemon did not become ready. Dumping /tmp/dockerd.log:"
  echo "------------------------------------------------------------"
  cat /tmp/dockerd.log || true
  echo "------------------------------------------------------------"
  exit 1
fi

# Report what the shared build cache already holds. Useful when diagnosing a slow
# build (cold cache) or a full volume.
echo "Build cache usage at start:"
docker system df || true

echo "Authenticating to registry via GCE metadata server..."
TOKEN=$(wget -q -O - --header="Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" 2>/dev/null | grep -o '"access_token": *"[^"]*"' | cut -d'"' -f4 || true)
if [ -n "$TOKEN" ]; then
  echo "$TOKEN" | docker login -u oauth2accesstoken --password-stdin "https://${REGISTRY_HOST}"
  echo "Authenticated successfully via metadata server."
else
  echo "WARNING: Failed to obtain metadata token. Proceeding with unauthenticated / existing creds."
fi

echo "Checking if target image exists in registry (fast-path)..."
# The output is captured rather than discarded: a permission error and a genuinely
# missing image are indistinguishable once suppressed, and mistaking the former for
# the latter means rebuilding for ~30 minutes only to be rejected by the final push.
if docker manifest inspect "${TARGET_IMAGE}" > /tmp/probe.log 2>&1 ||
   docker pull -q "${TARGET_IMAGE}" >> /tmp/probe.log 2>&1; then
  echo ">>> Image ${TARGET_IMAGE} already exists in registry. Skipping build! <<<"
  exit 0
fi

echo "Registry probe did not return the image. Details:"
sed 's/^/    /' /tmp/probe.log || true
if grep -qiE 'denied|unauthorized|forbidden|permission' /tmp/probe.log; then
  echo ""
  echo "ERROR: the registry rejected this pod's credentials, so the image cannot be"
  echo "       pushed either. Failing now instead of building for ~30 minutes first."
  echo "       Running as Kubernetes service account: ${K8S_SERVICE_ACCOUNT:-unknown}"
  echo "       This cluster uses Workload Identity (GKE_METADATA), so the effective"
  echo "       identity comes from that service account's iam.gke.io/gcp-service-account"
  echo "       annotation - not from the node. Set .Values.serviceAccount (or"
  echo "       .Values.builder.serviceAccountName) to one that can read and write"
  echo "       ${REGISTRY_HOST}."
  exit 1
fi
echo "Target image not found in registry. Proceeding with on-demand build..."
apk add --no-cache git bash curl

BUILD_DIR=$(mktemp -d)
echo "Cloning {{ $gitRepo }}..."
git clone {{ $gitRepo }} "${BUILD_DIR}/repo"
cd "${BUILD_DIR}/repo"

echo "Checking out commit: {{ $tpuCommit }}..."
git checkout {{ $tpuCommit }}

# --- Shared build cache garbage collection ----------------------------------
# Reached only when the registry fast-path missed, i.e. a real build is about to
# start, so a cache hit still costs nothing.
#
# The metric is filesystem usage of /var/lib/docker, not `docker system df`:
# the latter only accounts for what the daemon knows about (images, containers,
# build cache) and ignores overlay2/containerd leftovers, while the build dies
# on the real filesystem running out of space.
CACHE_HIGH_PCT={{ $highPct }}
CACHE_LOW_PCT={{ $lowPct }}

# Prints "<total_kb> <used_kb> <use_pct>". Fields are counted from the end so
# that a long device name wrapped onto its own line by df does not shift them.
# An unreadable mount yields zeros, which disables pruning rather than failing
# a build that might well have succeeded.
cache_usage() {
  df -k /var/lib/docker |
    awk 'END {
           if (NR == 0 || NF < 5) { print "0 0 0"; exit }
           gsub(/%/, "", $(NF-1))
           printf "%d %d %d\n", $(NF-4), $(NF-3), $(NF-1)
         }'
}

# The trailing zeros are a guard: if df fails we get 0% instead of an unbound
# variable under `set -u`, which would abort a build that could well succeed.
set -- $(cache_usage) 0 0 0
CACHE_TOTAL_KB=$1
CACHE_PCT=$3
echo "Build cache volume: ${CACHE_PCT}% used of $((CACHE_TOTAL_KB / 1024 / 1024))Gi (high watermark ${CACHE_HIGH_PCT}%)."

if [ "${CACHE_HIGH_PCT}" -gt 0 ] && [ "${CACHE_PCT}" -ge "${CACHE_HIGH_PCT}" ]; then
  echo "Above the high watermark. Stage 1: removing local images and stopped containers..."
  # Free in terms of rebuild speed: every image here has already been pushed to
  # the registry and the fast-path queries the registry, never the local daemon.
  # Layer reuse comes from the BuildKit cache, which stage 1 does not touch.
  docker container prune -f || true
  docker image prune -a -f || true

  set -- $(cache_usage) 0 0 0
  CACHE_PCT=$3
  echo "After stage 1: ${CACHE_PCT}% used."

  if [ "${CACHE_PCT}" -ge "${CACHE_HIGH_PCT}" ]; then
    # Only now give up cache that actually costs build time. `--keep-storage`
    # no longer exists (buildx v0.37 in docker:dind); --min-free-space states
    # how much free space to leave behind, so the low watermark is converted
    # into bytes.
    TARGET_FREE_BYTES=$(( CACHE_TOTAL_KB * (100 - CACHE_LOW_PCT) / 100 * 1024 ))
    echo "Stage 2: trimming the BuildKit cache to leave $((TARGET_FREE_BYTES / 1024 / 1024 / 1024))Gi free (low watermark ${CACHE_LOW_PCT}%)..."
    docker buildx prune --force --min-free-space "${TARGET_FREE_BYTES}" || true

    set -- $(cache_usage) 0 0 0
    CACHE_PCT=$3
    echo "After stage 2: ${CACHE_PCT}% used."
    if [ "${CACHE_PCT}" -ge "${CACHE_HIGH_PCT}" ]; then
      # Not fatal - the build may still fit - but worth shouting about, because
      # the usual cause is a volume too small for a single full build
      # (~25-40Gi of BuildKit cache plus ~10Gi of image).
      echo "WARNING: still ${CACHE_PCT}% used after pruning everything prunable."
      echo "         The volume is probably undersized; grow it with:"
      echo "           kubectl patch pvc vllm-build-cache -p '{\"spec\":{\"resources\":{\"requests\":{\"storage\":\"...\"}}}}'"
    fi
  fi
  echo "Cache state after pruning:"
  docker system df || true
fi

echo "Building Docker image: ${TARGET_IMAGE}..."
docker build \
  --build-arg IS_TEST=true \
  --build-arg BM_INFRA=true \
  --build-arg VLLM_COMMIT_HASH={{ $vllmCommit }} \
  -f docker/Dockerfile -t "${TARGET_IMAGE}" .

echo "Pushing ${TARGET_IMAGE} to registry..."
docker push "${TARGET_IMAGE}"
echo ">>> Successfully built and pushed ${TARGET_IMAGE} <<<"

echo "Build cache usage at end:"
docker system df || true
{{- end }}
