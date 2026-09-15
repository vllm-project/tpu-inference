{{/*
Expand the name of the chart.
*/}}
{{- define "tpu-vllm-benchmark.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name directly from Release.Name.
*/}}
{{- define "tpu-vllm-benchmark.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{/*
Determine served model name
*/}}
{{- define "tpu-vllm-benchmark.servedModelName" -}}
{{- if .Values.model.servedName -}}
{{- .Values.model.servedName -}}
{{- else -}}
{{- .Values.model.name -}}
{{- end -}}
{{- end }}

{{/*
Determine runtime model path: If storage.type is gcs-direct, resolve to /gcs-models/<subpath>
*/}}
{{- define "tpu-vllm-benchmark.modelPath" -}}
{{- if eq .Values.storage.type "gcs-direct" -}}
{{- if .Values.storage.cacheSubpath -}}
/gcs-models/{{ .Values.storage.cacheSubpath | trimPrefix "/" }}
{{- else -}}
/gcs-models
{{- end -}}
{{- else -}}
{{ .Values.model.name }}
{{- end -}}
{{- end }}

{{/*
Generalized helper: Calculate number of VMs from any topology string (e.g. 2x2x1 -> 1, 2x2x2 -> 2, 2x2x4 -> 4)
*/}}
{{- define "tpu-vllm-benchmark.numVmsForTopo" -}}
{{- $parts := splitList "x" . -}}
{{- $x := index $parts 0 | int -}}
{{- $y := index $parts 1 | int -}}
{{- $z := index $parts 2 | int -}}
{{- $chips := mul $x $y $z -}}
{{- if lt $chips 4 -}}
1
{{- else -}}
{{- div $chips 4 -}}
{{- end -}}
{{- end }}

{{/*
Calculate TPU chips per VM from topology string (e.g. 2x1x1 -> 2, 2x2x1 -> 4, 2x2x2 -> 4 per VM)
*/}}
{{- define "tpu-vllm-benchmark.chipsPerVmForTopo" -}}
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
Generalized helper: Calculate Tensor Parallel Size from any topology string (cores = chips * 2)
*/}}
{{- define "tpu-vllm-benchmark.tpSizeForTopo" -}}
{{- $parts := splitList "x" . -}}
{{- $x := index $parts 0 | int -}}
{{- $y := index $parts 1 | int -}}
{{- $z := index $parts 2 | int -}}
{{- $chips := mul $x $y $z -}}
{{- mul $chips 2 -}}
{{- end }}

{{/*
Generalized helper: Calculate TPU chips per process bounds (default: 2,2,1)
*/}}
{{- define "tpu-vllm-benchmark.chipsPerProcessBoundsForTopo" -}}
2,2,1
{{- end }}

{{/*
Generalized helper: Calculate TPU process bounds from any topology string
*/}}
{{- define "tpu-vllm-benchmark.processBoundsForTopo" -}}
{{- $parts := splitList "x" . -}}
{{- $x := index $parts 0 | int -}}
{{- $y := index $parts 1 | int -}}
{{- $z := index $parts 2 | int -}}
{{- $px := div (add $x 1) 2 -}}
{{- $py := div (add $y 1) 2 -}}
{{- $pz := div (add $z 0) 1 -}}
{{- printf "%d,%d,%d" $px $py $pz -}}
{{- end }}

{{/*
Generalized helper: Calculate total TPU cores from any topology string
*/}}
{{- define "tpu-vllm-benchmark.totalCoresForTopo" -}}
{{- $parts := splitList "x" . -}}
{{- $x := index $parts 0 | int -}}
{{- $y := index $parts 1 | int -}}
{{- $z := index $parts 2 | int -}}
{{- $chips := mul $x $y $z -}}
{{- mul $chips 2 -}}
{{- end }}

{{/*
Generalized helper: Calculate TPU accelerator type name from topology
*/}}
{{- define "tpu-vllm-benchmark.acceleratorTypeForTopo" -}}
{{- $cores := include "tpu-vllm-benchmark.totalCoresForTopo" . -}}
{{- printf "tpu7x-%s" $cores -}}
{{- end }}

{{/*
================================================================================
Monolithic / Aggregated Topology Helpers (mode == "aggregated")
================================================================================
*/}}
{{- define "tpu-vllm-benchmark.numVms" -}}
{{- include "tpu-vllm-benchmark.numVmsForTopo" .Values.tpu.topology -}}
{{- end }}

{{- define "tpu-vllm-benchmark.tpSize" -}}
{{- if gt (int .Values.tpu.tensorParallelSize) 0 -}}
{{- .Values.tpu.tensorParallelSize -}}
{{- else -}}
{{- include "tpu-vllm-benchmark.tpSizeForTopo" .Values.tpu.topology -}}
{{- end }}
{{- end }}

{{- define "tpu-vllm-benchmark.chipsPerProcessBounds" -}}
{{- if .Values.tpu.chipsPerProcessBounds -}}
{{- .Values.tpu.chipsPerProcessBounds -}}
{{- else -}}
{{- include "tpu-vllm-benchmark.chipsPerProcessBoundsForTopo" .Values.tpu.topology -}}
{{- end }}
{{- end }}

{{- define "tpu-vllm-benchmark.processBounds" -}}
{{- if .Values.tpu.processBounds -}}
{{- .Values.tpu.processBounds -}}
{{- else -}}
{{- include "tpu-vllm-benchmark.processBoundsForTopo" .Values.tpu.topology -}}
{{- end }}
{{- end }}

{{/*
================================================================================
Disaggregated Architecture Topology Helpers (mode == "disaggregated")
================================================================================
*/}}
{{- define "tpu-vllm-benchmark.prefillNumVms" -}}
{{- include "tpu-vllm-benchmark.numVmsForTopo" .Values.disaggregated.prefill.topology -}}
{{- end }}

{{- define "tpu-vllm-benchmark.prefillTpSize" -}}
{{- if and .Values.disaggregated.prefill.tensorParallelSize (gt (int .Values.disaggregated.prefill.tensorParallelSize) 0) -}}
{{- .Values.disaggregated.prefill.tensorParallelSize -}}
{{- else -}}
{{- include "tpu-vllm-benchmark.tpSizeForTopo" .Values.disaggregated.prefill.topology -}}
{{- end }}
{{- end }}

{{- define "tpu-vllm-benchmark.prefillChipsBounds" -}}
{{- if .Values.disaggregated.prefill.chipsPerProcessBounds -}}
{{- .Values.disaggregated.prefill.chipsPerProcessBounds -}}
{{- else -}}
{{- include "tpu-vllm-benchmark.chipsPerProcessBoundsForTopo" .Values.disaggregated.prefill.topology -}}
{{- end }}
{{- end }}

{{- define "tpu-vllm-benchmark.prefillProcessBounds" -}}
{{- if .Values.disaggregated.prefill.processBounds -}}
{{- .Values.disaggregated.prefill.processBounds -}}
{{- else -}}
{{- include "tpu-vllm-benchmark.processBoundsForTopo" .Values.disaggregated.prefill.topology -}}
{{- end }}
{{- end }}

{{- define "tpu-vllm-benchmark.decodeNumVms" -}}
{{- include "tpu-vllm-benchmark.numVmsForTopo" .Values.disaggregated.decode.topology -}}
{{- end }}

{{- define "tpu-vllm-benchmark.decodeTpSize" -}}
{{- if and .Values.disaggregated.decode.tensorParallelSize (gt (int .Values.disaggregated.decode.tensorParallelSize) 0) -}}
{{- .Values.disaggregated.decode.tensorParallelSize -}}
{{- else -}}
{{- include "tpu-vllm-benchmark.tpSizeForTopo" .Values.disaggregated.decode.topology -}}
{{- end }}
{{- end }}

{{- define "tpu-vllm-benchmark.decodeChipsBounds" -}}
{{- if .Values.disaggregated.decode.chipsPerProcessBounds -}}
{{- .Values.disaggregated.decode.chipsPerProcessBounds -}}
{{- else -}}
{{- include "tpu-vllm-benchmark.chipsPerProcessBoundsForTopo" .Values.disaggregated.decode.topology -}}
{{- end }}
{{- end }}

{{- define "tpu-vllm-benchmark.decodeProcessBounds" -}}
{{- if .Values.disaggregated.decode.processBounds -}}
{{- .Values.disaggregated.decode.processBounds -}}
{{- else -}}
{{- include "tpu-vllm-benchmark.processBoundsForTopo" .Values.disaggregated.decode.topology -}}
{{- end }}
{{- end }}

{{- define "tpu-vllm-benchmark.kvConnector" -}}
{{- if .Values.disaggregated.kvConnector -}}
{{- .Values.disaggregated.kvConnector -}}
{{- else if contains "torchtpu" (.Values.image | default "") -}}
TPURaidenConnector
{{- else -}}
TPUConnector
{{- end }}
{{- end }}

{{- define "tpu-vllm-benchmark.kvConnectorModule" -}}
{{- if .Values.disaggregated.kvConnectorModule -}}
{{- .Values.disaggregated.kvConnectorModule -}}
{{- else if contains "torchtpu" (.Values.image | default "") -}}
vllm_torchtpu.distributed.kv_transfer.tpu_connector
{{- else -}}
{{- $connector := include "tpu-vllm-benchmark.kvConnector" . -}}
{{- if eq $connector "TPURaidenConnector" -}}
tpu_inference.distributed.tpu_raiden_connector
{{- else -}}
tpu_inference.distributed.tpu_connector
{{- end -}}
{{- end }}
{{- end }}

{{/*
================================================================================
Shared Storage & Caching Snippets
================================================================================
*/}}
{{- define "tpu-vllm-benchmark.gcsFuseAnnotations" -}}
gke-gcsfuse/volumes: "true"
gke-gcsfuse/memory-limit: {{ .Values.storage.gcsFuse.memoryLimit | default "72Gi" | quote }}
gke-gcsfuse/cpu-limit: {{ .Values.storage.gcsFuse.cpuLimit | default "8" | quote }}
{{- end }}

{{- define "tpu-vllm-benchmark.storageVolumeMounts" -}}
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

{{- define "tpu-vllm-benchmark.storageVolumes" -}}
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
Advanced Inference & Serving Feature Flags Helper
================================================================================
*/}}
{{- define "tpu-vllm-benchmark.featureFlags" -}}
{{- if .Values.features -}}
{{- if hasKey .Values.features "asyncScheduling" -}}
{{- if .Values.features.asyncScheduling }}
--async-scheduling \
{{- else }}
--no-async-scheduling \
{{- end -}}
{{- end -}}
{{- if hasKey .Values.features "prefixCaching" -}}
{{- if .Values.features.prefixCaching }}
--enable-prefix-caching \
{{- else }}
--no-enable-prefix-caching \
{{- end -}}
{{- end -}}
{{- if .Values.features.chunkedPrefill -}}
{{- if .Values.features.chunkedPrefill.enabled }}
--enable-chunked-prefill \
--max-num-batched-tokens={{ .Values.features.chunkedPrefill.maxNumBatchedTokens | default 2048 }} \
{{- else }}
--no-enable-chunked-prefill \
{{- end -}}
{{- end -}}
{{- if .Values.features.wideEP -}}
{{- if .Values.features.wideEP.enabled }}
--enable-expert-parallel \
{{- end -}}
{{- end -}}
{{- if .Values.features.speculativeDecoding -}}
{{- if .Values.features.speculativeDecoding.enabled -}}
{{- if .Values.features.speculativeDecoding.draftModel }}
--speculative-model={{ .Values.features.speculativeDecoding.draftModel }} \
--num-speculative-tokens={{ .Values.features.speculativeDecoding.numSpeculativeTokens | default 3 }} \
{{- end -}}
{{- end -}}
{{- end -}}
{{- if .Values.features.structuredOutput -}}
{{- if .Values.features.structuredOutput.enabled }}
--structured-outputs-config='{"backend":"{{ .Values.features.structuredOutput.backend | default "auto" }}"}' \
{{- end -}}
{{- end -}}
{{- if and .Values.features.kvCacheOffload .Values.features.kvCacheOffload.enabled -}}
{{- if ne .Values.mode "disaggregated" }}
--kv-transfer-config='{"kv_connector":"{{ .Values.features.kvCacheOffload.connector | default "RaidenOffloadConnector" }}"}' \
{{- end -}}
{{- end -}}
{{- end -}}
{{- if and .Values.model .Values.model.maxNumSeqs -}}
--max-num-seqs={{ .Values.model.maxNumSeqs }} \
{{- end -}}
{{- end -}}

{{/*
================================================================================
Container Image Determination Helper
================================================================================
Tag format is strictly: <registry>:<tpuInferenceCommit>-<vllmCommit>-<tpuVersion>
Never uses any other tag format. If commit hashes are empty (""), they default to "latest".
*/}}
{{- define "tpu-vllm-benchmark.computedImage" -}}
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
Image Builder InitContainer Definition
================================================================================
Runs Docker-in-Docker to check if the target image exists in Google Artifact Registry (GAR).
If missing, clones the repo, builds the image with setup_docker_env.sh flags, and pushes it.
Tag format is strictly: <registry>:<tpuInferenceCommit>-<vllmCommit>-<tpuVersion>
*/}}
{{- define "tpu-vllm-benchmark.imageBuilderInitContainer" -}}
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
- name: image-builder
  image: docker:dind
  securityContext:
    privileged: true
  env:
  - name: DOCKER_TLS_CERTDIR
    value: ""
  command: ["/bin/sh", "-c"]
  args:
  - |
    set -euo pipefail
    TARGET_IMAGE="{{ include "tpu-vllm-benchmark.computedImage" . }}"
    REGISTRY_HOST="{{ (splitList "/" $registry) | first }}"
    echo "============================================================"
    echo " Image Builder InitContainer"
    echo " Target Image:       ${TARGET_IMAGE}"
    echo " Registry Host:      ${REGISTRY_HOST}"
    echo " TPU Inference Ref:  {{ $tpuCommit }}"
    echo " vLLM Commit Ref:    {{ $vllmCommit }}"
    echo "============================================================"

    echo "Starting dockerd in background..."
    dockerd > /tmp/dockerd.log 2>&1 &
    DOCKERD_PID=$!
    trap 'kill $DOCKERD_PID >/dev/null 2>&1 || true' EXIT

    echo "Waiting for Docker daemon to become ready..."
    for i in $(seq 1 30); do
      if docker info > /dev/null 2>&1; then
        echo "Docker daemon ready."
        break
      fi
      sleep 1
    done

    echo "Authenticating to registry via GCE metadata server..."
    TOKEN=$(wget -q -O - --header="Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" 2>/dev/null | grep -o '"access_token": *"[^"]*"' | cut -d'"' -f4 || true)
    if [ -n "$TOKEN" ]; then
      echo "$TOKEN" | docker login -u oauth2accesstoken --password-stdin "https://${REGISTRY_HOST}"
      echo "Authenticated successfully via metadata server."
    else
      echo "WARNING: Failed to obtain metadata token. Proceeding with unauthenticated / existing creds."
    fi

    echo "Checking if target image exists in registry (fast-path)..."
    if docker manifest inspect "${TARGET_IMAGE}" > /dev/null 2>&1 || docker pull -q "${TARGET_IMAGE}" > /dev/null 2>&1; then
      echo ">>> Image ${TARGET_IMAGE} already exists in registry. Skipping build! <<<"
      exit 0
    fi

    echo "Target image not found in registry. Proceeding with on-demand build..."
    apk add --no-cache git bash curl

    BUILD_DIR=$(mktemp -d)
    echo "Cloning {{ $gitRepo }}..."
    git clone {{ $gitRepo }} "${BUILD_DIR}/repo"
    cd "${BUILD_DIR}/repo"

    echo "Checking out commit: {{ $tpuCommit }}..."
    git checkout {{ $tpuCommit }}

    echo "Building Docker image: ${TARGET_IMAGE}..."
    docker build \
      --build-arg IS_TEST=true \
      --build-arg BM_INFRA=true \
      --build-arg VLLM_COMMIT_HASH={{ $vllmCommit }} \
      -f docker/Dockerfile -t "${TARGET_IMAGE}" .

    echo "Pushing ${TARGET_IMAGE} to registry..."
    docker push "${TARGET_IMAGE}"
    echo ">>> Successfully built and pushed ${TARGET_IMAGE} <<<"
{{- end }}
