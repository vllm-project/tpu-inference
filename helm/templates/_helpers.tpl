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
{{- div $chips 4 -}}
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
{{- end -}}
