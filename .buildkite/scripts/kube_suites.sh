#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Uploads the kube versions of the test suites. Sourced by bootstrap.sh and by
# upload_models_and_features.sh, after configs/pipeline_config.sh
# (upload_with_priority).

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/nightly_suites.sh"

# One generation of pipeline_jax_kube.yml: the kube shapes in place of the bare
# queues set_jax_envs names.
set_kube_jax_envs() {
    case $1 in
        v6)
            export TPU_VERSION="tpu6e"
            export KUBE_SHAPE_SINGLE="ct6e-standard-1t/1x1"
            export KUBE_SHAPE_MULTI="ct6e-standard-8t/2x4"
            export TENSOR_PARALLEL_SIZE_SINGLE=1
            ;;
        v7)
            export TPU_VERSION="tpu7x"
            export KUBE_SHAPE_SINGLE="tpu7x-standard-1t/1x1x1"
            export KUBE_SHAPE_MULTI="tpu7x-standard-4t/2x2x1"
            export TENSOR_PARALLEL_SIZE_SINGLE=2
            ;;
        unset)
            unset TPU_VERSION KUBE_SHAPE_SINGLE KUBE_SHAPE_MULTI TENSOR_PARALLEL_SIZE_SINGLE
            ;;
    esac
}

# The CI_TARGETs of the files a kube nightly uploads, models apart from the
# rest: the model-list and feature-list meta-data the support matrices read,
# which upload_models_and_features.sh sets from the same suites on bare metal.
KUBE_MODEL_LIST=()
KUBE_FEATURE_LIST=()
add_ci_targets() {
    local suite="$1" target
    shift
    while IFS= read -r target; do
        [[ -n "${target}" ]] || continue
        if [[ "${suite}" == "models" ]]; then
            KUBE_MODEL_LIST+=("${target}")
        else
            KUBE_FEATURE_LIST+=("${target}")
        fi
    done < <(grep -hE '^[[:space:]]*CI_TARGET:' "$@" | sed -E 's/^[^:]*:[[:space:]]*//' | tr -d "\"'" | sed -E 's/[[:space:]]+$//' || true)
}

# One kube lane. models and features keep a file per model or feature in
# .buildkite/<lane>/kube/, beside its bare-metal file, and go up together as
# one pipeline the way upload_models_and_features.sh sends the bare-metal ones:
# each file's own steps: line dropped and the rest concatenated. The other
# lanes are a single file each.
upload_kube_lane() {
    local lane="$1"
    local dir=".buildkite/${lane}/kube"
    if [[ ! -d "${dir}" ]]; then
        add_ci_targets "${lane}" ".buildkite/pipeline_${lane}_kube.yml"
        upload_with_priority ".buildkite/pipeline_${lane}_kube.yml" "$JOB_PRIORITY"
        return
    fi
    add_ci_targets "${lane}" "${dir}"/*.yml
    echo "--- :pipeline: Uploading ${dir}/*.yml with priority ${JOB_PRIORITY:-PRIORITY_DEFAULT}"
    {
        echo "priority: ${JOB_PRIORITY:-PRIORITY_DEFAULT}"
        echo "steps:"
        grep -hv '^steps:' "${dir}"/*.yml
    } | buildkite-agent pipeline upload
}

# A suite's bare-metal files whose steps all run on the cpu queue. They are
# placeholders that record a model, feature or kernel as unverified for the
# support matrix, with nothing to run on a TPU, so they have no kube version
# and a kube nightly uploads them as they are. A file with TPU steps that has
# no kube version beside it goes in KUBE_MISSING, for the build annotation.
KUBE_MISSING=()
upload_cpu_only_files() {
    local suite="$1" f queues
    local -a files=()
    for f in ".buildkite/${suite}"/*.yml ".buildkite/${suite}"/*/*.yml; do
        [[ -f "${f}" && "${f}" != */kube/* ]] || continue
        queues=$(grep -oE 'queue:[[:space:]]*"?[^"[:space:],}#]+' "${f}" | sed -E 's/queue:[[:space:]]*"?//' | sort -u || true)
        if [[ "${queues}" == "cpu" ]]; then
            files+=("${f}")
        elif [[ -d ".buildkite/${suite}/kube" && ! -f ".buildkite/${suite}/kube/$(basename "${f}")" ]]; then
            KUBE_MISSING+=("${f}")
        fi
    done
    [[ "${#files[@]}" -gt 0 ]] || return 0
    add_ci_targets "${suite}" "${files[@]}"
    echo "--- :pipeline: Uploading ${#files[@]} cpu-only file(s) from .buildkite/${suite} with priority ${JOB_PRIORITY:-PRIORITY_DEFAULT}"
    {
        echo "priority: ${JOB_PRIORITY:-PRIORITY_DEFAULT}"
        echo "steps:"
        grep -hv '^steps:' "${files[@]}"
    } | buildkite-agent pipeline upload
}

# What "Upload Tests" (nightly_verify.yml) sends up in a kube build: the
# suites nightly_suites names, for both generations, and the meta-data the
# support matrices read, as upload_models_and_features.sh does on bare metal.
# Their step keys carry TPU_VERSION, so each file uploads once per generation in
# the same build.
upload_kube_nightly_suites() {
    local gen suite
    for gen in v6 v7; do
        set_kube_jax_envs "${gen}"
        for suite in $(nightly_suites); do
            case "${suite}" in
                models|features|parallelism|rl) upload_kube_lane "${suite}" ;;
            esac
            upload_cpu_only_files "${suite}"
        done
        set_kube_jax_envs unset
        buildkite-agent meta-data set "run_${gen}_matrix" "true"
    done
    if [[ "${#KUBE_MODEL_LIST[@]}" -gt 0 ]]; then
        printf '%s\n' "${KUBE_MODEL_LIST[@]}" | sort -u | buildkite-agent meta-data set "model-list"
    fi
    if [[ "${#KUBE_FEATURE_LIST[@]}" -gt 0 ]]; then
        printf '%s\n' "${KUBE_FEATURE_LIST[@]}" | sort -u | buildkite-agent meta-data set "feature-list"
    fi

    local gaps=""
    if [[ "${KUBE_OWNS_NIGHTLY:-0}" != "1" ]]; then
        gaps="This kube nightly builds the support matrices but does not commit them, record verified commit hashes or notify anyone of a failure: the bare-metal nightly does, until a kube schedule sets KUBE_OWNS_NIGHTLY=1."
    fi
    if [[ "${#KUBE_MISSING[@]}" -gt 0 ]]; then
        gaps+=" Missing here, bare-metal files with TPU steps and no kube version: $(printf '%s\n' "${KUBE_MISSING[@]}" | sort -u | xargs)."
    fi
    if [[ -n "${gaps}" ]]; then
        buildkite-agent annotate --style warning --context ci-fleet-gaps "${gaps}"
    fi
}
