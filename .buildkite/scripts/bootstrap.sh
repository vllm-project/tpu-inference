#!/bin/bash
# Copyright 2025 Google LLC
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

# Exit on error, exit on unset variable, fail on pipe errors.
set -euo pipefail

# Resolve the absolute directory path of the current script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the shared pipeline config file.
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/configs/pipeline_config.sh"

determine_job_priority() {
  local priority=""
  echo "--- Determining job priority" >&2
  if [[ "${NIGHTLY:-0}" == "1" ]]; then
    # Nightly build (Lowest priority)
    priority="$PRIORITY_NIGHTLY"
    echo "Build type: Nightly - Priority: $priority" >&2
  elif [[ "$BUILDKITE_PIPELINE_SLUG" == "tpu-vllm-integration" ]]; then
    # Integration pipeline
    priority="$PRIORITY_INTEGRATION"
    echo "Build type: Integration - Priority: $priority" >&2
  elif [[ "$BUILDKITE_PULL_REQUEST" != "false" && -n "$BUILDKITE_PULL_REQUEST" ]]; then
    local labels="${BUILDKITE_PULL_REQUEST_LABELS:-}"
    if grep -qx "oncall-fix" <<< "${labels//,/$'\n'}"; then
      # PR that fixes a CI or nightly breakage (Highest priority)
      priority="$PRIORITY_ONCALL_FIX"
      echo "Build type: On-call fix (PR #$BUILDKITE_PULL_REQUEST) - Priority: $priority" >&2
    else
      # Pre-merge PR tests
      priority="$PRIORITY_PRE_MERGE"
      echo "Build type: Pre-merge (PR #$BUILDKITE_PULL_REQUEST) - Priority: $priority" >&2
    fi
  elif [[ "$BUILDKITE_BRANCH" == "main" && "$BUILDKITE_PULL_REQUEST" == "false" ]]; then
    # Post-merge tests on main
    priority="$PRIORITY_POST_MERGE"
    echo "Build type: Post-merge (Main branch) - Priority: $priority" >&2
  else
    # Default priority for other branches or manual builds
    priority="$PRIORITY_DEFAULT"
    echo "Build type: General - Priority: $priority" >&2
  fi

  echo "$priority"
}

JOB_PRIORITY=$(determine_job_priority)
export JOB_PRIORITY
buildkite-agent meta-data set "JOB_PRIORITY" "$JOB_PRIORITY"
if [[ "$JOB_PRIORITY" == "$PRIORITY_ONCALL_FIX" ]]; then
  buildkite-agent annotate --style warning --context oncall-fix \
    "Runs at on-call priority (\`oncall-fix\` label): ahead of every other build, including main."
fi

# --- Skip build if only docs, icons, or CODEOWNERS changed ---
echo "--- :git: Checking changed files"

BASE_BRANCH=${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-"main"}
FILES_CHANGED=""

if [ "$BUILDKITE_PULL_REQUEST" != "false" ]; then
    echo "PR detected. Target branch: $BASE_BRANCH"

    # Fetch base and current commit to ensure local history exists for diff
    git fetch origin "$BASE_BRANCH" --depth=20 --quiet || echo "Base fetch failed"
    git fetch origin "$BUILDKITE_COMMIT" --depth=20 --quiet || true

    # Get all changes in this PR using triple-dot diff (common ancestor to HEAD)
    # This correctly captures changes even if the last commit is a merge from main
    FILES_CHANGED=$(git diff --name-only origin/"$BASE_BRANCH"..."$BUILDKITE_COMMIT" 2>/dev/null || true)

    # Fallback to single commit diff if PR history is unavailable
    if [ -z "$FILES_CHANGED" ]; then
        echo "Warning: PR diff failed. Falling back to single commit check."
        FILES_CHANGED=$(git diff-tree --no-commit-id --name-only -r -m "$BUILDKITE_COMMIT")
    fi
    
    echo "Files changed:"
    echo "$FILES_CHANGED"

    # Filter out files we want to skip builds for.
    NON_SKIPPABLE_FILES=$(echo "$FILES_CHANGED" | grep -vE "(\.md$|\.ico$|\.png$|^README$|^docs\/|^\.github\/CODEOWNERS$|support_matrices\/.*\.csv$)" || true)

    if [ -z "$NON_SKIPPABLE_FILES" ]; then
      echo "Only documentation, icon, or CODEOWNERS files changed. Skipping build."
      # No pipeline will be uploaded, and the build will complete.
      exit 0
    else
      echo "Code files changed. Proceeding with pipeline upload."
    fi

    # Count files not matching the benchmark prefix
    NON_BENCHMARK_COUNT=$(printf "%s\n" "$NON_SKIPPABLE_FILES" | grep -c -v "^\.buildkite/benchmark" || true)
    
    # Validate custom pipeline metadata (Uniqueness & Completeness)
    if .buildkite/scripts/validate_pipeline_metadata.sh "$NON_SKIPPABLE_FILES"; then
      echo "Pipeline metadata validation passed."
    else
      echo "+++ ❌ Pipeline metadata validation failed. Failing build."
      exit 1
    fi

    # Validate modified YAML pipelines using bk pipeline validate
    if .buildkite/scripts/validate_buildkite_ymls.sh "$NON_SKIPPABLE_FILES"; then
      echo "All pipelines syntax are valid. Proceeding with pipeline upload."
    else
      echo "Some pipelines syntax are invalid. Failing build."
      exit 1
    fi

    MODEL_FILES="add_model_to_ci\.py|tpu_optimized_model_template\.yml|vllm_native_model_template\.yml"
    FEATURE_FILES="add_feature_to_ci\.py|feature_template\.yml|parallelism_template\.yml"

    if echo "$FILES_CHANGED" | grep -qE "$MODEL_FILES"; then
      .buildkite/pipeline_generation/test_generation.sh --models
    fi

    if echo "$FILES_CHANGED" | grep -qE "$FEATURE_FILES"; then
      .buildkite/pipeline_generation/test_generation.sh --features
    fi
else
    echo "Non-PR build. Bypassing file change check."
    FILES_CHANGED=$(git diff-tree --no-commit-id --name-only -r -m "$BUILDKITE_COMMIT")
fi

# Store changed files in metadata for sub-pipelines (newlines to commas)
echo "$FILES_CHANGED" | tr '\n' ',' | buildkite-agent meta-data set "changed_files"

# Evaluate which file paths were changed in the PR to dynamically trigger kernel tests.
# These variables are interpolated into pipeline_jax.yml to skip steps without provisioning TPUs.
export RUN_KERNEL_TESTS=0
export RUN_KERNEL_COLLECTIVES_TESTS=0

# Trigger general kernel tests if any of the following are modified:
# - tpu_inference/kernels/* or tests/kernels/*
# - tests/conftest.py or requirements.txt
if echo "$FILES_CHANGED" | grep -qE '^(tpu_inference/kernels|tests/kernels|tests/conftest.py|requirements.txt)'; then
    export RUN_KERNEL_TESTS=1
fi

# Trigger collective-specific kernel tests if any of the following are modified:
# - tpu_inference/kernels/collectives/* or tests/kernels/collectives/*
# - tests/conftest.py or requirements.txt
if echo "$FILES_CHANGED" | grep -qE '^(tpu_inference/kernels/collectives|tests/kernels/collectives|tests/conftest.py|requirements.txt)'; then
    export RUN_KERNEL_COLLECTIVES_TESTS=1
fi

# Handles the environment state for different TPU generations.
set_jax_envs() {
    case $1 in
        v6)
            export TESTS_GROUP_LABEL="[jax] TPU6e Tests Group"
            export TPU_VERSION="tpu6e"
            export TPU_QUEUE_SINGLE="tpu_v6e_queue"
            export TPU_QUEUE_MULTI="tpu_v6e_8_queue"
            export TENSOR_PARALLEL_SIZE_SINGLE=1
            export TENSOR_PARALLEL_SIZE_MULTI=8
            ;;
        v7)
            export TESTS_GROUP_LABEL="[jax] TPU7x Tests Group"
            export TPU_VERSION="tpu7x"
            export TPU_QUEUE_SINGLE="tpu_v7x_2_queue"
            export TPU_QUEUE_MULTI="tpu_v7x_8_queue"
            export TENSOR_PARALLEL_SIZE_SINGLE=2
            export TENSOR_PARALLEL_SIZE_MULTI=8
            export COV_FAIL_UNDER="67"
            ;;
        unset)
            unset TESTS_GROUP_LABEL TPU_VERSION TPU_QUEUE_SINGLE TPU_QUEUE_MULTI TENSOR_PARALLEL_SIZE_SINGLE TENSOR_PARALLEL_SIZE_MULTI COV_FAIL_UNDER
            ;;
    esac
}

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

# One kube lane. models and features keep a file per model or feature in
# .buildkite/<lane>/kube/, beside its bare-metal file, and go up together as
# one pipeline the way upload_models_and_features.sh sends the bare-metal ones:
# each file's own steps: line dropped and the rest concatenated. The other
# lanes are a single file each.
upload_kube_lane() {
    local lane="$1"
    local dir=".buildkite/${lane}/kube"
    if [[ ! -d "${dir}" ]]; then
        upload_with_priority ".buildkite/pipeline_${lane}_kube.yml" "$JOB_PRIORITY"
        return
    fi
    echo "--- :pipeline: Uploading ${dir}/*.yml with priority ${JOB_PRIORITY:-PRIORITY_DEFAULT}"
    {
        echo "priority: ${JOB_PRIORITY:-PRIORITY_DEFAULT}"
        echo "steps:"
        grep -hv '^steps:' "${dir}"/*.yml
    } | buildkite-agent pipeline upload
}

# The kube files in place of upload_pipeline's. A scheduled kube run sets
# CI_LANES to some of jax, models, features, parallelism and rl, and runs the
# one generation its schedule names (TPU_VERSION and the KUBE_SHAPE_* env); any
# other kube build runs jax for both generations, as upload_pipeline does.
upload_kube_pipeline() {
    if [[ -n "${CI_LANES:-}" ]]; then
        local lane
        for lane in ${CI_LANES}; do
            case "${lane}" in
                jax)
                    upload_with_priority .buildkite/pipeline_jax_kube.yml "$JOB_PRIORITY"
                    if [[ "${TPU_VERSION:-tpu6e}" == "tpu6e" ]]; then
                        upload_with_priority .buildkite/pipeline_pypi_kube.yml "$JOB_PRIORITY"
                    fi
                    ;;
                models|features|parallelism|rl)
                    upload_kube_lane "${lane}"
                    ;;
                *)
                    echo "ERROR: CI_LANES has '${lane}'; the kube lanes are jax, models, features, parallelism and rl" >&2
                    exit 1
                    ;;
            esac
        done
        return
    fi
    if [ "${MODEL_IMPL_TYPE:-auto}" == "auto" ]; then
      set_kube_jax_envs v6
      upload_with_priority .buildkite/pipeline_jax_kube.yml "$JOB_PRIORITY"
      set_kube_jax_envs unset
      set_kube_jax_envs v7
      upload_with_priority .buildkite/pipeline_jax_kube.yml "$JOB_PRIORITY"
      set_kube_jax_envs unset
      upload_with_priority .buildkite/pipeline_pypi_kube.yml "$JOB_PRIORITY"
    fi
    # What nightly_verify.yml runs on bare metal on nightly and tag builds: the
    # models, features, parallelism and rl suites for both generations. Their
    # step keys carry TPU_VERSION, so each file uploads once per generation in
    # the same build. The support matrices are built on bare metal only.
    if [[ "${NIGHTLY:-0}" == "1" || -n "${BUILDKITE_TAG:-}" ]]; then
      local gen suite
      for gen in v6 v7; do
        set_kube_jax_envs "${gen}"
        for suite in models features parallelism rl; do
          upload_kube_lane "${suite}"
        done
        set_kube_jax_envs unset
      done
      # The P/D benchmark, once a day: it is a v7x workload whatever the
      # generation, and the vllm and flax_nnx nightlies would run it again on
      # the same code.
      if [ "${MODEL_IMPL_TYPE:-auto}" == "auto" ]; then
        upload_with_priority .buildkite/pipeline_disagg_kube.yml "$JOB_PRIORITY"
      fi
      buildkite-agent annotate --style warning --context ci-fleet-gaps \
        "Not in this kube build: the support matrices nightly_verify.yml builds on bare metal."
    fi
}

upload_pipeline() {
    if [[ "${CI_FLEET:-bare}" == "kube" ]]; then
      upload_kube_pipeline
      return
    fi
    if [ "${MODEL_IMPL_TYPE:-auto}" == "auto" ]; then
      # Upload JAX pipeline for v6 (default)
      set_jax_envs v6
      upload_with_priority .buildkite/pipeline_jax.yml "$JOB_PRIORITY"
      set_jax_envs unset

      # Upload JAX pipeline for v7
      set_jax_envs v7
      upload_with_priority .buildkite/pipeline_jax.yml "$JOB_PRIORITY"
      set_jax_envs unset

      # buildkite-agent pipeline upload .buildkite/pipeline_torch.yml
      upload_with_priority .buildkite/pipeline_pypi.yml "$JOB_PRIORITY"
    fi

    upload_with_priority .buildkite/nightly_verify.yml "$JOB_PRIORITY"
}

upload_benchmark_pipeline() {
    export BM_INFRA="true"
    if [[ "${CI_FLEET:-bare}" == "kube" ]]; then
      export BENCHMARK_TARGET=kube
    fi
    VLLM_COMMIT_HASH=$(buildkite-agent meta-data get "VLLM_COMMIT_HASH")
    TPU_COMMIT_HASH=$(git rev-parse HEAD)
    CODE_HASH="${VLLM_COMMIT_HASH}-${TPU_COMMIT_HASH}-"
    buildkite-agent meta-data set "CODE_HASH" "${CODE_HASH}"
    TIMEZONE="America/Los_Angeles"
    JOB_REFERENCE="$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)"
    buildkite-agent meta-data set "JOB_REFERENCE" "${JOB_REFERENCE}"
    echo "[BM-DEBUG] Using vllm commit hash: $(buildkite-agent meta-data get "VLLM_COMMIT_HASH")"
    echo "[BM-DEBUG] Using vllm-tpu commit hash: $(buildkite-agent meta-data get "CODE_HASH")"

    # Upload benchmark pipelines
    local case_folder=".buildkite/benchmark/cases/ci"
    local generator_script="${SCRIPT_DIR}/../benchmark/scripts/generate_bk_pipeline.py"
    local changed_cases=""

    # For PR builds, identify changed benchmark cases outside of ci folder
    if [ "$BUILDKITE_PULL_REQUEST" != "false" ] && [ -n "${FILES_CHANGED:-}" ]; then
        echo "--- Identifying changed benchmark cases in PR"
        export UPLOAD_DB="false"
        changed_cases=$(echo "$FILES_CHANGED" | grep "^\.buildkite/benchmark/cases/.*\.json$" | grep -v "^\.buildkite/benchmark/cases/ci/" || true)
        if [ -n "$changed_cases" ]; then
            echo "Found changed benchmark cases to include:"
            echo "$changed_cases"
            # Note: We keep changed_cases as a newline-separated string
            # to correctly handle filenames with spaces in process_json_benchmark_cases.

            # Upload global case name validation step
            upload_with_priority .buildkite/validate_benchmark_case_name.yml "$JOB_PRIORITY"
            export BENCHMARK_VALIDATION_UPLOADED="true"
        fi
    fi

    # Process both fixed ci folder and any extra changed cases
    process_json_benchmark_cases "$case_folder" "$generator_script" "$JOB_PRIORITY" "$changed_cases"
}

# Decides bare metal or kube (ci_fleet.sh), before anything that depends on
# the answer is uploaded.
choose_ci_fleet() {
    echo "--- :kubernetes: Choosing bare metal or kube"
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/ci_fleet.sh"
    resolve_ci_fleet
    if [[ "${CI_FLEET}" == "kube" ]]; then
      buildkite-agent annotate --style info --context ci-fleet \
        "Runs on the kube fleet (${CI_FLEET_REASON})."
    fi
}

echo "--- Starting Buildkite Bootstrap"
echo "Running in pipeline: $BUILDKITE_PIPELINE_SLUG"

echo "Configure notification"
ONCALL_EMAIL="ullm-test-notifications-external@google.com"
NOTIFY_FILE="generated_notification.yml"

# Logic
# 1. Official Integration/Nightly: If it's triggered by schedule -> Notify Oncall & Slack.
# 2. Everything else (PRs, Manual Triggers): Notify the creator of this build.
#    - This ensures that if you manually trigger the integration pipeline for debugging, 
#      it won't alert the oncall team.

if [[ "$BUILDKITE_PIPELINE_SLUG" == "tpu-vllm-integration" && "$BUILDKITE_SOURCE" == "schedule" ]] || \
   [[ "${NIGHTLY:-0}" == "1" && "$BUILDKITE_SOURCE" == "schedule" ]]; then
    echo "Context: Scheduled Integration/Nightly. Notifying Oncall."
    cat <<EOF > "$NOTIFY_FILE"
notify:
  - email: "$ONCALL_EMAIL"
    if: build.state == "failed"
  - slack: "vllm#tpu-ci-notifications"
    if: build.state == "failed"
EOF
else
    echo "Context: PR/Manual. Notifying Owner ($BUILDKITE_BUILD_CREATOR_EMAIL)."
    cat <<EOF > "$NOTIFY_FILE"
notify:
  - email: "$BUILDKITE_BUILD_CREATOR_EMAIL"
    if: build.state == "failed"
EOF

fi

# A scheduled kube build gates nothing yet, so it notifies no one: a lane
# (CI_LANES set), or a shadow of the bare integration run (CI_FLEET=kube on its
# schedule), whose failures the bare run already reports.
if [[ -z "${CI_LANES:-}" ]] && \
   [[ "${CI_FLEET:-}" != "kube" || "$BUILDKITE_SOURCE" != "schedule" ]]; then
  upload_with_priority "$NOTIFY_FILE" "$JOB_PRIORITY"
fi
rm "$NOTIFY_FILE"

echo "Configure testing logic"
if [[ $BUILDKITE_PIPELINE_SLUG == "tpu-vllm-integration" ]]; then
    # Note: Integration pipeline always fetch latest vllm version
    VLLM_COMMIT_HASH=$(git ls-remote https://github.com/vllm-project/vllm.git HEAD | awk '{ print $1}')
    buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${VLLM_COMMIT_HASH}"
    echo "Using vllm commit hash: $(buildkite-agent meta-data get "VLLM_COMMIT_HASH")"
    choose_ci_fleet
    # The pin moves on the bare run's results. A kube run shadows it and must
    # not promote a vLLM commit the bare run has not passed.
    if [[ "${CI_FLEET}" != "kube" ]]; then
      # Note: upload are inserted in reverse order, so promote LKG should upload before tests
      upload_with_priority .buildkite/integration_promote.yml "$JOB_PRIORITY"
    fi
    if [[ "${CI_FLEET}" == "kube" ]]; then
      set_kube_jax_envs v7
      upload_with_priority .buildkite/pipeline_jax_kube.yml "$JOB_PRIORITY"
      set_kube_jax_envs unset
      set_kube_jax_envs v6
      upload_with_priority .buildkite/pipeline_jax_kube.yml "$JOB_PRIORITY"
      set_kube_jax_envs unset
    else
      # Upload JAX pipeline for v7
      set_jax_envs v7
      upload_with_priority .buildkite/pipeline_jax.yml "$JOB_PRIORITY"
      set_jax_envs unset

      # Upload JAX pipeline for v6 (default)
      set_jax_envs v6
      upload_with_priority .buildkite/pipeline_jax.yml "$JOB_PRIORITY"
      set_jax_envs unset
    fi

else
  # Note: PR and Nightly pipelines will load VLLM_COMMIT_HASH from vllm_lkg.version file, if not exists, get the latest commit hash from vllm repo
  VLLM_COMMIT_HASH=$(get_vllm_commit_hash)
  buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${VLLM_COMMIT_HASH}"
  echo "Using vllm commit hash: $(buildkite-agent meta-data get "VLLM_COMMIT_HASH")"
    
  # Check if the current build is a pull request
  if [ "$BUILDKITE_PULL_REQUEST" != "false" ]; then
    echo "This is a Pull Request build."

    # Wait for GitHub API to sync labels
    echo "Sleeping for 5 seconds to ensure GitHub API is updated..."
    sleep 5

    API_URL="https://api.github.com/repos/vllm-project/tpu-inference/pulls/$BUILDKITE_PULL_REQUEST"
    echo "Fetching PR details from: $API_URL"

    # Fetch the response body and save to a temporary file
    GITHUB_PR_RESPONSE_FILE="github_api_logs.json"
    curl -s "$API_URL" -o "$GITHUB_PR_RESPONSE_FILE"
    
    # Upload the full response body as a Buildkite artifact
    echo "Uploading GitHub API response as artifact..."
    buildkite-agent artifact upload "$GITHUB_PR_RESPONSE_FILE"

    # Extract labels using input redirection
    PR_LABELS=$(jq -r '.labels[].name' < "$GITHUB_PR_RESPONSE_FILE")
    echo "Extracted PR Labels: $PR_LABELS"

    # If it's a PR, check for the specific label
    if [[ $PR_LABELS == *"ready"* ]]; then
      echo "Found 'ready' label on PR. Uploading main pipeline..."
      choose_ci_fleet
      # Upload main pipeline if file list is empty or contains non-benchmark files
      if [ -z "${NON_SKIPPABLE_FILES:-}" ] || [ "${NON_BENCHMARK_COUNT:--1}" -ne 0 ]; then
        upload_pipeline
      fi
      upload_benchmark_pipeline
    else
      # Explicitly fail the build because the required 'ready' label is missing.
      echo "Missing 'ready' label on PR. Failing build."
      exit 1
    fi
  else
    # If it's NOT a Pull Request (e.g., branch push, tag, manual build)
    echo "This is not a Pull Request build. Uploading main pipeline."
    choose_ci_fleet
    upload_pipeline
    # A scheduled kube lane is its jax suite alone; the benchmark has its own.
    if [[ -z "${CI_LANES:-}" ]]; then
      upload_benchmark_pipeline
    fi
  fi
fi

# Since Buildkite inserts steps in reverse order, uploading this last 
# ensures the Docker build steps appear at the very top of the UI.
upload_with_priority .buildkite/pipeline_build.yml "$JOB_PRIORITY"

echo "--- Buildkite Bootstrap Finished"