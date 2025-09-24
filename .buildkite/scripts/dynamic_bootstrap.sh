#!/bin/bash

echo "--- Starting Special Buildkite Bootstrap ---"

# for loop features and models upload to buildkite
BUILDKITE_DIR=".buildkite"
TARGET_FOLDERS="models features models/informational"

MODEL_LIST_KEY="tpu-model-list"
INFORMATIONAL_MODEL_LIST_KEY="vllm-model-list"
POPURLAR_MODEL_LIST_KEY="popular-model-list"

FEATURE_LIST_METADATA_KEY="feature-list"

declare -a model_names
declare -a pipeline_steps

# Declare separate arrays for each list
declare -a tpu_model_list
declare -a vllm_model_list
declare -a popular_model_list
declare -a feature_list

echo "--- Scanning: ${TARGET_FOLDERS}"

for folder_path in $TARGET_FOLDERS; do
  folder=$BUILDKITE_DIR/$folder_path
  # Check if the folder exists
  if [[ ! -d "$folder" ]]; then
    echo "Warning: Folder '$folder' not found. Skipping."
    continue
  fi

  # Use find command to locate all .yml or .yaml files
  # -print0 and read -r -d '' are a safe way to handle filenames with special characters (like spaces)
  while IFS= read -r -d '' yml_file; do
    echo "--- handling yml file: ${yml_file}"

    # Read the first line for getting model name
    first_line=$(awk 'NR==1{print $0; exit}' "${yml_file}")

    # Check if the first line contains the '# ' comment marker
    if [[ "$first_line" == "# "* ]]; then
      model_name=${first_line#\# }
      echo "Model Name: ${model_name}"

      # folder_name=$(basename "$folder_path")

      # Based on the folder name, add the model to the correct list
      case "$folder_path" in
        "models")
          tpu_model_list+=("${model_name}")
          ;;
        "models/informational")
          vllm_model_list+=("${model_name}")
          ;;
        "models/popular")
          popular_model_list+=("${model_name}")
          ;;
        "features")
          feature_list+=("${model_name}")
          ;;
        *)
          echo "Warning: No specific list for folder '${folder_path}'. Ignoring model '${model_name}'."
          ;;
      esac


      model_names+=("${model_name}")
    else
      echo "Warning: The first line of ${yml_file} is not in the expected comment format (ex: '# model-name')."
    fi

    # --- Dynamic Buildkite Pipeline Step ---
    # For each found .yml file, generate a command step
    # Here we assume the .yml file itself is an executable buildkite pipeline step script
    pipeline_yaml=$(cat <<EOF
- label: "Upload: ${yml_file}"
  command: "buildkite-agent pipeline upload ${yml_file}"
  agents:
    queue: tpu_v6e_queue
EOF
)

  pipeline_steps+=("${pipeline_yaml}")

  done < <(find "$folder" -maxdepth 1 -type f \( -name "*.yml" -o -name "*.yaml" \) -print0)
done

echo "--- Scan Complete. Final Lists: ---"

# Convert array to a newline-separated string
echo "TPU Models (${#tpu_model_list[@]}):"
printf "%s\n" "${tpu_model_list[@]}"
tpu_model_list_str=$(printf "%s\n" "${tpu_model_list[@]}")

echo "VLLM Models (${#vllm_model_list[@]}):"
printf "%s\n" "${vllm_model_list[@]}"
vllm_model_list_str=$(printf "%s\n" "${vllm_model_list[@]}")

echo "Popular Models (${#popular_model_list[@]}):"
printf "%s\n" "${popular_model_list[@]}"
popular_model_list_str=$(printf "%s\n" "${popular_model_list[@]}")

model_list_string=$(printf "%s\n" "${model_names[@]}")

if [[ -n "$tpu_model_list_str" ]]; then
  echo "--- Uploading tpu_model_list_str to Meta-data:${MODEL_LIST_KEY}"
  echo "${tpu_model_list_str}" | buildkite-agent meta-data set "${MODEL_LIST_KEY}"
  echo "Testing: $(buildkite-agent meta-data get "${MODEL_LIST_KEY}")"
else
  echo "--- No tpu-support Models found to upload."
fi

if [[ -n "$vllm_model_list_str" ]]; then
  echo "--- Uploading vllm_model_list_str to Meta-data:${INFORMATIONAL_MODEL_LIST_KEY}"
  echo "${vllm_model_list_str}" | buildkite-agent meta-data set "${INFORMATIONAL_MODEL_LIST_KEY}"
  echo "Testing: $(buildkite-agent meta-data get "${INFORMATIONAL_MODEL_LIST_KEY}")"
else
  echo "--- No vllm-native Models found to upload."
fi

if [[ -n "$popular_model_list_str" ]]; then
  echo "--- Uploading popular_model_list_str to Meta-data:${POPURLAR_MODEL_LIST_KEY}"
  echo "${popular_model_list_str}" | buildkite-agent meta-data set "${POPURLAR_MODEL_LIST_KEY}"
  echo "Testing: $(buildkite-agent meta-data get "${POPURLAR_MODEL_LIST_KEY}")"
else
  echo "--- No popular Models found to upload."
fi


# --- Upload Dynamic Pipeline ---

if [[ -n "$pipeline_steps" ]]; then
  echo "--- Uploading Dynamic Pipeline Steps"
  final_pipeline_yaml="steps:"$'\n'
  final_pipeline_yaml+=$(printf "%s\n" "${pipeline_steps[@]}")
  echo "Upload YML: ${final_pipeline_yaml}"
  echo -e "${final_pipeline_yaml}" | buildkite-agent pipeline upload
else
  echo "--- No .yml files found, no new Pipeline Steps to upload."
  buildkite-agent step update --state "passed"
fi

echo "--- Buildkite Special Bootstrap Finished ---"
