#!/bin/bash

BUILDKITE_DIR=".buildkite"
TARGET_FOLDERS="models features"
MODEL_LIST_KEY="model-list"
FEATURE_LIST_KEY="feature-list"

declare -a pipeline_steps

# Declare separate arrays for each list
declare -a model_list
declare -a feature_list


for folder_path in $TARGET_FOLDERS; do
  folder=$BUILDKITE_DIR/$folder_path
  # Check if the folder exists
  if [[ ! -d "$folder" ]]; then
    echo "Warning: Folder '$folder' not found. Skipping."
    continue
  fi

  echo "Processing config ymls in ${folder}"

  # Use find command to locate all .yml or .yaml files
  # -print0 and read -r -d '' are a safe way to handle filenames with special characters (like spaces)
  while IFS= read -r -d '' yml_file; do
    echo "--- handling yml file: ${yml_file}"

    # Get model name or feature name from first line of yml config
    first_line=$(awk 'NR==1{print $0; exit}' "${yml_file}")
    # Check if the first line contains the '# ' comment marker
    if [[ "$first_line" == "# "* ]]; then
      subject_name=${first_line#\# }

      case "$folder_path" in
        "models")
          model_list+=("${subject_name}")
          ;;
        "features")
          feature_list+=("${subject_name}")
          ;;
      esac
    fi

#     For each found .yml file, generate a command step
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

# Convert array to a newline-separated string
model_list_string=$(printf "%s\n" "${model_list[@]}")
feature_list_string=$(printf "%s\n" "${feature_list[@]}")

if [[ -n "$model_list_string" ]]; then
  echo "${model_list_string}" | buildkite-agent meta-data set "${MODEL_LIST_KEY}"
  echo "Testing: $(buildkite-agent meta-data get "${MODEL_LIST_KEY}")"
fi

if [[ -n "$feature_list_string" ]]; then
  echo "${feature_list_string}" | buildkite-agent meta-data set "${FEATURE_LIST_KEY}"
  echo "Testing: $(buildkite-agent meta-data get "${FEATURE_LIST_KEY}")"
fi

# --- Upload Dynamic Pipeline ---

if [[ "${#pipeline_steps[@]}" -gt 0 ]]; then
  echo "--- Uploading Dynamic Pipeline Steps"
  final_pipeline_yaml="steps:"$'\n'
  final_pipeline_yaml+=$(printf "%s\n" "${pipeline_steps[@]}")
  echo "Upload YML: ${final_pipeline_yaml}"
  echo -e "${final_pipeline_yaml}" | buildkite-agent pipeline upload
else
  echo "--- No .yml files found, no new Pipeline Steps to upload."
  buildkite-agent step update --state "passed"
fi
