#!/bin/bash

echo "--- Starting Special Buildkite Bootstrap ---"

# for loop features and models upload to buildkite
BUILDKITE_DIR=".buildkite"
TARGET_FOLDERS="models features"
MODEL_LIST_METADATA_KEY="model-names-list"

declare -a model_names
declare -a pipeline_steps

echo "--- Scanning: ${TARGET_FOLDERS}"

for folder in $TARGET_FOLDERS; do
  folder=$BUILDKITE_DIR/$folder
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

  done < <(find "$folder" -type f \( -name "*.yml" -o -name "*.yaml" \) -print0)
done

# Convert array to a newline-separated string
model_list_string=$(printf "%s\n" "${model_names[@]}")

if [[ -n "$model_list_string" ]]; then
  echo "--- Uploading Model Name List to Meta-data"
  echo "${model_list_string}" | buildkite-agent meta-data set "${MODEL_LIST_METADATA_KEY}"
  echo "Testing: $(buildkite-agent meta-data get "model-names-list")"
else
  echo "--- No Model Names found to upload."
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
  # buildkite-agent step update --state "passed"
fi

echo "--- Buildkite Special Bootstrap Finished ---"
