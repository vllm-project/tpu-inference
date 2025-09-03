from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

# Set the model name
model_name = "meta-llama/Llama-Guard-4-12B"

# Specify the path to your downloaded model weights
# This path should point to the directory containing the .safetensors files
model_path = "/mnt/disks/jiries-disk_data/.vllm_models/models--meta-llama--Llama-Guard-4-12B/snapshots/87acb4b94e930c3d679e6e7ee9d57e2feab9ea71/"

# Load the safetensors file and print the keys
try:
    # Safetensors are often split into multiple files, so we'll load one to inspect
    safetensors_file = model_path + "model-00001-of-00005.safetensors"
    state_dict = load_file(safetensors_file)
    print("--- Keys from one .safetensors file ---")
    for key in state_dict.keys():
        print(key)

except FileNotFoundError:
    print(f"File not found: {safetensors_file}. Please check the path.")
    print("Trying to download the model to inspect keys...")

    # Fallback to loading via AutoModel if the local file isn't found
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("\n--- Keys from AutoModel.from_pretrained ---")
    for name, param in model.named_parameters():
        print(name)
