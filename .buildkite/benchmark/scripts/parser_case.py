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
# yapf: disable
import json
import shlex
import sys

# Map JSON command_type to actual CLI commands
CMD_MAP = {
    "vllm_serve": "vllm serve",
    "vllm_bench_serve": "vllm bench serve",
    "lm_eval": "lm_eval"
}

def get_current_machine_type():
    """
    Returns the current machine type string (e.g., 'v6e-8', 'v7x-2') 
    using the tpu_info library.
    """
    try:
        from tpu_info import device
        chip_type, num_chips = device.get_local_chips()
        if chip_type and num_chips > 0:
            name = chip_type.value.name
            # Normalize naming convention (e.g., '7x' -> 'v7x')
            if name == "7x":
                name = "v7x"
                # For v7x, each core exposes its own PCI endpoint.
                # Therefore, num_chips returned by get_local_chips() is already the total core count.
                num_devices = num_chips
            else:
                if not name.startswith("v"):
                    name = f"v{name}"
                # For other types (e.g. v2, v3, v6e...)
                num_devices = num_chips * chip_type.value.devices_per_chip

            machine_type = f"{name}-{num_devices}"
            print(f"echo '[DEBUG] Detected machine type: {machine_type}' >&2")
            return machine_type
        else:
            print(f"echo '[WARNING] No TPU chips detected: chip_type={chip_type}, num_chips={num_chips}' >&2")
    except ImportError:
        print("echo '[WARNING] tpu_info library not found. Cannot determine machine type.' >&2")
    except Exception as e:
        print(f"echo '[WARNING] Failed to determine machine type: {e}' >&2")
    return None


def resolve_device_args(args_dict, current_machine):
    """
    Resolves dictionary-based arguments based on the current machine type.
    """
    resolved_args = {}
    if not args_dict:
        return resolved_args

    for key, value in args_dict.items():
        # If the argument value is a dictionary, treat it as a machine-mapping configuration
        if isinstance(value, dict):
            if current_machine and current_machine in value:
                resolved_args[key] = value[current_machine]
            elif "default" in value:
                resolved_args[key] = value["default"]
            else:
                # Fatal error if resolution fails and no default is provided
                print(f"echo '[ERROR] Failed to resolve arg \"--{key}\" for machine \"{current_machine}\". No default found.' >&2")
                print("exit 1")
                sys.exit(1)
        else:
            resolved_args[key] = value

    return resolved_args

def build_command(cmd_type, args_dict):
    """Builds a safe shell command string from a dictionary of arguments."""
    base_cmd = CMD_MAP.get(cmd_type, cmd_type)
    cmd_parts = base_cmd.split()

    if not args_dict:
        return shlex.join(cmd_parts)

    for key, value in args_dict.items():
        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{key}")
        else:
            cmd_parts.append(f"--{key}")
            cmd_parts.append(str(value))

    return shlex.join(cmd_parts)


def main():
    if len(sys.argv) < 2:
        print("echo 'Error: Missing config file.' >&2; exit 1")
        sys.exit(1)

    config_file = sys.argv[1]
    target_case = sys.argv[2] if len(sys.argv) > 2 else None

    # Determine current machine type from tpu_info
    current_machine = get_current_machine_type()

    try:
        with open(config_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"echo 'Error reading JSON: {e}' >&2; exit 1")
        sys.exit(1)

    is_multi_case = "benchmark_cases" in data

    # Resolve target case data
    if is_multi_case:
        if not target_case:
            print(
                "echo 'Error: TARGET_CASE_NAME required for multi-case config.' >&2; exit 1"
            )
            sys.exit(1)

        cases = data.get("benchmark_cases", [])
        case_data = next(
            (c for c in cases if c.get("case_name") == target_case), None)
        if not case_data:
            print(
                f"echo 'Error: Case \"{target_case}\" not found.' >&2; exit 1")
            sys.exit(1)

        merged_env = data.get("global_env", {}).copy()
        merged_env.update(case_data.get("env", {}))

        # Inject global_env into case_data so it will be included in the DB `Config`
        case_data["global_env"] = data.get("global_env", {})
    else:
        case_data = data
        merged_env = case_data.get("env", {})

    config_json_str = json.dumps(case_data)
    print(f"export CASE_CONFIG_JSON={shlex.quote(config_json_str)}")

    # Export environment variables securely
    for k, v in merged_env.items():
        print(f"export {k}={shlex.quote(str(v))}")

    srv_opts = case_data.get("server_command_options", {})
    cli_opts = case_data.get("client_command_options", {})

    # Export specific environment for insert to db
    dataset = cli_opts.get("args", {}).get("dataset-name", {})
    print(f"export DATASET=\"{dataset}\"")
    num_prompts = cli_opts.get("args", {}).get("num-prompts", {})
    print(f"export NUM_PROMPTS=\"{num_prompts}\"")
    additional_config = srv_opts.get("args", {}).get("additional-config", {})
    print(f"export ADDITIONAL_CONFIG={shlex.quote(str(additional_config))}")
    model = srv_opts.get("args", {}).get("model", {})
    print(f"export MODEL=\"{model}\"")
    max_num_seqs = srv_opts.get("args", {}).get("max-num-seqs", {})
    print(f"export MAX_NUM_SEQS=\"{max_num_seqs}\"")
    max_num_batched_tokens = srv_opts.get("args", {}).get("max-num-batched-tokens", {})
    print(f"export MAX_NUM_BATCHED_TOKENS=\"{max_num_batched_tokens}\"")
    max_model_len = srv_opts.get("args", {}).get("max-model-len", {})
    print(f"export MAX_MODEL_LEN=\"{max_model_len}\"")
    cli_env = cli_opts.get("env", {})
    cli_env_parts = [f"{k}={v}" for k, v in cli_env.items()]
    quoted_cli_env = ' '.join(shlex.quote(p) for p in cli_env_parts)
    print(f"CLIENT_CMD_ENVS=({quoted_cli_env})")
    srv_env = srv_opts.get("env", {})
    srv_env_list = [f"{k}={v}" for k, v in srv_env.items()]
    srv_env_str = ' '.join(shlex.quote(item) for item in srv_env_list)
    print(f"SERVER_CMD_ENVS=({srv_env_str})")

    cli_cmd_type = cli_opts.get("command_type", "vllm_bench_serve")

    # Output execution strategy based on command_type
    if cli_cmd_type == "lm_eval":
        # Resolve machine-specific args before building command
        cli_raw_args = cli_opts.get("args", {})
        cli_resolved_args = resolve_device_args(cli_raw_args, current_machine)

        print("export COMMAND_TYPE=\"lm_eval\"")
        lm_cmd = build_command(cli_cmd_type, cli_resolved_args)
        quoted_lm_cmd = ' '.join(
            shlex.quote(arg) for arg in shlex.split(lm_cmd))
        print(f"LM_EVAL_CMD=({quoted_lm_cmd})")

        tensor_parallel_size = cli_resolved_args.get("tensor-parallel-size", {})
        print(f"export TENSOR_PARALLEL_SIZE=\"{tensor_parallel_size}\"")
    else:
        srv_cmd_type = srv_opts.get("command_type", "")
        srv_resolved_args = resolve_device_args(srv_opts.get("args", {}), current_machine)
        cli_resolved_args = resolve_device_args(cli_opts.get("args", {}), current_machine)

        srv_cmd = build_command(srv_cmd_type, srv_resolved_args)
        cli_cmd = build_command(cli_cmd_type, cli_resolved_args)

        print("export COMMAND_TYPE=\"server_client\"")
        quoted_srv_cmd = ' '.join(
            shlex.quote(arg) for arg in shlex.split(srv_cmd))
        print(f"SERVER_CMD=({quoted_srv_cmd})")
        quoted_cli_cmd = ' '.join(
            shlex.quote(arg) for arg in shlex.split(cli_cmd))
        print(f"CLIENT_CMD=({quoted_cli_cmd})")

        tensor_parallel_size = srv_resolved_args.get("tensor-parallel-size", {})
        print(f"export TENSOR_PARALLEL_SIZE=\"{tensor_parallel_size}\"")


if __name__ == '__main__':
    main()
