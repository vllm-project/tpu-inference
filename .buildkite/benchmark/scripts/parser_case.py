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
    "lm_eval": "lm_eval",
    "benchmark_serving": "python3 scripts/bench_serving/benchmark_serving.py"
}


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
    else:
        case_data = data
        merged_env = case_data.get("env", {})

    # Export environment variables securely
    for k, v in merged_env.items():
        print(f"export {k}={shlex.quote(str(v))}")

    srv_opts = case_data.get("server_command_options", {})
    cli_opts = case_data.get("client_command_options", {})

    # Export specific environment for insert to db
    dataset = cli_opts.get("args", {}).get("dataset-name", {})
    print(f"export DATASET=\"{dataset}\"")
    additional_config = srv_opts.get("args", {}).get("additional-config", {})
    print(f"export ADDITIONAL_CONFIG=\"{additional_config}\"")
    model = srv_opts.get("args", {}).get("model", {})
    print(f"export MODEL=\"{model}\"")
    max_num_seqs = srv_opts.get("args", {}).get("max-num-seqs", {})
    print(f"export MAX_NUM_SEQS=\"{max_num_seqs}\"")
    max_num_batched_tokens = srv_opts.get("args", {}).get("max-num-batched-tokens", {})
    print(f"export MAX_NUM_BATCHED_TOKENS=\"{max_num_batched_tokens}\"")
    tensor_parallel_size = srv_opts.get("args", {}).get("tensor-parallel-size", {})
    print(f"export TENSOR_PARALLEL_SIZE=\"{tensor_parallel_size}\"")
    max_model_len = srv_opts.get("args", {}).get("max-model-len", {})
    print(f"export MAX_MODEL_LEN=\"{max_model_len}\"")

    # TODO:
    # EXTRA_ENVS
    # EXTRA_ARGS

    cli_cmd_type = cli_opts.get("command_type", "vllm_bench_serve")

    # Output execution strategy based on command_type
    if cli_cmd_type == "lm_eval":
        print("export RUN_TYPE=\"lm_eval\"")
        lm_cmd = build_command(cli_cmd_type, srv_opts.get("args", {}))
        quoted_lm_cmd = ' '.join(
            shlex.quote(arg) for arg in shlex.split(lm_cmd))
        print(f"LM_EVAL_CMD=({quoted_lm_cmd})")
    else:
        srv_cmd_type = srv_opts.get("command_type", "")
        srv_cmd = build_command(srv_cmd_type, srv_opts.get("args", {}))
        cli_cmd = build_command(cli_cmd_type, cli_opts.get("args", {}))

        print("export RUN_TYPE=\"server_client\"")
        quoted_srv_cmd = ' '.join(
            shlex.quote(arg) for arg in shlex.split(srv_cmd))
        print(f"SERVER_CMD=({quoted_srv_cmd})")
        quoted_cli_cmd = ' '.join(
            shlex.quote(arg) for arg in shlex.split(cli_cmd))
        print(f"CLIENT_CMD=({quoted_cli_cmd})")


if __name__ == '__main__':
    main()
