# SPDX-License-Identifier: Apache-2.0
"""Builds a synthetic multi-turn dataset from R2E-Gym for benchmark_agentic.py.

R2E-Gym ships SWE task definitions, not trajectories: every row has an empty
`input` field, so there is nothing to replay. What each row does carry is real
content -- a GitHub-issue style problem statement, the pre-fix source files, and
the recorded stdout of the test suite before and after the fix.

This script turns that content into a scripted multi-turn conversation per task:
a realistic agent prompt, plus a fixed sequence of environment observations
built from the recorded tool output. The model still generates every assistant
turn live against the server, so the token distribution of the generated side is
real; only the environment side is pre-baked. That is the point -- it exercises
the serving stack under RL-shaped traffic without needing containers, test
execution, or a reward loop.

Usage:
    python prepare_r2e_dataset.py \
        --dataset hfilaretov/Benchmark-R2E-Gym-Easy \
        --split val \
        --model-path-or-id Qwen/Qwen3-4B \
        --output r2e_easy_val.jsonl
"""

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

# Tool block mirroring the R2E-Gym agent scaffold. This is prompt ballast that
# every real SWE rollout carries, so it belongs in the shared prefix.
SYSTEM_PROMPT = """You are a programming agent who is provided a GitHub issue \
and repository bash environment and is tasked to solve certain tasks (e.g., \
file localization, testing, code repair, and editing) to resolve the issue.

You have access to the following functions:

---- BEGIN FUNCTION #1: file_editor ----
Description: Custom editing tool for viewing, creating and editing files
  *  State is persistent across command calls and discussions with the user
  *  If `path` is a file, `view` displays the result of applying `cat -n`
  *  The `create` command cannot be used if the specified `path` already exists
  *  If a `command` generates a long output, it will be truncated
  *  The `undo_edit` command will revert the last edit made to the file at `path`

Parameters:
  (1) command (string, required): The command to run. Allowed: `view`, `create`,
      `str_replace`, `insert`, `undo_edit`.
  (2) path (string, required): Absolute path to file or directory
  (3) file_text (string, optional): Required for `create`
  (4) old_str (string, optional): Required for `str_replace`
  (5) new_str (string, optional): Replacement string
  (6) insert_line (integer, optional): Required for `insert`
  (7) view_range (array, optional): Line range to view
---- END FUNCTION #1 ----

---- BEGIN FUNCTION #2: execute_bash ----
Description: Execute a bash command in the terminal.
Parameters:
  (1) cmd (string, required): The bash command to execute.
---- END FUNCTION #2 ----

---- BEGIN FUNCTION #3: search ----
Description: Search for a term in a directory or a single file.
Parameters:
  (1) search_term (string, required): The term to search for.
  (2) path (string, optional): The file or directory to search in.
---- END FUNCTION #3 ----

---- BEGIN FUNCTION #4: finish ----
Description: Signals the completion of the current task.
Parameters:
  (1) command (string, required): The command to run: `submit`.
---- END FUNCTION #4 ----

Follow this format:
<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
</function>

Your thinking should be thorough. Take the following steps:
1. Explore the repository to familiarize yourself with its structure.
2. Create a script to reproduce the issue and execute it.
3. Edit the source code to resolve the issue.
4. Rerun your reproduce script to confirm the fix.
5. Consider edge cases and ensure your fix handles them.
"""


def load_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Loads raw R2E-Gym rows from a local JSONL file or the HF Hub.

    Args:
        args: Parsed command line arguments.

    Returns:
        List[Dict[str, Any]]: Raw dataset rows.
    """
    if args.input_jsonl:
        with open(args.input_jsonl) as f:
            return [json.loads(line) for line in f if line.strip()]

    from huggingface_hub import hf_hub_download

    split = "val" if args.split in ("val", "validation") else "train"
    path = hf_hub_download(
        repo_id=args.dataset,
        filename=f"benchmark_r2e_gym_easy_{split}.jsonl",
        repo_type="dataset",
    )
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def source_files(instance: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extracts the pre-fix source files touched by an R2E-Gym instance.

    Args:
        instance: Decoded `instance_dict` for one task.

    Returns:
        List[Dict[str, str]]: File dicts with `path` and `content` keys.
    """
    try:
        parsed = json.loads(instance["parsed_commit_content"])
    except (KeyError, json.JSONDecodeError):
        return []

    files = []
    for diff in parsed.get("file_diffs", []):
        content = diff.get("old_file_content")
        if not content:
            continue
        # `new_file_path` is unset in this dataset; the path lives on the diff
        # header, with minus/plus as the a/ b/ prefixed fallbacks.
        header = diff.get("header") or {}
        path = (header.get("file") or {}).get("path")
        if not path:
            minus = diff.get("minus_file") or {}
            path = (minus.get("path") or "").removeprefix("a/")
        if path:
            files.append({"path": path, "content": content})
    return files


def build_initial_prompt(
    instance: Dict[str, Any],
    tokenizer: AutoTokenizer,
    target_tokens: int,
) -> Optional[str]:
    """Builds one realistic agent prompt padded to a target token length.

    The problem statement alone is only ~250-700 tokens, far short of the 4k-16k
    a real SWE rollout carries. Real agents reach that length by pulling repo
    context into the prompt, so pad with the instance's own source files rather
    than filler, then trim to the target.

    Args:
        instance: Decoded `instance_dict` for one task.
        tokenizer: Tokenizer used to measure and trim the prompt.
        target_tokens: Desired prompt length in tokens.

    Returns:
        Optional[str]: The prompt, or None if the instance has no usable source.
    """
    files = source_files(instance)
    if not files:
        return None

    repo = instance.get("repo", "unknown/repo")
    parts = [
        f"I've uploaded a python code repository in the directory /testbed.\n"
        f"Repository: {repo}\n\n"
        f"Consider the following issue description:\n\n"
        f"<issue_description>\n{instance.get('problem_statement', '')}\n"
        f"</issue_description>\n\n"
        f"Can you help me implement the necessary changes to the repository so "
        f"that the requirements specified in the issue description are met?\n\n"
        f"Here is the current state of the relevant files:\n"
    ]

    for f in files:
        parts.append(f"\n<file path=\"/testbed/{f['path']}\">\n"
                     f"{f['content']}\n</file>\n")

    prompt = "".join(parts)
    ids = tokenizer.encode(prompt)

    # Repeat real repo context until the target is reached. Cycling the same
    # files keeps the content real; a prompt that is short stays short rather
    # than being padded with junk.
    if len(ids) < target_tokens and files:
        idx = 0
        while len(ids) < target_tokens and idx < 200:
            f = files[idx % len(files)]
            prompt += (f"\n<file path=\"/testbed/{f['path']}\" "
                       f"context_ref=\"{idx}\">\n{f['content']}\n</file>\n")
            ids = tokenizer.encode(prompt)
            idx += 1

    if len(ids) > target_tokens:
        prompt = tokenizer.decode(ids[:target_tokens],
                                  skip_special_tokens=True)

    return prompt


# A real `file_editor view` returns a bounded line range and truncates beyond
# it, so views stay roughly this size regardless of how large the file is.
VIEW_LINES = 120
MAX_VIEWS_PER_FILE = 4

# Agent harnesses clip long command output before it reaches the model. Without
# this, a recorded numpy run over 4k tests would feed back a single ~120k-token
# observation -- far past anything a real rollout sees.
MAX_BASH_OUTPUT_CHARS = 6000


def clip_output(text: str) -> str:
    """Clips command output the way an agent harness would.

    Keeps the head and tail, which is where the run summary and the failure
    report live, and drops the middle.

    Args:
        text: Raw recorded command output.

    Returns:
        str: Output clipped to roughly MAX_BASH_OUTPUT_CHARS.
    """
    if len(text) <= MAX_BASH_OUTPUT_CHARS:
        return text
    half = MAX_BASH_OUTPUT_CHARS // 2
    omitted = len(text) - 2 * half
    return (f"{text[:half]}\n\n<response clipped: {omitted} characters "
            f"omitted>\n\n{text[-half:]}")


def chunk_file(content: str) -> List[str]:
    """Splits file content into view-sized chunks, as a real editor would.

    Args:
        content: Full file content.

    Returns:
        List[str]: Up to MAX_VIEWS_PER_FILE chunks of VIEW_LINES lines each,
        rendered with line numbers like `cat -n`.
    """
    lines = content.splitlines()
    if not lines:
        return []

    chunks = []
    for start in range(0, len(lines), VIEW_LINES):
        window = lines[start:start + VIEW_LINES]
        body = "\n".join(f"{start + i + 1:6d}\t{line}"
                         for i, line in enumerate(window))
        remaining = len(lines) - (start + len(window))
        if remaining > 0:
            body += (f"\n<response clipped> {remaining} lines below. Use "
                     f"`view_range` to see more.")
        chunks.append(body)
        if len(chunks) >= MAX_VIEWS_PER_FILE:
            break
    return chunks


def build_env_turns(instance: Dict[str, Any], num_turns: int) -> List[str]:
    """Builds a sequence of environment observations from recorded output.

    Every observation is real recorded content: directory listings from the
    instance's file list, file views from the pre-fix sources, and test output
    captured from actual runs (failing before the fix, passing after). The
    sequence follows the shape of a real SWE rollout -- explore, view, edit,
    test, repeat -- and ends on the passing run.

    Args:
        instance: Decoded `instance_dict` for one task.
        num_turns: Number of observations to emit.

    Returns:
        List[str]: One observation per turn.
    """
    files = source_files(instance)
    try:
        execution = json.loads(instance["execution_result_content"])
    except (KeyError, json.JSONDecodeError):
        execution = {}

    failing = execution.get("old_commit_res_stdout", "")
    passing = execution.get("new_commit_res_stdout", "")
    setup = execution.get("setup_res_stderr", "")
    modified = instance.get("modified_files") or []
    if isinstance(modified, str):
        modified = json.loads(modified.replace("'", '"'))

    listing = "\n".join(f"/testbed/{p}" for p in modified) or "/testbed"
    views = []
    for f in files:
        views.extend(chunk_file(f["content"]))
    if not views:
        views = ["(file is empty)"]

    observations = []
    for turn in range(num_turns):
        # Cycle explore -> view -> edit -> test, the shape of a real rollout.
        kind = turn % 4
        if kind == 0:
            observations.append(
                f"Execution output of [execute_bash]:\n"
                f"find /testbed -name '*.py' | head -50\n{listing}\n"
                f"{setup[:600]}")
        elif kind == 1:
            observations.append("Execution output of [file_editor]:\n"
                                f"{views[turn % len(views)]}")
        elif kind == 2:
            path = modified[0] if modified else "file.py"
            observations.append(
                f"Execution output of [file_editor]:\nThe file /testbed/{path} "
                f"has been edited. Here's the result of running `cat -n` on a "
                f"snippet:\n{views[(turn + 1) % len(views)][:800]}\n"
                f"Review the changes and make sure they are correct.")
        else:
            # Real pytest output. Failing runs dominate a rollout; the last
            # test turn lands on the recorded passing run.
            is_last_test = turn >= num_turns - 2
            out = passing if (is_last_test and passing) else failing
            if not out:
                out = passing or "(no test output recorded)"
            observations.append(
                f"Execution output of [execute_bash]:\n{clip_output(out)}")

    return observations


def main():
    """Parses arguments and writes the converted dataset."""
    parser = argparse.ArgumentParser(
        description="Convert R2E-Gym task definitions into a synthetic "
        "multi-turn dataset for benchmark_agentic.py.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="hfilaretov/Benchmark-R2E-Gym-Easy",
        help="Hugging Face dataset ID to download.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default=None,
        help="Local R2E-Gym JSONL to convert instead of downloading.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "validation"],
        help="Dataset split to convert.",
    )
    parser.add_argument(
        "--model-path-or-id",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Model path or HF ID for the tokenizer used to size prompts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=None,
        help="Convert only the first N instances.",
    )
    parser.add_argument(
        "--initial-prompt-len-min",
        type=int,
        default=4000,
        help="Minimum initial prompt length in tokens.",
    )
    parser.add_argument(
        "--initial-prompt-len-max",
        type=int,
        default=16000,
        help="Maximum initial prompt length in tokens.",
    )
    parser.add_argument(
        "--turns-min",
        type=int,
        default=10,
        help="Minimum conversation turns per instance.",
    )
    parser.add_argument(
        "--turns-max",
        type=int,
        default=100,
        help="Maximum conversation turns per instance.",
    )
    parser.add_argument(
        "--env-len-max",
        type=int,
        default=0,
        help="Truncate each environment observation to this many tokens. "
        "0 keeps the real recorded length, which is what a real rollout "
        "feeds back.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print(f"Loading tokenizer from: {args.model_path_or_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_id,
                                              trust_remote_code=True)

    print("Loading R2E-Gym rows...")
    rows = load_rows(args)
    if args.num_instances:
        rows = rows[:args.num_instances]
    print(f"Converting {len(rows)} instances...")

    written = 0
    skipped = 0
    prompt_lens = []
    env_lens = []

    with open(args.output, "w") as out:
        for i, row in enumerate(rows):
            params = row.get("responses_create_params", {})
            metadata = params.get("metadata", {})

            if params.get("input"):
                # Nothing in this dataset has one, but if a future revision
                # ships real trajectories, do not silently overwrite them.
                print(f"  [{i}] has a real trajectory; converter would "
                      f"discard it. Skipping.")
                skipped += 1
                continue

            try:
                instance = json.loads(metadata["instance_dict"])
            except (KeyError, json.JSONDecodeError):
                skipped += 1
                continue

            target = rng.randint(args.initial_prompt_len_min,
                                 args.initial_prompt_len_max)
            prompt = build_initial_prompt(instance, tokenizer, target)
            if prompt is None:
                skipped += 1
                continue

            num_turns = rng.randint(args.turns_min, args.turns_max)
            env_turns = build_env_turns(instance, num_turns)

            if args.env_len_max:
                trimmed = []
                for text in env_turns:
                    ids = tokenizer.encode(text)
                    if len(ids) > args.env_len_max:
                        text = tokenizer.decode(ids[:args.env_len_max],
                                                skip_special_tokens=True)
                    trimmed.append(text)
                env_turns = trimmed

            prompt_tokens = len(tokenizer.encode(prompt))
            prompt_lens.append(prompt_tokens)
            env_lens.extend(len(tokenizer.encode(t)) for t in env_turns)

            out.write(
                json.dumps({
                    "instance_id": instance.get("instance_id"),
                    "repo": instance.get("repo"),
                    "system_prompt": SYSTEM_PROMPT,
                    "initial_prompt": prompt,
                    "initial_prompt_tokens": prompt_tokens,
                    "num_turns": num_turns,
                    "env_turns": env_turns,
                }) + "\n")
            written += 1

            if written % 25 == 0:
                print(f"  converted {written}/{len(rows)}")

    if not written:
        print("No instances converted.", file=sys.stderr)
        sys.exit(1)

    def pct(values: List[int], p: float) -> int:
        return sorted(values)[min(len(values) - 1, int(len(values) * p))]

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nWrote {written} instances ({skipped} skipped) to {args.output} "
          f"({size_mb:.1f} MB)")
    print(f"  initial prompt tokens: min {min(prompt_lens)} "
          f"p50 {pct(prompt_lens, 0.5)} max {max(prompt_lens)}")
    print(f"  env observation tokens: min {min(env_lens)} "
          f"p50 {pct(env_lens, 0.5)} p99 {pct(env_lens, 0.99)} "
          f"max {max(env_lens)}")
    print(f"  total env observations: {len(env_lens)}")


if __name__ == "__main__":
    main()
