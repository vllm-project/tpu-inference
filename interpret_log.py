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

import ast
import dataclasses
import json
import sys
from typing import Any


@dataclasses.dataclass
class TuningKey:
    max_num_tokens: int
    actual_num_q_heads: int
    actual_lkv_dim: int
    actual_r_dim: int
    kv_dtype: str
    q_dtype: str
    total_num_pages: int
    page_size_per_kv_packing: int
    kv_packing: int
    max_num_seqs: int
    pages_per_seq: int
    sm_scale: float
    s_dtype: str
    case: str
    soft_cap: float | None
    mask_value: float | None
    chunk_prefill_size: int | None = None
    sliding_window: int | None = None
    p_same_dtype_as_v: bool = True


@dataclasses.dataclass
class TunableParams:
    decode_batch_size: int
    num_kv_pages_per_block: int | tuple[int, int, int]
    num_queries_per_block: int | tuple[int, int, int]
    vmem_limit_bytes: int


def _split_kv_pairs(payload: str) -> list[str]:
    pairs: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in payload:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                pairs.append(part)
            buf = []
            continue
        buf.append(ch)
    part = "".join(buf).strip()
    if part:
        pairs.append(part)
    return pairs


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def parse_payload(payload: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for part in _split_kv_pairs(payload):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        result[key.strip()] = _parse_value(value)
    return result


def infer_case(decode_batch_size: int, max_num_tokens: int) -> str:
    if decode_batch_size <= 1:
        return "decode_only"
    if decode_batch_size == max_num_tokens:
        return "batched_decode"
    return "mixed"


def build_tuning_key(group: dict[str, dict[str, Any]]) -> TuningKey:
    arrays = group["arrays"]
    scalars = group["scalars"]
    kernel = group["kernel"]

    ql_nope = arrays["ql_nope"]
    q_pe = arrays["q_pe"]
    cache_kv = arrays["cache_kv"]
    kv_lens = arrays["kv_lens"]
    page_indices = arrays["page_indices"]

    max_num_tokens = int(ql_nope[0])
    max_num_seqs = int(kv_lens[0])
    pages_per_seq = int(page_indices[0]) // max_num_seqs if max_num_seqs else 0

    return TuningKey(
        max_num_tokens=max_num_tokens,
        actual_num_q_heads=int(q_pe[1]),
        actual_lkv_dim=int(ql_nope[2]),
        actual_r_dim=int(q_pe[2]),
        kv_dtype="fp8",
        q_dtype="fp8",
        total_num_pages=int(cache_kv[0]),
        page_size_per_kv_packing=int(cache_kv[1]),
        kv_packing=int(cache_kv[2]),
        max_num_seqs=max_num_seqs,
        pages_per_seq=pages_per_seq,
        sm_scale=float(scalars["sm_scale"]),
        s_dtype="bf16",
        case=infer_case(int(kernel["decode_batch_size"]), max_num_tokens),
        soft_cap=scalars.get("soft_cap"),
        mask_value=scalars.get("mask_value"),
        chunk_prefill_size=scalars.get("chunk_prefill_size"),
        sliding_window=scalars.get("sliding_window"),
        p_same_dtype_as_v=True,
    )


def build_tunable_params(group: dict[str, dict[str, Any]]) -> TunableParams:
    kernel = group["kernel"]
    return TunableParams(
        decode_batch_size=int(kernel["decode_batch_size"]),
        num_kv_pages_per_block=kernel["num_kv_pages_per_blocks"],
        num_queries_per_block=kernel["num_queries_per_blocks"],
        vmem_limit_bytes=int(kernel["vmem_limit_bytes"]),
    )


def parse_log_file(log_path: str) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    curr: dict[str, dict[str, Any]] = {}
    curr_raw: dict[str, str] = {}

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "Array shapes:" in line:
                payload = line.split("Array shapes:", 1)[1].strip()
                curr["arrays"] = parse_payload(payload)
                curr_raw["array_line"] = line.rstrip("\n")
            elif "Scalar parameters:" in line:
                payload = line.split("Scalar parameters:", 1)[1].strip()
                curr["scalars"] = parse_payload(payload)
                curr_raw["scalar_line"] = line.rstrip("\n")
            elif "Kernel optimization params:" in line:
                payload = line.split("Kernel optimization params:",
                                     1)[1].strip()
                curr["kernel"] = parse_payload(payload)
                curr_raw["kernel_line"] = line.rstrip("\n")

            if {"arrays", "scalars", "kernel"}.issubset(curr):
                tuning_key = build_tuning_key(curr)
                tunable_params = build_tunable_params(curr)
                record = {
                    "arrays": curr["arrays"],
                    "scalars": curr["scalars"],
                    "kernel_optimization": curr["kernel"],
                    "tuning_key": dataclasses.asdict(tuning_key),
                    "tunable_params": dataclasses.asdict(tunable_params),
                }
                groups.append(record)
                print(curr_raw["array_line"])
                print(curr_raw["scalar_line"])
                print(curr_raw["kernel_line"])
                print(f"TuningKey={tuning_key} TunableParams={tunable_params}")
                print()
                curr = {}
                curr_raw = {}

    return groups


def main() -> None:
    input_log = sys.argv[1] if len(sys.argv) > 1 else "ds_r1_kernel.log"
    output_json = sys.argv[2] if len(
        sys.argv) > 2 else "parsed_ds_r1_kernel.json"

    groups = parse_log_file(input_log)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2)

    print(f"Saved {len(groups)} parsed groups to {output_json}")


if __name__ == "__main__":
    main()
