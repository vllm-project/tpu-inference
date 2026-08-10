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
"""Dump the WRONG gsm8k generations from an lm_eval --log_samples run.

Usage: k3_dump_wrong_samples.py <lm_eval_output_dir>

Prints the first 8 samples whose strict extraction != target — question
tail, target, and the FULL untruncated generation (the failure mode may
be repetition or truncation, which live in the tail) — plus 2 correct
samples for contrast. A first-N dump is useless on a failing run whose
early questions happen to pass; filtering on wrongness is the point.
"""
import glob
import json
import sys


def rows(out_dir):
    for path in sorted(
            glob.glob(f"{out_dir}/**/samples_gsm8k*.jsonl", recursive=True)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def is_correct(row):
    em = row.get("exact_match")
    if em is None:
        m = row.get("metrics") or {}
        em = m.get("exact_match")
    return bool(em)


def show(row, idx, label):
    doc = row.get("doc") or {}
    q = (doc.get("question") or "")[-400:]
    target = row.get("target") or doc.get("answer") or ""
    resps = row.get("resps") or [[""]]
    gen = resps[0][0] if resps and resps[0] else ""
    filt = row.get("filtered_resps") or [""]
    print(f"===== [{label}] sample doc_id={row.get('doc_id', idx)} "
          f"extracted={filt[0]!r}")
    print(f"--- question (tail): {q}")
    print(f"--- target: {target}")
    print(f"--- generation (FULL, {len(gen)} chars):")
    print(gen)
    print("=====")


def main():
    out_dir = sys.argv[1]
    wrong = 0
    right = 0
    total = 0
    for row in rows(out_dir):
        total += 1
        if not is_correct(row) and wrong < 8:
            wrong += 1
            show(row, total, f"WRONG #{wrong}")
        elif is_correct(row) and right < 2:
            right += 1
            show(row, total, f"correct #{right}")
    print(f"[wrong-dump] scanned {total} samples, dumped {wrong} wrong + "
          f"{right} correct")
    if total == 0:
        print(f"[wrong-dump] NO samples found under {out_dir} — was "
              f"--log_samples/--output_path set?")


if __name__ == "__main__":
    main()
