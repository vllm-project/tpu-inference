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

import argparse
import json
import re
import shlex
import sys


class DbMetadataParser:

    def __init__(self):
        # Fields mapped to the RunRecord table schema
        self.record = {
            "model": None,
            "tensor_parallel_size": 1,
            "kv_cache_dtype": "auto",
            "max_model_len": None,
            "max_num_seqs": None,
            "max_num_batched_tokens": None,
            "input_len": None,
            "output_len": None,
            "prefix_len": 0
        }

    def _get_val(self, tokens, i):
        token = tokens[i]
        if "=" in token:
            return token.split("=", 1)[1]
        elif i + 1 < len(tokens):
            return tokens[i + 1]
        return None

    def parse_serve(self, cmd):
        if not cmd:
            return
        tokens = shlex.split(cmd)
        # Positional model detection
        if "serve" in tokens:
            idx = tokens.index("serve")
            if idx + 1 < len(tokens) and not tokens[idx + 1].startswith(
                    "-") and "=" not in tokens[idx + 1]:
                self.record["model"] = tokens[idx + 1]

        for i, t in enumerate(tokens):
            f = t.split("=")[0]
            if f == "--model":
                self.record["model"] = self._get_val(tokens, i)
            elif f == "--tensor-parallel-size":
                self.record["tensor_parallel_size"] = int(
                    self._get_val(tokens, i))
            elif f == "--kv-cache-dtype":
                self.record["kv_cache_dtype"] = self._get_val(tokens, i)
            elif f == "--max-model-len":
                self.record["max_model_len"] = int(self._get_val(tokens, i))
            elif f == "--max-num-seqs":
                self.record["max_num_seqs"] = int(self._get_val(tokens, i))
            elif f == "--max-num-batched-tokens":
                self.record["max_num_batched_tokens"] = int(
                    self._get_val(tokens, i))

    def parse_client(self, cmd):
        if not cmd:
            return
        tokens = shlex.split(cmd)
        for i, t in enumerate(tokens):
            f = t.split("=")[0]
            if f == "--dataset-path":
                path = self._get_val(tokens, i)
                match = re.search(
                    r"inlen(?P<in>\d+)_outlen(?P<out>\d+)_prefixlen(?P<prefix>\d+)",
                    path)
                if match:
                    self.record["input_len"] = int(match.group("in"))
                    self.record["output_len"] = int(match.group("out"))
                    self.record["prefix_len"] = int(match.group("prefix"))
            elif f == "--custom-output-len":
                self.record["output_len"] = int(self._get_val(tokens, i))

    def validate(self):
        # Mandatory fields for Spanner DB categorization
        mandatory = ["model", "input_len", "output_len"]
        missing = [k for k in mandatory if self.record.get(k) is None]
        if missing:
            print(f"Error: Missing DB mandatory parameters: {missing}",
                  file=sys.stderr)
            return False
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", required=True)
    parser.add_argument("--client", required=True)
    args = parser.parse_args()
    p = DbMetadataParser()
    p.parse_serve(args.serve)
    p.parse_client(args.client)
    if not p.validate():
        sys.exit(1)
    print(json.dumps(p.record))
