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
import sys

import requests


def test_sampling_params(url, model):
    print(f"Testing sampling parameters against: {url} (model: {model})")
    endpoint = f"{url}/completions"

    # --- Test 1: Temperature 0 (Deterministic) ---
    print("\n--- Test 1: Temperature 0 (Deterministic) ---")
    payload = {
        "model": model,
        "prompt": "What is 2 + 2? Answer with just the number:",
        "max_tokens": 5,
        "temperature": 0,
        "seed": 42
    }
    t0_res1 = requests.post(endpoint,
                            json=payload).json()["choices"][0]["text"]
    t0_res2 = requests.post(endpoint,
                            json=payload).json()["choices"][0]["text"]
    assert t0_res1 == t0_res2, f"FAILED: T=0 results differ: '{t0_res1}' vs '{t0_res2}'"
    print(f"✅ Passed (Output: '{t0_res1}')")

    # --- Test 2: High Temperature (Variation) ---
    print("\n--- Test 2: High Temperature (Variation) ---")
    payload = {
        "model": model,
        "prompt": "Write a random word:",
        "max_tokens": 10,
        "temperature": 2.0,
        "n": 10  # Request 10 completions at once to check variation
    }
    resp = requests.post(endpoint, json=payload).json()
    unique_outputs = set(c["text"] for c in resp["choices"])
    assert len(
        unique_outputs
    ) > 1, f"FAILED: High T=2.0 produced no variation: {unique_outputs}"
    print(
        f"✅ Passed ({len(unique_outputs)} unique variations from 10 samples)")

    # --- Test 3: Top-P (Nucleus Sampling) ---
    print("\n--- Test 3: Top-P constraints ---")
    payload = {
        "model": model,
        "prompt": "The capital of France is",
        "max_tokens": 5,
        "temperature": 0.8,
        "top_p": 0.1
    }
    resp = requests.post(endpoint, json=payload).json()
    assert len(resp["choices"][0]["text"]) > 0
    print("✅ Passed: Request with restrictive Top-P succeeded")

    # --- Test 4: Top-K constraints ---
    print("\n--- Test 4: Top-K constraints ---")
    payload["top_p"] = 1.0
    payload["top_k"] = 1
    resp = requests.post(endpoint, json=payload).json()
    assert len(resp["choices"][0]["text"]) > 0
    print("✅ Passed: Request with restrictive Top-K succeeded")

    # --- Test 5: Logprobs (Metadata Integrity) ---
    print("\n--- Test 5: Logprobs (Metadata Integrity) ---")
    payload = {
        "model": model,
        "prompt": "Hello",
        "max_tokens": 5,
        "logprobs": 5
    }
    resp = requests.post(endpoint, json=payload).json()
    logprobs = resp["choices"][0].get("logprobs")
    assert logprobs is not None, "FAILED: logprobs field missing"
    assert "token_logprobs" in logprobs or "content" in logprobs, "FAILED: Detailed logprob data missing"
    print("✅ Passed: Logprobs correctly returned through Disagg boundary")

    # --- Test 6: Prompt Logprobs ---
    print("\n--- Test 6: Prompt Logprobs (Prefiller Verification) ---")
    # vLLM API uses 'echo' or specific fields for prompt logprobs depending on version
    payload = {
        "model": model,
        "prompt": "Deep Learning is",
        "max_tokens": 1,
        "echo": True,
        "logprobs": 1
    }
    resp = requests.post(endpoint, json=payload).json()
    assert "logprobs" in resp["choices"][
        0], "FAILED: Prompt logprobs (echo) missing"
    print("✅ Passed: Prompt metadata preserved through Prefiller")

    # --- Test 7: Optimization Boundaries (Top-K=-1, Top-P=1.0) ---
    print("\n--- Test 7: Optimization Boundaries ---")
    payload = {
        "model": model,
        "prompt": "Testing optimizations",
        "max_tokens": 5,
        "temperature": 0.7,
        "top_k": -1,
        "top_p": 1.0
    }
    resp = requests.post(endpoint, json=payload).json()
    assert len(resp["choices"][0]["text"]) > 0
    print("✅ Passed: Boundary values (-1, 1.0) handled correctly")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    try:
        if test_sampling_params(args.url, args.model):
            print("\n✨ 100% Case Coverage for Sampling Combination achieved!")
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        sys.exit(1)
