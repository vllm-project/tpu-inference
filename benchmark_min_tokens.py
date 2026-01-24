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

import asyncio
from collections import Counter

import aiohttp
import numpy as np

# --- Configuration ---
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
URL = "http://localhost:8000/v1/chat/completions"
NUM_REQUESTS = 200  # Total requests to send
CONCURRENCY = 60  # Number of parallel requests (aiming for >50 QPS)
MIN_TOKENS = 30  # The setting we are testing


async def send_request(session):
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": "Say hello."
        }],  # Short prompt to hit cache easily
        "min_tokens": MIN_TOKENS,
        "temperature":
        0.0  # Deterministic to ensure variance is due to bug, not sampling
    }
    try:
        async with session.post(URL, json=payload) as response:
            if response.status != 200:
                print(f"Error: {response.status}")
                return 0
            data = await response.json()
            # Extract the actual number of tokens generated
            return data['usage']['completion_tokens']
    except Exception as e:
        print(f"Request failed: {e}")
        return 0


async def main():
    print("--- Starting Benchmark ---")
    print(f"Target: {URL}")
    print(f"Requests: {NUM_REQUESTS} | Concurrency: {CONCURRENCY}")
    print(f"Min Tokens Enforced: {MIN_TOKENS}")

    async with aiohttp.ClientSession() as session:
        tasks = []
        # Create a batch of concurrent tasks
        for _ in range(NUM_REQUESTS):
            tasks.append(send_request(session))
            # simple throttle to not blow up client immediately, but keep high QPS
            if len(tasks) % CONCURRENCY == 0:
                await asyncio.sleep(0.1)

        # Fire them off
        results = await asyncio.gather(*tasks)

    # --- Analysis ---
    results = [r for r in results if r > 0]  # Filter errors
    if not results:
        print("No successful requests.")
        return

    counts = Counter(results)
    mean_tokens = np.mean(results)
    p50 = np.percentile(results, 50)

    print("\n--- Results ---")
    print(f"Mean Token Length: {mean_tokens:.2f}")
    print(f"Median Token Length: {p50}")
    print("\nDistribution (Length : Count):")
    for length in sorted(counts.keys()):
        bar = "#" * counts[length]
        print(f"{length:3d} tokens: {counts[length]:3d} | {bar}")

    # Verdict
    if p50 < MIN_TOKENS:
        print("\n❌ BUG REPRODUCED: Median length is below min_tokens.")
    else:
        print("\n✅ PASSED: System respected min_tokens.")


if __name__ == "__main__":
    asyncio.run(main())
