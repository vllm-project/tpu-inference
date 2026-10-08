#!/usr/bin/env python3
"""Evaluate GSM8K accuracy against local vLLM chat completions endpoint."""

import argparse
import asyncio
import json
import re
import sys
import urllib.request

try:
    import aiohttp
except ImportError:
    aiohttp = None


def extract_answer(text: str) -> str:
    """Extract numeric answer from text, looking for #### <number> or last number."""
    # Look for #### <number>
    match = re.search(r"####\s*(-?[0-9\.,]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()

    # Look for \boxed{<number>}
    match = re.search(r"\\boxed\{([^{}]+)\}", text)
    if match:
        cand = match.group(1).replace(",", "").strip()
        num_m = re.search(r"-?[0-9\.]+", cand)
        if num_m:
            return num_m.group(0)

    # Fallback to last number in text
    nums = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", text)
    if nums:
        return nums[-1]
    return ""


def normalize_num(s: str) -> str:
    try:
        val = float(s)
        if val.is_integer():
            return str(int(val))
        return f"{val:.4f}"
    except (ValueError, TypeError):
        return s.strip()


async def query_model(session, base_url: str, model: str, question: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Please reason step by step, and put your final answer within #### <number>."},
            {"role": "user", "content": question},
        ],
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    async with session.post(base_url, json=payload, timeout=60) as resp:
        if resp.status != 200:
            err = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {err}")
        res = await resp.json()
        msg = res["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        if reasoning and not content:
            return reasoning
        elif reasoning and content:
            return f"{reasoning}\n{content}"
        return content


async def run_evaluation(base_url: str, model: str, limit: int, concurrency: int, output_file: str = "gsm8k_eval_results.json"):
    dataset_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    print(f"Downloading GSM8K dataset from {dataset_url}...")
    req = urllib.request.urlopen(dataset_url)
    lines = [json.loads(line) for line in req.read().decode('utf-8').splitlines() if line.strip()]
    if limit > 0:
        lines = lines[:limit]

    print(f"Loaded {len(lines)} test examples. Concurrency: {concurrency}")

    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async with aiohttp.ClientSession() as session:
        async def evaluate_item(idx, item):
            question = item["question"]
            ground_truth = extract_answer(item["answer"])
            async with semaphore:
                try:
                    output = await query_model(session, base_url, model, question)
                    pred = extract_answer(output)
                    correct = normalize_num(pred) == normalize_num(ground_truth)
                    return {"idx": idx, "correct": correct, "pred": pred, "gt": ground_truth, "error": None, "raw_output": output, "question": question}
                except Exception as e:
                    return {"idx": idx, "correct": False, "pred": None, "gt": ground_truth, "error": str(e), "raw_output": None, "question": question}

        tasks = [evaluate_item(i, item) for i, item in enumerate(lines)]
        completed = 0
        correct_count = 0
        for fut in asyncio.as_completed(tasks):
            res = await fut
            completed += 1
            if res["correct"]:
                correct_count += 1
            if completed % 10 == 0 or completed == len(tasks):
                print(f"Progress: {completed}/{len(tasks)} | Correct: {correct_count}/{completed} ({correct_count/completed*100:.2f}%)")
            results.append(res)

    results.sort(key=lambda r: r["idx"])
    accuracy = (correct_count / len(results)) * 100.0

    print("\n" + "=" * 60)
    print("GSM8K Evaluation Results")
    print("=" * 60)
    print(f"Model:                {model}")
    print(f"Endpoint:             {base_url}")
    print(f"Total Evaluated:      {len(results)}")
    print(f"Correct:              {correct_count}")
    print(f"Accuracy:             {accuracy:.2f}%")
    print("=" * 60)

    # Save detailed results to JSON
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {output_file}")

    # Show first 5 samples
    print("\nSample Predictions:")
    for r in results[:5]:
        status = "PASS" if r["correct"] else "FAIL"
        print(f"  [{status}] Example #{r['idx']}: GT={r['gt']}, Pred={r['pred']} (error: {r['error']})")
        if not r["correct"] and r["raw_output"]:
            print(f"    Raw Output Preview: {r['raw_output'][:300]!r}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate GSM8K accuracy on vLLM server")
    parser.add_argument("--base-url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--output", default="gsm8k_eval_results.json")
    args = parser.parse_args()

    acc = asyncio.run(run_evaluation(args.base_url, args.model, args.limit, args.concurrency, output_file=args.output))
    if acc < 20.0:
        print(f"WARNING: Accuracy {acc:.2f}% is lower than expected for Qwen3.5-35B.")


if __name__ == "__main__":
    main()
