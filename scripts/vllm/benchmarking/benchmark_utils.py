# Copied from vLLM: https://github.com/vllm-project/vllm/blob/02f0c7b/benchmarks/benchmark_utils.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This module provides utility functions for benchmarking vLLM.
"""

import argparse
import json
import math
import os
import re
from typing import Any, List, Tuple

import evaluate
import nltk
import numpy as np
from backend_request_func import RequestFuncOutput
from benchmark_dataset import SampleRequest
from math_utils import extract_numbers, post_processing_math_ans, sympify_set


def convert_to_pytorch_benchmark_format(args: argparse.Namespace,
                                        metrics: dict[str, list],
                                        extra_info: dict[str, Any]) -> list:
    """
    Save the benchmark results in the format used by PyTorch OSS benchmark with
    on metric per record
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    records = []
    if not os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):
        return records

    for name, benchmark_values in metrics.items():
        record = {
            "benchmark": {
                "name": "vLLM benchmark",
                "extra_info": {
                    "args": vars(args),
                },
            },
            "model": {
                "name": args.model,
            },
            "metric": {
                "name": name,
                "benchmark_values": benchmark_values,
                "extra_info": extra_info,
            },
        }

        tp = record["benchmark"]["extra_info"]["args"].get(
            "tensor_parallel_size")
        # Save tensor_parallel_size parameter if it's part of the metadata
        if not tp and "tensor_parallel_size" in extra_info:
            record["benchmark"]["extra_info"]["args"][
                "tensor_parallel_size"] = (extra_info["tensor_parallel_size"])

        records.append(record)

    return records


class InfEncoder(json.JSONEncoder):

    def clear_inf(self, o: Any):
        if isinstance(o, dict):
            return {k: self.clear_inf(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self.clear_inf(v) for v in o]
        elif isinstance(o, float) and math.isinf(o):
            return "inf"
        return o

    def iterencode(self, o: Any, *args, **kwargs) -> Any:
        return super().iterencode(self.clear_inf(o), *args, **kwargs)


def write_to_json(filename: str, records: list) -> None:
    with open(filename, "w") as f:
        json.dump(
            records,
            f,
            cls=InfEncoder,
            default=lambda o:
            f"<{type(o).__name__} object is not JSON serializable>",
        )


def postprocess_text_mmlu(preds: List[str],
                          targets: List[str]) -> Tuple[List[int], List[int]]:
    """
    Postprocess the generated text to get the predicted and target answers for the MMLU dataset.

    Args:
        preds (List[str]): List of generated text
        targets (List[str]): List of target text

    Returns:
        Tuple[List[int], List[int]]: List of predicted answers and list of target answers"""
    choices = ["A", "B", "C", "D", None]

    def _parse_answer(output):
        # ? marks the close parenthesis as optional
        re_str = r"\s*\(([A-D])\)?\s*\w*"
        match = re.search(re_str, output, re.IGNORECASE)
        predicted_answer = match.group(1).upper() if match else None
        return predicted_answer

    preds = [choices.index(_parse_answer(pred.strip())) for pred in preds]
    targets = [choices.index(target.strip().upper()) for target in targets]
    return preds, targets


def extract_boxed_answers(text):
    pieces = text.split("boxed{")
    if len(pieces) == 1:
        return [""]
    piece = pieces[1]
    ans = []
    for piece in pieces[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == "{":
                n += 1
            elif piece[i] == "}":
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == "%":
                        ans.append(piece[:i + 1])
                        break
                    else:
                        ans.append(piece[:i])
                        break
    if ans:
        return ans
    else:
        return [""]


def extract_answer(pred_str, exhaust=False):
    pred = []
    if "boxed{" in pred_str:
        pred = extract_boxed_answers(pred_str)
    elif "Answer:" in pred_str:
        matches = re.findall(r"Answer:[\*]*\s+(\S*.*)", pred_str)
        if matches:
            pred = [extract_numbers(matches[-1])]
    elif "the answer is" in pred_str:
        pred = [extract_numbers(pred_str.split("the answer is")[-1].strip())]
    elif "final answer is $" in pred_str and "$. I hope" in pred_str:
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = [tmp.split("$. I hope", 1)[0].strip()]
    else:  # use the last number
        pattern = r"-?\d*\.?\d+"
        ans = re.findall(pattern, pred_str.replace(",", ""))
        if len(ans) >= 1:
            ans = ans[-1]
        else:
            ans = ""
        if ans:
            pred.append(ans)
    # multiple line
    pred_list = []
    for ans in pred:
        ans = ans.replace("<|end_of_text|>", "")
        ans = ans.strip().split("\n")[0]
        ans = ans.lstrip(":")
        ans = ans.lstrip("$")
        ans = ans.rstrip("$")
        ans = ans.rstrip(".")
        ans = ans.rstrip("/")
        pred_list.append(ans)
    if exhaust:
        return pred_list
    else:
        return pred_list[-1] if pred_list else ""


def eval_accuracy_math_500(request_outputs: List[RequestFuncOutput]) -> dict:
    """
    Evaluate accuracy for Math500 dataset.
    """
    preds = []
    targets = []
    for output in request_outputs:
        preds.append(output.generated_text)
        targets.append(output.input_request.completion)

    correct_ans = 0
    wrong_ans = 0
    for p, t in zip(preds, targets):

        p = extract_answer(p)
        ans_set = post_processing_math_ans(p)
        sympified_ans_set = sympify_set(ans_set)

        target_set = post_processing_math_ans(t)
        sympified_target_set = sympify_set(target_set)

        if sympified_target_set == sympified_ans_set:
            correct_ans += 1
            continue
        wrong_ans += 1
    total_ans = correct_ans + wrong_ans
    result = {}
    result["literal"] = correct_ans / total_ans if total_ans > 0 else 0.0
    result["gen_len"] = total_ans
    result["gen_num"] = total_ans

    print("\nResults\n")
    print(result)
    return result


def eval_accuracy_mmlu(request_outputs: List[RequestFuncOutput]) -> dict:
    """
    Evaluate the accuracy of the results of a given benchmark on the MMLU dataset.

    Args:
        request_outputs (List[RequestFuncOutput]): The outputs of the benchmarking run.

    Returns:
        dict: A dictionary containing the accuracy of the model on the MMLU dataset
    """
    metric = evaluate.load("accuracy")
    nltk.download("punkt")
    nltk.download("punkt_tab")
    preds = []
    targets = []

    for output in request_outputs:
        preds.append(output.generated_text)
        targets.append(output.input_request.completion)
    preds, targets = postprocess_text_mmlu(preds, targets)
    result = metric.compute(
        predictions=preds,
        references=targets,
    )
    result = {k: float(round(np.mean(v), 4)) for k, v in result.items()}
    result["gen_num"] = len(preds)
    print("\nResults\n")
    print(result)
    return result


def postprocess_text_mlperf(pred: str, target: str):
    """Process a single prediction-target pair for the MLPerf benchmark.

    Args:
        pred (str): The generated text.
        target (str): The target text.

    Returns:
        tuple: A tuple containing the processed prediction and target text.
    """
    pred = pred.strip()
    target = target.strip()

    # rougeLSum expects newline after each sentence
    pred = "\n".join(nltk.sent_tokenize(pred))
    target = "\n".join(nltk.sent_tokenize(target))

    return pred, target


def eval_accuracy_mlperf(request_outputs: RequestFuncOutput) -> None:
    """
    Evaluate the accuracy of the results of a given benchmark on the MLPerf dataset.

    Args:
        request_outputs (RequestFuncOutput): The outputs of the benchmarking run.
    """
    metric = evaluate.load("rouge")
    nltk.download("punkt")
    nltk.download("punkt_tab")

    preds = []
    targets = []
    for output in request_outputs:
        pred, target = postprocess_text_mlperf(output.generated_text,
                                               output.input_request.completion)
        preds.append(pred)
        targets.append(target)

    result = metric.compute(
        predictions=preds,
        references=targets,
    )
    result = {k: float(round(np.mean(v) * 100, 4)) for k, v in result.items()}
    result["gen_num"] = len(preds)
    print("\nResults\n")
    print(result)


def eval_benchmark_dataset_result(request_outputs: RequestFuncOutput,
                                  dataset_name: str) -> None:
    """
    Evaluate the accuracy of the results of a given benchmark on a given dataset.

    Args:
        request_outputs (RequestFuncOutput): The outputs of the benchmarking run.
        dataset_name (str): The name of the dataset that the benchmark was run on.
    """
    if dataset_name == "mmlu":
        print("Evaluating MMLU...")
        eval_accuracy_mmlu(request_outputs)
    elif dataset_name == "mlperf":
        print("Evaluating MLPerf...")
        eval_accuracy_mlperf(request_outputs)
    elif dataset_name == "math500":
        eval_accuracy_math_500(request_outputs)
    else:
        raise NotImplementedError("Evaluation is not support for dataset: %s" %
                                  dataset_name)


def sample_warmup_requests(requests: List[SampleRequest]):
    """
    Sample warmup requests from a list of requests.

    Args:
        requests (List[SampleRequest]): A list of SampleRequest objects.

    Yields:
        SampleRequest: A warmup request from the input list.
    """
    interesting_buckets = [
        0,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
    ]

    for start, end in zip(interesting_buckets[:-1], interesting_buckets[1:]):
        for request in requests:
            if start < request.prompt_len <= end:
                yield request
                break
