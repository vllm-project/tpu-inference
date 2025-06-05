# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import math
import os
import re
from typing import Any

import evaluate
import nltk
import numpy as np
from backend_request_func import RequestFuncOutput


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


def postprocess_text(preds, targets):
    choices = ["A", "B", "C", "D", None]

    def _parse_answer(output):
        re_str = r"\s*\(([A-D])\)\s*\w*"
        match = re.search(re_str, output, re.IGNORECASE)
        predicted_answer = match.group(1).upper() if match else None
        return predicted_answer

    preds = [choices.index(_parse_answer(pred.strip())) for pred in preds]
    targets = [choices.index(target.strip().upper()) for target in targets]
    return preds, targets


def eval_accuracy_mmlu(request_outputs):
    metric = evaluate.load("accuracy")
    nltk.download("punkt")
    nltk.download("punkt_tab")
    preds = []
    targets = []

    for output in request_outputs:
        preds.append(output.generated_text)
        targets.append(output.input_request.completion)
    preds, targets = postprocess_text(preds, targets)
    result = metric.compute(
        predictions=preds,
        references=targets,
    )
    result = {k: float(round(np.mean(v), 4)) for k, v in result.items()}
    result["gen_num"] = len(preds)
    print("\nResults\n")
    print(result)
    return result


def eval_accuracy_mlperf(request_outputs):
    metric = evaluate.load("rouge")
    nltk.download("punkt")
    nltk.download("punkt_tab")

    def postprocess_text(pred, target):
        """Process a single prediction-target pair"""
        pred = pred.strip()
        target = target.strip()

        # rougeLSum expects newline after each sentence
        pred = "\n".join(nltk.sent_tokenize(pred))
        target = "\n".join(nltk.sent_tokenize(target))

        return pred, target

    preds = []
    targets = []
    for output in request_outputs:
        pred, target = postprocess_text(output.generated_text,
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
    else:
        raise NotImplementedError("Evaluation is not support for dataset: %s" %
                                  dataset_name)


def sample_warmup_requests(requests):
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
