# Copied from vLLM: https://github.com/vllm-project/vllm/blob/02f0c7b/benchmarks/benchmark_dataset.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This module defines a framework for sampling benchmark requests from various
datasets. Each dataset subclass of BenchmarkDataset must implement sample
generation. Supported dataset types include:
    - MMLMDataset
    - MLPerfDataset
"""

import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union

import pandas as pd
from transformers import PreTrainedTokenizerBase
from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalDataDict

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class SampleRequest:
    """
    Represents a single inference request for benchmarking.
    """

    prompt: Union[str, Any]
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[Union[MultiModalDataDict, dict]] = None
    lora_request: Optional[LoRARequest] = None
    completion: Optional[str] = None


# -----------------------------------------------------------------------------
# Benchmark Dataset Base Class
# -----------------------------------------------------------------------------


class BenchmarkDataset(ABC):
    DEFAULT_SEED = 0
    IS_MULTIMODAL = False

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        random_seed: int = DEFAULT_SEED,
    ) -> None:
        """
        Initialize the BenchmarkDataset with an optional dataset path and random
        seed.  Args:
            dataset_path (Optional[str]): Path to the dataset. If None, it
            indicates that a default or random dataset might be used.
            random_seed (int): Seed value for reproducible shuffling or
            sampling. Defaults to DEFAULT_SEED.
        """
        self.dataset_path = dataset_path
        # Set the random seed, ensuring that a None value is replaced with the
        # default seed.
        self.random_seed = random_seed if random_seed is not None else self.DEFAULT_SEED
        self.data = None

    def apply_multimodal_chat_transformation(
            self,
            prompt: str,
            mm_content: Optional[MultiModalDataDict] = None) -> list[dict]:
        """
        Transform a prompt and optional multimodal content into a chat format.
        This method is used for chat models that expect a specific conversation
        format.
        """
        content = [{"text": prompt, "type": "text"}]
        if mm_content is not None:
            content.append(mm_content)
        return [{"role": "user", "content": content}]

    def load_data(self) -> None:
        """
        Load data from the dataset path into self.data.

        This method must be overridden by subclasses since the method to load
        data will vary depending on the dataset format and source.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        # TODO (jenniferzhao): add support for downloading data
        raise NotImplementedError(
            "load_data must be implemented in subclasses.")

    @abstractmethod
    def sample(self, tokenizer: PreTrainedTokenizerBase,
               num_requests: int) -> list[SampleRequest]:
        """
        Abstract method to generate sample requests from the dataset.

        Subclasses must override this method to implement dataset-specific logic
        for generating a list of SampleRequest objects.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be used
             for processing the dataset's text.
            num_requests (int): The number of sample requests to generate.

        Returns:
            list[SampleRequest]: A list of sample requests generated from the
            dataset.
        """
        raise NotImplementedError("sample must be implemented in subclasses.")

    def maybe_oversample_requests(self, requests: list[SampleRequest],
                                  num_requests: int) -> None:
        """
        Oversamples the list of requests if its size is less than the desired
        number.

        Args:
            requests (List[SampleRequest]): The current list of sampled
            requests.  num_requests (int): The target number of requests.
        """
        if len(requests) < num_requests:
            random.seed(self.random_seed)
            additional = random.choices(requests,
                                        k=num_requests - len(requests))
            requests.extend(additional)
            logger.info("Oversampled requests to reach %d total samples.",
                        num_requests)


# -----------------------------------------------------------------------------
# Utility Functions and Global Caches
# -----------------------------------------------------------------------------


def is_valid_sequence(
    prompt_len: int,
    output_len: int,
    min_len: int = 4,
    max_prompt_len: int = 1024,
    max_total_len: int = 2048,
    skip_min_output_len_check: bool = False,
) -> bool:
    """
    Validate a sequence based on prompt and output lengths.

    Default pruning criteria are copied from the original `sample_hf_requests`
    and `sample_sharegpt_requests` functions in benchmark_serving.py, as well as
    from `sample_requests` in benchmark_throughput.py.
    """
    # Check for invalid conditions
    prompt_too_short = prompt_len < min_len
    output_too_short = (not skip_min_output_len_check) and (output_len
                                                            < min_len)
    prompt_too_long = prompt_len > max_prompt_len
    combined_too_long = (prompt_len + output_len) > max_total_len

    # Return True if none of the invalid conditions are met
    return not (prompt_too_short or output_too_short or prompt_too_long
                or combined_too_long)


# -----------------------------------------------------------------------------
# MMLU Dataset Implementation
# -----------------------------------------------------------------------------


class MMLUDataset(BenchmarkDataset):
    """
    Implements the MMLUDataset dataset.  Logic heavily inspired by Jetstream
    https://github.com/AI-Hypercomputer/JetStream/blob/bbfb5bd/benchmarks/benchmark_serving.py#L327.
    """

    def __init__(self, num_shots: int, mmlu_method: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mmlu_method = mmlu_method
        self.num_shots = num_shots
        self.load_data()

    def load_mmlu_dataset_csv(self,
                              dataset_path: str) -> tuple[Any, dict[str, str]]:
        assert dataset_path != ""
        dataset = []
        prompts_per_subject = dict()
        for cvs_file in os.listdir(dataset_path):
            if cvs_file.endswith(".csv"):
                subject = " ".join(cvs_file.split("_")[:-1])
                if subject not in prompts_per_subject:
                    prompts_per_subject[subject] = ""
                filepath = os.path.join(dataset_path, cvs_file)
                data = pd.read_csv(filepath, header=None)
                data["subject"] = subject
                dataset.append(data)

        combined_dataset = pd.concat(dataset, ignore_index=True)
        header_dict = {
            0: "question",
            1: "A",
            2: "B",
            3: "C",
            4: "D",
            5: "answer",
        }
        combined_dataset.rename(columns=header_dict, inplace=True)
        return combined_dataset, prompts_per_subject

    def gen_mmlu_qa(self, data: Any, mmlu_method: str = "") -> str:

        output = ""
        for _, row in data.iterrows():
            output += (f"Question: {row['question']}\n"
                       f"Choices:\n"
                       f"(A) {row['A']}\n"
                       f"(B) {row['B']}\n"
                       f"(C) {row['C']}\n"
                       f"(D) {row['D']}\n")

            output += "\nCorrect answer:"

            if mmlu_method == "HELM":
                output += f"({row['answer']})\n\n"
            elif mmlu_method == "Harness":
                content = row[row["answer"].upper()]
                output += f"({row['answer']}) {content}\n\n"

        return output

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        combined_dataset, prompts_per_subject = self.load_mmlu_dataset_csv(
            self.dataset_path)
        num_rows, _ = combined_dataset.shape
        print(f"Loaded {num_rows} data from mmlu dataset")

        for subject in prompts_per_subject:
            header = (
                f"The following are multiple choice questions (with answers) "
                f"about {subject}:\n")
            shots_data = combined_dataset[combined_dataset["subject"] ==
                                          subject].head(self.num_shots)
            prompts_per_subject[subject] = header + self.gen_mmlu_qa(
                shots_data, mmlu_method=self.mmlu_method)

        mmlu_data = []
        for _, row in combined_dataset.iloc[self.num_shots:].iterrows():
            question_prompt = self.gen_mmlu_qa(pd.DataFrame([row]))
            output = row["answer"]
            prompt = prompts_per_subject[row["subject"]] + question_prompt
            mmlu_data.append((prompt, output))

        self.data = mmlu_data

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list:
        samples: list = []
        for prompt, completion in self.data:
            if len(samples) >= num_requests:
                break

            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            new_output_len = len(
                completion_ids) if output_len is None else output_len
            # TODO @jacobplatin?
            # if not is_valid_sequence(
            #     prompt_len,
            #     new_output_len,
            #     skip_min_output_len_check=output_len is not None,
            # ):
            #     continue
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=new_output_len,
                    completion=completion,
                ))
        self.maybe_oversample_requests(samples, num_requests)
        return samples


# -----------------------------------------------------------------------------
# MLPerf Dataset Implementation
# -----------------------------------------------------------------------------


class MLPerfDataset(BenchmarkDataset):
    """
    Implements the MLPerf dataset.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        dataset = pd.read_pickle(self.dataset_path)
        mlperf_data = []
        print(f"Loaded {len(dataset)} data from mlperf dataset")
        # NOTE: an example row (entry in the dataset) looks like:
        # system_prompt        You are an AI assistant that helps people find...
        # question             Given the sentence "A woman with a fairy tatto...
        # output               To determine if we can conclude that "The woma...
        # input                <s>[INST] <<SYS>>\nYou are an AI assistant tha...
        # tok_input            [1, 1, 518, 25580, 29962, 3532, 14816, 29903, ...
        # tok_output           [1, 1763, 8161, 565, 591, 508, 17668, 393, 376...
        # origin                                                             cot
        # tok_input_length                                                   146
        # tok_output_length                                                  195
        # TODO: do we want text normalization or any other preprocessing?
        for _, row in dataset.iterrows():
            prompt = row["question"]
            output = row["output"]
            mlperf_data.append((prompt, output))

        self.data = mlperf_data

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        **kwargs,
    ) -> list:
        samples: list = []
        for prompt, completion in self.data:
            if len(samples) >= num_requests:
                break

            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            new_output_len = len(
                completion_ids) if output_len is None else output_len
            # NOTE (jacobplatin): I don't believe that we filter the MLPerf dataset
            # at all, but it could be done here
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=new_output_len,
                    completion=completion,
                ))
        self.maybe_oversample_requests(samples, num_requests)
        return samples
