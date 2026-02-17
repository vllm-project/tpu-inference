# SPDX-License-Identifier: Apache-2.0
"""
Implements a few utility functions for the various runners.
"""
import bisect
import datetime
import functools
import json
import os
import time
from enum import Enum
from typing import Any

import jax
from jax._src.interpreters import pxla
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput

from tpu_inference import envs
from tpu_inference.logger import init_logger
from tpu_inference.runner.input_batch import InputBatch

MIN_NUM_SEQS = 8

# These are used for determining the inference phase for a given batch in
# determine_phase_from_batch_composition_stats
# We will say that any batch who has at least 90% of its tokens scheduled for
# prefilling is in the PREFILL_HEAVY phase
PREFILL_HEAVY_RATIO_THRESHOLD = 0.9
# We will say that any batch who has at most 20% of its tokens scheduled for
# prefilling is in the DECODE_HEAVY phase
DECODE_HEAVY_RATIO_THRESHOLD = 0.2
# We will say that any batch who has between 40% and 60% of its tokens scheduled
# for prefilling is in the BALANCED phase
BALANCED_RATIO_THRESHOLD = (0.4, 0.6)
PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR = 15

logger = init_logger(__name__)


class InferencePhase(Enum):
    PREFILL_HEAVY = 0
    DECODE_HEAVY = 1
    BALANCED = 2
    AMBIGUOUS = 3


def get_padded_num_reqs_with_upper_limit(x: int, upper_limit: int) -> int:
    res = MIN_NUM_SEQS if x <= MIN_NUM_SEQS else 1 << (x - 1).bit_length()
    return min(res, upper_limit)


def get_req_paddings(min_req_size: int, max_req_size: int) -> list[int]:
    # assert min_req_size is power of 2
    assert (min_req_size & (min_req_size - 1) == 0) and min_req_size > 0
    paddings: list = []
    num = max(MIN_NUM_SEQS, min_req_size)
    while num <= max_req_size and (len(paddings) == 0 or paddings[-1] != num):
        paddings.append(num)
        num = get_padded_num_reqs_with_upper_limit(num + 1, max_req_size)
    logger.info(f"Prepared request paddings: {paddings}")
    return paddings


def get_token_paddings(min_token_size: int, max_token_size: int,
                       padding_gap: int) -> list[int]:
    """Generate a list of padding size, starting from min_token_size,
    ending with a number that can cover max_token_size

    If padding_gap == 0 then:
        increase 2X each time (exponential)
    else:
        first increase the size to twice,
        then increase the padding size by padding_gap.
    """
    # assert min_token_size is power of 2
    assert (min_token_size & (min_token_size - 1) == 0) and min_token_size > 0
    paddings = []
    num = min_token_size

    if padding_gap == 0:
        while True:
            paddings.append(num)
            if num >= max_token_size:
                break
            num *= 2
    else:
        while num <= padding_gap:
            paddings.append(num)
            num *= 2
        num //= 2
        while num < max_token_size:
            num += padding_gap
            paddings.append(num)
    logger.info(f"Prepared token paddings: {paddings}")
    return paddings


def get_padded_token_len(paddings: list[int], x: int) -> int:
    """Return the first element in paddings list greater or equal to x.
    """
    index = bisect.bisect_left(paddings, x)
    assert index < len(paddings)
    return paddings[index]


class LatencyTracker:

    def __init__(self, name="Operation"):
        self.name = name

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        elapsed_time = self.end_time - self.start_time
        logger.debug(f"Latency for '{self.name}': {elapsed_time:.3f} seconds")


class ForbidCompile:
    """
    A context manager to forbid JAX compilation in a specific block of code.

    It works by temporarily wrapping the internal JAX caching function
    `_cached_lowering_to_hlo`. If a call within the `with` block results
    in a cache miss (i.e., triggers a new compilation), it raises a
    RuntimeError.

    Usage:
        # This will raise an error because it's the first compilation.
        with ForbidCompile():
            jitted_func(x)

        # "Warm up" the cache first.
        jitted_func(x)
        # This will now succeed without error.
        with ForbidCompile():
            jitted_func(x)
    """

    def __init__(
            self,
            message="JAX compilation occurred but was forbidden in this context."
    ):
        self.message = message
        self._original_func = None

    def __enter__(self):
        # Store the original function
        self._original_func = pxla._cached_lowering_to_hlo
        original_cached_func = self._original_func

        # Create a wrapper
        @functools.wraps(original_cached_func)
        def wrapper(*args, **kwargs):
            # Get cache statistics before the call
            info_before = original_cached_func.cache_info()
            misses_before = info_before.misses

            # Execute the original cached function
            result = original_cached_func(*args, **kwargs)

            # Get cache statistics after the call
            info_after = original_cached_func.cache_info()
            misses_after = info_after.misses

            # Check if a cache miss occurred
            if misses_after > misses_before:
                raise RuntimeError(self.message)

            return result

        # Monkey-patch the function with our wrapper
        pxla._cached_lowering_to_hlo = wrapper

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the original function
        if self._original_func:
            pxla._cached_lowering_to_hlo = self._original_func
        # Don't suppress any exceptions that occurred inside the 'with' block
        return False


def get_batch_composition_stats(
        input_batch: InputBatch, total_num_scheduled_tokens: int,
        num_reqs: int, padded_total_num_scheduled_tokens: int,
        scheduler_output: "VllmSchedulerOutput") -> dict:
    """
    Logs the total number of tokens scheduled for the batch, the number of
    prefill tokens, the number of decode tokens, and the number of padded
    tokens scheduled for the batch.
    Args:
        input_batch: The input batch.
        total_num_scheduled_tokens: The total number of tokens scheduled for the batch.
        num_reqs: The number of requests in the batch.
        padded_total_num_scheduled_tokens: The padded total number of tokens scheduled for the batch.
        scheduler_output: The scheduler output.
    Returns:
        A string containing the total number of tokens scheduled for the batch, the number of
        prefill tokens, the number of decode tokens, and the number of padded tokens scheduled for the batch.
    """
    num_prefill_tokens = 0
    num_decode_tokens = 0

    # Get the number of scheduled tokens for each request.
    num_scheduled_tokens_per_req_list = []
    # Get the number of tokens already processed for each request.
    num_computed_tokens_per_req = input_batch.num_computed_tokens_cpu[:
                                                                      num_reqs]

    for i, req_id in enumerate(input_batch.req_ids[:num_reqs]):
        assert req_id is not None

        # This is the number of tokens to process in the current step for this request
        num_scheduled_for_req = scheduler_output.num_scheduled_tokens[req_id]
        num_scheduled_tokens_per_req_list.append(num_scheduled_for_req)

        # This is the number of tokens already processed for this request (before this step)
        num_already_computed = num_computed_tokens_per_req[i]

        if num_already_computed == 0:
            # Prefill
            num_prefill_tokens += num_scheduled_for_req
        # This means the request is ongoing
        else:
            if num_scheduled_for_req > 1:
                # It's a multi-token request, so it's chunked prefill
                num_prefill_tokens += num_scheduled_for_req
            else:
                # It's a single token for an ongoing request, so it's decode
                num_decode_tokens += 1
    return {
        "total_num_scheduled_tokens": total_num_scheduled_tokens,
        "num_prefill_tokens": num_prefill_tokens,
        "num_decode_tokens": num_decode_tokens,
        "padded_total_num_scheduled_tokens": padded_total_num_scheduled_tokens,
        "num_reqs": num_reqs
    }


def determine_phase_from_batch_composition_stats(
        batch_composition_stats: dict[str, Any]) -> InferencePhase:
    """
    Determines the inference phase based on the batch composition stats.

    Args:
        batch_composition_stats: The batch composition stats.
            This is a dict containing:
                total_num_scheduled_tokens: The total number of tokens scheduled for the batch.
                num_prefill_tokens: The number of prefill tokens.
                num_decode_tokens: The number of decode tokens.
                padded_total_num_scheduled_tokens: The padded total number of tokens scheduled for the batch.
                num_reqs: The number of requests in the batch.
    Returns:
        The inference phase enum value.
    """
    num_prefill_tokens = batch_composition_stats["num_prefill_tokens"]
    total_num_scheduled_tokens = batch_composition_stats[
        "total_num_scheduled_tokens"]
    prefill_ratio_for_batch = num_prefill_tokens / total_num_scheduled_tokens
    if prefill_ratio_for_batch >= PREFILL_HEAVY_RATIO_THRESHOLD:
        return InferencePhase.PREFILL_HEAVY
    elif prefill_ratio_for_batch <= DECODE_HEAVY_RATIO_THRESHOLD:
        return InferencePhase.DECODE_HEAVY
    elif prefill_ratio_for_batch >= BALANCED_RATIO_THRESHOLD[
            0] and prefill_ratio_for_batch <= BALANCED_RATIO_THRESHOLD[1]:
        return InferencePhase.BALANCED
    else:
        return InferencePhase.AMBIGUOUS


class PhasedBasedProfiler:
    """
    Implements a phased-based profiler, which will profile three phases:
        1. Prefill heavy
        2. Decode heavy
        3. Balanced

    A phase is determined based on the ratio of prefill tokens to total scheduled
    tokens for the given batch (see `determine_phase_from_batch_composition_stats`).

    Args:
        profile_dir: The directory to save the profiles to.

    Attributes:
        profiling_n_steps_left: The number of steps left to profile for the current phase.
        profile_dir_with_phase_suffix: The directory to save the profiles to.
        num_steps_to_profile_for: The number of steps to profile for each phase.
        profile_dir: The directory to save the profiles to.
        inference_phase_seen: A dictionary that keeps track of whether a given phase has been seen.
        default_profiling_options: The default profiling options.
        current_phase: The current phase.
    """

    def __init__(self, profile_dir: str):
        self.profiling_n_steps_left: int = 0
        self.profile_dir_with_phase_suffix: str = None
        self.num_steps_to_profile_for: int = int(
            os.getenv("PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR",
                      PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR))
        self.profile_dir: str = profile_dir
        # NOTE: we purposely don't have AMBIGUOUS here
        self.inference_phase_seen: dict = {
            InferencePhase.PREFILL_HEAVY: False,
            InferencePhase.DECODE_HEAVY: False,
            InferencePhase.BALANCED: False
        }
        self.default_profiling_options = jax.profiler.ProfileOptions()
        self.default_profiling_options.python_tracer_level = envs.PYTHON_TRACER_LEVEL

        self.current_phase: str = ""

        logger.info(
            "Phased-based profiler enabled. Traces will be saved to: %s",
            self.profile_dir)

    def _write_batch_composition_stats_to_file_helper(
            self, batch_composition_stats: dict) -> None:
        """
        Writes the batch composition stats to a file at the given time,
        e.g.: prefill_heavy/batch_composition_stats_2025_08_22_15_41_41_505018.json
        """
        now = datetime.datetime.now()
        date_string_in_profiler_format = now.strftime("%Y_%m_%d_%H_%M_%S_%f")

        with open(
                os.path.join(
                    self.profile_dir_with_phase_suffix,
                    f"batch_composition_stats_{date_string_in_profiler_format}.json"
                ), "w") as f:
            f.write(json.dumps(batch_composition_stats) + "\n")

    def _start_profiling(self, batch_composition_stats: dict) -> None:
        """
        Potentially starts profiling for a given unseen phase.

        Args:
            batch_composition_stats: The batch composition stats,  which is a dict
                containig:
                    total_num_scheduled_tokens: The total number of tokens scheduled for the batch.
                    num_prefill_tokens: The number of prefill tokens.
                    num_decode_tokens: The number of decode tokens.
                    padded_total_num_scheduled_tokens: The padded total number of tokens scheduled for the batch.
                    num_reqs: The number of requests in the batch.
        """
        current_determined_phase = determine_phase_from_batch_composition_stats(
            batch_composition_stats)
        for phase, has_been_seen in self.inference_phase_seen.items():
            if has_been_seen or phase != current_determined_phase:
                continue

            self.inference_phase_seen[phase] = True
            self.profiling_n_steps_left = self.num_steps_to_profile_for

            self.current_phase = phase.name.lower()

            logger.info(f"Starting profiling for {self.current_phase} phase")
            logger.info(f"Batch composition stats: {batch_composition_stats}")
            self.profile_dir_with_phase_suffix = os.path.join(
                self.profile_dir, self.current_phase)

            # Create the profile subdirectory if it doesn't exist
            os.makedirs(self.profile_dir_with_phase_suffix, exist_ok=True)

            # Write the batch composition stats to a file to make it easier to
            # align with the traces
            self._write_batch_composition_stats_to_file_helper(
                batch_composition_stats)

            jax.profiler.start_trace(
                self.profile_dir_with_phase_suffix,
                profiler_options=self.default_profiling_options)
            break

    def _step_or_stop_profiling(self, batch_composition_stats: dict) -> None:
        """
        Steps the profiler or stops it if we have profiled enough steps for the
        current phase.

        Args:
            batch_composition_stats: The batch composition stats,  which is a dict
                containig:
                    total_num_scheduled_tokens: The total number of tokens scheduled for the batch.
                    num_prefill_tokens: The number of prefill tokens.
                    num_decode_tokens: The number of decode tokens.
                    padded_total_num_scheduled_tokens: The padded total number of tokens scheduled for the batch.
                    num_reqs: The number of requests in the batch.
        """
        # We only should decrement the profiling_n_steps_left if we are profiling
        if self.current_phase != "":
            self._write_batch_composition_stats_to_file_helper(
                batch_composition_stats)
            self.profiling_n_steps_left -= 1
            if self.profiling_n_steps_left <= 0:
                jax.profiler.stop_trace()
                logger.info(
                    f"Profiling for {self.current_phase} phase finished")
                self.current_phase = ""

    def step(self, batch_composition_stats: dict) -> None:
        """
        Steps the profiler.

        Args:
            batch_composition_stats: The batch composition stats,  which is a dict
                containig:
                    total_num_scheduled_tokens: The total number of tokens scheduled for the batch.
                    num_prefill_tokens: The number of prefill tokens.
                    num_decode_tokens: The number of decode tokens.
                    padded_total_num_scheduled_tokens: The padded total number of tokens scheduled for the batch.
                    num_reqs: The number of requests in the batch.
        """
        have_seen_all_phases = all(self.inference_phase_seen.values())
        # We want to start profiling only after the first trial request
        is_past_initial_request = batch_composition_stats[
            "num_reqs"] > 1 and batch_composition_stats[
                "total_num_scheduled_tokens"] > 1
        if is_past_initial_request and (not have_seen_all_phases
                                        or self.current_phase != ""):
            # We haven't started profiling yet
            if self.profiling_n_steps_left <= 0:
                self._start_profiling(batch_composition_stats)
            # We are in the middle of profiling a given phase
            else:
                self._step_or_stop_profiling(batch_composition_stats)
