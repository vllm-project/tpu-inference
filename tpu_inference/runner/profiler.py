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

import datetime
import json
import os
from dataclasses import asdict, dataclass
from enum import StrEnum

import jax
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput

from tpu_inference import envs
from tpu_inference.logger import init_logger
from tpu_inference.runner.input_batch import InputBatch

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

logger = init_logger(__name__)


class InferencePhase(StrEnum):
    PREFILL_HEAVY = "prefill_heavy"
    DECODE_HEAVY = "decode_heavy"
    BALANCED = "balanced"
    NOT_DETERMINED = "not_determined"


@dataclass
class BatchCompositionStats:
    # The total number of tokens scheduled for the batch.
    total_num_scheduled_tokens: int
    # The number of prefill tokens.
    num_prefill_tokens: int
    # The number of decode tokens.
    num_decode_tokens: int
    # The padded total number of tokens scheduled for the batch.
    padded_total_num_scheduled_tokens: int
    # The number of requests in the batch.
    num_reqs: int


def get_batch_composition_stats(
        input_batch: InputBatch, total_num_scheduled_tokens: int,
        num_reqs: int, padded_total_num_scheduled_tokens: int,
        scheduler_output: "VllmSchedulerOutput") -> BatchCompositionStats:
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
        BatchCompositionStats
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
    return BatchCompositionStats(
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        padded_total_num_scheduled_tokens=padded_total_num_scheduled_tokens,
        num_reqs=num_reqs,
    )


def determine_phase_from_batch_composition_stats(
        batch_composition_stats: BatchCompositionStats) -> InferencePhase:
    """
    Determines the inference phase based on the batch composition stats.

    Args:
        batch_composition_stats: The batch composition stats.
    Returns:
        The inference phase enum value.
    """
    num_prefill_tokens = batch_composition_stats.num_prefill_tokens
    total_num_scheduled_tokens = batch_composition_stats.total_num_scheduled_tokens
    prefill_ratio_for_batch = num_prefill_tokens / total_num_scheduled_tokens
    if prefill_ratio_for_batch >= PREFILL_HEAVY_RATIO_THRESHOLD:
        return InferencePhase.PREFILL_HEAVY
    elif prefill_ratio_for_batch <= DECODE_HEAVY_RATIO_THRESHOLD:
        return InferencePhase.DECODE_HEAVY
    elif prefill_ratio_for_batch >= BALANCED_RATIO_THRESHOLD[
            0] and prefill_ratio_for_batch <= BALANCED_RATIO_THRESHOLD[1]:
        return InferencePhase.BALANCED
    else:
        return InferencePhase.NOT_DETERMINED


class PhasedBasedProfiler:
    """
    Implements a phased-based profiler, which will profile three phases:
        1. Prefill heavy
        2. Decode heavy
        3. Balanced

    A phase is determined based on the ratio of prefill tokens to total tokens
    for the given batch (see `determine_phase_from_batch_composition_stats`).

    Args:
        profile_dir: The directory to save the profiles to.

    Attributes:
        profiling_n_steps_left: The number of steps left to profile for the
            current phase.
        profile_dir_with_phase_suffix: The directory to save the profiles to.
        num_steps_to_profile_for: The number of steps to profile for each phase.
        profile_dir: The directory to save the profiles to.
        inference_phase_seen: A dictionary that keeps track of whether a given
            phase has been seen.
        default_profiling_options: The default profiling options.
        current_phase: The current phase.
    """

    def __init__(self, profile_dir: str):
        self.profiling_n_steps_left: int = 0
        self.profile_dir_with_phase_suffix: str = None
        self.num_steps_to_profile_for: int = envs.PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR
        self.profile_dir: str = profile_dir
        self.inference_phase_seen: dict = {
            InferencePhase.PREFILL_HEAVY: False,
            InferencePhase.DECODE_HEAVY: False,
            InferencePhase.BALANCED: False
        }
        self.default_profiling_options = jax.profiler.ProfileOptions()
        self.default_profiling_options.python_tracer_level = envs.PYTHON_TRACER_LEVEL

        self.current_phase: InferencePhase = InferencePhase.NOT_DETERMINED

        logger.info(
            "Phased-based profiler enabled. Traces will be saved to: %s",
            self.profile_dir)

    def _write_batch_composition_stats_to_file_helper(
            self, batch_composition_stats: BatchCompositionStats) -> None:
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
            f.write(json.dumps(asdict(batch_composition_stats)) + "\n")

    def _start_profiling(
            self, batch_composition_stats: BatchCompositionStats) -> None:
        """
        Potentially starts profiling for a given unseen phase.

        Args:
            batch_composition_stats: The batch composition stats.
        """
        determined_phase = determine_phase_from_batch_composition_stats(
            batch_composition_stats)
        if determined_phase == InferencePhase.NOT_DETERMINED:
            return
        if self.inference_phase_seen[determined_phase]:
            return

        self.current_phase = determined_phase
        self.inference_phase_seen[self.current_phase] = True
        self.profiling_n_steps_left = self.num_steps_to_profile_for

        logger.info(f"Starting profiling for {self.current_phase.value} phase")
        logger.info(
            f"Batch composition stats: {asdict(batch_composition_stats)}")
        self.profile_dir_with_phase_suffix = os.path.join(
            self.profile_dir, self.current_phase.value)

        # Create the profile subdirectory if it doesn't exist
        os.makedirs(self.profile_dir_with_phase_suffix, exist_ok=True)

        # Write the batch composition stats to a file to make it easier to
        # align with the traces
        self._write_batch_composition_stats_to_file_helper(
            batch_composition_stats)

        jax.profiler.start_trace(
            self.profile_dir_with_phase_suffix,
            profiler_options=self.default_profiling_options)

    def _step_or_stop_profiling(self, batch_composition_stats: dict) -> None:
        """
        Steps the profiler or stops it if we have profiled enough steps for the
        current phase.

        Args:
            batch_composition_stats: The batch composition stats.
        """
        # We only should decrement the profiling_n_steps_left if we are
        # profiling
        if self.current_phase != InferencePhase.NOT_DETERMINED:
            self._write_batch_composition_stats_to_file_helper(
                batch_composition_stats)
            self.profiling_n_steps_left -= 1
            if self.profiling_n_steps_left <= 0:
                jax.profiler.stop_trace()
                logger.info(
                    f"Profiling for {self.current_phase.value} phase finished")
                self.current_phase = InferencePhase.NOT_DETERMINED

    def step(self, batch_composition_stats: BatchCompositionStats) -> None:
        """
        Steps the profiler.

        Args:
            batch_composition_stats: The batch composition stats.
        """
        have_seen_all_phases = all(self.inference_phase_seen.values())
        # We want to start profiling only after the first trial request
        is_past_initial_request = batch_composition_stats.num_reqs > 1 and batch_composition_stats.total_num_scheduled_tokens > 1
        if is_past_initial_request and (not have_seen_all_phases
                                        or self.current_phase
                                        != InferencePhase.NOT_DETERMINED):
            # We haven't started profiling yet
            if self.profiling_n_steps_left <= 0:
                self._start_profiling(batch_composition_stats)
            # We are in the middle of profiling a given phase
            else:
                self._step_or_stop_profiling(batch_composition_stats)
