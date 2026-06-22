# SPDX-License-Identifier: Apache-2.0
"""
Implements a few utility functions for the various runners.
"""
import atexit
import bisect
import datetime
import functools
import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.interpreters import pxla
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput

from tpu_inference import envs
from tpu_inference.logger import init_logger
from tpu_inference.runner.input_batch import InputBatch


def trim_request_id_suffix(request_id: str) -> str:
    """Trims the suffix from a request ID, keeping only the base ID.

    Example: cmpl-f0d75fed-c25e-4ccf-b369-9bd0226021b3-0-a9cb3cca
          -> cmpl-f0d75fed-c25e-4ccf-b369-9bd0226021b3
    """
    parts = request_id.split("-")
    if len(parts) >= 6 and parts[0] == "cmpl":
        return "-".join(parts[:6])
    return request_id


def get_kv_transfer_metadata(kv: list[Any]) -> tuple[str, int]:
    """Returns the dimensions string and size in bytes for a list of KV cache tensors.

    Dimensions format: (num_layers, num_blocks, block_size, num_heads, 2 (K/V), head_dim)
    """
    if not kv:
        return "()", 0

    # Dimensions string
    dims_str = str((len(kv), ) + tuple(kv[0].shape))

    # Calculate bytes robustly (supports JAX/NumPy Tensors and ShapeDtypeStruct specs)
    kv_size_bytes = 0
    for k in kv:
        if hasattr(k, "nbytes"):
            kv_size_bytes += k.nbytes
        else:
            # Fallback for ShapeDtypeStruct specs
            kv_size_bytes += int(np.prod(k.shape) * np.dtype(k.dtype).itemsize)

    return dims_str, kv_size_bytes


def extract_request_ids_for_tracing(
    input_batch: InputBatch,
    scheduler_output: Optional[Any] = None,
) -> dict[str, str]:
    """Extracts request IDs from an InputBatch and formats them for XProf tracing."""
    req_id_kwargs = {}
    try:
        num_reqs = input_batch.num_reqs
        raw_req_ids = [
            str(rid) for rid in input_batch.req_ids[:num_reqs]
            if rid is not None
        ]

        active_req_ids = []
        for rid in raw_req_ids:
            # Only trace requests that have non-zero scheduled tokens in this step
            if scheduler_output is not None:
                num_tokens = scheduler_output.num_scheduled_tokens.get(rid, 0)
                if num_tokens == 0:
                    continue
            active_req_ids.append(rid)

        trimmed_req_ids = [
            trim_request_id_suffix(rid) for rid in active_req_ids
        ]
        for i, rid in enumerate(trimmed_req_ids):
            req_id_kwargs[f"request_id{i+1}"] = rid
    except Exception as e:
        logger.warning(
            f"Failed to extract request IDs for tracing from input_batch. Error: {e}"
        )
    return req_id_kwargs


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
PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP = 0
# For decode only batches, start capturing traces after all requests in the
# batch has KV caches that have reached this length threshold
PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD = -1

logger = init_logger(__name__)


class InferencePhase(Enum):
    PREFILL_HEAVY = 0
    DECODE_HEAVY = 1
    BALANCED = 2
    AMBIGUOUS = 3
    PREFILL_ONLY = 4
    DECODE_ONLY = 5


def _inject_dp_rank_into_filename(fname: str, dp_rank: int) -> str:
    """Prefix `dp<N>_` to an xplane or trace filename, e.g.

        t1v-n-3312659f-w-0.xplane.pb       -> dp0_t1v-n-3312659f-w-0.xplane.pb
        t1v-n-3312659f-w-0.trace.json.gz   -> dp0_t1v-n-3312659f-w-0.trace.json.gz

    Used by PhasedBasedProfiler under MPMD so per-DP-rank captures (which
    otherwise share an identical hostname+worker filename on a single host)
    can coexist in one `plugins/profile/<ts>/` dir for xprof.
    """
    return f"dp{dp_rank}_{fname}"


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


def get_attn_req_paddings(min_req_size: int, max_req_size: int) -> list[int]:
    """Get num reqs paddings with custom override to reduce compilation time"""
    if not envs.ATTN_BUCKETIZED_NUM_REQS:
        reqs = [max_req_size]
    elif envs.ATTN_CUSTOM_NUM_REQS_BUCKETS:
        reqs = envs.ATTN_CUSTOM_NUM_REQS_BUCKETS
    else:
        reqs = get_req_paddings(min_req_size, max_req_size)

    if max_req_size not in reqs:
        logger.info(
            "max_num_reqs must be supported but is not in ATTN_CUSTOM_NUM_REQS_BUCKETS. Adding max_num_reqs to the num_reqs buckets."
        )
        reqs.append(max_req_size)

    logger.info(f"Prepared attn request paddings: {reqs}")

    return reqs


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
    assert index < len(paddings), f"{paddings=}, {x=}"
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
        batch_id: int, input_batch: InputBatch,
        total_num_scheduled_tokens: int, num_reqs: int,
        padded_total_num_scheduled_tokens: int,
        scheduler_output: "VllmSchedulerOutput") -> dict:
    """
    Logs the total number of tokens scheduled for the batch, the number of
    prefill tokens, the number of decode tokens, and the number of padded
    tokens scheduled for the batch.
    Args:
        batch_id: The sequential id of the batch.
        input_batch: The input batch.
        total_num_scheduled_tokens: The total number of tokens scheduled for the batch.
        num_reqs: The number of requests in the batch.
        padded_total_num_scheduled_tokens: The padded total number of tokens scheduled for the batch.
        scheduler_output: The scheduler output.
    Returns:
        A dict containing the batch id, the total number of tokens scheduled for the batch, the number of
        prefill tokens, the number of decode tokens, the number of padded tokens scheduled for the batch,
        the number of requests in the batch, and the phase of the inference the batch is in.
    """
    num_prefill_tokens = 0
    num_decode_tokens = 0

    # Get the number of scheduled tokens for each request.
    num_scheduled_tokens_per_req_list = []
    # Get the number of tokens already processed for each request.
    num_computed_tokens_per_req = input_batch.num_computed_tokens_cpu[:
                                                                      num_reqs]

    scheduled_spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
    min_kv_len = float('inf') if num_reqs > 0 else 0
    for i, req_id in enumerate(input_batch.req_ids[:num_reqs]):
        assert req_id is not None

        # This is the number of tokens to process in the current step for this request
        num_scheduled_for_req = scheduler_output.num_scheduled_tokens[req_id]
        num_scheduled_tokens_per_req_list.append(num_scheduled_for_req)

        # This is the number of tokens already processed for this request (before this step)
        num_already_computed = int(
            num_computed_tokens_per_req[i])  # Cast from np.int32
        min_kv_len = min(min_kv_len, num_already_computed)

        # When speculative decoding is enabled for this request, the extra
        # tokens are draft tokens being verified, not chunked prefill tokens.
        num_spec_tokens = len(scheduled_spec_decode_tokens.get(req_id, ()))

        if num_already_computed == 0:
            # Prefill
            num_prefill_tokens += num_scheduled_for_req
        # This means the request is ongoing
        else:
            if num_spec_tokens > 0:
                # Verifying draft tokens for an ongoing request — count the
                # target token plus the draft tokens as decode.
                num_decode_tokens += num_scheduled_for_req
            elif num_scheduled_for_req > 1:
                # It's a multi-token request, so it's chunked prefill
                num_prefill_tokens += num_scheduled_for_req
            else:
                # It's a single token for an ongoing request, so it's decode
                num_decode_tokens += 1

    stats = {
        "batch_id": batch_id,
        "total_num_scheduled_tokens": total_num_scheduled_tokens,
        "num_prefill_tokens": num_prefill_tokens,
        "num_decode_tokens": num_decode_tokens,
        "padded_total_num_scheduled_tokens": padded_total_num_scheduled_tokens,
        "num_reqs": num_reqs,
        "min_kv_len": min_kv_len if min_kv_len != float('inf') else 0
    }
    stats["phase"] = determine_phase_from_batch_composition_stats(stats).name
    return stats


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
    if prefill_ratio_for_batch == 1.0:
        return InferencePhase.PREFILL_ONLY
    if prefill_ratio_for_batch == 0.0:
        return InferencePhase.DECODE_ONLY
    if prefill_ratio_for_batch >= PREFILL_HEAVY_RATIO_THRESHOLD:
        return InferencePhase.PREFILL_HEAVY
    if prefill_ratio_for_batch <= DECODE_HEAVY_RATIO_THRESHOLD:
        return InferencePhase.DECODE_HEAVY
    if prefill_ratio_for_batch >= BALANCED_RATIO_THRESHOLD[
            0] and prefill_ratio_for_batch <= BALANCED_RATIO_THRESHOLD[1]:
        return InferencePhase.BALANCED

    return InferencePhase.AMBIGUOUS


class AggregatedStatsLogger:
    """
    Logs batch composition stats continuously for all steps to a file and
    periodically flushes them to GCS if required.

    Args:
        profile_dir: The directory where the profile stats should be saved (local or GCS).
        flush_interval: The number of steps between flushes to storage.
    """

    def __init__(self, profile_dir: str, flush_interval: int = 100):
        self.profile_dir = profile_dir
        self.flush_interval = flush_interval
        self.step_count = 0

        now = datetime.datetime.now()
        date_string = now.strftime("%Y_%m_%d_%H_%M_%S")
        filename = f"all_batches_stats_{date_string}.jsonl"
        self.local_temp_file, self.target_file = \
            self._get_local_and_target_paths(self.profile_dir, filename)

        self._f_local = open(self.local_temp_file, "w")
        self._f_tmp = open(
            os.path.join(tempfile.gettempdir(), "all_batches_stats.jsonl"),
            "w")

        logger.info(
            f"Initialized AggregatedStatsLogger with output path: {self.target_file}"
        )
        atexit.register(self.close)

    def _get_local_and_target_paths(self, base_dir: str,
                                    filename: str) -> tuple[str, str]:
        """Helper to resolve local temp path vs final target path (e.g. for GCS)."""
        target = os.path.join(base_dir, filename)
        if base_dir.startswith("gs://"):
            return os.path.join(tempfile.gettempdir(), filename), target
        os.makedirs(base_dir, exist_ok=True)
        return target, target

    def _sync_to_gcs(self,
                     local_file: str,
                     target_file: str,
                     blocking: bool = False) -> None:
        """Helper to sync local file to GCS using the Python SDK."""
        if target_file.startswith("gs://") and os.path.exists(local_file):

            def _upload():
                try:
                    from google.cloud import storage  # type: ignore
                    client = storage.Client()
                    # e.g., gs://my-bucket/path/to/file.txt -> ("my-bucket", "path/to/file.txt")
                    bucket_name, blob_name = target_file[5:].split("/", 1)
                    client.bucket(bucket_name).blob(
                        blob_name).upload_from_filename(local_file)
                except Exception as e:
                    logger.error(
                        f"Failed to upload {local_file} to {target_file}: {e}",
                        exc_info=True)

            if blocking:
                _upload()
            else:
                threading.Thread(target=_upload, daemon=True).start()

    def log(self, batch_composition_stats: dict) -> None:
        """
        Logs a single batch's composition statistics to local temporary files.
        Automatically triggers a flush to storage if the step count reaches the flush interval.

        Args:
            batch_composition_stats: A dictionary containing the composition statistics
                for the current batch.
        """
        stats_json = json.dumps(batch_composition_stats) + "\n"
        self._f_local.write(stats_json)
        self._f_tmp.write(stats_json)

        self.step_count += 1
        if self.step_count % self.flush_interval == 0:
            self.flush()

    def flush(self, blocking: bool = False) -> None:
        """
        Flushes the current buffered logs to local disk and syncs to Google Cloud Storage (GCS)
        if the target profile directory is a GCS URI.

        Args:
            blocking: If True, waits for the GCS sync subprocess to finish before returning.
        """
        self._f_local.flush()
        self._f_tmp.flush()
        if self.target_file.startswith("gs://") and os.path.exists(
                self.local_temp_file):
            logger.info(
                f"Syncing continuous batch stats to {self.target_file} (Step {self.step_count})..."
            )
            self._sync_to_gcs(self.local_temp_file,
                              self.target_file,
                              blocking=blocking)

    def close(self) -> None:
        """
        Closes the file handles, ensuring a final blocking flush to storage.
        If syncing to GCS, cleans up the local temporary file.
        """
        self.flush(blocking=True)
        self._f_local.close()
        self._f_tmp.close()
        if self.profile_dir.startswith("gs://") and os.path.exists(
                self.local_temp_file):
            try:
                os.remove(self.local_temp_file)
            except OSError:
                pass


class PhasedBasedProfiler:
    """
    Implements a phased-based profiler, which will profile three phases:
        1. Prefill heavy
        2. Decode heavy
        3. Balanced
        4. Prefill Only
        5. Decode  Only

    A phase is determined based on the ratio of prefill tokens to total scheduled
    tokens for the given batch (see `determine_phase_from_batch_composition_stats`).

    Args:
        profile_dir: The directory to save the profiles to.
        worker_rank: The rank of the current worker process.
        flush_interval: The number of steps between continuous logger flushes to storage.

    Attributes:
        profiling_n_steps_left: The number of steps left to profile for the current phase.
        profile_dir_with_phase_suffix: The directory to save the profiles to.
        num_steps_to_profile_for: The number of steps to profile for each phase.
        num_decode_steps_to_skip: The number of decode steps to skip before profiling.
        decode_steps_skipped: The number of decode steps skipped so far.
        profile_dir: The directory to save the profiles to.
        inference_phase_seen: A dictionary that keeps track of whether a given phase has been seen.
        default_profiling_options: The default profiling options.
        current_phase: The current phase.
        worker_rank: The rank of the current worker process.
    """

    def __init__(self,
                 profile_dir: str,
                 worker_rank: int = 0,
                 flush_interval: int = 100):
        self.profiling_n_steps_left: int = 0
        self.profile_dir_with_phase_suffix: str = None
        self.num_steps_to_profile_for: int = int(
            os.getenv("PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR",
                      PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR))
        self.num_decode_steps_to_skip: int = int(
            os.getenv("PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP",
                      PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP))
        self.decode_steps_skipped: int = 0
        self.decode_kv_len_threshold: int = int(
            os.getenv("PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD",
                      PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD))
        self.profile_dir: str = profile_dir
        # NOTE: we purposely don't have AMBIGUOUS here
        self.inference_phase_seen: dict = {
            InferencePhase.PREFILL_ONLY: False,
            InferencePhase.PREFILL_HEAVY: False,
            InferencePhase.DECODE_ONLY: False,
            InferencePhase.DECODE_HEAVY: False,
            InferencePhase.BALANCED: False
        }
        self.default_profiling_options = jax.profiler.ProfileOptions()
        self.default_profiling_options.python_tracer_level = envs.PYTHON_TRACER_LEVEL
        self.default_profiling_options.advanced_configuration = {
            "tpu_trace_mode": "TRACE_COMPUTE",
            "tpu_num_sparse_cores_to_trace": 1,
            "tpu_num_sparse_core_tiles_to_trace": 1,
        }
        if envs.PROFILE_SINGLE_DEVICE:
            self.default_profiling_options.advanced_configuration = {
                "tpu_num_chips_to_profile_per_task": 1,
                "tpu_num_sparse_cores_to_trace": 1,
                "tpu_num_sparse_core_tiles_to_trace": 1,
            }

        self.current_phase: str = ""
        self.reset_sentinel_path: str = os.path.join(
            self.profile_dir, "reset_profiler")

        self.worker_rank = worker_rank
        self.aggregated_stats_logger = None

        logger.info(
            "Phased-based profiler enabled. Traces will be saved to: %s",
            self.profile_dir)
        if self.num_decode_steps_to_skip > 0:
            logger.info(
                "Will skip %d decode-heavy steps before profiling decode_heavy phase.",
                self.num_decode_steps_to_skip)
        if self.decode_kv_len_threshold >= 0:
            logger.info("Will skip decode-only steps until min KV len >= %d.",
                        self.decode_kv_len_threshold)

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
                containing:
                    batch_id: The sequential id of the batch.
                    total_num_scheduled_tokens: The total number of tokens scheduled for the batch.
                    num_prefill_tokens: The number of prefill tokens.
                    num_decode_tokens: The number of decode tokens.
                    padded_total_num_scheduled_tokens: The padded total number of tokens scheduled for the batch.
                    num_reqs: The number of requests in the batch.
                    phase: The phase of the inference the batch is in.
        """
        current_determined_phase = determine_phase_from_batch_composition_stats(
            batch_composition_stats)
        for phase, has_been_seen in self.inference_phase_seen.items():
            if has_been_seen or phase != current_determined_phase:
                continue

            # Skip a configurable number of decode-heavy steps before profiling
            if phase == InferencePhase.DECODE_HEAVY and \
                    self.decode_steps_skipped < self.num_decode_steps_to_skip:
                self.decode_steps_skipped += 1
                logger.debug(
                    "Skipping decode-heavy step %d/%d before profiling.",
                    self.decode_steps_skipped, self.num_decode_steps_to_skip)
                break

            # Skip decode-only steps until min KV len reaches threshold
            if phase == InferencePhase.DECODE_ONLY and \
                    self.decode_kv_len_threshold >= 0:
                min_kv_len = batch_composition_stats.get("min_kv_len", 0)
                if min_kv_len < self.decode_kv_len_threshold:
                    logger.debug(
                        "Skipping decode-only step as min KV len %d < threshold %d.",
                        min_kv_len, self.decode_kv_len_threshold)
                    break

            self.inference_phase_seen[phase] = True
            self.profiling_n_steps_left = self.num_steps_to_profile_for

            self.current_phase = phase.name.lower()

            logger.info(f"Starting profiling for {self.current_phase} phase")
            logger.info(f"Batch composition stats: {batch_composition_stats}")
            phase_dir = os.path.join(self.profile_dir, self.current_phase)
            os.makedirs(phase_dir, exist_ok=True)

            # Resolve the canonical destination ts before start_trace so all
            # DP ranks land in the same <phase>/plugins/profile/<ts>/ dir
            # when capture is moved out of the sandbox.
            self._canonical_dst_ts = self._resolve_canonical_dst_ts(phase_dir)

            self.profile_dir_with_phase_suffix = os.path.join(
                phase_dir, f"dp_rank_{self.worker_rank}")
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
                containing:
                    batch_id: The sequential id of the batch.
                    total_num_scheduled_tokens: The total number of tokens scheduled for the batch.
                    num_prefill_tokens: The number of prefill tokens.
                    num_decode_tokens: The number of decode tokens.
                    padded_total_num_scheduled_tokens: The padded total number of tokens scheduled for the batch.
                    num_reqs: The number of requests in the batch.
                    phase: The phase of the inference the batch is in.
        """
        # We only should decrement the profiling_n_steps_left if we are profiling
        if self.current_phase != "":
            self._write_batch_composition_stats_to_file_helper(
                batch_composition_stats)
            self.profiling_n_steps_left -= 1
            if self.profiling_n_steps_left <= 0:
                jax.profiler.stop_trace()
                self._merge_profile_directories()
                logger.info(
                    f"Profiling for {self.current_phase} phase finished")
                self.current_phase = ""

    def reset(self) -> None:
        """
        Resets the profiler so all phases can be captured again.
        Called when the sentinel file is detected, allowing the bench
        script to re-arm profiling between benchmark configurations.
        """
        for phase in self.inference_phase_seen:
            self.inference_phase_seen[phase] = False
        self.profiling_n_steps_left = 0
        self.current_phase = ""
        logger.info("Phased profiler reset — all phases will be re-captured")

    def _check_reset_sentinel(self) -> None:
        """
        Checks for a sentinel file that signals the profiler should reset.
        The bench script writes this file between benchmark configs so that
        traces are captured for every ISL/OSL/concurrency combination.
        """
        if os.path.exists(self.reset_sentinel_path):
            os.remove(self.reset_sentinel_path)
            self.reset()

    # How long non-zero DP ranks will wait for rank 0 to publish the
    # canonical-ts marker before falling back to their own timestamp.
    _CANONICAL_TS_POLL_TIMEOUT_S = 5.0
    _CANONICAL_TS_POLL_INTERVAL_S = 0.05

    def _resolve_canonical_dst_ts(self, phase_dir: str) -> str:
        """Resolve the canonical destination timestamp for this phase.

        Rank 0 picks the ts (wall clock now) and writes it atomically to a
        marker file keyed by parent PID; non-zero ranks poll for the marker
        and read the ts so all ranks end up moving their captures into the
        same <phase>/plugins/profile/<canonical_ts>/ dir.

        The parent PID in the marker name keeps a current-session marker
        distinct from any leftover marker from a prior `vllm serve` run
        sharing the same PHASED_PROFILING_DIR.
        """
        marker = os.path.join(phase_dir, f".canonical_ts_{os.getppid()}")
        if self.worker_rank == 0:
            canonical_ts = datetime.datetime.now().strftime(
                "%Y_%m_%d_%H_%M_%S")
            marker_tmp = f"{marker}.tmp"
            with open(marker_tmp, "w") as f:
                f.write(canonical_ts)
            os.replace(marker_tmp, marker)
            return canonical_ts

        deadline = time.monotonic() + self._CANONICAL_TS_POLL_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                with open(marker) as f:
                    ts = f.read().strip()
                if ts:
                    return ts
            except OSError:
                pass
            time.sleep(self._CANONICAL_TS_POLL_INTERVAL_S)

        fallback_ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        logger.warning(
            "dp_rank %d did not find rank 0's canonical-ts marker at %s "
            "within %.1fs; falling back to own timestamp %s — this rank's "
            "capture will land in a separate ts dir from rank 0's.",
            self.worker_rank, marker, self._CANONICAL_TS_POLL_TIMEOUT_S,
            fallback_ts)
        return fallback_ts

    def _merge_profile_directories(self) -> None:
        """
        Consolidates phase trace artifacts so downstream tools (c2xprof,
        TensorBoard profile plugin, `scripts/merge_xprof.py`) see a single
        distributed session.

        Two split states are handled in sequence; the function is safe to
        call from every rank after its own jax.profiler.stop_trace.

        1. MPMD on a single host: each DP rank captured into
           <phase>/dp_rank_<N>/plugins/profile/<ts>/ with an identically-
           named xplane.pb (same hostname + same JAX worker id across
           ranks). Hoist each rank's files up to
           <phase>/plugins/profile/<ts>/ under TPU_MULTIPROCESS_DP, with
           `dp<N>` injected into the filename so per-rank captures
           coexist instead of clobbering each other.

        2. Disjoint timestamp dirs: ray multi-host startup skew (or, after
           step 1, the per-rank stop_trace timestamps in MPMD) produces
           multiple <ts>/ subdirs under <phase>/plugins/profile/. Collapse
           them into the earliest timestamp dir.

        Example multi-host split state before merge:
          .../plugins/profile/2026_05_06_04_47_36/j-1b8d22de-2250-4697-9dfc-ray-node-1-0.xplane.pb
          .../plugins/profile/2026_05_06_04_47_38/j-1b8d22de-2250-4697-9dfc-ray-node-0-0.xplane.pb
        After merge: both files under 2026_05_06_04_47_36/.
        """
        source_profile_path = os.path.join(self.profile_dir_with_phase_suffix,
                                           "plugins", "profile")
        if not os.path.exists(source_profile_path):
            return
        phase_dir = os.path.dirname(self.profile_dir_with_phase_suffix)
        dst_ts_dir = os.path.join(phase_dir, "plugins", "profile",
                                  self._canonical_dst_ts)
        try:
            os.makedirs(dst_ts_dir, exist_ok=True)
            for ts in os.listdir(source_profile_path):
                src_ts_dir = os.path.join(source_profile_path, ts)
                if not os.path.isdir(src_ts_dir):
                    continue
                for fname in os.listdir(src_ts_dir):
                    new_fname = (_inject_dp_rank_into_filename(
                        fname, self.worker_rank)
                                 if envs.TPU_MULTIPROCESS_DP else fname)
                    shutil.move(os.path.join(src_ts_dir, fname),
                                os.path.join(dst_ts_dir, new_fname))
                try:
                    os.rmdir(src_ts_dir)
                except OSError:
                    pass
            for cleanup in (source_profile_path,
                            os.path.dirname(source_profile_path)):
                try:
                    os.rmdir(cleanup)
                except OSError:
                    pass
            logger.info(
                f"Successfully merged profile directories into: {dst_ts_dir}")
        except Exception as e:
            logger.warning("Failed to merge profile directories: %s", e)

    def step(self, batch_composition_stats: dict) -> None:
        """
        Steps the profiler and logs batch composition stats.

        Args:
            batch_composition_stats: The batch composition stats,  which is a dict
                containing:
                    batch_id: The sequential id of the batch.
                    total_num_scheduled_tokens: The total number of tokens scheduled for the batch.
                    num_prefill_tokens: The number of prefill tokens.
                    num_decode_tokens: The number of decode tokens.
                    padded_total_num_scheduled_tokens: The padded total number of tokens scheduled for the batch.
                    num_reqs: The number of requests in the batch.
                    phase: The phase of the inference the batch is in.
        """
        self._check_reset_sentinel()
        have_seen_all_phases = all(self.inference_phase_seen.values())
        # We want to start profiling only after the first trial request
        is_past_initial_request = batch_composition_stats[
            "total_num_scheduled_tokens"] > 1
        if is_past_initial_request and (not have_seen_all_phases
                                        or self.current_phase != ""):
            # We haven't started profiling yet
            if self.profiling_n_steps_left <= 0:
                self._start_profiling(batch_composition_stats)
            # We are in the middle of profiling a given phase
            else:
                self._step_or_stop_profiling(batch_composition_stats)


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "draft_lengths",
        "target_logits_indices",
        "bonus_logits_indices",
        "final_logits_indices",
    ],
    meta_fields=[],
    drop_fields=["draft_lengths_cpu", "req_indices_dp", "req_ids_dp"],
)
@dataclass
class SpecDecodeMetadata:
    """Metadata for speculative decoding on JAX/TPU, containing all necessary indices."""
    draft_lengths: jnp.ndarray
    target_logits_indices: jnp.ndarray
    bonus_logits_indices: jnp.ndarray
    final_logits_indices: jnp.ndarray

    draft_lengths_cpu: Any = field(init=False, default=None)
    req_indices_dp: dict = field(init=False, default_factory=dict)
    req_ids_dp: dict = field(init=False, default_factory=dict)


def host_extract_sampled_tokens(
        runner, spec_decode_metadata: Optional[SpecDecodeMetadata],
        sampled_output: jnp.ndarray, logits_indices_selector: np.ndarray,
        discard_sampled_tokens_req_indices: list, num_reqs: int):
    """host retrieve the sampled tokens for the current step."""
    next_tokens = sampled_output
    if spec_decode_metadata is None:
        next_tokens = np.asarray(jax.device_get(next_tokens))
        # Map tokens back to the pre-dp shuffling order
        if logits_indices_selector is not None:
            next_tokens = next_tokens[logits_indices_selector]
        selected_token_ids = np.expand_dims(next_tokens[:num_reqs], 1)
        valid_sampled_token_ids = selected_token_ids.tolist()
    else:
        valid_sampled_token_ids = runner.rejection_sampler.parse_output(
            next_tokens, runner.input_batch.vocab_size,
            spec_decode_metadata.draft_lengths_cpu, num_reqs,
            spec_decode_metadata.final_logits_indices.shape[0], runner.dp_size,
            spec_decode_metadata.req_indices_dp)
    # Mask out the sampled tokens that should not be sampled.
    for i in discard_sampled_tokens_req_indices:
        valid_sampled_token_ids[i].clear()

    return valid_sampled_token_ids


def get_eos_token_id(model_config: Any) -> tuple[int, ...]:
    """Extract EOS token ID from the model configuration with fallback."""
    eos_token_id = model_config.get_vocab_size() - 1
    if hasattr(model_config, "hf_config"):
        eos_token_id = getattr(model_config.hf_config, "eos_token_id",
                               eos_token_id)
        if eos_token_id is None:
            eos_token_id = model_config.get_vocab_size() - 1

    if isinstance(eos_token_id, int):
        return (eos_token_id, )
    elif isinstance(eos_token_id, list):
        return tuple(eos_token_id)
    elif eos_token_id is None:
        return ()
    else:
        return tuple(eos_token_id)


def get_pad_token_id(model_config: Any) -> int:
    """Extract padding token ID from the model configuration with fallback."""
    padding_token_id = 0
    if hasattr(model_config, "hf_config"):
        padding_token_id = getattr(model_config.hf_config, "pad_token_id", 0)
        if padding_token_id is None:
            padding_token_id = 0
    return padding_token_id
