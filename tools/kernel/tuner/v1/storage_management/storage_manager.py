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
"""Base class for kernel tuner storage backends.

This module defines the StorageManager abstract base class, which provides the
interface for persisting and retrieving kernel tuning data. Concrete subclasses
implement this interface against a specific backend — either Google Cloud Spanner
for production use, or a local JSON file for lightweight / offline use.

Typical usage:
    Instantiate a concrete subclass (e.g. SpannerStorageManager or
    JsonStorageManager) and pass it to the tuning pipeline. The pipeline interacts
    only with the StorageManager interface, making the backend swappable.
"""


class StorageManager:
    """Abstract base class for kernel tuner storage backends.

    Subclasses must implement all methods to provide a concrete storage backend.
    Two backends are currently supported:
      - Spanner: a fully managed, scalable backend for production tuning runs.
      - Local JSON file: a lightweight backend for offline or development use.
    """

    def __init__(self, instance_id='vllm-bm-inst', database_id='tune-gmm'):
        raise NotImplementedError("Subclasses must implement __init__")

    def init_case_set(self, case_set_id, scan_space, desc):
        """Creates a new CaseSet entry in storage.

        Called at the start of a tuning run to register the case set before any
        cases are written.

        Args:
            case_set_id: Unique string identifier for the case set.
            scan_space: Total number of configurations in the scan space.
            desc: Human-readable description of the case set.
        """
        raise NotImplementedError("Subclasses must implement init_case_set")

    def case_set_id_exists(self, case_set_id) -> bool:
        """Checks whether the given case_set_id already exists in the CaseSet table."""
        raise NotImplementedError(
            "Subclasses must implement case_set_id_exists")

    def get_case_set_desc(self, case_set_id) -> str:
        """Gets the description for the given case_set_id from the CaseSet table."""
        raise NotImplementedError(
            "Subclasses must implement get_case_set_desc")

    def finish_case_set(self, case_set_id, valid, invalid, duration):
        """Marks a CaseSet as completed and records summary statistics.

        Called at the end of a tuning run after all cases have been written.

        Args:
            case_set_id: Unique string identifier for the case set.
            valid: Number of valid cases that were written.
            invalid: Number of invalid cases that were skipped.
            duration: Total wall-clock time in seconds for the tuning run.
        """
        raise NotImplementedError("Subclasses must implement finish_case_set")

    def get_case_set_metadata(self, case_set_id):
        """Retrieves metadata associated with a case set.

        Args:
            case_set_id: Unique string identifier for the case set.

        Returns:
            A dict with keys:
                'tpu_inference_hash': git hash of the tpu-inference repo.
                'bm_infra_hash': git hash of the bm-infra repo.
                'kernel_runner': name of the KernelTunerRunner class used.
        """
        raise NotImplementedError(
            "Subclasses must implement get_case_set_metadata")

    def flush(self):
        """Flushes any buffered tuning cases to the backend storage.

        Implementations that buffer writes (e.g. for batching) must commit all
        pending data when this is called.
        """
        raise NotImplementedError("Subclasses must implement flush")

    def add_tuner_case(self, caseset_id: str, case_id: int, case: str):
        """Buffers a single tuning case for storage.

        Implementations may batch writes internally and flush automatically when
        a buffer threshold is reached.

        Args:
            caseset_id: Unique string identifier for the case set.
            case_id: Integer index of this case within the case set.
            case: String encoding of the case in 'key:value' format.
        """
        raise NotImplementedError("Subclasses must implement add_tuner_case")

    def create_bucket_for_run(self, cs_id: str, r_id: int, bucket_id: int,
                              start_case_id: int, end_case_id: int):
        """Creates a new work bucket for a tuning run.

        Used by tuner agents to define discrete units of work (buckets) that can
        be claimed and processed independently.

        Args:
            cs_id: Case set ID the bucket belongs to.
            r_id: Run ID the bucket belongs to.
            bucket_id: Unique integer identifier for the bucket within the run.
            start_case_id: Starting case ID (inclusive) for this bucket.
            end_case_id: Ending case ID (inclusive) for this bucket.
        """
        raise NotImplementedError(
            "Subclasses must implement create_buckets_for_run")

    def mark_bucket_in_progress(self, cs_id, r_id, b_id):
        """Marks a work bucket as IN_PROGRESS, claiming it for the current worker.

        Used by tuner agents to coordinate work distribution and avoid duplicate
        processing across workers.

        Args:
            cs_id: Case set ID the bucket belongs to.
            r_id: Run ID the bucket belongs to.
            b_id: Bucket ID to claim.
        """
        raise NotImplementedError(
            "Subclasses must implement mark_bucket_in_progress")

    def mark_bucket_completed(self, cs_id, r_id, b_id, tt_us):
        """Marks a work bucket as COMPLETED and records its total processing time.

        Args:
            cs_id: Case set ID the bucket belongs to.
            r_id: Run ID the bucket belongs to.
            b_id: Bucket ID to mark as completed.
            tt_us: Total processing time for the bucket in microseconds.
        """
        raise NotImplementedError(
            "Subclasses must implement mark_bucket_completed")

    def get_already_processed_ids(self, cs_id, r_id, start, end):
        """Returns case IDs that have already been processed within a range.

        Used by tuner agents to resume interrupted runs without reprocessing
        completed cases.

        Args:
            cs_id: Case set ID to query.
            r_id: Run ID to query.
            start: Start of the case ID range (inclusive).
            end: End of the case ID range (inclusive).

        Returns:
            A set of integer case IDs that have already been processed.
        """
        raise NotImplementedError(
            "Subclasses must implement get_already_processed_ids")

    def save_results_batch(self, results):
        """Persists a batch of tuning results to the backend.

        Called by tuner agents after completing a batch of cases.

        Args:
            results: A list of result tuples, each containing fields for
                CaseResults (ID, RunId, CaseId, ProcessedStatus, WorkerID,
                Latency, WarmupTime, TotalTime, ProcessedAt).
        """
        raise NotImplementedError(
            "Subclasses must implement save_results_batch")

    def get_bucket_configs(self, cs_id, start, end):
        """Retrieves the tuning case configurations for a range of case IDs.

        Called by tuner agents to fetch the cases they need to run.

        Args:
            cs_id: Case set ID to query.
            start: Start of the case ID range (inclusive).
            end: End of the case ID range (inclusive).

        Returns:
            A dict mapping case_id (int) to the corresponding storage row for
            that case.
        """
        raise NotImplementedError(
            "Subclasses must implement get_bucket_configs")

    def get_total_cases_in_case_set(self, case_set_id):
        """Returns the total number of cases in the given case set.

        Args:
            case_set_id: Unique string identifier for the case set.

        Returns:
            The total number of cases in the case set.
        """
        raise NotImplementedError(
            "Subclasses must implement get_total_cases_in_case_set")

    def get_timestamp_sec(self):
        """Returns the current timestamp in seconds since the epoch.

        Used for logging the time of events.

        Returns:
            Current timestamp in seconds.
        """
        raise NotImplementedError(
            "Subclasses must implement get_timestamp_sec")
