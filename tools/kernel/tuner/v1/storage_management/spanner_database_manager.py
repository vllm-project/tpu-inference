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

from absl import flags
from google.api_core import retry
from google.cloud import spanner

from tools.kernel.tuner.v1.storage_management.storage_manager import \
    StorageManager

BATCH_SIZE = 1000

FLAGS = flags.FLAGS


class SpannerStorageManager(StorageManager):
    # (TODO)For historical reason, the database_id is still tune-gmm, but it
    # actually contains tuning cases for different kernels, not just gmm. We
    # can consider to rename it in the future for better clarity.
    def __init__(self, worker_id=None, dry_run=False):
        gcp_project_id = FLAGS.gcp_project_id
        spanner_instance_id = FLAGS.spanner_instance_id
        spanner_database_id = FLAGS.spanner_database_id
        self.current_case_id = 0
        self.invalid_count = 0
        self.buffer = []
        self.worker_id = worker_id
        self.dry_run = dry_run
        if not self.dry_run:
            self.client = spanner.Client(project=gcp_project_id,
                                         disable_builtin_metrics=True)
            self.instance = self.client.instance(spanner_instance_id)
            self.database = self.instance.database(spanner_database_id)
        else:
            self.database = None

    def init_case_set(self, case_set_id, scan_space, desc):
        """Initializes the CaseSet row."""
        if self.dry_run:
            return

        def _do_insert(tx):
            tx.execute_update(
                "INSERT INTO CaseSet (ID, Description, Status, ScanSpace) VALUES (@id, @desc, 'CREATING', @scan)",
                params={
                    'id': case_set_id,
                    'desc': desc,
                    'scan': scan_space
                },
                param_types={
                    'id': spanner.param_types.STRING,
                    'desc': spanner.param_types.STRING,
                    'scan': spanner.param_types.INT64
                })

        self.database.run_in_transaction(_do_insert)

    def case_set_id_exists(self, case_set_id) -> bool:
        """Checks whether the given case_set_id already exists in the CaseSet table."""
        if self.dry_run:
            return False

        def _do_query(tx):
            result = tx.execute_sql(
                "SELECT COUNT(*) FROM CaseSet WHERE ID = @id",
                params={'id': case_set_id},
                param_types={'id': spanner.param_types.STRING})
            row = list(result)[0]
            return row[0] > 0

        return self.database.run_in_transaction(_do_query)

    def get_case_set_desc(self, case_set_id) -> str:
        """Gets the description for the given case_set_id from the CaseSet table."""
        if self.dry_run:
            return None

        def _do_query(tx):
            result = tx.execute_sql(
                "SELECT Description FROM CaseSet WHERE ID = @id",
                params={'id': case_set_id},
                param_types={'id': spanner.param_types.STRING})
            row = list(result)[0]
            return row[0]

        return self.database.run_in_transaction(_do_query)

    def finish_case_set(self, case_set_id, valid, invalid, duration):
        """Updates tracking columns upon completion."""
        if self.dry_run:
            return

        def _do_update(tx):
            tx.execute_update(
                "UPDATE CaseSet SET Status = 'COMPLETED', Valid = @v, Invalid = @i, DurationSeconds = @d WHERE ID = @id",
                params={
                    'id': case_set_id,
                    'v': valid,
                    'i': invalid,
                    'd': duration
                },
                param_types={
                    'id': spanner.param_types.STRING,
                    'v': spanner.param_types.INT64,
                    'i': spanner.param_types.INT64,
                    'd': spanner.param_types.FLOAT64
                })

        self.database.run_in_transaction(_do_update)

    def get_case_set_metadata(self, case_set_id):
        if self.dry_run:
            return {}

        def _do_query(tx):
            result = tx.execute_sql(
                "SELECT TpuInferenceHash, BmInfraHash, KernelRuner FROM CaseSet WHERE ID = @id",
                params={'id': case_set_id},
                param_types={'id': spanner.param_types.STRING})
            row = list(result)[0]
            return {
                'tpu_inference_hash': row[0],
                'bm_infra_hash': row[1],
                'kernel_runer': row[2]
            }

        return self.database.run_in_transaction(_do_query)

    @retry.Retry(predicate=retry.if_transient_error)
    def flush(self):
        if not self.buffer or self.dry_run:
            return
        with self.database.batch() as b:
            b.insert(table='KernelTuningCases',
                     columns=('ID', 'CaseId', 'CaseKeyValue'),
                     values=self.buffer)
        self.buffer = []

    def add_tuner_case(self, caseset_id: str, case_id: int, case: str):
        assert isinstance(
            caseset_id, str
        ), f'param caseset_id should be a string but got {type(caseset_id)}'
        assert isinstance(
            case_id,
            int), f'param case_id should be an integer but got {type(case_id)}'
        assert isinstance(
            case, str
        ), f'param case should be a string representing the key:value but got {type(case)}'
        self.buffer.append((caseset_id, case_id, case))
        self.current_case_id += 1
        if len(self.buffer) >= BATCH_SIZE:
            self.flush()

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
        if self.dry_run:
            return
        self.database.run_in_transaction(lambda tx: tx.insert(
            columns=('ID', 'RunId', 'BucketId', 'StartCaseId', 'EndCaseId',
                     'Status', 'WorkerID', 'UpdatedAt'),
            table='WorkBuckets',
            values=[(cs_id, r_id, bucket_id, start_case_id, end_case_id,
                     'PENDING', self.worker_id, spanner.COMMIT_TIMESTAMP)]))

    # tuner agents working on the a bucket will mark the bucket as IN_PROGRESS/COMPLETED
    def mark_bucket_in_progress(self, cs_id, r_id, b_id):
        self.database.run_in_transaction(lambda tx: tx.execute_update(
            "UPDATE WorkBuckets SET Status = 'IN_PROGRESS', WorkerID = @wid, UpdatedAt = PENDING_COMMIT_TIMESTAMP() WHERE ID = @id AND RunId = @rid AND BucketId = @bid",
            params={
                'id': cs_id,
                'rid': r_id,
                'bid': b_id,
                'wid': self.worker_id
            },
            param_types={
                'id': spanner.param_types.STRING,
                'rid': spanner.param_types.STRING,
                'bid': spanner.param_types.INT64,
                'wid': spanner.param_types.STRING
            }))

    def mark_bucket_completed(self, cs_id, r_id, b_id, tt_us):
        self.database.run_in_transaction(lambda tx: tx.execute_update(
            "UPDATE WorkBuckets SET Status = 'COMPLETED', TotalTime = @tt, UpdatedAt = PENDING_COMMIT_TIMESTAMP() WHERE ID = @id AND RunId = @rid AND BucketId = @bid",
            params={
                'id': cs_id,
                'rid': r_id,
                'bid': b_id,
                'tt': tt_us
            },
            param_types={
                'id': spanner.param_types.STRING,
                'rid': spanner.param_types.STRING,
                'bid': spanner.param_types.INT64,
                'tt': spanner.param_types.INT64
            }))

    def get_already_processed_ids(self, cs_id, r_id, start, end):
        query = "SELECT CaseId FROM CaseResults WHERE ID = @id AND RunId = @rid AND CaseId BETWEEN @s AND @e"
        with self.database.snapshot() as snp:
            return {
                row[0]
                for row in snp.execute_sql(query,
                                           params={
                                               'id': cs_id,
                                               'rid': r_id,
                                               's': start,
                                               'e': end
                                           })
            }

    # tuner agents will save the result after completing a tuning batch
    def save_results_batch(self, results):
        if not results:
            return
        with self.database.batch() as b:
            b.insert_or_update(table='CaseResults',
                               columns=('ID', 'RunId', 'CaseId',
                                        'ProcessedStatus', 'WorkerID',
                                        'Latency', 'WarmupTime', 'TotalTime',
                                        'ProcessedAt'),
                               values=results)

    # tuner agents will query from the KernelTuningCases table and run the cases
    def get_bucket_configs(self, cs_id, start, end):
        query = "SELECT ID, CaseId, CaseKeyValue FROM KernelTuningCases WHERE ID = @id AND CaseId BETWEEN @s AND @e ORDER BY CaseId ASC"
        with self.database.snapshot() as snp:
            return {
                row[1]: row
                for row in snp.execute_sql(query,
                                           params={
                                               'id': cs_id,
                                               's': start,
                                               'e': end
                                           })
            }

    def get_total_cases_in_case_set(self, case_set_id):
        """Returns the total number of cases in the given case set.

        Args:
            case_set_id: Unique string identifier for the case set.

        Returns:
            The total number of cases in the case set.
        """
        query = "SELECT Valid FROM CaseSet WHERE ID = @id"
        with self.database.snapshot() as snp:
            result = list(snp.execute_sql(query, params={'id': case_set_id}))
            return result[0][0] if result else 0

    def get_timestamp_sec(self):
        """Returns the current timestamp in seconds since the epoch.

        Used for logging the time of events.

        Returns:
            Current timestamp in seconds.
        """
        return spanner.COMMIT_TIMESTAMP
