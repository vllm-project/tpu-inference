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

import json
import os
from datetime import datetime

from tools.kernel.tuner.v1.common.storage_manager import StorageManager
from tools.kernel.tuner.v1.common.utils import get_host_ip

BATCH_SIZE = 1000


class LocalDbManager(StorageManager):
    """Local JSON-file-backed implementation of StorageManager.

    Models the database as a folder (default: /tmp/kernel_tuner_run_YYYY_MM_DD)
    where each Spanner table is persisted as a JSON file. All writes are also
    printed to the console for visibility.
    """

    def __init__(self,
                 instance_id='vllm-bm-inst',
                 database_id='tune-gmm',
                 worker_id=None,
                 dry_run=False,
                 db_path=None):
        self.current_case_id = 0
        self.invalid_count = 0
        self.buffer = []
        self.worker_id = worker_id or get_host_ip()
        self.dry_run = dry_run
        if db_path is None:
            date_str = datetime.now().strftime('%Y_%m_%d')
            db_path = f'/tmp/kernel_tuner_run_{date_str}'
        self.db_path = db_path
        if not self.dry_run:
            os.makedirs(self.db_path, exist_ok=True)
            print(f'[LocalDbManager] Database initialized at {self.db_path}')

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _table_path(self, table_name):
        return os.path.join(self.db_path, f'{table_name}.json')

    def _read_table(self, table_name):
        path = self._table_path(table_name)
        if not os.path.exists(path):
            return []
        with open(path, 'r') as f:
            return json.load(f)

    def _write_table(self, table_name, data):
        path = self._table_path(table_name)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # StorageManager interface
    # ------------------------------------------------------------------

    def init_case_set(self, case_set_id, scan_space, desc):
        if self.dry_run:
            return
        row = {
            'ID': case_set_id,
            'Description': desc,
            'Status': 'CREATING',
            'ScanSpace': scan_space
        }
        table = self._read_table('CaseSet')
        table.append(row)
        self._write_table('CaseSet', table)
        print(f'[LocalDbManager] init_case_set: {row}')

    def case_set_id_exists(self, case_set_id) -> bool:
        if self.dry_run:
            return False
        table = self._read_table('CaseSet')
        for row in table:
            if row['ID'] == case_set_id:
                return True
        return False

    def get_case_set_desc(self, case_set_id) -> str:
        if self.dry_run:
            return None
        table = self._read_table('CaseSet')
        for row in table:
            if row['ID'] == case_set_id:
                return row['Description']
        return None

    def finish_case_set(self, case_set_id, valid, invalid, duration):
        if self.dry_run:
            return
        table = self._read_table('CaseSet')
        for row in table:
            if row['ID'] == case_set_id:
                row.update({
                    'Status': 'COMPLETED',
                    'Valid': valid,
                    'Invalid': invalid,
                    'DurationSeconds': duration
                })
                break
        self._write_table('CaseSet', table)
        print(
            f'[LocalDbManager] finish_case_set: ID={case_set_id}, valid={valid}, invalid={invalid}, duration={duration}s'
        )

    def get_case_set_metadata(self, case_set_id):
        if self.dry_run:
            return {}
        table = self._read_table('CaseSet')
        for row in table:
            if row['ID'] == case_set_id:
                return {
                    'tpu_inference_hash': row.get('TpuInferenceHash'),
                    'bm_infra_hash': row.get('BmInfraHash'),
                    'kernel_runer': row.get('KernelRuner'),
                }
        return {}

    def flush(self):
        if not self.buffer or self.dry_run:
            return
        table = self._read_table('KernelTuningCases')
        for caseset_id, case_id, case_kv in self.buffer:
            table.append({
                'ID': caseset_id,
                'CaseId': case_id,
                'CaseKeyValue': case_kv
            })
        self._write_table('KernelTuningCases', table)
        print(
            f'[LocalDbManager] flush: wrote {len(self.buffer)} cases to KernelTuningCases'
        )
        self.buffer = []

    def add_tuner_case(self, caseset_id: str, case_id: int, case: str):
        assert type(
            caseset_id
        ) == str, f'param caseset_id should be a string but got {type(caseset_id)}'
        assert type(
            case_id
        ) == int, f'param case_id should be an integer but got {type(case_id)}'
        assert type(
            case
        ) == str, f'param case should be a string representing the key:value but got {type(case)}'
        self.buffer.append((caseset_id, case_id, case))
        self.current_case_id += 1
        if len(self.buffer) >= BATCH_SIZE:
            self.flush()

    def mark_bucket_in_progress(self, cs_id, r_id, b_id):
        table = self._read_table('WorkBuckets')
        for row in table:
            if row['ID'] == cs_id and row['RunId'] == r_id and row[
                    'BucketId'] == b_id:
                row.update({
                    'Status': 'IN_PROGRESS',
                    'WorkerID': self.worker_id,
                    'UpdatedAt': datetime.now().isoformat()
                })
                break
        else:
            # Row not found; insert it so local runs can proceed without pre-seeded buckets.
            table.append({
                'ID': cs_id,
                'RunId': r_id,
                'BucketId': b_id,
                'Status': 'IN_PROGRESS',
                'WorkerID': self.worker_id,
                'UpdatedAt': datetime.now().isoformat()
            })
        self._write_table('WorkBuckets', table)
        print(
            f'[LocalDbManager] mark_bucket_in_progress: cs_id={cs_id}, r_id={r_id}, b_id={b_id}, worker={self.worker_id}'
        )

    def mark_bucket_completed(self, cs_id, r_id, b_id, tt_us):
        table = self._read_table('WorkBuckets')
        for row in table:
            if row['ID'] == cs_id and row['RunId'] == r_id and row[
                    'BucketId'] == b_id:
                row.update({
                    'Status': 'COMPLETED',
                    'TotalTime': tt_us,
                    'UpdatedAt': datetime.now().isoformat()
                })
                break
        self._write_table('WorkBuckets', table)
        print(
            f'[LocalDbManager] mark_bucket_completed: cs_id={cs_id}, r_id={r_id}, b_id={b_id}, tt_us={tt_us}'
        )

    def get_already_processed_ids(self, cs_id, r_id, start, end):
        table = self._read_table('CaseResults')
        return {
            row['CaseId']
            for row in table if row['ID'] == cs_id and row['RunId'] == r_id
            and start <= row['CaseId'] <= end
        }

    def save_results_batch(self, results):
        if not results:
            return
        cols = ('ID', 'RunId', 'CaseId', 'ProcessedStatus', 'WorkerID',
                'Latency', 'WarmupTime', 'TotalTime', 'ProcessedAt')
        table = self._read_table('CaseResults')
        # Build lookup for insert-or-update semantics (mirrors Spanner's insert_or_update).
        index = {
            (row['ID'], row['RunId'], row['CaseId']): i
            for i, row in enumerate(table)
        }
        for result in results:
            row = dict(zip(cols, result))
            key = (row['ID'], row['RunId'], row['CaseId'])
            if key in index:
                table[index[key]] = row
            else:
                index[key] = len(table)
                table.append(row)
        self._write_table('CaseResults', table)
        print(
            f'[LocalDbManager] save_results_batch: saved {len(results)} results to CaseResults'
        )

    def get_bucket_configs(self, cs_id, start, end):
        table = self._read_table('KernelTuningCases')
        return {
            row['CaseId']: (row['ID'], row['CaseId'], row['CaseKeyValue'])
            for row in table
            if row['ID'] == cs_id and start <= row['CaseId'] <= end
        }
