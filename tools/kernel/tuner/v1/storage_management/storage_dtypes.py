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
"""Data types and dataclasses for storage management."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RetryTable:
    """Dataclass representing a record in the RetryTable Spanner database.

    Attributes:
        case_set_id: Unique identifier for the case set (CaseSetId).
        run_id: Unique identifier for the tuning run (RunId).
        case_id: Integer identifier for the specific case (CaseId).
        n_retried: Number of retries attempted for this case (n_retried).
    """

    case_set_id: str
    run_id: str
    case_id: int
    n_retried: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass fields to Spanner column names and values."""
        return {
            'CaseSetId': self.case_set_id,
            'RunId': self.run_id,
            'CaseId': self.case_id,
            'n_retried': self.n_retried,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetryTable':
        """Creates a RetryTable instance from a dictionary.

        Supports both Spanner schema column names (e.g. 'CaseSetId') and
        snake_case attribute names (e.g. 'case_set_id').
        """
        case_set_id = data.get('CaseSetId') if 'CaseSetId' in data else data.get('case_set_id')
        run_id = data.get('RunId') if 'RunId' in data else data.get('run_id')
        case_id = data.get('CaseId') if 'CaseId' in data else data.get('case_id')
        n_retried = data.get('n_retried')

        return cls(
            case_set_id=case_set_id,
            run_id=run_id,
            case_id=case_id,
            n_retried=n_retried,
        )
