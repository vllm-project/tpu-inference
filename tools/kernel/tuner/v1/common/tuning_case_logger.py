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

from tools.kernel.tuner.v1.common.tuner_datatypes import (TunableParams,
                                                          TuningCase,
                                                          TuningKey)


class TuningCaseLogger:
    """
    A utility class for persistently logging and retrieving kernel tuning cases.

    This logger accepts a `TuningKey` and `TunableParams`, constructs a `TuningCase`, 
    and appends it to a specified file. It also provides functionality to parse this 
    file and reconstruct a list of all logged cases.

    Typical Workflow:
        1. Logging: Instantiate this class inside `tuned_params.py` (alongside your 
           `TuningKey`, `TunableParams`, and `get_tuned_params` definitions) to record 
           the specific configurations targeted for tuning.
        2. Retrieval: Use this logger within the kernel tuner's `generate_cases` method 
           to load the logged cases and feed them into the tuning pipeline.

    Note:
        The resulting log files are typically stored as JSON in the 
        `kernel/tuner/v1/tuning_cases` directory.
    """

    def __init__(self, log_file_path: str, key_class=None, params_class=None):
        self.log_file_path = log_file_path
        self.key_class = key_class
        self.params_class = params_class

    def log_tuning_case(self, tuning_key: TuningKey,
                        tunable_params: TunableParams):
        tuning_case = TuningCase(tuning_key=tuning_key,
                                 tunable_params=tunable_params)

        with open(self.log_file_path, 'a') as f:
            f.write(f"{str(tuning_case)}\n")

    def get_logged_tuning_cases(self, ) -> list[TuningCase]:
        tuning_cases = []
        with open(self.log_file_path, 'r') as f:
            for line in f:
                if not line:
                    continue
                tuning_cases.append(
                    TuningCase.from_string(
                        line.strip(),
                        tuning_key_class=self.key_class,
                        tunable_params_class=self.params_class))
        return tuning_cases
