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

# define a class that can accept a TuningKey and TunableParams, then construct the TuningCase and log it to a file, which is specified when created.
# and it should also able to return the list of TuningCases that have been logged so far.

from tools.kernel.tuner.v1.common.kernel_tuner_base import (TunableParams,
                                                            TuningCase,
                                                            TuningKey)


class TuningCaseLogger:

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

    def append_dict_to_json_file(self, object: dict):
        import json
        with open(self.log_file_path, 'a') as f:
            f.write(f"{json.dumps(object)}\n")


if __name__ == "__main__":
    from tpu_inference.kernels.experimental.batched_rpa.tuned_params import (
        TunableParams, TuningKey)
    reader = TuningCaseLogger(
        log_file_path='/tmp/batched_rpa_gemma4_tuning_cases_1308693.log',
        key_class=TuningKey,
        params_class=TunableParams)
    tuning_cases = reader.get_logged_tuning_cases()
    for case in tuning_cases:
        print(case)
