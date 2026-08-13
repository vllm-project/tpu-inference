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
import tempfile

from parse_benchmark_log import parse_benchmark_log


def test_parse_benchmark_log_standard():
    log_content = """
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     1024      
Failed requests:                         0         
Benchmark duration (s):                  441.67    
Total input tokens:                      8388608   
Total generated tokens:                  1048576   
Request throughput (req/s):              2.32      
Output token throughput (tok/s):         2374.10   
Peak output token throughput (tok/s):    5918.00   
Peak concurrent requests:                1024.00   
Total token throughput (tok/s):          21366.87  
---------------Time to First Token----------------
Mean TTFT (ms):                          164004.30 
Median TTFT (ms):                        145423.02 
P99 TTFT (ms):                           364320.86 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          210.93    
Median TPOT (ms):                        241.14    
P99 TPOT (ms):                           273.17    
---------------Inter-token Latency----------------
Mean ITL (ms):                           210.95    
Median ITL (ms):                         238.27    
P99 ITL (ms):                            349.54    
----------------End-to-end Latency----------------
Mean E2EL (ms):                          379782.45 
Median E2EL (ms):                        392323.66 
P99 E2EL (ms):                           441251.24 
==================================================
"""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, "bench.log")
        result_file = os.path.join(temp_dir, "result.txt")

        with open(log_file, "w") as f:
            f.write(log_content)

        parse_benchmark_log(log_file, result_file)

        with open(result_file, "r") as f:
            lines = f.read().splitlines()

        results = dict(line.split("=") for line in lines)
        assert results["Throughput"] == "2.32"
        assert results["OutputTokenThroughput"] == "2374.10"
        assert results["TotalTokenThroughput"] == "21366.87"
        assert results["MedianTTFT"] == "145423.02"
        assert results["P99TTFT"] == "364320.86"
        assert results["MedianTPOT"] == "241.14"
        assert results["P99TPOT"] == "273.17"
        assert results["MedianITL"] == "238.27"
        assert results["P99ITL"] == "349.54"
        assert results["MedianETEL"] == "392323.66"
        assert results["P99ETEL"] == "441251.24"


def test_parse_benchmark_log_accuracy_json():
    log_content = """
AccuracyMetrics: {"accuracy": 0.95, "f1": 0.9}
"""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, "bench.log")
        result_file = os.path.join(temp_dir, "result.txt")

        with open(log_file, "w") as f:
            f.write(log_content)

        parse_benchmark_log(log_file, result_file)

        with open(result_file, "r") as f:
            lines = f.read().splitlines()

        results = dict(line.split("=", 1) for line in lines)
        assert "AccuracyMetrics" in results
        parsed_json = json.loads(results["AccuracyMetrics"])
        assert parsed_json["accuracy"] == 0.95


def test_parse_benchmark_log_accuracy_legacy():
    log_content = """
Some random log before
Results
{'accuracy': 0.88, 'other_metric': 0.5}
"""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, "bench.log")
        result_file = os.path.join(temp_dir, "result.txt")

        with open(log_file, "w") as f:
            f.write(log_content)

        parse_benchmark_log(log_file, result_file)

        with open(result_file, "r") as f:
            lines = f.read().splitlines()

        results = dict(line.split("=", 1) for line in lines)
        assert "AccuracyMetrics" in results
        parsed_json = json.loads(results["AccuracyMetrics"])
        assert parsed_json["accuracy"] == 0.88


def test_parse_benchmark_log_case_insensitive():
    log_content = """
Some random log before
============ Serving Benchmark Result ============
REQUEST THROUGHPUT (req/s):              15.00     
OuTpUt ToKeN tHrOuGhPuT (tok/s):         250.00    
MeDiAn TTFT (ms):                        95.00     
==================================================
"""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, "bench.log")
        result_file = os.path.join(temp_dir, "result.txt")

        with open(log_file, "w") as f:
            f.write(log_content)

        parse_benchmark_log(log_file, result_file)

        with open(result_file, "r") as f:
            lines = f.read().splitlines()

        results = dict(line.split("=") for line in lines)
        assert results["Throughput"] == "15.00"
        assert results["OutputTokenThroughput"] == "250.00"
        assert results["MedianTTFT"] == "95.00"


if __name__ == "__main__":
    print("Running tests without pytest...")
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            print(f"Executing {name}...")
            func()
    print("All tests passed successfully!")
