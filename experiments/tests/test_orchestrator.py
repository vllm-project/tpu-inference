import unittest
import os
import tempfile
import sys
import yaml
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
run_sweep = __import__('run_sweep_orchestrator')

class TestOrchestratorExp(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.test_dir.name, "test_exp_config.yaml")

    def tearDown(self):
        self.test_dir.cleanup()

    def create_mock_config(self, content):
        with open(self.config_path, "w") as f:
            yaml.dump(content, f)
            
    def write_raw_yaml_string(self, content):
        with open(self.config_path, "w") as f:
            f.write(content)

    @patch("run_sweep_orchestrator.subprocess.run")
    @patch("run_sweep_orchestrator.datetime")
    def test_chronological_id_generation(self, mock_datetime, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        mock_now = MagicMock()
        mock_now.strftime.return_value = "20261111_000000"
        mock_datetime.now.return_value = mock_now

        self.create_mock_config({
             "sweep_matrix": {
                 "model": ["test/model"],
                 "batch_size": [1],
                 "input_len": [128],
                 "output_len": [64]
             }
         })

        test_args = ["--config", self.config_path, "--result-dir", self.test_dir.name]
        with patch("sys.argv", ["run_sweep_orchestrator.py"] + test_args):
            run_sweep.main()

        expected_dir = os.path.join(self.test_dir.name, "20261111_000000")
        self.assertTrue(os.path.exists(expected_dir))

    @patch("run_sweep_orchestrator.subprocess.run")
    def test_custom_experiment_id_and_resumption(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        self.create_mock_config({
            "sweep_matrix": {
                 "model": ["meta-llama/test", "google/test2"],
                 "batch_size": [1, 2],
                 "input_len": [128],
                 "output_len": [16]
            }
        })

        test_args = ["--config", self.config_path, "--result-dir", self.test_dir.name, "--experiment-id", "RESUME_ID_X"]

        with patch("sys.argv", ["run_sweep_orchestrator.py"] + test_args):
            run_sweep.main()

        self.assertEqual(mock_run.call_count, 4)

        expected_dir = os.path.join(self.test_dir.name, "RESUME_ID_X")
        csv_file = os.path.join(expected_dir, "results.csv")
        
        import csv
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["model", "batch_size", "input_len", "output_len"])
            writer.writeheader()
            writer.writerow({"model": "meta-llama/test", "batch_size": "1", "input_len": "128", "output_len": "16"})
            writer.writerow({"model": "meta-llama/test", "batch_size": "2", "input_len": "128", "output_len": "16"})

        mock_run.reset_mock()
        
        with patch("sys.argv", ["run_sweep_orchestrator.py"] + test_args):
            run_sweep.main()

        self.assertEqual(mock_run.call_count, 2)
        
    @patch("run_sweep_orchestrator.subprocess.run")
    def test_backward_compatibility_migration(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        self.create_mock_config({
             "model": "google/model-old",
             "sweep_matrix": {
                 "batches": [4],
                 "inputs": [32],
                 "output_lens": [32]
             }
         })

        test_args = ["--config", self.config_path, "--result-dir", self.test_dir.name, "--experiment-id", "EXP_COMPAT"]
        with patch("sys.argv", ["run_sweep_orchestrator.py"] + test_args):
            run_sweep.main()

        cmd = mock_run.call_args[0][0]
        self.assertIn("--model", cmd)
        self.assertIn("google/model-old", cmd)
        self.assertIn("--batch-size", cmd)

    @patch("run_sweep_orchestrator.subprocess.run")
    def test_multiple_models_via_yaml_string(self, mock_run):
        # Validate exact inline string injection parsing flow lists uniformly simulating exact PyYAML behavior mapping over explicit multi-model boundaries
        mock_run.return_value = MagicMock(returncode=0)
        yaml_content = """
sweep_matrix:
  model: [meta-llama/test-1, google/test-2]
  batch_size: [1, 2]
  input_len: [128]
  output_len: [32]
"""
        self.write_raw_yaml_string(yaml_content)

        test_args = ["--config", self.config_path, "--result-dir", self.test_dir.name, "--experiment-id", "MULTI_MODEL_STR"]
        with patch("sys.argv", ["run_sweep_orchestrator.py"] + test_args):
            run_sweep.main()

        # 2 models * 2 batch_sizes * 1 * 1 = 4 total iterations
        self.assertEqual(mock_run.call_count, 4)
        
        # Capture the iterations
        models_called = []
        for call_arg in mock_run.call_args_list:
            cmd = call_arg[0][0] # The command list
            # The model is the index after "--model"
            if "--model" in cmd:
                model_name = cmd[cmd.index("--model") + 1]
                models_called.append(model_name)
                
        self.assertEqual(models_called.count("meta-llama/test-1"), 2)
        self.assertEqual(models_called.count("google/test-2"), 2)


if __name__ == "__main__":
    unittest.main()
