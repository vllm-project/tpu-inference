import yaml
import json
import unittest

class TestJaxOptionsParsing(unittest.TestCase):
    def test_yaml_to_python_dictionary_fidelity(self):
        # We simulate the exact YAML block we will introduce to the configs
        yaml_content = """
        trace_configs:
          - batch_size: 256
            input_len: 1024
            output_len: 64
            jax_advanced_configuration:
              tpu_enable_periodic_counter_sampling: true
              tpu_tc_perf_counter_sampling_options: 'interval_us:10 scaling:0 counter_size_bits:1 indices:56 indices:57 indices:58 indices:38 indices:105'
              tpu_cmn_perf_counter_sampling_options: 'interval_us:32 scaling:0 counter_size_bits:2 indices:1 indices:2 indices:58'
              num_tensor_cores_to_trace_per_device: 1
        """
        
        # 1. Orchestrator Loads YAML
        parsed_yaml = yaml.safe_load(yaml_content)
        trace_config = parsed_yaml["trace_configs"][0]
        
        # 2. Orchestrator extracts payload and json.dumps it into CLI args
        advanced_config = trace_config.get("jax_advanced_configuration")
        cli_argument_string = json.dumps(advanced_config)
        
        # 3. vLLM Subprocess json.loads it from CLI args
        restored_python_dict = json.loads(cli_argument_string)
        
        # 4. Compare it against the EXACT Hardcoded python representation requested by the user
        exact_python_representation = {
            "tpu_enable_periodic_counter_sampling" : True,
            "tpu_tc_perf_counter_sampling_options" : (
                'interval_us:10 scaling:0 counter_size_bits:1 indices:56 indices:57 indices:58 indices:38 indices:105'
            ),
            "tpu_cmn_perf_counter_sampling_options" : (
                'interval_us:32 scaling:0 counter_size_bits:2 indices:1 indices:2 indices:58'
            ),
            "num_tensor_cores_to_trace_per_device": 1,
        }
        
        self.assertEqual(restored_python_dict, exact_python_representation)
        self.assertEqual(type(restored_python_dict['tpu_enable_periodic_counter_sampling']), bool)
        self.assertEqual(type(restored_python_dict['num_tensor_cores_to_trace_per_device']), int)
        self.assertEqual(type(restored_python_dict['tpu_tc_perf_counter_sampling_options']), str)

if __name__ == '__main__':
    unittest.main(verbosity=2)
