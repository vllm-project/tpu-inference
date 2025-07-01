import unittest
from dataclasses import dataclass, field, fields
from vllm.config import ModelConfig
from typing import Any, List, Mapping

from tpu_commons.models.jax.common.base import Config

def setup_vllm_config(subconfig_types: List[str], overrides: List[Mapping[str, Any]]):
    vllm_config = SimpleVllmConfig()
    for (subconfig_type, override) in zip(subconfig_types, overrides):
        if subconfig_type == "model":
            for key in override:
                setattr(vllm_config.model_config, key, override[key])
        else:
            for key in override:
                setattr(vllm_config, key, override[key])
    return vllm_config

@dataclass
class SimpleVllmConfig():
    additional_config: Mapping[str, Any] = field(default_factory=dict)
    model_config: ModelConfig = field(default_factory=ModelConfig)

@dataclass
class SimpleConfig(Config):
    vllm_config: SimpleVllmConfig
    arg1: str
    arg2: str
    arg3: int

    def is_equal(self, other: Config):
        for f in fields(self):
            if f.name != "vllm_config":
                if getattr(self, f.name) != getattr(other, f.name):
                    return False
        return True


class ConfigOverrideTests(unittest.TestCase):

    
    def test_additional_config_overrides(self):
        subconfig_types = ['']
        overrides = [{"additional_config": {"arg1": "val1", "arg2": "val2"}}]
        override_vllm_config = setup_vllm_config(subconfig_types, overrides)
        default_vllm_config = SimpleVllmConfig()
        config = SimpleConfig(vllm_config=override_vllm_config, arg1="foo", arg2="bar", arg3=123)
        expected_config = SimpleConfig(vllm_config=default_vllm_config, arg1="val1", arg2="val2", arg3=123)
        self.assertTrue(config.is_equal(expected_config))

    def test_hf_overrides(self):
        subconfig_types = ['model']
        overrides = [{"hf_overrides": {"arg2": "val2", "arg3": 456}}]
        default_vllm_config = SimpleVllmConfig()
        override_vllm_config = setup_vllm_config(subconfig_types, overrides)
        config = SimpleConfig(vllm_config=override_vllm_config, arg1="foo", arg2="bar", arg3=123)
        expected_config = SimpleConfig(vllm_config=default_vllm_config, arg1="foo", arg2="val2", arg3=456)
        self.assertTrue(config.is_equal(expected_config))

    def test_additional_and_hf_overrides(self):
        subconfig_types = ['', 'model']
        overrides = [{"additional_config": {"arg1": "val1", "arg2": "val2"}},
                     {"hf_overrides": {"arg2": "val3", "arg3": 456}}]
        default_vllm_config = SimpleVllmConfig()
        override_vllm_config = setup_vllm_config(subconfig_types, overrides)
        config = SimpleConfig(vllm_config=override_vllm_config, arg1="foo", arg2="bar", arg3=123)
        expected_config = SimpleConfig(vllm_config=default_vllm_config, arg1="val1", arg2="val3", arg3=456)
        self.assertTrue(config.is_equal(expected_config))

    def test_additional_and_generate_overrides(self):
        subconfig_types = ['', 'model']
        overrides = [{"additional_config": {"arg1": "val1", "arg2": "val2"}},
                     {"override_generation_config": {"arg2": "val3", "arg3": 456}}]
        default_vllm_config = SimpleVllmConfig()
        override_vllm_config = setup_vllm_config(subconfig_types, overrides)
        config = SimpleConfig(vllm_config=override_vllm_config, arg1="foo", arg2="bar", arg3=123)
        expected_config = SimpleConfig(vllm_config=default_vllm_config, arg1="val1", arg2="val3", arg3=456)
        self.assertTrue(config.is_equal(expected_config))

    def test_hf_and_generate_overrides(self):
        subconfig_types = ['model', 'model']
        overrides = [{"hf_overrides": {"arg2": "val2", "arg3": 456}},
                     {"override_generation_config": {"arg2": "val4", "arg3": 789}}]
        default_vllm_config = SimpleVllmConfig()
        override_vllm_config = setup_vllm_config(subconfig_types, overrides)
        config = SimpleConfig(vllm_config=override_vllm_config, arg1="foo", arg2="bar", arg3=123)
        expected_config = SimpleConfig(vllm_config=default_vllm_config, arg1="foo", arg2="val4", arg3=789)
        self.assertTrue(config.is_equal(expected_config))

    def test_additional_and_hf_and_generate_overrides(self):
        subconfig_types = ['', 'model', 'model']
        overrides = [{"additional_config": {"arg1": "val1", "arg2": "val2"}},
                     {"hf_overrides": {"arg2": "val2", "arg3": 456}},
                     {"override_generation_config": {"arg1": "val3", "arg2": "val4", "arg3": 789}}]
        default_vllm_config = SimpleVllmConfig()
        override_vllm_config = setup_vllm_config(subconfig_types, overrides)
        config = SimpleConfig(vllm_config=override_vllm_config, arg1="foo", arg2="bar", arg3=123)
        expected_config = SimpleConfig(vllm_config=default_vllm_config, arg1="val3", arg2="val4", arg3=789)
        self.assertTrue(config.is_equal(expected_config))


if __name__ == '__main__':
    unittest.main()

