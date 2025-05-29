from dataclasses import dataclass, fields
from typing import Type, Union

from vllm.config import SchedulerConfig


@dataclass
class ExperimentalSchedulerConfig(SchedulerConfig):
    enable_chunked_prefill: bool = False
    policy: str = "fcfs"
    num_scheduler_steps: int = 1
    scheduler_cls: Union[str, Type[object]] = (
        "tpu_commons.core.experimental_scheduler.ExperimentalScheduler")

    @classmethod
    def initialize_from_config(
        cls,
        vllm_scheduler_config: SchedulerConfig,
        experimental_scheduler_config: dict,
    ):
        scheduler_config = {
            field.name: getattr(vllm_scheduler_config, field.name)
            for field in fields(vllm_scheduler_config) if field.init
        }
        scheduler_config["scheduler_cls"] = (
            "tpu_commons.core.experimental_scheduler.ExperimentalScheduler")
        for k, v in experimental_scheduler_config.items():
            scheduler_config[k] = v
        return cls(**scheduler_config)
