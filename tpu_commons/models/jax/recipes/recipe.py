from dataclasses import dataclass, fields
import json

from tpu_commons.models.jax.common.base import Config
from tpu_commons.models.jax.common.sharding import ShardingConfig
from tpu_commons.models.jax.common.model import ModelConfig

@dataclass(frozen=True)
class RecipeConfig():
    model: ModelConfig
    sharding: ShardingConfig
    serving: Config

    def __str__(self):
        def get_serializable_dict(dc_object):
            """Recursively build a dictionary, respecting the repr flag."""
            d = {}
            for f in fields(dc_object):
                if f.repr:
                    value = getattr(dc_object, f.name)
                    if hasattr(value, '__dataclass_fields__'): # Check if it's a dataclass
                        d[f.name] = get_serializable_dict(value)
                    else:
                        d[f.name] = value
            return d

        final_dict = get_serializable_dict(self)
        
        return json.dumps(final_dict, indent=4, default=str)
