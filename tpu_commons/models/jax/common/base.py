import dataclasses
from dataclasses import dataclass, fields
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from typing import Any, Callable, Iterable, Mapping

from tpu_commons.logger import init_logger

# Type alias for Initializer for cleaner type hints
Initializer = Callable[..., jax.Array]
logger = init_logger(__name__)

@dataclass
class Config:
    """Base configuration class with a robust factory method.

    This class provides a `from_cfg` classmethod that allows creating a config
    instance from a dictionary, ensuring that all required fields are present
    and ignoring any extraneous keys.
    """

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any] | None = None, **kwargs):
        """Creates a config instance from a dictionary and/or keyword arguments.

        This factory method validates that all fields without default values
        are provided in the input dictionary or keyword arguments.

        Args:
            cfg: A dictionary of configuration parameters.
            **kwargs: Additional configuration parameters passed as keyword arguments.

        Returns:
            An instance of the configuration class.

        Raises:
            ValueError: If any required parameters are missing.
        """
        if cfg is None:
            cfg = {}
        cfg.update(kwargs)

        required_params = {
            f.name
            for f in fields(cls) if f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        }

        # Check if any of the truly required parameters are missing from the provided config.
        missing_params = required_params - set(cfg.keys())
        if missing_params:
            raise ValueError(
                f"Missing required parameters for {cls.__name__}: {', '.join(sorted(list(missing_params)))}"
            )

        known_params = {f.name for f in fields(cls)}
        filtered_cfg = {k: v for k, v in cfg.items() if k in known_params}

        return cls(**filtered_cfg)


    # TODO: check logic with some unit tests.
    def maybe_apply_overrides(self):
        """Update the args with additional_configs, hf_overrides, and override_generation_config settings.
        If there is overlap in overrides between the configs, then print a warning declaring which 
        overrides will take precedent."""
        if not getattr(self, "vllm_config"):
            return
        def _overrides_keys(keys: Iterable[str], original_dict: Mapping[str, Any], new_dict: Mapping[str, Any]):
            overriden_keys = []
            for key in keys:
                if original_dict[key] != new_dict[key]:
                    overriden_keys.append(key)
            return overriden_keys
        
        def _overrides_str(original: str, original_val: Any, new_val: Any):
            return f"{original}: {original_val} ---> {new_val}"
        
        ## Update the args with overrides from each supplied config and log any overlapping
        # overrides that will take precedent.
        overrides_dict = {}
        vllm_model_config = self.vllm_config.model_config
        additional_config_keys, hf_overrides_keys, generation_overrides_keys = [set()] * 3
        
        #  Add additional_config args (which has the lowest priority).
        if self.vllm_config.additional_config:
            additional_config_keys = set(self.vllm_config.additional_config)
            overrides_dict.update(self.vllm_config.additional_config)
        
        # hf_overrides has higher priority than additional_config args
        if vllm_model_config.hf_overrides:
            hf_overrides_keys = set(vllm_model_config.hf_overrides)
            overrides_dict.update(vllm_model_config.hf_overrides)
            intersecting_keys = hf_overrides_keys.intersection(additional_config_keys)
            if intersecting_keys:
                overriden_keys = _overrides_keys(intersecting_keys,
                                                 self.vllm_config.additional_config,
                                                 vllm_model_config.hf_overrides)
                if overriden_keys:
                    overriden_keys_str = "\n".join([_overrides_str(key,
                                                                    self.vllm_config.additional_config[key],
                                                                    vllm_model_config.hf_overrides[key])
                                                        for key in overriden_keys]
                                                    )
                    logger.warning(f"Overriding additional_config arguments with the following hf_overrides:\n{overriden_keys_str}")

        # override_generation_cofnig has higher priority than the other args.
        if vllm_model_config.override_generation_config:
            intersecting_keys = set()
            generation_overrides_keys = set(vllm_model_config.override_generation_config)
            overrides_dict.update(vllm_model_config.override_generation_config)
            previous_configs = [vllm_model_config.hf_overrides, self.vllm_config.additional_config]
            previous_keys = [hf_overrides_keys, additional_config_keys]
            prev_config_type = ["hf_overrides", "additional_config"]
            for i in range(len(previous_keys)):
                new_intersecting_keys = (generation_overrides_keys & previous_keys[i]) - intersecting_keys
                # Update with intersecting keys only found with the current config.
                intersecting_keys = intersecting_keys | new_intersecting_keys
                overriden_keys = _overrides_keys(new_intersecting_keys,
                                                 previous_configs[i],
                                                 vllm_model_config.override_generation_config)
                if overriden_keys:
                    overriden_keys_str = "\n".join([_overrides_str(key,
                                                                    previous_configs[i][key],
                                                                    vllm_model_config.override_generation_config[key])
                                                        for key in overriden_keys]
                                                    )
                    logger.warning(f"Overriding {prev_config_type[i]} arguments with the following generation_overrides:\n{overriden_keys_str}")

        # Update the args with the overrides
        for field in fields(self):
            if field.name in overrides_dict:
                setattr(self, field.name, overrides_dict[field.name])


    def __post_init__(self):
        self.maybe_apply_overrides()
    

@dataclasses.dataclass
class ParamFactory:
    """A factory for creating nnx.Param objects with shared RNGs and initializers.

    This class simplifies the creation of parameters by holding common
    configuration like the RNG stream and the weight initialization function.

    Attributes:
        rngs: An `nnx.Rngs` object to provide RNG streams for parameter initialization.
        initializer: A callable (e.g., a kernel initializer from JAX) used to
            generate parameter data.
    """
    kernel_initializer: Initializer
    scale_initializer: Initializer

    def _create_param(self,
                      initializer: Initializer,
                      rngs: nnx.Rngs,
                      shape: tuple[int, ...],
                      sharding: NamedSharding,
                      dtype: Any = jnp.float32) -> nnx.Param:
        """Private helper to create a sharded parameter with a given initializer."""
        sharded_initializer = jax.jit(
            initializer,
            static_argnames=('shape', 'dtype'), 
            out_shardings=sharding
        )
        key = rngs.params()
        param_data = sharded_initializer(key, shape, dtype)
        return nnx.Param(param_data, sharding=sharding)

    def create_kernel_param(self, *args, **kwargs) -> nnx.Param:
        """Creates a kernel/weight parameter using the kernel_initializer."""
        return self._create_param(self.kernel_initializer, *args, **kwargs)

    def create_scale_param(self, *args, **kwargs) -> nnx.Param:
        """Creates a scale/gain parameter using the scale_initializer."""
        return self._create_param(self.scale_initializer, *args, **kwargs)