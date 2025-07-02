import dataclasses
from dataclasses import dataclass, fields
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding

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

        def _overrides_str(original: str, original_val: Any,
                           new_val: Any) -> str:
            return f"{original}: {original_val} ---> {new_val}"

        def _get_overrides_dict(self) -> Mapping[str, Any]:
            """Return the overrides from all of the possible vllm sections."""
            overrides_dict = {}
            vllm_model_config = self.vllm_config.model_config

            for override_type in ordered_override_types:
                if override_type == "additional_config":
                    overrides_dict[
                        override_type] = self.vllm_config.additional_config
                else:
                    overrides_dict[override_type] = getattr(
                        vllm_model_config, override_type)
            return overrides_dict

        ordered_override_types = [
            "additional_config", "hf_overrides", "override_generation_config"
        ]

        overrides_dict = _get_overrides_dict(self)

        # Override the config values using the vLLM sections with highest
        # precedence first.
        for field in fields(self):
            selected_type = None
            for override_type in reversed(ordered_override_types):
                if field.name in overrides_dict[override_type]:
                    setattr(self, field.name,
                            overrides_dict[override_type][field.name])
                    selected_type = override_type
                    break
            if selected_type is None:
                continue

            # If multiple vLLM sections contain overrides, print a warning.
            for override_type in ordered_override_types:
                if override_type == selected_type:
                    break
                else:
                    if field.name in overrides_dict[override_type]:
                        overriden_keys_str = _overrides_str(
                            field.name,
                            overrides_dict[override_type][field.name],
                            overrides_dict[selected_type][field.name])
                        logger.warning(
                            f"Overriding {override_type} arguments with the following {selected_type} args: {overriden_keys_str}"
                        )

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
        sharded_initializer = jax.jit(initializer,
                                      static_argnames=('shape', 'dtype'),
                                      out_shardings=sharding)
        key = rngs.params()
        param_data = sharded_initializer(key, shape, dtype)
        return nnx.Param(param_data, sharding=sharding)

    def create_kernel_param(self, *args, **kwargs) -> nnx.Param:
        """Creates a kernel/weight parameter using the kernel_initializer."""
        return self._create_param(self.kernel_initializer, *args, **kwargs)

    def create_scale_param(self, *args, **kwargs) -> nnx.Param:
        """Creates a scale/gain parameter using the scale_initializer."""
        return self._create_param(self.scale_initializer, *args, **kwargs)
