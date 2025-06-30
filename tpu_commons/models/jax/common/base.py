import dataclasses
from dataclasses import dataclass, fields
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from typing import Any, Callable

# Type alias for Initializer for cleaner type hints
Initializer = Callable[..., jax.Array]

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