import abc
from typing import List, Optional, Tuple

import jax
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.common.attention.attention import AttentionMetadata
from tpu_commons.models.jax.common.base import ParamFactory
from tpu_commons.models.jax.common.constants import KVCacheType

logger = init_logger(__name__)


class Model(nnx.Module, abc.ABC):

    def __init__(self, vllm_config: VllmConfig, rng: PRNGKey, mesh: Mesh):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, kv_caches: List[KVCacheType], input_ids: jax.Array,
                 attention_metadata: AttentionMetadata,
                 *args) -> Tuple[List[KVCacheType], jax.Array, jax.Array]:
        raise NotImplementedError

    @abc.abstractmethod
    def load_weights(self, rng: jax.Array, cache_dir: Optional[str] = None):
        raise NotImplementedError

    @classmethod
    def create_model_with_random_weights(cls, vllm_config: VllmConfig,
                                         rng: jax.Array, mesh: Mesh):
        """to create a model with random weights."""
        logger.info("Initializing model with random weights.")
        param_factory = ParamFactory(
            kernel_initializer=nnx.initializers.xavier_normal(),
            scale_initializer=nnx.initializers.ones,
            random_init=True)
        return cls(vllm_config, rng, mesh, param_factory)

    @classmethod
    def create_model_for_checkpoint_loading(cls, vllm_config: VllmConfig,
                                            rng: jax.Array, mesh: Mesh):
        """to create a model with abstract shapes for checkpoint loading."""
        logger.info("Initializing abstract model for checkpoint loading.")
        param_factory = ParamFactory(
            kernel_initializer=nnx.initializers.xavier_normal(),
            scale_initializer=nnx.initializers.ones,
            random_init=False)
        return cls(vllm_config, rng, mesh, param_factory)
