import typing as tp

import torch
import jax
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh

from tpu_inference.layers.jax.pool.pooler import Pooler
from vllm.config import VllmConfig

_T = tp.TypeVar("_T", bound=type[nnx.Module])

_GENERATE_SUFFIXES = (
    "ForCausalLM",
    "ForConditionalGeneration",
)

class PoolingMixin:
    """
    same as VllmModelForPooling 
    """
    is_pooling_model: tp.ClassVar[tp.Literal[True]] = True

    default_pooling_type: tp.ClassVar[str] = "LAST"
    pooler: Pooler


def _get_pooling_model_name(orig_model_name: str, pooling_suffix: str) -> str:
    model_name = orig_model_name
    for suffix in _GENERATE_SUFFIXES:
        model_name = model_name.removesuffix(suffix)
    return model_name + pooling_suffix


def _create_pooling_model_cls(orig_cls: _T) -> _T:
    class ModelForPooling(orig_cls, PoolingMixin): 
        is_pooling_model = True

        def __init__(
            self,
            vllm_config: VllmConfig,
            rng_key: jax.Array,
            mesh: Mesh,
        ) -> None:
            super().__init__(
                vllm_config=vllm_config,
                rng_key=rng_key,
                mesh=mesh,
            )


            # Pooling models do not require language modeling heads.
            # However, there is a problem: since the pattern for loading weights in nnx
            # is abstract_module -> module, removing the lm_head attribute or leaves from the abstract_module
            # results in an error, I think.
            # This is because, during hf_load_weights, we need to match between the hf_key and nnx_key.

            # for attr in ("model.lm_head"):
            #     if hasattr(self, attr):
            #         delattr(self, attr)

            if getattr(self, "pooler", None) is None:
                self._init_pooler(vllm_config=vllm_config)

        def _init_pooler(self, vllm_config: VllmConfig) -> None: 
            raise NotImplementedError

    return ModelForPooling 


def as_embedding_model(cls: _T) -> _T:

    class ModelForEmbedding(_create_pooling_model_cls(cls)):
        def _init_pooler(self, vllm_config: VllmConfig) -> None:
            pooler_config = vllm_config.model_config.pooler_config
            if pooler_config is None:
                raise ValueError(
                    "Embedding models require `pooler_config` to be set in the model configuration."
                )

            self.pooler = Pooler.for_embed(pooler_config)

    ModelForEmbedding.__name__ = _get_pooling_model_name(
        cls.__name__,
        "ForEmbedding",
    )
    return ModelForEmbedding  # type: ignore[return-value]



def init_pooler_from_vllm_model(
        vllm_model: torch.nn.Module,
        vllm_config: VllmConfig,
        rng_key: PRNGKey, 
        mesh: Mesh,
):
    class DummyModule:
        def __init__(self, vllm_config, rng_key, mesh):
            pass

    for suffix in _GENERATE_SUFFIXES:
        if suffix in vllm_model.__class__.__name__:
            return None

    if "ForEmbedding" in vllm_model.__class__.__name__:
        EmbedModel = as_embedding_model(DummyModule)

        embed_model = EmbedModel(vllm_config=vllm_config, rng_key=rng_key, mesh=mesh,)
        embed_model._init_pooler(vllm_config)
        return embed_model.pooler 
    else:
        raise NotImplementedError(
            f"Pooling initialization for {vllm_model.__class__.__name__} is not implemented."
        )

