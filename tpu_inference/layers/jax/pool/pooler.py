import enum
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from tpu_inference.layers.jax.pool.pooling_metadata import TPUSupportedPoolingMetadata

from vllm.config.pooler import PoolerConfig


# [padded_num_reqs, dim]
# or [padded_num_reqs, padded_max_num_batchec_token_per_req, dim] for allpool
PoolerOutput = jax.Array


class PoolingType(enum.Enum):
    LAST = "LAST"
    MEAN = "MEAN"
    CLS = "CLS"
    ALL = "ALL"


@dataclass(frozen=True)
class ResolvedPoolingConfig:
    task: str
    pooling_type: PoolingType
    normalize: bool

    @classmethod
    def from_config(
        cls,
        task: str,
        pooler_config: PoolerConfig | None,
    ) -> "ResolvedPoolingConfig":
        pooler_config = pooler_config or PoolerConfig()

        # The encode functionality is currently disabled because we cannot use DispatchPooler
        # as intended. (It was part of ModelForEmbedding, and in newer versions it was renamed to token_embed.)
        # This is because TPU does not support alternating requests between these two tasks, and it is
        # out of scope to change the vllm request handler/API server to separate these requests.
        # Therefore, this is disabled by default—users cannot use token_embed/encode functionality for now.

        if task == "embed":
            default_pooling_type = PoolingType.LAST
            default_normalize = True
        elif task == "encode":
            raise ValueError(f"Unsupported pooling task: {task}")
        else:
            raise ValueError(f"Unsupported pooling task: {task}")

        pooling_type_str = pooler_config.pooling_type or default_pooling_type.name
        pooling_type = PoolingType(pooling_type_str.upper())
        normalize = (
            pooler_config.normalize
            if pooler_config.normalize is not None
            else default_normalize
        )

        return cls(task=task, pooling_type=pooling_type, normalize=normalize)


class PoolingMethod(nnx.Module):
    @staticmethod
    def from_pooling_type(pooling_type: PoolingType) -> "PoolingMethod":
        if pooling_type is PoolingType.ALL:
            raise NotImplementedError("ALL pooling is not implemented yet.")
            # return AllPoolingMethod()
        if pooling_type is PoolingType.MEAN:
            return MeanPoolingMethod()
        if pooling_type is PoolingType.LAST:
            return LastPoolingMethod()
        if pooling_type is PoolingType.CLS:
            return CLSPoolingMethod()
        raise NotImplementedError(f"Unsupported pooling type: {pooling_type}")

    def __call__(
        self,
        hidden_states: jax.Array,
        pooling_metadata: TPUSupportedPoolingMetadata,
    ) -> jax.Array:
        raise NotImplementedError


class AllPoolingMethod(PoolingMethod):
    def __call__(
        self,
        hidden_states: jax.Array,
        pooling_metadata: TPUSupportedPoolingMetadata,
    ) -> jax.Array:
        pass


class MeanPoolingMethod(PoolingMethod):
    def __call__(
        self,
        hidden_states: jax.Array,
        pooling_metadata: TPUSupportedPoolingMetadata,
    ) -> jax.Array:
        padded_prompt_lens = pooling_metadata.prompt_lens
        padded_start_indices = pooling_metadata.first_token_indices
        padded_end_indices = pooling_metadata.last_token_indices
        cumsum = jnp.cumsum(hidden_states, dtype=jnp.float32)

        return (
            cumsum[padded_end_indices]
            - cumsum[padded_start_indices]
            + hidden_states[padded_start_indices]
        ) / padded_prompt_lens.unsqueeze(1)


class LastPoolingMethod(PoolingMethod):
    def __call__(
        self,
        hidden_states: jax.Array,
        pooling_metadata: TPUSupportedPoolingMetadata,
    ) -> jax.Array:
        return hidden_states[pooling_metadata.last_token_indices]


class CLSPoolingMethod(PoolingMethod):
    def __call__(
        self,
        hidden_states: jax.Array,
        pooling_metadata: TPUSupportedPoolingMetadata,
    ) -> jax.Array:
        return hidden_states[pooling_metadata.first_token_indices]


class PoolerHead(nnx.Module):
    def __call__(
        self,
        pooled: jax.Array,
        token_embeddings: jax.Array,
        token_mask: jax.Array,
        pooling_metadata: TPUSupportedPoolingMetadata,
    ) -> PoolerOutput:
        raise NotImplementedError


class EmbeddingPoolerHead(PoolerHead):
    def __init__(self, default_normalize: bool) -> None:
        super().__init__()
        self.default_normalize = default_normalize

    def __call__(
        self,
        pooled: jax.Array,
        pooling_metadata: TPUSupportedPoolingMetadata,
    ) -> PoolerOutput:

        # In the torch version, this part should handle other computations related to pooling_params, such as
        # normalization and truncating the embedding dimensions (for matryoshka models).
        # The problem with TPU is that we want a consistent output shape, and I feel like
        # the best we can do is to handle this outside JIT, on the CPU.
        # While you can actually do normalization with jnp.where, or maybe jax.lax.cond for branching in jax.jit,
        # for the sake of simplicity, we either normalize all requests or none of them based on pooling_config.
        # Pooler output: [padded_num_reqs, dim]


        if self.default_normalize:
            pooled = normalize(pooled)

        return pooled


class Pooler(nnx.Module):
    @staticmethod
    def for_encode(pooler_config: PoolerConfig | None) -> "Pooler":
        resolved = ResolvedPoolingConfig.from_config("encode", pooler_config)
        raise NotImplementedError("EncodePooler is currently disabled.")

    @staticmethod
    def for_embed(pooler_config: PoolerConfig | None) -> "Pooler":
        resolved = ResolvedPoolingConfig.from_config("embed", pooler_config)
        return EmbeddingPooler.from_config(resolved)

    def __call__(
        self,
        hidden_states: jax.Array,
        pooling_metadata: TPUSupportedPoolingMetadata,
    ) -> PoolerOutput:
        raise NotImplementedError

    def get_supported_tasks(self) -> set[str]:
        raise NotImplementedError


class EmbeddingPooler(Pooler):
    def __init__(
        self,
        pooling: PoolingMethod,
        head: EmbeddingPoolerHead,
    ) -> None:
        self.pooling = pooling 
        self.head = head 

    @classmethod
    def from_config(cls, config: ResolvedPoolingConfig) -> None:
        pooling = PoolingMethod.from_pooling_type(config.pooling_type)
        head = EmbeddingPoolerHead(config.normalize)
        return cls(pooling, head)

    def __call__(
        self,
        hidden_states: jax.Array,
        pooling_metadata: TPUSupportedPoolingMetadata,
    ) -> PoolerOutput:
        hidden_states = hidden_states.astype(jnp.float32)
        # the output mus be of type torch.tensor, but we cannot convert numpy to torch if the dtype is bf16
        pooled = self.pooling(hidden_states, pooling_metadata)
        return self.head(pooled, pooling_metadata)

    def get_supported_tasks(self) -> set[str]:
        return {"embed"}


def normalize(embeddings: jax.Array) -> jax.Array:
    norms = jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
    norms = jnp.maximum(norms, 1e-12)
    normalized = embeddings / norms
    return normalized
