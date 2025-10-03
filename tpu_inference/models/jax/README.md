# JAX Model Development Guide
tpu_commons provides a flexible framework for implementing Transformer-based architectures in Flax NNX. This readme will outline the requirements and guidelines for model development.


The ingredients for integrating a new model consist of the following:
 - model registration
 - model definition and custom layer implementation


# Code Organization
It is helpful to familiarize with the code organization before developing new models:
 - registration of new Jax model types should be performed in tpu_commons/tpu_commons/models/jax/model_loader.py
 - new Jax model definitions should be added to tpu_commons/tpu_commons/models/jax.
 - commonly used layers (e.g. embedding, feed-forward) can be imported from tpu_commons/tpu_commons/models/jax/common/.
 - model-specific layer implementations should be added to tpu_commons/tpu_commons/models/jax/common/<layer_type>/ (e.g. attention/, moe/).
 - Custom (Qwix) quantization config (yaml) files should be stored in tpu_commons/tpu_commons/models/jax/utils/quantization/configs


# Model Implementation
Implementing a new model requires creating a dedicated model file (e.g. [deepseek_v3.py](https://github.com/vllm-project/tpu_commons/blob/main/tpu_commons/models/jax/deepseek_v3.py)) that contains the following components:
 - The model class, which defines the architecture.
 - Forward pass implementation and logit computation.
 - Weight loading logic to import HuggingFace weights into the model definition.


## Defining the model architecture
The model file is intended to contain all of the information needed for defining the Transformer-based architecture.
The expected interface for the constructor is as follow:
class NewModel(nnx.Module):
   def __init__(self, vllm_config: VllmConfig, rng: nnx.Rngs,
                mesh: jax.sharding.Mesh)


The constructor should set the architecture configuration (e.g. num_layers, hidden_size) and initialize the model layers. Layers can be defined from scratch using flax NNX (e.g. [Llama3](TODO)) or can leverage tpu_commons to import or extend commonly used layer types (e.g. [Embedder](https://github.com/vllm-project/tpu_commons/blob/main/tpu_commons/models/jax/common/layers.py#L168), [RMSNorm](https://github.com/vllm-project/tpu_commons/blob/main/tpu_commons/models/jax/common/layers.py#L49), [MoE](https://github.com/vllm-project/tpu_commons/blob/main/tpu_commons/models/jax/common/moe/moe.py#L69), [Attention](TODO), [DenseFFW](https://github.com/vllm-project/tpu_commons/blob/main/tpu_commons/models/jax/common/layers.py#L98C7-L98C15), [TransformerBlock](TODO)).


## Implementing the forward pass
The forward pass contains the logic for stitching together the layers which are defined in the model constructor and is expected to use the following interface:


def __call__(
    self,
    kv_caches: List[jax.Array],
    input_ids: jax.Array,
    attention_metadata: AttentionMetadata,
    *args,
) -> Tuple[List[KVCacheType], jax.Array, List[jax.Array]]


The key assumption of this interface is that context is managed outside of the model (the exception being that the model is responsible for updating the KV cache tensors after self-attention), which is the case for vLLM (e.g. see [Block schedule and management](TODO) and [AttentionMetadata definition](TODO) for more details).
The returned output is expected to contain the updated KV cache, final model hidden states, and optional auxiliary hidden states (for speculative decoding)(TODO: Lihao).


In addition to the forward pass logic, each model needs to implement a method to generate the logits using the following interface:
def compute_logits(self, hidden_states: jax.Array) -> jax.Array:


# Weight Loading
Open source model weights are not universally standardized in their naming nor in their parameter shapes. Thus, it is necessary to implement per-model weight loading logic to correctly import open source weights to their corresponding model parameters and in the correct shape.
To do this, each model must implement a `load_weights` method with the following interface: `def load_weights(self, rng: PRNGKey)`


(TODO Jacob)
Weight loading logic is typically composed of several categories of steps:
 - Loading HuggingFace weights into an iterator (see [model_weights_generator](TODO))
 - Defining a mapping between loaded weight names and model weight names.
 - Defining a mapping of tensor transformations to apply on the loaded parameters. These transfprmations can include transposing or reshaping loaded tensors.
 - Performing model-specific logic (e.g. splitting a loaded weight tensor and loading into multiple parameters).


Please refer to [deepseek_v3.py](https://github.com/vllm-project/tpu_commons/blob/main/tpu_commons/models/jax/deepseek_v3.py#L355) or [llama4.py](TODO) for some examples on how to implement weight loading.


## Quantization Support (TODO: jacobplatin)
Many large LLMs like DeepSeek-V3 employ quantization to reduce hardware requirements and improve perf. tpu_commons supports quantized models via the Qwix library. 
In order to load a quantized model checkpoint, one needs to:
 - Set the quantization configuration using a Qwix config (see [Qwix documentation](TODO) and [example yaml](TODO) for reference) or natively in the model [weight loading code](https://github.com/vllm-project/tpu_commons/blob/main/tpu_commons/models/jax/deepseek_v3.py#L459).
 - update the weight loading logic to load quantized weights and scalars into Qwix-formatted tenors.
Please refer to [deepseek_v3.py](https://github.com/vllm-project/tpu_commons/blob/main/tpu_commons/models/jax/deepseek_v3.py#L459) for an example of how quantization is supported by default.


# Model Registration
Once a new model type is implemented, it must be added to the model registry in [model_loader.py](https://github.com/vllm-project/tpu_commons/blob/main/tpu_commons/models/jax/model_loader.py#L27).
WARNING: per vLLM’s validation process a model must be registered under a supported HuggingFace model name (see [here](TODO) for more detail).


# Guidelines for contributing (TODO)
- Style (naming convention)
- CI/CD test cases
- accuracy and perf monitoring through bm-infra