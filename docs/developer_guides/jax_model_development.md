# JAX Model Development Guide

tpu-inference provides a flexible framework for implementing Transformer-based architectures in Flax NNX.

The ingredients for integrating a new model type consist of:
- defining the model architecture and implementing any custom layers
- implementing weight loading logic
- (optional) adding quantization support
- registering the new model into tpu-inference

# Code Organization
It is helpful to familiarize with the code organization before beginning model development:

```bash
tpu_inference
├── layers
│   ├── jax # Provide pre-implemented building blocks for tpu-inference models.
│   │    ├── attention_interface.py # Core interfaces used for applying attention.
│   │    ├── base.py
│   │    ├── layers.py
│   │    ├── transformer_block.py
│   │    ├── sharding.py
│   │    ├── rope.py
│   │    ├── glossary.md
│   │    ├── attention
│   │    │    ├── attention.py # Pre-implemented attention layer.
│   │    └── moe
│   │         ├── moe.py
│   └── common # Functionalities shared between torchax and jax implementations.
└── models
   ├── common
   │   └── model_loader.py
   └── jax  # Contains model files for each type of model family.
       ├── deepseek_v3.py
       ├── llama3.py
       ├── qwen3.py
       └── utils
```

- Registration of new Jax model types should be performed in [`tpu_inference/models/common/model_loader.py`](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/models/common/model_loader.py)
- New Jax model definitions should be added to [`tpu_inference/models/jax`](https://github.com/vllm-project/tpu-inference/tree/main/tpu_inference/models/jax).
- Commonly used layers (e.g. embedding, feed-forward) can be imported from [`tpu_inference/layers/jax`](https://github.com/vllm-project/tpu-inference/tree/main/tpu_inference/layers/jax).
- Model-specific layer implementations should be added directly to the modeling file itself (e.g. for DeepSeek-V3: [`tpu_inference/models/jax/deepseek_v3.py`](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/models/jax/deepseek_v3.py)).
- Custom (Qwix) quantization configs (yaml files) should be stored in [`tpu_inference/models/jax/utils/quantization/configs`](https://github.com/vllm-project/tpu-inference/tree/main/tpu_inference/models/jax/utils/quantization/configs).

# Model Implementation
Implementing a new model requires creating a dedicated model file (e.g. [deepseek_v3.py](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/models/jax/deepseek_v3.py)) that contains the following components:
- The model class, which defines the architecture.
- Forward pass implementation and logit computation.
- Weight loading logic to import HuggingFace weights into the model definition.

## Defining the model architecture
The model file is intended to contain all of the information needed for defining the Transformer-based architecture.
Each model file contains a model class with the following constructor interface:

```python
class NewModel(nnx.Module):
  def __init__(self, vllm_config: VllmConfig, rng: nnx.Rngs,
               mesh: jax.sharding.Mesh)
```

"# TODO (jacobplatin): we need to update this once the JAX-path refactor is fully done"

The constructor should set the architecture configuration (e.g. num_layers, hidden_size) and initialize the model layers. Layers can be defined from scratch using flax NNX (e.g. [Llama3](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/models/jax/llama3.py)) or can leverage tpu-inference to import or extend commonly used layer types (e.g. [Embedder](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/layers/jax/layers.py#L168), [RMSNorm](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/layers/jax/layers.py#L49), [MoE](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/layers/jax/moe/moe.py#L69), [Attention](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/layers/jax/attention/attention.py#L23), [DenseFFW](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/layers/jax/layers.py#L98C7-L98C15), [TransformerBlock](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/layers/jax/transformer_block.py#L15
)).

## Implementing the forward pass
The forward pass contains the logic for stitching together the layers that are defined in the model constructor and is expected to use the following interface:

```python
def __call__(
   self,
   kv_caches: List[jax.Array],
   input_ids: jax.Array,
   attention_metadata: AttentionMetadata,
   *args,
) -> Tuple[List[KVCacheType], jax.Array, List[jax.Array]]
```

The key assumption of this interface is that context is managed outside of the model (the exception being that the model is responsible for updating the KV cache tensors after self-attention), which is the case in vLLM.
(See [vLLM's Block schedule and management design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager.html?h=kv+cache#implementation) and [tpu_jax_runner.py](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/runner/tpu_jax_runner.py#L556) for more details on how AttentionMetadata is prepared.)
The returned output is expected to contain the updated KV cache, final layer hidden states, and (optional) auxiliary final hidden state residuals (for speculative decoding).

In addition to the forward pass logic, each model needs to implement a method to generate the logits using the following interface:
`def compute_logits(self, hidden_states: jax.Array) -> jax.Array:`

# Weight Loading
Open source model weights are not universally standardized in their naming nor in their parameter shapes. Thus, it is necessary to implement per-model weight loading logic to correctly import open source weights into their corresponding model parameters.
To do this, each model must implement a `load_weights` method with the following interface: `def load_weights(self, rng: PRNGKey)`

Weight loading logic is typically composed of several categories of steps:
- Loading HuggingFace weights into an iterator (see [model_weights_generator](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/models/jax/utils/weight_utils.py#L73))
- Defining a mapping between loaded weight names and implementation
 weight names.
- Defining a mapping of tensor transformations to apply on the loaded parameters. (these transformations can include transposing or reshaping loaded tensors).
- Performing model-specific loading logic (e.g. splitting a loaded weight tensor and loading into multiple parameters).
- (optional) Support for loading pre-quantized models.

Please refer to [deepseek_v3.py](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/models/jax/deepseek_v3.py#L354) or [llama4.py](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/models/jax/llama4.py#L286) for some examples on how to implement weight loading.

# Quantization Support
Many large LLMs like DeepSeek-V3 employ quantization to reduce hardware requirements and improve performance. The tpu-inference codebase utilizes [Qwix](https://github.com/google/qwix) to load pre-quantized models and/or apply additional quantization settings to loaded model weights. In tpu-inference, there are no assumptions on how a pre-quantized checkpoint is generated (so you are free to use your choice of popular tools), as long as the results are saved in HuggingFace Safetensor format and the guidelines below are followed.
For more details on how to perform inference runs with Qwix on tpu-inference, please refer to the [general readme](https://github.com/vllm-project/tpu-inference/tree/31fa76a0187496ec161c634c98ac5eba144cb36c?tab=readme-ov-file#quantization).

**Please note** that you may need to update the list of supported quantization types on TPU [here](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/platforms/tpu_jax.py#L48). vLLM will trigger a validation error if the `quant_method` listed in the [HuggingFace quantization_config](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/config.json#L40) is not one of the supported types.

For the sake of demonstration, we will be referencing [deepseek_v3.py](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/models/jax/deepseek_v3.py) for implementation details in this section.

## Loading Pre-quantized Checkpoints and Applying Quantization Rules
To correctly load a pre-quantized checkpoint, the following steps need to be run:
- Define the quantization settings using a Qwix config, which can be exposed as a yaml file (e.g. [int8_default.yaml](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/models/jax/utils/quantization/configs/int8_default.yaml)) or [set within the code](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/models/jax/utils/quantization/quantization_utils.py#L37). Open source models' quantization settings are typically published in the respective HuggingFace quantization_config (e.g. [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/config.json#L37)).
(For more information on the supported Qwix quantization options, please refer to the [Qwix documentation](https://github.com/google/qwix?tab=readme-ov-file#quantization-config)).
- Set `use_abstract_model: True` in your Qwix config so that your NNX model graph is quantized before the weights are loaded in.
- If the pre-quantized model contains dequantization scales, update the weight loading logic to [store them](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/models/jax/deepseek_v3.py#L693) as well. If loading the model’s weights required [applying transformations](#weight-loading), ensure that [dequantization scales are also transformed accordingly](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/models/jax/deepseek_v3.py#L602). The scale dimensions can be determined by the `weight_block_size` in the [HuggingFace config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json#L41) and [set](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/models/jax/deepseek_v3.py#L484) in the weight loading logic. The scale dimensions can also be cross-referenced against the [Safetensor files](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/model-00001-of-000163.safetensors).

Conversely, if the checkpoint is not pre-quantized then no custom model loading code is needed and one should set `use_abstract_model: False` in the Qwix config.

**Please be aware** that the Qwix quantization settings are the source of truth and will override the data types used for the loaded weights (even if pre-quantized weights were provided).

# Model Registration
Once a new model type is implemented, it must be added to the model registry in [model_loader.py](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/models/common/model_loader.py#L29).

!!! warning
    per vLLM’s validation process, a model must be registered under a supported HuggingFace model name (see [here](https://github.com/vllm-project/vllm/blob/320feae6f506097c47b6b41a634a6197512cffc1/vllm/model_executor/models/registry.py#L428) for more detail).

To plug in external Jax NNX modeling implementations into tpu-inference, please refer to the [dedicated documentation](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/docs/getting_started/out-of-tree.md).
