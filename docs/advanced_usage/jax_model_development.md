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
│   │    ├── __init__.py # Definition of JaxModule to provide pytorch-like APIs (e.g. named_parameters)
│   │    ├── embed.py
│   │    ├── linear.py
│   │    ├── norm.py
│   │    └── moe
│   │         ├── moe.py
│   └── common # Functionalities shared between torchax and jax implementations.
└── models
   ├── common
   │   └── model_loader.py
   └── jax  # Contains model files for each type of model family.
       ├── deepseek_v3.py
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

## Reusable building blocks

With [#1485](https://github.com/vllm-project/tpu-inference/issues/1485), we provided several reusable building blocks that's similar to those in vLLM.
The building blocks can be found under layers/jax/, e.g. JaxLinear / JaxEinsum / JaxMoe etc. With these building blocks, contributors can draft the model
with similar model architecture as vLLM implementation.

## Defining the model architecture
The model file is intended to contain all of the information needed for defining the Transformer-based architecture.
Each model file contains a model class with the following constructor interface:

```python
class NewModel(JaxModule, LoadableWithIterator):
  def __init__(self, vllm_config: VllmConfig, rng: nnx.Rngs, quant_config):
    ...
```

The constructor should set the architecture configuration (e.g. num_layers, hidden_size) and initialize the model layers. Layers can leverage tpu-inference to import or extend commonly used layer types (e.g. [JaxEmbed](https://github.com/vllm-project/tpu-inference/blob/c2b3ff50f9a2a99026e67de26f122c1a46b3e366/tpu_inference/layers/jax/embed.py#L25), [JaxRmsNorm](https://github.com/vllm-project/tpu-inference/blob/c2b3ff50f9a2a99026e67de26f122c1a46b3e366/tpu_inference/layers/jax/norm.py#L25), [JaxMoE](https://github.com/vllm-project/tpu-inference/blob/c2b3ff50f9a2a99026e67de26f122c1a46b3e366/tpu_inference/layers/jax/moe/moe.py#L129)). (Not recommended) The model can also be defined from scratch using flax NNX (e.g. [Llama3](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/models/jax/llama3.py)).

## Implementing the forward pass
The forward pass contains the logic for stitching together the layers that are defined in the model constructor and is expected to use the following interface:

```python
def __call__(
    self,
    kv_caches: List[jax.Array],
    input_ids: jax.Array,
    attention_metadata: AttentionMetadata,
    ...
) -> Tuple[List[jax.Array], jax.Array | JaxIntermediateTensors,
            List[jax.Array]]:
```

The key assumption of this interface is that context is managed outside of the model (the exception being that the model is responsible for updating the KV cache tensors after self-attention), which is the case in vLLM.
(See [vLLM's Block schedule and management design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager.html?h=kv+cache#implementation) and [tpu_jax_runner.py](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/runner/tpu_jax_runner.py#L556) for more details on how AttentionMetadata is prepared.)
The returned output is expected to contain the updated KV cache, final layer hidden states, and (optional) auxiliary final hidden state residuals (for speculative decoding).

In addition to the forward pass logic, each model needs to implement a method to generate the logits using the following interface:

```
def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
```

# Weight Loading

With [#1623](https://github.com/vllm-project/tpu-inference/issues/1623) and [#1571](https://github.com/vllm-project/tpu-inference/issues/1571), new models are recommended to reuse weight loading mechanisms/utilities from vLLM repo.

## Parameter level loading

tpu-inference provides [default per-parameter weight loader](https://github.com/vllm-project/tpu-inference/blob/c2b3ff50f9a2a99026e67de26f122c1a46b3e366/tpu_inference/models/jax/utils/weight_utils.py#L840) if not specified otherwise. This is sufficient for most cases. However it's possible to provide specific weight loader by setting "weight_loader" attribute for the paramter. Typical usages are:
- unpack the weight, e.g. unpack uint8 into fp4
- reshape the weight before loading

## Module level loading

If a module has "load_weights" method like [JaxMoE](https://github.com/vllm-project/tpu-inference/blob/c2b3ff50f9a2a99026e67de26f122c1a46b3e366/tpu_inference/layers/jax/moe/moe.py#L247), the weight loading mechanism will use it and skip per-parameter loader.

Typical usages are:
- Fuse the weight, e.g. in MoE
- Split the weight, e.g. in MLA

## Model level loading

By default a model should inherit [LoadableWithIterator](https://github.com/vllm-project/tpu-inference/blob/c2b3ff50f9a2a99026e67de26f122c1a46b3e366/tpu_inference/models/jax/utils/weight_utils.py#L866) which relies on `JaxAutoWeightsLoader` to load model. But a model definition can definitly override this method like [Deepseek](https://github.com/vllm-project/tpu-inference/blob/c2b3ff50f9a2a99026e67de26f122c1a46b3e366/tpu_inference/models/jax/deepseek_v3.py#L1309).

## Process weights after loading

Each [quant_method](https://github.com/vllm-project/tpu-inference/blob/c2b3ff50f9a2a99026e67de26f122c1a46b3e366/tpu_inference/layers/jax/quantization/__init__.py#L69) can process the weight after loading. This is usually used to re-quantize the weight to be TPU-friendly block size.

# Quantization Support
Many large LLMs like DeepSeek-V3 employ quantization to reduce hardware requirements and improve performance. The tpu-inference codebase can load pre-quantized model checkpoint, and utilizes [Qwix](https://github.com/google/qwix) to apply additional quantization settings to unquantized model weights. In tpu-inference, there are no assumptions on how a pre-quantized checkpoint is generated (so you are free to use your choice of popular tools), as long as the results are saved in HuggingFace Safetensor format and proper quantization configuration is provided on HuggingFace.
For more details on how to perform inference runs with Qwix on tpu-inference, please refer to the [general readme](https://github.com/vllm-project/tpu-inference/tree/31fa76a0187496ec161c634c98ac5eba144cb36c?tab=readme-ov-file#quantization).

**Please note** that you may need to update the list of supported quantization types on TPU [here](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/platforms/tpu_jax.py#L48). vLLM will trigger a validation error if the `quant_method` listed in the [HuggingFace quantization_config](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/config.json#L40) is not one of the supported types.

For the sake of demonstration, we will be referencing [deepseek_v3.py](https://github.com/vllm-project/tpu-inference/blob/c38e48fe2530d1c25143979d715e7578bcc242f3/tpu_inference/models/jax/deepseek_v3.py) for implementation details in this section.

## Loading Pre-quantized Checkpoints

Similar to vLLM, tpu-inference relies on "quantization_config" from `config.json` on HuggingFace to find out the proper [QuantizationConfig](https://github.com/vllm-project/tpu-inference/blob/c38e48fe2530d1c25143979d715e7578bcc242f3/tpu_inference/layers/jax/quantization/configs.py#L27), which is propagated to each module inside a model. Each module then utilize `QuantizationConfig` to update its topology (e.g. add parameters for scale) such that the parameters in the model can map to weights from HuggingFace.

After weights are loaded, we also through [process_weights_after_loading](https://github.com/vllm-project/tpu-inference/blob/c38e48fe2530d1c25143979d715e7578bcc242f3/tpu_inference/layers/jax/quantization/__init__.py#L69) re-quantize the weights for linear layers, to make quantization block layout more TPU-friendly.

All above mechanism are natively implemented in tpu-inference, as an user, you don't need to do anything but

```bash
MODEL_IMPL_TYPE=flax_nnx vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
```

## Loading Unquantized Checkpoints with Online Qwix Quantization

With an unquantized model checkpoint, we also provide an option to apply quantization while loading the checkpoint, through [Qwix](https://github.com/google/qwix).

An user needs to proivde qwix configuration describing how they want to online quantize the model

```bash
MODEL_IMPL_TYPE=flax_nnx vllm serve Qwen/Qwen3-4B-Instruct-2507 \
    --additional_config='{"quantization": { "qwix": { "rules": [{ "module_path": ".*", "weight_qtype": "float8_e4m3fn", "act_qtype": "float8_e4m3fn"}]}}}'
```

# Model Registration
Once a new model type is implemented, it must be added to the model registry in [model_loader.py](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/tpu_inference/models/common/model_loader.py#L29).

!!! warning
    per vLLM’s validation process, a model must be registered under a supported HuggingFace model name (see [here](https://github.com/vllm-project/vllm/blob/320feae6f506097c47b6b41a634a6197512cffc1/vllm/model_executor/models/registry.py#L428) for more detail).

To plug in external Jax NNX modeling implementations into tpu-inference, please refer to the [dedicated documentation](https://github.com/vllm-project/tpu-inference/blob/31fa76a0187496ec161c634c98ac5eba144cb36c/docs/getting_started/out-of-tree.md).
