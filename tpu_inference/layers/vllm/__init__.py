# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.registry import TokenizerRegistry

from tpu_inference.layers.vllm import backends as backends
from tpu_inference.layers.vllm import custom_ops as custom_ops
from tpu_inference.layers.vllm import ops as ops
from tpu_inference.layers.vllm import quantization as quantization
from tpu_inference.logger import init_logger

# keras_hub is an optional dependency: when it is installed, the plugin
# hook below registers its serving pieces; when it is not, tpu-inference
# works exactly as before.
try:
    from keras_hub.src.vllm.hf_config import (KERAS_HUB_ARCHITECTURE,
                                              register_hf_config)
    from keras_hub.src.vllm.keras_hub_for_causal_lm import \
        KerasHubForCausalLM
except ImportError:
    KerasHubForCausalLM = None

logger = init_logger(__name__)


# NOTE: this function is the entry_points target for the vllm general plugin.
def register_layers():
    # Override vLLM's built-in model classes with TPU-specific implementations
    # (e.g. DeepseekV4ForCausalLM). Done here because this is the plugin hook
    # that vLLM invokes at startup, before any model is loaded.
    from tpu_inference.models.vllm.experimental import register_models
    register_models()
    _register_keras_hub()


def _register_keras_hub():
    """Registers everything needed to serve KerasHub presets, at plugin load.

    Four registrations, all through the standard mechanisms:

    - ``register_hf_config()`` teaches transformers the ``keras_hub``
      model_type, so ``AutoConfig`` can read the config KerasHub writes.
    - ``register_model`` puts keras-hub's ``KerasHubForCausalLM`` class in
      the model registry (and vLLM's), so the loader resolves it by
      architecture name exactly like any other model.
    - ``TokenizerRegistry`` maps ``tokenizer_mode="keras_hub"`` to
      ``KerasHubTokenizer``, so vLLM tokenizes with the preset's own
      KerasHub tokenizer.
    - ``RENDERER_REGISTRY`` maps the same mode to vLLM's standard HF prompt
      renderer, which drives any registered tokenizer.

    The preset to serve rides in the config's ``keras_hub_preset`` field,
    which the model class and tokenizer read. The model class itself lives
    in keras-hub (`keras_hub.src.vllm.keras_hub_for_causal_lm`); this hook
    only registers it.
    """
    if KerasHubForCausalLM is None:
        logger.debug(
            "keras_hub is not installed; skipping KerasHub registration.")
        return
    # model_loader must not be imported at module scope (see the
    # inline-import NOTE in `_get_model_architecture`), so it resolves at
    # registration time.
    from tpu_inference.models.common.model_loader import register_model

    # transformers side: resolve `model_type: keras_hub` in every worker.
    register_hf_config()
    register_model(KERAS_HUB_ARCHITECTURE, KerasHubForCausalLM)
    # Tokenizer side: serve the preset's own KerasHub tokenizer
    # (tokenizer_mode="keras_hub"); registered by module path, resolved
    # lazily by whichever process loads the tokenizer. vLLM keys its prompt
    # renderer by the same mode; the standard HF renderer drives any
    # TokenizerLike, so map "keras_hub" to it.
    TokenizerRegistry.register("keras_hub", "keras_hub.src.vllm.tokenizer",
                               "KerasHubTokenizer")
    RENDERER_REGISTRY.register("keras_hub", "vllm.renderers.hf", "HfRenderer")
