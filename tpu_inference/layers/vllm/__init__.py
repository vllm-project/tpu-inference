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
import torch

from tpu_inference.layers.vllm import backends as backends
from tpu_inference.layers.vllm import custom_ops as custom_ops
from tpu_inference.layers.vllm import ops as ops
from tpu_inference.layers.vllm import quantization as quantization


# NOTE: this empty function exists for an entry_points target for vllm plugin.
def register_layers():
    from torchtitan.experiments.tpu.afmv7 import AFMTextV7Wrapper
    from vllm.model_executor.models.registry import ModelRegistry

    if "AFMTextV7Wrapper" not in ModelRegistry.get_supported_archs():

        class VllmCompatibleAFMTextV7Wrapper(AFMTextV7Wrapper):

            def __init__(self, vllm_config, prefix=""):
                from torchtitan.experiments.tpu.afmv7 import AFMTextV7ModelArgs
                hf_config = vllm_config.model_config.hf_config

                model_args = AFMTextV7ModelArgs(
                    vocab_size=getattr(hf_config, "vocab_size", 153600),
                    hidden_dim=getattr(hf_config, "hidden_dim",
                                       getattr(hf_config, "hidden_size",
                                               2048)),
                    num_layers=getattr(
                        hf_config, "num_layers",
                        getattr(hf_config, "num_hidden_layers", 56)),
                    num_kv_reuse_layers=getattr(hf_config,
                                                "num_kv_reuse_layers", 21),
                    num_heads=getattr(
                        hf_config, "num_heads",
                        getattr(hf_config, "num_attention_heads", 16)),
                    num_kv_heads=getattr(
                        hf_config, "num_kv_heads",
                        getattr(hf_config, "num_key_value_heads", 2)),
                    hidden_dim_scale_factor=getattr(hf_config,
                                                    "hidden_dim_scale_factor",
                                                    3.25),
                    rope_theta=getattr(hf_config, "rope_theta", 500000.0),
                )
                super().__init__(model_args)

            def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
                return self.model.tok_embeddings(input_ids)

            def forward(self,
                        input_ids: torch.Tensor,
                        positions: torch.Tensor,
                        intermediate_tensors=None,
                        inputs_embeds=None) -> torch.Tensor:
                if input_ids.ndim == 1:
                    input_ids = input_ids.unsqueeze(0)
                output = self.model(input_ids, mode="SKIP_OUTPUT_LAYER")
                last_hidden_state = output.last_hidden_state
                # Flatten (B, S, H) -> (B*S, H) to match tpu-inference expectation
                return last_hidden_state.view(-1, last_hidden_state.size(-1))

            def compute_logits(self,
                               hidden_states: torch.Tensor) -> torch.Tensor:
                return self.model.output_transform(hidden_states)

        ModelRegistry.register_model("AFMTextV7Wrapper",
                                     VllmCompatibleAFMTextV7Wrapper)
