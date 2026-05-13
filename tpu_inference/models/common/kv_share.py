# Copyright 2026 Google LLC
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
"""Generic KV-cache sharing helpers driven by HF text_config attributes.

vLLM's PyTorch-side models declare KV-share via per-layer
`Attention(kv_sharing_target_layer_name=...)` — the runner discovers it
by scanning Attention modules. The JAX-side models have no equivalent
runtime declaration, so the runner derives the same mapping from the
HF text_config's `num_kv_shared_layers` + `layer_types` attributes.

Any JAX model whose HF config carries these attributes (currently
Gemma-4 E2B / E4B) can use this helper. New JAX models that introduce
KV-share via the same HF convention pick it up automatically.
"""


def compute_kv_share_map(text_config) -> dict:
    """Return `{shared_layer_idx: source_layer_idx}` for KV-shared layers.

    Source is the last preceding layer of the same attention type
    (sliding vs full). Mirrors the algorithm at
    `vllm/model_executor/models/gemma4.py:459-485`.

    Returns an empty dict when `num_kv_shared_layers == 0` (e.g. 26B/31B)
    or when `layer_types` is missing from the config.

    Raises:
      ValueError: if a layer's attention type has no preceding same-type
        layer to share K/V with — a configuration error that would
        silently mis-route the cache at the layer call site otherwise.
    """
    num_kv_shared_layers = getattr(text_config, "num_kv_shared_layers", 0)
    # Be defensive against mocked configs (e.g. MagicMock auto-creates
    # attributes as Mock objects — truthy but not int — which would
    # silently break the `> 0` check).
    if not isinstance(num_kv_shared_layers, int) or num_kv_shared_layers <= 0:
        return {}
    num_hidden_layers = text_config.num_hidden_layers
    raw_layer_types = getattr(text_config, "layer_types", None)
    if not isinstance(raw_layer_types, (list, tuple)) or not raw_layer_types:
        return {}
    layer_types = list(raw_layer_types)
    first_shared = num_hidden_layers - num_kv_shared_layers
    prev_types = layer_types[:first_shared]
    mapping: dict = {}
    for i in range(first_shared, num_hidden_layers):
        ctype = layer_types[i]
        if ctype not in prev_types:
            raise ValueError(
                f"KV-share: layer {i} of type {ctype!r} has no "
                f"preceding same-type layer in {prev_types!r}. "
                f"num_kv_shared_layers={num_kv_shared_layers}, "
                f"num_hidden_layers={num_hidden_layers}.")
        src = len(prev_types) - 1 - prev_types[::-1].index(ctype)
        mapping[i] = src
    return mapping
