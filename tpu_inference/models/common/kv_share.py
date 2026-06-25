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
    (sliding vs full). Mirrors the algorithm used by vllm-pytorch's
    `Gemma4Attention.__init__` to derive `kv_sharing_target_layer_name`.

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
            raise ValueError(f"KV-share: layer {i} of type {ctype!r} has no "
                             f"preceding same-type layer in {prev_types!r}. "
                             f"num_kv_shared_layers={num_kv_shared_layers}, "
                             f"num_hidden_layers={num_hidden_layers}.")
        src = len(prev_types) - 1 - prev_types[::-1].index(ctype)
        mapping[i] = src
    return mapping


def compute_mtp_kv_share_map(draft_config, target_config) -> dict[str, str]:
    """Return `{'draft_layer.i': 'layer.j'}` mapping speculative MTP layers

    to their cross-model KV sharing target verifier layer cache array indices.
    """
    from collections import defaultdict
    draft_text_config = getattr(draft_config, "text_config", draft_config)
    target_text_config = getattr(target_config, "text_config", target_config)

    target_layer_types = getattr(target_text_config, "layer_types", [])
    target_num_kv_shared = getattr(target_text_config, "num_kv_shared_layers",
                                   0)
    num_non_shared = len(target_layer_types) - target_num_kv_shared

    # Group verifier non-shared layers by attention type
    type_to_target_indices = defaultdict(list)
    for idx, lt in enumerate(target_layer_types[:num_non_shared]):
        type_to_target_indices[lt].append(idx)

    draft_layer_types = getattr(draft_text_config, "layer_types", [])
    draft_num_layers = getattr(draft_text_config, "num_hidden_layers", 4)

    redirects = {}
    for draft_idx in range(draft_num_layers):
        draft_layer_type = (draft_layer_types[draft_idx] if draft_idx
                            < len(draft_layer_types) else "full_attention")
        candidates = type_to_target_indices.get(draft_layer_type, [])
        if candidates:
            target_idx = candidates[-1]
            redirects[f"draft_layer.{draft_idx}"] = f"layer.{target_idx}"
        else:
            raise ValueError(
                f"MTP cross-model KV-share: draft layer {draft_idx} of type "
                f"{draft_layer_type!r} has no matching same-type verifier layer "
                f"in target configuration (candidates are empty).")
    return redirects
