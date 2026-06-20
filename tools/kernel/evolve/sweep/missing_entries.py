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
"""Walk tuned_block_sizes.py to find shape entries referenced by popular
models but missing tuned values.

The lookup logic falls back to a TPU-version default when an entry is
missing, leaving real perf on the table. This module catalogs popular
``(model, max_model_len)`` pairs, derives the lookup key for each, and
reports which ones currently miss the table.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass
class ModelSpec:
    """Minimal subset of HF config needed to derive a tuning lookup key."""
    name: str
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    q_dtype: str = "bfloat16"
    kv_dtype: str = "bfloat16"
    sliding_window: int | None = None


# Curated list of production-relevant models. Add freely.
POPULAR_MODELS: list[ModelSpec] = [
    ModelSpec("Qwen3-0.6B", num_q_heads=16, num_kv_heads=8, head_dim=128),
    ModelSpec("Qwen3-1.7B", num_q_heads=16, num_kv_heads=8, head_dim=128),
    ModelSpec("Qwen3-4B", num_q_heads=32, num_kv_heads=8, head_dim=128),
    ModelSpec("Qwen3-8B", num_q_heads=32, num_kv_heads=8, head_dim=128),
    ModelSpec("Qwen3-14B", num_q_heads=40, num_kv_heads=8, head_dim=128),
    ModelSpec("Qwen3-32B", num_q_heads=64, num_kv_heads=8, head_dim=128),
    ModelSpec("Llama-3.1-8B", num_q_heads=32, num_kv_heads=8, head_dim=128),
    ModelSpec("Llama-3.2-1B", num_q_heads=32, num_kv_heads=8, head_dim=64),
    ModelSpec("Llama-3.2-3B", num_q_heads=24, num_kv_heads=8, head_dim=128),
    ModelSpec("Gemma-2B", num_q_heads=8, num_kv_heads=1, head_dim=256),
]

POPULAR_CONTEXT_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]


@dataclasses.dataclass
class MissingEntry:
    """One unique missing key in the tuning table."""
    device: str  # e.g. 'TPU v7'
    page_size: int
    dtype_key: str  # e.g. 'q_bfloat16_kv_bfloat16'
    head_key: str  # e.g. 'q_head-16_kv_head-8_head-128'
    extra_key: str  # e.g. 'max_model_len-1024-sw-None'
    referencing_models: list[str] = dataclasses.field(default_factory=list)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _head_key(spec: ModelSpec) -> str:
    return (f"q_head-{spec.num_q_heads}"
            f"_kv_head-{spec.num_kv_heads}"
            f"_head-{spec.head_dim}")


def _dtype_key(spec: ModelSpec) -> str:
    return f"q_{spec.q_dtype}_kv_{spec.kv_dtype}"


def _extra_key(max_model_len: int, sliding_window: int | None) -> str:
    return f"max_model_len-{max_model_len}-sw-{sliding_window}"


def find_missing_entries(
        *,
        tuned_block_sizes_path: Path,
        models: list[ModelSpec] | None = None,
        context_lengths: list[int] | None = None,
        devices: list[str] = ("TPU v7", ),
        page_sizes: list[int] = (128, ),
) -> list[MissingEntry]:
    """Import the TUNED_BLOCK_SIZES table and find which keys are missing.

    Returns one ``MissingEntry`` per unique missing key; ``referencing_models``
    lists the popular models that would currently miss it.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_evolve_loaded_tuned",
                                                  tuned_block_sizes_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"could not load module from {tuned_block_sizes_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    table = mod.TUNED_BLOCK_SIZES

    models = models or POPULAR_MODELS
    context_lengths = context_lengths or POPULAR_CONTEXT_LENGTHS

    missing_by_key: dict[tuple, MissingEntry] = {}
    for dev in devices:
        for ps in page_sizes:
            for m in models:
                dt = _dtype_key(m)
                hd = _head_key(m)
                for ml in context_lengths:
                    ek = _extra_key(ml, m.sliding_window)
                    try:
                        _ = table[dev][ps][dt][hd][ek]
                        continue  # entry present
                    except KeyError:
                        pass
                    key = (dev, ps, dt, hd, ek)
                    if key not in missing_by_key:
                        missing_by_key[key] = MissingEntry(device=dev,
                                                           page_size=ps,
                                                           dtype_key=dt,
                                                           head_key=hd,
                                                           extra_key=ek)
                    missing_by_key[key].referencing_models.append(m.name)
    return list(missing_by_key.values())
