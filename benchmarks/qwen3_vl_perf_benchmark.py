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

"""
Performance Benchmark for Qwen3VL: JAX Native (flax_nnx) vs vLLM (TorchAX)

This script provides comprehensive performance comparison between:
1. Native JAX/Flax NNX implementation
2. vLLM/TorchAX wrapper implementation

Usage:
    python benchmarks/qwen3_vl_perf_benchmark.py \
        --model "Qwen/Qwen3-VL-4B" \
        --seq-lengths 128,256,512 \
        --num-warmup 3 \
        --num-iterations 10 \
        --profile-dir /tmp/qwen3vl_profile

Environment Variables:
    MODEL_IMPL_TYPE: Set to "flax_nnx" or "vllm" for specific implementation
    NEW_MODEL_DESIGN: Set to "True" for new model design features
"""

import argparse
import dataclasses
import gc
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Ensure JAX is configured before imports
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from tpu_inference import envs
from tpu_inference import utils as common_utils
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  MESH_AXIS_NAMES_2D)
from tpu_inference.models.common.model_loader import get_model
from tpu_inference.utils import make_optimized_mesh


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    model_path: str = "Qwen/Qwen3-VL-4B"
    seq_lengths: List[int] = field(default_factory=lambda: [128, 256, 512])
    batch_sizes: List[int] = field(default_factory=lambda: [1])
    num_warmup: int = 3
    num_iterations: int = 10
    profile_dir: Optional[str] = None
    output_file: Optional[str] = None
    use_dummy_weights: bool = True
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    test_vision: bool = True
    vision_grid_thw: Tuple[int, int, int] = (1, 4, 4)
    max_model_len: Optional[int] = None


@dataclass
class TimingResult:
    """Result of a timing measurement."""
    name: str
    impl_type: str
    seq_length: int
    batch_size: int
    compile_time_ms: float
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_tokens_per_sec: float
    memory_used_gb: float
    all_times_ms: List[float] = field(default_factory=list)


@dataclass
class BenchmarkResults:
    """Collection of benchmark results."""
    config: BenchmarkConfig
    results: List[TimingResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "config": dataclasses.asdict(self.config),
            "results": [dataclasses.asdict(r) for r in self.results],
            "metadata": self.metadata,
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


@dataclass
class ModelBundle:
    impl_type: str
    vllm_config: Any
    model_fn: Callable
    compute_logits_fn: Callable
    multimodal_fns: Optional[Dict[str, Callable]]
    state: Any
    model: Any
    text_config: Any
    layer_names: List[str]
    layer_name_to_kvcache_index: Tuple[Tuple[str, int], ...]


class Timer:
    """Context manager for timing operations."""

    def __init__(self):
        self.elapsed_ms: float = 0.0
        self.start: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000


def get_memory_usage_gb() -> float:
    """Get current HBM memory usage in GB."""
    try:
        devices = jax.local_devices()
        if devices:
            stats = devices[0].memory_stats()
            if stats:
                return stats.get("bytes_in_use", 0) / (1024**3)
    except Exception:
        pass
    return 0.0


def _block_until_ready(value: Any) -> Any:
    def _block(x):
        return x.block_until_ready() if hasattr(x, "block_until_ready") else x

    return jax.tree_util.tree_map(_block, value)


def _select_safe_token_id(vocab_size: int, disallowed: List[int]) -> int:
    disallowed_set = {token for token in disallowed if 0 <= token < vocab_size}
    for candidate in range(min(vocab_size, 8)):
        if candidate not in disallowed_set:
            return candidate
    return 0


def _sanitize_input_ids(input_ids: jax.Array, vocab_size: int,
                        disallowed: List[int]) -> jax.Array:
    if not disallowed:
        return input_ids
    safe_id = _select_safe_token_id(vocab_size, disallowed)
    sanitized = input_ids
    for token_id in disallowed:
        if 0 <= token_id < vocab_size:
            sanitized = jnp.where(sanitized == token_id, safe_id, sanitized)
    return sanitized


def _build_model_config(model_path: str, dtype: str,
                        max_model_len: int) -> Any:
    from vllm.config import ModelConfig

    try:
        return ModelConfig(
            model=model_path,
            tokenizer=model_path,
            tokenizer_mode="auto",
            trust_remote_code=True,
            seed=42,
            dtype=dtype,
            max_model_len=max_model_len,
        )
    except TypeError:
        model_config = ModelConfig(model_path)
        if hasattr(model_config, "tokenizer"):
            model_config.tokenizer = model_path
        if hasattr(model_config, "tokenizer_mode"):
            model_config.tokenizer_mode = "auto"
        if hasattr(model_config, "trust_remote_code"):
            model_config.trust_remote_code = True
        if hasattr(model_config, "seed"):
            model_config.seed = 42
        if hasattr(model_config, "dtype"):
            model_config.dtype = dtype
        if hasattr(model_config, "max_model_len"):
            model_config.max_model_len = max_model_len
        return model_config


def create_vllm_config(
    model_path: str,
    dtype: str,
    use_dummy_weights: bool,
    tensor_parallel_size: int,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
) -> Any:
    """Create a VllmConfig for benchmarking."""
    try:
        from vllm.config import (
            CacheConfig,
            DeviceConfig,
            MultiModalConfig,
            ParallelConfig,
            SchedulerConfig,
            VllmConfig,
        )
    except ImportError as e:
        print(f"Warning: vLLM is required for benchmarking: {e}")
        return None

    try:
        from vllm.config import LoadConfig
    except ImportError:
        from vllm.config.load import LoadConfig

    model_config = _build_model_config(model_path, dtype, max_model_len)
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        try:
            from transformers import AutoConfig
        except ImportError as e:
            raise RuntimeError(
                "Transformers is required to load the HF config.") from e
        hf_config = AutoConfig.from_pretrained(model_path,
                                               trust_remote_code=True)
        model_config.hf_config = hf_config

    if hasattr(model_config, "multimodal_config"):
        if (getattr(model_config, "multimodal_config", None) is None
                and hasattr(hf_config, "image_token_id")):
            model_config.multimodal_config = MultiModalConfig(
                image_input_type="pixel",
                image_token_id=hf_config.image_token_id,
                image_input_shape=None,
            )

    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        swap_space=4,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=tensor_parallel_size,
        worker_use_ray=False,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        is_encoder_decoder=False,
    )
    device_config = DeviceConfig(device="tpu")

    load_format = "dummy" if use_dummy_weights else "auto"
    try:
        load_config = LoadConfig(load_format=load_format)
    except TypeError:
        load_config = LoadConfig()
        if hasattr(load_config, "load_format"):
            load_config.load_format = load_format

    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        load_config=load_config,
        speculative_config=None,
        lora_config=None,
        observability_config={},
        additional_config={"enable_dynamic_image_sizes": False},
    )


def _is_multimodal_model(model_config: Any) -> bool:
    attr = getattr(model_config, "is_multimodal_model", False)
    return bool(attr() if callable(attr) else attr)


def _uses_mrope(model_config: Any, hf_config: Any) -> bool:
    attr = getattr(model_config, "uses_mrope", None)
    if attr is not None:
        return bool(attr() if callable(attr) else attr)
    rope_scaling = getattr(hf_config, "rope_scaling", None)
    return isinstance(rope_scaling, dict) and "mrope_section" in rope_scaling


def _get_text_config(hf_config: Any) -> Any:
    return getattr(hf_config, "text_config", hf_config)


def _collect_vllm_layer_names(vllm_wrapper: Any) -> List[str]:
    vllm_model = getattr(getattr(vllm_wrapper, "model", None), "vllm_model",
                         None)
    if vllm_model is None:
        return []
    names = []
    for module in vllm_model.modules():
        layer_name = getattr(module, "layer_name", None)
        if layer_name:
            names.append(layer_name)
    # Deduplicate while preserving order.
    seen = set()
    ordered = []
    for name in names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def create_mesh(tp_size: int, devices: Optional[List] = None) -> Mesh:
    """Create a JAX mesh for sharding."""
    if devices is None:
        devices = jax.devices()
    num_devices = len(devices)
    if num_devices % tp_size != 0:
        raise ValueError(
            f"Tensor parallel size {tp_size} does not divide {num_devices} devices."
        )
    dp_size = max(1, num_devices // tp_size)
    if envs.NEW_MODEL_DESIGN:
        axis_names = MESH_AXIS_NAMES
        axis_shapes = (dp_size, 1, 1, tp_size)
    else:
        axis_names = MESH_AXIS_NAMES_2D
        axis_shapes = (dp_size, tp_size)
    return make_optimized_mesh(axis_shapes, axis_names, devices=devices)


def build_attention_metadata(
    seq_len: int,
    batch_size: int,
    block_size: int,
    use_mrope: bool,
    dp_size: int,
) -> Tuple[jax.Array, AttentionMetadata, jax.Array, int]:
    """Create attention metadata and positions for benchmarking."""
    if dp_size < 1:
        dp_size = 1
    if batch_size % dp_size != 0:
        raise ValueError(
            f"Batch size {batch_size} must be divisible by data parallel size {dp_size}."
        )
    num_reqs_per_dp = batch_size // dp_size
    max_num_blocks_per_req = (seq_len + block_size - 1) // block_size
    tokens_per_dp = seq_len * num_reqs_per_dp

    positions_per_req = jnp.arange(seq_len, dtype=jnp.int32)
    positions_per_dp = jnp.tile(positions_per_req, num_reqs_per_dp)
    if use_mrope:
        positions_per_dp = jnp.broadcast_to(positions_per_dp[None, :],
                                            (3, tokens_per_dp))
        positions = jnp.concatenate([positions_per_dp] * dp_size, axis=1)
    else:
        positions = jnp.concatenate([positions_per_dp] * dp_size, axis=0)

    block_ids = np.arange(num_reqs_per_dp * max_num_blocks_per_req,
                          dtype=np.int32).reshape(num_reqs_per_dp,
                                                  max_num_blocks_per_req)
    block_tables = np.concatenate([block_ids.reshape(-1)] * dp_size, axis=0)
    block_tables = jnp.array(block_tables, dtype=jnp.int32)
    seq_lens = jnp.array([seq_len] * num_reqs_per_dp * dp_size,
                         dtype=jnp.int32)
    query_start_loc_per_dp = jnp.array(
        [seq_len * i for i in range(num_reqs_per_dp + 1)], dtype=jnp.int32)
    query_start_loc = jnp.concatenate(
        [query_start_loc_per_dp] * dp_size, axis=0)
    request_distribution = jnp.array([0, 0, num_reqs_per_dp] * dp_size,
                                     dtype=jnp.int32)

    base_indices = jnp.arange(1, num_reqs_per_dp + 1,
                              dtype=jnp.int32) * seq_len - 1
    logits_indices = jnp.concatenate(
        [base_indices + r * tokens_per_dp for r in range(dp_size)],
        axis=0)

    attention_metadata = AttentionMetadata(
        input_positions=positions,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
    )
    return positions, attention_metadata, logits_indices, max_num_blocks_per_req


def create_kv_caches(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    num_blocks: int,
    block_size: int,
    mesh: Mesh,
    dtype: jnp.dtype = jnp.bfloat16,
) -> List[jax.Array]:
    """Create KV caches for benchmarking."""
    from tpu_inference.runner.kv_cache import create_kv_caches as _create_kv_caches

    layer_names = [f"layer_{i}" for i in range(num_layers)]
    return _create_kv_caches(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        mesh=mesh,
        layer_names=layer_names,
        cache_dtype=dtype,
    )


class ModelBenchmarker:
    """Benchmarker for comparing model implementations."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.mesh = create_mesh(config.tensor_parallel_size)
        self.rng = jax.random.PRNGKey(42)
        self.results = BenchmarkResults(config=config)

        # Collect metadata
        self.results.metadata = {
            "jax_version": jax.__version__,
            "num_devices": jax.device_count(),
            "device_kind": str(jax.devices()[0].device_kind) if jax.devices() else "unknown",
            "platform": jax.default_backend(),
        }

    def _get_dtype(self) -> jnp.dtype:
        """Get JAX dtype from config."""
        dtype_map = {
            "float32": jnp.float32,
            "float16": jnp.float16,
            "bfloat16": jnp.bfloat16,
        }
        return dtype_map.get(self.config.dtype, jnp.bfloat16)

    def _load_model_bundle(self, impl_type: str) -> Optional[ModelBundle]:
        """Load model and return a bundle of callables and metadata."""
        envs.MODEL_IMPL_TYPE = impl_type
        max_model_len = (self.config.max_model_len
                         or max(self.config.seq_lengths))
        max_num_seqs = max(self.config.batch_sizes)
        max_num_batched_tokens = max(self.config.seq_lengths) * max_num_seqs

        vllm_config = create_vllm_config(
            model_path=self.config.model_path,
            dtype=self.config.dtype,
            use_dummy_weights=self.config.use_dummy_weights,
            tensor_parallel_size=self.config.tensor_parallel_size,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        if vllm_config is None:
            return None

        model_fn, compute_logits_fn, _, multimodal_fns, state, _, model = get_model(
            vllm_config, self.rng, self.mesh)
        hf_config = vllm_config.model_config.hf_config
        text_config = _get_text_config(hf_config)
        num_layers = getattr(text_config, "num_hidden_layers", 0)

        layer_names: List[str] = []
        if impl_type == "vllm":
            layer_names = _collect_vllm_layer_names(model)
        if not layer_names:
            layer_names = [f"layer.{i}" for i in range(num_layers)]
        layer_name_to_kvcache_index = tuple(
            (name, i) for i, name in enumerate(layer_names))

        return ModelBundle(
            impl_type=impl_type,
            vllm_config=vllm_config,
            model_fn=model_fn,
            compute_logits_fn=compute_logits_fn,
            multimodal_fns=multimodal_fns,
            state=state,
            model=model,
            text_config=text_config,
            layer_names=layer_names,
            layer_name_to_kvcache_index=layer_name_to_kvcache_index,
        )

    def _benchmark_single(
        self,
        name: str,
        impl_type: str,
        forward_fn: Callable,
        input_data: Dict[str, Any],
        seq_length: int,
        batch_size: int,
    ) -> TimingResult:
        """Run a single benchmark."""
        kv_caches = input_data.get("kv_caches")
        static_inputs = {
            key: value
            for key, value in input_data.items()
            if key != "kv_caches"
        }

        def _run_once(kv_state):
            if kv_state is None:
                return None, forward_fn(**static_inputs)
            kv_out, output = forward_fn(kv_caches=kv_state, **static_inputs)
            return kv_out, output

        # Warmup and measure compile time
        with Timer() as compile_timer:
            for _ in range(self.config.num_warmup):
                kv_caches, result = _run_once(kv_caches)
                _block_until_ready(result)

        compile_time_ms = compile_timer.elapsed_ms / max(self.config.num_warmup, 1)

        # Clear caches
        gc.collect()

        # Run benchmark iterations
        times_ms = []
        for _ in range(self.config.num_iterations):
            with Timer() as iter_timer:
                kv_caches, result = _run_once(kv_caches)
                _block_until_ready(result)
            times_ms.append(iter_timer.elapsed_ms)

        times_ms = np.array(times_ms)
        total_tokens = seq_length * batch_size

        return TimingResult(
            name=name,
            impl_type=impl_type,
            seq_length=seq_length,
            batch_size=batch_size,
            compile_time_ms=compile_time_ms,
            mean_time_ms=float(np.mean(times_ms)),
            std_time_ms=float(np.std(times_ms)),
            min_time_ms=float(np.min(times_ms)),
            max_time_ms=float(np.max(times_ms)),
            throughput_tokens_per_sec=total_tokens / (np.mean(times_ms) / 1000),
            memory_used_gb=get_memory_usage_gb(),
            all_times_ms=times_ms.tolist(),
        )

    def _benchmark_impl(self, impl_type: str) -> List[TimingResult]:
        """Benchmark a specific implementation."""
        print("\n" + "=" * 60)
        title = ("Native JAX/Flax NNX Implementation"
                 if impl_type == "flax_nnx" else "vLLM/TorchAX Implementation")
        print(f"Benchmarking: {title}")
        print("=" * 60)

        results: List[TimingResult] = []

        try:
            print(f"Creating Qwen3VL model ({impl_type})...")
            with Timer() as model_timer:
                bundle = self._load_model_bundle(impl_type)
            if bundle is None:
                print(f"Failed to create vLLM config for {impl_type}")
                return results
            print(f"Model creation time: {model_timer.elapsed_ms:.2f} ms")

            hf_config = bundle.vllm_config.model_config.hf_config
            text_config = bundle.text_config
            num_kv_heads = getattr(text_config, "num_key_value_heads",
                                   getattr(text_config, "num_attention_heads",
                                           0))
            num_attention_heads = getattr(text_config, "num_attention_heads",
                                           max(num_kv_heads, 1))
            hidden_size = getattr(text_config, "hidden_size",
                                  getattr(hf_config, "hidden_size", 0))
            head_dim = hidden_size // max(num_attention_heads, 1)
            num_kv_heads = common_utils.get_padded_num_heads(
                num_kv_heads, self.mesh.shape["model"])
            head_dim = common_utils.get_padded_head_dim(head_dim)
            if hasattr(bundle.vllm_config.model_config, "get_vocab_size"):
                vocab_size = bundle.vllm_config.model_config.get_vocab_size()
            else:
                vocab_size = getattr(text_config, "vocab_size",
                                     getattr(hf_config, "vocab_size", 0))
            if vocab_size <= 0:
                raise ValueError("Invalid vocab size for benchmarking.")

            block_size = getattr(bundle.vllm_config.cache_config, "block_size",
                                 16)
            use_mrope = _uses_mrope(bundle.vllm_config.model_config, hf_config)

            embed_input_ids_fn = None
            if bundle.multimodal_fns:
                embed_input_ids_fn = bundle.multimodal_fns.get(
                    "embed_input_ids_fn")
            use_inputs_embeds = (_is_multimodal_model(
                bundle.vllm_config.model_config) and embed_input_ids_fn is not None)
            mm_embeds = None
            if use_inputs_embeds:
                mm_embeds = jnp.zeros((0, hidden_size),
                                      dtype=self._get_dtype())

            # Benchmark for each sequence length and batch size
            for seq_len in self.config.seq_lengths:
                for batch_size in self.config.batch_sizes:
                    print(f"\n  seq_len={seq_len}, batch_size={batch_size}")

                    total_tokens = seq_len * batch_size
                    self.rng, data_rng = jax.random.split(self.rng)
                    input_ids = jax.random.randint(
                        data_rng, (total_tokens,), 0, vocab_size, dtype=jnp.int32)
                    disallowed_ids = []
                    for key in ("image_token_id", "video_token_id",
                                "vision_start_token_id"):
                        token_id = getattr(hf_config, key, None)
                        if isinstance(token_id, (int, np.integer)):
                            disallowed_ids.append(int(token_id))
                    input_ids = _sanitize_input_ids(input_ids, vocab_size,
                                                    disallowed_ids)

                    dp_size = int(self.mesh.shape.get("data", 1))
                    input_positions, attn_metadata, logits_indices, max_blocks = (
                        build_attention_metadata(seq_len, batch_size,
                                                 block_size, use_mrope,
                                                 dp_size))
                    kv_caches = create_kv_caches(
                        num_layers=len(bundle.layer_names),
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        num_blocks=max_blocks * batch_size,
                        block_size=block_size,
                        mesh=self.mesh,
                        dtype=self._get_dtype(),
                    )

                    def forward_fn(kv_caches, input_ids, attn_metadata,
                                   input_positions, logits_indices,
                                   layer_name_to_kvcache_index, mm_embeds):
                        inputs_embeds = None
                        if use_inputs_embeds:
                            inputs_embeds = embed_input_ids_fn(
                                bundle.state, input_ids, mm_embeds)

                        kv_caches, hidden_states, _ = bundle.model_fn(
                            bundle.state,
                            kv_caches,
                            input_ids,
                            attn_metadata,
                            inputs_embeds,
                            input_positions,
                            layer_name_to_kvcache_index,
                            None,  # lora_metadata
                            None,  # intermediate_tensors
                            True,  # is_first_rank
                            True,  # is_last_rank
                            None,  # deepstack_embeds
                        )

                        selected_hidden = hidden_states[logits_indices]
                        logits = bundle.compute_logits_fn(
                            bundle.state,
                            selected_hidden,
                            None,  # lora_metadata
                        )
                        return kv_caches, logits

                    result = self._benchmark_single(
                        name="forward_pass",
                        impl_type=impl_type,
                        forward_fn=forward_fn,
                        input_data={
                            "kv_caches": kv_caches,
                            "input_ids": input_ids,
                            "attn_metadata": attn_metadata,
                            "input_positions": input_positions,
                            "logits_indices": logits_indices,
                            "layer_name_to_kvcache_index":
                            bundle.layer_name_to_kvcache_index,
                            "mm_embeds": mm_embeds,
                        },
                        seq_length=seq_len,
                        batch_size=batch_size,
                    )
                    results.append(result)
                    self._print_result(result)

            if self.config.test_vision:
                results.extend(self._benchmark_vision_encoder(bundle))

        except Exception as e:
            print(f"Error benchmarking {impl_type}: {e}")
            import traceback
            traceback.print_exc()

        return results

    def benchmark_flax_nnx(self) -> List[TimingResult]:
        """Benchmark the native JAX/Flax NNX implementation."""
        return self._benchmark_impl("flax_nnx")

    def benchmark_vllm(self) -> List[TimingResult]:
        """Benchmark the vLLM/TorchAX implementation."""
        return self._benchmark_impl("vllm")

    def _benchmark_vision_encoder(self,
                                  bundle: ModelBundle) -> List[TimingResult]:
        """Benchmark the vision encoder."""
        print(f"\n  Benchmarking Vision Encoder ({bundle.impl_type})")
        results: List[TimingResult] = []

        embed_multimodal_fn = None
        if bundle.multimodal_fns:
            embed_multimodal_fn = bundle.multimodal_fns.get(
                "embed_multimodal_fn")
        if embed_multimodal_fn is None:
            print("  Vision encoder not available for this implementation.")
            return results

        try:
            grid_thw = self.config.vision_grid_thw
            t, h, w = grid_thw
            image_grid_thw = (grid_thw, )

            vc = bundle.vllm_config.model_config.hf_config.vision_config
            patch_dim = int(vc.in_channels * vc.temporal_patch_size *
                            vc.patch_size * vc.patch_size)
            num_patches = t * h * w

            self.rng, data_rng = jax.random.split(self.rng)
            pixel_values = jax.random.normal(
                data_rng, (num_patches, patch_dim), dtype=self._get_dtype())

            def forward_fn(state, pixel_values, image_grid_thw):
                return embed_multimodal_fn(state,
                                           image_grid_thw,
                                           pixel_values=pixel_values)

            result = self._benchmark_single(
                name="vision_encoder",
                impl_type=bundle.impl_type,
                forward_fn=forward_fn,
                input_data={
                    "state": bundle.state,
                    "pixel_values": pixel_values,
                    "image_grid_thw": image_grid_thw,
                },
                seq_length=num_patches,
                batch_size=1,
            )
            results.append(result)
            self._print_result(result)

        except Exception as e:
            print(f"  Vision encoder benchmark failed: {e}")

        return results

    def _print_result(self, result: TimingResult):
        """Print a single result."""
        print(f"    Mean: {result.mean_time_ms:.2f} ms (±{result.std_time_ms:.2f})")
        print(f"    Min/Max: {result.min_time_ms:.2f}/{result.max_time_ms:.2f} ms")
        print(f"    Throughput: {result.throughput_tokens_per_sec:.0f} tokens/sec")
        print(f"    Memory: {result.memory_used_gb:.2f} GB")

    def run_profile(self, impl_type: str = "flax_nnx"):
        """Run with JAX profiler enabled."""
        if not self.config.profile_dir:
            print("No profile directory specified, skipping profiling")
            return

        profile_path = Path(self.config.profile_dir) / impl_type
        profile_path.mkdir(parents=True, exist_ok=True)

        print(f"\nRunning with profiler, output: {profile_path}")

        jax.profiler.start_trace(str(profile_path))
        try:
            if impl_type == "flax_nnx":
                self.benchmark_flax_nnx()
            else:
                self.benchmark_vllm()
        finally:
            jax.profiler.stop_trace()

        print(f"Profile saved to: {profile_path}")

    def run_all(self) -> BenchmarkResults:
        """Run all benchmarks."""
        print("\n" + "#" * 60)
        print("# Qwen3VL Performance Benchmark")
        print("# Model: " + self.config.model_path)
        print("# Devices: " + str(jax.device_count()))
        print("#" * 60)

        # Benchmark both implementations
        flax_results = self.benchmark_flax_nnx()
        self.results.results.extend(flax_results)

        vllm_results = self.benchmark_vllm()
        self.results.results.extend(vllm_results)

        # Print comparison
        self._print_comparison()

        # Save results
        if self.config.output_file:
            self.results.save(self.config.output_file)
            print(f"\nResults saved to: {self.config.output_file}")

        return self.results

    def _print_comparison(self):
        """Print comparison between implementations."""
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)

        # Group results by (name, seq_length, batch_size)
        flax_results = {}
        vllm_results = {}

        for r in self.results.results:
            key = (r.name, r.seq_length, r.batch_size)
            if r.impl_type == "flax_nnx":
                flax_results[key] = r
            else:
                vllm_results[key] = r

        # Compare
        print(f"\n{'Benchmark':<20} {'SeqLen':<8} {'Batch':<6} "
              f"{'flax_nnx (ms)':<15} {'vllm (ms)':<15} {'Speedup':<10}")
        print("-" * 80)

        for key in sorted(flax_results.keys()):
            name, seq_len, batch_size = key
            flax_r = flax_results.get(key)
            vllm_r = vllm_results.get(key)

            flax_time = flax_r.mean_time_ms if flax_r else float("nan")
            vllm_time = vllm_r.mean_time_ms if vllm_r else float("nan")

            if flax_r and vllm_r:
                speedup = vllm_time / flax_time
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"

            print(f"{name:<20} {seq_len:<8} {batch_size:<6} "
                  f"{flax_time:<15.2f} {vllm_time:<15.2f} {speedup_str:<10}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3VL: JAX Native vs vLLM/TorchAX"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-4B",
        help="Model path or name",
    )
    parser.add_argument(
        "--seq-lengths",
        type=str,
        default="128,256,512",
        help="Comma-separated sequence lengths to test",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1",
        help="Comma-separated batch sizes to test",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=3,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=None,
        help="Directory for JAX profiler output",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override max_model_len (defaults to max seq length).",
    )
    parser.add_argument(
        "--real-weights",
        action="store_true",
        help="Load real model weights instead of dummy weights.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Skip vision encoder benchmarks",
    )
    parser.add_argument(
        "--impl",
        type=str,
        choices=["all", "flax_nnx", "vllm"],
        default="all",
        help="Which implementation to benchmark",
    )
    parser.add_argument(
        "--profile-only",
        type=str,
        choices=["flax_nnx", "vllm"],
        default=None,
        help="Only run profiling for specified implementation",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = BenchmarkConfig(
        model_path=args.model,
        seq_lengths=[int(x) for x in args.seq_lengths.split(",")],
        batch_sizes=[int(x) for x in args.batch_sizes.split(",")],
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        profile_dir=args.profile_dir,
        output_file=args.output,
        use_dummy_weights=not args.real_weights,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        test_vision=not args.no_vision,
        max_model_len=args.max_model_len,
    )

    benchmarker = ModelBenchmarker(config)

    if args.profile_only:
        benchmarker.run_profile(args.profile_only)
    elif args.impl == "all":
        benchmarker.run_all()
    elif args.impl == "flax_nnx":
        results = benchmarker.benchmark_flax_nnx()
        benchmarker.results.results.extend(results)
        if config.output_file:
            benchmarker.results.save(config.output_file)
    elif args.impl == "vllm":
        results = benchmarker.benchmark_vllm()
        benchmarker.results.results.extend(results)
        if config.output_file:
            benchmarker.results.save(config.output_file)


if __name__ == "__main__":
    main()
