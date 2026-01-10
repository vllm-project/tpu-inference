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
import contextlib
import dataclasses
import gc
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

# Ensure JAX is configured before imports
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


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


class Timer:
    """Context manager for timing operations with JAX synchronization."""

    def __init__(self, sync: bool = True):
        self.sync = sync
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        if self.sync:
            jax.block_until_ready(jax.device_count())
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.sync:
            jax.block_until_ready(jax.device_count())
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


def create_mock_vllm_config(
    model_path: str,
    dtype: str,
    use_dummy_weights: bool,
    tensor_parallel_size: int,
) -> Any:
    """Create a mock VllmConfig for testing."""
    try:
        from transformers import AutoConfig
        from vllm.config import (
            CacheConfig,
            DeviceConfig,
            LoadConfig,
            ModelConfig,
            MultiModalConfig,
            ParallelConfig,
            SchedulerConfig,
        )

        # Load actual HF config if available, otherwise create mock
        try:
            hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            # Create a minimal mock config for Qwen3VL
            from transformers import Qwen2VLConfig
            hf_config = Qwen2VLConfig(
                vocab_size=152064,
                hidden_size=3584,
                intermediate_size=18944,
                num_hidden_layers=28,
                num_attention_heads=28,
                num_key_value_heads=4,
                hidden_act="silu",
                max_position_embeddings=32768,
                rms_norm_eps=1e-6,
                rope_theta=1000000.0,
                tie_word_embeddings=False,
            )
            # Add vision config
            hf_config.vision_config = MagicMock()
            hf_config.vision_config.hidden_size = 1280
            hf_config.vision_config.intermediate_size = 5120
            hf_config.vision_config.num_hidden_layers = 32
            hf_config.vision_config.num_attention_heads = 16
            hf_config.vision_config.patch_size = 14
            hf_config.vision_config.image_size = 384
            hf_config.vision_config.temporal_patch_size = 2
            hf_config.vision_config.in_channels = 3
            hf_config.vision_config.spatial_merge_size = 2
            hf_config.vision_config.out_hidden_size = 3584
            hf_config.vision_config.depth = 32
            hf_config.vision_config.num_heads = 16

        # Ensure required attributes exist
        if not hasattr(hf_config, "image_token_id"):
            hf_config.image_token_id = 151655
        if not hasattr(hf_config, "video_token_id"):
            hf_config.video_token_id = 151656
        if not hasattr(hf_config, "vision_start_token_id"):
            hf_config.vision_start_token_id = 151652

        # Create vLLM config components
        model_config = MagicMock()
        model_config.hf_config = hf_config
        model_config.dtype = dtype
        model_config.model = model_path
        model_config.seed = 42

        cache_config = MagicMock(spec=CacheConfig)
        cache_config.cache_dtype = "auto"
        cache_config.block_size = 16
        cache_config.num_gpu_blocks = 1024

        parallel_config = MagicMock(spec=ParallelConfig)
        parallel_config.tensor_parallel_size = tensor_parallel_size
        parallel_config.pipeline_parallel_size = 1
        parallel_config.data_parallel_size = 1

        load_config = MagicMock(spec=LoadConfig)
        load_config.load_format = "dummy" if use_dummy_weights else "auto"

        scheduler_config = MagicMock(spec=SchedulerConfig)
        device_config = MagicMock(spec=DeviceConfig)

        # Create mock VllmConfig
        vllm_config = MagicMock()
        vllm_config.model_config = model_config
        vllm_config.cache_config = cache_config
        vllm_config.parallel_config = parallel_config
        vllm_config.load_config = load_config
        vllm_config.scheduler_config = scheduler_config
        vllm_config.device_config = device_config
        vllm_config.speculative_config = None
        vllm_config.lora_config = None
        vllm_config.additional_config = {"enable_dynamic_image_sizes": False}

        # Sharding config
        sharding_config = MagicMock()
        sharding_config.total_dp_size = 1
        sharding_config.tp_size = tensor_parallel_size
        vllm_config.sharding_config = sharding_config

        return vllm_config

    except ImportError as e:
        print(f"Warning: Could not import vLLM components: {e}")
        return None


def create_mesh(devices: Optional[List] = None) -> Mesh:
    """Create a JAX mesh for sharding."""
    if devices is None:
        devices = jax.local_devices()
    devices_array = np.array(devices).reshape((len(devices), 1, 1))
    return Mesh(devices_array, axis_names=("data", "attn_dp", "model"))


def create_attention_metadata(
    seq_len: int,
    batch_size: int = 1,
) -> Any:
    """Create attention metadata for benchmarking."""
    from tpu_inference.layers.common.attention_metadata import AttentionMetadata

    num_reqs = batch_size
    max_num_blocks_per_req = (seq_len + 15) // 16

    # Create 3D positions for MRoPE
    positions = jnp.broadcast_to(
        jnp.arange(seq_len, dtype=jnp.int32)[None, :],
        (3, seq_len)
    )
    block_tables = jnp.zeros(
        (num_reqs, max_num_blocks_per_req), dtype=jnp.int32
    ).reshape(-1)
    seq_lens = jnp.array([seq_len] * num_reqs, dtype=jnp.int32)
    query_start_loc = jnp.concatenate([
        jnp.array([0], dtype=jnp.int32),
        jnp.cumsum(seq_lens)
    ])
    request_distribution = jnp.array([0, 0, num_reqs], dtype=jnp.int32)

    return AttentionMetadata(
        input_positions=positions,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
    )


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
        self.mesh = create_mesh()
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
        # Warmup and measure compile time
        with Timer() as compile_timer:
            for _ in range(self.config.num_warmup):
                _ = forward_fn(**input_data)
                jax.block_until_ready(_)

        compile_time_ms = compile_timer.elapsed_ms / max(self.config.num_warmup, 1)

        # Clear caches
        gc.collect()

        # Run benchmark iterations
        times_ms = []
        for _ in range(self.config.num_iterations):
            with Timer() as iter_timer:
                result = forward_fn(**input_data)
                jax.block_until_ready(result)
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

    def benchmark_flax_nnx(self) -> List[TimingResult]:
        """Benchmark the native JAX/Flax NNX implementation."""
        print("\n" + "=" * 60)
        print("Benchmarking: Native JAX/Flax NNX Implementation")
        print("=" * 60)

        results = []

        try:
            # Set environment for flax_nnx
            import tpu_inference.envs as envs
            envs.MODEL_IMPL_TYPE = "flax_nnx"

            vllm_config = create_mock_vllm_config(
                model_path=self.config.model_path,
                dtype=self.config.dtype,
                use_dummy_weights=self.config.use_dummy_weights,
                tensor_parallel_size=self.config.tensor_parallel_size,
            )

            if vllm_config is None:
                print("Failed to create vLLM config for flax_nnx")
                return results

            # Import and create the model
            from tpu_inference.models.jax.qwen3_vl import Qwen3VLForConditionalGeneration

            print("Creating Qwen3VL model (flax_nnx)...")
            with Timer() as model_timer:
                model = Qwen3VLForConditionalGeneration(
                    vllm_config, self.rng, self.mesh
                )
            print(f"Model creation time: {model_timer.elapsed_ms:.2f} ms")

            # Get model config
            hf_config = vllm_config.model_config.hf_config
            num_layers = getattr(hf_config, "num_hidden_layers", 28)
            num_kv_heads = getattr(hf_config, "num_key_value_heads", 4)
            head_dim = getattr(hf_config, "hidden_size", 3584) // getattr(
                hf_config, "num_attention_heads", 28
            )
            vocab_size = getattr(hf_config, "vocab_size", 152064)

            # Create KV caches
            kv_caches = create_kv_caches(
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                num_blocks=64,
                block_size=16,
                mesh=self.mesh,
                dtype=self._get_dtype(),
            )

            # JIT compile the model
            @jax.jit
            def forward_pass(kv_caches, input_ids, attn_metadata):
                return model(kv_caches, input_ids, attn_metadata)

            # Benchmark for each sequence length and batch size
            for seq_len in self.config.seq_lengths:
                for batch_size in self.config.batch_sizes:
                    print(f"\n  seq_len={seq_len}, batch_size={batch_size}")

                    # Create inputs
                    total_tokens = seq_len * batch_size
                    input_ids = jax.random.randint(
                        self.rng, (total_tokens,), 0, vocab_size, dtype=jnp.int32
                    )
                    attn_metadata = create_attention_metadata(seq_len, batch_size)

                    # Benchmark forward pass
                    result = self._benchmark_single(
                        name="forward_pass",
                        impl_type="flax_nnx",
                        forward_fn=forward_pass,
                        input_data={
                            "kv_caches": kv_caches,
                            "input_ids": input_ids,
                            "attn_metadata": attn_metadata,
                        },
                        seq_length=seq_len,
                        batch_size=batch_size,
                    )
                    results.append(result)
                    self._print_result(result)

            # Benchmark vision encoder if enabled
            if self.config.test_vision:
                results.extend(self._benchmark_vision_encoder(model, "flax_nnx"))

        except Exception as e:
            print(f"Error benchmarking flax_nnx: {e}")
            import traceback
            traceback.print_exc()

        return results

    def benchmark_vllm(self) -> List[TimingResult]:
        """Benchmark the vLLM/TorchAX implementation."""
        print("\n" + "=" * 60)
        print("Benchmarking: vLLM/TorchAX Implementation")
        print("=" * 60)

        results = []

        try:
            # Set environment for vllm
            import tpu_inference.envs as envs
            envs.MODEL_IMPL_TYPE = "vllm"

            vllm_config = create_mock_vllm_config(
                model_path=self.config.model_path,
                dtype=self.config.dtype,
                use_dummy_weights=self.config.use_dummy_weights,
                tensor_parallel_size=self.config.tensor_parallel_size,
            )

            if vllm_config is None:
                print("Failed to create vLLM config for vllm")
                return results

            # Import and create the model wrapper
            from tpu_inference.models.vllm.vllm_model_wrapper import VllmModelWrapper

            print("Creating Qwen3VL model (vllm/TorchAX)...")
            with Timer() as model_timer:
                model = VllmModelWrapper(vllm_config, self.rng, self.mesh)
                params, lora_manager = model.load_weights()
            print(f"Model creation time: {model_timer.elapsed_ms:.2f} ms")

            # Get the JIT-compiled step function
            step_fn = model.jit_step_func()

            # Get model config
            hf_config = vllm_config.model_config.hf_config
            num_layers = getattr(hf_config, "num_hidden_layers", 28)
            num_kv_heads = getattr(hf_config, "num_key_value_heads", 4)
            head_dim = getattr(hf_config, "hidden_size", 3584) // getattr(
                hf_config, "num_attention_heads", 28
            )
            vocab_size = getattr(hf_config, "vocab_size", 152064)

            # Create KV caches
            kv_caches = create_kv_caches(
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                num_blocks=64,
                block_size=16,
                mesh=self.mesh,
                dtype=self._get_dtype(),
            )

            # Create layer name to kv cache index mapping
            layer_name_to_kvcache_index = tuple(
                (f"model.layers.{i}.self_attn", i) for i in range(num_layers)
            )

            # Benchmark for each sequence length and batch size
            for seq_len in self.config.seq_lengths:
                for batch_size in self.config.batch_sizes:
                    print(f"\n  seq_len={seq_len}, batch_size={batch_size}")

                    # Create inputs
                    total_tokens = seq_len * batch_size
                    input_ids = jax.random.randint(
                        self.rng, (total_tokens,), 0, vocab_size, dtype=jnp.int32
                    )
                    attn_metadata = create_attention_metadata(seq_len, batch_size)
                    input_positions = jnp.arange(total_tokens, dtype=jnp.int32)
                    input_embeds = jnp.zeros((0,), dtype=self._get_dtype())

                    def forward_fn(
                        params_and_buffers,
                        kv_caches,
                        input_ids,
                        attn_metadata,
                        input_embeds,
                        input_positions,
                        layer_name_to_kvcache_index,
                    ):
                        return step_fn(
                            params_and_buffers,
                            kv_caches,
                            input_ids,
                            attn_metadata,
                            input_embeds,
                            input_positions,
                            layer_name_to_kvcache_index,
                            None,  # lora_metadata
                            None,  # intermediate_tensors
                            True,  # is_first_rank
                            True,  # is_last_rank
                        )

                    # Benchmark forward pass
                    result = self._benchmark_single(
                        name="forward_pass",
                        impl_type="vllm",
                        forward_fn=forward_fn,
                        input_data={
                            "params_and_buffers": params,
                            "kv_caches": kv_caches,
                            "input_ids": input_ids,
                            "attn_metadata": attn_metadata,
                            "input_embeds": input_embeds,
                            "input_positions": input_positions,
                            "layer_name_to_kvcache_index": layer_name_to_kvcache_index,
                        },
                        seq_length=seq_len,
                        batch_size=batch_size,
                    )
                    results.append(result)
                    self._print_result(result)

        except Exception as e:
            print(f"Error benchmarking vllm: {e}")
            import traceback
            traceback.print_exc()

        return results

    def _benchmark_vision_encoder(
        self, model: Any, impl_type: str
    ) -> List[TimingResult]:
        """Benchmark the vision encoder."""
        print(f"\n  Benchmarking Vision Encoder ({impl_type})")
        results = []

        try:
            grid_thw = self.config.vision_grid_thw
            t, h, w = grid_thw

            # Get vision config
            vc = model.config.vision_config
            patch_dim = int(
                vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
            )
            num_patches = t * h * w

            # Create dummy pixel values
            pixel_values = jax.random.normal(
                self.rng, (num_patches, patch_dim), dtype=self._get_dtype()
            )

            @jax.jit
            def vision_forward(pixel_values, grid_thw):
                return model.embed_multimodal((grid_thw,), pixel_values=pixel_values)

            # Benchmark vision encoder
            result = self._benchmark_single(
                name="vision_encoder",
                impl_type=impl_type,
                forward_fn=lambda: vision_forward(pixel_values, grid_thw),
                input_data={},
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
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        test_vision=not args.no_vision,
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
