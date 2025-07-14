import os
import time
import traceback
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import jax.profiler
from absl import flags
from absl.testing import absltest, parameterized

# Assuming the kernel is in the expected path
# You may need to adjust this import based on your project structure
from tpu_commons.kernels.ragged_paged_attention import \
    kernel as paged_attention_kernel

FLAGS = flags.FLAGS

flags.DEFINE_string("profile_dir", "/mnt/disks/persist/paged_attention_bench",
                    "Directory to save JAX profiler traces.")
flags.DEFINE_boolean("profile", True,
                     "Enable JAX profiling for all test cases.")


# --- 1. Reference JAX implementation for DENSE inputs ---
@jax.jit
def reference_attention_dense_jit(q, k, v, sm_scale=1.0):
    """A reference implementation of GQA for dense, padded inputs."""
    batch, q_len, num_q_heads, head_dim = q.shape
    _, kv_len, num_kv_heads, _ = k.shape

    if num_kv_heads != num_q_heads:
        num_groups = num_q_heads // num_kv_heads
        k = jnp.repeat(k, num_groups, axis=2)
        v = jnp.repeat(v, num_groups, axis=2)

    logits = jnp.einsum(
        "bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32) * sm_scale

    mask_shape = (q_len, kv_len)
    q_idxs = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
    kv_idxs = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
    causal_mask = kv_idxs > (kv_len - q_len + q_idxs)
    causal_mask = jnp.broadcast_to(causal_mask,
                                   (batch, num_q_heads, q_len, kv_len))

    logits = jnp.where(causal_mask, jnp.finfo(jnp.float32).min, logits)
    weights = jax.nn.softmax(logits, axis=-1).astype(q.dtype)
    output = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return output


@jax.jit
def reference_attention_dense_masked_jit(q, k, v, mask, sm_scale=1.0):
    """A reference implementation of GQA for dense, padded inputs with a mask."""
    batch, q_len, num_q_heads, head_dim = q.shape
    _, kv_len, num_kv_heads, _ = k.shape

    if num_kv_heads != num_q_heads:
        num_groups = num_q_heads // num_kv_heads
        k = jnp.repeat(k, num_groups, axis=2)
        v = jnp.repeat(v, num_groups, axis=2)

    logits = jnp.einsum(
        "bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32) * sm_scale

    # The mask is broadcast to the logits shape (batch, num_q_heads, q_len, kv_len)
    logits = jnp.where(mask, jnp.finfo(jnp.float32).min, logits)

    weights = jax.nn.softmax(logits, axis=-1).astype(q.dtype)
    output = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return output


# --- 2. Input Setup Helper ---
def _setup_inputs(seq_lens: List[Tuple[int, int]], num_heads: Tuple[int, int],
                  head_dim: int, page_size: int, dtype: jnp.dtype,
                  num_pages: int, **kwargs) -> Tuple:
    """Creates dummy paged/ragged inputs for the Pallas kernel."""
    cu_q_lens = [0]
    kv_lens = []
    for q_len, kv_len in seq_lens:
        cu_q_lens.append(cu_q_lens[-1] + q_len)
        kv_lens.append(kv_len)

    max_num_batched_tokens = cu_q_lens[-1]
    max_num_seq = len(seq_lens)
    max_kv_len = max(kv_lens) if kv_lens else 0
    pages_per_seq = paged_attention_kernel.cdiv(max_kv_len, page_size)
    num_q_heads, num_kv_heads = num_heads

    prng_key = jax.random.key(1234)
    k0, k1, k2 = jax.random.split(prng_key, 3)

    q = jax.random.normal(k0, (max_num_batched_tokens, num_q_heads, head_dim),
                          dtype=dtype)
    kv_pages = jax.random.normal(
        k1, (num_pages, page_size, num_kv_heads * 2, head_dim), dtype=dtype)
    page_indices = jax.random.randint(k2, (max_num_seq, pages_per_seq), 0,
                                      num_pages).astype(jnp.int32)

    cu_q_lens_arr = jnp.array(cu_q_lens, dtype=jnp.int32)
    kv_lens_arr = jnp.array(kv_lens, dtype=jnp.int32)
    num_seqs_arr = jnp.array([len(seq_lens)], dtype=jnp.int32)

    padded_cu_q_lens = jnp.zeros(
        max_num_seq + 1,
        dtype=jnp.int32).at[:len(cu_q_lens_arr)].set(cu_q_lens_arr)
    padded_kv_lens = jnp.zeros(
        max_num_seq, dtype=jnp.int32).at[:len(kv_lens_arr)].set(kv_lens_arr)

    return q, kv_pages, padded_kv_lens, page_indices, padded_cu_q_lens, num_seqs_arr


# --- 3. Main Test Class ---
class PagedAttentionBenchmark(parameterized.TestCase):

    _benchmark_results: List[Dict[str, Any]] = []

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if FLAGS.profile:
            os.makedirs(FLAGS.profile_dir, exist_ok=True)
            print(
                f"JAX profiling enabled. Base trace directory: {FLAGS.profile_dir}"
            )

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls._print_final_synopsis()

    @classmethod
    def _print_final_synopsis(cls):
        pallas_ragged = next((r for r in cls._benchmark_results
                              if r["name"] == "Pallas_Decode_Ragged"), None)
        ref_ragged = next((r for r in cls._benchmark_results
                           if r["name"] == "Ref_Decode_Ragged"), None)

        if not pallas_ragged or not ref_ragged:
            # Don't print anything if the key tests weren't run
            return

        # --- Data Extraction ---
        lat_p, lat_r = pallas_ragged['latency_ms'], ref_ragged['latency_ms']
        tp_p, tp_r = pallas_ragged['effective_throughput_tok_s'], ref_ragged[
            'effective_throughput_tok_s']
        mem_p, mem_r = pallas_ragged['kv_cache_mem_mb'], ref_ragged[
            'kv_cache_mem_mb']
        waste_r = ref_ragged['wasted_computation_pct']

        # --- Calculations ---
        lat_ratio = lat_p / lat_r if lat_p > lat_r else lat_r / lat_p
        lat_slower = "slower" if lat_p > lat_r else "faster"

        mem_multiplier = mem_r / mem_p if mem_p > 0 else 1.0

        # Project system throughput by accounting for the larger batch size the paged kernel can handle.
        # We assume throughput (tok/s) scales linearly with batch size for this projection.
        proj_tp_p = tp_p * mem_multiplier
        proj_tp_r = tp_r  # Baseline, multiplier is 1.0

        tp_ratio = proj_tp_r / proj_tp_p if proj_tp_p < proj_tp_r else proj_tp_p / proj_tp_r
        tp_winner = "JAX (Padded)" if proj_tp_p < proj_tp_r else "Pallas (Ragged)"

        # --- Table Printing ---
        print("\n" + "=" * 120)
        print(
            "Final Synopsis: Paged (Ragged) vs JAX (Padded Dense) on a Mixed-Length Decode Batch"
            .center(120))
        print("=" * 120)
        print(
            "This comparison quantifies the trade-offs on a realistic, ragged batch to determine the bottom-line throughput."
        )

        print(
            "\n| Metric                        | Pallas (Ragged)      | JAX (Padded)         | Analysis                                                       |"
        )
        print(
            "| :---------------------------- | :------------------- | :------------------- | :------------------------------------------------------------- |"
        )

        latency_analysis = f"Pallas is {lat_ratio:.1f}x {lat_slower} in raw latency for a fixed batch."
        print(
            f"| 1. Per-Batch Latency (ms)     | {lat_p:<20.3f} | {lat_r:<20.3f} | {latency_analysis:<62} |"
        )

        mem_analysis = f"Dense kernel wastes {waste_r:.1f}% of compute on padding."
        print(
            f"| 2. Computational Waste        | 0.0%                 | {f'{waste_r:.1f}%':<20} | {mem_analysis:<62} |"
        )

        mem_multiplier_analysis = f"Enables a {mem_multiplier:.1f}x larger batch in the same K/V cache."
        print(
            f"| 3. Memory Advantage           | {f'{mem_multiplier:.1f}x batch multiplier':<20} | 1.0x (baseline)      | {mem_multiplier_analysis:<62} |"
        )

        proj_tp_analysis = f"--> {tp_winner} is {tp_ratio:.1f}x faster in overall throughput."
        print(
            f"| 4. Projected Throughput (tok/s) | {proj_tp_p:<20,.0f} | {proj_tp_r:<20,.0f} | {proj_tp_analysis:<62} |"
        )

        # --- Bottom-Line Analysis ---
        print("\nBottom-Line Analysis:")
        if tp_winner == "JAX (Padded)":
            print(
                f"For this specific workload (with {waste_r:.0f}% padding waste), the JAX kernel's raw performance on large matrix multiplies\n"
                f"outweighs the Paged Attention kernel's memory advantage, resulting in {tp_ratio:.1f}x higher projected system throughput."
            )
            print(
                "\nThe Paged Attention kernel would become more performant on workloads with higher sequence length variance\n"
                "(i.e., >50% padding waste) where its memory savings can better offset its higher per-operation latency."
            )
        else:
            print(
                "For this workload, the Paged Attention kernel's ability to avoid wasted computation and pack batches more\n"
                f"densely into memory overcomes its higher per-operation latency, resulting in {tp_ratio:.1f}x higher overall throughput."
            )
        print(
            "\nThis analysis assumes system throughput is limited by K/V cache memory and that attention latency scales with batch size."
        )
        print("=" * 120)

    def _print_performance_projections(self, benchmark_type: str,
                                       latency_ms: float, batch_size: int,
                                       benchmark_q_heads: int,
                                       benchmark_kv_heads: int,
                                       benchmark_dim: int,
                                       implementation_name: str):
        """Prints projected performance, accounting for GQA vs MHA."""
        models = [
            {
                'name': 'Llama 4 Scout (8B MoE)',
                'layers': 24,
                'q_heads': 32,
                'kv_heads': 8,
                'dim': 128
            },
            {
                'name': 'Llama 3 (8B) - GQA',
                'layers': 32,
                'q_heads': 32,
                'kv_heads': 8,
                'dim': 128
            },
            {
                'name': 'DeepSeek-R1 (20B)',
                'layers': 40,
                'q_heads': 40,
                'kv_heads': 8,
                'dim': 128
            },
            {
                'name': 'Llama 4 Maverick (95B)',
                'layers': 64,
                'q_heads': 64,
                'kv_heads': 16,
                'dim': 128
            },
            {
                'name': 'Llama 3 (70B) - GQA',
                'layers': 80,
                'q_heads': 64,
                'kv_heads': 8,
                'dim': 128
            },
        ]

        print(f"\n--- Performance Projections for {implementation_name} ---")
        if benchmark_type == 'decode':
            print(
                f"Based on {latency_ms:.3f} ms latency for a batch of {batch_size}"
            )
            print(
                "\n| Model                   | Est. Full Model Throughput (tokens/sec) |"
            )
            print(
                "| :---------------------- | :-------------------------------------- |"
            )
            for model in models:
                q_scale = model['q_heads'] / benchmark_q_heads
                kv_scale = model['kv_heads'] / benchmark_kv_heads
                scale_factor = q_scale * kv_scale
                full_latency_s = (latency_ms /
                                  1000) * model['layers'] * scale_factor * 3
                throughput = batch_size / full_latency_s if full_latency_s > 0 else 0
                print(
                    f"| {model['name']:<23} | ~{throughput:,.0f} tokens/sec |")
        elif benchmark_type == 'prefill':
            print(
                f"Based on {latency_ms:.3f} ms latency for a batch of {batch_size}"
            )
            print(
                "\n| Model                   | Est. Single-Chip Batch Prefill Time |"
            )
            print(
                "| :---------------------- | :---------------------------------- |"
            )
            for model in models:
                q_scale = model['q_heads'] / benchmark_q_heads
                kv_scale = model['kv_heads'] / benchmark_kv_heads
                scale_factor = q_scale * kv_scale
                prefill_time_ms = latency_ms * model[
                    'layers'] * scale_factor * 3
                print(f"| {model['name']:<23} | ~{prefill_time_ms:.0f} ms |")

    def _run_pallas_benchmark(self, config: Dict[str, Any],
                              benchmark_type: str):
        test_name = self.id().split('.')[-1]
        print(
            f"\n--- Running Perf Benchmark for Pallas Kernel: {test_name} ---")

        inputs = _setup_inputs(**config)
        target_device = jax.devices("tpu")[0]
        inputs_on_device = jax.device_put(inputs, device=target_device)

        num_iterations = 50
        jitted_kernel = jax.jit(paged_attention_kernel.ragged_paged_attention)

        @jax.jit
        def timed_loop(inputs_tuple):

            def loop_body(i, val):
                return jitted_kernel(*inputs_tuple)

            return jax.lax.fori_loop(0, num_iterations - 1, loop_body,
                                     jitted_kernel(*inputs_tuple))

        print(f"[{test_name}] Warming up Pallas Kernel...")
        warmup_output = timed_loop(inputs_on_device)
        warmup_output.block_until_ready()
        print(f"[{test_name}] Warm-up complete.")

        if FLAGS.profile:
            profile_logdir = os.path.join(FLAGS.profile_dir, "pallas",
                                          test_name)
            os.makedirs(profile_logdir, exist_ok=True)
            print(
                f"[{test_name}] Starting profile, saving to: {profile_logdir}")
            with jax.profiler.TraceAnnotation(f"Profile_Pallas_{test_name}"):
                jax.profiler.start_trace(profile_logdir)
                final_output = timed_loop(inputs_on_device)
                final_output.block_until_ready()
                jax.profiler.stop_trace()
            print(f"[{test_name}] Profile saved. View with TensorBoard.")
            # We still need to time outside the profiler block for accurate latency
            start_time = time.perf_counter()
            final_output = timed_loop(inputs_on_device)
            final_output.block_until_ready()
            end_time = time.perf_counter()
        else:
            start_time = time.perf_counter()
            final_output = timed_loop(inputs_on_device)
            final_output.block_until_ready()
            end_time = time.perf_counter()

        avg_latency_ms = (end_time - start_time) / num_iterations * 1000
        print("\n[Result] Implementation: Pallas Kernel (on Ragged Inputs)")
        print(f"[Result] Average Latency: {avg_latency_ms:.3f} ms")

        if "Ragged" in test_name and benchmark_type == "decode":
            total_query_tokens = sum(s[0] for s in config['seq_lens'])
            effective_throughput = total_query_tokens / (avg_latency_ms / 1000)

            _, num_kv_heads = config["num_heads"]
            head_dim, dtype = config["head_dim"], config["dtype"]
            dtype_bytes = jnp.dtype(dtype).itemsize
            total_kv_tokens = sum(s[1] for s in config['seq_lens'])
            kv_cache_mem_mb = (total_kv_tokens * num_kv_heads * head_dim *
                               dtype_bytes) / (1024**2)

            self._benchmark_results.append({
                "name": test_name,
                "type": "Pallas (Ragged)",
                "latency_ms": avg_latency_ms,
                "effective_throughput_tok_s": effective_throughput,
                "kv_cache_mem_mb": kv_cache_mem_mb,
                "wasted_computation_pct": 0.0,
            })

        self._print_performance_projections(benchmark_type, avg_latency_ms,
                                            len(config['seq_lens']),
                                            config['num_heads'][0],
                                            config['num_heads'][1],
                                            config['head_dim'],
                                            "Pallas Kernel")

    def _run_reference_benchmark(self, config: Dict[str, Any],
                                 benchmark_type: str):
        test_name = self.id().split('.')[-1]

        seq_lens = config["seq_lens"]
        is_ragged = len(set(s for s in seq_lens)) > 1

        if is_ragged:
            print(
                f"\n--- Running Perf Benchmark for JAX Reference on Ragged (Padded) Batch: {test_name} ---"
            )
        else:
            print(
                f"\n--- Running Perf Benchmark for JAX Reference: {test_name} ---"
            )

        num_q_heads, num_kv_heads = config["num_heads"]
        head_dim, dtype = config["head_dim"], config["dtype"]
        batch_size = len(seq_lens)

        # Determine padding from sequence lengths
        max_q_len = max(s[0] for s in seq_lens)
        max_kv_len = max(s[1] for s in seq_lens)

        prng_key = jax.random.key(4321)
        k_q, k_k, k_v = jax.random.split(prng_key, 3)

        q = jax.random.normal(k_q,
                              (batch_size, max_q_len, num_q_heads, head_dim),
                              dtype=dtype)
        k = jax.random.normal(k_k,
                              (batch_size, max_kv_len, num_kv_heads, head_dim),
                              dtype=dtype)
        v = jax.random.normal(k_v,
                              (batch_size, max_kv_len, num_kv_heads, head_dim),
                              dtype=dtype)

        num_iterations = 50

        if is_ragged:
            # Create attention mask for raggedness and causality
            q_lens_arr = jnp.array([s[0] for s in seq_lens],
                                   dtype=jnp.int32).reshape(batch_size, 1, 1)
            kv_lens_arr = jnp.array([s[1] for s in seq_lens],
                                    dtype=jnp.int32).reshape(batch_size, 1, 1)

            q_indices = jnp.arange(max_q_len).reshape(1, max_q_len, 1)
            kv_indices = jnp.arange(max_kv_len).reshape(1, 1, max_kv_len)

            # Mask for padding on Q and K/V
            padding_mask = (q_indices >= q_lens_arr) | (kv_indices
                                                        >= kv_lens_arr)

            # Causal mask
            q_positions = (kv_lens_arr - q_lens_arr) + q_indices
            causal_mask = kv_indices > q_positions

            attention_mask = jnp.logical_or(
                padding_mask, causal_mask)[:, None, :, :]  # Add head dim

            inputs_on_device = jax.device_put((q, k, v, attention_mask),
                                              device=jax.devices("tpu")[0])
            jitted_kernel = reference_attention_dense_masked_jit

            @jax.jit
            def timed_loop(q_d, k_d, v_d, mask_d):

                def loop_body(i, val):
                    return jitted_kernel(q_d, k_d, v_d, mask_d)

                return jax.lax.fori_loop(0, num_iterations - 1, loop_body,
                                         jitted_kernel(q_d, k_d, v_d, mask_d))
        else:
            # Original logic for uniform batches
            inputs_on_device = jax.device_put((q, k, v),
                                              device=jax.devices("tpu")[0])
            jitted_kernel = reference_attention_dense_jit

            @jax.jit
            def timed_loop(q_d, k_d, v_d):

                def loop_body(i, val):
                    return jitted_kernel(q_d, k_d, v_d)

                return jax.lax.fori_loop(0, num_iterations - 1, loop_body,
                                         jitted_kernel(q_d, k_d, v_d))

        print(f"[{test_name}] Warming up JAX reference kernel...")
        warmup_output = timed_loop(*inputs_on_device)
        warmup_output.block_until_ready()
        print(f"[{test_name}] Warm-up complete.")

        if FLAGS.profile:
            profile_logdir = os.path.join(FLAGS.profile_dir, "reference",
                                          test_name)
            os.makedirs(profile_logdir, exist_ok=True)
            print(
                f"[{test_name}] Starting profile, saving to: {profile_logdir}")
            with jax.profiler.TraceAnnotation(
                    f"Profile_Reference_{test_name}"):
                jax.profiler.start_trace(profile_logdir)
                final_output = timed_loop(*inputs_on_device)
                final_output.block_until_ready()
                jax.profiler.stop_trace()
            print(f"[{test_name}] Profile saved. View with TensorBoard.")
            # Time outside profiler block
            start_time = time.perf_counter()
            final_output = timed_loop(*inputs_on_device)
            final_output.block_until_ready()
            end_time = time.perf_counter()
        else:
            start_time = time.perf_counter()
            final_output = timed_loop(*inputs_on_device)
            final_output.block_until_ready()
            end_time = time.perf_counter()

        avg_latency_ms = (end_time - start_time) / num_iterations * 1000

        if is_ragged:
            total_tokens = sum(s[0] for s in seq_lens)
            effective_throughput = total_tokens / (avg_latency_ms / 1000)
            print(
                "\n[Result] Implementation: JAX Reference (on Padded Dense Inputs)"
            )
            print(
                f"[Result] Average Latency: {avg_latency_ms:.3f} ms (for padded shape {q.shape})"
            )
            print(f"[Result] Total non-pad query tokens: {total_tokens}")
            print(
                f"[Result] Effective Throughput: {effective_throughput:,.0f} tokens/sec"
            )

            dtype_bytes = jnp.dtype(dtype).itemsize
            padded_kv_tokens = batch_size * max_kv_len
            kv_cache_mem_mb = (padded_kv_tokens * num_kv_heads * head_dim *
                               dtype_bytes) / (1024**2)
            actual_kv_tokens = sum(s[1] for s in seq_lens)
            wasted_computation_pct = (
                padded_kv_tokens - actual_kv_tokens
            ) / padded_kv_tokens * 100 if padded_kv_tokens > 0 else 0

            self._benchmark_results.append({
                "name":
                test_name,
                "type":
                "JAX (Padded)",
                "latency_ms":
                avg_latency_ms,
                "effective_throughput_tok_s":
                effective_throughput,
                "kv_cache_mem_mb":
                kv_cache_mem_mb,
                "wasted_computation_pct":
                wasted_computation_pct,
            })

        else:
            print("\n[Result] Implementation: JAX Reference (on Dense Inputs)")
            print(f"[Result] Average Latency: {avg_latency_ms:.3f} ms")
            self._print_performance_projections(benchmark_type, avg_latency_ms,
                                                len(config['seq_lens']),
                                                config['num_heads'][0],
                                                config['num_heads'][1],
                                                config['head_dim'],
                                                "JAX Reference")

    def _run_accuracy_test(self, config: Dict[str, Any]):
        test_name = self.id().split('.')[-1]
        print(f"\n--- Running Accuracy Validation: {test_name} ---")

        inputs = _setup_inputs(**config)

        # Accuracy tests are usually quick, so profiling might not be as critical
        # but can be added if needed.
        print("[Accuracy] Running Pallas kernel...")
        pallas_output = paged_attention_kernel.ragged_paged_attention(
            *inputs).block_until_ready()

        print("[Accuracy] Running internal Python-loop reference kernel...")
        reference_output = paged_attention_kernel.ref_ragged_paged_attention(
            *inputs).block_until_ready()

        print("[Accuracy] Comparing outputs...")
        are_close = jnp.allclose(pallas_output,
                                 reference_output,
                                 rtol=1e-2,
                                 atol=1e-2)

        self.assertTrue(
            are_close,
            "Validation FAILED: Pallas kernel output does not match reference."
        )

        mae = jnp.mean(jnp.abs(pallas_output - reference_output))
        print("\n[Result] ✅ Validation PASSED!")
        print(f"[Result] Mean Absolute Error: {mae.item():.6f}")

    # --- 4. Parameterized Test Definitions ---
    @parameterized.named_parameters(
        {
            "testcase_name": "Accuracy_Decode",
            "test_type": "accuracy",
            "benchmark_type": "decode",
            "config": {
                "seq_lens": [(1, 1024)],
                "num_heads": (32, 8),
                "head_dim": 128,
                "page_size": 16,
                "dtype": jnp.bfloat16,
                "num_pages": 1024,
            },
        },
        {
            "testcase_name": "Accuracy_Prefill",
            "test_type": "accuracy",
            "benchmark_type": "prefill",
            "config": {
                "seq_lens": [(512, 512)],
                "num_heads": (32, 8),
                "head_dim": 128,
                "page_size": 16,
                "dtype": jnp.bfloat16,
                "num_pages": 1024,
            },
        },
        {
            "testcase_name": "Perf_Pallas_Decode",
            "test_type": "pallas_perf",
            "benchmark_type": "decode",
            "config": {
                "seq_lens": [(1, 2048)] * 64,
                "num_heads": (32, 8),
                "head_dim": 128,
                "page_size": 16,
                "dtype": jnp.bfloat16,
                "num_pages": 8192,
            },
        },
        {
            "testcase_name": "Perf_Reference_Decode",
            "test_type": "reference_perf",
            "benchmark_type": "decode",
            "config": {
                "seq_lens": [(1, 2048)] * 64,
                "num_heads": (32, 8),
                "head_dim": 128,
                "page_size": 16,
                "dtype": jnp.bfloat16,
                "num_pages": 8192,
            },
        },
        {
            "testcase_name": "Perf_Pallas_Prefill",
            "test_type": "pallas_perf",
            "benchmark_type": "prefill",
            "config": {
                "seq_lens": [(1024, 1024)] * 8,
                "num_heads": (32, 8),
                "head_dim": 128,
                "page_size": 16,
                "dtype": jnp.bfloat16,
                "num_pages": 8192,
            },
        },
        {
            "testcase_name": "Perf_Reference_Prefill",
            "test_type": "reference_perf",
            "benchmark_type": "prefill",
            "config": {
                "seq_lens": [(1024, 1024)] * 8,
                "num_heads": (32, 8),
                "head_dim": 128,
                "page_size": 16,
                "dtype": jnp.bfloat16,
                "num_pages": 8192,
            },
        },
        {
            "testcase_name": "Perf_Pallas_Decode_Large_Batch",
            "test_type": "pallas_perf",
            "benchmark_type": "decode",
            "config": {
                "seq_lens": [(1, 2048)] * 256,
                "num_heads": (32, 8),
                "head_dim": 128,
                "page_size": 16,
                "dtype": jnp.bfloat16,
                "num_pages": 8192,
            },
        },
        {
            "testcase_name": "Perf_Pallas_Decode_Long_Sequence",
            "test_type": "pallas_perf",
            "benchmark_type": "decode",
            "config": {
                "seq_lens": [(1, 8192)] * 64,
                "num_heads": (32, 8),
                "head_dim": 128,
                "page_size": 16,
                "dtype": jnp.bfloat16,
                "num_pages": 8192,
            },
        },
        {
            "testcase_name": "Perf_Pallas_Prefill_Large_Batch",
            "test_type": "pallas_perf",
            "benchmark_type": "prefill",
            "config": {
                "seq_lens": [(1024, 1024)] * 32,
                "num_heads": (32, 8),
                "head_dim": 128,
                "page_size": 16,
                "dtype": jnp.bfloat16,
                "num_pages": 8192,
            },
        },
        {
            "testcase_name": "Perf_Pallas_Decode_Ragged",
            "test_type": "pallas_perf",
            "benchmark_type": "decode",
            "config": {
                "seq_lens": [(1, 512)] * 32 + [(1, 1024)] * 16 +
                [(1, 2048)] * 16,  # Total batch size 64
                "num_heads": (32, 8),
                "head_dim": 128,
                "page_size": 16,
                "dtype": jnp.bfloat16,
                "num_pages": 8192,
            },
        },
        {
            "testcase_name": "Perf_Reference_Decode_Ragged",
            "test_type": "reference_perf",
            "benchmark_type": "decode",
            "config": {
                "seq_lens": [(1, 512)] * 32 + [(1, 1024)] * 16 +
                [(1, 2048)] * 16,  # Total batch size 64
                "num_heads": (32, 8),
                "head_dim": 128,
                "page_size": 16,
                "dtype": jnp.bfloat16,
                "num_pages": 8192,
            },
        },
    )
    def test_paged_attention(self, test_type: str, benchmark_type: str,
                             config: Dict[str, Any]):
        if not jax.devices("tpu"):
            self.skipTest("This benchmark is designed for TPU.")

        try:
            if test_type == "accuracy":
                self._run_accuracy_test(config=config)
            elif test_type == "pallas_perf":
                self._run_pallas_benchmark(config=config,
                                           benchmark_type=benchmark_type)
            elif test_type == "reference_perf":
                self._run_reference_benchmark(config=config,
                                              benchmark_type=benchmark_type)
            else:
                raise ValueError(f"Unknown test type: {test_type}")
        except Exception as e:
            print(f"\n--- ERROR during test: {self.id()} ---")
            traceback.print_exc()
            self.fail(f"Test failed with exception: {e}")


if __name__ == "__main__":
    absltest.main()
