# Copyright 2026 Google LLC
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    ragged_paged_attention, ref_ragged_paged_attention)
from tpu_inference.kernels.ragged_paged_attention.v3.per_token_scale_kernel import \
    ref_ragged_paged_attention_per_token_non_jit
from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing)
from tpu_inference.layers.common.quantization import quantize_kv


# Helper to print diff
def print_diff(name, actual, baseline):
    # Flatten
    a = actual.reshape(-1).astype(jnp.float32)
    b = baseline.reshape(-1).astype(jnp.float32)

    # MSE
    mse = jnp.mean(jnp.square(a - b))

    # Relative Error
    norm_diff = jnp.linalg.norm(a - b)
    norm_base = jnp.linalg.norm(b) + 1e-6
    rel_err = norm_diff / norm_base

    print(f"\n--- {name} ---")
    print(f"MSE: {mse:.8f}")
    print(f"Rel: {rel_err:.8f}")


seq_lens = [(192, 328), (128, 180), (64, 255)]
num_heads = (32, 8)
head_dim = 128
page_size = 16
num_pages = 1000
max_num_batched_tokens = 4096
max_num_seq = 512
# 1. Generate in High Precision First
q_dtype = jnp.bfloat16
kv_dtype_storage = jnp.float4_e2m1fn

rng = np.random.default_rng(1234)


def gen_random(shape, dtype):
    return jnp.array(rng.random(size=shape, dtype=np.float32)).astype(dtype)


cu_q_lens = [0]
kv_lens = []
for q_len, kv_len in seq_lens:
    assert q_len <= kv_len
    cu_q_lens.append(cu_q_lens[-1] + q_len)
    kv_lens.append(kv_len)

max_num_batched_tokens = max(align_to(cu_q_lens[-1], 128),
                             max_num_batched_tokens)
max_num_seq = max(align_to(len(seq_lens), 8), max_num_seq)
max_kv_len = max(kv_lens)
pages_per_seq = cdiv(max_kv_len, page_size)
num_q_heads, num_kv_heads = num_heads

# Generate Inputs in BF16
q = gen_random((max_num_batched_tokens, num_q_heads, head_dim), q_dtype)
# k = gen_random((max_num_batched_tokens, num_kv_heads, head_dim), jnp.bfloat16)
# v = gen_random((max_num_batched_tokens, num_kv_heads, head_dim), jnp.bfloat16)


def gen_outliers(shape, dtype):
    # Standard normal data (most values small)
    data = rng.standard_normal(size=shape).astype(np.float32)

    # Inject massive outliers into sporadic tokens/heads
    # This forces Per-Tensor to squash the 'normal' data to zero,
    # causing high error. Per-Token will adapt.
    mask = rng.random(size=shape) > 0.99
    data = data + (mask * 100.0)

    return jnp.array(data).astype(dtype)


k = gen_outliers((max_num_batched_tokens, num_kv_heads, head_dim),
                 jnp.bfloat16)
v = gen_outliers((max_num_batched_tokens, num_kv_heads, head_dim),
                 jnp.bfloat16)

# --- CACHE GENERATION ---
# We generate BF16 history, then quantize it for the quantized run.

page_cnt = 0
page_indices_list = []
kv_pages_bf16_list = []
kv_pages_quant_list = []
kv_packing = get_dtype_packing(kv_dtype_storage)
padded_head_dim = align_to(head_dim, 128)
num_kv_heads_x2 = align_to(num_kv_heads * 2, kv_packing)

# 2. Fix Scale Initialization
# Initialize scale caches to 1.0 (identity) so history doesn't explode
# Or 0.0 if you assume empty cache, but here we fill it.
# Ideally, we should compute scales for the history, but for simplicity
# in this test script, we will initialize history scales to 1.0.
k_scale_cache = jnp.ones((num_pages, page_size, num_kv_heads, 1),
                         dtype=jnp.float32)
v_scale_cache = jnp.ones((num_pages, page_size, num_kv_heads, 1),
                         dtype=jnp.float32)

for kv_len in kv_lens:
    # Generate High Precision Page Data
    kv_bf16_unpacked = gen_random(
        (
            kv_len,
            num_kv_heads,
            2,  # K and V
            head_dim),
        jnp.bfloat16)

    # Pack for storage (BF16 baseline usually expects specific layout, assuming standard here)
    # Pading to page size
    pad_len = cdiv(kv_len, page_size) * page_size - kv_len
    kv_bf16_padded = jnp.pad(kv_bf16_unpacked,
                             ((0, pad_len), (0, 0), (0, 0), (0, 0)))

    # Reshape for page list
    kv_page_bf16 = kv_bf16_padded.reshape(-1, page_size, num_kv_heads, 2,
                                          head_dim)

    # Create Quantized Version for the Test Subject
    # Note: Simplification - we are casting HF->Quant directly.
    # Realistically you should calculate scales here if populating cache.
    # For this test, we assume history is within range.
    kv_page_quant = kv_page_bf16.astype(kv_dtype_storage)

    # Reshape logic for packed storage
    # [num_pages, page_size, num_heads, 2, head_dim] -> [..., num_heads*2, head_dim]
    temp = kv_page_quant.reshape(kv_page_quant.shape[0], page_size,
                                 num_kv_heads * 2, head_dim)
    # Pack dims
    temp = temp.reshape(temp.shape[0], page_size,
                        num_kv_heads_x2 // kv_packing, kv_packing, head_dim)

    indices = page_cnt + jnp.arange(kv_page_bf16.shape[0], dtype=jnp.int32)
    indices = jnp.pad(indices, ((0, pages_per_seq - indices.shape[0]), ))

    page_indices_list.append(indices)
    page_cnt += kv_page_bf16.shape[0]

    kv_pages_bf16_list.append(kv_page_bf16)
    kv_pages_quant_list.append(temp)

# Concatenate Caches
kv_cache_bf16 = jnp.concatenate(kv_pages_bf16_list, axis=0)
kv_cache_bf16 = jnp.pad(kv_cache_bf16,
                        ((0, num_pages - kv_cache_bf16.shape[0]), (0, 0),
                         (0, 0), (0, 0), (0, 0)))

kv_cache_quant = jnp.concatenate(kv_pages_quant_list, axis=0)
kv_cache_quant = jnp.pad(kv_cache_quant,
                         ((0, num_pages - kv_cache_quant.shape[0]), (0, 0),
                          (0, 0), (0, 0), (0, 0)))

# Indices setup
page_indices = jnp.stack(page_indices_list, axis=0)
page_indices = jnp.pad(page_indices,
                       ((0, max_num_seq - page_indices.shape[0]), (0, 0)))
page_indices = page_indices.reshape(-1)

cu_q_lens = jnp.array(cu_q_lens, dtype=jnp.int32)
cu_q_lens = jnp.pad(cu_q_lens, (0, max_num_seq + 1 - cu_q_lens.shape[0]))
kv_lens = jnp.array(kv_lens, dtype=jnp.int32)
kv_lens = jnp.pad(kv_lens, (0, max_num_seq - kv_lens.shape[0]))
distribution = jnp.array([0, 0, len(seq_lens)], dtype=jnp.int32)

# --- RUN 1: BASELINE (BF16) ---
# This is the "Ground Truth"
print("Running BF16 Baseline...")
# Assuming ref_ragged_paged_attention handles the BF16 unpacked layout or adjust layout as needed by your kernel
# If the kernel expects packed layout even for BF16, you need to adjust kv_cache_bf16 shape
# For this example, assuming the reference kernel can take [P, S, H, 2, D]
args_base = (q, k, v, kv_cache_bf16, kv_lens, page_indices, cu_q_lens,
             distribution)
baseline_out, _ = ref_ragged_paged_attention(*args_base)

# --- RUN 2: PER-TOKEN SCALING ---
print("Running Per-Token Quantized...")
# Note: k/v inputs are BF16, kernel quantizes them internally
args_pt = (
    q,
    k,
    v,
    kv_cache_quant,  # Initialized with quantized data
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    k_scale_cache,
    v_scale_cache  # Initialized to 1.0
)
pt_out, _, _, _ = ref_ragged_paged_attention_per_token_non_jit(*args_pt)

print_diff("Per-Token vs Baseline", pt_out, baseline_out)

# --- RUN 3: PER-TENSOR SCALING ---
print("Running Per-Tensor Quantized (Ref)...")

# Quantize inputs using global scale (1.0 for random 0-1 data is fine)
k_q, v_q = quantize_kv(kv_dtype_storage, k, v, 1.0, 1.0)

args_tensor = (
    q,
    k_q,
    v_q,
    kv_cache_quant,  # Same quantized cache
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution)
# Note: Explicitly passing k_scale/v_scale is important if they are not None defaults
tensor_out, _ = ref_ragged_paged_attention(*args_tensor,
                                           k_scale=1.0,
                                           v_scale=1.0)

print_diff("Per-Tensor vs Baseline", tensor_out, baseline_out)

# --- RUN 4: PER-TENSOR KERNEL ---
print("Running Per-Tensor Kernel (Actual)...")
# Ensure inputs match the reference run
kernel_out, _ = ragged_paged_attention(*args_tensor, k_scale=1.0, v_scale=1.0)

print_diff("Per-Tensor Kernel vs Baseline", kernel_out, baseline_out)
# print_diff("Per-Tensor Kernel vs Ref", kernel_out, tensor_out) # sanity check implementation
