#!/usr/bin/env python3
"""Compare JAX vs PyTorch implementations of MLA attention.

This script verifies that both implementations produce identical results
for the same random inputs, helping debug numerical discrepancies.

Run with: python -m pytest tests/mla_compare_jax_torch.py -v -s
Or directly: python tests/mla_compare_jax_torch.py
"""

import numpy as np
import torch
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any

# Force CPU execution for reproducibility
jax.config.update('jax_platform_name', 'cpu')

# GLM-4 MLA dimensions
GLM4_CONFIG = {
    'num_heads': 20,
    'qk_nope_head_dim': 192,
    'qk_rope_head_dim': 64,
    'v_head_dim': 256,
    'kv_lora_rank': 512,
    'hidden_size': 4096,
}

def create_test_inputs(
    num_tokens: int = 4,
    config: dict = GLM4_CONFIG,
    seed: int = 42,
    dtype_np: np.dtype = np.float32,
) -> Dict[str, np.ndarray]:
    """Create random test inputs with fixed seed."""
    np.random.seed(seed)

    num_heads = config['num_heads']
    qk_nope = config['qk_nope_head_dim']
    qk_rope = config['qk_rope_head_dim']
    v_head = config['v_head_dim']
    kv_lora = config['kv_lora_rank']
    qk_head = qk_nope + qk_rope

    # Query: (num_tokens, num_heads, qk_head_dim)
    q = np.random.randn(num_tokens, num_heads, qk_head).astype(dtype_np)

    # Compressed KV (already normalized): (num_tokens, kv_lora_rank)
    kv_c_normed = np.random.randn(num_tokens, kv_lora).astype(dtype_np)

    # K position embeddings (already RoPE'd): (num_tokens, qk_rope_head_dim)
    k_pe = np.random.randn(num_tokens, qk_rope).astype(dtype_np)

    # kv_b_proj weight: (num_heads * (qk_nope + v_head), kv_lora_rank)
    # Layout: [head0_K, head0_V, head1_K, head1_V, ...]
    kv_b_proj_weight = np.random.randn(
        num_heads * (qk_nope + v_head), kv_lora
    ).astype(dtype_np) * 0.02  # Small init like transformers

    # Scale factor
    scale = 1.0 / np.sqrt(qk_head)

    return {
        'q': q,
        'kv_c_normed': kv_c_normed,
        'k_pe': k_pe,
        'kv_b_proj_weight': kv_b_proj_weight,
        'scale': scale,
        'config': config,
    }


def mla_attention_torch(
    q: torch.Tensor,  # (num_tokens, num_heads, qk_head_dim)
    kv_c_normed: torch.Tensor,  # (num_tokens, kv_lora_rank)
    k_pe: torch.Tensor,  # (num_tokens, qk_rope_head_dim)
    kv_b_proj_weight: torch.Tensor,  # (num_heads * (qk_nope + v), kv_lora_rank)
    scale: float,
    config: dict,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """PyTorch implementation of MLA attention (decode mode / data-movement friendly).

    Returns:
        output: (num_tokens, num_heads, v_head_dim)
        intermediates: Dict of intermediate values for debugging
    """
    num_heads = config['num_heads']
    qk_nope = config['qk_nope_head_dim']
    v_head = config['v_head_dim']
    kv_lora = config['kv_lora_rank']

    num_tokens = q.shape[0]
    intermediates = {}

    # Step 1: Split Q into nope and pe parts
    q_nope = q[..., :qk_nope]  # (num_tokens, num_heads, qk_nope)
    q_pe = q[..., qk_nope:]    # (num_tokens, num_heads, qk_rope)
    intermediates['q_nope'] = q_nope
    intermediates['q_pe'] = q_pe

    # Step 2: Extract W_K and W_V from kv_b_proj
    # Reshape: (num_heads * (qk_nope + v), kv_lora) -> (num_heads, qk_nope + v, kv_lora)
    kv_b_reshaped = kv_b_proj_weight.view(num_heads, qk_nope + v_head, kv_lora)
    w_k = kv_b_reshaped[:, :qk_nope, :]  # (num_heads, qk_nope, kv_lora)
    w_v = kv_b_reshaped[:, qk_nope:, :]  # (num_heads, v_head, kv_lora)
    intermediates['w_k'] = w_k
    intermediates['w_v'] = w_v

    # Step 3: Project q_nope to latent space: ql_nope = q_nope @ W_K
    # einsum: (tokens, heads, qk_nope) @ (heads, qk_nope, kv_lora) -> (tokens, heads, kv_lora)
    ql_nope = torch.einsum('thd,hdk->thk', q_nope, w_k)
    intermediates['ql_nope'] = ql_nope

    # Step 4: Concatenate Q = [ql_nope, q_pe]
    q_combined = torch.cat([ql_nope, q_pe], dim=-1)  # (tokens, heads, kv_lora + qk_rope)
    intermediates['q_combined'] = q_combined

    # Step 5: Concatenate K = [kv_c, k_pe]
    k_combined = torch.cat([kv_c_normed, k_pe], dim=-1)  # (tokens, kv_lora + qk_rope)
    intermediates['k_combined'] = k_combined

    # Step 6: Compute attention scores
    # Q: (tokens, heads, dim), K: (tokens, dim) -> attn: (heads, tokens, tokens)
    attn_scores = torch.einsum('qnh,kh->nqk', q_combined, k_combined)  # (heads, q_tokens, k_tokens)
    attn_scores = attn_scores * scale
    intermediates['attn_scores_raw'] = attn_scores

    # Step 7: Apply causal mask
    causal_mask = torch.tril(torch.ones(num_tokens, num_tokens, device=q.device))
    attn_scores = attn_scores.masked_fill(causal_mask[None, :, :] == 0, float('-inf'))
    intermediates['attn_scores_masked'] = attn_scores

    # Step 8: Softmax
    attn_weights = torch.softmax(attn_scores, dim=-1)
    intermediates['attn_weights'] = attn_weights

    # Step 9: Compute output in latent space: output_latent = attn @ kv_c
    # attn: (heads, q_tokens, k_tokens), kv_c: (tokens, kv_lora) -> (tokens, heads, kv_lora)
    output_latent = torch.einsum('nqk,kl->qnl', attn_weights, kv_c_normed)
    intermediates['output_latent'] = output_latent

    # Step 10: Project to value space: output = output_latent @ W_V^T
    # einsum: (tokens, heads, kv_lora) @ (heads, v_head, kv_lora)^T -> (tokens, heads, v_head)
    output = torch.einsum('thk,hvk->thv', output_latent, w_v)
    intermediates['output'] = output

    return output, intermediates


def mla_attention_jax(
    q: jnp.ndarray,  # (num_tokens, num_heads, qk_head_dim)
    kv_c_normed: jnp.ndarray,  # (num_tokens, kv_lora_rank)
    k_pe: jnp.ndarray,  # (num_tokens, qk_rope_head_dim)
    kv_b_proj_weight: jnp.ndarray,  # (num_heads * (qk_nope + v), kv_lora_rank)
    scale: float,
    config: dict,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """JAX implementation of MLA attention (decode mode / data-movement friendly).

    Returns:
        output: (num_tokens, num_heads, v_head_dim)
        intermediates: Dict of intermediate values for debugging
    """
    num_heads = config['num_heads']
    qk_nope = config['qk_nope_head_dim']
    qk_rope = config['qk_rope_head_dim']
    v_head = config['v_head_dim']
    kv_lora = config['kv_lora_rank']

    num_tokens = q.shape[0]
    intermediates = {}

    # Step 1: Split Q into nope and pe parts
    q_nope = q[..., :qk_nope]  # (num_tokens, num_heads, qk_nope)
    q_pe = q[..., qk_nope:]    # (num_tokens, num_heads, qk_rope)
    intermediates['q_nope'] = q_nope
    intermediates['q_pe'] = q_pe

    # Step 2: Extract W_K and W_V from kv_b_proj
    # Reshape: (num_heads * (qk_nope + v), kv_lora) -> (num_heads, qk_nope + v, kv_lora)
    kv_b_reshaped = kv_b_proj_weight.reshape(num_heads, qk_nope + v_head, kv_lora)
    w_k = kv_b_reshaped[:, :qk_nope, :]  # (num_heads, qk_nope, kv_lora)
    w_v = kv_b_reshaped[:, qk_nope:, :]  # (num_heads, v_head, kv_lora)
    intermediates['w_k'] = w_k
    intermediates['w_v'] = w_v

    # Step 3: Project q_nope to latent space: ql_nope = q_nope @ W_K
    # einsum: (tokens, heads, qk_nope) @ (heads, qk_nope, kv_lora) -> (tokens, heads, kv_lora)
    ql_nope = jnp.einsum('thd,hdk->thk', q_nope, w_k)
    intermediates['ql_nope'] = ql_nope

    # Step 4: Concatenate Q = [ql_nope, q_pe]
    q_combined = jnp.concatenate([ql_nope, q_pe], axis=-1)  # (tokens, heads, kv_lora + qk_rope)
    intermediates['q_combined'] = q_combined

    # Step 5: Concatenate K = [kv_c, k_pe]
    k_combined = jnp.concatenate([kv_c_normed, k_pe], axis=-1)  # (tokens, kv_lora + qk_rope)
    intermediates['k_combined'] = k_combined

    # Step 6: Compute attention scores
    # Q: (tokens, heads, dim), K: (tokens, dim) -> attn: (heads, tokens, tokens)
    attn_scores = jnp.einsum('qnh,kh->nqk', q_combined, k_combined)  # (heads, q_tokens, k_tokens)
    attn_scores = attn_scores * scale
    intermediates['attn_scores_raw'] = attn_scores

    # Step 7: Apply causal mask
    causal_mask = jnp.tril(jnp.ones((num_tokens, num_tokens)))
    attn_scores = jnp.where(causal_mask[None, :, :] == 0, -1e9, attn_scores)
    intermediates['attn_scores_masked'] = attn_scores

    # Step 8: Softmax
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    intermediates['attn_weights'] = attn_weights

    # Step 9: Compute output in latent space: output_latent = attn @ kv_c
    # attn: (heads, q_tokens, k_tokens), kv_c: (tokens, kv_lora) -> (tokens, heads, kv_lora)
    output_latent = jnp.einsum('nqk,kl->qnl', attn_weights, kv_c_normed)
    intermediates['output_latent'] = output_latent

    # Step 10: Project to value space: output = output_latent @ W_V^T
    # einsum: (tokens, heads, kv_lora) @ (heads, v_head, kv_lora)^T -> (tokens, heads, v_head)
    output = jnp.einsum('thk,hvk->thv', output_latent, w_v)
    intermediates['output'] = output

    return output, intermediates


def compare_values(
    name: str,
    torch_val: torch.Tensor,
    jax_val: jnp.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    allow_mask_value_diff: bool = False,
) -> Tuple[bool, str]:
    """Compare torch and jax values, return (match, message)."""
    torch_np = torch_val.detach().cpu().numpy()
    jax_np = np.array(jax_val)

    if torch_np.shape != jax_np.shape:
        return False, f"Shape mismatch: torch={torch_np.shape}, jax={jax_np.shape}"

    # Handle comparison with masked values (inf vs -1e9)
    if allow_mask_value_diff:
        # Replace -inf with -1e9 in torch for comparison
        torch_np_compare = np.where(np.isinf(torch_np), -1e9, torch_np)
        # Replace large negative values in jax with -1e9 for consistency
        jax_np_compare = np.where(jax_np < -1e8, -1e9, jax_np)

        # Compare finite values only for diff calculation
        finite_mask = np.isfinite(torch_np) & np.isfinite(jax_np)
        if np.any(finite_mask):
            finite_diff = np.abs(torch_np[finite_mask] - jax_np[finite_mask])
            max_diff = np.max(finite_diff) if finite_diff.size > 0 else 0.0
            mean_diff = np.mean(finite_diff) if finite_diff.size > 0 else 0.0
        else:
            max_diff = 0.0
            mean_diff = 0.0

        matches = np.allclose(torch_np_compare, jax_np_compare, rtol=rtol, atol=atol)
    else:
        max_diff = np.max(np.abs(torch_np - jax_np))
        mean_diff = np.mean(np.abs(torch_np - jax_np))
        matches = np.allclose(torch_np, jax_np, rtol=rtol, atol=atol)

    status = "✓" if matches else "✗"
    msg = f"{status} {name}: shape={torch_np.shape}, max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"

    if not matches:
        # Show sample values
        msg += f"\n    torch sample: {torch_np.flatten()[:5]}"
        msg += f"\n    jax sample:   {jax_np.flatten()[:5]}"

    return matches, msg


def run_comparison(num_tokens: int = 4, seed: int = 42, dtype: str = 'float32'):
    """Run full comparison between PyTorch and JAX implementations."""
    print(f"\n{'='*70}")
    print(f"MLA Attention Comparison: JAX vs PyTorch")
    print(f"num_tokens={num_tokens}, seed={seed}, dtype={dtype}")
    print(f"{'='*70}\n")

    # Create test inputs
    dtype_np = np.float32 if dtype == 'float32' else np.float16
    inputs = create_test_inputs(num_tokens=num_tokens, seed=seed, dtype_np=dtype_np)
    config = inputs['config']

    print(f"Config: {config}\n")
    print(f"Scale: {inputs['scale']:.6f}\n")

    # Convert to torch
    q_torch = torch.from_numpy(inputs['q'])
    kv_c_torch = torch.from_numpy(inputs['kv_c_normed'])
    k_pe_torch = torch.from_numpy(inputs['k_pe'])
    kv_b_torch = torch.from_numpy(inputs['kv_b_proj_weight'])

    # Convert to jax
    q_jax = jnp.array(inputs['q'])
    kv_c_jax = jnp.array(inputs['kv_c_normed'])
    k_pe_jax = jnp.array(inputs['k_pe'])
    kv_b_jax = jnp.array(inputs['kv_b_proj_weight'])

    # Run both implementations
    print("Running PyTorch implementation...")
    output_torch, intermediates_torch = mla_attention_torch(
        q_torch, kv_c_torch, k_pe_torch, kv_b_torch, inputs['scale'], config
    )

    print("Running JAX implementation...")
    output_jax, intermediates_jax = mla_attention_jax(
        q_jax, kv_c_jax, k_pe_jax, kv_b_jax, inputs['scale'], config
    )

    # Compare all intermediate values
    print("\n" + "-"*70)
    print("Comparing intermediate values:")
    print("-"*70)

    all_match = True
    comparison_order = [
        'q_nope', 'q_pe', 'w_k', 'w_v', 'ql_nope',
        'q_combined', 'k_combined', 'attn_scores_raw',
        'attn_scores_masked', 'attn_weights', 'output_latent', 'output'
    ]

    # Fields where mask values (-inf vs -1e9) are acceptable
    mask_value_fields = {'attn_scores_masked'}

    for key in comparison_order:
        if key in intermediates_torch and key in intermediates_jax:
            matches, msg = compare_values(
                key, intermediates_torch[key], intermediates_jax[key],
                allow_mask_value_diff=(key in mask_value_fields)
            )
            print(msg)
            if not matches:
                all_match = False

    # Final summary
    print("\n" + "="*70)
    if all_match:
        print("✓ SUCCESS: All values match between PyTorch and JAX!")
    else:
        print("✗ FAILURE: Some values do not match!")
    print("="*70 + "\n")

    return all_match, output_torch, output_jax, intermediates_torch, intermediates_jax


def test_mla_comparison_float32():
    """Test MLA comparison with float32."""
    all_match, _, _, _, _ = run_comparison(num_tokens=4, seed=42, dtype='float32')
    assert all_match, "PyTorch and JAX implementations should match for float32"


def test_mla_comparison_different_seq_lengths():
    """Test MLA with different sequence lengths."""
    for num_tokens in [1, 2, 8, 16]:
        all_match, _, _, _, _ = run_comparison(num_tokens=num_tokens, seed=123, dtype='float32')
        assert all_match, f"Failed for num_tokens={num_tokens}"


def test_mla_comparison_different_seeds():
    """Test MLA with different random seeds."""
    for seed in [0, 42, 123, 999]:
        all_match, _, _, _, _ = run_comparison(num_tokens=4, seed=seed, dtype='float32')
        assert all_match, f"Failed for seed={seed}"


if __name__ == "__main__":
    # Run comparison with various configurations
    run_comparison(num_tokens=4, seed=42, dtype='float32')
    run_comparison(num_tokens=8, seed=123, dtype='float32')

    print("\n" + "="*70)
    print("Testing different sequence lengths...")
    print("="*70)
    for n in [1, 2, 4, 8, 16]:
        match, _, _, _, _ = run_comparison(num_tokens=n, seed=42, dtype='float32')
        if not match:
            print(f"FAILED for num_tokens={n}")
            break
    else:
        print("\nAll sequence length tests passed!")
