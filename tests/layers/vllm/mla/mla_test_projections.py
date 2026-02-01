#!/usr/bin/env python3
"""Test MLA projection matrices against vLLM's reference implementation.

This script verifies that our W_K and W_V extraction from kv_b_proj matches
vLLM's expected computation for both prefill (compute-friendly) and
decode (data-movement-friendly) modes.

Run with: python tests/mla_test_projections.py
"""

import numpy as np
import torch
import jax

# Force CPU execution
jax.config.update('jax_platform_name', 'cpu')

# GLM-4 MLA dimensions
GLM4_CONFIG = {
    'num_heads': 20,
    'qk_nope_head_dim': 192,  # P
    'qk_rope_head_dim': 64,   # R
    'v_head_dim': 256,         # V
    'kv_lora_rank': 512,       # L (Lkv)
}


def create_kv_b_proj_weight(config: dict, seed: int = 42) -> np.ndarray:
    """Create kv_b_proj weight matrix with known values."""
    np.random.seed(seed)
    num_heads = config['num_heads']
    qk_nope = config['qk_nope_head_dim']
    v_head = config['v_head_dim']
    kv_lora = config['kv_lora_rank']

    # kv_b_proj weight: (num_heads * (qk_nope + v_head), kv_lora_rank)
    # This is a PyTorch Linear layer weight [out_features, in_features]
    weight = np.random.randn(num_heads * (qk_nope + v_head), kv_lora).astype(np.float32) * 0.02
    return weight


def test_prefill_mode(config: dict = GLM4_CONFIG, seed: int = 42):
    """Test prefill mode (compute-friendly) k_nope and v computation.

    In prefill mode, vLLM computes:
        kv = kv_b_proj(kv_c)  # Linear forward: kv_c @ weight.T
        kv = kv.view(-1, num_heads, qk_nope + v_head)
        k_nope, v = kv.split([qk_nope, v_head], dim=-1)

    So k_nope[t,n,p] = sum_l(kv_c[t,l] * weight[n*(qk_nope+v) + p, l])
    """
    print("\n" + "="*70)
    print("Testing PREFILL mode (compute-friendly)")
    print("="*70)

    num_heads = config['num_heads']
    qk_nope = config['qk_nope_head_dim']
    v_head = config['v_head_dim']
    kv_lora = config['kv_lora_rank']

    # Create test data
    np.random.seed(seed)
    kv_b_weight = create_kv_b_proj_weight(config, seed)
    kv_c = np.random.randn(4, kv_lora).astype(np.float32)  # 4 tokens

    # --- vLLM's prefill computation ---
    # Linear forward: output = input @ weight.T
    kv_output = kv_c @ kv_b_weight.T  # (4, num_heads * (qk_nope + v))
    kv_output = kv_output.reshape(-1, num_heads, qk_nope + v_head)
    k_nope_vllm = kv_output[..., :qk_nope]  # (4, num_heads, qk_nope)
    v_vllm = kv_output[..., qk_nope:]       # (4, num_heads, v_head)

    print(f"vLLM prefill k_nope shape: {k_nope_vllm.shape}")
    print(f"vLLM prefill v shape: {v_vllm.shape}")

    # --- Our extraction approach ---
    # Reshape weight to (num_heads, qk_nope + v, kv_lora)
    weight_reshaped = kv_b_weight.reshape(num_heads, qk_nope + v_head, kv_lora)
    w_k = weight_reshaped[:, :qk_nope, :]  # (num_heads, qk_nope, kv_lora)
    w_v = weight_reshaped[:, qk_nope:, :]  # (num_heads, v_head, kv_lora)

    # Compute k_nope and v using our extracted W_K and W_V
    # For prefill: k_nope[t,n,p] = sum_l(kv_c[t,l] * W_K[n,p,l])
    # This is: kv_c @ W_K.T per head
    k_nope_ours = np.einsum('tl,npd->tnp', kv_c, w_k.transpose(0, 2, 1))
    # Wait, that's wrong. Let me think again...

    # Actually for prefill: kv_c @ weight.T = kv_c @ [out, in].T = kv_c @ [in, out]
    # The weight row n*(qk_nope+v)+p maps to k_nope[n,p]
    # So: k_nope[t,n,p] = sum_l(kv_c[t,l] * weight[n*(qk_nope+v)+p, l])
    #                   = sum_l(kv_c[t,l] * weight_reshaped[n, p, l])

    # einsum: (tokens, kv_lora) @ (heads, qk_nope, kv_lora) -> (tokens, heads, qk_nope)
    k_nope_ours = np.einsum('tl,npl->tnp', kv_c, w_k)
    v_ours = np.einsum('tl,nvl->tnv', kv_c, w_v)

    print(f"Our k_nope shape: {k_nope_ours.shape}")
    print(f"Our v shape: {v_ours.shape}")

    # Compare
    k_diff = np.max(np.abs(k_nope_vllm - k_nope_ours))
    v_diff = np.max(np.abs(v_vllm - v_ours))

    print(f"\nk_nope max diff: {k_diff:.2e}")
    print(f"v max diff: {v_diff:.2e}")

    k_match = np.allclose(k_nope_vllm, k_nope_ours, rtol=1e-5, atol=1e-5)
    v_match = np.allclose(v_vllm, v_ours, rtol=1e-5, atol=1e-5)

    print(f"\nk_nope match: {'✓' if k_match else '✗'}")
    print(f"v match: {'✓' if v_match else '✗'}")

    return k_match and v_match


def test_decode_mode(config: dict = GLM4_CONFIG, seed: int = 42):
    """Test decode mode (data-movement-friendly) Q projection.

    In decode mode, vLLM computes:
        ql_nope = einsum("snh,lnh->snl", q_nope, W_UK)

    Where W_UK has shape [L, N, P] = [kv_lora_rank, num_heads, qk_nope_head_dim]
    and is related to kv_b_proj.weight by:
        W_UK[l,n,p] = weight[n*(qk_nope+v)+p, l]
    """
    print("\n" + "="*70)
    print("Testing DECODE mode (data-movement-friendly)")
    print("="*70)

    num_heads = config['num_heads']
    qk_nope = config['qk_nope_head_dim']
    v_head = config['v_head_dim']
    kv_lora = config['kv_lora_rank']

    # Create test data
    np.random.seed(seed)
    kv_b_weight = create_kv_b_proj_weight(config, seed)
    q_nope = np.random.randn(4, num_heads, qk_nope).astype(np.float32)

    # --- Construct vLLM's W_UK from kv_b_proj ---
    # W_UK[l,n,p] = weight[n*(qk_nope+v)+p, l]
    # Shape: [kv_lora, num_heads, qk_nope]
    W_UK_vllm = np.zeros((kv_lora, num_heads, qk_nope), dtype=np.float32)
    for n in range(num_heads):
        for p in range(qk_nope):
            W_UK_vllm[:, n, p] = kv_b_weight[n * (qk_nope + v_head) + p, :]

    # vLLM's decode projection
    ql_nope_vllm = np.einsum('snh,lnh->snl', q_nope, W_UK_vllm)

    print(f"vLLM W_UK shape: {W_UK_vllm.shape}")
    print(f"vLLM ql_nope shape: {ql_nope_vllm.shape}")

    # --- Our extraction approach ---
    weight_reshaped = kv_b_weight.reshape(num_heads, qk_nope + v_head, kv_lora)
    w_k = weight_reshaped[:, :qk_nope, :]  # (num_heads, qk_nope, kv_lora)

    # Our projection: einsum('thd,hdk->thk', q_nope, w_k)
    ql_nope_ours = np.einsum('thd,hdk->thk', q_nope, w_k)

    print(f"Our w_k shape: {w_k.shape}")
    print(f"Our ql_nope shape: {ql_nope_ours.shape}")

    # Compare
    diff = np.max(np.abs(ql_nope_vllm - ql_nope_ours))
    print(f"\nql_nope max diff: {diff:.2e}")

    match = np.allclose(ql_nope_vllm, ql_nope_ours, rtol=1e-5, atol=1e-5)
    print(f"ql_nope match: {'✓' if match else '✗'}")

    # Also verify W_UK relationship
    # Our w_k[n,p,l] should equal W_UK[l,n,p]
    w_k_transposed = w_k.transpose(2, 0, 1)  # (kv_lora, num_heads, qk_nope)
    w_uk_diff = np.max(np.abs(W_UK_vllm - w_k_transposed))
    print(f"\nW_UK vs w_k.transpose(2,0,1) max diff: {w_uk_diff:.2e}")
    w_uk_match = np.allclose(W_UK_vllm, w_k_transposed, rtol=1e-5, atol=1e-5)
    print(f"W_UK structure match: {'✓' if w_uk_match else '✗'}")

    return match and w_uk_match


def test_output_projection(config: dict = GLM4_CONFIG, seed: int = 42):
    """Test output projection through W_V.

    In decode mode, vLLM computes:
        o = einsum("snl,lnv->snv", spda_o, W_UV)

    Where W_UV has shape [L, N, V] = [kv_lora_rank, num_heads, v_head_dim]
    and is related to kv_b_proj.weight by:
        W_UV[l,n,v] = weight[n*(qk_nope+v)+qk_nope+v, l]
    """
    print("\n" + "="*70)
    print("Testing OUTPUT projection through W_V")
    print("="*70)

    num_heads = config['num_heads']
    qk_nope = config['qk_nope_head_dim']
    v_head = config['v_head_dim']
    kv_lora = config['kv_lora_rank']

    # Create test data
    np.random.seed(seed)
    kv_b_weight = create_kv_b_proj_weight(config, seed)
    output_latent = np.random.randn(4, num_heads, kv_lora).astype(np.float32)

    # --- Construct vLLM's W_UV from kv_b_proj ---
    # W_UV[l,n,v] = weight[n*(qk_nope+v)+qk_nope+v, l]
    # Shape: [kv_lora, num_heads, v_head]
    W_UV_vllm = np.zeros((kv_lora, num_heads, v_head), dtype=np.float32)
    for n in range(num_heads):
        for v in range(v_head):
            W_UV_vllm[:, n, v] = kv_b_weight[n * (qk_nope + v_head) + qk_nope + v, :]

    # vLLM's output projection
    output_vllm = np.einsum('snl,lnv->snv', output_latent, W_UV_vllm)

    print(f"vLLM W_UV shape: {W_UV_vllm.shape}")
    print(f"vLLM output shape: {output_vllm.shape}")

    # --- Our extraction approach ---
    weight_reshaped = kv_b_weight.reshape(num_heads, qk_nope + v_head, kv_lora)
    w_v = weight_reshaped[:, qk_nope:, :]  # (num_heads, v_head, kv_lora)

    # Our projection: einsum('thk,hvk->thv', output_latent, w_v)
    output_ours = np.einsum('thk,hvk->thv', output_latent, w_v)

    print(f"Our w_v shape: {w_v.shape}")
    print(f"Our output shape: {output_ours.shape}")

    # Compare
    diff = np.max(np.abs(output_vllm - output_ours))
    print(f"\nOutput max diff: {diff:.2e}")

    match = np.allclose(output_vllm, output_ours, rtol=1e-5, atol=1e-5)
    print(f"Output match: {'✓' if match else '✗'}")

    # Also verify W_UV relationship
    # Our w_v[n,v,l] should equal W_UV[l,n,v]
    w_v_transposed = w_v.transpose(2, 0, 1)  # (kv_lora, num_heads, v_head)
    w_uv_diff = np.max(np.abs(W_UV_vllm - w_v_transposed))
    print(f"\nW_UV vs w_v.transpose(2,0,1) max diff: {w_uv_diff:.2e}")
    w_uv_match = np.allclose(W_UV_vllm, w_v_transposed, rtol=1e-5, atol=1e-5)
    print(f"W_UV structure match: {'✓' if w_uv_match else '✗'}")

    return match and w_uv_match


def test_full_mla_decode(config: dict = GLM4_CONFIG, seed: int = 42):
    """Test full MLA decode path end-to-end."""
    print("\n" + "="*70)
    print("Testing FULL MLA DECODE path")
    print("="*70)

    num_heads = config['num_heads']
    qk_nope = config['qk_nope_head_dim']
    qk_rope = config['qk_rope_head_dim']
    v_head = config['v_head_dim']
    kv_lora = config['kv_lora_rank']
    qk_head = qk_nope + qk_rope

    num_tokens = 4
    scale = 1.0 / np.sqrt(qk_head)

    # Create test data
    np.random.seed(seed)
    kv_b_weight = create_kv_b_proj_weight(config, seed)
    q = np.random.randn(num_tokens, num_heads, qk_head).astype(np.float32)
    kv_c = np.random.randn(num_tokens, kv_lora).astype(np.float32)
    k_pe = np.random.randn(num_tokens, qk_rope).astype(np.float32)

    # Split Q
    q_nope = q[..., :qk_nope]
    q_pe = q[..., qk_nope:]

    # --- vLLM approach (construct W_UK and W_UV explicitly) ---
    W_UK = np.zeros((kv_lora, num_heads, qk_nope), dtype=np.float32)
    W_UV = np.zeros((kv_lora, num_heads, v_head), dtype=np.float32)
    for n in range(num_heads):
        for p in range(qk_nope):
            W_UK[:, n, p] = kv_b_weight[n * (qk_nope + v_head) + p, :]
        for v in range(v_head):
            W_UV[:, n, v] = kv_b_weight[n * (qk_nope + v_head) + qk_nope + v, :]

    # Step 1: Project q_nope
    ql_nope_vllm = np.einsum('snh,lnh->snl', q_nope, W_UK)

    # Step 2: Combine Q and K
    q_combined_vllm = np.concatenate([ql_nope_vllm, q_pe], axis=-1)
    k_combined_vllm = np.concatenate([kv_c, k_pe], axis=-1)

    # Step 3: Attention
    attn_scores = np.einsum('qnh,kh->nqk', q_combined_vllm, k_combined_vllm) * scale
    causal_mask = np.tril(np.ones((num_tokens, num_tokens)))
    attn_scores = np.where(causal_mask[None, :, :] == 0, -1e9, attn_scores)
    attn_weights = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)

    # Step 4: Output in latent space
    output_latent_vllm = np.einsum('nqk,kl->qnl', attn_weights, kv_c)

    # Step 5: Project to v_head_dim
    output_vllm = np.einsum('snl,lnv->snv', output_latent_vllm, W_UV)

    # --- Our approach (extract w_k and w_v directly) ---
    weight_reshaped = kv_b_weight.reshape(num_heads, qk_nope + v_head, kv_lora)
    w_k = weight_reshaped[:, :qk_nope, :]
    w_v = weight_reshaped[:, qk_nope:, :]

    # Step 1: Project q_nope
    ql_nope_ours = np.einsum('thd,hdk->thk', q_nope, w_k)

    # Step 2: Combine Q and K (same as vLLM)
    q_combined_ours = np.concatenate([ql_nope_ours, q_pe], axis=-1)
    k_combined_ours = np.concatenate([kv_c, k_pe], axis=-1)

    # Step 3: Attention (same as vLLM)
    attn_scores_ours = np.einsum('qnh,kh->nqk', q_combined_ours, k_combined_ours) * scale
    attn_scores_ours = np.where(causal_mask[None, :, :] == 0, -1e9, attn_scores_ours)
    attn_weights_ours = np.exp(attn_scores_ours - np.max(attn_scores_ours, axis=-1, keepdims=True))
    attn_weights_ours = attn_weights_ours / np.sum(attn_weights_ours, axis=-1, keepdims=True)

    # Step 4: Output in latent space
    output_latent_ours = np.einsum('nqk,kl->qnl', attn_weights_ours, kv_c)

    # Step 5: Project to v_head_dim
    output_ours = np.einsum('thk,hvk->thv', output_latent_ours, w_v)

    # Compare at each step
    print(f"\nql_nope max diff: {np.max(np.abs(ql_nope_vllm - ql_nope_ours)):.2e}")
    print(f"q_combined max diff: {np.max(np.abs(q_combined_vllm - q_combined_ours)):.2e}")
    print(f"attn_weights max diff: {np.max(np.abs(attn_weights - attn_weights_ours)):.2e}")
    print(f"output_latent max diff: {np.max(np.abs(output_latent_vllm - output_latent_ours)):.2e}")
    print(f"final output max diff: {np.max(np.abs(output_vllm - output_ours)):.2e}")

    match = np.allclose(output_vllm, output_ours, rtol=1e-5, atol=1e-5)
    print(f"\nFull MLA decode match: {'✓' if match else '✗'}")

    return match


if __name__ == "__main__":
    print("="*70)
    print("MLA PROJECTION MATRIX TESTS")
    print("="*70)

    results = []

    results.append(("Prefill mode", test_prefill_mode()))
    results.append(("Decode Q projection", test_decode_mode()))
    results.append(("Output projection", test_output_projection()))
    results.append(("Full MLA decode", test_full_mla_decode()))

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    all_pass = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
        exit(1)
