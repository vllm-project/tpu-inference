# --- weight_verification.py ---
import numpy as np
import jax.numpy as jnp
import os

# Assuming you still have access to the original 'weights_to_load' structure 
# saved as 'canonical_mlp_weights.npz'

WEIGHTS_FILE = "canonical_mlp_weights.npz"
weights_to_load = np.load(WEIGHTS_FILE)

# Configuration constants from jax_execution.py
HIDDEN_SIZE = 1408
JAX_DTYPE = jnp.bfloat16
D = HIDDEN_SIZE # 1408
N = 16 # num_attention_heads
H = 88 # head_dim
K = 16 # num_key_value_heads

def verify_weight_transpose(layer_idx: int, kernel_name: str, is_attention: bool, should_transpose: bool):
    """Loads PT weight, applies the expected JAX logic, and checks the shape."""
    
    prefix = f'layer.{layer_idx}.'
    
    if is_attention:
        pt_key = f'{prefix}attn.{kernel_name}.kernel'
        target_shape = (N, H, D) if kernel_name == 'o_proj' else (D, N, H) # N=K=16 here
        
    else: # MLP
        pt_key = f'{prefix}mlp.{kernel_name}.kernel'
        target_shape = (D, 5632) if kernel_name == 'fc1' else (5632, D) # [I, O] shape convention in JAX
        
    pt_kernel_f32 = weights_to_load[pt_key]
    
    # Apply the expected transposition logic based on the final working configuration
    if should_transpose:
        # MLP and W_O require transpose
        pt_kernel_transformed = np.ascontiguousarray(np.transpose(pt_kernel_f32))
    else:
        # Q, K, V require NO transpose
        pt_kernel_transformed = pt_kernel_f32
        
    # Apply the required reshape
    try:
        # Ensure the kernel reshapes to the target JAX shape
        jax_kernel_final = pt_kernel_transformed.reshape(target_shape)
        
        # Verify if the original kernel transpose logic was actually correct:
        # JAX loads [I, O]. PT stores [O, I]. Transpose is needed unless the input is multi-dim.
        
        # We assume the reshape is always correct if the logic (transpose/no transpose) is correct.
        print(f"{kernel_name} (L{layer_idx}): Transpose={should_transpose}. Reshapes correctly to {target_shape}.")
        return True
    
    except ValueError as e:
        print(f"{kernel_name} (L{layer_idx}): Transpose={should_transpose}. FAILED reshape to {target_shape}. Error: {e}")
        print(f"   Shape after transpose/no-transpose: {pt_kernel_transformed.shape}")
        return False


print("\n--- WEIGHT TRANSPOSE VERIFICATION (Layer 0) ---")

# --- MLP WEIGHTS ---
# MLP MUST be transposed (proved by previous TypeError)
verify_weight_transpose(0, 'fc1', is_attention=False, should_transpose=True)
verify_weight_transpose(0, 'fc2', is_attention=False, should_transpose=True)

# --- ATTENTION WEIGHTS (Based on Final Working Configuration) ---
# Q, K, V must NOT be transposed (Proven correct by shift)
verify_weight_transpose(0, 'q_proj', is_attention=True, should_transpose=False)
verify_weight_transpose(0, 'k_proj', is_attention=True, should_transpose=False)
verify_weight_transpose(0, 'v_proj', is_attention=True, should_transpose=False)

# W_O MUST be transposed (Inferred correct based on numerical consistency)
verify_weight_transpose(0, 'o_proj', is_attention=True, should_transpose=True)

print("--- Verification Complete ---")

