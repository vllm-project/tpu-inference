# --- Minimal Comparison Runner (Final Fix) ---
import numpy as np
import jax.numpy as jnp
# NOTE: Need to import jax directly if jnp.load is used, though jnp.load might suffice.

PT_OUTPUT_FILE = "pytorch_encoder_output.npy"
JAX_OUTPUT_FILE = "jax_encoder_output.npy"

pt_output_np = np.load(PT_OUTPUT_FILE)

# *** FIX: Load the JAX file using jax.numpy, and explicitly convert to float32 NumPy array ***
# 1. Use jnp.load to correctly interpret the bfloat16 saved format.
# 2. Convert the result to float32.
# 3. Use jax.device_get to pull the array back to the CPU/host memory as a standard NumPy array.
import jax # Ensure JAX is imported for jax.device_get

jax_output_np = jax.device_get(jnp.load(JAX_OUTPUT_FILE))
jax_output_f32 = jax_output_np.astype(np.float32) # This conversion should now be safe


assert pt_output_np.shape == jax_output_np.shape, f"SHAPE MISMATCH: HF {pt_output_np.shape} != JAX {jax_output_np.shape}"
        
# Ensure PT output is at least bfloat16
JAX_DTYPE_NP = jnp.bfloat16.dtype
pt_output_b16 = pt_output_np.astype(JAX_DTYPE_NP)
pt_output_f32 = pt_output_b16.astype(np.float32)

# Perform subtraction on the compatible float32 arrays
max_diff = np.max(np.abs(pt_output_f32 - jax_output_f32))

print(f"\n--- FINAL HETEROGENEOUS VALIDATION ---")
if max_diff < 1e-3:
    print(f"SUCCESS: JAX Vision Encoder matches PyTorch reference.")
    print(f"Max Difference: {max_diff:.5f}")
else:
    print(f"FAILURE: Numerical Mismatch (Max Diff: {max_diff:.5f})")
    print(f"HF Slice: {pt_output_f32[0, 0, :5]}")
    print(f"JAX Slice: {jax_output_f32[0, 0, :5]}")