import jax
import jax.numpy as jnp
import numpy as np
import os
from typing import Dict, Any, List, Tuple

# --- Configuration & Files ---
RNG_SEED = 42
JAX_DTYPE = jnp.bfloat16
INPUT_FILE = "canonical_input_data.npz"
WEIGHTS_FILE = "canonical_mlp_weights.npz"
OUTPUT_FILE = "jax_encoder_output.npy"
PT_OUTPUT_FILE = "pytorch_encoder_output.npy" 
HIDDEN_SIZE = 1408

# --- Import JAX Components (Must be available on TPU host) ---
from flax import nnx
from jax.sharding import Mesh 
from tpu_inference.models.jax.llama4 import (
    JAXLlama4VisionEncoder,
)
from transformers.models.llama4.configuration_llama4 import Llama4VisionConfig 

# NOTE: You must include the definition of create_config from the generator script here for consistency.
def create_config() -> Llama4VisionConfig:
    """Creates a mock configuration object."""
    return Llama4VisionConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=16,
        num_hidden_layers=34,
        intermediate_size=5632,
        patch_size=14,
        image_size=336,
        norm_eps=1e-5,
        vision_output_dim=4096,
        projector_input_dim=4096,
        projector_output_dim=4096,
        rope_theta=10000.0,
        _attn_implementation="eager",
    )

def inject_weights(jax_model: JAXLlama4VisionEncoder, weights_to_load: Dict[str, np.ndarray]):
    """
    Loads MLP, LayerNorm, and Attention kernels into the JAX model.
    The transpose is necessary as PyTorch stores weights as [O, I], and JAX/Flax often expects [I, O].
    """
    
    print("--- Injecting ALL Vision Encoder Weights ---")
    
    for i in range(len(jax_model.layers)):
        jax_layer = jax_model.layers[i]
        
        # --- LayerNorm (Scale/Bias) Injection ---
        for name in ['input_layernorm', 'post_attention_layernorm']:
            jax_ln = getattr(jax_layer, name)
            prefix = f'layer.{i}.ln.{name}'
            
            # Load Scale (weight/gamma) and Bias (beta)
            jax_ln.scale.value = jnp.asarray(weights_to_load[f'{prefix}.scale'], dtype=JAX_DTYPE)
            jax_ln.bias.value = jnp.asarray(weights_to_load[f'{prefix}.bias'], dtype=JAX_DTYPE)

        # --- MLP Kernels (FC1/FC2) Injection ---
        for jax_linear, layer_name in [
            (jax_layer.mlp.fc1, 'fc1'), 
            (jax_layer.mlp.fc2, 'fc2')
        ]:
            prefix = f'layer.{i}.mlp.{layer_name}'
            pt_kernel_f32 = weights_to_load[f'{prefix}.kernel']
            pt_bias_f32 = weights_to_load[f'{prefix}.bias']
            
            # FIX: TRANSPOSE MANDATORY: [O, I] -> [I, O], and ensure contiguity
            jax_kernel = np.ascontiguousarray(np.transpose(pt_kernel_f32)) 
            
            jax_linear.kernel.value = jnp.asarray(jax_kernel, dtype=JAX_DTYPE)
            jax_linear.bias.value = jnp.asarray(pt_bias_f32, dtype=JAX_DTYPE)

            if i == 0 and layer_name == 'fc1':
                # --- Retrieve loaded JAX value ---
                jax_loaded_f32 = np.asarray(jax.device_get(jax_linear.kernel.value)).astype(np.float32)
                
                # --- Prepare expected value (Source f32, transposed) ---
                expected_f32 = jax_kernel.astype(np.float32)
                
                # --- Set Tolerance ---
                # Expected bfloat16 noise is around 1e-4. We set a generous tolerance.
                TOLERANCE = 5e-4 
                
                # --- Calculate Differences ---
                differences = np.abs(expected_f32 - jax_loaded_f32)
                max_diff_load = np.max(differences)
                
                # Count elements that differ more than expected bfloat16 noise
                num_elements = differences.size
                num_out_of_tolerance = np.sum(differences > TOLERANCE)
                
                print("\n--- WEIGHT LOAD VERIFICATION (Layer 0, fc1) ---")
                print(f"Total Elements Checked: {num_elements}")
                print(f"Max Difference Found: {max_diff_load:.6f}")
                print(f"Elements > {TOLERANCE:.1e} Tolerance: {num_out_of_tolerance}")

                if num_out_of_tolerance == 0 and max_diff_load < TOLERANCE * 2:
                    print("LOAD CHECK: ALL weights match within expected bfloat16 noise.")
                else:
                    print("LOAD CHECK: Structural/Precision discrepancy detected in loaded weights.")
        
    for i in range(len(jax_model.layers)):
        jax_layer = jax_model.layers[i]
        
        # Determine the target dimensions
        H = jax_layer.self_attn.head_dim # 88
        N = jax_layer.self_attn.num_attention_heads # 16
        D = jax_layer.self_attn.hidden_size # 1408
        K = jax_layer.self_attn.num_key_value_heads # 16 (for this model)
        
        # --- ATTENTION KERNELS (Q, K, V, O) INJECTION ---
        for name, jax_kernel_attr in [
            ('q_proj', jax_layer.self_attn.kernel_q_proj_DNH),
            ('k_proj', jax_layer.self_attn.kernel_k_proj_DKH),
            ('v_proj', jax_layer.self_attn.kernel_v_proj_DKH),
            ('o_proj', jax_layer.self_attn.kernel_o_proj_NHD),
        ]:
            prefix = f'layer.{i}.attn.{name}'
            pt_kernel_f32 = weights_to_load[f'{prefix}.kernel'] 
            
            # 1. Determine transposition
            if name in ['q_proj', 'k_proj', 'v_proj']:
                # FIX: Q, K, V must NOT be transposed (required for numerical correctness)
                pt_kernel_transposed = pt_kernel_f32
            else: # name == 'o_proj'
                # FIX: W_O MUST be transposed (required for structural/numerical correctness)
                pt_kernel_transposed = np.ascontiguousarray(np.transpose(pt_kernel_f32)) 
            
            # 2. Reshape to target JAX format
            if name == 'o_proj':
                # W_O: PT [D, D] -> JAX [N, H, D]
                jax_kernel_final = pt_kernel_transposed.reshape(N, H, D) 
            else: 
                # W_Q, W_K, W_V: PT [D, D] -> JAX [D, N/K, H]
                jax_kernel_final = pt_kernel_transposed.reshape(D, N, H) 

            # 3. Final Assignment
            jax_kernel_attr.value = jnp.asarray(jax_kernel_final, dtype=JAX_DTYPE)
            
    print("--- Injection Complete. ---")


def run_jax_encoder_on_tpu():
    # 1. Load Inputs and Weights
    if not os.path.exists(INPUT_FILE) or not os.path.exists(WEIGHTS_FILE):
        print("\nFATAL ERROR: Input/Weight files not found. Please run input_generator.py first and transfer files.")
        return False
        
    data = np.load(INPUT_FILE)
    jax_hiddens = jnp.asarray(data['hiddens'], dtype=JAX_DTYPE)
    jax_freqs = jnp.asarray(data['freqs'], dtype=JAX_DTYPE)
    weights_to_load = np.load(WEIGHTS_FILE)

    # 2. Setup and Initialization
    config = create_config()
    rng_key = jax.random.PRNGKey(RNG_SEED)
    rngs = nnx.Rngs(rng_key)
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ('data'))
    
    jax_encoder = JAXLlama4VisionEncoder(config=config, rngs=rngs, mesh=mesh)
    inject_weights(jax_encoder, weights_to_load)

    # 3. Execute JAX Forward Pass
    print("\n[PART 3: JAX] Executing JAX model on target hardware...")
    
    jax_output = jax_encoder(
        jax_hiddens, 
        jax_freqs,
        num_kv_pages_per_block=1, 
        num_queries_per_block=1,
    )
    
    # 4. Save Output
    jax_output_np = np.asarray(jax.device_get(jax_output))
    np.save(OUTPUT_FILE, jax_output_np)
    
    print(f"JAX Execution Complete. Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_jax_encoder_on_tpu()