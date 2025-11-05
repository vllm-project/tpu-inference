import jax
import jax.numpy as jnp
import numpy as np
import torch
import os
from typing import Any, Dict, Tuple
from flax import nnx

# --- JAX IMPORTS ---
# Assuming JAXLlama4VisionEncoder is in tpu_inference.models.jax.llama_guard_4
from tpu_inference.models.jax.llama4 import (
    JAXLlama4VisionEncoder, 
)

from tpu_inference.layers.jax.llama4_vision_rope import Llama4VisionRotaryEmbedding as JAXLlama4VisionRotaryEmbedding

# --- HUGGINGFACE IMPORTS ---
# Assuming Llama4VisionEncoder and necessary config are imported from the correct location
from transformers.models.llama4.modeling_llama4 import Llama4VisionEncoder, Llama4VisionConfig, Llama4VisionRotaryEmbedding as PytorchLlama4VisionRotaryEmbedding
from transformers.models.llama4.configuration_llama4 import Llama4VisionConfig # Re-import for clarity


# ----------------------------------------------------------------------
# 1. SETUP AND UTILITIES
# ----------------------------------------------------------------------

# Set devices to CPU for direct comparison (no JAX sharding issues)
os.environ['CUDA_VISIBLE_DEVICES'] = ""
DEVICE = "cpu"
DTYPE = torch.bfloat16
JAX_DTYPE = jnp.bfloat16

# --- Helper function to capture intermediate output ---
captured_output = {}
def capture_hook(module, input, output):
    # Store the output of the layer that was just executed
    # We use .cpu() to move it for NumPy conversion
    captured_output['ln_output'] = output.detach().cpu()


def inject_vision_mlp_weights(hf_model, jax_model):
    """Manually loads fc1 and fc2 kernels/biases from PyTorch model into the JAX model."""
    
    print("\n--- Injecting Vision MLP Weights (34 Layers) ---")
    
    # Iterate through all encoder layers (34 layers)
    for i in range(len(jax_model.layers)):
        pt_layer = hf_model.layers[i]
        jax_layer = jax_model.layers[i]

        # --- CRITICAL FIX 1: LAYER NORM (LN) WEIGHTS ---
        for name in ['input_layernorm', 'post_attention_layernorm']:
            pt_ln = getattr(pt_layer, name)
            jax_ln = getattr(jax_layer, name)
            
            # Load Scale (weight/gamma) and Bias (beta)
            # We are assuming LayerNorm has scale and bias parameters (which is true for nn.LayerNorm in HF)
            pt_ln_scale = pt_ln.weight.data.to(torch.float32).numpy()
            jax_ln.scale.value = jnp.asarray(pt_ln_scale, dtype=JAX_DTYPE)
            
            pt_ln_bias = pt_ln.bias.data.to(torch.float32).numpy()
            jax_ln.bias.value = jnp.asarray(pt_ln_bias, dtype=JAX_DTYPE)
        # -------------------------------------------------------------
        
        # --- CRITICAL FIX 2: MLP FC1/FC2 KERNEL INJECTION (Correcting Transposition/Contiguity) ---
        
        # --- FC1 WEIGHTS ---
        pt_kernel_fc1_f32 = pt_layer.mlp.fc1.weight.data.to(torch.float32).numpy()
        pt_bias_fc1 = pt_layer.mlp.fc1.bias.data.to(torch.float32).numpy()
        
        # FIX: TRANSPOSE IS MANDATORY
        jax_kernel_fc1 = np.ascontiguousarray(np.transpose(pt_kernel_fc1_f32))
        
        # The assertion now checks [1408, 5632] == [1408, 5632]
        assert jax_layer.mlp.fc1.kernel.value.shape == jax_kernel_fc1.shape 
        
        # Assign:
        jax_layer.mlp.fc1.kernel.value = jnp.asarray(jax_kernel_fc1, dtype=JAX_DTYPE)
        jax_layer.mlp.fc1.bias.value = jnp.asarray(pt_bias_fc1, dtype=JAX_DTYPE)
        
        # --- FC2 WEIGHTS (Transpose [O, I] -> [I, O]) ---
        pt_kernel_fc2_f32 = pt_layer.mlp.fc2.weight.data.to(torch.float32).numpy()
        pt_bias_fc2 = pt_layer.mlp.fc2.bias.data.to(torch.float32).numpy()
        
        # FIX: TRANSPOSE AND ENFORCE CONTIGUOUS MEMORY ORDER
        jax_kernel_fc2 = np.ascontiguousarray(np.transpose(pt_kernel_fc2_f32)) 
        
        # Assign:
        jax_layer.mlp.fc2.kernel.value = jnp.asarray(jax_kernel_fc2, dtype=JAX_DTYPE)
        jax_layer.mlp.fc2.bias.value = jnp.asarray(pt_bias_fc2, dtype=JAX_DTYPE)
        
        if i == 0:
            print(f"Layer 0 MLP fc1 Kernel slice (JAX Loaded): {jax_layer.mlp.fc1.kernel.value[0, :5]}")
            print(f"Layer 0 MLP fc1 Bias slice (JAX Loaded): {jax_layer.mlp.fc1.bias.value[:5]}")
            pt_ln_output_check = pt_layer.post_attention_layernorm.bias.data.to(torch.float32).numpy()[:5]
            jax_ln_output_check = jax_layer.post_attention_layernorm.bias.value[:5]
            print(f"Layer 0 LN Bias Check (PT): {pt_ln_output_check}")
            print(f"Layer 0 LN Bias Check (JAX Loaded): {jax_ln_output_check}")
            
    print("--- Injection Complete. ---")

def create_mock_config(hidden_size=1408, num_layers=34):
    """Creates a mock configuration dict/object for initializing models."""
    return Llama4VisionConfig(
        hidden_size=hidden_size,
        num_attention_heads=16,
        num_hidden_layers=num_layers,
        intermediate_size=5632,
        patch_size=14,
        image_size=336,
        norm_eps=1e-5,
        vision_output_dim=4096,
        projector_input_dim=4096,
        projector_output_dim=4096,
        projector_dropout=0.0,
        pixel_shuffle_ratio=0.5,
        rope_theta=10000.0,
        _attn_implementation="eager",
    )

def create_random_inputs(config: Llama4VisionConfig, batch_size=1, sequence_length=577):
    """Creates random inputs matching the encoder's intermediate shape."""    
    # Ensure DTYPE and JAX_DTYPE are available in the scope (as defined externally)
    # DTYPE = torch.bfloat16
    # JAX_DTYPE = jnp.bfloat16
    
    # Encoder input shape: [batch_size * num_images, sequence_length (patches+CLS), hidden_size]
    # 1. Create PyTorch Hidden States (BFloat16, converted to Float32 for numpy compatibility)
    hidden_states_bfloat16 = torch.randn(batch_size, sequence_length, config.hidden_size, dtype=DTYPE).to(DEVICE)
    
    # 2. Create RoPE Frequencies (complex tensor for PyTorch, split for JAX)
    head_dim = config.hidden_size // config.num_attention_heads
    
    # Generate random real/imaginary parts in Float32 to avoid NumPy errors
    torch_freqs_real = torch.randn(sequence_length, head_dim // 2, dtype=torch.float32)
    torch_freqs_imag = torch.randn(sequence_length, head_dim // 2, dtype=torch.float32)
    
    # Combine into a PyTorch complex tensor [S, D//2] for the HF model call
    torch_freqs = torch.complex(torch_freqs_real, torch_freqs_imag).to(DTYPE).to(DEVICE)
    
    
    # --- CRITICAL FIXES START HERE ---
    
    # A. Define jax_hidden_states: Convert PyTorch input to JAX bfloat16
    # We must first convert the PyTorch input to float32 before calling .numpy()
    jax_hidden_states = jnp.asarray(hidden_states_bfloat16.to(torch.float32).numpy(), dtype=JAX_DTYPE)
    
    # B. Define jax_freqs_ci: Stack real/imaginary parts for JAX RoPE function [S, D//2, 2]
    jax_freqs_real = jnp.asarray(torch_freqs_real.numpy(), dtype=JAX_DTYPE)
    jax_freqs_imag = jnp.asarray(torch_freqs_imag.numpy(), dtype=JAX_DTYPE)
    jax_freqs_ci = jnp.stack([jax_freqs_real, jax_freqs_imag], axis=-1) 
    
    # --- CRITICAL FIXES END HERE ---
    
    # Return 4 values: (PyTorch Hiddens, PyTorch Freqs, JAX Hiddens, JAX Freqs)
    return hidden_states_bfloat16, torch_freqs, jax_hidden_states, jax_freqs_ci

def create_canonical_inputs(config: Llama4VisionConfig, batch_size=1, sequence_length=577):
    """
    Creates inputs using a random hidden state but the canonical, pre-calculated
    Llama4 Vision RoPE frequencies from the HuggingFace implementation.
    """    
    # Encoder input shape: [batch_size, sequence_length (patches+CLS), hidden_size]
    
    # 1. Create PyTorch Hidden States
    hidden_states_bfloat16 = torch.randn(batch_size, sequence_length, config.hidden_size, dtype=DTYPE).to(DEVICE)
    
    # 2. Get the CANONICAL RoPE Frequencies from the HF class
    hf_rope_module = PytorchLlama4VisionRotaryEmbedding(config)
    
    # The .forward() on this module returns the pre-calculated complex tensor [S, D_rot]
    # We must pass a dummy tensor to the forward call, as per the HF implementation
    dummy_hiddens = torch.empty(1, 1, 1) 
    torch_freqs_complex_cpu = hf_rope_module.forward(dummy_hiddens).cpu().to(torch.complex64)
    torch_freqs = torch_freqs_complex_cpu.to(DTYPE).to(DEVICE)
    
    # 3. Define jax_hidden_states: Convert PyTorch input to JAX bfloat16
    jax_hiddens = jnp.asarray(hidden_states_bfloat16.to(torch.float32).numpy(), dtype=JAX_DTYPE)
    
    # 4. Define jax_freqs_ci: Stack real/imaginary parts for JAX RoPE function [S, D//2, 2]
    # NOTE: The HF RoPE module returns the complex tensor. We need to extract the real/imag parts.
    
    # a. Convert complex tensor back to real/imaginary parts
    # torch_freqs is [S, D_rot] (complex). torch.view_as_real converts to [S, D_rot, 2] (real)
    freqs_real_imag = torch.view_as_real(torch_freqs_complex_cpu).numpy()# [S, D//2, 2]
    
    # b. Extract and convert to JAX array
    jax_freqs_real = jnp.asarray(freqs_real_imag[..., 0], dtype=JAX_DTYPE)
    jax_freqs_imag = jnp.asarray(freqs_real_imag[..., 1], dtype=JAX_DTYPE)
    
    # c. Stack them for the JAX required format
    jax_freqs_ci = jnp.stack([jax_freqs_real, jax_freqs_imag], axis=-1) 
    
    # Return 4 values: (PyTorch Hiddens, PyTorch Freqs, JAX Hiddens, JAX Freqs)
    return hidden_states_bfloat16, torch_freqs, jax_hiddens, jax_freqs_ci

def compare_outputs(hf_output: torch.Tensor, jax_output: jax.Array, tolerance: float = 1e-3):
    """Converts JAX output to NumPy/PyTorch and performs a numerical comparison."""
    jax_output_np = np.asarray(jax.device_get(jax_output))
    hf_output_np = hf_output.cpu().to(torch.float32).numpy()

    # Convert HF to the expected JAX dtype precision (bfloat16) for fair comparison
    hf_output_bfloat16_np = hf_output_np.astype(JAX_DTYPE.dtype)
    
    # Check shape equality first
    if hf_output_np.shape != jax_output_np.shape:
        print(f"SHAPE MISMATCH: HF {hf_output_np.shape} != JAX {jax_output_np.shape}")
        return False
    
    # Use torch's built-in check for robust bfloat16 comparison
    max_diff = np.max(np.abs(hf_output_bfloat16_np - jax_output_np))

    if max_diff > tolerance:
        print(f"NUMERICAL MISMATCH (Tolerance: {tolerance})")
        print(f"Max Absolute Difference: {max_diff}")
        # Print slices for visual inspection
        print("\n--- HF Slice (bfloat16 precision) ---")
        print(hf_output_bfloat16_np[0, 0, :5])
        print("\n--- JAX Slice ---")
        print(jax_output_np[0, 0, :5])
        return False
    
    print(f"COMPARISON PASSED. Max Difference: {max_diff}")
    return True


# ----------------------------------------------------------------------
# 2. VERIFICATION EXECUTION
# ----------------------------------------------------------------------

def verify_vision_encoder():
    print("--- Starting Llama4 Vision Encoder Verification ---")
    config = create_mock_config()
    
    # Initialize JAX and PyTorch encoders
    rng_key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(params=rng_key)
    
    # NOTE: mesh is initialized for CPU devices when JAX_PLATFORMS=cpu is set
    # Ensure mesh uses available devices (CPUs)
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ('data')) 
    
    DUMMY_NUM_KV_PAGES_PER_BLOCK = 1
    DUMMY_NUM_QUERIES_PER_BLOCK = 1

    # JAX Model Initialization
    jax_encoder = JAXLlama4VisionEncoder(config=config, rngs=rngs, mesh=mesh)
    
    # PyTorch Model Initialization (Ensure weights are identical before calling)
    hf_encoder = Llama4VisionEncoder(config=config)
    
    hf_encoder.to(DTYPE)

    # inject MLP weights
    inject_vision_mlp_weights(hf_encoder, jax_encoder)

    # --- CRITICAL PATCH: Hook the LayerNorm Layer ---
    # The LayerNorm we want is the 'input_layernorm' of the first encoder layer (hf_encoder.layers[0]).
    # NOTE: The HF structure is usually encoder.layers[0].input_layernorm
    
    # Check the structure in your HF library's model (assuming standard naming)
    # The MLP LayerNorm is post_attention_layernorm, but the divergence starts earlier. 
    # Let's target the post-attention layer norm, as that is the input to the MLP.
    ln_layer = hf_encoder.layers[0].post_attention_layernorm
    
    # Register the hook to capture the output of this layer
    hook_handle = ln_layer.register_forward_hook(capture_hook)

    # Generate identical inputs
    torch_hiddens, torch_freqs, jax_hiddens, jax_freqs = create_canonical_inputs(config)

    # --- Step 1: Execute HF Forward Pass ---
    # HF expects [S, B, H]
    with torch.no_grad():
        hf_output_tuple = hf_encoder.forward(
            hidden_states=torch_hiddens, #.transpose(0, 1), 
            freqs_ci=torch_freqs,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False
        )
        # Convert back to [B, S, H] for comparison
        hf_last_hidden_state = hf_output_tuple[0] #.transpose(0, 1) 
    
    # Remove the hook immediately after execution
    hook_handle.remove() 
    
    # --- NUMERICAL DIAGNOSTIC STEP ---
    # Extract the PyTorch LayerNorm output slice (index 0, token 0)
    pt_ln_output = captured_output['ln_output'][0, 0, :5]
    
    print("\n--- PYTORCH LAYER NORM DIAGNOSTIC (Layer 0, Pre-MLP) ---")
    # Convert to float32 for clean printing and NumPy safety
    pt_ln_np = pt_ln_output.to(torch.float32).numpy() 
    print(f"PyTorch LayerNorm Output Slice: {pt_ln_np}")
    print("---------------------------------------------------------")

    # 1. Get a reference to the MLP fc1 layer of the first encoder layer
    # We use .data to access the raw tensor content, which is already loaded from disk.
    pt_fc1_kernel = hf_encoder.layers[0].mlp.fc1.weight.data
    pt_fc1_bias = hf_encoder.layers[0].mlp.fc1.bias.data

    # 2. Extract the required slices
    # Slices must be converted to float32 for clean printing (BF16 tensors can be messy to print)
    pt_kernel_slice = pt_fc1_kernel[0, :5].to(torch.float32).numpy()
    pt_bias_slice = pt_fc1_bias[:5].to(torch.float32).numpy()

    # 3. Print the results for manual comparison with JAX output
    print("\n--- PYTORCH KERNEL DIAGNOSTIC (Layer 0 MLP) ---")
    print(f"HF fc1 Kernel (Row 0, Cols 0-4) Slice: {pt_kernel_slice}")
    print(f"HF fc1 Bias (Cols 0-4) Slice: {pt_bias_slice}")
    print("---------------------------------------------")
        
    # --- Step 2: Execute JAX Forward Pass ---
    # JAX encoder __call__ expects (hidden_states: jax.Array, freqs_ci_stacked: jax.Array)
    jax_last_hidden_state = jax_encoder(
        jax_hiddens, 
        jax_freqs,
        num_kv_pages_per_block=DUMMY_NUM_KV_PAGES_PER_BLOCK, 
        num_queries_per_block=DUMMY_NUM_QUERIES_PER_BLOCK,
    )
    
    # --- Step 3: Compare Results ---
    print("\n--- Comparing Llama4VisionEncoder Outputs (Layer 34 Output) ---")
    
    is_correct = compare_outputs(hf_last_hidden_state, jax_last_hidden_state, tolerance=1e-3)

    return is_correct


if __name__ == "__main__":
    # --- CRITICAL FIX: Force JAX to use CPU backend ---
    # This prevents the TPU initialization failure during verification.
    os.environ['JAX_PLATFORMS'] = 'cpu'
    
    # Temporarily remove JAX log prints to clean up output
    os.environ['JAX_DEBUG_LOG_LEVEL'] = 'DEBUG'

    # Ensure jax is fully initialized with the new platform variable before use
    jax.devices() 

    if verify_vision_encoder():
        print("\nSUCCESS: JAX Llama4VisionEncoder matches HuggingFace reference.")
    else:
        print("\nFAILURE: JAX Llama4VisionEncoder output mismatch.")