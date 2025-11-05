import jax.numpy as jnp
import numpy as np
import torch
import os
from typing import Tuple, Dict, Any

# --- CONFIGURATION ---
RNG_SEED = 42
SEQUENCE_LENGTH = 577
HIDDEN_SIZE = 1408
DTYPE = torch.bfloat16
JAX_DTYPE = jnp.bfloat16

# --- INPUT/OUTPUT FILES ---
INPUT_FILE = "canonical_input_data.npz"
WEIGHTS_FILE = "canonical_mlp_weights.npz"
OUTPUT_FILE = "pytorch_encoder_output.npy"

# --- Import PyTorch/HF Components (Must be run in a Python/PyTorch environment) ---
try:
    from transformers.models.llama4.modeling_llama4 import (
        Llama4VisionEncoder, 
        Llama4VisionConfig, 
        Llama4VisionRotaryEmbedding
    )
except ImportError:
    print("FATAL ERROR: Please run this script in an environment with HuggingFace Transformers and PyTorch installed.")
    exit()

# -------------------------- CORE LOGIC --------------------------

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

def generate_canonical_data(config: Llama4VisionConfig) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generates random inputs and canonical weights/frequencies."""
    
    # 1. Generate Random Hidden States (Initial Input to the Encoder)
    torch_hiddens = torch.randn(
        1, config.hidden_size, SEQUENCE_LENGTH, dtype=DTYPE
    ).transpose(1, 2) # Shape [B, S, D]
    
    # 2. Generate Canonical RoPE Frequencies (Fixed 2D position embedding math)
    hf_rope = Llama4VisionRotaryEmbedding(config)
    dummy_hiddens = torch.empty(1, 1, 1) 
    torch_freqs = hf_rope.forward(dummy_hiddens).to(DTYPE) # Complex tensor
    
    # 3. Save JAX-Friendly Inputs
    jax_hiddens_np = torch_hiddens.to(torch.float32).numpy()
    
    torch_freqs_complex_cpu = torch_freqs.cpu().to(torch.complex64)
    freqs_real_imag = torch.view_as_real(torch_freqs_complex_cpu).numpy()
    jax_freqs_ci_np = freqs_real_imag # [S, D_rot, 2]
    
    # 4. Extract Ground-Truth MLP Weights (for later injection into JAX model)
    hf_encoder = Llama4VisionEncoder(config=config)
    weights_to_save = {}

    for i in range(config.num_hidden_layers):
        pt_layer = hf_encoder.layers[i]

        prefix = f'layer.{i}.attn.' # New prefix for attention weights

        # --- Attention Kernels ---
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            pt_attn_linear = getattr(pt_layer.self_attn, name)
            
            # Extract WEIGHT
            attn_weight = pt_attn_linear.weight.data.to(torch.float32).numpy()
            weights_to_save[f'{prefix}{name}.kernel'] = attn_weight
        
        # --- MLP Kernels/Biases ---
        for name, layer in [('fc1', pt_layer.mlp.fc1), ('fc2', pt_layer.mlp.fc2)]:
            prefix = f'layer.{i}.mlp.{name}'
            # Kernel [O, I] -> Save as is, will be transposed on JAX side
            weights_to_save[f'{prefix}.kernel'] = layer.weight.data.to(torch.float32).numpy()
            weights_to_save[f'{prefix}.bias'] = layer.bias.data.to(torch.float32).numpy()
            
        # --- LayerNorm Scales/Biases ---
        for name in ['input_layernorm', 'post_attention_layernorm']:
            pt_ln = getattr(pt_layer, name)
            prefix = f'layer.{i}.ln.{name}'
            weights_to_save[f'{prefix}.scale'] = pt_ln.weight.data.to(torch.float32).numpy()
            weights_to_save[f'{prefix}.bias'] = pt_ln.bias.data.to(torch.float32).numpy()
            
    return torch_hiddens, torch_freqs, jax_hiddens_np, jax_freqs_ci_np, weights_to_save

def run_pytorch_execution(config: Llama4VisionConfig, torch_hiddens: torch.Tensor, torch_freqs: torch.Tensor) -> np.ndarray:
    """Runs the PyTorch encoder to get the ground truth numerical output."""
    
    hf_encoder = Llama4VisionEncoder(config=config)
    hf_encoder.to(DTYPE).eval()

    print("[PART 2: PYTORCH] Executing HF model to get ground truth...")
    with torch.no_grad():
        hf_output_tuple = hf_encoder.forward(
            hidden_states=torch_hiddens,
            freqs_ci=torch_freqs,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False
        )
        pt_output = hf_output_tuple[0]
        # Save output in float32 NumPy format
        pt_output_np = pt_output.cpu().to(torch.float32).numpy()
    
    return pt_output_np

def save_inputs(jax_hiddens_np, jax_freqs_ci_np):
    """Saves the JAX-ready inputs."""
    np.savez(INPUT_FILE, hiddens=jax_hiddens_np, freqs=jax_freqs_ci_np)
    print(f"\nInputs saved to {INPUT_FILE}")


if __name__ == "__main__":
    config = create_config()
    
    # 1. Generate all data (hiddens, freqs, weights)
    torch_hiddens, torch_freqs, jax_hiddens_np, jax_freqs_ci_np, weights_to_save = generate_canonical_data(config)

    # 2. Save Inputs and Weights to files
    save_inputs(jax_hiddens_np, jax_freqs_ci_np)
    np.savez(WEIGHTS_FILE, **weights_to_save)
    print(f"Weights saved to {WEIGHTS_FILE}")
    
    # 3. Run PyTorch Execution and save the ground truth output
    pt_output_np = run_pytorch_execution(config, torch_hiddens, torch_freqs)
    np.save(OUTPUT_FILE, pt_output_np)
    print(f"PyTorch Ground Truth Output saved to {OUTPUT_FILE}")
    
    print("\n--- Execution Complete. Transfer files to TPU host. ---")