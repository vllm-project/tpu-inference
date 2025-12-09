import functools
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
from fused_moe_input_gen import gen_moe_inputs
from fused_moe_orig import fused_ep_moe as fused_ep_moe_orig
from g3_quant_fused_moe import fused_ep_moe
use_quant = True
mesh_devices = sorted(
    jax.devices(),
    key=lambda x: (
        x.coords[0],
        (-1 if x.coords[0] % 2 else 1) * x.coords[1],
    ),
)
mesh = Mesh(np.array(mesh_devices).reshape(1, -1), axis_names=("data", "model"))

num_experts = 256  # 256(orig), 160 crashes
hidden_size = 6144
intermediate_size = 2560
seed = 54321
top_k = 8

num_tokens = 8192  # 8 * 8 * 4096
dtype = jnp.bfloat16
w_dtype = jnp.float8_e4m3fn
a, w1, w2, gating_output = gen_moe_inputs(
    dtype,
    top_k,
    num_experts,
    hidden_size,
    intermediate_size,
    num_tokens,
    seed=seed,
)
w1_scale = None
w2_scale = None
subc_quant_wsz = None

# w1, w1_scale = sub_channel_quantize(w1, w_dtype, subc_quant_wsz)
# w2, w2_scale = sub_channel_quantize(w2, w_dtype, subc_quant_wsz)

print(f"Inputs generated. a.shape: {a.shape}, w1.shape: {w1.shape}, w2.shape: {w2.shape}, gating_output.shape: {gating_output.shape}")

renormalize_topk_logits = False
bt = 64
bf = 512  # Changed from 1024 to 512
bd1 = 512
bd2 = 512
btc = 32
bfc = 256
bd1c = 256
bd2c = 256
act_fn = "silu"
STATIC_KWARGS = {
    "mesh": mesh,
    "top_k": top_k,
    "bt": 64,
    "bf": 512,  # Changed from 1024 to 512
    "bd1": 512,
    "bd2": 512,
    "btc": 32,
    "bfc": 256,
    "bd1c": 256,
    "bd2c": 256,

    "subc_quant_wsz": subc_quant_wsz,
}
orig_STATIC_KWARGS = STATIC_KWARGS.copy()
orig_STATIC_KWARGS.pop("subc_quant_wsz")
if not use_quant:
    fused_ep_moe_orig_p = functools.partial(fused_ep_moe_orig, **orig_STATIC_KWARGS)
    print("Compiling fused_ep_moe_orig_kernel...")
    fused_ep_moe_orig_kernel  = jax.jit(fused_ep_moe_orig_p).lower(
            tokens=a,
            w1=w1,
            w2=w2,
            gating_output=gating_output,
        ).compile({'xla_tpu_enable_log_recorder': 'true'})
    print("Done compiling fused_ep_moe_orig_kernel.")
    orig_out = fused_ep_moe_orig_kernel(
        tokens=a,
        w1=w1,
        w2=w2,
        gating_output=gating_output,
    ).block_until_ready()
    print("Original output shape:", orig_out.shape)
else:
    fused_ep_moe_debug_p = functools.partial(fused_ep_moe, **STATIC_KWARGS)
    print("Compiling fused_ep_moe_debug_kernel...")
    fused_ep_moe_debug_kernel  = jax.jit(fused_ep_moe_debug_p).lower(
            tokens=a,
            w1=w1,
            w2=w2,
            gating_output=gating_output,
        ).compile({'xla_tpu_enable_log_recorder': 'true'})
    print("Done compiling fused_ep_moe_debug_kernel.")
    debug_out = fused_ep_moe_debug_kernel(
        tokens=a,
        w1=w1,
        w2=w2,
        gating_output=gating_output,
    ).block_until_ready()
    print("Actual output shape:", debug_out.shape)
