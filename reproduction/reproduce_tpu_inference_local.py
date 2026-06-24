import time
import sys
import os

# Ensure we import the local tpu_inference package from the repository
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# 1. Enable JAX compilation cache if JAX_COMPILATION_CACHE_DIR is set
cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
if cache_dir:
    print(f"Enabling JAX compilation cache at: {cache_dir}", file=sys.stderr, flush=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

# 2. Get active mesh (local TPU devices)
devices = jax.local_devices()
mesh = Mesh(devices, ('data',))
print(f"Device mesh initialized with {len(devices)} local devices: {mesh}", file=sys.stderr, flush=True)

from tpu_inference.layers.jax import JaxModule
from tpu_inference.models.common.pathways_dummy_loader import load_dummy_weights_jax

# 3. Define a minimal dummy JaxModule that simulates a large model structure
NUM_PARAMETERS = 185
dim = 1024

class ToyJaxModel(JaxModule):
    def __init__(self, num_layers=NUM_PARAMETERS, dim=dim):
        sharding_spec = P('data',)
        
        # Generate 20 unique shapes
        unique_shapes = []
        for i in range(20):
            unique_shapes.append((dim, 1024 * (i + 1)))
            
        # Add parameters
        for i in range(num_layers):
            shape = unique_shapes[i % len(unique_shapes)]
            val = jax.random.uniform(jax.random.PRNGKey(0), shape=shape, dtype=jnp.float32)
            param = nnx.Param(val, sharding=sharding_spec, mesh=mesh)
            setattr(self, f"weight_{i}", param)

# 4. Instantiate the model
print("Instantiating Toy JAX Model...", file=sys.stderr, flush=True)
model = ToyJaxModel()

# 5. Measure weight loading time
print("Loading dummy weights via tpu_inference load_dummy_weights_jax...", file=sys.stderr, flush=True)
with jax.set_mesh(mesh):
    t0 = time.perf_counter()
    load_dummy_weights_jax(model, mesh)
    t1 = time.perf_counter()

print(f"\n==============================================", file=sys.stdout, flush=True)
print(f"LOCAL TPU_INFERENCE REPRODUCTION RESULTS", file=sys.stdout, flush=True)
print(f"Total parameters loaded: {NUM_PARAMETERS}", file=sys.stdout, flush=True)
print(f"Total weight loading time: {t1 - t0:.2f} seconds", file=sys.stdout, flush=True)
print(f"Average time per parameter: {(t1 - t0)/NUM_PARAMETERS:.4f} seconds", file=sys.stdout, flush=True)
print(f"JAX Caching State: TPU_INF_ENABLE_JAX_CACHE={os.environ.get('TPU_INF_ENABLE_JAX_CACHE', '1')}", file=sys.stdout, flush=True)
print(f"==============================================", file=sys.stdout, flush=True)
