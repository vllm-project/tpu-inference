import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax._src.layout import Layout, Format
from jax._src.lib import xla_client as xc
from jax._src import api
# Import the internal handler that causes the crash
from jax._src.array import _array_shard_arg

def repro_internal_handler_recursion():
    print("Setting up INTERNAL HANDLER reproduction...")

    # 1. Setup Mesh (1x1)
    devices = jax.devices()
    if not devices:
        print("No JAX devices found.")
        return

    device_array = np.array(devices[:1]).reshape(1, 1)
    mesh = Mesh(device_array, ('data', 'model'))
    sharding = NamedSharding(mesh, PartitionSpec())

    shape = (32768, 64)
    dtype = jnp.bfloat16

    # 2. Create Input 'x' with Forced (1, 0) Layout
    print("Creating Input 'x' with Col-Major (1, 0) layout...")
    try:
        col_major_layout = Layout((1, 0)) 
        col_major_format = Format(col_major_layout, sharding)
        x_host = jnp.ones(shape, dtype=dtype)
        x = jax.device_put(x_host, col_major_format)
    except:
        # Fallback if forcing fails, though it might not repro if layout matches target
        x = jax.device_put(jnp.ones(shape, dtype=dtype), sharding)

    # 3. Define Target Layout (0, 1) with Tiling
    # This must be DIFFERENT from input to trigger the conversion logic
    try:
        target_layout = Layout((0, 1), tiling=((16, 128), (2, 1)))
    except TypeError:
        target_layout = Layout((0, 1))

    print(f"\nInput Layout: {getattr(x, 'layout', 'Unknown')}")
    print(f"Target Layout: {target_layout}")
    print("\n--- Triggering Recursion ---")
    print("We are calling `_array_shard_arg` directly.")
    print("This simulates JIT discovering a mismatch deeply nested in compilation.")
    
    try:
        # We pass copy_semantics to match the signature required by the handler
        copy_semantics = [xc.ArrayCopySemantics.REUSE_INPUT] 
        
        # This call is what happens inside JIT when it checks arguments.
        # 1. It sees mismatch (1,0) != (0,1).
        # 2. It calls api.device_put(x, target_format).
        # 3. api.device_put triggers a JIT compilation for the transpose.
        # 4. That internal JIT checks 'x' again -> calls _array_shard_arg -> Loop.
        _array_shard_arg([x], [sharding], [target_layout], copy_semantics)
        
        print("\n[Unexpected] Success! No recursion.")
    except RecursionError:
        print("\n[SUCCESS] RecursionError Captured!")
        print("This confirms the infinite loop in array.py.")
        
        # If you see this, you are ready to apply the fix.
        print("\nFix: In jax/_src/array.py, replace `api.device_put(...)` with `dispatch._device_put_impl(...)`.")
    except Exception as e:
        print(f"\nCaught unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    repro_internal_handler_recursion()