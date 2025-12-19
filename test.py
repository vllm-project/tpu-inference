import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax._src.layout import Layout
from jax._src.lib import xla_client as xc
# Import internal handler
from jax._src.array import _array_shard_arg

def repro_synthetic_recursion():
    print("Setting up SYNTHETIC reproduction...")
    print("This script simulates the JIT behavior to prove the infinite loop in array.py.")

    devices = jax.devices()
    if not devices:
        print("No JAX devices found.")
        return

    # 1. FIX: Setup Mesh correctly with reshape
    # We need a 1x1 array for the 2 axes ('data', 'model')
    device_array = np.array(devices[:1]).reshape(1, 1)
    mesh = Mesh(device_array, ('data', 'model'))
    sharding = NamedSharding(mesh, PartitionSpec())
    
    # 2. Create Input 'x'
    shape = (32768, 64)
    dtype = jnp.bfloat16
    x_host = jnp.ones(shape, dtype=dtype)
    x = jax.device_put(x_host, sharding)
    
    print(f"Input X created: {x.shape}")

    # 3. Define Layouts
    try:
        # We need ANY layout here, just to pass into the function
        target_layout = Layout((1, 0)) # Col Major
    except:
        print("Layouts not supported in this JAX version.")
        return

    print(f"\nTarget Layout: {target_layout}")
    print("Triggering recursion logic...")

    # 4. Mock api.device_put to verify the loop
    # In the real crash: _array_shard_arg -> api.device_put -> jit -> _array_shard_arg
    
    original_device_put = jax._src.api.device_put
    
    def mock_device_put(val, fmt):
        print("  [Mock] api.device_put called (simulating JIT compilation...)")
        print(f"  [Mock] JIT checks input 'x' and recurses back into shard_arg...")
        
        # This simulates the JIT argument checker seeing the input 'val'
        # and checking if it matches the requirements.
        copy_semantics = [xc.ArrayCopySemantics.REUSE_INPUT] 
        _array_shard_arg([val], [fmt.sharding], [fmt.layout], copy_semantics)
        
        return val

    # Apply Patch
    jax._src.api.device_put = mock_device_put

    try:
        print("Entering _array_shard_arg (Loop Start)...")
        copy_semantics = [xc.ArrayCopySemantics.REUSE_INPUT] 
        
        # We try to shard 'x' to 'target_layout'.
        # Since we patched device_put, this MUST recurse if it enters the mismatch block.
        _array_shard_arg([x], [sharding], [target_layout], copy_semantics)
        
    except RecursionError:
        print("\n[CONFIRMED] RecursionError captured!")
        print("The logic in _array_shard_arg creates an infinite loop when JIT enforces layout.")
        raise
    finally:
        # Restore Patch
        jax._src.api.device_put = original_device_put

if __name__ == "__main__":
    repro_synthetic_recursion()