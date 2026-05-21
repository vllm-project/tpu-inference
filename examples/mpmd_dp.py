import multiprocessing
import jax 
import jax.numpy as jnp
import pathwaysutils 
from jax.sharding import NamedSharding
pathwaysutils.initialize()

### Experiment 1
# run two workloads on two subslices (16 devices total, run one workload on 8 devices and the other workload on the other 8 devices)

mesh_a = jax.sharding.Mesh(jax.devices()[:8], ('dp',))
mesh_b = jax.sharding.Mesh(jax.devices()[8:], ('dp',))
sharding_a = NamedSharding(mesh_a, jax.sharding.PartitionSpec('dp'))
sharding_b = NamedSharding(mesh_b, jax.sharding.PartitionSpec('dp'))


a = jax.device_put(jnp.ones((16, 16)), sharding_a)
b = jax.device_put(jnp.ones((16, 16)), sharding_b)
print(a)
print(b)
# what happens if I do a + b? will it automatically transfer data between the two subslices?

# c = a + b
# print(c)
# Fails with ValueError: Received incompatible devices for jitted computation. Got argument x of add with shape float32[16,16] and device ids [0, 1, 2, 3, 4, 5, 6, 7] on platform TPU and argument y of add with shape float32[16,16] and device ids [8, 9, 10, 11, 12, 13, 14, 15] on platform TPU

### Experiment 2
# Run two worklods on two processes. 

def my_function(sharding):
    a = jax.device_put(jnp.ones((16, 16)), sharding)
    jax.block_until_ready(a)
    print(f"Process {multiprocessing.current_process().name} has sharding {sharding} and a {a}", flush=True)
    return a

p1 = multiprocessing.Process(target=my_function, args=(sharding_a,))
p2 = multiprocessing.Process(target=my_function, args=(sharding_b,))

# Start the process
p1.start()
p2.start()

# Optional: Wait for the process to finish
p1.join()
p2.join()

print("Both processes have finished execution.")