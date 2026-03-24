import pathwaysutils 
pathwaysutils.initialize()
from jax.experimental import colocated_python
import jax 
import jax.numpy as jnp

@colocated_python.colocated_python
def twice(x):
  out_arrays = []
  for shard in x.addressable_shards:
    out_arrays.append(2 * shard.data)
  return jax.make_array_from_single_device_arrays(
      sharding=x.sharding, shape=x.shape, arrays=out_arrays
  )

tpu_devices = jax.local_devices()
cpu_devices = colocated_python.colocated_cpu_devices(tpu_devices)
cpu_devices_mesh = jax.sharding.Mesh(cpu_devices, "x")
tpu_devices_mesh = jax.sharding.Mesh(tpu_devices, "x")

# Construct input that is sharded across all cpu_devices
x = jnp.array([1] * len(cpu_devices))
x = jax.device_put(
    x, jax.NamedSharding(cpu_devices_mesh, jax.sharding.PartitionSpec("x"))
)

# Get output that is sharded across all cpu_devices
out = twice(x)

# Copy output from cpu_devices into corresponding tpu_devices (without going through the pathways client)
with (
    jax.transfer_guard_device_to_host("disallow_explicit"),
    jax.transfer_guard_host_to_device("disallow_explicit"),
):
  out_tpus = jax.device_put(out, jax.NamedSharding(tpu_devices_mesh, jax.sharding.PartitionSpec("x")))

print(str(out_tpus))