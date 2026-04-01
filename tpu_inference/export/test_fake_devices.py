"""Unit test to verify fake TPU devices on CPU using jax.experimental.topologies."""

import unittest
import jax
from jax.experimental import topologies

class TestFakeDevices(unittest.TestCase):
    def test_get_topology_desc(self):
        # This test should run on CPU to verify we can get fake TPU devices
        if jax.devices()[0].device_kind != "cpu":
             self.skipTest("This test is for CPU only (to simulate TPU on CPU)")
             
        topology_name = "tpu7x_2x4"

        try:
             topology_desc = topologies.get_topology_desc(topology_name)
             devices = topology_desc.devices
             print(f"Fake devices: {devices}")
             self.assertGreater(len(devices), 0, "No fake devices returned")
             print(f"Successfully loaded {topology_name} topology with {len(devices)} devices")
             print(f"Fake device kind (version): {devices[0].device_kind}")
             
             # Verify we can create a mesh
             from jax.sharding import Mesh
             # Create a simple 1D mesh
             mesh = Mesh(devices, ('x',))
             self.assertEqual(mesh.shape['x'], len(devices))
             print(f"Successfully created mesh with fake devices: {mesh}")
             
        except Exception as e:
             self.fail(f"Failed to load topology or create mesh for {topology_name}: {e}")

if __name__ == '__main__':
    unittest.main()
