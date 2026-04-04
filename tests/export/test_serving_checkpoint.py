import os
import tempfile
import unittest

import jax
import jax.numpy as jnp
from tpu_inference.export import serving_checkpoint

class TestServingCheckpoint(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_load_basic(self):
        """Test saving and loading a simple PyTree without sharding overrides."""
        params = {
            "linear": {
                "w": jnp.ones((2, 4), jnp.float32),
                "b": jnp.zeros((4,), jnp.float32),
            },
            "embedding": jnp.ones((10, 8), jnp.float32),
        }
        
        # Create an AbstractMesh for saving
        abstract_mesh = jax.sharding.AbstractMesh(axis_sizes=(1,), axis_names=("x",))
        
        # Save
        serving_checkpoint.save(
            params=params,
            output_dir=self.output_dir,
            abstract_mesh=abstract_mesh,
            description="BasicPyTreeTest",
        )
        
        # Verify files created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "metadata.json")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "model_params_shape_dtype_struct.jax_exported")))
        
        # Load
        loaded_params = serving_checkpoint.load(self.output_dir)
        
        # Verify shape and dtype
        self.assertEqual(loaded_params["linear"]["w"].shape, (2, 4))
        self.assertEqual(loaded_params["linear"]["w"].dtype, jnp.float32)
        self.assertEqual(loaded_params["embedding"].shape, (10, 8))
        self.assertEqual(loaded_params["embedding"].dtype, jnp.float32)

    def test_save_and_load_with_sharding_and_mesh(self):
        """Test saving and loading with sharding overrides using a simulated mesh."""
        
        # We need a real mesh for testing load with sharding, requiring some devices.
        # Assuming xla_force_host_platform_device_count is set to 8 in environment.
        devices = jax.devices()
        if len(devices) < 2:
            self.skipTest("Requires at least 2 devices for sharding test")
            
        mesh = jax.sharding.Mesh(devices[:2], ("x",))
        abstract_mesh = jax.sharding.AbstractMesh(axis_sizes=(2,), axis_names=("x",))
        
        params = {
            "w": jnp.ones((8, 8), jnp.float32),
        }
        
        # Define sharding for override
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x", None))
        params_sharding_override = {
            "w": sharding,
        }
        
        # Save with override
        serving_checkpoint.save(
            params=params,
            output_dir=self.output_dir,
            abstract_mesh=abstract_mesh,
            params_sharding_override=params_sharding_override,
            description="ShardingOverrideTest",
        )
        
        # Load with mesh
        loaded_params = serving_checkpoint.load(self.output_dir, mesh=mesh)
        
        # Verify shape, dtype and sharding
        self.assertEqual(loaded_params["w"].shape, (8, 8))
        self.assertEqual(loaded_params["w"].dtype, jnp.float32)
        
        # In lazy mode, it returns ShapeDtypeStruct which has a 'sharding' attribute
        self.assertTrue(hasattr(loaded_params["w"], "sharding"))
        self.assertEqual(loaded_params["w"].sharding, sharding)

if __name__ == "__main__":
    unittest.main()
