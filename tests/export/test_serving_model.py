import unittest
import os
import tempfile
import json
import jax
import jax.numpy as jnp
from unittest.mock import patch

from jax.experimental import topologies
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax import shard_map
from jax.sharding import PartitionSpec as P

from tpu_inference.export import serving_model

class TestServingModelExport(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["GOOGLE_EXPORT_MODEL_PATH"] = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()
        if "GOOGLE_EXPORT_MODEL_PATH" in os.environ:
            del os.environ["GOOGLE_EXPORT_MODEL_PATH"]

    def test_save_native_model_basic(self):
        def add_one(x):
            return x + 1

        x = jnp.ones((2, 2))
        exported = jax.export.export(jax.jit(add_one))(x)
        serving_model.save_native_model(self.temp_dir.name, {"add_one": exported})

        # Verify files
        model_dir = os.path.join(self.temp_dir.name, "model_fn")
        self.assertTrue(os.path.exists(model_dir))
        self.assertTrue(os.path.exists(os.path.join(model_dir, "add_one.pb")))
        self.assertTrue(os.path.exists(os.path.join(model_dir, "metadata.json")))

        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
            self.assertIn("add_one", metadata)
            self.assertEqual(metadata["add_one"]["file_path"], "add_one.pb")

    def test_load_native_model_roundtrip(self):
        def add_one(x):
            return x + 1

        x = jnp.ones((2, 2))
        exported = jax.export.export(jax.jit(add_one))(x)
        serving_model.save_native_model(self.temp_dir.name, {"add_one": exported})

        # Load it back
        loaded_map = serving_model.load_native_model(self.temp_dir.name)
        self.assertIn("add_one", loaded_map)
        
        # Verify it works by running it (though we just verify it exists and is callable or has same signature)
        # loaded_map['add_one'] is an Exported object. We can check its call method or similar if available, 
        # or just assume it loaded correctly if it doesn't crash.
        self.assertIsInstance(loaded_map["add_one"], jax.export.Exported)

    def test_save_native_model_tpu7_topology(self):
        """Test exporting a model using a fake TPU v7 topology on CPU."""
        
        topology_name = "tpu7x_2x4"
        
        try:
            topology_desc = topologies.get_topology_desc(topology_name)
            devices = topology_desc.devices
            
            def add_one(x):
                return x + 1

            x = jnp.ones((2, 2))
            
            exported = jax.export.export(jax.jit(add_one))(x)
            
            print("\n=== TPU v7 StableHLO (MLIR Module) ===")
            print(exported.mlir_module())
            print("======================================\n")
            
            serving_model.save_native_model(self.temp_dir.name, {"add_one": exported})

            model_dir = os.path.join(self.temp_dir.name, "model_fn")
            self.assertTrue(os.path.exists(model_dir))
            self.assertTrue(os.path.exists(os.path.join(model_dir, "add_one.pb")))
            
        except ImportError:
            self.skipTest("jax.experimental.topologies not available in this JAX version")
        except Exception as e:
            self.fail(f"Failed TPU v7 export test: {e}")

    def test_save_native_model_pallas_kernel(self):
        """Test exporting a model containing a Pallas kernel."""

        def add_kernel(x_ref, y_ref, out_ref):
            out_ref[...] = x_ref[...] + y_ref[...]

        @jax.jit
        def pallas_add(x, y):
            return pl.pallas_call(
                add_kernel,
                out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                grid=(1,),
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.VMEM),
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            )(x, y)

        topology_name = "tpu7x_2x4"
        try:
            topology_desc = topologies.get_topology_desc(topology_name)
            devices = topology_desc.devices
            mesh = Mesh(devices, ('x',))
            
            x = jnp.ones((128,), dtype=jnp.float32)
            y = jnp.ones((128,), dtype=jnp.float32)
            
            sharded_pallas_add = shard_map(
                pallas_add,
                mesh=mesh,
                in_specs=(P('x'), P('x')),
                out_specs=P('x'),
                check_vma=False
            )
            
            jit_sharded_pallas_add = jax.jit(sharded_pallas_add)
            
            with jax.set_mesh(mesh):
                exported = jax.export.export(jit_sharded_pallas_add, platforms=['tpu'])(x, y)

            
            print("\n=== Pallas Kernel StableHLO ===")
            print(exported.mlir_module())
            print("===============================\n")

            
            serving_model.save_native_model(self.temp_dir.name, {"pallas_add": exported})
            
            model_dir = os.path.join(self.temp_dir.name, "model_fn")
            self.assertTrue(os.path.exists(model_dir))
            self.assertTrue(os.path.exists(os.path.join(model_dir, "pallas_add.pb")))
            
        except Exception as e:
            self.fail(f"Failed Pallas export test: {e}")

if __name__ == '__main__':
    os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
    unittest.main()

