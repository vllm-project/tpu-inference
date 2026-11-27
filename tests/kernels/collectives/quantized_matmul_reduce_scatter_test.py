# SPDX-License-Identifier: Apache-2.0

import collections

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference import utils
from tpu_inference.kernels.collectives import \
    quantized_matmul_reduce_scatter as qmrs
from tpu_inference.kernels.quantized_matmul.util import quantize_tensor

jax.config.parse_flags_with_absl()

P = jax.sharding.PartitionSpec
NamedSharding = jax.sharding.NamedSharding

FakeDevice = collections.namedtuple("FakeDevice", ["coords"])


def _reference(x, w, s, axis, int8_act):
    """xla_quantized_matmul followed by psum_scatter, in f32."""
    if int8_act:
        xq, sx = qmrs_quantize(x)
        acc = jax.lax.dot(xq, w, preferred_element_type=jnp.int32)
        acc = acc.astype(jnp.float32) * sx.reshape(-1, 1)
    else:
        acc = jax.lax.dot(x,
                          w.astype(jnp.bfloat16),
                          preferred_element_type=jnp.float32)
    return jax.lax.psum_scatter(acc * s[None, :],
                                axis,
                                scatter_dimension=0,
                                tiled=True)


def qmrs_quantize(x):
    return quantize_tensor(x, jnp.int8)


class RingOrderTest(parameterized.TestCase):

    def _check_cycle(self, coords):
        devices = [FakeDevice(coords=c) for c in coords]
        order = qmrs.ring_order(devices)
        self.assertEqual(sorted(order), list(range(len(coords))))
        for a, b in zip(order, order[1:] + order[:1]):
            ca, cb = coords[a], coords[b]
            dist = abs(ca[0] - cb[0]) + abs(ca[1] - cb[1])
            self.assertEqual(dist, 1, (order, ca, cb))

    @parameterized.parameters((2, 2), (2, 4), (4, 2), (4, 4), (3, 2), (2, 3),
                              (4, 3))
    def test_grid_is_neighbour_cycle(self, nx, ny):
        coords = [(x, y, 0) for y in range(ny) for x in range(nx)]
        # Shuffle so the result cannot rely on device order.
        rng = np.random.default_rng(nx * 10 + ny)
        coords = [coords[i] for i in rng.permutation(len(coords))]
        self._check_cycle(coords)

    def test_odd_by_odd_falls_back_to_axis_order(self):
        coords = [(x, y, 0) for y in range(3) for x in range(3)]
        devices = [FakeDevice(coords=c) for c in coords]
        self.assertEqual(qmrs.ring_order(devices), list(range(9)))

    def test_no_coords(self):
        self.assertEqual(qmrs.ring_order([object()] * 4), [0, 1, 2, 3])


class ChooseBlocksTest(parameterized.TestCase):

    def test_gemma_shapes(self):
        # Gemma 4 31B down_proj / o_proj per chip at TP=4, T=2048.
        self.assertEqual(qmrs.choose_blocks(2048, 5376, 5376, 4, True),
                         (896, 896, True))
        self.assertEqual(qmrs.choose_blocks(2048, 4096, 5376, 4, False),
                         (896, 1024, True))

    @parameterized.parameters((16, 5376, 5376, 4), (128, 5376, 5376, 4),
                              (2048 + 128, 5376, 5376, 4),
                              (16384, 5376, 5376, 4))
    def test_unsupported(self, T, K, N, D):
        self.assertIsNone(qmrs.choose_blocks(T, K, N, D, True))

    def test_odd_tile_count_is_unidirectional(self):
        self.assertEqual(qmrs.choose_blocks(1024, 1024, 640, 4, False),
                         (640, 1024, False))


@jtu.with_config(jax_numpy_dtype_promotion='standard')
class QuantizedMatmulReduceScatterTest(jtu.JaxTestCase):

    @parameterized.product(
        int8_act=[True, False],
        bidir=[True, False],
        shape=[(1024, 4096, 1792), (2048, 21504, 5376), (2048, 8192, 5376)],
    )
    def test_matches_psum_scatter(self, int8_act, bidir, shape):
        if not jtu.is_device_tpu() or jax.device_count() < 2:
            self.skipTest('Needs at least 2 TPU devices')
        axis = 'x'
        D = jax.device_count()
        T, K, N = shape
        if T % (D * 128) or K % (D * 128):
            self.skipTest('Shape does not split over this many devices')
        mesh = utils.make_optimized_mesh((D, ), (axis, ))
        order = qmrs.ring_order(list(mesh.devices.flat))

        def fused(x, w, s):
            sx = None
            if int8_act:
                x, sx = qmrs_quantize(x)
            plan = qmrs.choose_blocks(x.shape[0], x.shape[1], w.shape[1], D,
                                      int8_act)
            return qmrs.quantized_matmul_reduce_scatter(x,
                                                        w,
                                                        s,
                                                        axis=axis,
                                                        order=order,
                                                        sx=sx,
                                                        bn=plan[0],
                                                        bk=plan[1],
                                                        bidir=bidir
                                                        and plan[2])

        def ref(x, w, s):
            return _reference(x, w, s, axis, int8_act)

        in_specs = (P(None, axis), P(axis, None), P(None))
        f_fused = jax.jit(
            jax.shard_map(fused,
                          mesh=mesh,
                          in_specs=in_specs,
                          out_specs=P(axis, None),
                          check_vma=False))
        f_ref = jax.jit(
            jax.shard_map(ref,
                          mesh=mesh,
                          in_specs=in_specs,
                          out_specs=P(axis, None),
                          check_vma=False))

        # Several calls on fresh data to expose races between calls.
        for i in range(5):
            k0, k1, k2 = jax.random.split(jax.random.key(1234 + i), 3)
            x = jax.device_put(jax.random.normal(k0, (T, K), jnp.bfloat16),
                               NamedSharding(mesh, in_specs[0]))
            w = jax.device_put(
                jax.random.randint(k1, (K, N), -127, 128, jnp.int8),
                NamedSharding(mesh, in_specs[1]))
            s = jax.device_put(
                jax.random.uniform(k2, (N, ), jnp.float32, 1e-3, 2e-3),
                NamedSharding(mesh, in_specs[2]))
            out = np.asarray(f_fused(x, w, s), np.float32)
            expected = np.asarray(f_ref(x, w, s), np.float32)
            # Partial sums travel in bf16: allow bf16 rounding per ring step.
            err = np.max(np.abs(out - expected)) / np.max(np.abs(expected))
            self.assertLess(err, 2e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
