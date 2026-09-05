# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for hierarchical_reduce_scatter, including FP8 comm quality eval."""

import os
from unittest import mock

import importlib.util as _importlib_util

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental import shard_map

# Load the kernel by file path, bypassing tpu_inference/__init__.py (which
# eagerly imports env_override -> vllm -> transformers -> torchvision). The
# kernel is pure jax with zero tpu_inference imports, so loading it standalone
# keeps this kernel unit test hermetic and runnable even when the serving
# stack (torch / torch_tpu / torchvision) is broken or version-skewed -- a
# kernel test shouldn't depend on the whole vLLM import graph.
# hierrs_tc is a PACKAGE with intra-package imports, so the standalone
# spec_from_file_location trick the monolith used cannot load it. A normal
# import works and is verified not to drag in the vLLM graph: the package
# imports only jax + its own modules.
from tpu_inference.kernels.collectives.hierrs_tc import config as hrs_config
from tpu_inference.kernels.collectives.hierrs_tc import wrapper as hrs

jax.config.parse_flags_with_absl()

P = jax.sharding.PartitionSpec


def _make_mesh(num_devices: int, axis_name: str) -> jax.sharding.Mesh:
    """Physically-ordered 1D mesh, inlined from the tpu7x path of
    tpu_inference.utils.make_optimized_mesh.

    Importing tpu_inference.utils pulls in torchax, which re-registers the
    PrivateUse1 backend and collides with the already-registered 'tpu' backend
    under pytest. The kernel is jax-only, so this test builds its mesh directly:
    sort devices by physical coords (so ids 2k/2k+1 are the two chiplets of one
    chip, which the kernel's topology math assumes) and hand them to make_mesh.
    """
    devices = sorted(jax.devices(), key=lambda d: d.coords)[:num_devices]
    try:
        axis_types = (jax.sharding.AxisType.Auto,)
        return jax.make_mesh((num_devices,), (axis_name,), axis_types,
                             devices=devices)
    except Exception:
        return jax.sharding.Mesh(
            np.asarray(devices).reshape((num_devices,)), (axis_name,))

SpongeDir: str | None = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', None)

# Minimum SNR (dB) we require for FP8 comm mode. FP8 E4M3FN gives ~2.4 bits
# of mantissa, so we expect meaningful but bounded degradation.
_MIN_SNR_DB = 20.0

# Static scaling uses a single fixed scale for every chunk instead of a
# per-chunk max-abs, so small partial sums under-utilize the fp8 range and
# quality is expected to be somewhat below the dynamic path. A well-chosen
# scale should still clear this bar.
_MIN_STATIC_SNR_DB = 12.0

# The bf16 kernel and the psum reference sum the same values in different
# orders, so they differ by bf16 reassociation error -- not by one ULP. An
# elementwise rtol cannot express that: wherever the true value sits near zero
# the denominator vanishes while the absolute error does not. Measured gap on
# the shape below is a flat 47.9 dB across every micro-batch count and seed
# (results/measure_bf16_tol.py), so 40 dB leaves ~2x margin in power while
# still catching a genuinely wrong reduction, which lands far lower.
_MIN_BF16_SNR_DB = 40.0

# How much worse than the psum reference the kernel may be against an exact
# f32 ground truth. Measured: kernel 0.0053-0.0061, reference 0.0058-0.0063 --
# the kernel is actually the more accurate of the two at several seeds, so this
# is a regression guard rather than a fitted bound.
_BF16_TRUTH_SLACK = 1.5


def _snr_db(signal: jax.Array, noise: jax.Array) -> float:
    """Signal-to-noise ratio in dB: 10*log10(||signal||^2 / ||noise||^2)."""
    signal_power = float(jnp.sum(signal.astype(jnp.float32)**2))
    noise_power = float(jnp.sum(noise.astype(jnp.float32)**2))
    if noise_power == 0.0:
        return float('inf')
    return 10.0 * jnp.log10(signal_power / noise_power).item()


def _reference_reduce_scatter(x: jax.Array, mesh: jax.sharding.Mesh,
                               in_specs: P) -> jax.Array:
    """Reference: psum over 'x' axis then slice each device's shard."""
    axis_name = mesh.axis_names[0]
    num_devices = mesh.devices.size

    def inner(local_x):
        reduced = jax.lax.psum(local_x, axis_name=axis_name)
        idx = jax.lax.axis_index(axis_name)
        chunk = local_x.shape[0] // num_devices
        return jax.lax.dynamic_slice_in_dim(reduced, idx * chunk, chunk, axis=0)

    return shard_map.shard_map(
        inner,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=in_specs,
        check_rep=False,
    )(x)


@jtu.with_config(jax_numpy_dtype_promotion='standard')
class HierarchicalReduceScatterTest(jtu.JaxTestCase):

    def _requires_devices(self, n: int):
        if jax.device_count() < n:
            self.skipTest(f'Need {n} devices, got {jax.device_count()}')

    # ------------------------------------------------------------------ #
    # Correctness: bf16 mode should match reference psum+slice            #
    # ------------------------------------------------------------------ #
    @parameterized.product(
        num_micro_batches=[1, 2, 4],
    )
    def test_correctness_bf16(self, num_micro_batches):
        self._requires_devices(8)
        axis_name = 'x'
        num_devices = jax.device_count()
        mesh = _make_mesh(num_devices, axis_name)

        seq_len, hidden = 1024, 4096
        in_specs = P(axis_name, None)

        for seed in range(3):
            x = jax.random.normal(jax.random.key(seed), (seq_len, hidden),
                                  dtype=jnp.bfloat16)
            x_sharded = jax.device_put(
                x, jax.sharding.NamedSharding(mesh, in_specs))

            ref = _reference_reduce_scatter(x_sharded, mesh, in_specs)
            out = hrs.hierarchical_reduce_scatter(
                x_sharded,
                mesh=mesh,
                in_specs=in_specs,
                num_micro_batches=num_micro_batches,
                fp8_comm=False,
            )

            ref_f32 = jnp.asarray(ref, jnp.float32)
            out_f32 = jnp.asarray(out, jnp.float32)

            # 1. Agreement with the reference, as SNR. Both are valid bf16
            #    reduce-scatters that sum in different orders; see
            #    _MIN_BF16_SNR_DB for why elementwise rtol cannot be used here.
            snr = _snr_db(ref_f32, out_f32 - ref_f32)
            self.assertGreater(
                snr, _MIN_BF16_SNR_DB,
                msg=f'bf16 mb={num_micro_batches} seed={seed}: SNR {snr:.1f} dB '
                    f'< {_MIN_BF16_SNR_DB} dB vs psum reference')

            # 2. Accuracy against an exact f32 ground truth -- no collective
            #    involved, so this catches the case where kernel AND reference
            #    drift together, which (1) would happily pass.
            truth = jnp.asarray(x, jnp.float32).reshape(
                num_devices, seq_len // num_devices, hidden).sum(axis=0)
            scale = float(jnp.max(jnp.abs(truth))) or 1.0
            err_kernel = float(jnp.max(jnp.abs(out_f32 - truth))) / scale
            err_ref = float(jnp.max(jnp.abs(ref_f32 - truth))) / scale
            self.assertLess(
                err_kernel, max(err_ref * _BF16_TRUTH_SLACK, 1e-3),
                msg=f'bf16 mb={num_micro_batches} seed={seed}: kernel error '
                    f'{err_kernel:.4g} vs f32 truth exceeds reference '
                    f'{err_ref:.4g} by more than {_BF16_TRUTH_SLACK}x')

            # 3. Determinism. The kernel raced for months precisely here:
            #    identical input gave different answers run to run. A
            #    correctness test that only ever calls the kernel once cannot
            #    see that, which is how it stayed hidden.
            again = hrs.hierarchical_reduce_scatter(
                x_sharded,
                mesh=mesh,
                in_specs=in_specs,
                num_micro_batches=num_micro_batches,
                fp8_comm=False,
            )
            self.assertTrue(
                bool(jnp.array_equal(jnp.asarray(again, jnp.float32), out_f32)),
                msg=f'bf16 mb={num_micro_batches} seed={seed}: two runs on '
                    f'identical input disagree -- race, not precision')

    # ------------------------------------------------------------------ #
    # Race regression gate                                                #
    # ------------------------------------------------------------------ #
    @parameterized.product(
        num_micro_batches=[2, 4],
        fp8_comm=[False, True],
    )
    def test_no_race_multi_micro_batch(self, num_micro_batches, fp8_comm):
        """Repeat one shape many times and require bit-identical output.

        Shape matters more than repetition count here. Two races were found in
        this kernel, and NEITHER reproduced at the 128 local rows that
        test_correctness_bf16 uses -- that test passes even with the bug
        deliberately restored. 512 local rows with num_micro_batches=4 was wrong
        in 157 of 200 runs, so this is the configuration a gate has to exercise.

        Both failures were write-after-read hazards on a shared buffer, which
        makes them timing-dependent: a single call can easily come back correct.
        Comparing repeated calls is what exposes them.
        """
        self._requires_devices(8)
        axis_name = 'x'
        num_devices = jax.device_count()
        mesh = _make_mesh(num_devices, axis_name)

        seq_len, hidden = 4096, 4096          # 512 local rows on 8 devices
        in_specs = P(axis_name, None)
        runs = 12

        x = jax.random.normal(jax.random.key(0), (seq_len, hidden),
                              dtype=jnp.bfloat16)
        x_sharded = jax.device_put(
            x, jax.sharding.NamedSharding(mesh, in_specs))
        truth = jnp.asarray(x, jnp.float32).reshape(
            num_devices, seq_len // num_devices, hidden).sum(axis=0)
        scale = float(jnp.max(jnp.abs(truth))) or 1.0
        kw = dict(fp8_min_rows=0, fp8_static_scale=1.0) if fp8_comm else {}
        tol = 0.15 if fp8_comm else 0.05

        first, worst = None, 0.0
        for i in range(runs):
            out = jnp.asarray(hrs.hierarchical_reduce_scatter(
                x_sharded, mesh=mesh, in_specs=in_specs,
                num_micro_batches=num_micro_batches, fp8_comm=fp8_comm, **kw),
                jnp.float32)
            worst = max(worst, float(jnp.max(jnp.abs(out - truth))) / scale)
            if first is None:
                first = out
            else:
                self.assertTrue(
                    bool(jnp.array_equal(first, out)),
                    msg=f'mb={num_micro_batches} fp8={fp8_comm}: run {i} '
                        f'differs from run 0 on identical input -- race')

        self.assertLess(
            worst, tol,
            msg=f'mb={num_micro_batches} fp8={fp8_comm}: max relative error '
                f'{worst:.4g} over {runs} runs exceeds {tol}')

    # ------------------------------------------------------------------ #
    # Quality evaluation: FP8 comm vs bf16 baseline                      #
    # ------------------------------------------------------------------ #
    @parameterized.product(
        seq_len=[512, 1024, 2048],
        hidden=[2048, 4096, 8192],
    )
    def test_fp8_comm_quality(self, seq_len, hidden):
        """FP8 comm output should have acceptable SNR vs bf16 baseline.

        This is the primary evaluation gate before enabling real FP8 wire
        transfers in Phase 2. If this test fails, FP8 quality is too poor
        for production use at that shape.
        """
        self._requires_devices(8)
        axis_name = 'x'
        num_devices = jax.device_count()
        mesh = _make_mesh(num_devices, axis_name)
        in_specs = P(axis_name, None)

        snr_values = []
        max_errors = []
        for seed in range(5):
            x = jax.random.normal(jax.random.key(seed), (seq_len, hidden),
                                  dtype=jnp.bfloat16)
            x_sharded = jax.device_put(
                x, jax.sharding.NamedSharding(mesh, in_specs))

            out_bf16 = hrs.hierarchical_reduce_scatter(
                x_sharded,
                mesh=mesh,
                in_specs=in_specs,
                fp8_comm=False,
            )
            out_fp8 = hrs.hierarchical_reduce_scatter(
                x_sharded,
                mesh=mesh,
                in_specs=in_specs,
                fp8_comm=True,
            )

            noise = out_fp8.astype(jnp.float32) - out_bf16.astype(jnp.float32)
            snr = _snr_db(out_bf16, noise)
            max_err = float(jnp.max(jnp.abs(noise)))

            snr_values.append(snr)
            max_errors.append(max_err)

        avg_snr = sum(snr_values) / len(snr_values)
        avg_max_err = sum(max_errors) / len(max_errors)

        print(
            f'[fp8_quality seq={seq_len} hidden={hidden}] '
            f'SNR={avg_snr:.1f} dB, avg_max_err={avg_max_err:.4f}')

        self.assertGreater(
            avg_snr,
            _MIN_SNR_DB,
            msg=f'FP8 comm SNR {avg_snr:.1f} dB < threshold {_MIN_SNR_DB} dB '
            f'for seq_len={seq_len}, hidden={hidden}. FP8 quality too poor.',
        )

    # ------------------------------------------------------------------ #
    # Smoke: fp8_comm=True doesn't crash and produces finite outputs      #
    # ------------------------------------------------------------------ #
    def test_fp8_comm_no_nan(self):
        self._requires_devices(8)
        axis_name = 'x'
        num_devices = jax.device_count()
        mesh = _make_mesh(num_devices, axis_name)
        in_specs = P(axis_name, None)

        # Use a larger range to stress the FP8 scale computation.
        x = jax.random.normal(jax.random.key(42), (1024, 4096),
                               dtype=jnp.bfloat16) * 10.0
        x_sharded = jax.device_put(x,
                                    jax.sharding.NamedSharding(mesh, in_specs))

        out = hrs.hierarchical_reduce_scatter(
            x_sharded,
            mesh=mesh,
            in_specs=in_specs,
            fp8_comm=True,
        )
        self.assertTrue(jnp.all(jnp.isfinite(out)),
                        'fp8_comm produced NaN or Inf')

    # ------------------------------------------------------------------ #
    # Edge case: all-zero input must not produce NaN with FP8 comm        #
    # ------------------------------------------------------------------ #
    def test_fp8_comm_zero_input(self):
        self._requires_devices(8)
        axis_name = 'x'
        num_devices = jax.device_count()
        mesh = _make_mesh(num_devices, axis_name)
        in_specs = P(axis_name, None)

        x = jnp.zeros((1024, 4096), dtype=jnp.bfloat16)
        x_sharded = jax.device_put(x,
                                    jax.sharding.NamedSharding(mesh, in_specs))

        out = hrs.hierarchical_reduce_scatter(
            x_sharded,
            mesh=mesh,
            in_specs=in_specs,
            fp8_comm=True,
        )
        self.assertTrue(jnp.all(jnp.isfinite(out)),
                        'fp8_comm produced NaN on zero input')
        self.assertAllClose(out, jnp.zeros_like(out), atol=0.0)

    # ------------------------------------------------------------------ #
    # Static scaling: fixed wire scale vs bf16 baseline                   #
    # ------------------------------------------------------------------ #
    @parameterized.product(
        seq_len=[1024, 2048],
        hidden=[4096, 8192],
    )
    def test_fp8_static_scale_quality(self, seq_len, hidden):
        """Static-scale FP8 output should have acceptable SNR vs bf16.

        Uses fp8_min_rows=0 to force the FP8 wire at every tested size (the
        default env gate would otherwise silently fall back to bf16 below
        ~2048 rows and make this comparison vacuous).
        """
        self._requires_devices(8)
        axis_name = 'x'
        num_devices = jax.device_count()
        mesh = _make_mesh(num_devices, axis_name)
        in_specs = P(axis_name, None)

        snr_values = []
        for seed in range(5):
            x = jax.random.normal(jax.random.key(seed), (seq_len, hidden),
                                  dtype=jnp.bfloat16)
            x_sharded = jax.device_put(
                x, jax.sharding.NamedSharding(mesh, in_specs))

            out_bf16 = hrs.hierarchical_reduce_scatter(
                x_sharded, mesh=mesh, in_specs=in_specs, fp8_comm=False)

            # Data-driven static scale with 2x headroom: map the largest final
            # value to ~half of fp8's range so partial sums that momentarily
            # exceed the final (sign cancellation) still avoid saturation.
            max_abs = float(jnp.max(jnp.abs(out_bf16.astype(jnp.float32))))
            static_scale =  hrs_config.FP8_E4M3_MAX / (2.0 * max(max_abs, 1e-6))

            out_static = hrs.hierarchical_reduce_scatter(
                x_sharded, mesh=mesh, in_specs=in_specs, fp8_comm=True,
                fp8_static_scale=static_scale, fp8_min_rows=0)

            self.assertTrue(jnp.all(jnp.isfinite(out_static)),
                            'static-scale fp8 produced NaN or Inf')
            noise = out_static.astype(jnp.float32) - out_bf16.astype(jnp.float32)
            snr_values.append(_snr_db(out_bf16, noise))

        avg_snr = sum(snr_values) / len(snr_values)
        print(f'[fp8_static seq={seq_len} hidden={hidden}] '
              f'SNR={avg_snr:.1f} dB (scale from bf16 max, 2x headroom)')
        self.assertGreater(
            avg_snr, _MIN_STATIC_SNR_DB,
            msg=f'static-scale SNR {avg_snr:.1f} dB < {_MIN_STATIC_SNR_DB} dB '
            f'for seq_len={seq_len}, hidden={hidden}.')

    # ------------------------------------------------------------------ #
    # Static scaling: zero input must not divide-by-zero / NaN            #
    # ------------------------------------------------------------------ #
    def test_fp8_static_scale_zero_input(self):
        self._requires_devices(8)
        axis_name = 'x'
        num_devices = jax.device_count()
        mesh = _make_mesh(num_devices, axis_name)
        in_specs = P(axis_name, None)

        x = jnp.zeros((1024, 4096), dtype=jnp.bfloat16)
        x_sharded = jax.device_put(x,
                                    jax.sharding.NamedSharding(mesh, in_specs))

        out = hrs.hierarchical_reduce_scatter(
            x_sharded, mesh=mesh, in_specs=in_specs, fp8_comm=True,
            fp8_static_scale=0.0625, fp8_min_rows=0)
        self.assertTrue(jnp.all(jnp.isfinite(out)),
                        'static-scale fp8 produced NaN on zero input')
        self.assertAllClose(out, jnp.zeros_like(out), atol=0.0)

    # ------------------------------------------------------------------ #
    # Static vs dynamic: both paths run and land in the same ballpark     #
    # ------------------------------------------------------------------ #
    def test_fp8_static_vs_dynamic(self):
        self._requires_devices(8)
        axis_name = 'x'
        num_devices = jax.device_count()
        mesh = _make_mesh(num_devices, axis_name)
        in_specs = P(axis_name, None)

        x = jax.random.normal(jax.random.key(7), (2048, 4096),
                              dtype=jnp.bfloat16)
        x_sharded = jax.device_put(x,
                                    jax.sharding.NamedSharding(mesh, in_specs))

        out_bf16 = hrs.hierarchical_reduce_scatter(
            x_sharded, mesh=mesh, in_specs=in_specs, fp8_comm=False)
        max_abs = float(jnp.max(jnp.abs(out_bf16.astype(jnp.float32))))
        static_scale =  hrs_config.FP8_E4M3_MAX / (2.0 * max(max_abs, 1e-6))

        out_dyn = hrs.hierarchical_reduce_scatter(
            x_sharded, mesh=mesh, in_specs=in_specs, fp8_comm=True,
            fp8_min_rows=0)
        out_static = hrs.hierarchical_reduce_scatter(
            x_sharded, mesh=mesh, in_specs=in_specs, fp8_comm=True,
            fp8_static_scale=static_scale, fp8_min_rows=0)

        self.assertTrue(jnp.all(jnp.isfinite(out_dyn)))
        self.assertTrue(jnp.all(jnp.isfinite(out_static)))
        # A well-chosen static scale should track dynamic closely.
        diff = out_static.astype(jnp.float32) - out_dyn.astype(jnp.float32)
        snr = _snr_db(out_dyn, diff)
        print(f'[fp8_static_vs_dynamic] static-vs-dynamic SNR={snr:.1f} dB')
        self.assertGreater(snr, 10.0,
                           msg=f'static diverges from dynamic (SNR {snr:.1f} dB)')


@jtu.with_config(jax_numpy_dtype_promotion='standard')
class HierarchicalReduceScatterPlanningTest(jtu.JaxTestCase):
  """Planning rules that decide HOW the kernel runs, tested without a TPU.

  These cover the two decisions taken before any device work happens: how many
  micro-batches to use, and whether the working set can live in VMEM scratch.
  Both are pure arithmetic, so they run anywhere and in milliseconds -- which
  matters because the on-device suite only ever exercises whatever the
  heuristics happen to pick, and would stay green if these rules regressed.
  """

  # local_seq_len values a decode-heavy server actually reaches, per RS_PIPE_PROBE
  # at MNBT 1024 / MoE chunk 256 on 8 devices. This is the PER-DEVICE, PRE-scatter
  # row count -- 8x the post-scatter count that appears in the HLO.
  PRODUCTION_ROWS = (32, 64, 128, 256, 512, 1024, 2048)
  HIDDEN = 4096
  BF16_ITEMSIZE = 2

  def test_micro_batch_floor_is_bf16_only(self):
    """mb=1 returns the PREVIOUS call's result on the BF16 wire.

    [Step C] reads the accumulator on the line after the emit_pipeline that
    writes it, with nothing ordering the two. Measured with a fresh input per
    run against a psum+dynamic_slice reference: 26/30, 24/30 and 11/30 runs
    below 40 dB SNR at 128, 256 and 512 local rows. mb>=2 is 0/30 everywhere.

    The FP8 wire is exempt because quantize_chunks_to_fp8_staging already
    separates the write from the wire read (0/200 at the production shape).
    If that staging step is ever removed, this exemption must be re-validated
    with fresh inputs BEFORE this test is relaxed.
    """
    self.assertEqual(hrs_config._MIN_SAFE_MICRO_BATCHES, 2,
                     msg='the floor is disabled -- is RS_ALLOW_UNSAFE_MB1 set?')
    for rows in self.PRODUCTION_ROWS:
      bf16_mb = hrs_config.pick_num_micro_batches(rows, self.HIDDEN,
                                                  self.BF16_ITEMSIZE, False)
      self.assertGreaterEqual(
          bf16_mb, 2,
          msg=f'bf16 at local_seq_len={rows} picked mb={bf16_mb}; mb=1 is '
              'measurably incorrect, see the docstring above')
      self.assertLessEqual(bf16_mb, hrs_config._MAX_MICRO_BATCHES)

    # The exemption is real, not vacuous: at small shapes the fitted rule wants
    # mb=1 and the FP8 wire is allowed to take it while BF16 is not.
    fp8_mb = hrs_config.pick_num_micro_batches(128, self.HIDDEN,
                                               self.BF16_ITEMSIZE, True)
    self.assertEqual(fp8_mb, 1,
                     msg='FP8 no longer takes mb=1, so the BF16-only floor is '
                         'not being exercised by this test')

  def test_work_set_is_six_bytes_per_element_on_both_wires(self):
    """FP8 halves what crosses the wire but does NOT shrink the working set.

    running_sum and recv_buf are BF16 on both wires (2 B/elem each); the FP8
    wire swaps the BF16 phase-2 landing buffer for two 1-byte staging buffers.
    Both land on 6 B/elem. A comment that assumed otherwise under-counted these
    buffers by 8x and claimed they fit in VMEM at every shape.
    """
    for rows in self.PRODUCTION_ROWS:
      elems = rows * self.HIDDEN
      bf16 = hrs._work_set_bytes(rows, self.HIDDEN, self.BF16_ITEMSIZE,
                                False, 8, 8)
      self.assertEqual(bf16, 6 * elems)

      # num_scale_slots only adds the two f32 scale buffers, which are tiny.
      fp8 = hrs._work_set_bytes(rows, self.HIDDEN, self.BF16_ITEMSIZE,
                               True, 8, 8)
      scale_bytes = 2 * 8 * 8 * hrs_config.SCALE_LANE * 4
      self.assertEqual(fp8, 6 * elems + scale_bytes)

  def test_work_scratch_falls_back_when_it_cannot_fit(self):
    """The VMEM claim is sized from the shape, and refuses shapes that do not fit.

    At 2048 local rows the working set (48.1 MiB) plus scoped scratch plus the
    operand is 74.5 MiB against 58.9 MiB usable -- no setting makes that fit, so
    the kernel must fall back to the pl.ANY/HBM form rather than fail to
    compile. At 1024 the same total is 42.5 MiB and must be accepted.
    """
    fake_info = mock.Mock(vmem_capacity_bytes=64 * 2**20)
    with mock.patch.object(hrs.pltpu, 'get_tpu_info', return_value=fake_info):
      for rows, mb, expected in ((256, 1, True), (512, 1, True),
                                 (1024, 1, True), (2048, 2, False)):
        enabled, frac = hrs._plan_work_scratch(rows, self.HIDDEN,
                                               self.BF16_ITEMSIZE, True, 8,
                                               8 * mb, mb, 0.95)
        self.assertEqual(
            enabled, expected,
            msg=f'local_seq_len={rows} mb={mb}: VMEM scratch enabled={enabled}, '
                f'expected {expected}')
        if enabled:
          # The claim must cover the need exactly, not a fixed fraction: an
          # under-claim raises CompileTimeScopedVmemOom at compile time.
          self.assertGreater(frac, 0.0)
          self.assertLess(frac, 1.0)


if __name__ == '__main__':
    absltest.main()
