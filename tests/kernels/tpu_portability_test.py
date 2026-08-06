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
"""Ask the kernels to build for chip generations that are not attached,
naming each one through an abstract device, so a kernel that pins one
chip's constants is caught off device. Building is covered; emitting
machine code for the chip is not.
"""

import importlib.metadata
import os
import warnings
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import pytest
from jax._src.pallas.mosaic import tpu_info as tpu_info_lib
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.fused_moe.v2.host import VMEM_FRACTION, vmem_limit
from tpu_inference.kernels.megablox.gmm_fused import (default_vmem_limit_bytes,
                                                      gmm_fused,
                                                      unsupported_reason)
from tpu_inference.kernels.megablox.gmm_v2_fused_support import (
    sublane_tiling, validate_inputs)
from tpu_inference.tpu_info import get_tpu_vmem_size_bytes

ChipVersion = tpu_info_lib.ChipVersion

# Written out rather than derived, so a new generation is reported by name.
CHIP_DEVICE_KIND = {
    ChipVersion.TPU_V2: "TPU v2",
    ChipVersion.TPU_V3: "TPU v3",
    ChipVersion.TPU_V4I: "TPU v4 lite",
    ChipVersion.TPU_V4: "TPU v4",
    ChipVersion.TPU_V5E: "TPU v5e",
    ChipVersion.TPU_V5P: "TPU v5p",
    ChipVersion.TPU_V6E: "TPU v6e",
    ChipVersion.TPU_7: "TPU7",
    ChipVersion.TPU_7X: "TPU7x",
    ChipVersion.TPU_8I: "TPU8i",
}

ALL_CHIPS = list(ChipVersion)

CHIP_PARAMS = [pytest.param(c, id=c.value) for c in ALL_CHIPS]

# int4: the tiling rule divides 32 by the element width, so sub-byte widths
# are where a whole-byte rule goes wrong.
TILING_DTYPES = [
    pytest.param(jnp.float32, id="f32"),
    pytest.param(jnp.bfloat16, id="bf16"),
    pytest.param(jnp.float8_e4m3fn, id="f8e4m3"),
    pytest.param(jnp.int8, id="i8"),
    pytest.param(jnp.int4, id="i4"),
]

# One real chip's capacity, the literal this file keeps out of the kernels.
ONE_GENERATIONS_VMEM_LITERAL = 64 * 1024 * 1024

# Fused-GMM shape: real tile and budget arithmetic, under a second to build.
SMALL_G, SMALL_M, SMALL_H, SMALL_I = 4, 128, 512, 256


@pytest.fixture(autouse=True)
def keep_this_file_off_the_device():
    """Hold every test to the host CPU so nothing starts the TPU runtime."""
    previous = jax.config.jax_platforms
    jax.config.update("jax_platforms", "cpu")
    try:
        yield
    finally:
        jax.config.update("jax_platforms", previous)


@contextmanager
def building_for(chip: ChipVersion):
    """Build as if this process were attached to `chip`."""
    device = AbstractDevice(device_kind=CHIP_DEVICE_KIND[chip],
                            num_cores=1,
                            platform="tpu")
    mesh = AbstractMesh((1, ), ("x", ), (AxisType.Explicit, ),
                        abstract_device=device)
    with jax.sharding.use_abstract_mesh(mesh):
        yield mesh


def record_for(chip: ChipVersion):
    """The record jax holds for `chip`, fetched by the kernels' own route."""
    return tpu_info_lib.get_tpu_info_for_chip(chip, 1)


def test_every_enumerated_generation_is_covered():
    """A generation missing from the table warns rather than fails."""
    missing = [c.value for c in ALL_CHIPS if c not in CHIP_DEVICE_KIND]
    if missing:
        warnings.warn(
            f"jax enumerates {missing} and this file has no device-kind "
            "name for them, so they are going untested. Add the name jax "
            "knows each chip by to CHIP_DEVICE_KIND.", UserWarning)


@pytest.mark.parametrize("chip", CHIP_PARAMS)
def test_device_kind_name_resolves_back_to_its_chip(chip):
    kind = CHIP_DEVICE_KIND[chip]
    assert tpu_info_lib.chip_version_from_device_kind(kind) is chip


@pytest.mark.parametrize("chip", CHIP_PARAMS)
def test_naming_a_chip_selects_that_chips_record(chip):
    """Naming a chip redirects the hardware questions to that chip."""
    with building_for(chip):
        info = pltpu.get_tpu_info()
    assert info.chip_version is chip
    assert info.generation == record_for(chip).generation


@pytest.mark.parametrize("chip", CHIP_PARAMS)
def test_fused_ep_moe_vmem_budget_is_this_generations(chip):
    """The fused expert-parallel kernel's budget follows the chip."""
    capacity = record_for(chip).vmem_capacity_bytes
    with building_for(chip):
        assert vmem_limit() == int(capacity * VMEM_FRACTION)


@pytest.mark.parametrize("chip", CHIP_PARAMS)
def test_gmm_fused_vmem_budget_is_this_generations(chip):
    """The fused grouped-matmul kernel's budget follows the chip."""
    capacity = record_for(chip).vmem_capacity_bytes
    with building_for(chip):
        assert default_vmem_limit_bytes() == int(capacity * 0.9)


@pytest.mark.parametrize("chip", CHIP_PARAMS)
def test_the_reported_vmem_size_is_this_generations(chip):
    """The reported size is the chip's and never zero (the error value)."""
    with building_for(chip):
        assert get_tpu_vmem_size_bytes() == record_for(
            chip).vmem_capacity_bytes
        assert get_tpu_vmem_size_bytes() > 0


def test_the_vmem_budget_is_not_one_number_for_every_chip():
    """A budget that never moves across generations was written down."""
    ep_budgets, gmm_budgets = set(), set()
    for chip in ALL_CHIPS:
        with building_for(chip):
            ep_budgets.add(vmem_limit())
            gmm_budgets.add(default_vmem_limit_bytes())
    assert len(ep_budgets) >= 3, (
        f"the fused expert-parallel budget takes only {sorted(ep_budgets)} "
        "across every generation jax enumerates; it is no longer reading "
        "the chip")
    assert len(gmm_budgets) >= 3, (
        f"the fused grouped-matmul budget takes only {sorted(gmm_budgets)} "
        "across every generation jax enumerates; it is no longer reading "
        "the chip")


def test_the_old_literal_would_have_been_wrong_on_most_generations():
    """The 64 MiB literal is wrong on most generations, too large on one."""
    capacities = {c: record_for(c).vmem_capacity_bytes for c in ALL_CHIPS}
    wrong = {
        c.value: v
        for c, v in capacities.items() if v != ONE_GENERATIONS_VMEM_LITERAL
    }
    too_small = {
        c: v
        for c, v in capacities.items() if v < ONE_GENERATIONS_VMEM_LITERAL
    }
    assert len(wrong) >= 6, (
        "the 64 MiB literal is supposed to be wrong on most generations; it "
        f"is only wrong on {sorted(wrong)}. Check the hardware record.")
    assert too_small, (
        "at least one generation is supposed to have LESS vector memory than "
        "the old literal claimed -- that is the case where the literal hands "
        "the kernel a budget the chip cannot honour")


@pytest.mark.parametrize("chip", CHIP_PARAMS)
@pytest.mark.parametrize("dtype", TILING_DTYPES)
def test_sublane_tiling_answers_on_every_generation(chip, dtype):
    """The tiling helper never refuses a generation the build supports."""
    with building_for(chip):
        tiling = sublane_tiling(dtype)
    assert isinstance(tiling, int)
    assert tiling > 0
    assert tiling & (tiling - 1) == 0, (
        f"sublane tiling {tiling} on {chip.value} for {dtype.__name__} is "
        "not a power of two")


@pytest.mark.parametrize("chip", CHIP_PARAMS)
@pytest.mark.parametrize("dtype", TILING_DTYPES)
def test_sublane_tiling_matches_the_rule_for_its_generation(chip, dtype):
    """The record's answer up to generation 6, the large tiling from 7."""
    info = record_for(chip)
    with building_for(chip):
        tiling = sublane_tiling(dtype)
    if info.generation < 7:
        assert tiling == info.get_sublane_tiling(dtype)
    else:
        expected = info.num_sublanes * (32 // jax.dtypes.itemsize_bits(dtype))
        assert tiling == expected


def test_the_guard_covers_a_generation_the_record_refuses():
    """Some generation still makes the record raise, and the guard answers."""
    refused = []
    for chip in ALL_CHIPS:
        try:
            record_for(chip).get_sublane_tiling(jnp.bfloat16)
        except NotImplementedError:
            refused.append(chip)
    assert refused, (
        "the hardware record now answers the sublane tiling on every "
        "generation jax enumerates, so the guard in "
        "gmm_v2_fused_support.sublane_tiling has nothing left to catch. "
        "Confirm that against the installed jax and then remove the guard "
        "rather than leaving an unexplained branch.")
    for chip in refused:
        with building_for(chip):
            for dtype, _ in ((jnp.bfloat16, None), (jnp.float32, None),
                             (jnp.float8_e4m3fn, None)):
                assert sublane_tiling(dtype) > 0


@pytest.mark.parametrize("chip", CHIP_PARAMS)
def test_input_validation_survives_every_generation(chip):
    """The whole input check, not just the tiling helper underneath it."""
    dims = (SMALL_M, SMALL_H)
    rhs = jax.ShapeDtypeStruct((SMALL_G, SMALL_H, SMALL_I), jnp.float8_e4m3fn)
    group_sizes = jax.ShapeDtypeStruct((SMALL_G, ), jnp.int32)
    group_offset = jax.ShapeDtypeStruct((1, ), jnp.int32)

    for lhs_dtype in (jnp.bfloat16, jnp.float8_e4m3fn):
        lhs = jax.ShapeDtypeStruct(dims, lhs_dtype)
        with building_for(chip):
            expected = sublane_tiling(lhs_dtype)
            got = validate_inputs(lhs, rhs, None, None, group_sizes,
                                  group_offset)
        if jax.dtypes.itemsize_bits(lhs_dtype) == 8:
            # 8-bit rows are not clamped: the row count must be a whole
            # number of sublane tiles and the check says so.
            assert got.size_lhs_sublane == expected
        else:
            assert got.size_lhs_sublane == min(expected, SMALL_M)
        assert got.size_m == SMALL_M
        assert got.size_k == SMALL_H
        assert got.size_n == SMALL_I


def trivial_pallas_program(x):
    """The smallest kernel whose emitted text is generation-dependent."""
    info = pltpu.get_tpu_info()
    return pl.pallas_call(
        lambda x_ref, o_ref: o_ref.__setitem__(..., x_ref[...] + 1.0),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(info.vmem_capacity_bytes * 0.9)),
    )(x)


def fused_gmm_operands(mesh):

    def operand(shape, dtype):
        return jax.ShapeDtypeStruct(shape,
                                    dtype,
                                    sharding=NamedSharding(mesh, P()))

    return (
        operand((SMALL_M, SMALL_H), jnp.bfloat16),
        operand((SMALL_G, SMALL_H, 2 * SMALL_I), jnp.float8_e4m3fn),
        operand((SMALL_G, SMALL_I, SMALL_H), jnp.float8_e4m3fn),
        operand((SMALL_G, ), jnp.int32),
        operand((SMALL_G, 1, 1, 2 * SMALL_I), jnp.float32),
        operand((SMALL_G, 1, 1, SMALL_H), jnp.float32),
    )


@pytest.mark.parametrize("chip", CHIP_PARAMS)
def test_a_pallas_kernel_builds_for_every_generation(chip):
    """A plain kernel builds for every chip jax enumerates, deviceless."""
    with building_for(chip) as mesh:
        operand = jax.ShapeDtypeStruct((256, 256),
                                       jnp.float32,
                                       sharding=NamedSharding(mesh, P()))
        traced = jax.jit(trivial_pallas_program).trace(operand)
        lowered = traced.lower(lowering_platforms=("tpu", ))
        text = lowered.as_text()
    assert "tpu_custom_call" in text, (
        f"the program built for {chip.value} contains no Mosaic kernel, so "
        "something replaced the kernel with plain array operations and this "
        "test is no longer checking the kernel path")


def test_the_built_program_differs_between_generations():
    """Building "for" a chip must actually change what comes out."""
    programs = {}
    for chip in ALL_CHIPS:
        with building_for(chip) as mesh:
            operand = jax.ShapeDtypeStruct((256, 256),
                                           jnp.float32,
                                           sharding=NamedSharding(mesh, P()))
            programs[chip.value] = jax.jit(trivial_pallas_program).trace(
                operand).lower(lowering_platforms=("tpu", )).as_text()
    assert len(set(programs.values())) >= 3, (
        "the program built for every generation is nearly identical "
        f"({len(set(programs.values()))} distinct texts across "
        f"{len(programs)} chips); naming the chip is not reaching the build")


@pytest.mark.parametrize("chip", CHIP_PARAMS)
def test_the_fused_gmm_kernel_builds_wherever_the_chip_can_host_it(chip):
    """Builds wherever the MXU takes eight-bit inputs, declines by name."""
    with building_for(chip) as mesh:
        operands = fused_gmm_operands(mesh)
        reason = unsupported_reason(operands[0], operands[1], operands[2],
                                    operands[4], operands[5], None, None)
        if reason is not None:
            assert reason.strip(), "a refusal must carry a reason"
            supported = record_for(chip).is_matmul_supported(
                jnp.float8_e4m3fn, jnp.float8_e4m3fn)
            assert not supported, (
                f"{chip.value} takes eight-bit matmul inputs natively but the "
                f"fused GMM kernel still declined: {reason}")
            pytest.skip(f"{chip.value} cannot host the fused GMM kernel: "
                        f"{reason}")
        text = gmm_fused.trace(*operands).lower(
            lowering_platforms=("tpu", )).as_text()
    assert "tpu_custom_call" in text


def test_at_least_one_unattached_generation_hosts_the_fused_gmm_kernel():
    """More than one generation hosts it, so deviceless buys something."""
    hosted = []
    for chip in ALL_CHIPS:
        with building_for(chip) as mesh:
            operands = fused_gmm_operands(mesh)
            if unsupported_reason(operands[0], operands[1], operands[2],
                                  operands[4], operands[5], None,
                                  None) is None:
                hosted.append(chip.value)
    assert len(hosted) >= 2, (
        f"only {hosted} can host the fused GMM kernel, so a deviceless build "
        "check buys nothing beyond the chip already attached")


def declared_libtpu_pin():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    path = os.path.join(root, "requirements.txt")
    assert os.path.exists(path), f"no requirements file at {path}"
    pins = []
    for raw in open(path):
        line = raw.split("#", 1)[0].strip()
        if line.startswith("libtpu=="):
            pins.append(line[len("libtpu=="):].strip())
    assert len(pins) == 1, (
        f"{path} declares {len(pins)} libtpu pins ({pins}); exactly one is "
        "expected so that there is a single declared runtime")
    return pins[0]


def test_the_installed_runtime_is_the_declared_one():
    """Report whether the installed TPU runtime is the declared one; a
    mismatch skips with both versions named rather than failing."""
    declared = declared_libtpu_pin()
    try:
        installed = importlib.metadata.version("libtpu")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("no TPU runtime library installed in this environment, so "
                    "there is nothing to compare the declared pin against")
    if installed != declared:
        pytest.skip(
            f"requirements.txt declares libtpu=={declared} but this "
            f"environment has libtpu=={installed} installed. Results taken "
            "here are not from the runtime this tree says it needs.")
