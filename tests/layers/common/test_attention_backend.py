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
"""The TPU attention backend interface, exercised with a fake backend."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from vllm.config.attention import AttentionConfig
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import \
    QwenGatedDeltaNetAttention
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.attention.backends.registry import AttentionBackendEnum

import tpu_inference.kernels.ragged_paged_attention.v3.kernel as rpa
import tpu_inference.kernels.ragged_paged_attention.v3.kernel_hd64 as rpa_hd64
from tpu_inference.kernels.gdn.v3 import wrapper as v3_gdn
from tpu_inference.layers.common import (attention_backend,
                                         attention_interface, gdn_attention)
from tpu_inference.layers.common.attention_backend import (
    GDNLinearAttentionBackend, RPAAttentionBackend)
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.layers.vllm.backends.flash_attn import (
    PallasAttentionBackend, PallasAttentionBackendImpl)
from tpu_inference.layers.vllm.custom_ops.gdn_attention_op import \
    VllmGatedDeltaNetAttention
from tpu_inference.platforms.tpu_platform import TpuPlatform
from tpu_inference.runner import kv_cache as kv_cache_mod

FAKE = "fake"
FAKE_VLLM_BACKEND = f"{__name__}.FakePallasAttentionBackend"
FAKE_OUTPUT = 7
NUM_TOKENS, NUM_HEADS, HEAD_DIM, KERNEL_SIZE = 8, 2, 128, 4
QKV_DIM = 3 * NUM_HEADS * HEAD_DIM
NUM_STATE_SLOTS, WRITE_SLOT, READ_SLOT = 4, 2, 1
V3_MARKER, FAKE_MARKER = 1, 2


def stub_gdn_kernel(marker: int):
    """A GDN kernel that copies the read slot plus marker into the write slot,
    and outputs marker."""

    def run(qkv, b, a, conv_state, recurrent_state, conv_weight, conv_bias,
            a_log, dt_bias, query_start_loc, state_indices, distribution,
            seq_lens, read_state_indices, **options):
        new_conv_state = conv_state.at[state_indices].set(
            conv_state[read_state_indices] + marker)
        new_recurrent_state = recurrent_state.at[state_indices].set(
            recurrent_state[read_state_indices] + marker)
        output = jnp.full((qkv.shape[0], options["n_v"] * options["d_v"]),
                          marker, qkv.dtype)
        return (new_conv_state, new_recurrent_state), output

    return run


class FakeAttentionBackend(attention_backend.AttentionBackend):

    def get_kv_cache_shape(self, total_num_pages, page_size, num_kv_heads,
                           head_dim, kv_dtype):
        return (total_num_pages, page_size, 2 * num_kv_heads, head_dim)

    def attention(self, kv_cache, q, k, v, md, shared_md, mesh, **kwargs):
        return kv_cache, jnp.full(q.shape, FAKE_OUTPUT, q.dtype)


class FakeLinearAttentionBackend(attention_backend.LinearAttentionBackend):

    def conv1d_gdn_kernel(self):
        return stub_gdn_kernel(FAKE_MARKER)


class FakePallasAttentionBackendImpl(PallasAttentionBackendImpl):
    attention_backend = FAKE


class FakePallasAttentionBackend(PallasAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return FAKE.upper()

    @staticmethod
    def get_impl_cls() -> type[FakePallasAttentionBackendImpl]:
        return FakePallasAttentionBackendImpl


def register_fake() -> None:
    attention_backend.register_attention_backend(
        FAKE,
        attention=FakeAttentionBackend,
        linear_attention=FakeLinearAttentionBackend,
        vllm_backend=FAKE_VLLM_BACKEND)


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention_backend, "_REGISTERED_BACKENDS", {})
    register_fake()


@pytest.fixture
def mesh() -> Mesh:
    """One CPU device over every axis the shard_map specs name."""
    devices = np.array([jax.devices("cpu")[0]])
    return Mesh(devices.reshape((1, ) * len(MESH_AXIS_NAMES)), MESH_AXIS_NAMES)


def make_vllm_config(backend: str | None):
    return SimpleNamespace(attention_config=AttentionConfig(backend=backend))


def make_gdn_inputs(mesh: Mesh) -> dict[str, object]:
    """One request of NUM_TOKENS tokens on a cached prefix, resuming from
    READ_SLOT and checkpointing into WRITE_SLOT."""
    slot_ids = jnp.arange(NUM_STATE_SLOTS, dtype=jnp.float32)
    conv_state = jnp.broadcast_to(
        slot_ids[:, None, None],
        (NUM_STATE_SLOTS, KERNEL_SIZE - 1, QKV_DIM)).astype(jnp.bfloat16)
    recurrent_state = jnp.broadcast_to(
        slot_ids[:, None, None,
                 None], (NUM_STATE_SLOTS, NUM_HEADS, HEAD_DIM, HEAD_DIM))
    return {
        "j_mixed_qkv": jnp.ones((NUM_TOKENS, QKV_DIM), jnp.bfloat16),
        "j_b": jnp.ones((NUM_TOKENS, NUM_HEADS), jnp.bfloat16),
        "j_a": jnp.ones((NUM_TOKENS, NUM_HEADS), jnp.bfloat16),
        "conv_state": conv_state,
        "recurrent_state": recurrent_state,
        "j_conv_weight": jnp.ones((QKV_DIM, 1, KERNEL_SIZE), jnp.float32),
        "j_conv_bias": None,
        "j_A_log": jnp.ones((NUM_HEADS, ), jnp.float32),
        "j_dt_bias": jnp.ones((NUM_HEADS, ), jnp.float32),
        "state_indices": jnp.array([WRITE_SLOT], jnp.int32),
        "query_start_loc": jnp.array([0, NUM_TOKENS], jnp.int32),
        "distribution": jnp.array([0, 0, 1], jnp.int32),
        "seq_lens": jnp.array([2 * NUM_TOKENS], jnp.int32),
        "n_kq": NUM_HEADS,
        "n_v": NUM_HEADS,
        "d_k": HEAD_DIM,
        "d_v": HEAD_DIM,
        "kernel_size": KERNEL_SIZE,
        "mesh": mesh,
        "read_state_indices": jnp.array([READ_SLOT], jnp.int32),
    }


def test_attention_runs_the_given_backend(mesh):
    q = jnp.ones((NUM_TOKENS, NUM_HEADS, HEAD_DIM), jnp.bfloat16)
    k = v = jnp.ones((NUM_TOKENS, 1, HEAD_DIM), jnp.bfloat16)
    kv_cache = jnp.zeros((2, 16, 2, HEAD_DIM), jnp.bfloat16)

    new_kv_cache, output = attention_interface.attention(
        kv_cache,
        q,
        k,
        v,
        AttentionMetadata(input_positions=jnp.arange(NUM_TOKENS)),
        mesh,
        attention_backend=FakeAttentionBackend())

    assert new_kv_cache is kv_cache
    np.testing.assert_array_equal(output.astype(jnp.float32), FAKE_OUTPUT)


def test_kv_cache_shape_comes_from_the_given_backend(mesh):
    args = (mesh, 8, 16, 4, HEAD_DIM, jnp.bfloat16)

    shape = kv_cache_mod.get_kv_cache_shape_with_mesh(
        *args, attention_backend=FakeAttentionBackend())

    assert tuple(shape) == (8, 16, 8, HEAD_DIM)
    assert tuple(
        kv_cache_mod.get_kv_cache_shape_with_mesh(*args)) != tuple(shape)


@pytest.mark.parametrize(("head_dim", "kernel"), [(128, rpa), (64, rpa_hd64)])
def test_rpa_kv_cache_shape(head_dim, kernel):
    args = (8, 16, 4, head_dim, jnp.bfloat16)

    assert tuple(RPAAttentionBackend().get_kv_cache_shape(*args)) == tuple(
        kernel.get_kv_cache_shape(*args))


@pytest.mark.parametrize(("backend_cls", "marker"), [
    (None, V3_MARKER),
    (GDNLinearAttentionBackend, V3_MARKER),
    (FakeLinearAttentionBackend, FAKE_MARKER),
])
def test_gdn_runs_the_given_kernel(mesh, monkeypatch, backend_cls, marker):
    monkeypatch.setattr(v3_gdn, "fused_conv1d_gdn", stub_gdn_kernel(V3_MARKER))
    kernel = backend_cls().conv1d_gdn_kernel() if backend_cls else None

    states, output = gdn_attention.run_jax_gdn_attention(
        **make_gdn_inputs(mesh), conv1d_gdn_kernel=kernel)

    np.testing.assert_array_equal(output.astype(jnp.float32), marker)
    for state in states:
        np.testing.assert_array_equal(
            np.asarray(state[WRITE_SLOT], np.float32), READ_SLOT + marker)


@pytest.mark.parametrize(("backend", "expected"), [
    (FAKE, FakeLinearAttentionBackend),
    (None, GDNLinearAttentionBackend),
])
def test_gdn_layer_resolves_its_backend_at_init(monkeypatch, backend,
                                                expected):
    monkeypatch.setattr(QwenGatedDeltaNetAttention, "__init__",
                        lambda self, config, vllm_config, *args: None)
    layer = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)

    VllmGatedDeltaNetAttention.__init__(layer, None, make_vllm_config(backend))

    assert type(layer.linear_attention_backend) is expected


def test_register_lets_vllm_parse_the_name():
    # vLLM runs the plugins in every process, and more than once in one.
    register_fake()

    assert (AttentionConfig.validate_backend_before(FAKE)
            is AttentionBackendEnum.FAKE)
    assert AttentionBackendEnum.FAKE.get_path() == FAKE_VLLM_BACKEND


def test_register_rejects_a_vllm_backend_name():
    with pytest.raises(ValueError, match="FLASH_ATTN"):
        attention_backend.register_attention_backend(
            "flash_attn",
            attention=FakeAttentionBackend,
            vllm_backend=FAKE_VLLM_BACKEND)


@pytest.mark.parametrize(("backend", "name"), [
    (FAKE, FAKE),
    (None, "rpa"),
    ("FLASH_ATTN", "rpa"),
    ("TRITON_ATTN", "rpa"),
])
def test_attention_backend_name(backend, name):
    assert attention_backend.attention_backend_name(
        make_vllm_config(backend)) == name


def test_one_name_selects_both_backends():
    assert isinstance(attention_backend.create_attention_backend(FAKE),
                      FakeAttentionBackend)
    assert isinstance(attention_backend.create_linear_attention_backend(FAKE),
                      FakeLinearAttentionBackend)
    assert type(attention_backend.create_attention_backend(
        "rpa")) is RPAAttentionBackend
    assert type(attention_backend.create_linear_attention_backend(
        "rpa")) is GDNLinearAttentionBackend


@pytest.mark.parametrize(("backend", "expected"), [
    ("FAKE", FakePallasAttentionBackend),
    ("FLASH_ATTN", PallasAttentionBackend),
    ("TRITON_ATTN", PallasAttentionBackend),
])
def test_platform_selects_the_vllm_backend(backend, expected):
    path = TpuPlatform.get_attn_backend_cls(AttentionBackendEnum[backend],
                                            SimpleNamespace(use_mla=False))

    assert resolve_obj_by_qualname(path) is expected


def test_platform_rejects_a_registered_backend_for_mla():
    with pytest.raises(ValueError, match="does not support MLA"):
        TpuPlatform.get_attn_backend_cls(AttentionBackendEnum.FAKE,
                                         SimpleNamespace(use_mla=True))


def test_default_impl_runs_rpa():
    assert PallasAttentionBackendImpl.attention_backend == "rpa"


@pytest.mark.parametrize(("env", "value"), [
    ("TPU_MULTIHOST_BACKEND", "ray"),
    ("VLLM_WORKER_MULTIPROC_METHOD", "spawn"),
])
def test_rejects_a_config_that_crosses_a_process_boundary(
        monkeypatch, env, value):
    monkeypatch.setenv(env, value)

    with pytest.raises(ValueError, match=f"{env}={value}"):
        attention_backend.check_attention_backend_config(
            make_vllm_config(FAKE))


def test_default_backend_skips_the_process_check(monkeypatch):
    monkeypatch.setenv("TPU_MULTIHOST_BACKEND", "ray")

    attention_backend.check_attention_backend_config(make_vllm_config(None))
