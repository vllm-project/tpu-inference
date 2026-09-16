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

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

import tpu_inference.layers.common.attention_interface as attention_interface
import tpu_inference.layers.vllm.ops.scaled_dot_product_attention as sdpa_ops
from tpu_inference.kernels.flash_attention.kernel import mha_reference
from tpu_inference.layers.vllm.ops.scaled_dot_product_attention import (
    scaled_dot_product_attention, vllm_vit_sdpa)


@pytest.fixture
def dp_mesh():
    """A 2-way 'data' x 1-way 'model' mesh, mirroring `enable_dp_attention`."""
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip('Need >=2 devices to exercise the DP axis.')
    return Mesh(np.array(devices[:2]).reshape(2, 1), ('data', 'model'))


class TestAttnDpBatchAxisFix:
    """
    With `--additional_config='{"sharding": {"sharding_strategy":
    {"enable_dp_attention": true}}}'`, the JAX mesh's 'data' axis has size > 1.
    vLLM's ViT flattens every image in a request into a single sequence via
    cu_seqlens/segment_ids, so the batch dimension seen by
    `scaled_dot_product_attention` / `vllm_vit_sdpa` is always 1 -- sharding that
    size-1 axis by a DP>1 'data' axis fails shard_map's divisibility check.
    The fix pins `batch_axis=None` (replicated) instead of the default
    `batch_axis="data"` for both ops.
    """

    def test_default_batch_axis_fails_under_dp_mesh_with_batch_one(
            self, dp_mesh):
        """Documents the bug: batch=1 can't be sharded by a size-2 'data' axis."""
        q = k = v = jnp.ones((1, 2, 128, 64), dtype=jnp.float32)
        ab = jnp.zeros((1, 2, 128, 128), dtype=jnp.float32)

        with jax.set_mesh(dp_mesh):
            attn_fn = attention_interface.sharded_flash_attention(
                dp_mesh, causal=False, sm_scale=1.0, use_attention_bias=True)
            with pytest.raises(ValueError, match='divisible'):
                attn_fn(q, k, v, ab, None)

    def test_scaled_dot_product_attention_survives_dp_mesh(self, dp_mesh):
        q = k = v = jnp.ones((1, 2, 128, 64), dtype=jnp.float32)

        with jax.set_mesh(dp_mesh):
            out = scaled_dot_product_attention(q, k, v)

        assert out.shape == (1, 2, 128, 64)

    def test_vllm_vit_sdpa_survives_dp_mesh(self, dp_mesh):
        # (batch, seq_len, num_heads, head_dim), as vllm_vit_sdpa expects.
        q = k = v = jnp.ones((1, 128, 2, 64), dtype=jnp.float32)

        with jax.set_mesh(dp_mesh):
            out = vllm_vit_sdpa(q, k, v, cu_seqlens=[0, 64, 128])

        assert out.shape == (1, 128, 2, 64)


class TestVitSdpaPadSegments:
    """`vllm_vit_sdpa` must hand the kernel segment ids in which every padded
    position (kernel 128-alignment padding AND mm-encoder budget padding) sits
    in its own trailing segment."""

    def _capture_seg_ids(self, q_seq_len, cu_seqlens):
        q = k = v = jnp.ones((1, q_seq_len, 2, 64), dtype=jnp.float32)
        with mock.patch.object(sdpa_ops,
                               'sharded_flash_attention') as mock_sfa:
            mock_sfa.return_value = mock.MagicMock(
                return_value=jnp.ones((1, 2, q_seq_len,
                                       64), dtype=jnp.float32))
            vllm_vit_sdpa(q, k, v, cu_seqlens=cu_seqlens)
        attn_fn = mock_sfa.return_value
        seg_ids = attn_fn.call_args[0][3]
        return np.asarray(seg_ids.q[0]), np.asarray(seg_ids.kv[0])

    def test_budget_padding_gets_dedicated_segment(self):
        """Full batch (no empty trailing cu_seqlens entry) with padded q/k/v:
        the tail must NOT inherit the last image's segment id."""
        # 2 images covering [0, 64) and [64, 192); 192..255 is budget padding.
        q_seg, kv_seg = self._capture_seg_ids(256, [0, 64, 192])
        expected = np.array([0] * 64 + [1] * 128 + [2] * 64)
        np.testing.assert_array_equal(q_seg, expected)
        np.testing.assert_array_equal(kv_seg, expected)

    def test_kernel_alignment_padding_gets_dedicated_segment(self):
        """The 128-alignment pad rows keep their dedicated trailing segment."""
        # seq_len 100 -> q_pad 28; one image covering the whole real range.
        q_seg, kv_seg = self._capture_seg_ids(100, [0, 100])
        assert q_seg.shape == (128, )
        np.testing.assert_array_equal(q_seg[:100], np.zeros(100))
        np.testing.assert_array_equal(q_seg[100:], np.ones(28))
        np.testing.assert_array_equal(kv_seg, q_seg)


class TestVitVmemLimit:
    """The ViT flash attention must raise the scoped-vmem limit above the
    32MiB pallas default: large flattened image sequences exceed it
    (CompileTimeScopedVmemOom at bf16[1,16,25344,80])."""

    def test_vit_vmem_limit_covers_failing_shape(self):
        # 37.22M was the observed scoped allocation that OOMed at 32M.
        assert sdpa_ops._VIT_FLASH_ATTENTION_VMEM_LIMIT_BYTES >= int(
            38 * 1024 * 1024)

    @pytest.mark.parametrize('op,extra_kwargs', [
        (scaled_dot_product_attention, {}),
        (vllm_vit_sdpa, {
            'cu_seqlens': [0, 128]
        }),
    ])
    def test_ops_pass_vmem_limit_to_flash_attention(self, op, extra_kwargs):
        q = k = v = jnp.ones((1, 2, 128, 64), dtype=jnp.float32)
        if op is vllm_vit_sdpa:
            q = k = v = jnp.ones((1, 128, 2, 64), dtype=jnp.float32)

        with mock.patch.object(sdpa_ops,
                               'sharded_flash_attention') as mock_sfa:
            mock_sfa.return_value = mock.MagicMock(
                return_value=jnp.ones((1, 2, 128, 64), dtype=jnp.float32))
            op(q, k, v, **extra_kwargs)

        assert mock_sfa.call_count == 1
        assert mock_sfa.call_args.kwargs['vmem_limit_bytes'] == \
            sdpa_ops._VIT_FLASH_ATTENTION_VMEM_LIMIT_BYTES


class TestSdpaSkipsBiasWithoutMask:
    """Without a caller-supplied `attn_mask`, `scaled_dot_product_attention`
    must not materialize a dense (batch, num_heads, q_seq_len, kv_seq_len)
    float32 bias: that spends O(S^2) memory to express the O(S) 128-alignment
    padding constraint, and OOMs on long vision/video sequences. Segment ids
    carry the same information."""

    def _capture_call(self, seq_len, **kwargs):
        """Runs the op with the kernel mocked out, returning the mock."""
        padded = seq_len + (128 - (seq_len % 128)) % 128
        q = k = v = jnp.ones((1, 2, seq_len, 64), dtype=jnp.float32)
        with mock.patch.object(sdpa_ops,
                               'sharded_flash_attention') as mock_sfa:
            mock_sfa.return_value = mock.MagicMock(
                return_value=jnp.ones((1, 2, padded, 64), dtype=jnp.float32))
            scaled_dot_product_attention(q, k, v, **kwargs)
        return mock_sfa

    def test_unaligned_seq_len_uses_segment_ids_not_bias(self):
        # seq_len 100 -> q_pad = kv_pad = 28, so padding must be masked.
        mock_sfa = self._capture_call(100)

        assert mock_sfa.call_args.kwargs['use_attention_bias'] is False
        args = mock_sfa.return_value.call_args[0]
        assert len(args) == 4  # (q, k, v, segment_ids) -- no bias tensor
        seg_ids = args[3]
        q_seg = np.asarray(seg_ids.q[0])
        kv_seg = np.asarray(seg_ids.kv[0])
        assert q_seg.shape == (128, )
        # Real tokens share segment 0; the pad tail gets its own segment, so
        # real queries cannot attend to padded keys.
        np.testing.assert_array_equal(q_seg[:100], np.zeros(100))
        np.testing.assert_array_equal(q_seg[100:], np.ones(28))
        np.testing.assert_array_equal(kv_seg, q_seg)

    def test_aligned_seq_len_needs_no_segment_ids(self):
        # seq_len 128 needs no padding, so there is nothing to mask.
        mock_sfa = self._capture_call(128)

        assert mock_sfa.call_args.kwargs['use_attention_bias'] is False
        assert mock_sfa.return_value.call_args[0][3] is None

    def test_explicit_attn_mask_still_uses_bias(self):
        mock_sfa = self._capture_call(128,
                                      attn_mask=jnp.zeros((1, 2, 128, 128),
                                                          dtype=jnp.float32))

        assert mock_sfa.call_args.kwargs['use_attention_bias'] is True
        args = mock_sfa.return_value.call_args[0]
        assert len(args) == 5  # (q, k, v, attention_bias, segment_ids)
        assert args[3].shape == (1, 2, 128, 128)
        assert args[4] is None

    def test_segment_ids_match_bias_numerically(self):
        """The segment-id path must produce the same attention as the bias it
        replaces. An all-zero `attn_mask` is semantically 'no mask', so the two
        paths must agree."""

        def mock_flash_attention(q,
                                 k,
                                 v,
                                 ab=None,
                                 segment_ids=None,
                                 sm_scale=1.0,
                                 causal=False,
                                 **kwargs):
            return mha_reference(q,
                                 k,
                                 v,
                                 ab,
                                 segment_ids,
                                 causal=causal,
                                 sm_scale=sm_scale)

        key = jax.random.PRNGKey(42)
        q, k, v = jax.random.normal(key, (3, 1, 2, 100, 64), dtype=jnp.float32)
        mesh = Mesh(
            np.array(jax.devices()[:1]).reshape(1, 1), ('data', 'model'))

        # The real Pallas kernel calls pltpu.get_tpu_info(), which rejects a
        # CPU device kind, so swap in the reference implementation.
        with jax.set_mesh(mesh), mock.patch.object(
                attention_interface,
                'flash_attention',
                side_effect=mock_flash_attention):
            out_seg = scaled_dot_product_attention(q, k, v, attn_mask=None)
            out_bias = scaled_dot_product_attention(q,
                                                    k,
                                                    v,
                                                    attn_mask=jnp.zeros(
                                                        (1, 2, 100, 100),
                                                        dtype=jnp.float32))

        assert out_seg.shape == (1, 2, 100, 64)
        np.testing.assert_allclose(out_seg, out_bias, rtol=1e-5, atol=1e-5)
