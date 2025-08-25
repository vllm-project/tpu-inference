# SPDX-License-Identifier: Apache-2.0

import jax
import torch
import torch.nn.functional as F
import torchax
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_commons.models.torchax.torchax_wrapper import flash_attention

jax.config.parse_flags_with_absl()


def ref_attention(
    q: torch.Tensor,  # [batch, num_heads, q_len, head_dim]
    k: torch.Tensor,  # [batch, num_heads, kv_len, head_dim]
    v: torch.Tensor,  # [batch, num_heads, kv_len, head_dim]
    causal: bool = False,
    sm_scale: float = 1.0,
) -> torch.Tensor:
    """Reference implementation for attention."""
    # using the same padding as our wrapper function
    # NOTE: vllm uses flash attention for multimodal encoders and always pads
    # inputs to a power of 2 for warming-up and inference stages.
    q_len, kv_len = q.size(-2), k.size(-2)
    block_size = 128

    pad_q_len = (block_size - q_len % block_size) % block_size
    if pad_q_len > 0:
        q = F.pad(q, (0, 0, 0, pad_q_len))

    pad_kv_len = (block_size - kv_len % block_size) % block_size
    if pad_kv_len > 0:
        k = F.pad(k, (0, 0, 0, pad_kv_len))
        v = F.pad(v, (0, 0, 0, pad_kv_len))

    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    if causal:
        padded_q_len, padded_kv_len = q.size(-2), k.size(-2)
        # NOTE: This assumes q_len == kv_len for causal attention.
        mask = torch.triu(torch.ones(padded_q_len,
                                     padded_kv_len,
                                     device=q.device,
                                     dtype=torch.bool),
                          diagonal=1)
        scores = scores.masked_fill(mask, -float('inf'))

    attn = torch.nn.functional.softmax(scores, dim=-1).to(v.dtype)
    output = torch.matmul(attn, v)

    if pad_q_len > 0:
        output = output[:, :, :q_len, :]

    return output


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class FlashAttentionTest(jtu.JaxTestCase):

    def _test_flash_attention(
        self,
        batch: int,
        num_heads: int,
        q_len: int,
        kv_len: int,
        head_dim: int,
        causal: bool,
        dtype: torch.dtype,
    ):
        if causal and q_len != kv_len:
            self.skipTest(
                "Causal attention requires q_len == kv_len for this test.")

        with torchax.default_env():
            sm_scale = 1.0 / (head_dim**0.5)
            q = torch.randn(batch, num_heads, q_len, head_dim,
                            dtype=dtype).to('jax')
            k = torch.randn(batch, num_heads, kv_len, head_dim,
                            dtype=dtype).to('jax')
            v = torch.randn(batch, num_heads, kv_len, head_dim,
                            dtype=dtype).to('jax')

            # Run flash attention from torchax_wrapper
            output = flash_attention(q,
                                     k,
                                     v,
                                     q_len=q_len,
                                     kv_len=kv_len,
                                     causal=causal,
                                     sm_scale=sm_scale)

            # Run reference implementation
            ref_out = ref_attention(q, k, v, causal=causal, sm_scale=sm_scale)
            # Transpose to match the output format of flash_attention
            ref_out = ref_out.transpose(1, 2)

            # The output of flash_attention is [batch, q_len, num_heads, head_dim]
            self.assertEqual(output.shape, (batch, q_len, num_heads, head_dim))

            output = torchax.interop.jax_view(output)
            ref_out = torchax.interop.jax_view(ref_out)

            self.assertAllClose(output, ref_out, atol=2e-2, rtol=2e-2)

    # TODO(vladkarp)?:  although there should not be vllm inputs sizes that are not a power of 2,
    # the current code will break on close assertion if any of q_len, kv_len will actually be.
    @parameterized.product(
        batch=[1, 8],
        num_heads=[8],
        q_len=[128, 256],
        kv_len=[128, 256],
        head_dim=[64],
        causal=[False, True],
        dtype=[torch.bfloat16, torch.float32],  # torch.float32
    )
    def test_flash_attention(self, batch: int, num_heads: int, q_len: int,
                             kv_len: int, head_dim: int, causal: bool,
                             dtype: torch.dtype):
        self._test_flash_attention(batch, num_heads, q_len, kv_len, head_dim,
                                   causal, dtype)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
