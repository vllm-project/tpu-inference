import jax
import torch
import torchax

from tpu_inference.lora.torch_lora_ops import bgmv_torch


def test_bgmv_torch():
    num_tokens = 16
    hidden_size = 128
    max_loras = 9
    max_lora_rank = 8

    with torchax.default_env(), jax.default_device(jax.devices("tpu")[0]):
        inputs = torch.rand(num_tokens, hidden_size, device='jax')
        loras = torch.rand(max_loras,
                           1,
                           max_lora_rank,
                           hidden_size,
                           device='jax')
        idxs = torch.randint(0, max_loras, (num_tokens, ), device='jax')

        actual = bgmv_torch(inputs, loras, idxs)
        expected = _ref_bgmv_torch(inputs, loras, idxs)
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=1e-3)


def _ref_bgmv_torch(inputs, loras, idxs):
    if len(loras.shape) == 4:
        loras = loras.squeeze(axis=1)

    # Another equivalent ref impl is as the 2 lines below.
    # selected_loras = loras[idxs]
    # return torch.einsum('td,tld->tl', inputs, selected_loras)
    num_tokens, _ = inputs.shape
    outputs = []
    for i in range(num_tokens):
        input = inputs[i]  # [hidden_size]
        lora = loras[idxs[i]]  # [max_lora_rank, hidden_size]
        out = torch.matmul(lora, input)
        outputs.append(out)

    return torch.stack(outputs, axis=0)
