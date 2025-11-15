import time
from typing import List

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from vllm import LLM

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MAX_NUM_BATCHED_TOKENS = 128
MAX_NUM_SEQS = 8
RTOL = 5e-3
ATOL = 5e-3


def last_token_pool(last_hidden_states: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
    """Reference pooling implementation from Qwen3 embedding docs."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size,
                                           device=last_hidden_states.device),
                             sequence_lengths]


def hf_embeddings(texts: List[str], model: AutoModel,
                   tokenizer: AutoTokenizer) -> np.ndarray:
    """Get reference embeddings using HF Transformers""" 
    batch_dict = tokenizer(texts,
                           padding=True,
                           truncation=True,
                           max_length=MAX_NUM_BATCHED_TOKENS,
                           return_tensors="pt")
    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state,
                                      batch_dict["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()


def vllm_embeddings(texts: List[str]) -> np.ndarray:
    """Get embeddings via vLLM """
    llm = LLM(model=MODEL_ID,
              runner="pooling",
              convert="embed",
              max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
              max_num_seqs=MAX_NUM_SEQS,
              max_model_len=MAX_NUM_BATCHED_TOKENS)
    outputs = llm.embed(texts)
    embeddings = np.asarray(
        [np.array(output.outputs.embedding, dtype=np.float32) for output in outputs])
    del llm
    # Wait for TPU runtime tear down before next test.
    time.sleep(10)
    return embeddings


def compare_embeddings(vllm_emb: np.ndarray,
                        hf_emb: np.ndarray) -> List[tuple[bool, float, float]]:
    """Compare embeddings with diagnostics."""
    results = []
    for v_emb, h_emb in zip(vllm_emb, hf_emb):
        is_close = np.allclose(v_emb, h_emb, rtol=RTOL, atol=ATOL)
        max_diff = float(np.max(np.abs(v_emb - h_emb)))
        cos_sim = float(np.dot(v_emb, h_emb) /
                        (np.linalg.norm(v_emb) * np.linalg.norm(h_emb)))
        results.append((is_close, max_diff, cos_sim))
    return results


@pytest.mark.tpu
def test_last_token_embedding_pooling(monkeypatch: pytest.MonkeyPatch):
    prompts = [
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "Machine learning systems process large datasets to extract useful information.",
        "Neural networks learn hierarchical representations from raw data automatically.",
        "Transformer architectures power modern language models used in production today.",
        "Vector embeddings capture semantic meaning in high dimensional spaces for retrieval.",
        "Artificial intelligence continues to transform industries across the global economy.",
        "Gradient descent iteratively updates parameters to minimize model loss functions.",
        "Attention mechanisms allow models to focus on the most relevant parts of input."
    ]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
                                              padding_side="left",
                                              trust_remote_code=True)
    hf_model = AutoModel.from_pretrained(MODEL_ID,
                                         trust_remote_code=True,
                                         torch_dtype=torch.float32)
    hf_model.eval()

    with monkeypatch.context():
        vllm_embeds = vllm_embeddings(prompts)
    hf_embeds = hf_embeddings(prompts, hf_model, tokenizer)

    assert vllm_embeds.shape == hf_embeds.shape == (len(prompts), hf_embeds.shape[1])

    comparisons = compare_embeddings(vllm_embeds, hf_embeds)
    for idx, (is_close, max_diff, cos_sim) in enumerate(comparisons):
        assert is_close, (
            f"Embedding {idx} mismatch (max_diff={max_diff:.2e}, cos_sim={cos_sim:.6f})")
