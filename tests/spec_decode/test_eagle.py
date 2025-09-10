import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax.sharding import Mesh
from vllm.config import CacheConfig

from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.spec_decode.jax.eagle import EagleProposer


class MockModel(nnx.Module):

    def __init__(self, vocab_size, hidden_size, rngs: nnx.Rngs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.rngs = rngs

    def __call__(self, kv_caches, input_ids, hidden_states, positions,
                 attention_metadata):
        # This mock model will generate hidden states that encode the next
        # token ID to be sampled. The token ID is based on the input_ids,
        # which are the previously sampled tokens.
        new_hidden_states = jnp.zeros_like(hidden_states)
        new_hidden_states = new_hidden_states.at[:, 0].set(input_ids + 1)
        return None, new_hidden_states

    def compute_logits(self, hidden_states):
        # Create deterministic logits.
        # Sample token ids based on the first element of the hidden_states.
        token_ids = hidden_states[:, 0].astype(jnp.int32)
        return jax.nn.one_hot(token_ids, self.vocab_size)


@pytest.fixture
def proposer():
    # Mock vllm_config
    class MockVllmConfig:

        def __init__(self):
            self.speculative_config = self.MockSpeculativeConfig()
            self.model_config = self.MockModelConfig()
            self.cache_config = CacheConfig(block_size=16)

        class MockSpeculativeConfig:

            def __init__(self):
                self.num_speculative_tokens = 3
                self.method = "eagle"
                self.draft_model_config = self.MockDraftModelConfig()

            class MockDraftModelConfig:
                pass

        class MockModelConfig:
            pass

    class MockRunner:

        def __init__(self):
            self.mesh = Mesh(jax.devices(), ('model', ))

    proposer = EagleProposer(MockVllmConfig(), MockRunner())
    proposer.model = MockModel(vocab_size=100, hidden_size=4, rngs=nnx.Rngs(0))
    return proposer


def test_propose(proposer):
    # Create dummy inputs
    target_token_ids = jnp.array([1, 2, 3, 4, 5, 6, 7, 8])
    target_positions = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])
    target_hidden_states = jnp.zeros((8, 4))
    # The first draft token will be this + 1.
    next_token_ids = jnp.array([41, 59])

    # Create a realistic AttentionMetadata object
    attn_metadata = AttentionMetadata(
        seq_lens=jnp.array([4, 4]),
        input_positions=jnp.array([4, 12]),
        query_start_loc=jnp.array([0, 4, 8]),
        block_tables=jnp.array([[0, 1], [2, 3]]),
    )

    # Call the propose method
    draft_tokens = proposer.propose(target_token_ids, target_positions,
                                    target_hidden_states, next_token_ids,
                                    attn_metadata)

    # The mock model generates tokens by incrementing the previous token ID.
    # 1. First draft token is next_token_ids + 1 -> [42, 60]
    # 2. Second draft token is [42, 60] + 1 -> [43, 61]
    # 3. Third draft token is [43, 61] + 1 -> [44, 62]
    expected_draft_tokens = jnp.array([[42, 43, 44], [60, 61, 62]])

    assert jnp.array_equal(draft_tokens, expected_draft_tokens)
