from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.attention.backends.abstract import AttentionType
from vllm.config import ModelConfig, SchedulerConfig, VllmConfig

from tpu_commons.attention.backends.pallas_torchax import (
    PallasAttentionBackend, PallasAttentionBackendImpl, PallasMetadata,
    write_to_kv_cache)


class TestPallasMetadata:

    def test_init(self):
        slot_mapping = torch.tensor([1, 2, 3])
        block_tables = torch.tensor([[1, 2], [3, 4]])
        context_lens = torch.tensor([10, 20])
        query_start_loc = torch.tensor([0, 10])
        num_seqs = torch.tensor([2])
        num_slices = torch.tensor([1])

        metadata = PallasMetadata(slot_mapping=slot_mapping,
                                  block_tables=block_tables,
                                  context_lens=context_lens,
                                  query_start_loc=query_start_loc,
                                  num_seqs=num_seqs,
                                  num_slices=num_slices)

        assert torch.equal(metadata.slot_mapping, slot_mapping)
        assert torch.equal(metadata.block_tables, block_tables)
        assert torch.equal(metadata.context_lens, context_lens)
        assert torch.equal(metadata.query_start_loc, query_start_loc)
        assert torch.equal(metadata.num_seqs, num_seqs)
        assert torch.equal(metadata.num_slices, num_slices)

    def test_tree_flatten_unflatten(self):
        slot_mapping = torch.tensor([1, 2, 3])
        block_tables = torch.tensor([[1, 2], [3, 4]])
        context_lens = torch.tensor([10, 20])
        query_start_loc = torch.tensor([0, 10])
        num_seqs = torch.tensor([2])
        num_slices = torch.tensor([1])

        metadata = PallasMetadata(slot_mapping=slot_mapping,
                                  block_tables=block_tables,
                                  context_lens=context_lens,
                                  query_start_loc=query_start_loc,
                                  num_seqs=num_seqs,
                                  num_slices=num_slices)

        children, aux_data = metadata.tree_flatten()
        reconstructed = PallasMetadata.tree_unflatten(aux_data, children)

        assert torch.equal(reconstructed.slot_mapping, metadata.slot_mapping)
        assert torch.equal(reconstructed.block_tables, metadata.block_tables)
        assert torch.equal(reconstructed.context_lens, metadata.context_lens)
        assert torch.equal(reconstructed.query_start_loc,
                           metadata.query_start_loc)
        assert torch.equal(reconstructed.num_seqs, metadata.num_seqs)
        assert torch.equal(reconstructed.num_slices, metadata.num_slices)
        assert aux_data is None


class TestPallasAttentionBackend:

    def test_get_state_cls(self):
        from vllm.attention.backends.utils import CommonAttentionState
        assert PallasAttentionBackend.get_state_cls() == CommonAttentionState

    def test_get_name(self):
        assert PallasAttentionBackend.get_name() == "PALLAS_VLLM_V1"

    def test_get_impl_cls(self):
        assert PallasAttentionBackend.get_impl_cls(
        ) == PallasAttentionBackendImpl

    def test_get_metadata_cls(self):
        assert PallasAttentionBackend.get_metadata_cls() == PallasMetadata

    def test_get_kv_cache_shape(self):
        num_blocks = 10
        block_size = 16
        num_kv_heads = 8
        head_size = 256

        shape = PallasAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size)

        expected_shape = (num_blocks, block_size, num_kv_heads * 2, head_size)
        assert shape == expected_shape

    def test_get_kv_cache_shape_unaligned_head_size(self):
        num_blocks = 10
        block_size = 16
        num_kv_heads = 8
        head_size = 96  # Not aligned to 128

        shape = PallasAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size)

        # 96 should be padded to 128
        padded_head_size = 128
        expected_shape = (num_blocks, block_size, num_kv_heads * 2,
                          padded_head_size)
        assert shape == expected_shape

    def test_swap_blocks_raises_error(self):
        src_kv_cache = torch.empty(0)
        dst_kv_cache = torch.empty(0)
        src_to_dst = torch.empty(0)

        with pytest.raises(
                RuntimeError,
                match="swap_blocks is not used for the TPU backend"):
            PallasAttentionBackend.swap_blocks(src_kv_cache, dst_kv_cache,
                                               src_to_dst)

    def test_get_min_page_size(self):
        model_config = MagicMock(spec=ModelConfig)
        model_config.max_model_len = 2048

        scheduler_config = MagicMock(spec=SchedulerConfig)
        scheduler_config.max_num_seqs = 256

        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.model_config = model_config
        vllm_config.scheduler_config = scheduler_config

        min_page_size = PallasAttentionBackend.get_min_page_size(vllm_config)
        assert min_page_size > 0
        # Should be a power of 2
        assert (min_page_size & (min_page_size - 1)) == 0

    def test_get_page_size(self):
        model_config = MagicMock(spec=ModelConfig)
        model_config.max_model_len = 2048

        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.model_config = model_config

        page_size = PallasAttentionBackend.get_page_size(vllm_config)
        assert 16 <= page_size <= 256
        # Should be a power of 2
        assert (page_size & (page_size - 1)) == 0

    def test_get_page_size_small_model_len(self):
        model_config = MagicMock(spec=ModelConfig)
        model_config.max_model_len = 64  # Small model length

        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.model_config = model_config

        page_size = PallasAttentionBackend.get_page_size(vllm_config)
        assert page_size == 16

    def test_get_page_size_large_model_len(self):
        model_config = MagicMock(spec=ModelConfig)
        model_config.max_model_len = 8192  # Large model length

        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.model_config = model_config

        page_size = PallasAttentionBackend.get_page_size(vllm_config)
        assert page_size == 256


class TestPallasAttentionBackendImpl:

    def test_init_valid_params(self):
        impl = PallasAttentionBackendImpl(
            num_heads=32,
            head_size=128,
            scale=0.088,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
        )

        assert impl.num_heads == 32
        assert impl.head_size == 128
        assert impl.scale == 0.088
        assert impl.num_kv_heads == 8
        assert impl.num_queries_per_kv == 4
        assert impl.sliding_window is None

    def test_init_with_alibi_slopes_raises_error(self):
        with pytest.raises(NotImplementedError,
                           match="Alibi slopes is not supported"):
            PallasAttentionBackendImpl(
                num_heads=32,
                head_size=128,
                scale=0.088,
                num_kv_heads=8,
                alibi_slopes=[1.0, 2.0],
                sliding_window=None,
                kv_cache_dtype="auto",
                attn_type=AttentionType.DECODER,
            )

    def test_init_with_fp8_kv_cache_raises_error(self):
        with pytest.raises(NotImplementedError,
                           match="FP8 KV cache dtype is not supported"):
            PallasAttentionBackendImpl(
                num_heads=32,
                head_size=128,
                scale=0.088,
                num_kv_heads=8,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="fp8",
                attn_type=AttentionType.DECODER,
            )

    def test_init_with_encoder_attention_raises_error(self):
        with pytest.raises(NotImplementedError,
                           match="Encoder self-attention"):
            PallasAttentionBackendImpl(
                num_heads=32,
                head_size=128,
                scale=0.088,
                num_kv_heads=8,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="auto",
                attn_type=AttentionType.ENCODER,
            )

    @patch(
        'tpu_commons.attention.backends.pallas_torchax.ragged_paged_attention')
    @patch('tpu_commons.attention.backends.pallas_torchax.get_forward_context')
    def test_forward_empty_kv_cache(self, mock_get_context, mock_attention):
        impl = PallasAttentionBackendImpl(
            num_heads=32,
            head_size=128,
            scale=0.088,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
        )

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        query = torch.randn(2, 4096)  # 2 tokens, 32 heads * 128 head_size
        key = torch.randn(2, 1024)  # 2 tokens, 8 kv_heads * 128 head_size
        value = torch.randn(2, 1024)
        kv_cache = torch.empty(0)  # Empty cache

        metadata = PallasMetadata(slot_mapping=torch.tensor([0, 1]),
                                  block_tables=torch.tensor([[0, 1]]),
                                  context_lens=torch.tensor([2]),
                                  query_start_loc=torch.tensor([0, 2]),
                                  num_seqs=torch.tensor([1]),
                                  num_slices=torch.tensor([1]))

        result = impl.forward(layer, query, key, value, kv_cache, metadata)
        assert result.shape == query.shape
        mock_attention.assert_not_called()

    def test_forward_with_output_scale_raises_error(self):
        impl = PallasAttentionBackendImpl(
            num_heads=32,
            head_size=128,
            scale=0.088,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
        )

        layer = MagicMock()
        query = torch.randn(2, 512)
        key = torch.randn(2, 256)
        value = torch.randn(2, 256)
        kv_cache = torch.randn(10, 16, 16, 128)
        metadata = MagicMock()
        output_scale = torch.tensor([1.0])

        with pytest.raises(NotImplementedError,
                           match="fused output quantization"):
            impl.forward(layer,
                         query,
                         key,
                         value,
                         kv_cache,
                         metadata,
                         output_scale=output_scale)

    def test_init_with_irope_warning(self):
        with patch('tpu_commons.attention.backends.pallas_torchax.logger'
                   ) as mock_logger:
            _ = PallasAttentionBackendImpl(
                num_heads=32,
                head_size=128,
                scale=0.088,
                num_kv_heads=8,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="auto",
                attn_type=AttentionType.DECODER,
                use_irope=True,
            )
            mock_logger.warning_once.assert_called_once_with(
                "Using irope in Pallas is not supported yet, it will fall back "
                "to global attention for long context.")

    @patch(
        'tpu_commons.attention.backends.pallas_torchax.ragged_paged_attention')
    @patch('tpu_commons.attention.backends.pallas_torchax.get_forward_context')
    @patch('tpu_commons.attention.backends.pallas_torchax.write_to_kv_cache')
    def test_forward_full_flow(self, mock_write_kv, mock_get_context,
                               mock_attention):
        impl = PallasAttentionBackendImpl(
            num_heads=32,
            head_size=128,
            scale=0.088,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
        )

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0
        layer.kv_cache = {}

        mock_context = MagicMock()
        mock_context.virtual_engine = 0
        mock_get_context.return_value = mock_context

        query = torch.randn(2, 4096)  # 2 tokens, 32 heads * 128 head_size
        key = torch.randn(2, 1024)  # 2 tokens, 8 kv_heads * 128 head_size
        value = torch.randn(2, 1024)
        kv_cache = torch.randn(10, 16, 16, 128)  # Non-empty cache

        metadata = PallasMetadata(slot_mapping=torch.tensor([0, 1]),
                                  block_tables=torch.tensor([[0, 1]]),
                                  context_lens=torch.tensor([2]),
                                  query_start_loc=torch.tensor([0, 2]),
                                  num_seqs=torch.tensor([1]),
                                  num_slices=torch.tensor([1]))

        # Mock write_to_kv_cache to return the same cache
        mock_write_kv.return_value = kv_cache

        # Mock attention output
        mock_attention.return_value = torch.randn(2, 32, 128)

        result = impl.forward(layer, query, key, value, kv_cache, metadata)

        # Verify write_to_kv_cache was called
        mock_write_kv.assert_called_once()

        # Verify ragged_paged_attention was called
        mock_attention.assert_called_once()

        # Check result shape
        assert result.shape == (2, 4096)

    @patch(
        'tpu_commons.attention.backends.pallas_torchax.ragged_paged_attention')
    @patch('tpu_commons.attention.backends.pallas_torchax.get_forward_context')
    def test_forward_with_kv_sharing(self, mock_get_context, mock_attention):
        impl = PallasAttentionBackendImpl(
            num_heads=32,
            head_size=128,
            scale=0.088,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name="earlier_layer",
        )

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        query = torch.randn(2, 4096)  # 2 tokens, 32 heads * 128 head_size
        key = torch.randn(2, 1024)  # 2 tokens, 8 kv_heads * 128 head_size
        value = torch.randn(2, 1024)
        kv_cache = torch.randn(10, 16, 16, 128)

        metadata = PallasMetadata(slot_mapping=torch.tensor([0, 1]),
                                  block_tables=torch.tensor([[0, 1]]),
                                  context_lens=torch.tensor([2]),
                                  query_start_loc=torch.tensor([0, 2]),
                                  num_seqs=torch.tensor([1]),
                                  num_slices=torch.tensor([1]))

        mock_attention.return_value = torch.randn(2, 32, 128)

        result = impl.forward(layer, query, key, value, kv_cache, metadata)

        # Verify get_forward_context was not called (KV cache sharing skips write)
        mock_get_context.assert_not_called()
        assert result.shape == (2, 4096)

    @patch(
        'tpu_commons.attention.backends.pallas_torchax.ragged_paged_attention')
    @patch('tpu_commons.attention.backends.pallas_torchax.get_forward_context')
    @patch('tpu_commons.attention.backends.pallas_torchax.write_to_kv_cache')
    def test_forward_with_head_padding(self, mock_write_kv, mock_get_context,
                                       mock_attention):
        impl = PallasAttentionBackendImpl(
            num_heads=8,
            head_size=96,  # Not aligned to 128, will need padding
            scale=0.25,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
        )

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0
        layer.kv_cache = {}

        mock_context = MagicMock()
        mock_context.virtual_engine = 0
        mock_get_context.return_value = mock_context

        query = torch.randn(2, 768)  # 2 tokens, 8 heads * 96 head_size
        key = torch.randn(2, 768)  # 2 tokens, 8 kv_heads * 96 head_size
        value = torch.randn(2, 768)
        kv_cache = torch.randn(10, 16, 16, 128)  # Padded head size

        metadata = PallasMetadata(slot_mapping=torch.tensor([0, 1]),
                                  block_tables=torch.tensor([[0, 1]]),
                                  context_lens=torch.tensor([2]),
                                  query_start_loc=torch.tensor([0, 2]),
                                  num_seqs=torch.tensor([1]),
                                  num_slices=torch.tensor([1]))

        mock_write_kv.return_value = kv_cache
        # Return padded output that will be trimmed
        mock_attention.return_value = torch.randn(2, 8, 128)

        result = impl.forward(layer, query, key, value, kv_cache, metadata)

        # Result should be trimmed back to original head size
        assert result.shape == (2, 768)  # 8 heads * 96 original head_size


@patch('tpu_commons.attention.backends.pallas_torchax.call_jax')
@patch('tpu_commons.attention.backends.pallas_torchax.kv_cache_update')
def test_write_to_kv_cache(mock_kv_cache_update, mock_call_jax):
    # Mock the JAX function call to return the same kv_cache
    mock_call_jax.return_value = torch.randn(160, 16, 128)  # reshaped size

    key = torch.randn(2, 8, 128)  # 2 tokens, 8 kv_heads, 128 head_size
    value = torch.randn(2, 8, 128)  # 2 tokens, 8 kv_heads, 128 head_size
    kv_cache = torch.randn(
        10, 16, 16,
        128)  # 10 blocks, 16 block_size, 16 kv_heads*2, 128 head_size
    slot_mapping = torch.tensor([[0, 1, 0], [16, 17, 0], [0, 0, 0]])
    num_slices = torch.tensor([1])

    result = write_to_kv_cache(key, value, kv_cache, slot_mapping, num_slices)

    # Verify the JAX function was called
    mock_call_jax.assert_called_once()

    # Check the result shape matches input kv_cache
    assert result.shape == kv_cache.shape

    # Verify kv_cache_update was passed to call_jax
    args, kwargs = mock_call_jax.call_args
    assert args[0] == mock_kv_cache_update
    assert kwargs['page_size'] == 16


def test_write_to_kv_cache_tensor_shapes():
    # Create tensors with correct shapes: key/value should be flattened
    key = torch.randn(3, 1024)  # 3 tokens, 8 kv_heads * 128 head_size
    value = torch.randn(3, 1024)  # 3 tokens, 8 kv_heads * 128 head_size
    kv_cache = torch.randn(
        5, 8, 16, 128)  # 5 blocks, 8 block_size, 16 kv_heads*2, 128 head_size
    slot_mapping = torch.tensor([[0, 1, 2], [8, 9, 10], [0, 0, 0]])
    num_slices = torch.tensor([1])

    with patch('tpu_commons.attention.backends.pallas_torchax.call_jax'
               ) as mock_call_jax:
        # Mock return value with correct shape
        mock_call_jax.return_value = torch.randn(40, 16, 128)  # reshaped size

        result = write_to_kv_cache(key, value, kv_cache, slot_mapping,
                                   num_slices)

        # Check input tensor shapes passed to JAX
        args, kwargs = mock_call_jax.call_args
        kv_input = args[1]  # The concatenated kv tensor

        # kv should be [3 tokens, 16 combined_heads, 128 padded_head_size]
        assert kv_input.shape == (3, 16, 128)
        assert result.shape == kv_cache.shape
