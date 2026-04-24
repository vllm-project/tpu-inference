import pytest
from unittest.mock import MagicMock, patch

from tpu_inference.models.vllm.vllm_model_wrapper import VllmModelWrapper

def test_vllm_model_wrapper_applies_generic_patches():
    # Setup dummy dependencies
    mock_vllm_config = MagicMock()
    mock_vllm_config.speculative_config = None
    mock_vllm_config.model_config.is_multimodal_model = False
    
    import torch
    mock_vllm_config.model_config.dtype = torch.bfloat16
    
    mock_rng = MagicMock()
    mock_mesh = MagicMock()
    
    # We patch multiple internal methods/imports of VllmModelWrapper so it doesn't crash
    # when we only want to test the invocation of apply_generic_tpu_patches.
    with patch('tpu_inference.models.vllm.vllm_model_wrapper.get_tpu_quantization_config'), \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.VllmModelWrapper._apply_pp_patch'), \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.set_current_vllm_config'), \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.vllm_get_model') as mock_vllm_get_model, \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.replace_set_lora'), \
         patch('tpu_inference.models.vllm.vllm_model_wrapper._VllmRunner'), \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.shard_model_to_tpu') as mock_shard, \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.load_lora_model') as mock_load_lora_model, \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.apply_generic_tpu_patches') as mock_apply_generic_tpu_patches, \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.maybe_apply_qwen3_vl_patches'):
         
        mock_vllm_get_model.return_value = MagicMock()
        mock_load_lora_model.return_value = (MagicMock(), mock_vllm_get_model.return_value)
        mock_shard.return_value = {}
        
        # Instantiate
        wrapper = VllmModelWrapper(
            vllm_config=mock_vllm_config,
            rng=mock_rng,
            mesh=mock_mesh,
        )
        
        # Call load_weights
        wrapper.load_weights()
        
        # Assert the patches are applied!
        mock_apply_generic_tpu_patches.assert_called_once()
