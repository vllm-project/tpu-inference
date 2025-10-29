from tpu_inference.models.common.model_loader import register_model


class DummyModel:
    def __init__(self, vllm_config=None): pass
    def __call__(self, kv_caches=None, input_ids=None, attention_metadata=None): pass

def test_vllm_wrapper_has_required_methods():
    register_model("DummyForCausalLM", DummyModel)

    from vllm.model_executor.models.registry import ModelRegistry
    wrapper_cls = ModelRegistry.models.get("DummyForCausalLM").model_cls
    assert hasattr(wrapper_cls, "get_input_embeddings")
    m = wrapper_cls()
    try:
        m.get_input_embeddings(input_ids=None, positions=None, inputs_embeds=None)
    except NotImplementedError:
        pass

    from vllm.model_executor.models.interfaces_base import is_vllm_model
    assert is_vllm_model(wrapper_cls)
