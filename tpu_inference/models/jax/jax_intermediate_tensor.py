import jax
from jax.tree_util import register_pytree_node_class
from typing import TYPE_CHECKING, Any, Dict, Union
from vllm.sequence import IntermediateTensors
from dataclasses import dataclass
from torchax.interop import jax_view, torch_view
if TYPE_CHECKING:
    from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput
else:
    KVConnectorOutput = Any

@register_pytree_node_class
@dataclass
class JaxIntermediateTensors:
    tensors: Dict[str, Any]
    kv_connector_output: KVConnectorOutput = None
    
    def tree_flatten(self):
        children = (self.tensors, )
        aux_data = self.kv_connector_output
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], aux_data)

    @classmethod
    def from_torch(cls, torch_obj: IntermediateTensors):
        kv_connector_output = getattr(torch_obj, 'kv_connector_output', None)
        jax_tensors = {k: jax_view(v) for k, v in torch_obj.tensors.items()}
        return cls(jax_tensors, kv_connector_output)

    def to_torch(self) -> IntermediateTensors:
        torch_tensors = {k: torch_view(v) for k, v in self.tensors.items()}
        return IntermediateTensors(torch_tensors)
    
    def __getitem__(self, key: Union[str, slice]):
        if isinstance(key, str):
            return self.tensors[key]
        elif isinstance(key, slice):
            return self.__class__({k: v[key] for k, v in self.tensors.items()})
    
    def __setitem__(self, key: str, value: Any):
        self.tensors[key] = value

    def keys(self):
        return self.tensors.keys()
    
    def items(self):
        return self.tensors.items()

    def __len__(self):
        return len(self.tensors)

    def block_until_ready(self):
        for tensor in self.tensors.values():
            assert isinstance(tensor, jax.Array), "block_until_ready needs to be applied on jax arrays"
            tensor.block_until_ready()