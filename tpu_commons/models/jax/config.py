from functools import cached_property
from typing import Any, List, Optional

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.utils import is_tpu_v5e, pad_to_multiple

logger = init_logger(__name__)

HEAD_DIM_MULTIPLE = 128
MAX_ALLOWED_PAGE_INDICES_N = (
    128 * 1024
)  # Based on experiments on v5e, 256x1024 results in smem oom but 128x1024 not. TODO: Adjust this based on TPU version.

# When chunked prefill is enabled, this is the max number of prefill segments that
# could scheduled in one token batch.
MAX_PREFILL_SEQS_PER_TOKEN_BATCH = 5


class ModelConfig:

    def __init__(
        self,
        model: str,
        tokenizer: str,
        tokenizer_mode: str,
        trust_remote_code: bool,
        backend: str,
        device: str,
        load_format: str,
        dtype: str,
        max_model_len: int,
        seed: int,
        enable_jit: bool,
        warmup: bool,
        sliding_window: int,
        profile_logdir: str,
        log_level: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.backend = backend
        self.device = device
        self.load_format = load_format
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.seed = seed
        self.enable_jit = enable_jit
        self.warmup = warmup
        self.sliding_window = sliding_window
        self.profile_logdir = profile_logdir
        self.log_level = log_level

        if self.tokenizer is None:
            self.tokenizer = self.model
        self._reset_dtype()
        # TODO: circular import
        from tpu_commons.models.jax.config import get_config
        self.hf_config = get_config(self.model, self.trust_remote_code)
        self._validate_hf_config()

    def _validate_hf_config(self) -> None:
        # Huggingface config mixes using some properties sometimes, so unify them.
        if hasattr(self.hf_config, "sliding_window_size"):
            if not hasattr(self.hf_config, "sliding_window"):
                self.hf_config.sliding_window = self.hf_config.sliding_window_size
            delattr(self.hf_config, "sliding_window_size")

        if hasattr(self.hf_config, "hidden_activation"):
            if not hasattr(self.hf_config, "hidden_act"):
                self.hf_config.hidden_act = self.hf_config.hidden_activation
            delattr(self.hf_config, "hidden_activation")

        if self.sliding_window is not None:
            self.hf_config.sliding_window = self.sliding_window

        if self.max_model_len is not None:
            self.config_to_use.max_position_embeddings = self.max_model_len

        config_objs = []
        if hasattr(self.hf_config, "text_config"):
            config_objs.append(self.hf_config.text_config)
        if hasattr(self.hf_config, "vision_config"):
            config_objs.append(self.hf_config.vision_config)
        if not config_objs:
            config_objs = [self.hf_config]

        for config_obj in config_objs:
            if not hasattr(config_obj, "head_dim"):
                config_obj.head_dim = (config_obj.hidden_size //
                                       config_obj.num_attention_heads)

            # PagedAttention kernel requires the head_dim to be multiple of 128.
            # So we pad it for the PagedAttention kernel, and store the original
            # value for the use of scaling q in the attention layer.
            if config_obj.head_dim % HEAD_DIM_MULTIPLE != 0:
                config_obj.head_dim_original = config_obj.head_dim
                config_obj.head_dim = pad_to_multiple(config_obj.head_dim,
                                                      HEAD_DIM_MULTIPLE)

        # Phi-3 mini and medium 4k - sliding window
        if (getattr(self.hf_config, "model_type", None) == "phi3"
                and getattr(self.hf_config, "sliding_window", None) == 2047):
            self.hf_config.sliding_window = self.hf_config.sliding_window + 1
            self.sliding_window = self.hf_config.sliding_window

        # Phi-3 medium variants kv heads padding due to kv heads being 10
        # KV heads must be divisible by tensor parallel size
        # Can only use Phi-3 medium variant with 4 TP due to memory
        if (getattr(self.hf_config, "model_type", None) == "phi3" and
                getattr(self.hf_config, "num_key_value_heads", None) % 4 != 0):
            self.hf_config.original_num_kv_heads = self.hf_config.num_key_value_heads

            # If not TPU_V5E, then megacore mode "kv_head" is enabled
            # In this case, kv heads by tp must be an even number
            # Duplicates kv_head value until desired condition is met
            while (self.hf_config.num_key_value_heads % 4 != 0) or (
                    not is_tpu_v5e() and
                (self.hf_config.num_key_value_heads / 4) % 2 != 0):
                self.hf_config.num_key_value_heads = (
                    self.hf_config.num_key_value_heads * 2)

        # Gemma2 model mixes global and sliding attentions.
        if getattr(self.hf_config, "model_type", None) == "gemma2":
            self.hf_config.attn_types = ["global", "sliding"]

        # Llama 3.2 model uses embedding as the last layer.
        if (getattr(self.hf_config, "model_type", None) == "llama"
                and "3.2" in self.model):
            self.hf_config.use_embedding_as_last_layer = True

        if self.is_moe() and not (
                hasattr(self.hf_config, "num_experts_per_tok") or
            (hasattr(self.hf_config, "text_config")
             and hasattr(self.hf_config.text_config, "num_experts_per_tok"))):
            raise ValueError(
                "`num_experts_per_tok` not found in the HF config for a MoE model."
            )

        # Add rope scaling if Qwen 2.5 7B or 14B
        # Recommendation from HuggingFace to add rope_scaling dict manually:
        # https://huggingface.co/Qwen/Qwen2.5-7B-Instruct#processing-long-texts
        # ruff: noqa: E712
        if (getattr(self.hf_config, "model_type", None) == "qwen2" and getattr(
                self.hf_config, "tie_word_embeddings", None) == False):
            self.hf_config.rope_scaling = {
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
                "type": "yarn",
            }

        # Llama 4 HF implementation has hard-coded nope_layer_interval into the code.
        nope_layer_interval = getattr(self.config_to_use,
                                      "nope_layer_interval", None)
        if self.hf_config.model_type == "llama4" and nope_layer_interval is None:
            logger.warning("Setting `nope_layer_interval` to 4 for Llama 4.")
            self.config_to_use.nope_layer_interval = 4

    def _reset_dtype(self) -> None:
        dtype_str_to_dtype_map = None
        if self.backend == "pytorch":
            import torch

            dtype_str_to_dtype_map = {
                "float16": torch.float16,
                "float": torch.float32,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }
        else:
            import jax.numpy as jnp

            dtype_str_to_dtype_map = {
                "float16":
                jnp.bfloat16,  # attention kernels don't support f16.
                "float": jnp.float32,
                "float32": jnp.float32,
                "bfloat16": jnp.bfloat16,
            }
        self.dtype = dtype_str_to_dtype_map[self.dtype]

    @cached_property
    def config_to_use(self):
        """This returns text config to use.

        Multi-modal models in HF store text config in `text_config` attribute.
        """
        config_to_use = self.hf_config
        if hasattr(config_to_use, "text_config"):
            config_to_use = config_to_use.text_config

        return config_to_use

    def set_eos_token_ids(self, tokenizer: Any) -> None:
        eos_token_ids = []
        if hasattr(self.config_to_use, "eos_token_id"):
            if isinstance(self.config_to_use.eos_token_id, int):
                eos_token_ids.append(self.config_to_use.eos_token_id)
            elif isinstance(self.config_to_use.eos_token_id, list):
                eos_token_ids.extend(self.config_to_use.eos_token_id)
        eos_token_ids.append(tokenizer.eos_token_id)
        self.eos_token_ids = list(set(eos_token_ids))

    def is_mixed_attentions(self) -> bool:
        return hasattr(self.hf_config, "attn_types")

    def is_moe(self) -> bool:
        return hasattr(self.config_to_use, "num_local_experts")

    def is_mistral(self) -> bool:
        return (self.hf_config.model_type == "mistral"
                or self.hf_config.model_type == "mixtral")

    def get_sliding_window(self) -> Optional[int]:
        return getattr(self.hf_config, "sliding_window", None)

    def get_vocab_size(self) -> int:
        return self.config_to_use.vocab_size

    def get_hidden_size(self) -> int:
        return self.config_to_use.hidden_size

    def get_num_attention_heads(self) -> int:
        return self.config_to_use.num_attention_heads

    def get_num_kv_heads(self, tp_size: Optional[int] = None) -> int:
        if hasattr(self.config_to_use, "num_key_value_heads"):
            num_kv_heads = self.config_to_use.num_key_value_heads
        else:
            num_kv_heads = self.config_to_use.num_attention_heads
        if tp_size is None:
            return num_kv_heads
        return max(num_kv_heads, tp_size)

    def get_head_dim(self) -> int:
        return self.config_to_use.head_dim

    def get_max_model_len(self) -> int:
        return self.config_to_use.max_position_embeddings

    def get_num_layers(self) -> int:
        return self.config_to_use.num_hidden_layers

    def get_intermediate_size(self) -> int:
        return self.config_to_use.intermediate_size

    def __repr__(self) -> str:
        return get_repr(self)


class ParallelConfig:

    def __init__(
        self,
        data_parallel_size: int,
        tensor_parallel_size: int,
        num_hosts: int,
        worker_distributed_method: str,
        disagg_topo: str,
    ) -> None:
        self.data_parallel_size = data_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.num_hosts = num_hosts
        self.worker_distributed_method = worker_distributed_method
        self.disagg_topo = disagg_topo

        self.num_workers = None
        self.num_prefill_workers = None
        self.num_decode_workers = None
        self.num_runners_per_worker = None
        self.num_devices_per_runner = None

    def __repr__(self) -> str:
        return get_repr(self)


class CacheConfig:

    def __init__(
        self,
        hbm_utilization_factor: float,
        block_size: int,
        num_blocks: Optional[int] = None,
        output_logits: bool = False,
        num_tokens_in_logits_cache: Optional[int] = None,
        kv_cache_eviction_algorithm: Optional[str] = None,
        sink_size: Optional[int] = None,
        cache_attention_scores: bool = False,
        cache_all_prefill_logits: bool = False,
        perplexity_reference_text_token_ids: Optional[List[int]] = None,
        enable_prefix_cache_hbm: bool = False,
    ) -> None:
        self.hbm_utilization_factor = hbm_utilization_factor
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.output_logits = output_logits
        self.num_tokens_in_logits_cache = num_tokens_in_logits_cache or 1
        self.kv_cache_eviction_algorithm = kv_cache_eviction_algorithm
        self.sink_size = sink_size
        self.cache_attention_scores = cache_attention_scores
        self.cache_all_prefill_logits = cache_all_prefill_logits
        self.perplexity_reference_text_token_ids = perplexity_reference_text_token_ids
        self.enable_prefix_cache_hbm = enable_prefix_cache_hbm

    def __repr__(self) -> str:
        return get_repr(self)


class SchedulerConfig:

    def __init__(
        self,
        max_prefill_seqs: int,
        prefill_seqs_padding: int,
        prefill_len_padding: int,
        max_decode_seqs: int,
        decode_seqs_padding: int,
        decode_blocks_padding: int,
        enable_chunked_prefill: bool = False,
        num_tokens_per_batch: int = 768,
    ) -> None:
        self.max_prefill_seqs = max_prefill_seqs or 1
        self.prefill_seqs_padding = prefill_seqs_padding or 8
        print("Padding prefill batch size (prefill_seqs_padding) to: ",
              self.prefill_seqs_padding)
        self.prefill_len_padding = prefill_len_padding or 512
        print("Prefill padding len is: ", self.prefill_len_padding)
        self.max_decode_seqs = max_decode_seqs or 256
        self.decode_seqs_padding = decode_seqs_padding or 8
        self.decode_blocks_padding = decode_blocks_padding or 128
        self.enable_chunked_prefill = enable_chunked_prefill
        self.num_tokens_per_batch = num_tokens_per_batch
        # When chunked prefill is enabled, each token batch will be padded
        # to multiple of `chunked_prefill_tokens_padding` to reduce number
        # of needed XLA compiled graphs.
        self.chunked_prefill_tokens_padding = 256

    def __repr__(self) -> str:
        return get_repr(self)


class LoRAConfig:

    def __init__(
        self,
        enable_lora: bool,
        max_lora_rank: int,
        max_num_lora: int,
        dtype: str,
        enable_lora_cache: bool,
        max_num_mem_cached_lora: int,
    ) -> None:
        self.enable_lora = enable_lora
        self.max_lora_rank = max_lora_rank
        self.max_num_lora = max_num_lora
        self.dtype = dtype
        self.enable_lora_cache = enable_lora_cache
        self.max_num_mem_cached_lora = max_num_mem_cached_lora

    def __repr__(self) -> str:
        return get_repr(self)


def get_repr(config):
    properties = vars(config)
    s = ""
    for k, v in properties.items():
        s += f"{k}={v}, "
    return f"({s})"
