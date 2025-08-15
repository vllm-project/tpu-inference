# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from transformers import LlamaConfig
from vllm.config import VllmConfig

from tpu_commons.models.jax.llama3 import LlamaForCausalLM
from tpu_commons.models.jax.utils.weight_utils import (TpuCommonAnnotation,
                                                       build_annotation_map)


class Llama3AnnotationMapTest(unittest.TestCase):

    def test_build_annotation_map_for_llama3(self):
        if not jax.devices():
            self.skipTest("No JAX devices available for mesh creation.")
        devices = np.array(jax.local_devices())
        mesh = Mesh(devices, axis_names=("model", ))

        rng = nnx.Rngs(0)
        # Use a small config for the test to keep it fast and memory-efficient
        vllm_config = MagicMock(spec=VllmConfig)
        hf_config = LlamaConfig(
            num_hidden_layers=2,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=False,
        )
        vllm_config.model_config.hf_config = hf_config
        vllm_config.model_config.get_vocab_size.return_value = 1000
        vllm_config.model_config.get_hidden_size.return_value = 128
        vllm_config.model_config.dtype = jnp.bfloat16

        model = LlamaForCausalLM(vllm_config, rng.params(), mesh)

        annotation_map = build_annotation_map(model)

        anno_embed = TpuCommonAnnotation(hf_name="model.embed_tokens")
        anno_lm_head = TpuCommonAnnotation(hf_name="lm_head",
                                           transpose_param=(1, 0))
        anno_norm = TpuCommonAnnotation(hf_name="norm")
        anno_input_ln = TpuCommonAnnotation(hf_name="input_layernorm")
        anno_post_attn_ln = TpuCommonAnnotation(
            hf_name="post_attention_layernorm")

        anno_q_proj = TpuCommonAnnotation(hf_name="q_proj",
                                          reshape_param=(8, 32, 128),
                                          bias_reshape_param=(8, 32),
                                          transpose_param=(2, 0, 1),
                                          pad_param=(1, 1),
                                          bias_pad_param=(1, 1))

        anno_k_proj = TpuCommonAnnotation(hf_name="k_proj",
                                          reshape_param=(8, 32, 128),
                                          bias_reshape_param=(8, 32),
                                          transpose_param=(2, 0, 1),
                                          pad_param=(1, 1),
                                          bias_pad_param=(1, 1))

        anno_v_proj = TpuCommonAnnotation(hf_name="v_proj",
                                          reshape_param=(8, 32, 128),
                                          bias_reshape_param=(8, 32),
                                          transpose_param=(2, 0, 1),
                                          pad_param=(1, 1),
                                          bias_pad_param=(1, 1))

        anno_o_proj = TpuCommonAnnotation(hf_name="o_proj",
                                          reshape_param=(128, 8, 32),
                                          transpose_param=(1, 2, 0),
                                          pad_param=(1, 1))

        anno_gate_proj = TpuCommonAnnotation(hf_name="gate_proj",
                                             transpose_param=(1, 0))
        anno_up_proj = TpuCommonAnnotation(hf_name="up_proj",
                                           transpose_param=(1, 0))
        anno_down_proj = TpuCommonAnnotation(hf_name="down_proj",
                                             transpose_param=(1, 0))

        expected_map = {
            "model.embed_tokens": ("embed.embedding", anno_embed),
            "lm_head": ("lm_head", anno_lm_head),
            "model.norm": ("model.norm.scale", anno_norm),
            "model.layers.0.input_layernorm":
            ("model.layers.0.input_layernorm.scale", anno_input_ln),
            "model.layers.0.post_attention_layernorm":
            ("model.layers.0.post_attention_layernorm.scale",
             anno_post_attn_ln),
            "model.layers.0.self_attn.q_proj":
            ("model.layers.0.self_attn.q_proj.kernel", anno_q_proj),
            "model.layers.0.self_attn.k_proj":
            ("model.layers.0.self_attn.k_proj.kernel", anno_k_proj),
            "model.layers.0.self_attn.v_proj":
            ("model.layers.0.self_attn.v_proj.kernel", anno_v_proj),
            "model.layers.0.self_attn.o_proj":
            ("model.layers.0.self_attn.o_proj.kernel", anno_o_proj),
            "model.layers.0.mlp.gate_proj":
            ("model.layers.0.mlp.gate_proj.kernel", anno_gate_proj),
            "model.layers.0.mlp.up_proj": ("model.layers.0.mlp.up_proj.kernel",
                                           anno_up_proj),
            "model.layers.0.mlp.down_proj":
            ("model.layers.0.mlp.down_proj.kernel", anno_down_proj),
            "model.layers.1.input_layernorm":
            ("model.layers.1.input_layernorm.scale", anno_input_ln),
            "model.layers.1.post_attention_layernorm":
            ("model.layers.1.post_attention_layernorm.scale",
             anno_post_attn_ln),
            "model.layers.1.self_attn.q_proj":
            ("model.layers.1.self_attn.q_proj.kernel", anno_q_proj),
            "model.layers.1.self_attn.k_proj":
            ("model.layers.1.self_attn.k_proj.kernel", anno_k_proj),
            "model.layers.1.self_attn.v_proj":
            ("model.layers.1.self_attn.v_proj.kernel", anno_v_proj),
            "model.layers.1.self_attn.o_proj":
            ("model.layers.1.self_attn.o_proj.kernel", anno_o_proj),
            "model.layers.1.mlp.gate_proj":
            ("model.layers.1.mlp.gate_proj.kernel", anno_gate_proj),
            "model.layers.1.mlp.up_proj": ("model.layers.1.mlp.up_proj.kernel",
                                           anno_up_proj),
            "model.layers.1.mlp.down_proj":
            ("model.layers.1.mlp.down_proj.kernel", anno_down_proj),
        }

        self.assertEqual(len(annotation_map), len(expected_map),
                         f"{annotation_map}")
        for hf_key, (model_path, anno) in expected_map.items():
            self.assertIn(hf_key, annotation_map)
            actual_model_path, actual_anno = annotation_map[hf_key]
            self.assertEqual(actual_model_path, model_path)
            self.assertEqual(actual_anno, anno, f"{model_path}, {anno}")
