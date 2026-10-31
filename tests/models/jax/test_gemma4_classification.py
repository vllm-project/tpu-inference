# SPDX-License-Identifier: Apache-2.0

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from jax.sharding import Mesh
from transformers import Gemma4Config
from vllm.config import ModelConfig, PoolerConfig, set_current_vllm_config
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.pooler.seqwise import (ClassifierPoolerHead,
                                                       LastPool,
                                                       SequencePooler)
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.models.common import model_loader
from tpu_inference.models.jax.gemma4_classification import \
    Gemma4ForSequenceClassification
from tpu_inference.models.jax.sequence_classification import (
    NativeSequenceClassificationModel, NativeSequenceClassificationPooler,
    classify_rows, supports_native_sequence_classification)
from tpu_inference.runner.tpu_runner import TPUModelRunner


def _make_single_device_mesh() -> Mesh:
    devices = np.asarray(jax.local_devices()[:1], dtype=object).reshape(
        (1, ) * len(MESH_AXIS_NAMES))
    return Mesh(devices, MESH_AXIS_NAMES)


def _make_tp_mesh() -> Mesh:
    local = jax.local_devices()
    tp = 2 if len(local) >= 2 else 1
    shape = [1] * len(MESH_AXIS_NAMES)
    shape[MESH_AXIS_NAMES.index("model")] = tp
    devices = np.asarray(local[:tp], dtype=object).reshape(tuple(shape))
    return Mesh(devices, MESH_AXIS_NAMES)


def _make_hf_config(hidden_size: int = 4, **kwargs) -> Gemma4Config:
    text_config = {"hidden_size": hidden_size, **kwargs.pop("text_config", {})}
    return Gemma4Config(architectures=["Gemma4ForSequenceClassification"],
                        text_config=text_config,
                        **kwargs)


def _make_supported_vllm_config(hidden_size: int = 4, num_labels: int = 2):

    class HeadConfig(SimpleNamespace):
        head_dtype = ModelConfig.head_dtype

    return SimpleNamespace(
        model_config=HeadConfig(
            hf_config=_make_hf_config(
                hidden_size,
                id2label={i: str(i)
                          for i in range(num_labels)},
                problem_type="single_label_classification",
            ),
            runner_type="pooling",
            convert_type="classify",
            dtype=torch.bfloat16,
            quantization=None,
            get_hidden_size=lambda: hidden_size,
            pooler_config=PoolerConfig(
                task="classify",
                seq_pooling_type="LAST",
                tok_pooling_type="ALL",
            ),
        ),
        additional_config={},
        cache_config=SimpleNamespace(cache_dtype="auto"),
        load_config=SimpleNamespace(load_format="auto"),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=2,
        ),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=True,
            async_scheduling=False,
        ),
        sharding_config=SimpleNamespace(
            total_dp_size=1,
            prefill_cp_size=1,
        ),
        lora_config=None,
        speculative_config=None,
    )


class TestGemma4SequenceClassification:

    @pytest.mark.parametrize(
        ("mutation", "expected"),
        [
            pytest.param(lambda cfg: None, True, id="supported"),
            pytest.param(
                lambda cfg: setattr(cfg.model_config.hf_config,
                                    "architectures", ["Gemma4ForCausalLM"]),
                False,
                id="bad_arch",
            ),
            pytest.param(
                lambda cfg: setattr(cfg.model_config.pooler_config,
                                    "seq_pooling_type", "MEAN"),
                False,
                id="bad_pool_mean",
            ),
            pytest.param(
                lambda cfg: setattr(cfg.model_config, "quantization", "fp8"),
                False,
                id="bad_quant_fp8",
            ),
            pytest.param(
                lambda cfg: setattr(cfg.sharding_config, "total_dp_size", 2),
                False,
                id="bad_dp_2",
            ),
            pytest.param(
                lambda cfg: setattr(cfg.parallel_config,
                                    "pipeline_parallel_size", 2),
                False,
                id="bad_pp_2",
            ),
            pytest.param(
                lambda cfg: setattr(cfg.sharding_config, "prefill_cp_size", 2),
                False,
                id="bad_cp_2",
            ),
            pytest.param(
                lambda cfg: setattr(cfg.model_config.hf_config,
                                    "classifier_from_token", ["no", "yes"]),
                False,
                id="bad_online_classifier_from_token",
            ),
            pytest.param(
                lambda cfg: setattr(cfg.model_config.hf_config.text_config,
                                    "method", "from_2_way_softmax"),
                False,
                id="bad_online_method",
            ),
        ],
    )
    def test_supports_native_sequence_classification_guards(
            self, mutation, expected):
        cfg = _make_supported_vllm_config()
        mutation(cfg)
        assert supports_native_sequence_classification(cfg) is expected

    @pytest.mark.parametrize(
        ("hf_kwargs", "expected_labels"),
        [
            pytest.param({}, 2, id="default_id2label"),
            pytest.param({"id2label": {
                0: "a",
                1: "b",
                2: "c"
            }},
                         3,
                         id="top_level_labels"),
            pytest.param(
                {"text_config": {
                    "num_labels": 3
                }}, 3, id="text_config_labels"),
        ],
    )
    def test_init_resolves_num_labels_from_real_config(self, hf_kwargs,
                                                       expected_labels):
        mesh = _make_single_device_mesh()
        cfg = _make_supported_vllm_config(hidden_size=4)
        cfg.model_config.dtype = jnp.bfloat16
        cfg.model_config.hf_config = _make_hf_config(4, **hf_kwargs)

        with jax.set_mesh(mesh), patch(
                "tpu_inference.models.jax.gemma4_classification.Gemma4Model",
                return_value=nnx.Module(),
        ):
            model = Gemma4ForSequenceClassification(cfg, jax.random.key(0),
                                                    mesh)
        assert model.num_labels == expected_labels
        assert model.score.weight.get_value().shape == (4, expected_labels)

    def test_init_and_float32_weight_load(self):
        mesh = _make_single_device_mesh()
        cfg = _make_supported_vllm_config(hidden_size=4, num_labels=3)
        cfg.model_config.dtype = jnp.bfloat16

        with jax.set_mesh(mesh), patch(
                "tpu_inference.models.jax.gemma4_classification.Gemma4Model",
                return_value=nnx.Module(),
        ):
            model = Gemma4ForSequenceClassification(cfg, jax.random.key(0),
                                                    mesh)
            assert model.num_labels == 3
            assert model.score.weight.get_value().shape == (4, 3)

            score_f32 = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            with set_current_vllm_config(cfg):
                loaded = model.load_weights([
                    ("score.weight", score_f32),
                    ("lm_head.weight", torch.ones((8, 4),
                                                  dtype=torch.bfloat16)),
                ])
            assert "score.weight" in loaded
            assert model.score.weight.get_value().dtype == jnp.float32
            rows = jnp.ones((2, 4), dtype=jnp.bfloat16)
            logits = model.compute_logits(rows)
            assert logits.dtype == jnp.float32
            assert logits.shape == (2, 3)

    def test_classify_rows_fp32_projection(self):
        rows = jnp.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.0, 3.0]],
                           dtype=jnp.bfloat16)
        weight = jnp.arange(8, dtype=jnp.float32).reshape(4, 2) / 8.0
        logits = classify_rows(rows, weight)
        expected = np.asarray(rows, dtype=np.float32) @ np.asarray(weight)
        assert logits.dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(logits),
                                   expected,
                                   rtol=1e-6,
                                   atol=1e-6)

    def test_load_weights_validates_score_shape(self):
        weight_param = MagicMock()
        weight_param.get_value.return_value = jnp.zeros((4, 2),
                                                        dtype=jnp.bfloat16)
        dummy = SimpleNamespace(
            num_labels=2,
            score=SimpleNamespace(weight=weight_param),
        )

        captured = []

        class FakeLoader:

            def __init__(self, model, skip_prefixes, skip_substrs):
                assert skip_prefixes == ["lm_head"]
                assert "vision" in skip_substrs
                assert "audio" in skip_substrs
                assert "multi_modal" in skip_substrs

            def load_weights(self, iterator):
                captured.extend(list(iterator))
                return {"loaded"}

        with patch(
                "tpu_inference.models.jax.gemma4_classification.JaxAutoWeightsLoader",
                FakeLoader,
        ):
            res = Gemma4ForSequenceClassification.load_weights(
                dummy,
                [
                    ("model.language_model.layers.0.weight", jnp.ones((4, 4))),
                    ("score.weight", jnp.ones((2, 4))),
                ],
            )
            assert res == {"loaded"}
            assert [name for name, _ in captured] == [
                "language_model.layers.0.weight",
                "score.weight",
            ]

            with pytest.raises(
                    ValueError,
                    match=
                    r"score\.weight must have shape \(2, 4\); got \(3, 4\)",
            ):
                Gemma4ForSequenceClassification.load_weights(
                    dummy, [("score.weight", jnp.ones((3, 4)))])

    def test_pooler_and_flax_model_state_refresh(self):
        mesh = _make_tp_mesh()
        cfg = _make_supported_vllm_config(hidden_size=4, num_labels=3)
        rows = jnp.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.0, 3.0]],
                           dtype=jnp.bfloat16)
        weight = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 8.0
        expected = np.asarray(rows, dtype=np.float32) @ np.asarray(weight)
        metadata = PoolingMetadata(
            prompt_lens=torch.tensor([1, 1], dtype=torch.int32),
            prompt_token_ids=None,
            prompt_token_ids_cpu=None,
            pooling_params=[
                PoolingParams(task="classify", use_activation=False)
                for _ in range(2)
            ],
            pooling_states=[PoolingStates(), PoolingStates()],
        )

        class SmallClassifier(nnx.Module, NativeSequenceClassificationModel):

            def __init__(self):
                parameter = nnx.Param(weight)
                parameter.set_metadata("_is_loaded", True)
                self.score = nnx.Dict(weight=parameter)
                self.num_labels = 3

            def compute_logits(self, values):
                return Gemma4ForSequenceClassification.compute_logits(
                    self, values)

        with jax.set_mesh(mesh), contextlib.ExitStack() as stack:
            for target, value in (
                ("_get_nnx_model", SmallClassifier()),
                ("get_tpu_quantization_config", None),
                ("update_vllm_config_for_qwix_quantization", None),
                ("maybe_create_mm_encoder_jit_manager", None),
            ):
                stack.enter_context(
                    patch.object(model_loader, target, return_value=value))
            stack.enter_context(
                patch.object(model_loader.envs, "MODEL_IMPL_TYPE", "flax_nnx"))
            with set_current_vllm_config(cfg):
                interface = model_loader.get_flax_model(
                    cfg, jax.random.key(0), mesh)

            logits = interface.compute_logits_fn(interface.state_leaves, rows)
            np.testing.assert_allclose(np.asarray(logits),
                                       expected,
                                       rtol=1e-6,
                                       atol=1e-6)
            outputs = interface.pooler_fn(logits, metadata)
            np.testing.assert_allclose(torch.stack(outputs).numpy(),
                                       expected,
                                       rtol=1e-6,
                                       atol=1e-6)

            runner = SimpleNamespace(
                state=interface.state,
                state_leaves=interface.state_leaves,
            )
            runner.state["score"]["weight"].value = weight + 1.0
            TPUModelRunner.refresh_state_leaves(runner)
            updated = interface.compute_logits_fn(runner.state_leaves, rows)
            np.testing.assert_allclose(
                np.asarray(updated),
                np.asarray(rows, dtype=np.float32) @ np.asarray(weight + 1.0),
                rtol=1e-6,
                atol=1e-6,
            )

    def test_pooler_mixed_activation_and_task_validation(self):
        weight = jnp.ones((4, 2), dtype=jnp.float32)
        param = nnx.Param(weight)
        param.set_metadata("_is_loaded", True)
        model = SimpleNamespace(score=SimpleNamespace(weight=param),
                                num_labels=2)
        head = ClassifierPoolerHead(
            head_dtype=torch.float32,
            activation=torch.nn.Softmax(dim=-1),
        )
        dispatch = DispatchPooler(
            {"classify": SequencePooler(LastPool(), head)})
        pooler = NativeSequenceClassificationPooler(model,
                                                    dispatch,
                                                    hidden_size=4)

        logits_dev = jnp.asarray([[0.0, 0.0], [1.0, 2.0]], dtype=jnp.float32)
        metadata = PoolingMetadata(
            prompt_lens=torch.tensor([1, 1], dtype=torch.int32),
            prompt_token_ids=None,
            prompt_token_ids_cpu=None,
            pooling_params=[
                PoolingParams(task="classify", use_activation=True),
                PoolingParams(task="classify", use_activation=False),
            ],
            pooling_states=[PoolingStates(), PoolingStates()],
        )
        outputs = pooler(logits_dev, metadata)
        np.testing.assert_allclose(outputs[0].numpy(),
                                   np.array([0.5, 0.5], dtype=np.float32),
                                   rtol=1e-6)
        np.testing.assert_allclose(outputs[1].numpy(),
                                   np.array([1.0, 2.0], dtype=np.float32),
                                   rtol=1e-6)

        bad_metadata = PoolingMetadata(
            prompt_lens=torch.tensor([1], dtype=torch.int32),
            prompt_token_ids=None,
            prompt_token_ids_cpu=None,
            pooling_params=[PoolingParams(task="embed")],
            pooling_states=[PoolingStates()],
        )
        with pytest.raises(RuntimeError, match="classify-only batches"):
            pooler(logits_dev[:1], bad_metadata)
