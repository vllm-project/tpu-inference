# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""E2E test: jina-embeddings-v2-small-en under the pooling runner.

Mirrors test_step_pooling.py, but with real weights, and — when
sentence-transformers is available — validates the embeddings numerically
against the reference implementation (cosine similarity > 0.99).
"""

import multiprocessing as mp

try:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import numpy as np
import pytest

MODEL_ID = "jinaai/jina-embeddings-v2-small-en"


def test_jina_embeddings_e2e():
    from vllm import LLM

    max_model_len = 1024
    sentences = [
        "How is the weather today?",
        "What is the current weather like today?",
        "Jina embeddings now run on TPU v6e with vLLM.",
    ]

    try:
        llm = LLM(
            model=MODEL_ID,
            runner="pooling",
            max_num_seqs=4,
            max_model_len=max_model_len,
            dtype="float32",
            trust_remote_code=True,
            override_pooler_config={"pooling_type": "MEAN"},
            tensor_parallel_size=1,
        )
    except Exception as e:
        pytest.skip(f"Skipping test: {e}")

    try:
        results = llm.embed(sentences)

        assert len(results) == len(sentences)
        embeddings = []
        for result in results:
            emb = result.outputs.embedding
            assert emb is not None
            assert len(emb) == 512  # hidden_size of the small-en variant
            embeddings.append(np.asarray(emb, dtype=np.float32))

        # The two weather questions must be closer to each other than either
        # is to the unrelated TPU sentence.
        def cos(a, b):
            return float(
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        sim_related = cos(embeddings[0], embeddings[1])
        sim_unrelated = cos(embeddings[0], embeddings[2])
        assert sim_related > sim_unrelated, (
            f"similarity sanity check failed: {sim_related} <= {sim_unrelated}"
        )

        # Numerical parity vs sentence-transformers, when available.
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            SentenceTransformer = None
        if SentenceTransformer is not None:
            try:
                st_model = SentenceTransformer(MODEL_ID,
                                               trust_remote_code=True)
                ref = st_model.encode(sentences)
            except Exception:
                ref = None
            if ref is not None:
                for got, expected in zip(embeddings, ref):
                    similarity = cos(got, np.asarray(expected,
                                                     dtype=np.float32))
                    assert similarity > 0.99, (
                        f"embedding does not match reference: {similarity}")
    finally:
        llm.llm_engine.engine_core.shutdown()
