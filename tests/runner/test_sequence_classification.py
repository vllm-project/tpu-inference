# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh

from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.runner.sequence_classification import \
    pool_sequence_classification


def _make_single_device_mesh() -> Mesh:
    devices = np.asarray(jax.local_devices()[:1], dtype=object).reshape(
        (1, ) * len(MESH_AXIS_NAMES))
    return Mesh(devices, MESH_AXIS_NAMES)


def test_pool_sequence_classification_chunked_completion():
    mesh = _make_single_device_mesh()
    hidden = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                         dtype=jnp.bfloat16)
    logits_indices = jnp.asarray([0, 2], dtype=jnp.int32)

    batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["req-done", "req-partial"],
        req_id_to_index={
            "req-done": 0,
            "req-partial": 1
        },
        num_computed_tokens_cpu=np.array([4, 2], dtype=np.int32),
        num_prompt_tokens=np.array([8, 8], dtype=np.int32),
        get_pooling_metadata=lambda: "meta",
    )
    runner = SimpleNamespace(
        input_batch=batch,
        mesh=mesh,
        vllm_config=SimpleNamespace(sharding_config=SimpleNamespace(
            prefill_cp_size=1)),
        state_leaves="leaves",
        _select_from_array_fn=lambda arr, idx, m, pcp: arr[idx],
        compute_logits_fn=lambda leaves, rows: rows.astype(jnp.float32),
        pooler_fn=lambda logits, meta: [
            torch.tensor([1.0, 2.0]),
            torch.tensor([5.0, 6.0]),
        ],
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={
        "req-done": 4,
        "req-partial": 3
    })

    out = pool_sequence_classification(
        runner,
        hidden,
        logits_indices,
        logits_indices_selector=None,
        scheduler_output=scheduler_output,
    )
    assert len(out.pooler_output) == 2
    assert torch.equal(out.pooler_output[0], torch.tensor([1.0, 2.0]))
    assert out.pooler_output[1] is None

    with pytest.raises(RuntimeError, match="without in-process DP reordering"):
        pool_sequence_classification(
            runner,
            hidden,
            logits_indices,
            logits_indices_selector=np.array([0, 1]),
            scheduler_output=scheduler_output,
        )
