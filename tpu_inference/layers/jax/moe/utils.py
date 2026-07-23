# Copyright 2025 Google LLC
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

import math

from jax.sharding import Mesh

from tpu_inference import envs
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.jax.layers import FlaxUtils
from tpu_inference.logger import init_logger

logger = init_logger(__name__)
modeling_flax_utils = FlaxUtils()


def get_expert_parallelism(expert_axis_name: str, mesh: Mesh) -> int:
    """
    Returns the expert parallelism number from the mesh.

    Args:
        expert_axis_name: The expert axis name.
        mesh: The mesh.

    Returns:
        The expert parallelism number.
    """
    if expert_axis_name is None:
        return 1
    else:
        if isinstance(expert_axis_name, str):
            return mesh.shape[expert_axis_name]
        else:
            return math.prod(mesh.shape[axis] for axis in expert_axis_name)


# TODO(#3041): Unify with torchax `select_moe_backend_from_fused_moe_config()`
def select_moe_backend(use_ep: bool) -> MoEBackend:
    """
    Selects the MoE backend for the JAX path.

    Args:
        use_ep: Whether to use expert parallelism.

    Returns:
        The selected MoE backend.
    """
    if envs.USE_MOE_EP_KERNEL:
        if use_ep:
            logger.info_once("[MoE]: Using fused MoE EP kernel")
            return MoEBackend.FUSED_MOE

    if use_ep:
        logger.warning_once(
            "USE_MOE_EP_KERNEL=1 but expert parallelism is not "
            "enabled. Falling back to gmm implementation.")
        logger.info_once("[MoE]: Using GMM EP kernel")
        return MoEBackend.GMM_EP

    if envs.USE_DENSE_MOE:
        logger.info_once("[MoE]: Using DENSE_MOE")
        logger.warning_once(
            "[MoE]: DENSE_MOE is naive and not intended for production.")
        return MoEBackend.DENSE_MAT

    logger.info_once("[MoE]: Using GMM TP kernel")
    return MoEBackend.GMM_TP
