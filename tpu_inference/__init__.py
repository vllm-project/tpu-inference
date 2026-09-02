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

# Engine-first XLA load. Each Raiden module statically links its own XLA and its
# own copy of `xla/pjrt/proto/execute_options.proto`; whichever is dlopened first
# wins the protobuf registry and a later one aborts the process. Must precede
# env_override, which does `import vllm` and brings up jaxlib's XLA. All modules,
# not just the engine: the FFI ones are imported at module scope by tunix's
# synchronizer, so they would otherwise land mid-run. The wheel is distributed
# as `tpu_raiden_jax` but installs `tpu_sync`.
for _mod in ("_tpu_raiden_jax", "weight_synchronizer_ffi",
             "_kv_cache_manager_ffi"):
    try:
        __import__(f"tpu_sync.frameworks.jax.{_mod}")
    except ImportError:
        # Raiden absent (e.g. the benchmark client), or an older build that
        # does not ship this module. Only EngineCore workers need the load.
        pass

# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import tpu_inference.env_override  # noqa: F401,E402
from tpu_inference import envs  # noqa: E402
from tpu_inference import tpu_info as ti  # noqa: E402
from tpu_inference.logger import init_logger  # noqa: E402

logger = init_logger(__name__)

if "proxy" in envs.JAX_PLATFORMS:
    logger.info("Running vLLM on TPU via Pathways proxy.")
    # Must run pathwaysutils.initialize() before any JAX operations
    try:
        import traceback

        import pathwaysutils
        import vllm
        from vllm.platforms import (resolve_current_platform_cls_qualname,
                                    resolve_obj_by_qualname)
        pathwaysutils.initialize()
        logger.info("Module pathwaysutils is imported.")

        # Pathways requires eager resolution of vllm.current_platform instead of
        # lazy resolution in the normal code path. Since this part involves
        # global topology discovery across multiple hosts, the platform
        # resolution must happen before other components are loaded.
        logger.info("Eagerly resolving vLLM current_platform for Pathways.")
        platform_cls_qualname = resolve_current_platform_cls_qualname()
        resolved_platform_instance = resolve_obj_by_qualname(
            platform_cls_qualname)()
        vllm.platforms._current_platform = resolved_platform_instance
        vllm.platforms._init_trace = "".join(traceback.format_stack())
        logger.info(
            f"vLLM platform resolved to: {resolved_platform_instance.__class__.__name__}"
        )

    except Exception as e:
        logger.error(
            f"Error occurred while importing pathwaysutils or logging TPU info: {e}"
        )
else:
    # Either running on TPU or CPU
    try:
        logger.info(f"TPU info: node_name={ti.get_node_name()} | "
                    f"tpu_type={ti.get_tpu_type()} | "
                    f"worker_id={ti.get_node_worker_id()} | "
                    f"num_chips={ti.get_num_chips()} | "
                    f"num_cores_per_chip={ti.get_num_cores_per_chip()}")
    except Exception as e:
        logger.error(
            f"Error occurred while logging TPU info: {e}. Are you running on CPU?"
        )
