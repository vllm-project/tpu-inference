import os

# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import tpu_inference.env_override  # noqa: F401
from tpu_inference import tpu_info as ti
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

if "proxy" in os.environ.get('JAX_PLATFORMS', '').lower():
    logger.info("Running vLLM on TPU via Pathways proxy.")
    # Must run pathwaysutils.initialize() before any JAX operations
    try:
        import pathwaysutils
        pathwaysutils.initialize()
        logger.info("Module pathwaysutils is imported.")
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
