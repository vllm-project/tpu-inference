import os

from tpu_commons import tpu_info as ti
from tpu_commons.logger import init_logger

logger = init_logger(__name__)

# The warning message to be displayed to the user
PROD_WARNING = (
    "🚨  CAUTION: You are using 'tpu_commons' , which is experimental "
    "and NOT intended for production use yet. Please see the README for more details."
)

logger.warn(PROD_WARNING)

logger.info(f"TPU info: node_name={ti.get_node_name()} | "
            f"tpu_type={ti.get_tpu_type()} | "
            f"worker_id={ti.get_node_worker_id()} | "
            f"num_chips={ti.get_num_chips()} | "
            f"num_cores_per_chip={ti.get_num_cores_per_chip()}")

# Must run pathwaysutils.initialize() before any JAX operations.
if "proxy" in os.environ.get('JAX_PLATFORMS', '').lower():
    import pathwaysutils
    pathwaysutils.initialize()
    logger.info("Running vLLM with Pathways. "
                "Module pathwaysutils is imported.")
else:
    logger.info("Running vLLM without Pathways. "
                "Module pathwaysutils is not imported.")
