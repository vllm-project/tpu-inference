import os

from tpu_commons import tpu_info as ti
from tpu_commons.logger import init_logger
import jax 

logger = init_logger(__name__)

# The warning message to be displayed to the user
PROD_WARNING = (
    "🚨  CAUTION: You are using 'tpu_commons' , which is experimental "
    "and NOT intended for production use yet. Please see the README for more details."
)

logger.warn(PROD_WARNING)


def detect_platform() -> str:
    if "proxy" in os.environ.get('JAX_PLATFORMS', '').lower():
        return "pathways"
    return jax.lib.xla_bridge.get_backend().platform # 'tpu' or 'cpu'
        
_platform = detect_platform()

if _platform == "tpu":
    try:
        logger.info("Running vLLM on TPU.")
        logger.info(f"TPU info: node_name={ti.get_node_name()} | "
                    f"tpu_type={ti.get_tpu_type()} | "
                    f"worker_id={ti.get_node_worker_id()} | "
                    f"num_chips={ti.get_num_chips()} | "
                    f"num_cores_per_chip={ti.get_num_cores_per_chip()}")
    except Exception as e:
        logger.error(f"Error occurred while logging TPU info: {e}")
elif _platform == "cpu":
    logger.info("Running vLLM on CPU.")
elif _platform== "pathways":
    logger.info("Running vLLM on TPU via Pathways proxy.")
    import pathwaysutils
    pathwaysutils.initialize()
    logger.info("Running vLLM with Pathways. "
                "Module pathwaysutils is imported.")
else:
    logger.error(f"Unsupported platform: {_platform}")