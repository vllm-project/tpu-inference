from vllm.v1.executor.ray_distributed_executor import RayDistributedExecutor as RayDistributedExecutorV1 

from tpu_commons.logger import init_logger


logger = init_logger(__name__)

class TpuDistributedExecutor(RayDistributedExecutorV1):
    """Ray-based distributed executor"""

    def _init_executor(self) -> None:
        logger.info("Initializing TPU distributed executor")
        super()._init_executor()