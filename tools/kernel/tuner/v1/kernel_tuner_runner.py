import logging
from absl import app
from absl import flags

from tools.kernel.tuner.v1.storage_management.spanner_database_manager import SpannerStorageManager
from tools.kernel.tuner.v1.storage_management.local_db_manager import LocalDbManager
from tools.kernel.tuner.v1.test_kernel_tuner import TestKernelTuner
from tools.kernel.tuner.v1.common.utils import get_host_ip

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_DEBUG = flags.DEFINE_bool(
    'debug', False, 'If true, prints results after each case iteration.')
_RUN_LOCALLY = flags.DEFINE_bool('run_locally', False, 'If true, uses local storage instead of cloud storage.')
_KERNEL_TUNER_NAME = flags.DEFINE_string('kernel_tuner_name', 'test_kernel_tuner', 'Name of the kernel tuner to run.')    
_CASE_SET_ID = flags.DEFINE_string('case_set_id', None, 'The case set ID to use for this run.')
_RUN_ID = flags.DEFINE_string('run_id', None, 'The run ID to use for this run. If not specified, a timestamp-based ID will be generated.')
_CASE_SET_DESC = flags.DEFINE_string('case_set_desc', '', 'Description of the case set.')
_GENERATE_BUILDKITE_PIPELINE = flags.DEFINE_bool('generate_buildkite_pipeline', False, 'If true, generates Buildkite pipeline YAML instead of running tuning jobs.')
_BEGIN_CASE_ID = flags.DEFINE_integer('begin_case_id', None, 'The begin case ID for tuning. Only used when --generate_buildkite_pipeline is false and --run_locally is false.')
_END_CASE_ID = flags.DEFINE_integer('end_case_id', None, 'The end case ID for tuning. Only used when --generate_buildkite_pipeline is false and --run_locally is false.')
_WORKER_ID = flags.DEFINE_string('worker_id', get_host_ip(), 'The worker id')


KERNEL_TUNER_REGISTRY = {
    'test_kernel_tuner': TestKernelTuner,
}



def main(argv):
    del argv  # Unused.

    # Initialize storage manager
    if _RUN_LOCALLY.value:
        storage_manager = LocalDbManager()
    else:
        storage_manager = SpannerStorageManager()

    # Initialize kernel tuner
    kernel_tuner_cls = KERNEL_TUNER_REGISTRY.get(_KERNEL_TUNER_NAME.value, None)

    if kernel_tuner_cls is None:
        raise ValueError(f'Unknown kernel tuner: {_KERNEL_TUNER_NAME.value}')
    kernel_tuner = kernel_tuner_cls(storage_manager)


    case_set_id = _CASE_SET_ID.value
    run_id = _RUN_ID.value
    case_set_desc = _CASE_SET_DESC.value
    if _RUN_LOCALLY.value:
        logger.info('Running in locally mode. Skipping Buildkite pipeline generation and running tuning jobs directly.')
        buckets = kernel_tuner._generate_tuning_jobs(case_set_id, desc=case_set_desc)
        for bucket in buckets:
            begin_case_id, end_case_id = bucket
            logger.info(f'Bucket: [{begin_case_id}, {end_case_id})')
            kernel_tuner.measure_latency(case_set_id, run_id, begin_case_id, end_case_id)
    else:
        logger.info('Running in cloud mode. Generating Buildkite pipeline YAML and printing to stdout.')
        generate_pipeline_yaml = _GENERATE_BUILDKITE_PIPELINE.value
        if generate_pipeline_yaml:
            pipeline_yaml = kernel_tuner.generate_buildkite_pipeline_yaml(case_set_id, desc=case_set_desc)
            print(pipeline_yaml)
        else:
            begin_case_id = _BEGIN_CASE_ID.value
            end_case_id = _END_CASE_ID.value
            kernel_tuner.measure_latency(case_set_id, run_id=run_id, begin_case_id=begin_case_id, end_case_id=end_case_id)

        
if __name__ == '__main__':
    app.run(main)