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

import dataclasses
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

from absl import app, flags

from tools.kernel.tuner.v1.common.kernel_tuner_base import (RunConfig,
                                                            TuningStatus)
from tools.kernel.tuner.v1.example_kernel_tuner import ExampleKernelTuner
from tools.kernel.tuner.v1.mla_kernel_tuner import MlaKernelTuner
from tools.kernel.tuner.v1.rpa_v3_kernel_tuner import RpaV3KernelTuner

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_DEBUG = flags.DEFINE_bool(
    'debug', False, 'If true, prints results after each case iteration.')
_RUN_LOCALLY = flags.DEFINE_bool(
    'run_locally', False,
    'If true, uses local storage instead of cloud storage.')
_KERNEL_TUNER_NAME = flags.DEFINE_string('kernel_tuner_name',
                                         'example_kernel_tuner',
                                         'Name of the kernel tuner to run.')
_CASE_SET_ID = flags.DEFINE_string('case_set_id', '',
                                   'The case set ID to use for this run.')
_RUN_ID = flags.DEFINE_string(
    'run_id', '',
    'The run ID to use for this run. If not specified, a timestamp-based ID will be generated.'
)
_CASE_SET_DESC = flags.DEFINE_string('case_set_desc', '',
                                     'Description of the case set.')
_GENERATE_BUILDKITE_PIPELINE = flags.DEFINE_bool(
    'generate_buildkite_pipeline', False,
    'If true, generates Buildkite pipeline YAML instead of running tuning jobs.'
)
_BEGIN_CASE_ID = flags.DEFINE_integer(
    'begin_case_id', None,
    'The begin case ID for tuning. Only used when --generate_buildkite_pipeline is false and --run_locally is false.'
)
_END_CASE_ID = flags.DEFINE_integer(
    'end_case_id', None,
    'The end case ID for tuning. Only used when --generate_buildkite_pipeline is false and --run_locally is false.'
)
_GCP_PROJECT_ID = flags.DEFINE_string(
    'gcp_project_id', 'cloud-tpu-inference-test',
    'The GCP project ID to use for Spanner. Only used when --run_locally is false.'
)
_SPANNER_INSTANCE_ID = flags.DEFINE_string(
    'spanner_instance_id', 'vllm-bm-inst',
    'The Spanner instance ID to use. Only used when --run_locally is false.')
_SPANNER_DATABASE_ID = flags.DEFINE_string(
    'spanner_database_id', 'tune-gmm',
    'The Spanner database ID to use. Only used when --run_locally is false.')
_WORKER_ID = flags.DEFINE_string('worker_id',
                                 os.getenv('HOST_NAME',
                                           'unknown'), 'The worker id')
_TPU_VERSION = flags.DEFINE_string(
    'tpu_version', '',
    'The TPU version to use for tuning. Supported values are "tpu6e" and "tpu7x".'
)

_TPU_CORES = flags.DEFINE_integer(
    'tpu_cores', 0,
    'The number of TPU cores to use for tuning. Default is 0. TPU tpu6e has 1 core per chip, TPU tpu7x has 2 cores per chip.'
)

_TPU_QUEUE_MULTI = flags.DEFINE_string(
    'tpu_queue_multi', '',
    'The TPU queue to use for tuning. This will be automatically determined based on the TPU version and cores if not specified. Supported values are "tpu_v6e_queue", "tpu_v6e_8_queue", "tpu_v7x_2_queue", "tpu_v7x_8_queue", and "tpu_v7x_16_queue".'
)

_JOB_PRIORITY = flags.DEFINE_integer(
    'job_priority', -10,
    'The priority to use for kernel tuning jobs. Higher priority jobs will be scheduled before lower priority ones. Default is -10, which is lower than typical user jobs to avoid impacting them.'
)

_MAX_EXECUTION_MINUTES = flags.DEFINE_integer(
    'max_execution_minutes', 20,
    'Only used when the kernel tuning job is scheduled through Buildkite. The maximum execution time in minutes for each kernel tuning job. If the job exceeds this time, it will save the job progresss, generate a new job to be scheduled by Buildkite and exit.'
)

# ----------------------------------------------------------------------------
# Smart-search flags (Phase 0 + 1). When --search_strategy=grid is set, the
# runner falls back to the legacy v1 measure_latency / Spanner / Buildkite
# orchestration. For any other strategy we drive the per-trial loop in
# ``run_smart_search`` below, using the tuner's smart-search hooks
# (``get_default_tuning_key``, ``get_search_space``, ``get_oracle``, etc.).
# ----------------------------------------------------------------------------
_SEARCH_STRATEGY = flags.DEFINE_enum(
    'search_strategy', 'grid', ['grid', 'tpe', 'evolutionary'],
    'Search strategy for the new smart-search loop. `grid` falls back to the '
    'legacy v1 measure_latency path (back-compat).')
_TRIAL_BUDGET = flags.DEFINE_integer(
    'trial_budget', 200,
    'Max number of candidates to evaluate in the smart-search loop. Ignored '
    'when --search_strategy=grid.')
_VERIFIER_MODE = flags.DEFINE_enum(
    'verifier_mode', 'fast', ['off', 'fast', 'strict'],
    'Correctness gating for each candidate. off=skip; fast=numerics only; '
    'strict=numerics + anti-cheat with a second seeded input set.')
_COST_MODEL_PREFILTER = flags.DEFINE_bool(
    'cost_model_prefilter', True,
    'If true, skip candidates that the tuner-supplied cost model marks '
    'infeasible before issuing a real TPU run.')
_INTERPRET_CHECK = flags.DEFINE_bool(
    'interpret_check', False,
    'If true, run an off-TPU pltpu.InterpretParams correctness pre-check on '
    'each candidate before the real-TPU run.')
_FINAL_EVAL = flags.DEFINE_bool(
    'final_eval', False,
    'If true, run lm-eval-harness on the winning config as an outer gate.')
_FINAL_EVAL_MODEL_ARGS = flags.DEFINE_string(
    'final_eval_model_args', '',
    'lm-eval --model_args payload for the final-eval gate '
    '(e.g. "pretrained=Qwen/Qwen3-0.6B").')
_FINAL_EVAL_TASKS = flags.DEFINE_list(
    'final_eval_tasks', ['gsm8k'],
    'Tasks to run via lm-eval for the final-eval gate.')
_RESULTS_PATH = flags.DEFINE_string(
    'results_path', '',
    'If set, write the smart-search trial log + winner JSON to this path. '
    'Default empty = "/tmp/kernel_tuning/{case_set_id}_{run_id}.jsonl".')
_TIMING_ITERS = flags.DEFINE_integer(
    'timing_iters', 10,
    'Number of timed iterations per candidate in the smart-search loop.')
_SECOND_SEED = flags.DEFINE_integer(
    'second_seed', 7919,
    'Seed offset for the second input set used in strict verifier mode.')

# Note: For simplicity, we are directly referencing the kernel tuner class
# here. In the future, we can consider a more flexible plugin-based system
# if we have more kernel tuners. For example, we can define an interface for
# kernel tuners and dynamically load kernel tuner classes based on the
# --kernel_tuner_name flag. This would allow us to add new kernel tuners
# without modifying this runner code. For now, after we implement more kernel
# tuners, we can simply add them to the KERNEL_TUNER_REGISTRY dictionary below.
KERNEL_TUNER_REGISTRY = {
    'example_kernel_tuner': ExampleKernelTuner,
    'rpa_v3_kernel_tuner': RpaV3KernelTuner,
    'mla_kernel_tuner': MlaKernelTuner,
}


def _resolve_results_path(case_set_id: str, run_id: str) -> Path:
    if _RESULTS_PATH.value:
        p = Path(_RESULTS_PATH.value)
    else:
        p = Path("/tmp/kernel_tuning") / f"{case_set_id}_{run_id}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _dataclass_to_jsonable(x: Any) -> Any:
    """Best-effort recursive conversion to a JSON-serializable structure."""
    if x is None or isinstance(x, (bool, int, float, str)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return str(x)
        return x
    if isinstance(x, (list, tuple)):
        return [_dataclass_to_jsonable(e) for e in x]
    if isinstance(x, dict):
        return {str(k): _dataclass_to_jsonable(v) for k, v in x.items()}
    if hasattr(x, "__dict__"):
        return {k: _dataclass_to_jsonable(v) for k, v in vars(x).items()}
    return str(x)


@dataclasses.dataclass
class SmartSearchConfig:
    """Plain-Python config for ``run_smart_search``.

    Decoupled from absl flags so the loop is unit-testable. ``main()``
    constructs one of these from the runner's flags before calling.
    """
    search_strategy: str = 'tpe'
    trial_budget: int = 200
    verifier_mode: str = 'fast'  # 'off' | 'fast' | 'strict'
    cost_model_prefilter: bool = True
    interpret_check: bool = False
    timing_iters: int = 10
    second_seed: int = 7919
    final_eval: bool = False
    final_eval_model_args: str = ''
    final_eval_tasks: tuple[str, ...] = ('gsm8k', )
    anti_cheat_input_skip_keys: tuple[str, ...] = ('kv_cache', )


def run_smart_search(
    kernel_tuner,
    config: SmartSearchConfig | None = None,
) -> dict[str, Any]:
    """Smart-search driver. Returns a JSON-serializable summary dict.

    Loop semantics per trial:
        1. Strategy proposes ``params``.
        2. (optional) Cost-model pre-filter — math.inf on infeasible.
        3. (optional) Off-TPU ``InterpretParams`` correctness preview.
        4. Real-TPU run via ``run_with_outputs``; OOM / errors → math.inf.
        5. Numerics + anti-cheat verifier against the oracle.
        6. (optional, strict mode) Second-seed cross-trial independence.
        7. Score = ``avg_latency_ns`` on success, ``math.inf`` on any failure.

    Failure modes — cost-model skip, interpret fail, OOM, kernel error,
    numerics fail, anti-cheat fail, cross-trial fail — all produce
    ``math.inf`` so the strategy treats them uniformly as "no info."
    """
    from tools.kernel.tuner.v1.search import SEARCH_STRATEGY_REGISTRY
    from tools.kernel.tuner.v1.verifier.anti_cheat import (
        AntiCheatGuard, cross_trial_independence)
    from tools.kernel.tuner.v1.verifier.interpret_check import \
        interpret_params_available

    if config is None:
        config = SmartSearchConfig()

    space = kernel_tuner.get_search_space()
    if space is None:
        raise RuntimeError(
            f"{type(kernel_tuner).__name__}.get_search_space() returned None. "
            f"The selected kernel tuner does not support smart-search. "
            f"Use --search_strategy=grid to fall back to the legacy v1 path.")
    tuning_key = kernel_tuner.get_default_tuning_key()
    if tuning_key is None:
        raise RuntimeError(
            f"{type(kernel_tuner).__name__}.get_default_tuning_key() returned "
            f"None. Smart-search needs a concrete TuningKey to explore.")
    oracle = kernel_tuner.get_oracle()
    cost_model = kernel_tuner.get_cost_model()
    inputs = kernel_tuner.generate_inputs(tuning_key)
    second_inputs = None
    if config.verifier_mode == 'strict':
        second_inputs = _try_second_seed_inputs(kernel_tuner, tuning_key,
                                                config.second_seed)

    strategy_cls = SEARCH_STRATEGY_REGISTRY[config.search_strategy]
    strategy = strategy_cls(space=space, trial_budget=config.trial_budget)

    guard = AntiCheatGuard(input_skip_keys=config.anti_cheat_input_skip_keys)

    results_log: list[dict[str, Any]] = []
    started_at = time.time()
    while not strategy.done():
        trial_idx = strategy.trials_observed
        params = strategy.suggest()
        record: dict[str, Any] = {
            "trial": trial_idx,
            "params": params,
            "strategy": config.search_strategy,
        }

        if config.cost_model_prefilter and cost_model is not None:
            feasible, reason = cost_model.is_feasible(tuning_key, params)
            if not feasible:
                record.update({
                    "status": "COST_MODEL_SKIP",
                    "reason": reason,
                    "score": "inf",
                })
                results_log.append(record)
                strategy.observe(params, math.inf, {"reason": reason})
                continue

        if (config.interpret_check and oracle is not None
                and interpret_params_available()):
            try:
                rep = _run_interpret_check(kernel_tuner, tuning_key, params,
                                           inputs, oracle)
            except Exception as err:  # pragma: no cover - kernel-specific
                logger.warning("interpret_check raised: %s", err)
                rep = None
            if rep is not None and not rep.passed:
                record.update({
                    "status": "INTERPRET_FAIL",
                    "reason": rep.reason,
                    "score": "inf",
                })
                results_log.append(record)
                strategy.observe(params, math.inf, {"reason": rep.reason})
                continue

        result = kernel_tuner.run_with_outputs(tuning_key,
                                               params,
                                               iters=config.timing_iters)
        record["latency_ns_p50"] = (result.bench.p50_ns
                                    if result.bench is not None else None)
        record["latency_ns_p95"] = (result.bench.p95_ns
                                    if result.bench is not None else None)
        record["latency_ns_mean"] = result.avg_latency_ns
        if result.status != TuningStatus.SUCCESS:
            record.update({
                "status": result.status.value,
                "score": "inf",
                "aux": result.aux,
            })
            results_log.append(record)
            strategy.observe(params, math.inf, {"status": result.status.value})
            continue

        if config.verifier_mode != 'off':
            verify_report = kernel_tuner.verify(tuning_key,
                                                params,
                                                result.output,
                                                inputs=inputs)
            record["numerics"] = {
                "passed": verify_report.passed,
                "max_abs_diff": verify_report.max_abs_diff,
                "cosine": verify_report.cosine,
                "nan_count": verify_report.nan_count,
                "inf_count": verify_report.inf_count,
                "reason": verify_report.reason,
            }
            if not verify_report.passed:
                record.update({
                    "status": "NUMERICS_FAIL",
                    "score": "inf",
                })
                results_log.append(record)
                strategy.observe(params, math.inf,
                                 {"reason": verify_report.reason})
                continue
            primary = (result.output[0] if isinstance(
                result.output, (tuple, list)) else result.output)
            ac = guard.inspect(primary, inputs)
            if not ac.passed:
                record.update({
                    "status": "ANTI_CHEAT_FAIL",
                    "score": "inf",
                    "anti_cheat_reason": ac.reason,
                })
                results_log.append(record)
                strategy.observe(params, math.inf, {"reason": ac.reason})
                continue

        if config.verifier_mode == 'strict' and second_inputs is not None:
            second_result = kernel_tuner.run_with_outputs(
                tuning_key, params, iters=max(2, config.timing_iters // 2))
            if second_result.status == TuningStatus.SUCCESS:
                primary_a = (result.output[0] if isinstance(
                    result.output, (tuple, list)) else result.output)
                primary_b = (second_result.output[0] if isinstance(
                    second_result.output,
                    (tuple, list)) else second_result.output)
                indep = cross_trial_independence(primary_a, primary_b)
                record["cross_trial"] = {
                    "passed": indep.passed,
                    "reason": indep.reason,
                }
                if not indep.passed:
                    record.update({
                        "status": "CROSS_TRIAL_FAIL",
                        "score": "inf",
                    })
                    results_log.append(record)
                    strategy.observe(params, math.inf,
                                     {"reason": indep.reason})
                    continue

        score = result.avg_latency_ns
        record.update({"status": "SUCCESS", "score": score})
        results_log.append(record)
        strategy.observe(params, score, {"bench": result.bench})

    best_params, best_score = strategy.best()
    summary = {
        "kernel_tuner_name": kernel_tuner.tuner_config.kernel_tuner_name,
        "search_strategy": config.search_strategy,
        "trial_budget": config.trial_budget,
        "trials_observed": strategy.trials_observed,
        "wall_time_sec": time.time() - started_at,
        "tuning_key": _dataclass_to_jsonable(tuning_key),
        "best_params": best_params,
        "best_score_ns":
        (None if not math.isfinite(best_score) else best_score),
        "verifier_mode": config.verifier_mode,
        "interpret_check": config.interpret_check,
        "cost_model_prefilter": config.cost_model_prefilter,
        "trials": [_dataclass_to_jsonable(r) for r in results_log],
    }

    if config.final_eval and best_params is not None:
        summary["final_eval"] = _run_final_eval(config)

    return summary


def _try_second_seed_inputs(kernel_tuner, tuning_key, seed: int):
    """Best-effort fresh inputs via a different numpy RNG seed.

    The existing ``generate_inputs`` doesn't accept a seed, so we swap the
    numpy RNG state around the call. Matches how legacy tuners draw inputs
    (``np.random.rand``).
    """
    import numpy as np
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        prev_key = kernel_tuner._tuning_key
        prev_cache = kernel_tuner._kernel_inputs_cache
        kernel_tuner._tuning_key = None
        kernel_tuner._kernel_inputs_cache = {}
        try:
            second = kernel_tuner.generate_inputs(tuning_key)
        finally:
            kernel_tuner._tuning_key = prev_key
            kernel_tuner._kernel_inputs_cache = prev_cache
        return second
    except Exception as err:  # pragma: no cover - kernel-specific
        logger.warning(
            "second-seed inputs unavailable (%s); "
            "strict mode will skip the cross-trial check", err)
        return None
    finally:
        np.random.set_state(state)


def _run_interpret_check(kernel_tuner, tuning_key, params, inputs, oracle):
    """Run a candidate under pltpu.InterpretParams and verify vs oracle.

    Best-effort: not every kernel will support interpret mode (custom DMAs,
    semaphores). Returns ``None`` if the tuner doesn't expose a hook.
    """
    from tools.kernel.tuner.v1.verifier.interpret_check import interpret_check
    fn = kernel_tuner.build_kernel_fn(tuning_key, params, inputs)
    if fn is None:
        return None
    # The tuner's build_kernel_fn returns a closure over real-TPU kernel; we
    # can only run it under interpret mode if the underlying kernel supports
    # it via a passthrough flag. For RPA v3 this is handled by recompiling
    # with pl.pallas_call(..., interpret=pltpu.InterpretParams()). At this
    # tier we just run it as-is; if the kernel doesn't support emulation it
    # falls through to a TPU run inside fn().
    return interpret_check(fn, oracle, inputs=inputs)


def _run_final_eval(config: SmartSearchConfig) -> dict[str, Any]:
    from tools.kernel.tuner.v1.verifier.lm_eval_gate import (lm_eval_available,
                                                             run_lm_eval)
    if not lm_eval_available():
        return {"skipped": True, "reason": "lm_eval CLI not on $PATH"}
    if not config.final_eval_model_args:
        return {
            "skipped": True,
            "reason": "final_eval_model_args not provided",
        }
    try:
        results = run_lm_eval(
            model_args=config.final_eval_model_args,
            tasks=list(config.final_eval_tasks),
            baselines={},
        )
        return {
            "skipped": False,
            "tasks": [_dataclass_to_jsonable(r) for r in results],
        }
    except Exception as err:
        return {"skipped": True, "reason": f"lm-eval failed: {err}"}


# Keep in sync with the logic in bootstrap_kernel_tuning.sh:set_jax_envs
def get_tpu_queue_by_version_and_cores(tpu_version, tpu_cores,
                                       tpu_queue_multi):
    """Gets and validates TPU queue based on version and core configuration."""
    _queue_by_version_and_cores = {
        ('tpu6e', 1): 'tpu_v6e_queue',
        ('tpu6e', 8): 'tpu_v6e_8_queue',
        ('tpu7x', 2): 'tpu_v7x_2_queue',
        ('tpu7x', 8): 'tpu_v7x_8_queue',
        ('tpu7x', 16): 'tpu_v7x_16_queue',
    }
    assert (
        tpu_version, tpu_cores
    ) in _queue_by_version_and_cores, f'Unsupported combination of TPU version {tpu_version} and cores {tpu_cores}. Supported combinations are: {list(_queue_by_version_and_cores.keys())}'
    expected_queue = _queue_by_version_and_cores[(tpu_version, tpu_cores)]
    assert not tpu_queue_multi or tpu_queue_multi == expected_queue, f'Inconsistent TPU queue {tpu_queue_multi} for version {tpu_version} and cores {tpu_cores}. Expected queue is {expected_queue}. Please check your flags.'
    return tpu_queue_multi or expected_queue


def main(argv):
    del argv  # Unused.

    # env validation
    if _KERNEL_TUNER_NAME.value not in KERNEL_TUNER_REGISTRY:
        raise ValueError(
            f'Kernel tuner {_KERNEL_TUNER_NAME.value} is not registered. Available tuners: {list(KERNEL_TUNER_REGISTRY.keys())}'
        )

    case_set_id = _CASE_SET_ID.value
    run_id = _RUN_ID.value
    case_set_desc = _CASE_SET_DESC.value
    assert case_set_id, 'case_set_id is required. Please specify it through --case_set_id flag.'
    assert run_id, 'run_id is required. Please specify it through --run_id flag.'
    logger.info(
        f'Using case_set_id: {case_set_id}, run_id: {run_id}, case_set_desc: {case_set_desc} for this tuning run.'
    )

    tpu_version = _TPU_VERSION.value
    tpu_cores = _TPU_CORES.value
    tpu_queue_multi = _TPU_QUEUE_MULTI.value

    tpu_queue_multi = get_tpu_queue_by_version_and_cores(
        tpu_version, tpu_cores, tpu_queue_multi)

    run_config = RunConfig(case_set_id=case_set_id,
                           run_id=run_id,
                           case_set_desc=case_set_desc,
                           tpu_version=tpu_version,
                           tpu_cores=tpu_cores,
                           tpu_queue_multi=tpu_queue_multi,
                           run_locally=_RUN_LOCALLY.value,
                           job_priority=_JOB_PRIORITY.value,
                           max_execution_minutes=_MAX_EXECUTION_MINUTES.value)
    kernel_tuner_cls = KERNEL_TUNER_REGISTRY.get(_KERNEL_TUNER_NAME.value)
    kernel_tuner = kernel_tuner_cls(run_config=run_config)

    # Smart-search routes (any strategy other than grid) bypass the legacy
    # case-set / Spanner / Buildkite pipeline entirely.
    if _SEARCH_STRATEGY.value != 'grid':
        cfg = SmartSearchConfig(
            search_strategy=_SEARCH_STRATEGY.value,
            trial_budget=_TRIAL_BUDGET.value,
            verifier_mode=_VERIFIER_MODE.value,
            cost_model_prefilter=_COST_MODEL_PREFILTER.value,
            interpret_check=_INTERPRET_CHECK.value,
            timing_iters=_TIMING_ITERS.value,
            second_seed=_SECOND_SEED.value,
            final_eval=_FINAL_EVAL.value,
            final_eval_model_args=_FINAL_EVAL_MODEL_ARGS.value,
            final_eval_tasks=tuple(_FINAL_EVAL_TASKS.value),
        )
        logger.info("Running smart-search with strategy=%s budget=%d",
                    cfg.search_strategy, cfg.trial_budget)
        summary = run_smart_search(kernel_tuner, cfg)
        out_path = _resolve_results_path(case_set_id, run_id)
        with out_path.open("w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Wrote smart-search results to %s", out_path)
        if summary["best_params"] is not None:
            logger.info("Best params: %s (score=%.2fns)",
                        summary["best_params"], summary["best_score_ns"] or -1)
        else:
            logger.warning(
                "Smart-search produced no verified winner (all candidates "
                "failed cost-model / numerics / kernel run). Review %s.",
                out_path)
        return

    if kernel_tuner.run_config.run_locally:
        logger.info(
            'Running in locally mode. Skipping Buildkite pipeline generation and running tuning jobs directly.'
        )
        buckets = kernel_tuner._generate_tuning_jobs()
        for bucket in buckets:
            begin_case_id, end_case_id = bucket
            kernel_tuner.measure_latency(begin_case_id, end_case_id)
    else:
        logger.info(
            'Running in cloud mode. Generating Buildkite pipeline YAML or running tuning jobs directly.'
        )
        if _GENERATE_BUILDKITE_PIPELINE.value:
            logger.info(
                'Generating Buildkite pipeline YAML. No tuning jobs will be run.'
            )

            kernel_tuner.generate_buildkite_pipeline()
        else:
            begin_case_id = _BEGIN_CASE_ID.value
            end_case_id = _END_CASE_ID.value
            logger.debug(
                'Running tuning jobs directly. Skipping Buildkite pipeline generation. Bucket [%d, %d)',
                begin_case_id, end_case_id)
            kernel_tuner.measure_latency(begin_case_id=begin_case_id,
                                         end_case_id=end_case_id)


if __name__ == '__main__':
    app.run(main)
