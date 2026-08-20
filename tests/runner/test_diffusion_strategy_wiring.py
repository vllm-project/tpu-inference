# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import pathlib

RUNNER = (pathlib.Path(__file__).resolve().parents[2] / "tpu_inference" /
          "runner" / "tpu_runner.py")
STRATEGY = (pathlib.Path(__file__).resolve().parents[2] / "tpu_inference" /
            "runner" / "diffusion" / "strategy.py")
PROGRAM = STRATEGY.parent / "program.py"
DECODE_LOOP = STRATEGY.parent.parent / "decode_loop.py"


def _method(name):
    module = ast.parse(RUNNER.read_text())
    runner_class = next(
        node for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "TPUModelRunner")
    for node in runner_class.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Method {name!r} not found")


def _strategy_method(name):
    module = ast.parse(STRATEGY.read_text())
    strategy_class = next(node for node in module.body
                          if isinstance(node, ast.ClassDef)
                          and node.name == "BlockDiffusionStrategy")
    for node in strategy_class.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Strategy method {name!r} not found")


def _module_function(name):
    module = ast.parse(STRATEGY.read_text())
    return next(node for node in module.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def _program_function(name):
    module = ast.parse(PROGRAM.read_text())
    return next(node for node in module.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def _decorator_keyword(function, name):
    jit_decorator = next(
        decorator for decorator in function.decorator_list
        if isinstance(decorator, ast.Call) and ast.unparse(decorator.func) ==
        "functools.partial" and ast.unparse(decorator.args[0]) == "jax.jit")
    return next(keyword.value for keyword in jit_decorator.keywords
                if keyword.arg == name)


def _calls(node, name):
    return any(
        isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        and child.func.id == name for child in ast.walk(node))


def _assert_donated_cache_is_immediately_replaced(method):
    for node in ast.walk(method):
        for _, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for index, statement in enumerate(value[:-1]):
                if not isinstance(statement, ast.If) or not _calls(
                        statement, "denoise_block_dual_cache"):
                    continue
                assert _calls(statement.body[-1], "denoise_block_dual_cache")
                replacement = value[index + 1]
                assert ast.unparse(
                    replacement) == "self.runner.kv_caches = output.kv_caches"
                return
    raise AssertionError("dual-cache branch not found")


def test_runner_resolves_generation_strategy_once_at_startup():
    init_source = ast.unparse(_method("__init__"))

    assert "resolve_generation_strategy(vllm_config)" in init_source
    assert "BlockDiffusionStrategy" in init_source


def test_diffusion_dispatch_precedes_autoregressive_phase_dispatch():
    execute_source = ast.unparse(_method("_execute_model"))
    diffusion_dispatch = execute_source.index(
        "self.block_diffusion_strategy.execute(scheduler_output)")
    autoregressive_dispatch = execute_source.index(
        "self.enable_continue_decode")

    assert diffusion_dispatch < autoregressive_dispatch


def test_finished_requests_are_cleaned_before_empty_cycle_return():
    execute_source = ast.unparse(_method("_execute_model"))
    cleanup = execute_source.index("on_scheduler_update")
    empty_cycle = execute_source.index(
        "if not scheduler_output.total_num_scheduled_tokens")

    assert cleanup < empty_cycle


def test_diffusion_precompile_uses_the_runtime_mesh_context():
    capture_source = ast.unparse(_method("capture_model"))

    assert "with jax.set_mesh(self.mesh)" in capture_source
    assert "self.block_diffusion_strategy.precompile()" in capture_source


def test_diffusion_forward_uses_nested_jit_safe_model_callable():
    forward_source = ast.unparse(_strategy_method("_run_model"))

    assert "runner.model_fn_no_options" in forward_source
    assert "runner.model_fn(" not in forward_source


def test_dual_cache_has_distinct_full_partial_and_final_forwards():
    denoise_source = ast.unparse(_strategy_method("_denoise_blocks"))
    partial_source = ast.unparse(
        _strategy_method("_model_forward_partial_subblock"))
    final_source = ast.unparse(_strategy_method("_model_forward_final"))

    assert "denoise_block_dual_cache" in denoise_source
    assert "dynamic_slice" in partial_source
    assert "sub_block_size" in partial_source
    assert "[:, -1, :]" in final_source


def test_dual_cache_jit_donates_the_kv_cache_argument():
    function = _program_function("_denoise_block_dual_cache_jit")
    positional_args = [argument.arg for argument in function.args.args]
    assert positional_args[9] == "kv_caches"

    donate_argnums = _decorator_keyword(function, "donate_argnums")
    assert ast.literal_eval(donate_argnums) == (9, )


def test_dual_cache_jit_uses_ar_collective_matmul_compiler_options():
    function = _program_function("_denoise_block_dual_cache_jit")
    compiler_options = ast.literal_eval(
        _decorator_keyword(function, "compiler_options"))

    decode_module = ast.parse(DECODE_LOOP.read_text())
    decode_core = next(
        node for node in decode_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_decode_core")
    ar_compiler_options = ast.literal_eval(
        _decorator_keyword(decode_core, "compiler_options"))

    expected_options = {
        "xla_tpu_all_gather_collective_matmul_mode": "post_spmd_conservative",
        "xla_tpu_reduce_scatter_collective_matmul_mode":
        "post_spmd_conservative",
        "xla_tpu_use_minor_sharding_for_major_trivial_input": "true",
    }
    assert compiler_options == expected_options
    assert compiler_options == ar_compiler_options


def test_runtime_and_precompile_immediately_replace_donated_kv_cache():
    _assert_donated_cache_is_immediately_replaced(
        _strategy_method("_denoise_blocks"))
    _assert_donated_cache_is_immediately_replaced(
        _strategy_method("precompile"))


def test_dual_cache_trace_separates_padding_and_straggler_waste():
    denoise_source = ast.unparse(_strategy_method("_denoise_blocks"))

    assert "padding_row_iterations" in denoise_source
    assert "straggler_row_iterations" in denoise_source


def test_dual_cache_acceptance_trace_is_opt_in_and_host_formatted():
    init_source = ast.unparse(_strategy_method("__init__"))
    denoise_source = ast.unparse(_strategy_method("_denoise_blocks"))
    precompile_source = ast.unparse(_strategy_method("precompile"))
    log_source = ast.unparse(
        _module_function("_log_dual_cache_acceptance_trace"))
    program_source = (STRATEGY.parent / "program.py").read_text()

    assert "get_commit_diagnostics_algorithm" in init_source
    assert "trace_acceptance_steps=self.config.runtime.trace_acceptance_steps" \
        in denoise_source
    assert "if self.config.runtime.trace_acceptance_steps" in denoise_source
    assert "device_outputs += (output.acceptance_trace,)" in denoise_source
    assert "trace_acceptance_steps=self.config.runtime.trace_acceptance_steps" \
        in precompile_source
    assert "q8_log_confidence_bias=self.config.runtime.q8_log_confidence_bias" \
        in denoise_source
    assert "q8_log_confidence_bias=self.config.runtime.q8_log_confidence_bias" \
        in precompile_source
    assert "force_q32_anchor_commit=self.config.runtime.force_q32_anchor_commit" \
        in denoise_source
    assert "force_q32_anchor_commit=self.config.runtime.force_q32_anchor_commit" \
        in precompile_source
    assert "forced_q32_anchor_commits" in denoise_source
    assert "row0_forced_anchor" in log_source
    assert "row0_selected_log_confidence" in log_source
    assert "row0_threshold_margin" in log_source
    assert "json.dumps" in log_source
    assert "debug.callback" not in program_source


def test_diffusion_uses_configured_capacity_and_partial_cache_metadata():
    build_source = ast.unparse(_strategy_method("_build_batch"))
    precompile_source = ast.unparse(_strategy_method("precompile"))

    assert "select_diffusion_batch_size(num_active, capacity)" in build_source
    assert "batch_size = runner.max_num_reqs" not in build_source
    assert "replace_cached_kv=True" in build_source
    assert build_source.count("fp32_rpa_accumulator") == 1
    assert "fp32_rpa_accumulator=self.config.runtime.fp32_partial_rpa" \
        in build_source
    assert "partial_query_start_loc" in build_source
    assert "[0, 0, num_active]" in build_source
    assert "diffusion_batch_sizes(self.batch_size)" in precompile_source


def test_diffusion_rejects_cache_reuse_and_transfer():
    validation_source = ast.unparse(
        _strategy_method("_validate_runner_capabilities"))

    assert "enable_prefix_caching" in validation_source
    assert "kv_transfer_config" in validation_source


def test_diffusion_passes_request_eos_policy_into_denoising():
    prefill_source = ast.unparse(_strategy_method("_process_prefill"))
    decode_source = ast.unparse(_strategy_method("_process_decode"))
    denoise_source = ast.unparse(_strategy_method("_denoise_blocks"))

    assert "sampling_params.ignore_eos" in prefill_source
    assert "sampling_params.ignore_eos" in decode_source
    assert "stop_on_eos_rows" in denoise_source
    assert "eos_token_ids" in denoise_source


def test_final_block_candidate_trimming_is_opt_in_and_host_side():
    prefill_source = ast.unparse(_strategy_method("_process_prefill"))
    decode_source = ast.unparse(_strategy_method("_process_decode"))
    denoise_source = ast.unparse(_strategy_method("_denoise_blocks"))
    precompile_source = ast.unparse(_strategy_method("precompile"))

    assert "self.config.runtime.trim_final_block_candidates" in prefill_source
    assert "trim_generation_mask" in prefill_source
    assert "self.config.runtime.trim_final_block_candidates" in decode_source
    assert "trim_generation_mask" in decode_source
    assert "trim_final_block_candidates" in denoise_source
    assert "trim_generation_mask" not in precompile_source


def test_prompt_prefill_is_one_block_causal_forward_per_length_group():
    build_source = ast.unparse(_strategy_method("_build_prompt_batch"))
    prefill_source = ast.unparse(_strategy_method("_process_prefill"))

    assert "AttentionMaskKind.BLOCK_CAUSAL" in build_source
    assert "rpa_static_query_len=sequence_length" in build_source
    assert "request_distribution = np.array([0, num_active, num_active]" in \
        build_source
    assert "_forward_prompt_blocks(group, prompts)" in prefill_source
    assert "for block_index in range" not in prefill_source
