# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

import sys

import pytest

import tpu_inference.envs as envs
from tpu_inference.envs import enable_envs_cache, environment_variables


def test_getattr_without_cache(monkeypatch: pytest.MonkeyPatch):
    # The test reads each of these before setting it, so it is asserting the
    # default rather than whatever the shell running pytest happened to
    # export. JAX_PLATFORMS in particular is set by every host-only run.
    for name in ("JAX_PLATFORMS", "PHASED_PROFILING_DIR", "TPU_NAME",
                 "TPU_ACCELERATOR_TYPE"):
        monkeypatch.delenv(name, raising=False)
    assert envs.JAX_PLATFORMS == ""
    assert envs.PHASED_PROFILING_DIR == ""
    monkeypatch.setenv("JAX_PLATFORMS", "tpu")
    monkeypatch.setenv("PHASED_PROFILING_DIR", "/tmp/profiling")
    assert envs.JAX_PLATFORMS == "tpu"
    assert envs.PHASED_PROFILING_DIR == "/tmp/profiling"

    assert envs.TPU_NAME is None
    assert envs.TPU_ACCELERATOR_TYPE is None
    monkeypatch.setenv("TPU_NAME", "my-tpu")
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5litepod-16")
    assert envs.TPU_NAME == "my-tpu"
    assert envs.TPU_ACCELERATOR_TYPE == "v5litepod-16"

    # __getattr__ is not decorated with functools.cache
    assert not hasattr(envs.__getattr__, "cache_info")


def test_getattr_with_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JAX_PLATFORMS", "tpu")
    monkeypatch.setenv("TPU_NAME", "my-tpu")

    # __getattr__ is not decorated with functools.cache
    assert not hasattr(envs.__getattr__, "cache_info")

    enable_envs_cache()

    # __getattr__ is decorated with functools.cache
    assert hasattr(envs.__getattr__, "cache_info")
    start_hits = envs.__getattr__.cache_info().hits

    # 2 more hits due to JAX_PLATFORMS and TPU_NAME accesses
    assert envs.JAX_PLATFORMS == "tpu"
    assert envs.TPU_NAME == "my-tpu"
    assert envs.__getattr__.cache_info().hits == start_hits + 2

    # All environment variables are cached
    for environment_variable in environment_variables:
        envs.__getattr__(environment_variable)
    assert envs.__getattr__.cache_info(
    ).hits == start_hits + 2 + len(environment_variables)

    # Reset envs.__getattr__ back to non-cached version to
    # avoid affecting other tests
    envs.__getattr__ = envs.__getattr__.__wrapped__


def test_boolean_env_vars(monkeypatch: pytest.MonkeyPatch):
    # Ensure clean environment for boolean vars by setting to default "0"
    monkeypatch.setenv("SKIP_JAX_PRECOMPILE", "0")
    monkeypatch.setenv("VLLM_XLA_CHECK_RECOMPILATION", "0")
    monkeypatch.setenv("NEW_MODEL_DESIGN", "0")
    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "0")
    monkeypatch.setenv("USE_MOE_EP_KERNEL", "0")
    monkeypatch.setenv("LAYOUT_Q_PROJ_AS_NDH", "0")
    monkeypatch.setenv("USE_BATCHED_RPA_KERNEL", "0")
    monkeypatch.setenv("DISABLE_WEIGHT_REQUANTIZATION", "0")

    # Test SKIP_JAX_PRECOMPILE (default False)
    assert envs.SKIP_JAX_PRECOMPILE is False
    monkeypatch.setenv("SKIP_JAX_PRECOMPILE", "1")
    assert envs.SKIP_JAX_PRECOMPILE is True
    monkeypatch.setenv("SKIP_JAX_PRECOMPILE", "0")
    assert envs.SKIP_JAX_PRECOMPILE is False

    # Test DISABLE_WEIGHT_REQUANTIZATION (default False)
    assert envs.DISABLE_WEIGHT_REQUANTIZATION is False
    monkeypatch.setenv("DISABLE_WEIGHT_REQUANTIZATION", "1")
    assert envs.DISABLE_WEIGHT_REQUANTIZATION is True
    monkeypatch.setenv("DISABLE_WEIGHT_REQUANTIZATION", "0")
    assert envs.DISABLE_WEIGHT_REQUANTIZATION is False

    # Test VLLM_XLA_CHECK_RECOMPILATION (default False)
    assert envs.VLLM_XLA_CHECK_RECOMPILATION is False
    monkeypatch.setenv("VLLM_XLA_CHECK_RECOMPILATION", "1")
    assert envs.VLLM_XLA_CHECK_RECOMPILATION is True
    monkeypatch.setenv("VLLM_XLA_CHECK_RECOMPILATION", "0")
    assert envs.VLLM_XLA_CHECK_RECOMPILATION is False

    # Test NEW_MODEL_DESIGN (default False)
    assert envs.NEW_MODEL_DESIGN is False
    monkeypatch.setenv("NEW_MODEL_DESIGN", "1")
    assert envs.NEW_MODEL_DESIGN is True

    # Test USE_MOE_EP_KERNEL (default False)
    assert envs.USE_MOE_EP_KERNEL is False
    monkeypatch.setenv("USE_MOE_EP_KERNEL", "1")
    assert envs.USE_MOE_EP_KERNEL is True

    # Test ENABLE_QUANTIZED_MATMUL_KERNEL (default False)
    assert envs.ENABLE_QUANTIZED_MATMUL_KERNEL is False
    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "1")
    assert envs.ENABLE_QUANTIZED_MATMUL_KERNEL is True

    # Test LAYOUT_Q_PROJ_AS_NDH (default False)
    assert envs.LAYOUT_Q_PROJ_AS_NDH is False
    monkeypatch.setenv("LAYOUT_Q_PROJ_AS_NDH", "1")
    assert envs.LAYOUT_Q_PROJ_AS_NDH is True

    # Test USE_BATCHED_RPA_KERNEL (default False)
    assert envs.USE_BATCHED_RPA_KERNEL is False
    monkeypatch.setenv("USE_BATCHED_RPA_KERNEL", "1")
    assert envs.USE_BATCHED_RPA_KERNEL is True


def test_boolean_env_vars_string_values(monkeypatch: pytest.MonkeyPatch):
    """Test that boolean env vars accept string values like 'True' and 'False'"""

    # Test NEW_MODEL_DESIGN with string "True"
    monkeypatch.setenv("NEW_MODEL_DESIGN", "True")
    assert envs.NEW_MODEL_DESIGN is True

    monkeypatch.setenv("NEW_MODEL_DESIGN", "true")
    assert envs.NEW_MODEL_DESIGN is True

    monkeypatch.setenv("NEW_MODEL_DESIGN", "False")
    assert envs.NEW_MODEL_DESIGN is False

    monkeypatch.setenv("NEW_MODEL_DESIGN", "false")
    assert envs.NEW_MODEL_DESIGN is False

    # Test SKIP_JAX_PRECOMPILE with string values
    monkeypatch.setenv("SKIP_JAX_PRECOMPILE", "True")
    assert envs.SKIP_JAX_PRECOMPILE is True

    monkeypatch.setenv("SKIP_JAX_PRECOMPILE", "false")
    assert envs.SKIP_JAX_PRECOMPILE is False

    # Test VLLM_XLA_CHECK_RECOMPILATION with string values
    monkeypatch.setenv("VLLM_XLA_CHECK_RECOMPILATION", "TRUE")
    assert envs.VLLM_XLA_CHECK_RECOMPILATION is True

    monkeypatch.setenv("VLLM_XLA_CHECK_RECOMPILATION", "FALSE")
    assert envs.VLLM_XLA_CHECK_RECOMPILATION is False

    # Test USE_MOE_EP_KERNEL with string values
    monkeypatch.setenv("USE_MOE_EP_KERNEL", "true")
    assert envs.USE_MOE_EP_KERNEL is True

    monkeypatch.setenv("USE_MOE_EP_KERNEL", "False")
    assert envs.USE_MOE_EP_KERNEL is False


def test_boolean_env_vars_invalid_values(monkeypatch: pytest.MonkeyPatch):
    """Test that boolean env vars raise errors for invalid values"""

    # Test invalid value for NEW_MODEL_DESIGN
    monkeypatch.setenv("NEW_MODEL_DESIGN", "yes")
    with pytest.raises(
            ValueError,
            match="Invalid boolean value 'yes' for NEW_MODEL_DESIGN"):
        _ = envs.NEW_MODEL_DESIGN

    monkeypatch.setenv("NEW_MODEL_DESIGN", "2")
    with pytest.raises(ValueError,
                       match="Invalid boolean value '2' for NEW_MODEL_DESIGN"):
        _ = envs.NEW_MODEL_DESIGN

    # Test invalid value for SKIP_JAX_PRECOMPILE
    monkeypatch.setenv("SKIP_JAX_PRECOMPILE", "invalid")
    with pytest.raises(
            ValueError,
            match="Invalid boolean value 'invalid' for SKIP_JAX_PRECOMPILE"):
        _ = envs.SKIP_JAX_PRECOMPILE


def test_boolean_env_vars_empty_string(monkeypatch: pytest.MonkeyPatch):
    """Test that empty string returns default value"""

    monkeypatch.setenv("NEW_MODEL_DESIGN", "")
    assert envs.NEW_MODEL_DESIGN is False  # Should return default

    monkeypatch.setenv("SKIP_JAX_PRECOMPILE", "")
    assert envs.SKIP_JAX_PRECOMPILE is False  # Should return default


def test_integer_env_vars(monkeypatch: pytest.MonkeyPatch):
    # Ensure clean environment for integer vars by setting to defaults
    monkeypatch.setenv("PYTHON_TRACER_LEVEL", "1")
    monkeypatch.setenv("NUM_SLICES", "1")
    monkeypatch.delenv("REQUANTIZE_BLOCK_SIZE", raising=False)
    monkeypatch.delenv("MOE_REQUANTIZE_BLOCK_SIZE", raising=False)

    assert envs.PYTHON_TRACER_LEVEL == 1
    monkeypatch.setenv("PYTHON_TRACER_LEVEL", "3")
    assert envs.PYTHON_TRACER_LEVEL == 3
    monkeypatch.setenv("PYTHON_TRACER_LEVEL", "0")
    assert envs.PYTHON_TRACER_LEVEL == 0

    # Test NUM_SLICES (default 1)
    assert envs.NUM_SLICES == 1
    monkeypatch.setenv("NUM_SLICES", "2")
    assert envs.NUM_SLICES == 2
    monkeypatch.setenv("NUM_SLICES", "4")
    assert envs.NUM_SLICES == 4

    # Test REQUANTIZE_BLOCK_SIZE: default should be None
    assert envs.REQUANTIZE_BLOCK_SIZE is None
    monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "512")
    assert envs.REQUANTIZE_BLOCK_SIZE == 512

    # Test MOE_REQUANTIZE_BLOCK_SIZE default should be None
    assert envs.MOE_REQUANTIZE_BLOCK_SIZE is None
    monkeypatch.setenv("MOE_REQUANTIZE_BLOCK_SIZE", "512")
    assert envs.MOE_REQUANTIZE_BLOCK_SIZE == 512


def test_model_impl_type_choices(monkeypatch: pytest.MonkeyPatch):
    # Test case sensitive choices
    monkeypatch.setenv("MODEL_IMPL_TYPE", "flax_nnx")
    assert envs.MODEL_IMPL_TYPE == "flax_nnx"

    monkeypatch.setenv("MODEL_IMPL_TYPE", "vllm")
    assert envs.MODEL_IMPL_TYPE == "vllm"

    monkeypatch.setenv("DRAFT_MODEL_IMPL_TYPE", "flax_nnx")
    assert envs.DRAFT_MODEL_IMPL_TYPE == "flax_nnx"

    monkeypatch.setenv("DRAFT_MODEL_IMPL_TYPE", "vllm")
    assert envs.DRAFT_MODEL_IMPL_TYPE == "vllm"


def test_string_env_vars_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("PREFILL_SLICES", raising=False)
    monkeypatch.delenv("DECODE_SLICES", raising=False)
    monkeypatch.delenv("REQUANTIZE_WEIGHT_DTYPE", raising=False)
    monkeypatch.delenv("AGGREGATED_STATS_DIR", raising=False)
    monkeypatch.delenv("MOE_REQUANTIZE_WEIGHT_DTYPE", raising=False)

    assert envs.JAX_PLATFORMS == ""
    assert envs.PREFILL_SLICES == ""
    assert envs.DECODE_SLICES == ""
    assert envs.PHASED_PROFILING_DIR == ""
    assert envs.AGGREGATED_STATS_DIR == ""
    assert envs.REQUANTIZE_WEIGHT_DTYPE == "float8_e4m3fn"
    assert envs.MOE_REQUANTIZE_WEIGHT_DTYPE == ""


def test_none_default_env_vars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TPU_ACCELERATOR_TYPE", raising=False)
    monkeypatch.delenv("TPU_NAME", raising=False)
    monkeypatch.delenv("TPU_WORKER_ID", raising=False)

    assert envs.TPU_ACCELERATOR_TYPE is None
    assert envs.TPU_NAME is None
    assert envs.TPU_WORKER_ID is None


def test_ray_env_vars(monkeypatch: pytest.MonkeyPatch):
    assert not envs.RAY_USAGE_STATS_ENABLED
    monkeypatch.setenv("RAY_USAGE_STATS_ENABLED", "1")
    assert envs.RAY_USAGE_STATS_ENABLED

    assert envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE == "shm"


def test_invalid_attribute_raises_error():
    with pytest.raises(AttributeError,
                       match="has no attribute 'NONEXISTENT_VAR'"):
        _ = envs.NONEXISTENT_VAR


def test_dir_returns_all_env_vars():
    env_vars = envs.__dir__()
    assert isinstance(env_vars, list)
    assert len(env_vars) == len(environment_variables)
    assert "JAX_PLATFORMS" in env_vars
    assert "TPU_NAME" in env_vars
    assert "SKIP_JAX_PRECOMPILE" in env_vars
    assert "VLLM_XLA_CHECK_RECOMPILATION" in env_vars
    assert "MODEL_IMPL_TYPE" in env_vars


def test_tpu_multihost_env_vars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TPU_WORKER_ID", "0")
    assert envs.TPU_WORKER_ID == "0"

    monkeypatch.setenv("TPU_MULTIHOST_BACKEND", "ray")
    assert envs.TPU_MULTIHOST_BACKEND == "ray"


def test_disaggregated_serving_env_vars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PREFILL_SLICES", "0,1,2,3")
    assert envs.PREFILL_SLICES == "0,1,2,3"

    monkeypatch.setenv("DECODE_SLICES", "4,5,6,7")
    assert envs.DECODE_SLICES == "4,5,6,7"


def test_model_impl_type_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MODEL_IMPL_TYPE", raising=False)
    assert envs.MODEL_IMPL_TYPE == "auto"


def test_cache_preserves_values_across_env_changes(
        monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JAX_PLATFORMS", "tpu")

    enable_envs_cache()

    assert envs.JAX_PLATFORMS == "tpu"

    # Change environment variable
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")

    # Cached value should still be "tpu"
    assert envs.JAX_PLATFORMS == "tpu"

    # Reset envs.__getattr__ back to non-cached version
    envs.__getattr__ = envs.__getattr__.__wrapped__

    # Now it should reflect the new value
    assert envs.JAX_PLATFORMS == "cpu"


def test_gdn_bf16_recurrent_state_env_var(monkeypatch: pytest.MonkeyPatch):
    """Read unset first, so the default is the reader's rather than
    whatever the shell running pytest happened to export.

    A value the reader cannot parse is refused by name, which env_bool
    does for every setting and is covered above.
    """
    monkeypatch.delenv("GDN_BF16_STATE", raising=False)
    monkeypatch.delenv("GDN_BF16_RECURRENT_STATE", raising=False)
    assert envs.GDN_BF16_RECURRENT_STATE is False
    monkeypatch.setenv("GDN_BF16_RECURRENT_STATE", "1")
    assert envs.GDN_BF16_RECURRENT_STATE is True
    monkeypatch.setenv("GDN_BF16_RECURRENT_STATE", "0")
    assert envs.GDN_BF16_RECURRENT_STATE is False


def test_retired_switch_spelling_is_refused(monkeypatch: pytest.MonkeyPatch):
    """The old spelling of a renamed setting is an error, not a no-op."""
    monkeypatch.delenv("USE_MOE_FUSED_EP_KERNEL", raising=False)
    monkeypatch.setenv("MOE_FUSED_EP", "1")
    with pytest.raises(ValueError, match="USE_MOE_FUSED_EP_KERNEL"):
        envs.USE_MOE_FUSED_EP_KERNEL


def test_retired_threshold_spelling_is_refused(
        monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", raising=False)
    monkeypatch.setenv("MOE_FUSED_EP_MIN_TOKENS", "2048")
    with pytest.raises(ValueError, match="MOE_FUSED_EP_KERNEL_MIN_TOKENS"):
        envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS


def test_a_retired_name_turned_off_warns_and_the_boot_continues(
        monkeypatch: pytest.MonkeyPatch, caplog):
    """MOE_FUSED_EP=0 is the most conservative thing a deployment could have
    written -- it pinned the feature off -- and refusing it made that the one
    config that could not start. The values that ask for nothing warn."""
    monkeypatch.delenv("USE_MOE_FUSED_EP_KERNEL", raising=False)
    for off in ("0", "", "false", "False", " 0 "):
        monkeypatch.setenv("MOE_FUSED_EP", off)
        assert envs.USE_MOE_FUSED_EP_KERNEL is False


def test_a_retired_name_turned_off_names_the_rename(
        monkeypatch: pytest.MonkeyPatch):
    """The warning has to carry the new spelling and the value that was
    written, so the operator can find the line and rewrite it."""
    monkeypatch.delenv("USE_MOE_FUSED_EP_KERNEL", raising=False)
    monkeypatch.setenv("MOE_FUSED_EP", "0")
    said = []

    class _Recorder:

        def warning_once(self, msg, *args):
            said.append(msg % args)

    # tpu_inference.logger the ATTRIBUTE is a Logger instance, so the module
    # this patches has to be reached through sys.modules rather than by
    # dotted name.
    monkeypatch.setattr(sys.modules["tpu_inference.logger"], "init_logger",
                        lambda name: _Recorder())
    assert envs.USE_MOE_FUSED_EP_KERNEL is False
    assert len(said) == 1
    assert "USE_MOE_FUSED_EP_KERNEL" in said[0]
    assert "MOE_FUSED_EP='0'" in said[0]


def test_a_retired_name_asking_for_something_still_refuses(
        monkeypatch: pytest.MonkeyPatch):
    """The half worth keeping: a value the deployment meant to act on would
    silently not be honoured, so the boot stops and quotes it."""
    monkeypatch.delenv("USE_MOE_FUSED_EP_KERNEL", raising=False)
    for asking in ("1", "true", "True", "not-a-bool"):
        monkeypatch.setenv("MOE_FUSED_EP", asking)
        with pytest.raises(ValueError, match="USE_MOE_FUSED_EP_KERNEL") as e:
            envs.USE_MOE_FUSED_EP_KERNEL
        assert repr(asking) in str(e.value)


def test_a_retired_threshold_turned_off_warns_rather_than_refusing(
        monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", raising=False)
    monkeypatch.setenv("MOE_FUSED_EP_MIN_TOKENS", "")
    assert envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS == 1024


def test_a_retired_switch_beside_the_current_one_boots(
        monkeypatch: pytest.MonkeyPatch):
    """A config that has to start trees from both sides of the rename writes
    the switch under both spellings, because a tree reads only the spelling
    it knows. Whatever the old name carries, this tree reads the new one."""
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "1")
    for anything in ("1", "0", "", "true", "False", " 1 ", "not-a-bool"):
        monkeypatch.setenv("MOE_FUSED_EP", anything)
        assert envs.USE_MOE_FUSED_EP_KERNEL is True
    # And the tree reads the new spelling even when the two disagree: the
    # old one is ignored, not merged.
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "0")
    monkeypatch.setenv("MOE_FUSED_EP", "1")
    assert envs.USE_MOE_FUSED_EP_KERNEL is False


def test_a_retired_threshold_beside_the_current_one_boots(
        monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", "2048")
    for anything in ("2048", "1024", "0", "", "not-an-int"):
        monkeypatch.setenv("MOE_FUSED_EP_MIN_TOKENS", anything)
        assert envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS == 2048


def test_the_dual_spelling_warning_names_both_spellings(
        monkeypatch: pytest.MonkeyPatch):
    """The operator has to be able to tell from the line which spelling ran
    the boot and that the other one did nothing."""
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "1")
    monkeypatch.setenv("MOE_FUSED_EP", "1")
    said = []

    class _Recorder:

        def warning_once(self, msg, *args):
            said.append(msg % args)

    monkeypatch.setattr(sys.modules["tpu_inference.logger"], "init_logger",
                        lambda name: _Recorder())
    assert envs.USE_MOE_FUSED_EP_KERNEL is True
    assert len(said) == 1
    assert "MOE_FUSED_EP='1'" in said[0]
    assert "USE_MOE_FUSED_EP_KERNEL" in said[0]
    assert "ignored" in said[0]


def test_a_retired_name_beside_an_empty_current_one_still_refuses(
        monkeypatch: pytest.MonkeyPatch):
    """The readers below take an empty value as unset, so a switch turned on
    under the old name with the new one blank is the silent-inert case the
    refusal is for, not the compatibility idiom."""
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "")
    monkeypatch.setenv("MOE_FUSED_EP", "1")
    with pytest.raises(ValueError, match="USE_MOE_FUSED_EP_KERNEL"):
        envs.USE_MOE_FUSED_EP_KERNEL


def test_the_retired_recurrent_state_spelling_is_refused(
        monkeypatch: pytest.MonkeyPatch):
    """GDN_BF16_STATE is the spelling every config written before the
    rename carries, and nothing reads it now."""
    monkeypatch.delenv("GDN_BF16_RECURRENT_STATE", raising=False)
    monkeypatch.setenv("GDN_BF16_STATE", "1")
    with pytest.raises(ValueError, match="GDN_BF16_RECURRENT_STATE") as e:
        envs.GDN_BF16_RECURRENT_STATE
    assert "GDN_BF16_STATE='1'" in str(e.value)


def test_the_retired_recurrent_state_spelling_turned_off_boots(
        monkeypatch: pytest.MonkeyPatch):
    """A config that pinned the cache to float32 under the old name asked
    for nothing it is not already getting, so it warns and continues."""
    monkeypatch.delenv("GDN_BF16_RECURRENT_STATE", raising=False)
    for off in ("0", "", "false", "False", " 0 "):
        monkeypatch.setenv("GDN_BF16_STATE", off)
        assert envs.GDN_BF16_RECURRENT_STATE is False


def test_the_retired_recurrent_state_spelling_beside_the_current_one_boots(
        monkeypatch: pytest.MonkeyPatch):
    """One config often has to start trees from both sides of the rename,
    and this tree reads the new spelling."""
    monkeypatch.setenv("GDN_BF16_RECURRENT_STATE", "1")
    for anything in ("1", "0", "", "true", "False", " 1 ", "not-a-bool"):
        monkeypatch.setenv("GDN_BF16_STATE", anything)
        assert envs.GDN_BF16_RECURRENT_STATE is True
    monkeypatch.setenv("GDN_BF16_RECURRENT_STATE", "0")
    monkeypatch.setenv("GDN_BF16_STATE", "1")
    assert envs.GDN_BF16_RECURRENT_STATE is False


def test_both_readers_of_one_feature_take_the_same_leading_sign(
        monkeypatch: pytest.MonkeyPatch):
    """They had opposite tolerances for one typo: the integer reader took
    '+512' and the boolean beside it refused '+1' by name."""
    monkeypatch.delenv("MOE_FUSED_EP", raising=False)
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", "+2048")
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "+1")
    assert envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS == 2048
    assert envs.USE_MOE_FUSED_EP_KERNEL is True
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "+0")
    assert envs.USE_MOE_FUSED_EP_KERNEL is False


def test_a_bounds_refusal_quotes_what_was_written(
        monkeypatch: pytest.MonkeyPatch):
    """'+0' and '-0' both parse to 0, and a report of '=0' alone loses the
    spelling the operator has to go and find in their config."""
    monkeypatch.delenv("MOE_FUSED_EP_MIN_TOKENS", raising=False)
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", "-0")
    with pytest.raises(ValueError, match="'-0'"):
        envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS


def test_current_switch_spelling_is_unaffected(
        monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MOE_FUSED_EP", raising=False)
    monkeypatch.delenv("MOE_FUSED_EP_MIN_TOKENS", raising=False)
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "1")
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", "2048")
    assert envs.USE_MOE_FUSED_EP_KERNEL is True
    assert envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS == 2048


def test_the_integer_reader_takes_one_spelling_of_an_integer(
        monkeypatch: pytest.MonkeyPatch):
    """Python's int() takes any Unicode decimal digit, underscore separators
    and a leading sign, so a fullwidth numeral pasted out of a document
    configured a setting with no signal that anything was unusual."""
    for bad in ("\uff11\uff10\uff12\uff14", "\u0661\u0660\u0662\u0664",
                "1_024", "0x400", "1.0", "1e3"):
        monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", bad)
        with pytest.raises(ValueError, match="Invalid integer value"):
            envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", "+512")
    assert envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS == 512


def test_both_readers_of_one_feature_take_the_same_whitespace(
        monkeypatch: pytest.MonkeyPatch):
    """They had opposite tolerances for the same typo: the integer stripped
    it and the boolean beside it hard-failed on a single trailing space."""
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", " 2048 ")
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", " 1 ")
    assert envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS == 2048
    assert envs.USE_MOE_FUSED_EP_KERNEL is True
