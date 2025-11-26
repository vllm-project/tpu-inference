"""Quantize a DeepSeek-V3 checkpoint to TPU FP4 and export packed safetensors."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from huggingface_hub import snapshot_download
from jax.sharding import Mesh
from safetensors import safe_open
from safetensors.torch import save_file
from vllm.config import ModelConfig, VllmConfig

from tpu_inference.models.common.model_loader import apply_qwix_quantization
# IMPORT THE DEDICATED MODEL CLASS
from tpu_inference.models.jax.deepseek_v3 import DeepSeekV3
from tpu_inference.models.jax.utils import file_utils
from tpu_inference.models.jax.utils.quantization.quantization_utils import (  # DEFAULT_GPT_OSS_TPU_FP4_CONFIG, # Removed unused config
    DEFAULT_DEEPSEEK_TPU_FP4_CONFIG, update_vllm_config_for_qwix_quantization)
from tpu_inference.models.jax.utils.quantization.tpu_fp4_utils import (
    TPU_FP4_SUBCHANNEL_SIZE, pack_tpu_fp4_from_fp32)
from tpu_inference.models.jax.utils.weight_utils import get_param

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# JAX path to HF tensor mapping for packed weights.
# This mapping must align with the JAX paths in deepseek_v3.py
# and the HF keys in the source checkpoint.
_PACK_TARGETS = {
    # DS-V3 MAPPINGS
    # Standard experts
    "layers.{layer}.custom_module.kernel_gating_EDF":
    "model.layers.{layer}.mlp.experts.*.gate_proj.weight",
    "layers.{layer}.custom_module.kernel_down_proj_EFD":
    "model.layers.{layer}.mlp.experts.*.down_proj.weight",
    "layers.{layer}.custom_module.kernel_up_proj_EDF":
    "model.layers.{layer}.mlp.experts.*.up_proj.weight",
    # Shared experts
    "layers.{layer}.shared_experts.kernel_gating_DF":
    "model.layers.{layer}.mlp.shared_experts.gate_proj.weight",
    "layers.{layer}.shared_experts.kernel_down_proj_FD":
    "model.layers.{layer}.mlp.shared_experts.down_proj.weight",
    "layers.{layer}.shared_experts.kernel_up_proj_DF":
    "model.layers.{layer}.mlp.shared_experts.up_proj.weight",
}


class QuantizationError(RuntimeError):
    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=
        "Quantize DeepSeek-V3 experts to TPU FP4 and emit packed safetensors")
    parser.add_argument("source",
                        type=Path,
                        help="Path to the source Hugging Face checkpoint")
    parser.add_argument("output",
                        type=Path,
                        help="Directory to write the quantized checkpoint")
    parser.add_argument("--seed",
                        type=int,
                        default=0,
                        help="PRNG seed for deterministic operations")
    parser.add_argument("--overwrite",
                        action="store_true",
                        help="Allow overwriting an existing output directory")
    return parser.parse_args()


def _prepare_output_dir(source: Path, output: Path, overwrite: bool) -> Path:
    if source.exists():
        source_root = source
    else:
        repo_id = str(source)
        if file_utils.is_hf_repo(repo_id):
            LOGGER.info("Downloading model snapshot for %s", repo_id)
            source_root = Path(snapshot_download(repo_id))
        else:
            raise FileNotFoundError(f"Source checkpoint not found: {source}")

    # --- HANDLES OVERWRITE AND DISK SPACE ISSUE (Selective Copy) ---
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory {output} already exists; use --overwrite")
        shutil.rmtree(output)
        LOGGER.info("Deleted existing output directory: %s", output)

    output.mkdir(parents=True, exist_ok=False)

    LOGGER.info("Copying essential metadata files...")
    for item in source_root.iterdir():
        if item.is_file() and not item.name.endswith(".safetensors"):
            # shutil.copy2(item, output / item.name
            shutil.copyfile(item, output / item.name)
    LOGGER.info("Creating empty placeholders for %d model shards...", 163)
    for i in range(1, 164):
        FILENAME = f"model-{i:05}-of-000163.safetensors"
        (output / FILENAME).touch()

    return output


def _build_vllm_config(model_dir: Path) -> VllmConfig:
    model_config = ModelConfig(model=str(model_dir),
                               trust_remote_code=True,
                               dtype="bfloat16")
    hf_config = model_config.hf_config
    quant_method = None
    if hasattr(hf_config, "quantization_config") and \
            hf_config.quantization_config is not None:  # type: ignore[attr-defined]
        quant_method = hf_config.quantization_config.get(  # type: ignore[attr-defined]
            "quant_method")
    if quant_method:
        model_config.quantization = quant_method  # type: ignore[assignment]
    model_config.hf_config = hf_config

    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.load_config.download_dir = str(model_dir)
    update_vllm_config_for_qwix_quantization(vllm_config)

    # --- START CRITICAL PATCH: INJECT TPU FP4 QUANTIZATION CONFIG ---
    if not hasattr(hf_config, "quantization_config"):
        hf_config.quantization_config = {
            "quant_method": "tpu_fp4",
            "activation_scheme": "static",  # Use static for weight-only quant
            "fmt": "e2m1fn",  # Specific to TPU FP4
            "weight_block_size":
            [1, TPU_FP4_SUBCHANNEL_SIZE],  # Use derived block size
        }
        LOGGER.info(
            "PATCHED: Injected TPU FP4 quantization_config into hf_config.")
    # --- END CRITICAL PATCH ---

    # Use the DeepSeek specific quantization config
    if "quantization" not in vllm_config.additional_config:
        vllm_config.additional_config[
            "quantization"] = DEFAULT_DEEPSEEK_TPU_FP4_CONFIG

    return vllm_config


def _create_mesh() -> Mesh:
    devices = jax.devices()
    if not devices:
        raise QuantizationError("No JAX devices available for quantization")
    mesh_array = np.array(devices).reshape(1, len(devices))
    return Mesh(mesh_array, ("data", "model"))


def _apply_quantization(vllm_config: VllmConfig, mesh: Mesh,
                        seed: int) -> DeepSeekV3:

    # NO MANUAL PATCHES NEEDED HERE. DeepSeekV3 handles its own config keys.
    rng = jax.random.PRNGKey(seed)

    vllm_config.model_config.hf_config.num_hidden_layers = 4 
    LOGGER.warning("DEBUG: Temporarily limiting model layers to 4 for compilation test.")


    with jax.default_device(jax.devices('cpu')[0]):
        # Model creation (initializes weights as large BF16 buffers) happens on CPU RAM
        model = DeepSeekV3(vllm_config, rng, mesh)

    #model.weight_loader.load_weights_metadata()

    # This must be done because we skipped model.load_weights(rng)
    model.initialize_cache()

    with mesh:
        #model.load_weights(rng)
        model = apply_qwix_quantization(vllm_config,
                                        model,
                                        rng,
                                        mesh,
                                        apply_to_abstract_model=False) 
    return model


def _to_host_float32(array: jax.Array) -> np.ndarray:
    host = jnp.asarray(array, dtype=jnp.float32)
    return np.asarray(jax.device_get(host))


def _collect_packed_weights(
        model: DeepSeekV3) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    LOGGER.info(f"DEBUG: Model object reference: {model}")
    import pdb; pdb.set_trace()
    params = nnx.state(model)
    replacements: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    num_layers = model.vllm_config.model_config.hf_config.num_hidden_layers
    for layer in range(num_layers):

        if layer < 3:
            # Dense layers (0, 1, 2) do not use the MoE paths defined in _PACK_TARGETS.
            # We skip them entirely.
            continue

        for jax_path_tpl, hf_tpl in _PACK_TARGETS.items():
            jax_path = jax_path_tpl.format(layer=layer)
            hf_key = hf_tpl.format(layer=layer)
            param = get_param(params, jax_path)
            if not hasattr(param, "array"):
                raise QuantizationError(
                    f"Expected quantized array at {jax_path}")
            q_array = param.array
            codes_fp32 = np.array(_to_host_float32(q_array.qvalue.value),
                                  copy=True)
            scales_fp32 = np.array(_to_host_float32(q_array.scale.value),
                                   copy=True)

            codes_tensor = torch.from_numpy(codes_fp32).to(torch.float32)
            scales_tensor = torch.from_numpy(scales_fp32).to(torch.bfloat16)

            # Reorder to (experts, hidden, channels)
            codes_tensor = codes_tensor.permute(0, 2, 1).contiguous()
            scales_tensor = scales_tensor.permute(0, 2, 1).contiguous()

            blocks_tensor = pack_tpu_fp4_from_fp32(codes_tensor).contiguous()
            replacements[hf_key] = (blocks_tensor, scales_tensor)
    return replacements


def _rewrite_safetensors(
        model_dir: Path,
        replacements: Mapping[str, Tuple[torch.Tensor, torch.Tensor]]) -> None:
    pending = set(replacements.keys())
    for st_path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(st_path, framework="pt") as handle:
            keys = list(handle.keys())
            metadata = handle.metadata()
            tensors = {name: handle.get_tensor(name) for name in keys}

        local_updates: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        new_tensors = {}
        for name, tensor in tensors.items():
            base_key = None
            if name in replacements:
                base_key = name
            elif name.endswith("_blocks") or name.endswith("_scales"):
                candidate = name[:-7]
                if candidate in replacements:
                    base_key = candidate
            if base_key is None:
                new_tensors[name] = tensor
            else:
                local_updates[base_key] = replacements[base_key]
        if not local_updates:
            continue
        for base_key, bundle in local_updates.items():
            blocks, scales = bundle
            new_tensors[f"{base_key}_blocks"] = blocks.cpu()
            new_tensors[f"{base_key}_scales"] = scales.cpu()
            pending.discard(base_key)
        save_file(new_tensors, st_path, metadata=metadata)
        LOGGER.info("Updated %s", st_path)
    if pending:
        missing = ", ".join(sorted(pending))
        raise QuantizationError(
            f"Failed to rewrite tensors for keys: {missing}")


def _update_config_json(model_dir: Path) -> None:
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())
    quant_config = config.get("quantization_config", {})
    quant_config.update({
        "quant_method": "tpu_fp4",
        "weight_block_size": [1, TPU_FP4_SUBCHANNEL_SIZE],
    })
    config["quantization_config"] = quant_config
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = _parse_args()
    output_dir = _prepare_output_dir(args.source, args.output, args.overwrite)
    vllm_config = _build_vllm_config(output_dir)
    mesh = _create_mesh()
    model = _apply_quantization(vllm_config, mesh, args.seed)
    replacements = _collect_packed_weights(model)
    _rewrite_safetensors(output_dir, replacements)
    _update_config_json(output_dir)
    LOGGER.info("Quantized checkpoint written to %s", output_dir)


if __name__ == "__main__":
    main()
