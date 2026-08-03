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
"""Generator for K3-tiny: a small random-weight checkpoint with the Kimi-K3
text-stack architecture (HF `moonshotai/Kimi-K3`, `model_type=kimi_linear`).

Produces two variants that represent the exact same function:
  * bf16:   all weights bf16 (`k3-tiny/`)
  * mxfp4:  routed-expert weights packed in the compressed-tensors
            `mxfp4-pack-quantized` format (`k3-tiny-mxfp4/`), everything else
            identical to the bf16 variant.

The routed-expert weights are drawn directly on the MXFP4 grid (random E2M1
codes x power-of-two E8M0 group scales), so `dequantize(mxfp4 variant) ==
bf16 variant` holds bit-exactly by construction. Packing/unpacking uses
`compressed_tensors`' own helpers so the byte format is canonical.

Topology (scaled-down but architecture-faithful to Kimi-K3):
  * 9 layers: 6 KDA + 3 MLA; 1-indexed `kda_layers=[1,2,3,5,6,7]`,
    `full_attn_layers=[4,8,9]` - the 3:1 KDA:MLA pattern with K3's extra
    final MLA layer (two consecutive MLA at the end, like K3's layers 92/93).
  * Layer 0 dense MLP, layers 1..8 MoE (`first_k_dense_replace=1`).
  * hidden 1024, 8 heads x head_dim 128 (KDA); MLA q/kv-lora with
    qk_nope 128 / qk_rope 64 / v 128 (NoPE + sigmoid output gate).
  * LatentMoE: 32 routed experts, top-4, 1 shared expert, latent dim 512,
    expert intermediate 384, latent RMSNorm; SiTU activation.
  * AttnRes with `attn_res_block_size=4` (depth checkpoints at layers 0/4/8).
  * Full-rank KDA output gate; `gate_lower_bound=-5.0`.

Weight names/shapes follow the HF reference implementation in
`moonshotai/Kimi-K3` (`modeling_kimi_linear.py`); the quantization_config
mirrors the real K3 `config.json` (same ignore regexes, group_size 32).
Where a released checkpoint's *storage* layout differs from the reference's
parameter shape, the checkpoint wins -- see `store_a_log`, which reproduces
Kimi-K3's zero-padded `A_log` buffer and Kimi-Linear-48B's `[1, 1, H, 1]`.

Usage:
    python tests/models/jax/utils/k3_tiny_generator.py \
        --output-root /mnt/disks/persist/models --seed 42
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file

TINY_TEXT_CONFIG = {
    "architectures": ["KimiLinearForCausalLM"],
    "auto_map": {
        "AutoConfig": "configuration_kimi_k3.KimiLinearConfig",
        "AutoModel": "modeling_kimi_linear.KimiLinearModel",
        "AutoModelForCausalLM": "modeling_kimi_linear.KimiLinearForCausalLM",
    },
    "model_type": "kimi_linear",
    "dtype": "bfloat16",
    "vocab_size": 8192,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
    "hidden_size": 1024,
    "num_hidden_layers": 9,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "intermediate_size": 4096,
    "hidden_act": "situ",
    "activation_situ_beta": 4.0,
    "activation_situ_linear_beta": 25.0,
    "attn_res_block_size": 4,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "num_experts": 32,
    "num_experts_per_token": 4,
    "num_shared_experts": 1,
    "moe_intermediate_size": 384,
    "routed_expert_hidden_size": 512,
    "latent_moe_use_norm": True,
    "moe_renormalize": True,
    "moe_router_activation_func": "sigmoid",
    "num_expert_group": 1,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "use_grouped_topk": True,
    "routed_scaling_factor": 1.0,
    "q_lora_rank": 256,
    "kv_lora_rank": 128,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "mla_use_nope": True,
    "mla_use_output_gate": True,
    "num_nextn_predict_layers": 0,
    "linear_attn_config": {
        "full_attn_layers": [4, 8, 9],
        "kda_layers": [1, 2, 3, 5, 6, 7],
        "head_dim": 128,
        "num_heads": 8,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "use_full_rank_gate": True,
    },
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-05,
    "tie_word_embeddings": False,
    "use_cache": True,
}

# Mirrors the real Kimi-K3 text_config.quantization_config (ignore regexes
# intentionally identical - only `experts.N.w{1,2,3}` are quantized; the
# latent routed_expert_{down,up}_proj stay bf16, matching the K3 checkpoint's
# safetensors index).
QUANTIZATION_CONFIG = {
    "config_groups": {
        "group_0": {
            "format": "mxfp4-pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": 32,
                "num_bits": 4,
                "observer": "minmax",
                "observer_kwargs": {},
                "scale_dtype": "torch.uint8",
                "strategy": "group",
                "symmetric": True,
                "type": "float",
                "zp_dtype": None,
            },
        }
    },
    "format":
    "mxfp4-pack-quantized",
    "global_compression_ratio":
    None,
    "ignore": [
        "re:.*self_attn.*",
        "re:.*shared_experts.*",
        "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
        "re:.*lm_head.*",
        "re:.*vision_tower.*",
        "re:.*mm_projector.*",
    ],
    "kv_cache_scheme":
    None,
    "quant_method":
    "compressed-tensors",
    "quantization_status":
    "compressed",
}


def softplus_gate_config():
    """K3-tiny with Kimi-Linear-48B-style KDA gate flags: no gate_lower_bound
    (-exp(A_log)*softplus decay form) and a low-rank g_a/g_b output gate.
    Same dims as K3-tiny otherwise; gives the softplus gate path CPU goldens
    that the 48B real-weight testbed exercises."""
    cfg = json.loads(json.dumps(TINY_TEXT_CONFIG))
    del cfg["linear_attn_config"]["gate_lower_bound"]
    cfg["linear_attn_config"]["use_full_rank_gate"] = False
    return cfg


E2M1_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def store_a_log(values: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Lay out A_log the way the released checkpoint of this variant does.

    `A_log` holds one log-decay scalar per KDA head (the HF reference
    allocates it with `num_heads` entries; vLLM's KDA gate kernel indexes it
    by head and broadcasts over the head's channels). The two released
    checkpoints store that same vector differently, and the fixtures mirror
    each one so the loader is covered against both:

      * Kimi-K3 (`gate_lower_bound` present): a flat buffer `head_dim` wide
        whose leading `num_heads` entries are the parameter and whose tail is
        zero padding. Read off moonshotai/Kimi-K3's safetensors headers, every
        one of the 69 KDA layers stores `[128]` with indices 0..95 non-zero
        and 96..127 exactly 0.0, at num_heads=96 / head_dim=128. (`o_norm`,
        genuinely per-head-dim and also `[128]`, is dense in the same layers.)
      * Kimi-Linear-48B (`gate_lower_bound` absent): `[1, 1, num_heads, 1]`,
        unpadded (moonshotai/Kimi-Linear-48B-A3B-Instruct).

    Deriving the fixture layout from the released checkpoints rather than from
    what our model builds is the point: an `A_log` written at `[num_heads]`
    agrees with the loader by construction and cannot catch a padded
    checkpoint.

    NOTE for anyone regenerating goldens: the HF reference
    `modeling_kimi_linear.py` allocates `A_log` with `num_heads` entries, so
    loading the K3-form fixture (or the real Kimi-K3 checkpoint) into the
    reference needs the padding dropped first.
    """
    lac = cfg["linear_attn_config"]
    if lac.get("gate_lower_bound") is None:
        return values.reshape(1, 1, -1, 1)
    padded = torch.zeros(lac["head_dim"], dtype=values.dtype)
    padded[:values.numel()] = values
    return padded


class _Rng:
    """Deterministic tensor factory: draw order defines the checkpoint."""

    def __init__(self, seed: int):
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def normal(self, *shape, std=0.02):
        return torch.randn(*shape, generator=self.gen,
                           dtype=torch.float32) * std

    def uniform(self, *shape, low=0.0, high=1.0):
        u = torch.rand(*shape, generator=self.gen, dtype=torch.float32)
        return low + (high - low) * u

    def norm_weight(self, dim):
        # Near-1 with jitter so a missing weight multiplication in a port
        # cannot cancel out.
        return 1.0 + self.normal(dim, std=0.05)

    def randint(self, *shape, high):
        return torch.randint(0,
                             high,
                             shape,
                             generator=self.gen,
                             dtype=torch.int64)


def _make_mxfp4_expert_weight(rng: _Rng, out_dim: int, in_dim: int):
    """Draw a weight directly on the MXFP4 grid.

    Returns (bf16_weight, packed_uint8, scale_e8m0_uint8). The bf16 weight is
    exactly `unpack(packed) * 2**(scale-127)`, i.e. what compressed-tensors
    decompression produces.
    """
    assert in_dim % 32 == 0
    num_groups = in_dim // 32
    # E2M1 magnitude codes (0..7) and signs, uniform over the grid.
    grid = torch.tensor(E2M1_GRID, dtype=torch.float32)
    codes = rng.randint(out_dim, in_dim, high=8)
    signs = torch.where(rng.randint(out_dim, in_dim, high=2) > 0, 1.0, -1.0)
    values = grid[codes] * signs  # in [-6, 6]
    # E8M0 group scales around 2**-8 so dequantized weights have std ~0.01,
    # comparable to a normal(0, 0.02) init.
    scale_exp = rng.randint(out_dim, num_groups, high=3) - 9  # {-9,-8,-7}
    scale = (2.0**scale_exp.to(torch.float32))
    bf16_weight = (values * scale.repeat_interleave(32, dim=1)).to(
        torch.bfloat16)

    from compressed_tensors.compressors.mx_utils import compress_mx_scale
    from compressed_tensors.compressors.nvfp4.helpers import pack_fp4_to_uint8
    packed = pack_fp4_to_uint8(values)
    scale_u8 = compress_mx_scale(scale, torch.uint8)
    return bf16_weight, packed, scale_u8


def build_state_dicts(seed: int, cfg=None):
    """Build (bf16_state_dict, mxfp4_state_dict) sharing all non-expert
    tensors. Names follow moonshotai/Kimi-K3 modeling_kimi_linear.py."""
    cfg = cfg or TINY_TEXT_CONFIG
    rng = _Rng(seed)
    hidden = cfg["hidden_size"]
    lac = cfg["linear_attn_config"]
    n_heads = lac["num_heads"]
    head_dim = lac["head_dim"]
    proj = n_heads * head_dim  # 1024
    conv_k = lac["short_conv_kernel_size"]
    q_head_dim = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]  # 192
    kda_layers_0idx = {i - 1 for i in lac["kda_layers"]}

    bf16 = {}
    mxfp4_only = {}  # replaces experts.* in the mxfp4 variant
    mxfp4_removed = set()  # bf16 expert names not present in mxfp4 variant

    def lin(name, out_d, in_d):
        bf16[name] = rng.normal(out_d, in_d).to(torch.bfloat16)

    bf16["model.embed_tokens.weight"] = rng.normal(cfg["vocab_size"],
                                                   hidden).to(torch.bfloat16)

    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        bf16[f"{p}.input_layernorm.weight"] = rng.norm_weight(hidden).to(
            torch.bfloat16)
        bf16[f"{p}.post_attention_layernorm.weight"] = rng.norm_weight(
            hidden).to(torch.bfloat16)
        # AttnRes site parameters (per layer).
        bf16[f"{p}.self_attention_res_norm.weight"] = rng.norm_weight(
            hidden).to(torch.bfloat16)
        bf16[f"{p}.mlp_res_norm.weight"] = rng.norm_weight(hidden).to(
            torch.bfloat16)
        lin(f"{p}.self_attention_res_proj.weight", 1, hidden)
        lin(f"{p}.mlp_res_proj.weight", 1, hidden)

        if i in kda_layers_0idx:
            a = f"{p}.self_attn"
            lin(f"{a}.q_proj.weight", proj, hidden)
            lin(f"{a}.k_proj.weight", proj, hidden)
            lin(f"{a}.v_proj.weight", proj, hidden)
            for c in ("q_conv1d", "k_conv1d", "v_conv1d"):
                bf16[f"{a}.{c}.weight"] = rng.normal(
                    proj, 1, conv_k, std=0.2).to(torch.bfloat16)
            bf16[f"{a}.A_log"] = store_a_log(
                torch.log(rng.uniform(n_heads, low=1.0,
                                      high=16.0)).to(torch.bfloat16), cfg)
            bf16[f"{a}.dt_bias"] = rng.uniform(proj, low=-0.5,
                                               high=0.5).to(torch.bfloat16)
            lin(f"{a}.f_a_proj.weight", head_dim, hidden)
            lin(f"{a}.f_b_proj.weight", proj, head_dim)
            lin(f"{a}.b_proj.weight", n_heads, hidden)
            if lac.get("use_full_rank_gate", False):
                lin(f"{a}.g_proj.weight", proj, hidden)
            else:
                lin(f"{a}.g_a_proj.weight", head_dim, hidden)
                lin(f"{a}.g_b_proj.weight", proj, head_dim)
            bf16[f"{a}.o_norm.weight"] = rng.norm_weight(head_dim).to(
                torch.bfloat16)
            lin(f"{a}.o_proj.weight", hidden, proj)
        else:
            a = f"{p}.self_attn"
            lin(f"{a}.q_a_proj.weight", cfg["q_lora_rank"], hidden)
            bf16[f"{a}.q_a_layernorm.weight"] = rng.norm_weight(
                cfg["q_lora_rank"]).to(torch.bfloat16)
            lin(f"{a}.q_b_proj.weight", n_heads * q_head_dim,
                cfg["q_lora_rank"])
            lin(f"{a}.kv_a_proj_with_mqa.weight",
                cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"], hidden)
            bf16[f"{a}.kv_a_layernorm.weight"] = rng.norm_weight(
                cfg["kv_lora_rank"]).to(torch.bfloat16)
            lin(f"{a}.kv_b_proj.weight",
                n_heads * (cfg["qk_nope_head_dim"] + cfg["v_head_dim"]),
                cfg["kv_lora_rank"])
            lin(f"{a}.g_proj.weight", n_heads * cfg["v_head_dim"], hidden)
            lin(f"{a}.o_proj.weight", hidden, n_heads * cfg["v_head_dim"])

        if i < cfg["first_k_dense_replace"]:
            lin(f"{p}.mlp.gate_proj.weight", cfg["intermediate_size"], hidden)
            lin(f"{p}.mlp.up_proj.weight", cfg["intermediate_size"], hidden)
            lin(f"{p}.mlp.down_proj.weight", hidden, cfg["intermediate_size"])
        else:
            m = f"{p}.block_sparse_moe"
            lin(f"{m}.gate.weight", cfg["num_experts"], hidden)
            bf16[f"{m}.gate.e_score_correction_bias"] = rng.normal(
                cfg["num_experts"], std=0.1).to(torch.bfloat16)
            latent = cfg["routed_expert_hidden_size"]
            inter = cfg["moe_intermediate_size"]
            for e in range(cfg["num_experts"]):
                for w_name, (od, idim) in {
                        "w1": (inter, latent),
                        "w2": (latent, inter),
                        "w3": (inter, latent),
                }.items():
                    base = f"{m}.experts.{e}.{w_name}"
                    w_bf16, packed, scale = _make_mxfp4_expert_weight(
                        rng, od, idim)
                    bf16[f"{base}.weight"] = w_bf16
                    mxfp4_only[f"{base}.weight_packed"] = packed
                    mxfp4_only[f"{base}.weight_scale"] = scale
                    mxfp4_removed.add(f"{base}.weight")
            shared_inter = inter * cfg["num_shared_experts"]
            lin(f"{m}.shared_experts.gate_proj.weight", shared_inter, hidden)
            lin(f"{m}.shared_experts.up_proj.weight", shared_inter, hidden)
            lin(f"{m}.shared_experts.down_proj.weight", hidden, shared_inter)
            lin(f"{m}.routed_expert_down_proj.weight", latent, hidden)
            lin(f"{m}.routed_expert_up_proj.weight", hidden, latent)
            bf16[f"{m}.routed_expert_norm.weight"] = rng.norm_weight(
                latent).to(torch.bfloat16)

    bf16["model.norm.weight"] = rng.norm_weight(hidden).to(torch.bfloat16)
    bf16["model.output_attn_res_norm.weight"] = rng.norm_weight(hidden).to(
        torch.bfloat16)
    lin("model.output_attn_res_proj.weight", 1, hidden)
    lin("lm_head.weight", cfg["vocab_size"], hidden)

    mxfp4 = {k: v for k, v in bf16.items() if k not in mxfp4_removed}
    mxfp4.update(mxfp4_only)
    return bf16, mxfp4


def write_checkpoint(out_dir: str,
                     state_dict: dict,
                     quantized: bool,
                     remote_code_dir: str | None,
                     cfg=None):
    os.makedirs(out_dir, exist_ok=True)
    config = dict(cfg or TINY_TEXT_CONFIG)
    if quantized:
        config["quantization_config"] = QUANTIZATION_CONFIG
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "generation_config.json"), "w") as f:
        json.dump({"do_sample": False}, f)
    save_file(state_dict, os.path.join(out_dir, "model.safetensors"))
    if remote_code_dir:
        for fname in ("configuration_kimi_k3.py", "modeling_kimi_linear.py"):
            shutil.copy(os.path.join(remote_code_dir, fname),
                        os.path.join(out_dir, fname))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="/mnt/disks/persist/models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--remote-code-dir",
        default=None,
        help="Directory holding configuration_kimi_k3.py and "
        "modeling_kimi_linear.py (e.g. a moonshotai/Kimi-K3 snapshot) to copy "
        "into the checkpoints so transformers trust_remote_code loading "
        "works. Omit to skip.")
    args = parser.parse_args()

    bf16, mxfp4 = build_state_dicts(args.seed)
    bf16_dir = os.path.join(args.output_root, "k3-tiny")
    mxfp4_dir = os.path.join(args.output_root, "k3-tiny-mxfp4")
    write_checkpoint(bf16_dir, bf16, False, args.remote_code_dir)
    write_checkpoint(mxfp4_dir, mxfp4, True, args.remote_code_dir)

    sp_cfg = softplus_gate_config()
    sp, _ = build_state_dicts(args.seed, cfg=sp_cfg)
    sp_dir = os.path.join(args.output_root, "k3-tiny-softplus")
    write_checkpoint(sp_dir, sp, False, args.remote_code_dir, cfg=sp_cfg)

    for label, d, sd in (("bf16", bf16_dir, bf16), ("mxfp4", mxfp4_dir, mxfp4),
                         ("softplus", sp_dir, sp)):
        total = sum(v.numel() * v.element_size() for v in sd.values())
        print(f"k3-tiny ({label}): {d}  "
              f"{len(sd)} tensors, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
