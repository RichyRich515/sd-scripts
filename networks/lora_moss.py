# LoRA network module for MOSS-Image
#
# Reuses the Anima LoRA implementation with MOSS-Image targets: the 42 joint blocks (proj.q/k/v/o, ffn_up_gate,
# ffn_down) and optionally the two text refinement blocks (network_args "train_text_refiner=true").
# Names follow the kohya/ComfyUI convention (lora_unet_layers_0_proj_q), which ComfyUI and Forge map automatically.
#
# Fused export: single-file MOSS checkpoints (e.g. moss_image_sft_bf16.safetensors) store attention as one
# layers.N.proj.qkv linear. Forge/ComfyUI only apply LoRA keys that match the loaded model's modules, so split
# q/k/v keys would be silently ignored there. On save, q/k/v LoRAs are additionally (or only) exported as an exact
# fused proj_qkv LoRA: down = cat(down_q, down_k, down_v), up = blockdiag(up_q * s_q, up_k * s_k, up_v * s_v),
# alpha = fused rank. "both" keeps split and fused keys in one file; a loader applies whichever matches its model.

import os
import re
from typing import Dict, Optional

import torch

from networks import lora_anima
from networks.lora_anima import LoRAModule, LoRAInfModule  # noqa: F401  (re-exported for network loaders)
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

FUSED_QKV_PATTERN = re.compile(r"^(lora_unet_layers_\d+_proj_)([qkv])$")


class LoRANetwork(lora_anima.LoRANetwork):
    ANIMA_TARGET_REPLACE_MODULE = ["MossBlock"]
    ANIMA_ADAPTER_TARGET_REPLACE_MODULE = ["MossTextBlock"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = []  # Qwen3.5 runs in a worker process and is never trained
    DEFAULT_EXCLUDE_PATTERN = r".*(_modulation|_norm|final_).*"
    ADAPTER_LORA_NAME_MARKER = "text_embedder"

    save_format = "both"  # set by the trainer from --moss_lora_save_format

    def fused_qkv_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Build exact fused proj_qkv LoRA tensors from split q/k/v LoRA tensors."""
        groups: Dict[str, Dict[str, str]] = {}
        for key in state_dict:
            lora_name = key.split(".")[0]
            match = FUSED_QKV_PATTERN.match(lora_name)
            if match:
                groups.setdefault(match.group(1), {})[match.group(2)] = lora_name

        fused = {}
        for prefix, parts in groups.items():
            if set(parts) != {"q", "k", "v"}:
                continue
            downs, ups = [], []
            for part in ("q", "k", "v"):
                name = parts[part]
                down = state_dict[f"{name}.lora_down.weight"].float()
                up = state_dict[f"{name}.lora_up.weight"].float()
                alpha = float(state_dict[f"{name}.alpha"])
                downs.append(down)
                ups.append(up * (alpha / down.shape[0]))  # fold each part's scale into its up weights
            down = torch.cat(downs, 0)
            rank = down.shape[0]
            up = torch.block_diag(*ups)
            dtype = state_dict[f"{parts['q']}.lora_down.weight"].dtype
            fused[f"{prefix}qkv.lora_down.weight"] = down.to(dtype)
            fused[f"{prefix}qkv.lora_up.weight"] = up.to(dtype)
            fused[f"{prefix}qkv.alpha"] = torch.tensor(float(rank), dtype=dtype)  # scale 1: per-part scales are in up
        return fused

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()
        if dtype is not None:
            for key in list(state_dict.keys()):
                state_dict[key] = state_dict[key].detach().clone().to("cpu").to(dtype)

        if self.save_format in ("fused", "both"):
            fused = self.fused_qkv_state_dict(state_dict)
            if self.save_format == "fused":
                for key in list(state_dict.keys()):
                    if FUSED_QKV_PATTERN.match(key.split(".")[0]):
                        del state_dict[key]
            state_dict.update({k: v.to("cpu") for k, v in fused.items()})

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library import train_util

            if metadata is None:
                metadata = {}
            metadata["ss_moss_lora_save_format"] = self.save_format
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)


def _translate_kwargs(kwargs: dict) -> dict:
    kwargs = dict(kwargs)
    if "train_text_refiner" in kwargs:
        kwargs["train_llm_adapter"] = kwargs.pop("train_text_refiner")
    return kwargs


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoders: list,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    return lora_anima.create_network(
        multiplier, network_dim, network_alpha, vae, text_encoders, unet, neuron_dropout,
        network_class=LoRANetwork, **_translate_kwargs(kwargs),
    )


def create_network_from_weights(multiplier, file, ae, text_encoders, unet, weights_sd=None, for_inference=False, **kwargs):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
    # fused proj_qkv keys are an export format; the training model has split q/k/v
    weights_sd = {k: v for k, v in weights_sd.items() if not k.startswith(tuple(_fused_prefixes(weights_sd)))}
    return lora_anima.create_network_from_weights(
        multiplier, file, ae, text_encoders, unet, weights_sd=weights_sd, for_inference=for_inference,
        network_class=LoRANetwork, **_translate_kwargs(kwargs),
    )


def _fused_prefixes(weights_sd) -> set:
    return {k.split(".")[0] + "." for k in weights_sd if re.match(r"^lora_unet_layers_\d+_proj_qkv\.", k)}
