# MOSS-Image VAE: FLUX.2 AutoencoderKL (small decoder) with 2x2 patchify and BatchNorm latent normalization.
# Encode: posterior [B, 32, H/8, W/8] -> patchify -> [B, 128, H/16, W/16] -> (z - running_mean) / sqrt(running_var + eps).
# Decode is the exact inverse. Reference: OpenMOSS-Team/MOSS-Image pipeline_moss_image.py (Apache-2.0).

import os

import torch
from torch import nn

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

# vae/config.json of OpenMOSS-Team/MOSS-Image-Base, used to build the model from a single (BFL-format) file
MOSS_VAE_CONFIG = {
    "act_fn": "silu",
    "batch_norm_eps": 0.0001,
    "batch_norm_momentum": 0.1,
    "block_out_channels": [128, 256, 512, 512],
    "decoder_block_out_channels": [96, 192, 384, 384],
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "force_upcast": True,
    "in_channels": 3,
    "latent_channels": 32,
    "layers_per_block": 2,
    "mid_block_add_attention": True,
    "norm_num_groups": 32,
    "out_channels": 3,
    "patch_size": [2, 2],
    "sample_size": 1024,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "use_post_quant_conv": True,
    "use_quant_conv": True,
}


def patchify(z: torch.Tensor) -> torch.Tensor:
    """[B, C, 2h, 2w] -> [B, 4C, h, w]; channel = c * 4 + i * 2 + j for pixel (2y+i, 2x+j)."""
    b, c, hh, ww = z.shape
    z = z.reshape(b, c, hh // 2, 2, ww // 2, 2).permute(0, 1, 3, 5, 2, 4)
    return z.reshape(b, c * 4, hh // 2, ww // 2)


def unpatchify(z: torch.Tensor) -> torch.Tensor:
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return z.reshape(b, c // 4, 2 * h, 2 * w)


class MossImageVAE(nn.Module):
    """Wraps diffusers AutoencoderKLFlux2. Always computes in fp32 (the reference VAE is force_upcast)."""

    def __init__(self, vae, sample_posterior: bool = False):
        super().__init__()
        self.vae = vae
        self.sample_posterior = sample_posterior
        eps = vae.config.batch_norm_eps
        self.register_buffer("latent_mean", vae.bn.running_mean.detach().clone().float().view(1, -1, 1, 1), persistent=False)
        self.register_buffer(
            "latent_std", (vae.bn.running_var.detach().float() + eps).sqrt().view(1, -1, 1, 1), persistent=False
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @torch.no_grad()
    def encode_pixels_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 3, H, W] in [-1, 1] -> normalized packed latents [B, 128, H/16, W/16] (fp32)."""
        dist = self.vae.encode(images.to(self.device, dtype=self.dtype)).latent_dist
        z = dist.sample() if self.sample_posterior else dist.mode()
        z = patchify(z.float())
        return (z - self.latent_mean.to(z.device)) / self.latent_std.to(z.device)

    @torch.no_grad()
    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        """normalized packed latents [B, 128, h, w] -> images [B, 3, 16h, 16w] in [-1, 1]."""
        z = latents.to(self.device, dtype=torch.float32)
        z = z * self.latent_std.to(z.device) + self.latent_mean.to(z.device)
        z = unpatchify(z).to(self.dtype)
        return self.vae.decode(z, return_dict=False)[0].clamp(-1, 1)


def _convert_bfl_vae_state_dict(state_dict: dict, config: dict) -> dict:
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

    src = {}
    for key, value in state_dict.items():
        if key.startswith("vae."):
            key = key[len("vae.") :]
        if key.startswith("encoder.quant_conv."):
            key = "quant_conv." + key[len("encoder.quant_conv.") :]
        elif key.startswith("decoder.post_quant_conv."):
            key = "post_quant_conv." + key[len("decoder.post_quant_conv.") :]
        src[key] = value
    converted = convert_ldm_vae_checkpoint(src, config)
    for key, value in src.items():
        if key.startswith("bn."):
            converted[key] = value
    return converted


def load_moss_image_vae(path: str, device="cpu", sample_posterior: bool = False) -> MossImageVAE:
    """Load the MOSS-Image VAE from a pipeline folder, a diffusers vae folder, or a single BFL/ComfyUI-format file."""
    from diffusers import AutoencoderKLFlux2

    if os.path.isdir(path):
        folder = os.path.join(path, "vae") if os.path.isfile(os.path.join(path, "vae", "config.json")) else path
        logger.info(f"Loading MOSS-Image VAE from {folder}")
        vae = AutoencoderKLFlux2.from_pretrained(folder, torch_dtype=torch.float32)
    else:
        from safetensors.torch import load_file

        logger.info(f"Loading MOSS-Image VAE from single file {path}")
        state_dict = load_file(path)
        if "bn.running_mean" not in state_dict and "vae.bn.running_mean" not in state_dict:
            raise ValueError("VAE file has no BatchNorm statistics (bn.running_mean); it is not a FLUX.2 VAE")
        vae = AutoencoderKLFlux2(**MOSS_VAE_CONFIG)
        if "encoder.conv_in.weight" in state_dict or "vae.encoder.conv_in.weight" in state_dict:
            if "decoder.up_blocks.0.resnets.0.conv1.weight" not in state_dict:
                state_dict = _convert_bfl_vae_state_dict(state_dict, dict(vae.config))
        decoder_in = state_dict["decoder.conv_in.weight"].shape[0]
        if decoder_in != MOSS_VAE_CONFIG["decoder_block_out_channels"][-1]:
            # full FLUX.2 decoder: rebuild with the default (wide) decoder
            config = dict(MOSS_VAE_CONFIG)
            config.pop("decoder_block_out_channels")
            vae = AutoencoderKLFlux2(**config)
            logger.warning("VAE has the full FLUX.2 decoder; MOSS-Image ships the small decoder (flux2-small-decoder)")
        logger.warning(
            "Single-file FLUX.2 VAEs can differ slightly from the official MOSS-Image VAE in encoder weights and "
            "BatchNorm statistics. Prefer the vae folder of OpenMOSS-Team/MOSS-Image-Base for caching latents."
        )
        vae.load_state_dict(state_dict, strict=True)
        vae.to(torch.float32)

    vae.eval().requires_grad_(False)
    return MossImageVAE(vae, sample_posterior=sample_posterior).to(device)
