# MOSS-Image Training Utilities

import argparse
import gc
import math
import os
import time
from typing import Optional

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm

from library.device_utils import init_ipex, clean_memory_on_device, synchronize_device
from library import moss_image_models, train_util

init_ipex()

from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def add_moss_training_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--moss_text_encoder",
        type=str,
        default=None,
        help="Qwen3.5-9B text encoder: MOSS-Image pipeline folder or its text_encoder folder (bf16, HF format)."
        " Defaults to --pretrained_model_name_or_path when that is a pipeline folder.",
    )
    parser.add_argument(
        "--moss_tokenizer",
        type=str,
        default=None,
        help="Tokenizer folder. Defaults to the tokenizer folder next to the text encoder.",
    )
    # --vae (base parser): pipeline folder, vae folder, or single FLUX.2 small-decoder file;
    # defaults to --pretrained_model_name_or_path when that is a pipeline folder
    parser.add_argument(
        "--moss_te_overlay_dir",
        type=str,
        default=None,
        help="Folder with the transformers 5 overlay used by the text encoder worker (installed automatically if missing).",
    )
    parser.add_argument(
        "--moss_max_token_length",
        type=int,
        default=512,
        help="Maximum caption length in Qwen3.5 tokens, including the chat template (upstream allows up to 4096).",
    )
    parser.add_argument(
        "--moss_vae_sample_posterior",
        action="store_true",
        help="Sample the VAE posterior when caching latents instead of using its mode.",
    )
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=1.0,
        help="Timestep distribution shift for rectified flow training (MOSS-Image trains with 1.0)",
    )
    parser.add_argument(
        "--timestep_sampling",
        type=str,
        default="moss",
        choices=["moss", "sigma", "uniform", "sigmoid", "shift", "flux_shift"],
        help="Timestep sampling method. 'moss' is the MOSS-Image recipe: sigma = 1 - sigmoid(mu + s*N(0,1)),"
        " truncated at --moss_ts_cutoff and rescaled, mixed with a uniform distribution.",
    )
    parser.add_argument("--sigmoid_scale", type=float, default=1.0, help="Scale factor for sigmoid timestep sampling")
    parser.add_argument("--moss_ts_mu", type=float, default=-1.09, help="'moss' timestep sampling: logit-normal location")
    parser.add_argument("--moss_ts_s", type=float, default=0.95, help="'moss' timestep sampling: logit-normal scale")
    parser.add_argument("--moss_ts_cutoff", type=float, default=0.95, help="'moss' timestep sampling: truncation point (1.0 = none)")
    parser.add_argument("--moss_ts_uniform_mix", type=float, default=0.01, help="'moss' timestep sampling: uniform mixture weight")
    parser.add_argument(
        "--moss_attn_backend",
        type=str,
        default="auto",
        choices=["auto", "cudnn", "efficient", "flash", "math", "torch"],
        help="Attention kernel. 'auto'/'torch': PyTorch's default. 'cudnn' is ~2.4x faster on Windows; it needs"
        " cuDNN >= 9.24 (torch 2.14.1+cu130): cuDNN 9.7.1 (torch 2.7.1) produced NaN gradients on real training inputs.",
    )
    parser.add_argument(
        "--moss_compile",
        type=str,
        default="none",
        choices=["none", "situ", "blocks"],
        help="torch.compile (needs Triton): 'situ' fuses the SiTU activation, 'blocks' compiles each joint block"
        " (fastest; ~16 s one-time compile). On an RTX 4090 at 1024px: none 2.58 s/img, situ 2.11, blocks 1.94."
        " With --blocks_to_swap, 'blocks' falls back to 'situ'.",
    )
    parser.add_argument(
        "--moss_int8",
        action="store_true",
        help="Quantize the frozen joint-block projections to ConvRot INT8 (the recipe of the ComfyUI *_int8_convrot"
        " files) and run them through cuBLAS int8. Faster on GeForce GPUs with --moss_compile blocks (slower"
        " without it), ~4.5 GB less VRAM. Not compatible with --blocks_to_swap.",
    )
    parser.add_argument(
        "--moss_int8_bwd",
        type=str,
        default="bf16",
        choices=["bf16", "int8"],
        help="Backward of the --moss_int8 layers: 'bf16' dequantizes the weight (exact gradient through the int8"
        " base), 'int8' also quantizes the incoming gradient for the int8 GEMM (faster, noisier).",
    )
    parser.add_argument(
        "--moss_nonfinite_patience",
        type=int,
        default=3,
        help="Micro-batches with non-finite loss or LoRA gradients are skipped (never applied to the weights);"
        " this many in a row aborts training as diverged. 1 = abort on the first one.",
    )
    parser.add_argument(
        "--moss_unchecked_blocks",
        type=int,
        default=0,
        help="With --gradient_checkpointing, skip checkpointing for the last N of the 42 blocks: faster, more VRAM.",
    )
    parser.add_argument(
        "--moss_fp32_modulation",
        action="store_true",
        help="Keep the timestep MLP and modulation projections (time_embedding, shared_block_modulation,"
        " video_head_modulation) in fp32. They are frozen; this avoids bf16 rounding of their outputs.",
    )
    parser.add_argument(
        "--moss_sample_uncond_skip_layers",
        type=str,
        default="22",
        help="Comma-separated joint blocks skipped in the negative CFG pass of sample images (official pipeline default: 22)."
        " 'none' (or empty) for plain CFG.",
    )
    parser.add_argument(
        "--moss_lora_save_format",
        type=str,
        default="both",
        choices=["split", "fused", "both"],
        help="Attention LoRA keys to save. 'split' targets proj.q/k/v (diffusers/HF and ComfyUI models), 'fused' targets"
        " proj.qkv (fused single-file checkpoints such as Forge/ComfyUI exports). 'both' writes both; each loader"
        " only applies the keys that match its model, so nothing is applied twice.",
    )


class MossTimestepSampler:
    """MOSS-Image recipe sampler over the discrete grid sigma_j = j / steps, j = 1..steps.

    S = 1 - sigmoid(mu + s * Z) restricted to [0, cutoff] and rescaled to [0, 1], mixed with a uniform pmf.
    Follows the MOSS-Image technical report's stage-3 parameters (mu=-1.09, s=0.95, cutoff=0.95, uniform mix 0.01).
    """

    def __init__(self, mu: float, s: float, cutoff: Optional[float], uniform_mix: float, shift: float = 1.0, steps: int = 1000):
        a = 1.0 if cutoff is None else float(cutoff)
        j = torch.arange(steps + 1, dtype=torch.float64)
        x = (a * j / steps).clamp(0.0, 1.0)
        # P(S <= x) = P(Z >= (logit(1 - x) - mu) / s), computed with erfc on both tails for precision
        u = (torch.log1p(-x) - torch.log(x) - mu) / (s * math.sqrt(2.0))
        cdf = torch.where(x <= 0, 0.0, torch.where(x >= 1, 1.0, 0.5 * torch.erfc(u)))
        sf = torch.where(x <= 0, 1.0, torch.where(x >= 1, 0.0, 0.5 * torch.erfc(-u)))
        lower = cdf[1:] <= 0.5
        mass = torch.where(lower, cdf[1:] - cdf[:-1], sf[:-1] - sf[1:])
        mass = mass / cdf[-1].clamp_min(torch.finfo(torch.float64).tiny)
        pmf = (1.0 - uniform_mix) * mass + uniform_mix / steps
        self.cdf = (pmf / pmf.sum()).cumsum(0).float()
        self.cdf[-1] = 1.0
        grid = torch.arange(1, steps + 1, dtype=torch.float32) / steps
        self.sigmas = grid if shift == 1.0 else shift * grid / (1.0 + (shift - 1.0) * grid)

    def sample(self, batch: int, device) -> torch.Tensor:
        u = torch.rand(batch)
        idx = torch.searchsorted(self.cdf, u).clamp_max(self.cdf.numel() - 1)
        return self.sigmas[idx].to(device)


def compute_loss_weighting_for_moss(weighting_scheme: str, sigmas: torch.Tensor) -> torch.Tensor:
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


# region sampling


def get_sampling_sigmas(num_steps: int, shift: float = 5.0, training_shift: float = 1.0, num_train_timesteps: int = 1000) -> torch.Tensor:
    """Inference sigmas of the official MOSSImageScheduler: shifted grid snapped to the nearest training sigma."""

    def grid(count, s):
        lower = s / (num_train_timesteps + s - 1)
        values = torch.linspace(1, lower, count, dtype=torch.float64)
        return s * values / (1 + (s - 1) * values)

    sigmas = grid(num_steps, shift)
    training = grid(num_train_timesteps, training_shift).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=num_train_timesteps - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat((sigmas, sigmas.new_zeros(1))).float()


@torch.no_grad()
def do_sample(
    model: moss_image_models.MossImageTransformer,
    height: int,
    width: int,
    seed: Optional[int],
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    neg_embeds: Optional[torch.Tensor],
    neg_mask: Optional[torch.Tensor],
    steps: int,
    guidance_scale: float,
    flow_shift: float,
    device: torch.device,
    uncond_skip_layers: Optional[list[int]] = None,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    # sample in fp32: promoting bf16 noise afterwards cannot restore its precision
    latents = torch.randn((1, model.latent_channels, height // 16, width // 16), generator=generator, dtype=torch.float32).to(device)
    sigmas = get_sampling_sigmas(steps, flow_shift).to(device)
    use_cfg = guidance_scale != 1.0 and neg_embeds is not None

    for i in tqdm(range(len(sigmas) - 1), desc="Sampling"):
        t = (sigmas[i] * 1000).reshape(1)
        prediction = model(latents, t, prompt_embeds, prompt_mask).float()
        if use_cfg:
            # official pipeline: the negative branch skips block 22 by default
            uncond = model(latents, t, neg_embeds, neg_mask, skip_layers=uncond_skip_layers).float()
            prediction = uncond + guidance_scale * (prediction - uncond)
        latents = latents + (sigmas[i + 1] - sigmas[i]) * prediction
    return latents


def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch,
    steps,
    model,
    vae,
    sample_prompts_te_outputs: dict,
):
    if steps == 0:
        if not args.sample_at_first:
            return
    else:
        if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
            return
        if args.sample_every_n_epochs is not None:
            if epoch is None or epoch % args.sample_every_n_epochs != 0:
                return
        else:
            if steps % args.sample_every_n_steps != 0 or epoch is not None:
                return

    if not sample_prompts_te_outputs:
        logger.error("No cached text encoder outputs for sample prompts; skipping sampling")
        return

    logger.info(f"Generating sample images at step {steps}")
    model = accelerator.unwrap_model(model)
    model.switch_block_swap_for_inference()

    prompts = train_util.load_prompts(args.sample_prompts)
    save_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(save_dir, exist_ok=True)

    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    was_training = model.training
    model.eval()
    with torch.no_grad(), accelerator.autocast():
        for prompt_dict in prompts:
            model.prepare_block_swap_before_forward()
            _sample_image_inference(accelerator, args, model, vae, save_dir, prompt_dict, epoch, steps, sample_prompts_te_outputs)
    model.train(was_training)

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    model.switch_block_swap_for_training()
    clean_memory_on_device(accelerator.device)


def _sample_image_inference(accelerator, args, model, vae, save_dir, prompt_dict, epoch, steps, sample_prompts_te_outputs):
    prompt = prompt_dict.get("prompt", "")
    negative_prompt = prompt_dict.get("negative_prompt", "")
    sample_steps = prompt_dict.get("sample_steps", 30)
    width = prompt_dict.get("width", 1024)
    height = prompt_dict.get("height", 1024)
    scale = prompt_dict.get("scale", 3.0)
    seed = prompt_dict.get("seed")
    flow_shift = float(prompt_dict.get("flow_shift", 5.0))  # "--fs" arrives as a string

    height = max(64, height - height % 16)
    width = max(64, width - width % 16)
    logger.info(f"  prompt: {prompt}, size: {width}x{height}, steps: {sample_steps}, scale: {scale}, flow_shift: {flow_shift}, seed: {seed}")

    def get_cond(p):
        embeds, mask = sample_prompts_te_outputs[p]
        return embeds.to(accelerator.device, dtype=model.dtype), mask.to(accelerator.device)

    if prompt not in sample_prompts_te_outputs:
        logger.warning(f"Prompt was not cached, skipping: {prompt}")
        return
    embeds, mask = get_cond(prompt)
    neg_embeds = neg_mask = None
    if scale != 1.0 and negative_prompt in sample_prompts_te_outputs:
        neg_embeds, neg_mask = get_cond(negative_prompt)

    clean_memory_on_device(accelerator.device)
    skip_arg = str(getattr(args, "moss_sample_uncond_skip_layers", "") or "")
    skip = [int(i) for i in skip_arg.split(",") if i.strip().isdigit()]  # "" or "none" = plain CFG
    skip = [i for i in skip if 0 <= i < len(model.layers)]
    latents = do_sample(
        model, height, width, seed, embeds, mask, neg_embeds, neg_mask, sample_steps, scale, flow_shift, accelerator.device, skip
    )

    gc.collect()
    synchronize_device(accelerator.device)
    clean_memory_on_device(accelerator.device)
    org_vae_device = vae.device
    vae.to(accelerator.device)
    with torch.autocast(device_type=accelerator.device.type, enabled=False):
        decoded = vae.decode_to_pixels(latents)
    vae.to(org_vae_device)
    clean_memory_on_device(accelerator.device)

    image = torch.clamp((decoded.float() + 1.0) / 2.0, min=0.0, max=1.0)[0]
    decoded_np = (255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)).astype(np.uint8)
    image = Image.fromarray(decoded_np)

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i = prompt_dict.get("enum", 0)
    img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
    image.save(os.path.join(save_dir, img_filename))

    if "wandb" in [tracker.name for tracker in accelerator.trackers]:
        wandb_tracker = accelerator.get_tracker("wandb")
        import wandb

        wandb_tracker.log({f"sample_{i}": wandb.Image(image, caption=prompt)}, commit=False)


# endregion
