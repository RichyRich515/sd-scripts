# MOSS-Image LoRA training script
#
# The Qwen3.5-9B text encoder needs transformers>=5.3, so it runs in a worker subprocess (library/moss_te_client.py)
# and its outputs are always cached before the DiT is loaded. Training itself only needs the DiT and cached data.

import argparse
import json
import os
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from accelerate import Accelerator
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import (
    flux_train_utils,
    moss_image_models,
    moss_image_vae,
    moss_te_client,
    moss_train_utils,
    sd3_train_utils,
    strategy_base,
    strategy_moss,
    train_util,
)
from library.custom_train_functions import (
    apply_snr_weight_for_flow_matching,
    maybe_apply_antithetic_noise_pairing,
)
import train_network
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def _is_pipeline_folder(path: Optional[str]) -> bool:
    return path is not None and os.path.isdir(path) and os.path.isfile(os.path.join(path, "model_index.json"))


class MossNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.uncond_te_outputs = None  # empty-prompt embedding used for caption dropout
        self.timestep_sampler = None
        # flow matching; get_noise_pred_and_target returns timesteps as sigmas in [0, 1]
        self.hf_prediction_mode = "flow"
        self.hf_timesteps_in_sigma = True
        self.anchor_prediction_mode = "flow"
        self.anchor_timesteps_in_sigma = True
        self._ot_logged = False

    def get_adaptive_model_type(self, args) -> str:
        return "flow_matching"

    def build_adaptive_model_fn(self, unet, accelerator, weight_dtype):
        model: moss_image_models.MossImageTransformer = unet
        text_conds = self._adaptive_last_text_conds
        prompt_embeds = text_conds[0].to(accelerator.device, dtype=weight_dtype)
        attn_mask = text_conds[1].to(accelerator.device)

        def model_fn(noisy_latents, timesteps, wdtype):
            n = noisy_latents.shape[0]
            embeds = prompt_embeds if prompt_embeds.shape[0] == n else prompt_embeds[:1].expand(n, -1, -1).contiguous()
            mask = attn_mask if attn_mask.shape[0] == n else attn_mask[:1].expand(n, -1).contiguous()
            embeds, mask = self.apply_uncond_for_dropped(embeds, mask)
            with torch.no_grad(), accelerator.autocast():
                return model(noisy_latents.to(wdtype), timesteps.float(), embeds, mask)

        return model_fn

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        if args.fp8_base or args.fp8_base_unet:
            logger.warning("fp8_base and fp8_base_unet are not supported for MOSS-Image.")
            args.fp8_base = False
            args.fp8_base_unet = False
        args.fp8_scaled = False

        if not args.cache_text_encoder_outputs:
            logger.info("MOSS-Image: the Qwen3.5 text encoder runs out of process, enabling --cache_text_encoder_outputs")
            args.cache_text_encoder_outputs = True
        if not args.network_train_unet_only:
            logger.info("MOSS-Image: the text encoder is never trained, enabling --network_train_unet_only")
            args.network_train_unet_only = True

        assert train_dataset_group.is_text_encoder_output_cacheable(
            cache_supports_dropout=True
        ), "MOSS-Image caches Text Encoder outputs: shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used (use --cache_caption_variants for all but token_warmup_step)"
        if getattr(args, "aug_refresh_epochs", 0) and getattr(args, "cache_caption_variants", 0):
            raise ValueError(
                "--aug_refresh_epochs with caption variants re-encodes captions during training, which would load the "
                "18 GB text encoder next to the DiT. Cache caption variants up front instead (aug_refresh_epochs=0)."
            )

        assert (
            args.blocks_to_swap is None or args.blocks_to_swap == 0
        ) or not args.cpu_offload_checkpointing, "blocks_to_swap is not supported with cpu_offload_checkpointing"
        if args.moss_int8 and args.blocks_to_swap:
            raise ValueError("--moss_int8 cannot be combined with --blocks_to_swap (int8 already saves ~5 GB of VRAM)")

        if args.timestep_sampling == "moss":
            self.timestep_sampler = moss_train_utils.MossTimestepSampler(
                args.moss_ts_mu, args.moss_ts_s, args.moss_ts_cutoff, args.moss_ts_uniform_mix, shift=args.discrete_flow_shift
            )
            logger.info(
                f"MOSS-Image timestep sampling: mu={args.moss_ts_mu}, s={args.moss_ts_s}, cutoff={args.moss_ts_cutoff}, "
                f"uniform_mix={args.moss_ts_uniform_mix}, shift={args.discrete_flow_shift}"
            )

        # 8x VAE downscale, 2x2 packing
        train_dataset_group.verify_bucket_reso_steps(16)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(16)

    def text_encoder_path(self, args) -> str:
        if args.moss_text_encoder is not None:
            return args.moss_text_encoder
        if _is_pipeline_folder(args.pretrained_model_name_or_path):
            return args.pretrained_model_name_or_path
        raise ValueError("--moss_text_encoder is required unless --pretrained_model_name_or_path is a MOSS-Image pipeline folder")

    def load_target_model(self, args, weight_dtype, accelerator):
        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0

        text_encoder = moss_te_client.MossTextEncoderProxy(
            self.text_encoder_path(args),
            tokenizer_path=args.moss_tokenizer,
            overlay_dir=args.moss_te_overlay_dir or moss_te_client.DEFAULT_OVERLAY_DIR,
            max_length=args.moss_max_token_length,
            encode_device=str(accelerator.device),
        )

        vae_path = args.vae
        if vae_path is None:
            if not _is_pipeline_folder(args.pretrained_model_name_or_path):
                raise ValueError("--vae is required unless --pretrained_model_name_or_path is a MOSS-Image pipeline folder")
            vae_path = args.pretrained_model_name_or_path
        vae = moss_image_vae.load_moss_image_vae(vae_path, device="cpu", sample_posterior=args.moss_vae_sample_posterior)

        return "moss_image", [text_encoder], vae, None  # DiT is loaded lazily, after text encoder caching

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders) -> tuple[nn.Module, list[nn.Module]]:
        loading_device = "cpu" if self.is_swapping_blocks else accelerator.device
        moss_image_models.set_attention_backend(args.moss_attn_backend)
        model = moss_image_models.load_moss_image_model(args.pretrained_model_name_or_path, loading_device, weight_dtype)
        if args.moss_int8:
            model.quantize_blocks_int8(args.moss_int8_bwd)

        if self.is_swapping_blocks:
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)

        compile_mode = args.moss_compile
        if compile_mode == "blocks" and self.is_swapping_blocks:
            # swapped blocks change weight devices between steps, which would force recompiles
            logger.warning("--moss_compile blocks is not used with block swap; compiling the SiTU activation only")
            compile_mode = "situ"
        model.set_compile_mode(compile_mode)

        return model, text_encoders

    def cast_vae(self, args):
        return False  # the FLUX.2 VAE runs in fp32 (force_upcast)

    def get_tokenize_strategy(self, args):
        return strategy_moss.MossTokenizeStrategy(args.moss_max_token_length)

    def get_tokenizers(self, tokenize_strategy):
        return [None]  # tokenization happens in the text encoder worker

    def get_latents_caching_strategy(self, args):
        return strategy_moss.MossLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check,
            cache_dtype=getattr(args, "cache_latents_dtype", "auto"),
        )

    def get_text_encoding_strategy(self, args):
        return strategy_moss.MossTextEncodingStrategy()

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        if hasattr(network, "save_format"):
            network.save_format = args.moss_lora_save_format

    def install_nonfinite_grad_guard(self, network):
        """Non-finite guard, gradient half (loss half in process_batch). The base loop has none, so a single NaN/inf
        gradient would reach AdamW and poison every LoRA weight from then on. Non-finite gradient entries are zeroed
        as they are accumulated, before clipping and the optimizer step see them. Installed on the first batch:
        post_process_network runs before apply_to registers the LoRA modules, when the network has no parameters."""
        self._nonfinite_grad_events = 0

        def zero_nonfinite_grad(param: torch.Tensor):
            if param.grad is not None and not torch.isfinite(param.grad).all():
                self._nonfinite_grad_events += 1
                if self._nonfinite_grad_events <= 5 or self._nonfinite_grad_events % 100 == 0:
                    logger.warning(f"MOSS-Image: non-finite LoRA gradient zeroed (event {self._nonfinite_grad_events})")
                param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        params = [p for p in network.parameters() if p.requires_grad]
        for param in params:
            param.register_post_accumulate_grad_hook(zero_nonfinite_grad)
        logger.info(f"MOSS-Image: non-finite gradient guard on {len(params)} LoRA parameters")
        self._grad_guard_installed = True

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        return None  # outputs are always cached

    def get_text_encoder_outputs_caching_strategy(self, args):
        return strategy_moss.MossTextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check, False,
            cache_dtype=getattr(args, "cache_text_encoder_outputs_dtype", "auto"),
        )

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        text_encoder: moss_te_client.MossTextEncoderProxy = text_encoders[0]
        vae.to("cpu")
        clean_memory_on_device(accelerator.device)
        text_encoder.to(accelerator.device)

        dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

        if self.uncond_te_outputs is None:
            prompts = [""]
            if args.sample_prompts is not None:
                for prompt_dict in train_util.load_prompts(args.sample_prompts):
                    for p in (prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")):
                        if p not in prompts:
                            prompts.append(p)
            rows = self.encode_prompts_with_cache(args, text_encoder, prompts)
            outputs = {p: (row.unsqueeze(0), torch.ones(1, row.shape[0], dtype=torch.long)) for p, row in zip(prompts, rows)}
            self.uncond_te_outputs = outputs[""]
            self.sample_prompts_te_outputs = outputs

        accelerator.wait_for_everyone()
        # stops the worker and frees its VRAM before the DiT is loaded
        text_encoder.to("cpu")
        clean_memory_on_device(accelerator.device)

    def encode_prompts_with_cache(self, args, text_encoder: moss_te_client.MossTextEncoderProxy, prompts: list[str]) -> list[torch.Tensor]:
        """Encode the empty prompt and sample prompts, reusing a small cache in output_dir so that runs whose
        dataset outputs are already cached do not have to start the 18 GB text encoder at all."""
        from safetensors.torch import load_file, save_file

        cache_path = os.path.join(args.output_dir, ".moss_prompt_te_cache.safetensors")
        source = f"{os.path.abspath(text_encoder.model_path)}|{args.moss_max_token_length}"
        cached = {}
        if os.path.isfile(cache_path):
            try:
                from safetensors import safe_open

                with safe_open(cache_path, "pt") as f:
                    metadata = f.metadata() or {}
                if metadata.get("source") == source:
                    tensors = load_file(cache_path)
                    keys = json.loads(metadata["prompts"])
                    cached = {p: tensors[f"emb_{i}"] for i, p in enumerate(keys)}
            except Exception as e:
                logger.warning(f"ignoring unreadable prompt cache {cache_path}: {e}")

        missing = [p for p in prompts if p not in cached]
        if missing:
            logger.info(f"encode Text Encoder outputs for {len(missing)} prompt(s) (empty prompt and sample prompts)")
            cached.update(zip(missing, text_encoder.encode(missing)))
            os.makedirs(args.output_dir, exist_ok=True)
            keys = list(cached.keys())
            save_file(
                {f"emb_{i}": cached[p].contiguous() for i, p in enumerate(keys)},
                cache_path,
                metadata={"source": source, "prompts": json.dumps(keys)},
            )
        return [cached[p] for p in prompts]

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        moss_train_utils.sample_images(accelerator, args, epoch, global_step, unet, vae, self.sample_prompts_te_outputs)

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        return sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)

    def encode_images_to_latents(self, args, vae, images):
        return vae.encode_pixels_to_latents(images)

    def shift_scale_latents(self, args, latents):
        return latents  # already normalized by the VAE's BatchNorm statistics

    def apply_uncond_for_dropped(self, prompt_embeds: torch.Tensor, attn_mask: torch.Tensor):
        """Samples whose mask is all zero (caption dropout) get the empty-prompt embedding, as at inference."""
        attn_mask = attn_mask.to(prompt_embeds.device)
        dropped = attn_mask.sum(1) == 0
        if not bool(dropped.any()):
            return prompt_embeds, attn_mask
        uncond_embeds, uncond_mask = self.uncond_te_outputs
        length = max(prompt_embeds.shape[1], uncond_embeds.shape[1])
        if prompt_embeds.shape[1] < length:
            pad = length - prompt_embeds.shape[1]
            prompt_embeds = torch.nn.functional.pad(prompt_embeds, (0, 0, 0, pad))
            attn_mask = torch.nn.functional.pad(attn_mask, (0, pad))
        else:
            prompt_embeds = prompt_embeds.clone()
            attn_mask = attn_mask.clone()
        u_len = uncond_embeds.shape[1]
        for i in torch.nonzero(dropped).flatten().tolist():
            prompt_embeds[i] = 0
            prompt_embeds[i, :u_len] = uncond_embeds[0].to(prompt_embeds.device, prompt_embeds.dtype)
            attn_mask[i] = 0
            attn_mask[i, :u_len] = 1
        return prompt_embeds, attn_mask

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        text_encoder_masks,
        unet,
        network,
        weight_dtype,
        train_unet,
        fixed_timesteps=None,
        is_train=True,
    ):
        model: moss_image_models.MossImageTransformer = unet

        adaptive_fixed_timesteps = fixed_timesteps
        if is_train and self.adaptive_manager is not None and fixed_timesteps is None:
            adaptive_fixed_timesteps = self.adaptive_manager.sample_timesteps(latents, noise_scheduler.config.num_train_timesteps)
            if self._adaptive_update_pending:
                self._adaptive_last_latents = latents.detach()
                self._adaptive_last_args = args
        elif adaptive_fixed_timesteps is None and self.timestep_sampler is not None:
            sigmas = self.timestep_sampler.sample(latents.shape[0], latents.device)
            adaptive_fixed_timesteps = sigmas * noise_scheduler.config.num_train_timesteps

        noise = torch.randn_like(latents)

        if getattr(args, "flow_use_ot", False) and latents.size(0) > 1:
            with torch.no_grad():
                b_size = latents.size(0)
                _, (_, col_indices) = train_util.cosine_optimal_transport(latents.view(b_size, -1), noise.view(b_size, -1))
                noise = noise[col_indices.squeeze(0)]
            if not self._ot_logged:
                logger.info(f"[MOSS OT] First batch: noise reordered by cosine OT: {col_indices.squeeze(0).tolist()}")
                self._ot_logged = True

        noise = maybe_apply_antithetic_noise_pairing(args, noise, is_train=is_train)

        if is_train and self.adaptive_manager is not None and fixed_timesteps is None and self._adaptive_update_pending:
            self._adaptive_last_noise = noise.detach()

        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args,
            noise_scheduler,
            latents,
            noise,
            accelerator.device,
            weight_dtype,
            fixed_timesteps=adaptive_fixed_timesteps,
            is_train=is_train,
        )

        if is_train and getattr(self, "wavelet_masking_enabled", False):
            self._noisy_latents = noisy_model_input.detach()
        if is_train and self.hf_scale > 0.0:
            self._hf_noisy_latents = noisy_model_input.detach()
        if is_train and self.anchor_scale > 0.0:
            self._anchor_noisy_latents = noisy_model_input.detach()
        if is_train and self.adv_scale > 0.0:
            self._adv_noisy_latents = noisy_model_input.detach()

        self.apply_tlora_mask(timesteps)

        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)

        prompt_embeds, attn_mask = text_encoder_conds[:2]  # ignore caption_dropout_rate
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        prompt_embeds, attn_mask = self.apply_uncond_for_dropped(prompt_embeds, attn_mask.to(accelerator.device).long())
        if args.gradient_checkpointing:
            prompt_embeds.requires_grad_(True)

        # the model takes sigma * 1000; prefer the exact fp32 values over the weight_dtype-rounded copy
        model_timesteps = adaptive_fixed_timesteps.float() if adaptive_fixed_timesteps is not None else timesteps.float()
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            model_pred = model(noisy_model_input, model_timesteps, prompt_embeds, attn_mask)

        self.clear_tlora_mask_if_needed()

        timesteps = timesteps.float() / 1000.0  # sigmas in [0, 1] for the shared flow-matching losses

        latents = latents.to(torch.float64)
        noise = noise.to(torch.float64)
        target = noise - latents  # rectified flow velocity

        weighting = moss_train_utils.compute_loss_weighting_for_moss(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
        return model_pred, target, timesteps, weighting, noise

    def process_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy,
        tokenize_strategy,
        is_train=True,
        train_text_encoder=True,
        train_unet=True,
        edm2_model=None,
    ) -> torch.Tensor:
        """Caption dropout on cached text encoder outputs, plus the non-finite loss guard."""
        if is_train and not getattr(self, "_grad_guard_installed", False):
            self.install_nonfinite_grad_guard(network)
        if is_train:
            # the previous micro-batch is complete (loss and backward): judge it once
            bad = getattr(self, "_last_loss_nonfinite", False) or self._nonfinite_grad_events > getattr(self, "_grad_events_seen", 0)
            self.update_nonfinite_streak(args, bad)
            self._grad_events_seen = self._nonfinite_grad_events
            self._last_loss_nonfinite = False

        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            caption_dropout_rates = text_encoder_outputs_list[-1]
            text_encoder_outputs_list = text_encoding_strategy.drop_cached_text_encoder_outputs(
                *text_encoder_outputs_list[:-1], caption_dropout_rates=caption_dropout_rates
            )
            batch["text_encoder_outputs_list"] = text_encoder_outputs_list + [caption_dropout_rates]

        result = super().process_batch(
            batch,
            text_encoders,
            unet,
            network,
            vae,
            noise_scheduler,
            vae_dtype,
            weight_dtype,
            accelerator,
            args,
            text_encoding_strategy,
            tokenize_strategy,
            is_train,
            train_text_encoder,
            train_unet,
            edm2_model,
        )

        # (loss for backward, pre-scaling loss and scaled loss for logging)
        loss = result[0]
        if is_train and not torch.isfinite(loss).all():
            # skip this micro-batch: a zero loss still attached to the LoRA weights gives zero gradients
            self._nonfinite_loss_events = getattr(self, "_nonfinite_loss_events", 0) + 1
            logger.warning(
                f"MOSS-Image: non-finite loss ({loss.item()}) skipped (event {self._nonfinite_loss_events}); "
                f"images: {[os.path.basename(p) for p in batch.get('absolute_paths') or []]}"
            )
            self._last_loss_nonfinite = True
            params = [p for p in network.parameters() if p.requires_grad]
            zero = sum(p.sum() for p in params) * 0.0
            result = (zero,) + tuple(zero.detach() if torch.is_tensor(r) else r for r in result[1:])
        return result

    def update_nonfinite_streak(self, args, nonfinite: bool):
        """Isolated non-finite micro-batches are skipped; a run of them means training diverged, so abort
        (like KohakuMossTrainer's NONFINITE_PATIENCE)."""
        self._nonfinite_streak = getattr(self, "_nonfinite_streak", 0) + 1 if nonfinite else 0
        patience = args.moss_nonfinite_patience
        if nonfinite and self._nonfinite_streak >= patience:
            raise RuntimeError(
                f"MOSS-Image: {self._nonfinite_streak} consecutive micro-batches had non-finite loss or gradients; "
                "training has diverged. The LoRA weights are still finite (bad updates were skipped): lower the "
                "learning rate or check the data, and resume from the last saved checkpoint (network_weights)."
            )

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        if args.min_snr_gamma:
            # timesteps are already sigmas in [0, 1]
            loss = apply_snr_weight_for_flow_matching(loss, timesteps, args.min_snr_gamma, soft=args.min_snr_gamma_soft)
        return loss

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec_dataclass(None, args, False, True, False, moss_image="base").to_metadata_dict()

    def update_metadata(self, metadata, args):
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift
        if args.timestep_sampling == "moss":
            metadata["ss_moss_ts_mu"] = args.moss_ts_mu
            metadata["ss_moss_ts_s"] = args.moss_ts_s
            metadata["ss_moss_ts_cutoff"] = args.moss_ts_cutoff
            metadata["ss_moss_ts_uniform_mix"] = args.moss_ts_uniform_mix
        metadata["ss_moss_max_token_length"] = args.moss_max_token_length
        metadata["ss_moss_fp32_modulation"] = args.moss_fp32_modulation
        metadata["ss_moss_compile"] = args.moss_compile
        metadata["ss_moss_attn_backend"] = args.moss_attn_backend
        metadata["ss_moss_int8"] = f"convrot (backward {args.moss_int8_bwd})" if args.moss_int8 else "off"

    def is_text_encoder_not_needed_for_training(self, args):
        return True

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        pass

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        if args.gradient_checkpointing and args.moss_unchecked_blocks > 0:
            unet.enable_gradient_checkpointing(cpu_offload=args.cpu_offload_checkpointing, unchecked_blocks=args.moss_unchecked_blocks)

        # after the base trainer's dtype cast of the DiT, before accelerate wraps it
        if args.moss_fp32_modulation:
            unet.modulation_to_fp32()
            logger.info("MOSS-Image: modulation layers kept in fp32")

        if not self.is_swapping_blocks:
            return super().prepare_unet_with_accelerator(args, accelerator, unet)

        model = accelerator.prepare(unet, device_placement=[not self.is_swapping_blocks])
        accelerator.unwrap_model(model).move_to_device_except_swap_blocks(accelerator.device)
        accelerator.unwrap_model(model).prepare_block_swap_before_forward()
        return model

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self.is_swapping_blocks:
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    moss_train_utils.add_moss_training_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    if args.network_module in ("networks.lora", "networks.lora_anima"):
        print(f"Override network module: {args.network_module} -> networks.lora_moss")
        args.network_module = "networks.lora_moss"

    # rectified flow: bypasses the CFM guard in train_network.py like Anima
    args._anima_model = True

    trainer = MossNetworkTrainer()
    trainer.train(args)
