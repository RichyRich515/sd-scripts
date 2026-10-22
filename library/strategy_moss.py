# MOSS-Image Strategy Classes

import os
import random
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from library import train_util
from library.cache_utils import load_npz, save_npz
from library.moss_te_client import MossTextEncoderProxy
from library.strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy, variant_key

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class MossTokenizeStrategy(TokenizeStrategy):
    """Tokenization (chat template + Qwen tokenizer) happens in the text encoder worker; captions pass through as text."""

    def __init__(self, max_length: int = 512) -> None:
        self.max_length = max_length

    def tokenize(self, text: Union[str, List[str]]) -> List[List[str]]:
        return [[text] if isinstance(text, str) else list(text)]


class MossTextEncodingStrategy(TextEncodingStrategy):
    """Encodes captions with Qwen3.5 (last hidden state, unpadded per caption).

    Outputs are right padded to the longest caption: [prompt_embeds [B, L, 4096], attn_mask [B, L] int].
    A sample whose mask is all zero (caption dropout) is replaced with the empty-prompt embedding at train time.
    """

    def __init__(self) -> None:
        super().__init__()

    def encode_tokens(self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[List[str]]) -> List[torch.Tensor]:
        text_encoder: MossTextEncoderProxy = models[0]
        rows = text_encoder.encode(tokens[0])
        length = max(row.shape[0] for row in rows)
        embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
        mask = torch.zeros(len(rows), length, dtype=torch.long)
        for i, row in enumerate(rows):
            embeds[i, : row.shape[0]] = row
            mask[i, : row.shape[0]] = 1
        return [embeds, mask]

    def drop_cached_text_encoder_outputs(
        self,
        prompt_embeds: torch.Tensor,
        attn_mask: torch.Tensor,
        caption_dropout_rates: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Caption dropout on cached outputs: zero the mask; the trainer substitutes the empty-prompt embedding."""
        if caption_dropout_rates is None or torch.all(caption_dropout_rates == 0.0).item():
            return [prompt_embeds, attn_mask]

        attn_mask = attn_mask.clone()
        for i in range(prompt_embeds.shape[0]):
            if random.random() < caption_dropout_rates[i].item():
                attn_mask[i] = 0
        return [prompt_embeds, attn_mask]


class MossTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    """Caches unpadded prompt_embeds [L, 4096] and attn_mask [L] per image; batches are padded by the collator."""

    MOSS_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_moss_te.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
        cache_dtype: str = "auto",
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial, cache_dtype=cache_dtype)

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + self.MOSS_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str, num_caption_variants: int = 0, caption_aug_hash: Optional[str] = None) -> bool:
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = load_npz(npz_path)
            if "prompt_embeds" not in npz or "attn_mask" not in npz or "caption_dropout_rate" not in npz:
                return False
            if not self._check_cached_variant_keys(npz, {"prompt_embeds", "attn_mask"}, num_caption_variants, caption_aug_hash):
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str, variant: int = 0) -> List[np.ndarray]:
        data, dtypes = load_npz(npz_path, return_dtypes=True)
        prompt_embeds = self._npz_get(data, "prompt_embeds", variant)
        attn_mask = self._npz_get(data, "attn_mask", variant)
        caption_dropout_rate = data["caption_dropout_rate"]
        if dtypes.get("prompt_embeds") == "bf16":
            prompt_embeds = torch.from_numpy(np.ascontiguousarray(prompt_embeds)).to(dtype=torch.bfloat16)
        return [prompt_embeds, attn_mask, caption_dropout_rate]

    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        infos: List,
    ):
        canonical_captions, variant_groups = self._get_variant_caption_groups(infos)

        def encode_captions(captions: List[str]) -> List[dict]:
            # one unpadded entry per caption
            text_encoder: MossTextEncoderProxy = models[0]
            rows = text_encoder.encode(captions)
            results = []
            for row in rows:
                row = row.detach().to("cpu")
                results.append(
                    {
                        "prompt_embeds": row if row.dtype == torch.bfloat16 else row.numpy(),
                        "attn_mask": np.ones(row.shape[0], dtype=np.int32),
                    }
                )
            return results

        batched = encode_captions(canonical_captions)
        variant_batched = [(group, encode_captions([c for _, c in group])) for group in variant_groups]

        base_keys = ["prompt_embeds", "attn_mask"]

        for i, info in enumerate(infos):
            n_variants = len(info.caption_variants) if getattr(info, "caption_variants", None) else 0

            # when caption variants are cached, dropout is carried inside the variants
            effective_dropout_rate = info.caption_dropout_rate if n_variants <= 1 else 0.0
            caption_dropout_rate = torch.tensor(effective_dropout_rate, dtype=torch.float32)

            variant_entries = []
            for k, (group, vb) in enumerate(variant_batched, start=1):
                idxs = [idx for idx, _ in group]
                if i in idxs:
                    variant_entries.append((k, vb[idxs.index(i)]))

            if self.cache_to_disk:
                save_kwargs = {key: batched[i][key] for key in base_keys}
                save_kwargs["caption_dropout_rate"] = caption_dropout_rate
                for k, entry in variant_entries:
                    for key in base_keys:
                        save_kwargs[variant_key(key, k)] = entry[key]
                if n_variants > 1:
                    save_kwargs["caption_variants"] = np.array(n_variants)
                    save_kwargs["caption_aug_hash"] = np.array(getattr(info, "caption_aug_hash", None) or "")
                save_npz(info.text_encoder_outputs_npz, save_kwargs, cache_dtype=self.cache_dtype)
            else:
                info.text_encoder_outputs = tuple([batched[i][key] for key in base_keys] + [caption_dropout_rate])
                if n_variants > 1:
                    info.text_encoder_outputs_variants = [
                        tuple([entry[key] for key in base_keys] + [caption_dropout_rate]) for _, entry in variant_entries
                    ]


class MossLatentsCachingStrategy(LatentsCachingStrategy):
    """Latent caching for the MOSS-Image VAE: normalized packed latents [128, H/16, W/16]."""

    MOSS_LATENTS_NPZ_SUFFIX = "_moss.npz"
    LATENT_SCALE = 16

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool, cache_dtype: str = "auto") -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, cache_dtype=cache_dtype)

    @property
    def cache_suffix(self) -> str:
        return self.MOSS_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return os.path.splitext(absolute_path)[0] + f"_{image_size[0]:04d}x{image_size[1]:04d}" + self.MOSS_LATENTS_NPZ_SUFFIX

    def is_disk_cached_latents_expected(
        self,
        bucket_reso: Tuple[int, int],
        npz_path: str,
        flip_aug: bool,
        alpha_mask: bool,
        num_aug_variants: int = 0,
        aug_config_hash: Optional[str] = None,
    ):
        return self._default_is_disk_cached_latents_expected(
            self.LATENT_SCALE, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True,
            num_aug_variants=num_aug_variants, aug_config_hash=aug_config_hash,
        )

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int], variant: int = 0
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray], Optional[bool]]:
        return self._default_load_latents_from_disk(self.LATENT_SCALE, npz_path, bucket_reso, variant=variant)

    def cache_batch_latents(
        self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool, random_crop_padding_percent: float = 0.05,
        num_aug_variants: int = 0, augmentor: Optional[Callable] = None, aug_config_hash: Optional[str] = None,
    ):
        vae_device = vae.device
        vae_dtype = vae.dtype

        def encode_by_vae(img_tensor):
            return vae.encode_pixels_to_latents(img_tensor).to("cpu")

        self._default_cache_batch_latents(
            encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop, multi_resolution=True,
            random_crop_padding_percent=random_crop_padding_percent,
            num_aug_variants=num_aug_variants, augmentor=augmentor, aug_config_hash=aug_config_hash,
        )

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae_device)
