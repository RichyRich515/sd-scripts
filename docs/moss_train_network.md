# MOSS-Image LoRA training (`moss_train_network.py`)

LoRA training for [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image), a 5B single-stream DiT with a Qwen3.5-9B
text encoder and the FLUX.2 VAE (small decoder). Base is recommended for training; LoRAs trained on Base also apply to SFT.

## Models

Download the diffusers-format pipeline (~27 GB):

```
huggingface-cli download OpenMOSS-Team/MOSS-Image-Base --local-dir MOSS-Image-Base \
  --include "model_index.json" "scheduler/*" "text_encoder/*" "tokenizer/*" "transformer/*" "vae/*"
```

Pass the folder as `--pretrained_model_name_or_path`; the text encoder, tokenizer and VAE default to its subfolders.
A single bf16 transformer file (split or fused `proj.qkv`, e.g. ComfyUI exports) also works with `--moss_text_encoder`
and `--vae`. Quantized checkpoints (int8 ConvRot, w4a8, fp8) cannot be trained, and the text encoder must be the bf16
HF-format Qwen3.5-9B. For `--vae`, prefer the pipeline's `vae/`; a single `flux2-small-decoder.safetensors` also loads,
but its encoder and BatchNorm statistics differ slightly from the official MOSS-Image VAE.

## Text encoder worker

Qwen3.5 needs `transformers>=5.3`, while this environment pins 4.x. On first use, transformers 5 is installed into
`moss_te_deps/` (shared dependencies pinned to the venv's versions; run `python library/moss_te_client.py install`
to do it manually). Captions are encoded in a subprocess that sees only that overlay, one unpadded caption at a time
with the official chat template, before the DiT is loaded; the worker then exits and frees its ~16 GB of VRAM.

Consequences: text encoder outputs are always cached (`--cache_text_encoder_outputs` is forced), so caption
shuffling, tag dropout and token warmup are not available (use `--cache_caption_variants`), and the text encoder is
never trained. The empty-prompt embedding (for caption dropout) and sample prompt embeddings are cached in
`<output_dir>/.moss_prompt_te_cache.safetensors`, so runs with fully cached datasets never start the worker.

## Arguments

| Argument | Default | |
|---|---|---|
| `--moss_text_encoder` | pipeline folder | Qwen3.5-9B folder (bf16, HF format) |
| `--moss_tokenizer` | next to the text encoder | |
| `--vae` | pipeline folder | VAE folder or single file |
| `--moss_max_token_length` | 512 | caption tokens including the chat template (max 4096) |
| `--timestep_sampling` | `moss` | MOSS recipe: σ = 1 − sigmoid(μ + s·N(0,1)), truncated at cutoff and rescaled, plus a uniform mix |
| `--moss_ts_mu` / `--moss_ts_s` / `--moss_ts_cutoff` / `--moss_ts_uniform_mix` | −1.09 / 0.95 / 0.95 / 0.01 | technical report stage-3 values |
| `--discrete_flow_shift` | 1.0 | MOSS-Image trains unshifted (inference uses 5.0) |
| `--moss_lora_save_format` | `both` | `split` (proj.q/k/v), `fused` (proj.qkv), or both |
| `--moss_vae_sample_posterior` | off | sample the VAE posterior instead of its mode when caching latents |
| `--blocks_to_swap` | 0 | up to 40 of the 42 blocks |
| `--moss_attn_backend` | `auto` | PyTorch's default (the memory-efficient kernel on Windows). `cudnn` is much faster but needs cuDNN ≥ 9.24 (torch 2.14.1+cu130): cuDNN 9.7.1 (torch 2.7.1) returned NaN gradients on real inputs, and its compiled path fails |
| `--moss_nonfinite_patience` | 3 | non-finite loss/gradients are skipped (never reach the weights); this many in a row aborts |
| `--moss_compile` | `none` | `situ` or `blocks` (torch.compile, needs Triton; on Windows the MSVC environment is loaded automatically when `cl` is not on PATH, which torch ≥ 2.8 needs) |
| `--moss_int8` | off | ConvRot INT8 base weights for the 252 block projections (same recipe as the ComfyUI `*_int8_convrot` files), run through cuBLAS int8. Faster with `--moss_compile blocks` (slower without), ~4.5 GB less VRAM; not with `--blocks_to_swap` |
| `--moss_int8_bwd` | `bf16` | `int8` also quantizes the input-gradient GEMM; not faster at 1024² |
| `--moss_unchecked_blocks` | 0 | skip checkpointing for the last N blocks (~1 GB each at 1024², small gain) |
| `--moss_fp32_modulation` | off | keep `time_embedding` / `shared_block_modulation` / `video_head_modulation` in fp32 (as KohakuMossTrainer does); outputs are rounded once to bf16 |
| `--moss_sample_uncond_skip_layers` | `22` | blocks skipped in the negative CFG pass of sample images (official pipeline default); `none` for plain CFG |

Bucket resolutions must be multiples of 16 (8× VAE + 2×2 packing).

## LoRA format

Keys follow the kohya/ComfyUI convention (`lora_unet_layers_0_proj_q`), so ComfyUI and Forge map them without
conversion. Single-file checkpoints such as `moss_image_sft_bf16.safetensors` fuse attention into `layers.N.proj.qkv`;
loaders only apply LoRA keys that match the loaded model, so split q/k/v keys would be silently ignored there. With the
default `both`, the saved file also holds an exact fused `proj_qkv` LoRA (down = cat(q, k, v), up = block-diagonal with
per-part scales folded in), and each loader applies whichever set matches its model. Network args:
`train_text_refiner=true` also targets the two text refinement blocks.

## Sample prompts

Defaults follow the official pipeline: CFG 3.0 (`--l`), 30 steps (`--s`), flow shift 5.0 (`--fs`), 1024×1024
(`--w`/`--h`). The negative prompt (`--n`) defaults to the empty string.

## Speed (RTX 4090, Windows, LoRA rank 16, bf16, gradient checkpointing, batch 1)

| torch | Setup | 1024² s/img | 512² s/img | peak allocated |
|---|---|---|---|---|
| 2.7.1 | eager | 2.96 | | 13.3 GB |
| 2.7.1 | `--moss_compile blocks` | 1.91 | 0.39 | 11.9 GB |
| 2.7.1 | blocks + `--moss_int8` | 1.74 | 0.40 | 7.5 GB |
| 2.14.1 | eager | 2.70 | 0.54 | 12.9 GB |
| 2.14.1 | eager + cuDNN | 2.31 | 0.54 | 12.9 GB |
| 2.14.1 | blocks + cuDNN | 1.42 | 0.33 | 12.0 GB |
| 2.14.1 | blocks + cuDNN + `--moss_int8` | **1.19** | **0.28** | 7.5 GB |
| 2.14.1 | same + `--moss_unchecked_blocks 42` | | 0.22 | 10.8 GB |

Eager SiTU (the FFN activation) is the largest single cost in eager mode: about a dozen fp32 kernels over
`[tokens, 8192]` tensors, run again in the checkpoint recompute and backward; compiling fuses it.

cuDNN attention is ~2.4x faster than the memory-efficient kernel PyTorch's Windows wheels use (they have no
FlashAttention). With cuDNN 9.7.1 (torch 2.7.1+cu128) its backward returned NaN gradients for specific real inputs
(a full run diverged at ~step 90; reproduced deterministically at σ 0.97 with three of twelve seeds on one image) and
inductor fails on it; cuDNN 9.24 (torch 2.14.1+cu130) was finite on the same inputs. torch 2.7 also ignores an
`sdpa_kernel` context inside compiled code, so the backend is set as the process-wide SDPA priority.

INT8: GeForce cards run bf16 tensor-core matmuls at half rate (fp32 accumulation) and int8 at full rate; compiled,
the six projections of a block take 3.7 ms instead of 5.9 ms at 1024². The fused Triton kernel of upstream sd-scripts
was as fast with Triton 3.3 but ran at bf16 speed with Triton 3.8, so cuBLAS `torch._int_mm` is used with the
quantization fused by inductor. Accuracy on real samples with a trained LoRA, relative to bf16: prediction error 2.9%
(fp8 weight-only: 5.7%), loss within 0.2%, LoRA gradient cosine 0.72 (swapping the attention kernel alone gives 0.66).

Skipping checkpointing costs ~1 GB per block at 1024² for ~0.4% speed per block; at 512² all 42 blocks fit (with
int8) and save ~20%. Batch 4 at 512² is ~9% faster per image than batch 1. Inference-style sampling peaks around 10.6 GB.
