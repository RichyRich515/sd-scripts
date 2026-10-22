# MOSS-Image Model Architecture
# Original code: OpenMOSS-Team, MOSS-Image (https://github.com/OpenMOSS/MOSS-Image), licensed under Apache-2.0
# Adapted for training: batched forward with masked text padding, gradient checkpointing and block swap.

import json
import os
from typing import Optional

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from library import custom_offloading_utils
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


MOSS_IMAGE_CONFIG = {
    "dim": 3072,
    "ffn_dim": 8192,
    "num_heads": 24,
    "num_layers": 42,
    "text_dim": 4096,
    "freq_dim": 256,
    "latent_channels": 128,
    "rope_base": 10000.0,
}


def situ_eager(hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    # computed in fp32 and rounded once, matching the reference training kernel
    h, g = hidden.float(), gate.float()
    return ((4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g)) * (25.0 * torch.tanh(h / 25.0))).to(hidden.dtype)


# Eager SiTU is ~a dozen separate fp32 kernels over [tokens, 8192] tensors (forward, recompute and backward);
# on an RTX 4090 at 1024px it was the largest single cost in a LoRA step. set_compile_mode can fuse it.
_situ_fn = situ_eager


def situ(hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return _situ_fn(hidden, gate)


def rotate(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    # x: [B, S, H, D], rope: [B, S, D] (sin | cos halves)
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


# SDPA kernel priority; None leaves the choice to PyTorch. See set_attention_backend.
_SDPA_BACKENDS = None
_DEFAULT_SDP_PRIORITY = None

# cuDNN SDPA versions whose backward returned NaN for real MOSS inputs (9.7.1, torch 2.7.1+cu128); 9.24 (torch
# 2.14.1+cu130) was finite on the same inputs. Versions in between are untested.
CUDNN_SDPA_FIRST_GOOD = 92400


def set_attention_backend(name: str = "auto"):
    """auto / torch: PyTorch's own choice. Explicit names put that backend first with fallbacks.

    cuDNN attention is ~2.4x faster than the memory-efficient kernel PyTorch picks on Windows for MOSS shapes. With
    cuDNN 9.7.1 (torch 2.7.1 + cu128) on an RTX 4090 its backward returned NaN query gradients for real training
    inputs (deterministic, e.g. a 51x28 latent + 128-token caption at sigma 0.97, seeds 213-215, block 40) and a full
    LoRA run went NaN at ~step 90; cuDNN 9.24 was finite on the same inputs.

    The priority is also set process-wide: torch 2.7 ignores an sdpa_kernel context inside torch.compile'd code
    (compiled blocks kept the efficient kernel), while the global order is honored when the graph is traced."""
    global _SDPA_BACKENDS, _DEFAULT_SDP_PRIORITY
    from torch.nn.attention import SDPBackend

    if _DEFAULT_SDP_PRIORITY is None:
        _DEFAULT_SDP_PRIORITY = list(torch._C._get_sdp_priority_order())
    fallbacks = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    if name in ("auto", "torch"):
        _SDPA_BACKENDS = None
    else:
        first = {
            "cudnn": SDPBackend.CUDNN_ATTENTION,
            "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "math": SDPBackend.MATH,
        }[name]
        _SDPA_BACKENDS = [first] + [b for b in fallbacks if b != first]
        if name == "cudnn" and (torch.backends.cudnn.version() or 0) < CUDNN_SDPA_FIRST_GOOD:
            logger.warning(
                f"MOSS-Image: cuDNN {torch.backends.cudnn.version()} attention produced NaN gradients in training "
                f"(fixed in {CUDNN_SDPA_FIRST_GOOD}, e.g. torch 2.14.1+cu130); prefer 'auto' with this torch"
            )
    if _SDPA_BACKENDS is None:
        torch._C._set_sdp_priority_order(_DEFAULT_SDP_PRIORITY)
    else:
        # backends left out keep their default relative order behind the chosen ones
        order = [int(b) for b in _SDPA_BACKENDS]
        torch._C._set_sdp_priority_order(order + [b for b in _DEFAULT_SDP_PRIORITY if b not in order])
    logger.info(f"MOSS-Image attention backend '{name}': {_SDPA_BACKENDS or 'PyTorch default'}")


def attention(q, k, v, rope, heads: int, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """q/k/v: [B, S, dim]; attn_mask: [B, 1, 1, S] bool (True = attend) or None."""
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = rotate(q.reshape(shape), rope).transpose(1, 2)
    k = rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    if _SDPA_BACKENDS is None:
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    else:
        from torch.nn.attention import sdpa_kernel

        with sdpa_kernel(_SDPA_BACKENDS, set_priority=True):
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return x.transpose(1, 2).flatten(2)


class MossTextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.q_norm = nn.RMSNorm(dim // heads, eps=1e-6)
        self.k_norm = nn.RMSNorm(dim // heads, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate = nn.Linear(dim, ffn_dim)
        self.ffn_up = nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope, attn_mask):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(attention(q, k, v, rope, self.heads, attn_mask))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class MossTextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([MossTextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope, attn_mask):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope, attn_mask)
        return x


def ensure_msvc_environment():
    """Windows: newer inductor (torch 2.8+) needs the MSVC compiler for dynamic-shape graphs ("cl is not found").
    If cl is not on PATH, load the environment of the newest Visual Studio install with the C++ tools (vcvars64)."""
    import shutil
    import subprocess

    if os.name != "nt" or shutil.which("cl"):
        return
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.isfile(vswhere):
        logger.warning("MSVC (cl) not found and Visual Studio is not installed; torch.compile may fail on newer torch")
        return
    install = subprocess.run(
        [vswhere, "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
        capture_output=True, text=True,
    ).stdout.strip()
    vcvars = os.path.join(install, "VC", "Auxiliary", "Build", "vcvars64.bat") if install else ""
    if not os.path.isfile(vcvars):
        logger.warning("MSVC (cl) not found: no Visual Studio install with the C++ tools; torch.compile may fail on newer torch")
        return
    out = subprocess.run(f'cmd /s /c ""{vcvars}" >nul && set"', capture_output=True, text=True, shell=False).stdout
    for line in out.splitlines():
        key, sep, value = line.partition("=")
        if sep and key:
            os.environ[key] = value
    logger.info(f"Loaded the MSVC environment for torch.compile from {vcvars}")


INT8_GROUPSIZE = 256


def convrot_hadamard(size: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Normalized regular Hadamard matrix (symmetric, orthogonal: H @ H = I) used by ConvRot (arXiv:2512.03673)."""
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    if h.shape[0] != size:
        raise ValueError(f"ConvRot group size must be a power of 4, got {size}")
    return (h / size**0.5).to(device=device, dtype=dtype)


def quantize_convrot_weight(weight: torch.Tensor, group_size: int = INT8_GROUPSIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """W_rot = W @ H^T per group of input features, then per-output-channel int8 (rotation in fp32). Same recipe as
    the ComfyUI *_int8_convrot checkpoints. Returns (int8 [N, K], fp32 scale [N])."""
    out_features, in_features = weight.shape
    h = convrot_hadamard(group_size, weight.device)
    w = (weight.float().reshape(out_features, -1, group_size) @ h.T).reshape(out_features, in_features)
    scale = (w.abs().amax(-1, keepdim=True) / 127.0).clamp(min=1e-30)
    return (w / scale).round().clamp(-128, 127).to(torch.int8), scale.reshape(-1)


def _quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic symmetric int8 quantization per row: (int8 [M, K], fp32 scale [M, 1])."""
    scale = x.abs().amax(-1, keepdim=True).float().clamp(min=1e-12) / 127.0
    return (x.float() / scale).round().clamp(-128, 127).to(torch.int8), scale


class _Int8MatmulFn(torch.autograd.Function):
    """x [M, K] @ (weight [N, K] int8 * scale [N])^T through cuBLAS int8 (torch._int_mm).

    Plain torch ops, so torch.compile traces it and inductor fuses the quantization and dequantization into the
    neighbouring kernels. (The fused Triton kernel of sd-scripts / Musubi Tuner was as fast with Triton 3.3 but ran at
    bf16 speed with Triton 3.8 on an RTX 4090.) The base weight is frozen: backward only returns grad_x."""

    @staticmethod
    def forward(ctx, x, weight, scale, bwd_int8: bool):
        ctx.save_for_backward(weight, scale)
        ctx.bwd_int8 = bwd_int8
        xq, xs = _quantize_rows(x)
        return (torch._int_mm(xq, weight.t()).float() * xs * scale).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        weight, scale = ctx.saved_tensors
        if ctx.bwd_int8:
            # fold the per-channel weight scale into the gradient, then quantize it row-wise too
            gq, gs = _quantize_rows(grad * scale.to(grad.dtype))
            grad_x = (torch._int_mm(gq, weight).float() * gs).to(grad.dtype)
        else:
            grad_x = grad @ (weight.to(grad.dtype) * scale[:, None].to(grad.dtype))
        return grad_x, None, None, None


def _int8_linear_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """ConvRot: rotate each group of 256 input features with the Hadamard matrix H (W was stored as W @ H^T)."""
    if torch.is_autocast_enabled(x.device.type):
        x = x.to(torch.get_autocast_dtype(x.device.type))
    features = x.shape[-1]
    x_rot = (x.reshape(-1, features // INT8_GROUPSIZE, INT8_GROUPSIZE) @ self.hadamard.to(x.dtype)).reshape(-1, features)
    out = _Int8MatmulFn.apply(x_rot, self.weight, self.scale_weight, self.int8_bwd_mode == "int8")
    return out.reshape(*x.shape[:-1], out.shape[-1])


class MossBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.q_norm = nn.RMSNorm(dim // heads, eps=1e-6)
        self.k_norm = nn.RMSNorm(dim // heads, eps=1e-6)
        self.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def _forward(self, x, rope, modulation, attn_mask):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        dtype = self.norm1.weight.dtype  # the projection weights may be int8 (quantize_blocks_int8)
        y = (self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]).to(dtype)
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](attention(q, k, self.proj["v"](y), rope, self.heads, attn_mask)) * gate_a[:, None]
        y = (self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]).to(dtype)
        hidden, gate = self.proj["ffn_up_gate"](y).chunk(2, -1)
        return x + self.proj["ffn_down"](situ(hidden, gate)) * gate_f[:, None]

    def forward(self, x, rope, modulation, attn_mask):
        if self.training and self.gradient_checkpointing:
            if self.cpu_offload_checkpointing:
                func = custom_offloading_utils.create_cpu_offloading_wrapper(self._forward, self.proj["q"].weight.device)
                return torch_checkpoint(func, x, rope, modulation, attn_mask, use_reentrant=False)
            return torch_checkpoint(self._forward, x, rope, modulation, attn_mask, use_reentrant=False)
        return self._forward(x, rope, modulation, attn_mask)


class MossRoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        # kept for checkpoint key compatibility; frequencies are always rebuilt in fp32
        self.register_buffer("bands", self._default_bands(torch.device("cpu")))

    def _default_bands(self, device):
        exponent = torch.arange(self.num_bands, device="cpu", dtype=torch.float32) / self.num_bands
        return (1 / (self.base**exponent)).to(device)

    def forward(self, height: int, width: int, text_lengths: list[int], padded_length: int, device) -> torch.Tensor:
        """Per-sample RoPE [B, H*W + padded_length, head_dim]. Image tokens carry the sample's text length."""
        bands = self._default_bands(device)
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        hw = torch.stack((h, w), -1).flatten(0, 1).float()
        hw[:, 0] -= (height - 1) / 2
        hw[:, 1] -= (width - 1) / 2
        text_pos = torch.arange(padded_length, device=device, dtype=torch.float32)

        ropes = []
        for length in text_lengths:
            # coordinates: (r, f, h, w); sequence order is image followed by text
            image = torch.zeros(height * width, 4, device=device, dtype=torch.float32)
            image[:, 1] = length
            image[:, 2:] = hw
            text = torch.zeros(padded_length, 4, device=device, dtype=torch.float32)
            text[:, 1] = text_pos
            phase = torch.cat((image, text), 0).unsqueeze(-1) * bands
            ropes.append(torch.cat((phase.sin(), phase.cos()), -2).flatten(-2))
        return torch.stack(ropes, 0)


class MossImageTransformer(nn.Module):
    """MOSS-Image dense single-stream DiT.

    Text is padded to the longest sample in the batch; padded positions are masked out as attention keys,
    so each sample is numerically equivalent to the upstream per-sample forward at its true text length.
    """

    def __init__(
        self,
        dim=3072,
        ffn_dim=8192,
        num_heads=24,
        num_layers=42,
        text_dim=4096,
        freq_dim=256,
        latent_channels=128,
        rope_base=10000.0,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.freq_dim = freq_dim
        self.num_blocks = num_layers
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = MossTextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = MossRoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([MossBlock(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = nn.RMSNorm(dim, eps=1e-6)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

        self.blocks_to_swap = None
        self.offloader: Optional[custom_offloading_utils.ModelOffloader] = None

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def set_compile_mode(self, mode: str = "none"):
        """none: eager. situ: fuse the SiTU activation only. blocks: compile each joint block (one graph shared by
        all 42 blocks; fuses RoPE, norms, modulation, SiTU and residual gating). Dynamic shapes avoid a recompile
        per bucket resolution."""
        global _situ_fn
        if mode != "none":
            ensure_msvc_environment()
        _situ_fn = situ_eager
        for block in self.layers:
            block._forward = MossBlock._forward.__get__(block)
        if mode == "situ":
            _situ_fn = torch.compile(situ_eager, dynamic=True)
        elif mode == "blocks":
            for block in self.layers:
                block._forward = torch.compile(MossBlock._forward.__get__(block), dynamic=True)
        elif mode != "none":
            raise ValueError(f"unknown compile mode {mode}")
        logger.info(f"MOSS-Image compile mode: {mode}")

    def enable_gradient_checkpointing(self, cpu_offload: bool = False, unchecked_blocks: int = 0):
        """unchecked_blocks: the last N joint blocks keep their activations instead of recomputing them in
        backward (faster, more VRAM)."""
        checked = max(0, len(self.layers) - unchecked_blocks)
        for i, block in enumerate(self.layers):
            if i < checked:
                block.enable_gradient_checkpointing(cpu_offload=cpu_offload)
            else:
                block.disable_gradient_checkpointing()
        logger.info(
            f"MOSS-Image: Gradient checkpointing enabled for {checked}/{len(self.layers)} blocks. CPU offload: {cpu_offload}"
        )

    def disable_gradient_checkpointing(self):
        for block in self.layers:
            block.disable_gradient_checkpointing()

    def enable_block_swap(self, num_blocks: int, device: torch.device):
        self.blocks_to_swap = num_blocks
        assert (
            self.blocks_to_swap <= self.num_blocks - 2
        ), f"Cannot swap more than {self.num_blocks - 2} blocks. Requested: {self.blocks_to_swap} blocks."
        self.offloader = custom_offloading_utils.ModelOffloader(self.layers, self.blocks_to_swap, device)
        logger.info(f"MOSS-Image: Block swap enabled. Swapping {num_blocks} blocks, total blocks: {self.num_blocks}, device: {device}.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        if self.blocks_to_swap:
            save_layers = self.layers
            self.layers = None
        self.to(device)
        if self.blocks_to_swap:
            self.layers = save_layers

    def switch_block_swap_for_inference(self):
        if not self.blocks_to_swap:
            return
        self.offloader.set_forward_only(True)
        self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        if not self.blocks_to_swap:
            return
        self.offloader.set_forward_only(False)
        self.prepare_block_swap_before_forward()

    def prepare_block_swap_before_forward(self):
        if not self.blocks_to_swap:
            return
        self.offloader.prepare_block_devices_before_forward(self.layers)

    MODULATION_MODULES = ("time_embedding", "shared_block_modulation", "video_head_modulation")

    def modulation_to_fp32(self):
        """Keep the timestep MLP and modulation projections in fp32 (as KohakuMossTrainer's KEEP_FP32 does).
        They are frozen during LoRA training; this only avoids bf16 rounding of their outputs."""
        for name in self.MODULATION_MODULES:
            getattr(self, name).to(torch.float32)

    def quantize_blocks_int8(self, bwd_mode: str = "bf16"):
        """Quantize the frozen joint-block projections (252 Linear layers) to ConvRot INT8 for LoRA training.

        On GeForce GPUs bf16 tensor-core matmuls run at half rate with fp32 accumulation while int8 runs at full
        rate: compiled, the six projections of a block take 3.7 ms instead of 5.9 ms at 1024px on an RTX 4090, and
        the weights take half the memory. The recipe matches the ComfyUI `*_int8_convrot.safetensors` files. Call
        before the LoRA is applied (LoRA wraps the patched forward) and not together with block swap (the offloader
        does not move the scales). Without --moss_compile blocks it is slower than bf16 (unfused quantization)."""
        if bwd_mode not in ("bf16", "int8"):
            raise ValueError(f"unknown int8 backward mode {bwd_mode}")
        if self.blocks_to_swap:
            raise ValueError("int8 quantization cannot be combined with block swap")
        hadamard = convrot_hadamard(INT8_GROUPSIZE, self.layers[0].proj["q"].weight.device, self.layers[0].norm1.weight.dtype)
        for block in self.layers:
            for linear in block.proj.values():
                weight = linear.weight
                quantized, scale = quantize_convrot_weight(weight.data)
                linear.weight = nn.Parameter(quantized.to(weight.device), requires_grad=False)
                linear.register_buffer("scale_weight", scale.to(weight.device))
                linear.hadamard = hadamard  # shared, not a buffer (stays out of state_dict)
                linear.int8_bwd_mode = bwd_mode
                linear.forward = _int8_linear_forward.__get__(linear)
        logger.info(f"MOSS-Image: {len(self.layers) * 6} block projections quantized to ConvRot INT8 (backward: {bwd_mode})")

    def time_embed(self, timestep: torch.Tensor, batch: int) -> torch.Tensor:
        times = torch.as_tensor(timestep, device=self.video_embedder.weight.device, dtype=torch.float32).reshape(-1)
        times = times.expand(batch)
        # the reference computes the sinusoid in float64 and feeds float32 into the MLP
        half = self.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        return self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()

    def modulation(self, timestep: torch.Tensor, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        # computed in the modules' own dtype: autocast would otherwise run fp32-kept weights in bf16
        with torch.autocast(device_type=self.video_embedder.weight.device.type, enabled=False):
            time = self.time_embed(timestep, batch)
            mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
            head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)
        # round once to the block compute dtype so fp32-kept modulation does not promote the residual stream
        dtype = self.video_embedder.weight.dtype
        return mod.to(dtype), head.to(dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        skip_layers: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """
        hidden_states: [B, 128, H, W] packed latents
        timestep: [B] or scalar, sigma * 1000
        encoder_hidden_states: [B, S, 4096] Qwen3.5 last hidden states (right padded)
        encoder_attention_mask: [B, S] binary, 1 = valid token
        skip_layers: zero-based joint block indices to skip (inference only, e.g. the official CFG uncond skip of 22)
        """
        skip_layers = set(skip_layers or ())
        batch, _, height, width = hidden_states.shape
        device = hidden_states.device
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=device)
        mask = encoder_attention_mask.to(device).bool()
        text_lengths = mask.sum(1).tolist()
        if min(text_lengths) == 0:
            raise ValueError("Each sample needs at least one valid text token.")

        # trim to the longest sample; the trailing columns are padding for every sample
        padded_length = max(text_lengths)
        context = encoder_hidden_states[:, :padded_length]
        mask = mask[:, :padded_length]

        mod, head = self.modulation(timestep, batch)

        video = hidden_states.flatten(2).transpose(1, 2)
        video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
        rope = self.rope(height, width, text_lengths, padded_length, device).to(video.dtype)

        # padded text keys are masked out; without padding, mask=None keeps the flash kernel eligible
        all_valid = bool(mask.all())
        text_attn_mask = None if all_valid else mask[:, None, None, :]
        joint_attn_mask = None
        if not all_valid:
            image_mask = torch.ones(batch, height * width, dtype=torch.bool, device=device)
            joint_attn_mask = torch.cat((image_mask, mask), 1)[:, None, None, :]

        text = self.text_embedder(context, rope[:, height * width :], text_attn_mask)
        x = torch.cat((video, text), 1)

        for block_idx, layer in enumerate(self.layers):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(block_idx)
            if block_idx not in skip_layers:  # swap bookkeeping still runs for skipped blocks
                x = layer(x, rope, mod, joint_attn_mask)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks(self.layers, block_idx)

        x = self.final_norm_video(x[:, : height * width].to(self.final_norm_video.weight.dtype))
        x = x * (1 + head[:, 1, None]) + head[:, 0, None]
        x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
        return x.transpose(1, 2).reshape(batch, -1, height, width)


# region loading


def _read_transformer_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Read a diffusers transformer folder (sharded or single) or a single safetensors file."""
    if os.path.isdir(path):
        folder = os.path.join(path, "transformer") if os.path.isdir(os.path.join(path, "transformer")) else path
        index = os.path.join(folder, "diffusion_pytorch_model.safetensors.index.json")
        if os.path.isfile(index):
            with open(index, "r", encoding="utf-8") as f:
                weight_map = json.load(f)["weight_map"]
            state_dict = {}
            for shard in sorted(set(weight_map.values())):
                state_dict.update(load_file(os.path.join(folder, shard)))
            return state_dict
        return load_file(os.path.join(folder, "diffusion_pytorch_model.safetensors"))
    return load_file(path)


def convert_moss_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip known prefixes and split fused qkv exports (ComfyUI/Forge single files) into q/k/v."""
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
        if any(k.startswith(prefix) for k in state_dict):
            state_dict = {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

    converted = {}
    for key, value in state_dict.items():
        if key.endswith(".proj.qkv.weight") and key.startswith("layers."):
            base = key[: -len("qkv.weight")]
            q, k, v = value.chunk(3, dim=0)
            converted[base + "q.weight"] = q
            converted[base + "k.weight"] = k
            converted[base + "v.weight"] = v
        else:
            converted[key] = value
    return converted


def is_moss_state_dict_fused(state_dict_keys) -> bool:
    return any(k.endswith("layers.0.proj.qkv.weight") for k in state_dict_keys)


def load_moss_image_model(
    path: str,
    device: torch.device,
    dtype: Optional[torch.dtype],
) -> MossImageTransformer:
    logger.info(f"Loading MOSS-Image transformer from {path}")
    state_dict = convert_moss_state_dict(_read_transformer_state_dict(path))

    for key in ("video_embedder.bias", "shared_block_modulation.weight", "layers.0.proj.ffn_up_gate.weight", "final_linear_video.weight"):
        if key not in state_dict:
            raise ValueError(f"Not a complete MOSS-Image transformer; missing {key}")
    if any(v.dtype in (torch.int8, torch.uint8) or v.dtype.itemsize == 1 for v in state_dict.values() if v.dtype != torch.bool):
        raise ValueError("Quantized (int8/fp8) MOSS-Image checkpoints cannot be used for training. Use a bf16/fp32 checkpoint.")

    dim = state_dict["video_embedder.bias"].shape[0]
    head_dim = state_dict["layers.0.q_norm.weight"].shape[0]
    num_layers = 1 + max(int(k.split(".")[1]) for k in state_dict if k.startswith("layers."))
    config = dict(MOSS_IMAGE_CONFIG)
    config.update(
        dim=dim,
        num_heads=dim // head_dim,
        num_layers=num_layers,
        ffn_dim=state_dict["layers.0.proj.ffn_up_gate.weight"].shape[0] // 2,
        latent_channels=state_dict["final_linear_video.weight"].shape[0],
        text_dim=state_dict["text_embedder.proj.weight"].shape[1],
        freq_dim=state_dict["time_embedding.0.weight"].shape[1],
    )
    logger.info(f"MOSS-Image config: {config}")

    with torch.device("meta"):
        model = MossImageTransformer(**config)
    if "rope.bands" not in state_dict:
        state_dict["rope.bands"] = model.rope._default_bands(torch.device("cpu"))
    if dtype is not None:
        state_dict = {k: (v.to(dtype) if v.is_floating_point() and k != "rope.bands" else v) for k, v in state_dict.items()}
    info = model.load_state_dict(state_dict, strict=True, assign=True)
    logger.info(f"Loaded MOSS-Image transformer: {info}")
    model.rope.bands = model.rope._default_bands(torch.device("cpu"))
    return model.to(device)


# endregion
