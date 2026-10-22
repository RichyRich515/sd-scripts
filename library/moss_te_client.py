# Client side of the MOSS-Image text encoder worker (see moss_te_worker.py).
#
# The training environment pins transformers 4.x, but Qwen3.5 needs transformers>=5.3. Instead of upgrading the
# shared venv, a transformers 5 "overlay" is pip-installed into a separate folder (pinned to the venv's versions of
# every shared dependency) and only the worker subprocess sees it via PYTHONPATH.
#
# MossTextEncoderProxy is an nn.Module stand-in for the text encoder, so the regular sd-scripts caching flow
# (caption variants, dropout rates, sample prompts, disk/memory caches) works unchanged. Moving it to the CPU
# stops the worker and frees its VRAM; the worker is (re)started lazily on the next encode.

import atexit
import glob
import json
import os
import subprocess
import sys
import tempfile
from typing import List, Optional

import torch
from safetensors.torch import load_file
from torch import nn

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

REPLY_PREFIX = "@@MOSS_TE@@ "
WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moss_te_worker.py")
DEFAULT_OVERLAY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "moss_te_deps")
OVERLAY_REQUIREMENT = "transformers>=5.3,<6"
# packages the overlay is allowed to provide in newer versions than the venv; everything else is pinned
OVERLAY_UNPINNED = {"transformers", "huggingface-hub", "tokenizers", "safetensors", "hf-xet"}


def overlay_transformers_version(overlay_dir: str) -> Optional[str]:
    found = glob.glob(os.path.join(overlay_dir, "transformers-*.dist-info"))
    if not found:
        return None
    return os.path.basename(found[0])[len("transformers-") : -len(".dist-info")]


def install_overlay(overlay_dir: str = DEFAULT_OVERLAY_DIR) -> None:
    """pip-install transformers 5 into overlay_dir, pinning shared dependencies to the venv's installed versions."""
    logger.info(f"Installing {OVERLAY_REQUIREMENT} overlay for the MOSS-Image text encoder into {overlay_dir}")
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--exclude-editable"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    pins = []
    for line in freeze:
        if "==" not in line or " @ " in line or line.startswith("-"):
            continue
        name = line.split("==")[0].strip().lower().replace("_", "-")
        if name not in OVERLAY_UNPINNED:
            pins.append(line)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(pins))
        constraints = f.name
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--target", overlay_dir, "--upgrade", "--constraint", constraints, OVERLAY_REQUIREMENT],
            check=True,
        )
    finally:
        os.remove(constraints)
    logger.info(f"Installed transformers {overlay_transformers_version(overlay_dir)} overlay")


def ensure_overlay(overlay_dir: str = DEFAULT_OVERLAY_DIR, auto_install: bool = True) -> None:
    version = overlay_transformers_version(overlay_dir)
    if version is not None and int(version.split(".")[0]) >= 5:
        return
    if not auto_install:
        raise RuntimeError(
            f"The MOSS-Image text encoder needs a transformers 5 overlay in {overlay_dir}. "
            f"Install it with: python {os.path.abspath(__file__)} install"
        )
    install_overlay(overlay_dir)


class MossTextEncoderProxy(nn.Module):
    """Stands in for the Qwen3.5 text encoder; encodes captions in a worker subprocess."""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        overlay_dir: str = DEFAULT_OVERLAY_DIR,
        max_length: int = 512,
        encode_device: str = "cuda",
    ):
        super().__init__()
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.overlay_dir = overlay_dir
        self.max_length = max_length
        self.encode_device = encode_device
        self.process: Optional[subprocess.Popen] = None
        # base trainer code inspects devices/dtypes through parameters (and logs the 3rd one)
        self.p0 = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.p1 = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.p2 = nn.Parameter(torch.zeros(1), requires_grad=False)
        atexit.register(self.stop)

    @property
    def device(self):
        return self.p0.device

    @property
    def dtype(self):
        return torch.bfloat16

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        if self.p0.device.type == "cpu":
            self.stop()
        return result

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return
        ensure_overlay(self.overlay_dir)
        env = dict(os.environ)
        env["PYTHONPATH"] = self.overlay_dir
        env["PYTHONIOENCODING"] = "utf-8"
        # -P: keep library/ (utils.py, attention.py, ...) off sys.path so it cannot shadow third-party modules
        cmd = [sys.executable, "-P", WORKER_PATH, "--model", self.model_path, "--device", self.encode_device, "--max_length", str(self.max_length)]
        if self.tokenizer_path is not None:
            cmd += ["--tokenizer", self.tokenizer_path]
        logger.info(f"Starting MOSS-Image text encoder worker: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, env=env, text=True, encoding="utf-8", bufsize=1
        )
        ready = self._read_reply()
        if not ready.get("ready"):
            self.stop()
            raise RuntimeError(f"MOSS-Image text encoder worker failed to start: {ready.get('error', ready)}")
        logger.info(f"MOSS-Image text encoder worker ready (transformers {ready.get('transformers')})")

    def stop(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                self.process.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                self.process.stdin.flush()
                self.process.wait(timeout=60)
            except Exception:
                self.process.kill()
        logger.info("MOSS-Image text encoder worker stopped")
        self.process = None

    def _read_reply(self) -> dict:
        while True:
            line = self.process.stdout.readline()
            if not line:
                code = self.process.wait()
                raise RuntimeError(f"MOSS-Image text encoder worker exited unexpectedly (code {code}); see the log above")
            if line.startswith(REPLY_PREFIX):
                return json.loads(line[len(REPLY_PREFIX) :])
            sys.stderr.write(line)

    @torch.no_grad()
    def encode(self, captions: List[str]) -> List[torch.Tensor]:
        """Returns one unpadded [L, 4096] bf16 CPU tensor per caption."""
        self.start()
        fd, out_path = tempfile.mkstemp(suffix=".safetensors", prefix="moss_te_")
        os.close(fd)
        try:
            self.process.stdin.write(json.dumps({"captions": list(captions), "out": out_path}) + "\n")
            self.process.stdin.flush()
            result = self._read_reply()
            if not result.get("ok"):
                raise RuntimeError(f"MOSS-Image text encoder worker error: {result.get('error')}")
            tensors = load_file(out_path)
            return [tensors[f"emb_{i}"] for i in range(len(captions))]
        finally:
            os.remove(out_path)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "install":
        install_overlay(sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_OVERLAY_DIR)
    else:
        print(f"usage: python {os.path.basename(__file__)} install [overlay_dir]")
