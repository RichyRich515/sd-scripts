# MOSS-Image text encoder worker.
#
# Qwen3.5 needs transformers>=5.3, which is not installed in the training environment. The trainer runs this
# script as a subprocess with a transformers 5 overlay prepended to PYTHONPATH (see moss_te_client.py).
# It must not import anything from `library`.
#
# Protocol: one JSON request per stdin line, one reply per stdout line prefixed with REPLY_PREFIX.
#   {"captions": [...], "out": "<path>.safetensors"} -> writes emb_<i> ([L, hidden] tensors), replies {"ok": true, "n": N}
#   {"cmd": "exit"}                                  -> exits
# Everything else the libraries print goes to stderr.

import argparse
import json
import os
import sys
import traceback

REPLY_PREFIX = "@@MOSS_TE@@ "


def reply(stdout, payload: dict):
    stdout.write(REPLY_PREFIX + json.dumps(payload) + "\n")
    stdout.flush()


def resolve_paths(path: str, tokenizer_path: str | None):
    """Accept a pipeline root (with text_encoder/ and tokenizer/) or a text_encoder folder."""
    model_dir = os.path.join(path, "text_encoder") if os.path.isdir(os.path.join(path, "text_encoder")) else path
    if tokenizer_path is None:
        for cand in (os.path.join(path, "tokenizer"), os.path.join(os.path.dirname(model_dir), "tokenizer"), model_dir):
            if os.path.isfile(os.path.join(cand, "tokenizer.json")) or os.path.isfile(os.path.join(cand, "tokenizer_config.json")):
                tokenizer_path = cand
                break
    if tokenizer_path is None:
        raise FileNotFoundError(f"No tokenizer found next to {model_dir}; pass --tokenizer")
    return model_dir, tokenizer_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    # keep the protocol channel clean: libraries print to stdout
    protocol_out = sys.stdout
    sys.stdout = sys.stderr

    import torch
    import transformers
    from safetensors.torch import save_file
    from transformers import AutoModel, AutoTokenizer

    major = int(transformers.__version__.split(".")[0])
    if major < 5:
        reply(protocol_out, {"ok": False, "error": f"transformers>=5.3 required in the worker, found {transformers.__version__}"})
        return 1

    model_dir, tokenizer_dir = resolve_paths(args.model, args.tokenizer)
    dtype = getattr(torch, args.dtype)
    print(f"[moss-te] transformers {transformers.__version__}, loading {model_dir} ({args.dtype}) on {args.device}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModel.from_pretrained(model_dir, dtype=dtype)
    # only the language model is needed for text prompts; the vision tower stays on the CPU
    text_model = model.language_model if hasattr(model, "language_model") else model
    text_model.to(args.device).eval().requires_grad_(False)
    print("[moss-te] ready", file=sys.stderr)
    reply(protocol_out, {"ready": True, "transformers": transformers.__version__})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("cmd") == "exit":
                break
            tensors = {}
            with torch.no_grad():
                for i, caption in enumerate(request["captions"]):
                    # exact training-time chat template: thinking enabled, no system prompt
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": caption}], tokenize=False, add_generation_prompt=True, enable_thinking=True
                    )
                    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)["input_ids"]
                    # unpadded single forwards: padding changes Qwen3.5's gated delta-net (linear attention) states
                    out = text_model(input_ids=ids.to(args.device), use_cache=False, return_dict=True)
                    tensors[f"emb_{i}"] = out.last_hidden_state[0].to("cpu", dtype=dtype).contiguous()
            save_file(tensors, request["out"])
            reply(protocol_out, {"ok": True, "n": len(tensors)})
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            reply(protocol_out, {"ok": False, "error": f"{type(e).__name__}: {e}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
