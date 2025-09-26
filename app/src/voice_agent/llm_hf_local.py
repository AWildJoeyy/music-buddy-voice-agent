from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_DIR = Path(r"app\model\models\Qwen2.5-7B-Instruct-GPTQ")
MODEL_DIR = Path(os.getenv("HF_LOCAL_PATH", str(DEFAULT_DIR))).expanduser().resolve()

DEVICE = os.getenv("HF_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
TEMP = float(os.getenv("HF_TEMP", "0.6"))
MAX_NEW = int(os.getenv("HF_MAX_NEW_TOKENS", "256"))

_tok = None
_model = None
_loaded_once = False

def _has_autogptq_files(p: Path) -> bool:
    return (p / "quantize_config.json").exists()

def _load():
    global _tok, _model
    if _model is not None:
        return
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model dir not found: {MODEL_DIR}")

    # Tokenizer (works for both cases)
    _tok = AutoTokenizer.from_pretrained(
        MODEL_DIR, use_fast=True, trust_remote_code=True, local_files_only=True
    )
    if getattr(_tok, "pad_token_id", None) is None and getattr(_tok, "eos_token_id", None) is not None:
        _tok.pad_token = _tok.eos_token

    if _has_autogptq_files(MODEL_DIR):
        # ---- Community GPTQ export (TheBloke-style) via AutoGPTQ ----
        from auto_gptq import AutoGPTQForCausalLM
        print("[hf] loading GPTQ via AutoGPTQ (quantize_config.json found)")
        _m = AutoGPTQForCausalLM.from_quantized(
            MODEL_DIR,
            trust_remote_code=True,
            device=DEVICE,
            use_safetensors=True,
            inject_fused_attention=False,
            use_triton=False,
            disable_exllama=True,  
        )
    else:
        # ---- Official Qwen GPTQ via Transformers+Optimum ----
        print("[hf] loading GPTQ via Transformers (official Qwen format)")
        _m = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto",        #
        )

    _m.eval()
    _model = _m

def _chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = []
        for m in messages:
            parts.append(f"{m['role'].capitalize()}: {m['content']}")
        parts.append("Assistant:")
        return "\n".join(parts)

@torch.inference_mode()
def complete(messages: List[Dict[str, str]]) -> str:
    _load()
    prompt = _chat_template(_tok, messages)
    inputs = _tok(prompt, return_tensors="pt")
    dev = next(_model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    out = _model.generate(
        **inputs,
        do_sample=True,
        temperature=TEMP,
        top_p=0.9,
        max_new_tokens=MAX_NEW,
        eos_token_id=getattr(_tok, "eos_token_id", None),
        pad_token_id=getattr(_tok, "pad_token_id", getattr(_tok, "eos_token_id", None)),
    )
    gen = out[0, inputs["input_ids"].shape[1]:]
    return _tok.decode(gen, skip_special_tokens=True).strip()

def preload():
    """
    Force-load the tokenizer/model and do a 1-token generate to warm CUDA kernels.
    Safe to call multiple times; it’s a no-op after the first.
    """
    global _loaded_once, _tok, _model
    if _loaded_once:
        return
    _load()  
    
    try:
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hi"},
        ]
        prompt = _chat_template(_tok, msgs)
        inputs = _tok(prompt, return_tensors="pt")
        dev = next(_model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        _model.generate(**inputs, max_new_tokens=1)
        if dev.type == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass

    _loaded_once = True
